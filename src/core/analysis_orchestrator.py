#!/usr/bin/env python
"""Analysis Orchestrator for coordinating analysis components."""

import os
import time
import logging
from typing import List, Dict, Optional

from ..utils.logmanager import LoggerManager
from ..io.path_manager import PathManager
from .system_analyser import SystemAnalyser, BatchAnalyser
from ..io.result_saver import ResultSaver
from ..utils.common import ErrorHandler
from ..utils.common import FileUtils
from .task_scheduler import TaskScheduler, AnalysisTask
from .process_scheduler import ProcessScheduler, ProcessAnalysisTask
from ..io.path_manager import lightweight_discover_systems, load_sampling_reuse_map


class AnalysisOrchestrator:
    """协调器类，用于组织和协调各个分析组件，降低耦合度。"""

    def __init__(self):
        self.logger = None
        self.path_manager = None
        self.system_analyser = None
        self.result_saver = None

    def initialize_components(self, args):
        """初始化所有分析组件"""
        # 日志系统
        analysis_results_dir = os.path.join(FileUtils.get_project_root(), "analysis_results")
        os.makedirs(analysis_results_dir, exist_ok=True)

        self.log_queue, self.log_listener = LoggerManager.create_multiprocess_logging_setup(
            output_dir=analysis_results_dir,
            log_filename="main.log",
            when="D",
            backup_count=14
        )

        self.log_listener.start()

        # 创建主进程日志器
        self.logger = LoggerManager.setup_worker_logger(
            name="AnalysisOrchestrator",
            queue=self.log_queue,
            level=logging.INFO,
            add_console=True
        )

        # 初始化其他组件
        self.path_manager = PathManager(args.output_dir)
        self.system_analyser = SystemAnalyser(
            include_hydrogen=False,
            sample_ratio=args.sample_ratio,
            power_p=args.power_p,
            pca_variance_ratio=args.pca_variance_ratio
        )
        self.result_saver = ResultSaver()

        return self.logger

    def discover_systems(self, search_paths: List[str], include_project: bool) -> List[str]:
        """发现系统目录"""
        return FileUtils.discover_systems(search_paths, include_project)

    def setup_analysis_directory(self, current_analysis_params: Dict) -> str:
        """设置分析参数专用目录"""
        actual_output_dir = self.path_manager.set_output_dir_for_params(current_analysis_params)
        return actual_output_dir

    def check_existing_results(self) -> bool:
        """检查是否已有完整结果"""
        return self.path_manager.check_existing_complete_results()

    def load_sampled_frames(self):
        """加载采样帧信息"""
        self.path_manager.load_sampled_frames_from_csv()

    def run_system_analysis(self, system_path: str, pre_sampled_frames: Optional[List[int]] = None):
        """运行单个系统分析"""
        return self.system_analyser.analyse_system(system_path, pre_sampled_frames=pre_sampled_frames)

    def run_sampling_evaluation(self, output_dir: str):
        """运行采样效果评估"""
        try:
            from ..analysis.sampling_comparison import analyse_sampling_compare as SamplingComparisonRunner
            self.logger.info("开始采样效果评估...")
            SamplingComparisonRunner(output_dir)
        except Exception as e:
            self.logger.warning(f"采样效果评估失败: {e}")

    # --- 新增：轻量发现 + 多线程调度 ---
    def run_parallel_lightweight(self, search_paths, include_project: bool, max_workers: int = 8):
        """轻量化扫描 + 构建任务池 + 多线程分析

        Args:
            search_paths (List[str]): 搜索路径
            include_project (bool): 是否包含项目根
            max_workers (int): 线程数
        Returns:
            List[Any]: analyse_system 返回的结果元组列表
        """
        if self.logger is None:
            # 确保已经初始化
            raise RuntimeError("必须先调用 initialize_components")
        t0 = time.time()
        records = lightweight_discover_systems(search_paths, include_project=include_project)
        self.logger.info(f"轻量发现耗时 {time.time()-t0:.1f}s, 去重后体系数={len(records)}")

        # 复用采样 (如果已有 targets 文件)
        reuse_map = {}
        if self.path_manager and self.path_manager.targets_file:
            reuse_map = load_sampling_reuse_map(self.path_manager.targets_file)
            self.logger.info(f"采样复用候选体系: {len(reuse_map)}")

        scheduler = TaskScheduler(max_workers=max_workers)
        reused = 0
        for rec in records:
            pre = None
            reuse_sampling = False
            if rec.system_name in reuse_map:
                meta = reuse_map[rec.system_name]
                if meta.get('source_hash') == rec.source_hash and meta.get('sampled_frames'):
                    pre = meta.get('sampled_frames')
                    reuse_sampling = True
                    reused += 1
            scheduler.add_task(AnalysisTask(
                system_path=rec.system_path,
                system_name=rec.system_name,
                pre_sampled_frames=pre,
                pre_stru_files=rec.selected_files,
                reuse_sampling=reuse_sampling
            ))
        self.logger.info(f"构建任务: {len(scheduler.tasks)} (其中复用采样 {reused})")
        def _analyse_task(task: AnalysisTask):
            return self.system_analyser.analyse_system(
                task.system_path,
                pre_sampled_frames=task.pre_sampled_frames,
                pre_stru_files=task.pre_stru_files
            )
        results = scheduler.run(_analyse_task)
        # 同步采样帧到 PathManager (如果需要后续保存)
        if self.path_manager:
            try:
                self.path_manager.update_sampled_frames_from_results(results)
            except Exception as e:
                self.logger.warning(f"同步采样帧失败: {e}")
        return results

    def run_parallel_process(self, search_paths, include_project: bool, max_workers: int = 4):
        """使用进程池的体系粒度并行（单体系单核）。"""
        if self.logger is None:
            raise RuntimeError("必须先初始化组件")
        import time
        t0 = time.time()
        records = lightweight_discover_systems(search_paths, include_project=include_project)
        self.logger.info(f"轻量发现耗时 {time.time()-t0:.1f}s, 体系={len(records)}")
        reuse_map = {}
        if self.path_manager and self.path_manager.targets_file:
            reuse_map = load_sampling_reuse_map(self.path_manager.targets_file)
        scheduler = ProcessScheduler(max_workers=max_workers, analyser_params={
            'sample_ratio': getattr(self.system_analyser, 'sample_ratio', 0.1),
            'power_p': getattr(self.system_analyser, 'power_p', 0.5),
            'pca_variance_ratio': getattr(self.system_analyser, 'pca_variance_ratio', 0.90),
        })
        reused = 0
        for rec in records:
            pre = None
            if rec.system_name in reuse_map:
                meta = reuse_map[rec.system_name]
                if meta.get('source_hash') == rec.source_hash and meta.get('sampled_frames'):
                    pre = meta.get('sampled_frames')
                    reused += 1
            scheduler.add_task(ProcessAnalysisTask(
                system_path=rec.system_path,
                system_name=rec.system_name,
                pre_sampled_frames=pre,
                pre_stru_files=rec.selected_files,
            ))
        self.logger.info(f"进程任务: {len(scheduler.tasks)} (复用 {reused})")
        results = scheduler.run()
        if self.path_manager:
            try:
                self.path_manager.update_sampled_frames_from_results(results)
            except Exception as e:  # noqa
                self.logger.warning(f"更新采样帧失败: {e}")
        return results

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'log_listener') and self.log_listener:
            try:
                LoggerManager.stop_listener(self.log_listener)
                if self.logger:
                    self.logger.info("日志监听器已停止")
            except Exception as e:
                print(f"停止日志监听器时出错: {str(e)}")

    def get_worker_count(self, requested_workers: int) -> int:
        """确定工作进程数"""
        import multiprocessing as mp
        if requested_workers <= 0:
            return max(1, mp.cpu_count() - 1)  # 保留一个核心给主进程
        return min(requested_workers, mp.cpu_count())
