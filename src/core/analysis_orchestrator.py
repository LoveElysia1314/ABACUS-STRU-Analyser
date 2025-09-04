#!/usr/bin/env python
"""Analysis Orchestrator for coordinating analysis components."""

import os
import logging
import glob
import multiprocessing as mp
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from ..utils.common import FileUtils
from ..utils.logmanager import LoggerManager
from ..utils.common import ErrorHandler
from ..io.path_manager import PathManager, load_sampling_reuse_map
from ..core.system_analyser import SystemAnalyser
from ..io.result_saver import ResultSaver


@dataclass
class AnalysisConfig:
    """分析配置类"""
    # 核心参数
    sample_ratio: float = 0.1
    power_p: float = -0.5
    pca_variance_ratio: float = 0.90
    
    # 运行配置
    workers: int = -1
    output_dir: str = 'analysis_results'
    search_paths: List[str] = None
    include_project: bool = False
    force_recompute: bool = False
    
    # 流程控制
    steps: List[int] = None  # 要执行的步骤列表，如[1,2,3]
    
    def __post_init__(self):
        if self.search_paths is None:
            self.search_paths = []
        if self.steps is None:
            self.steps = [1, 2, 3]  # 默认执行所有步骤


# 多进程工作上下文
class WorkerContext:
    """多进程工作上下文，避免使用全局变量"""
    _analyser = None
    _reuse_map: Dict[str, List[int]] = {}

    @classmethod
    def initialize(cls, analyser, reuse_map: Dict[str, List[int]]):
        """初始化工作上下文"""
        cls._analyser = analyser
        cls._reuse_map = reuse_map or {}

    @classmethod
    def get_analyser(cls):
        """获取分析器实例"""
        return cls._analyser

    @classmethod
    def get_reuse_map(cls):
        """获取复用映射"""
        return cls._reuse_map

    @classmethod
    def set_sampling_only(cls, sampling_only: bool):
        """设置采样模式"""
        if cls._analyser:
            pass


class AnalysisOrchestrator:
    """分析流程编排器 - 核心逻辑提取"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger: Optional[logging.Logger] = None
        self.log_queue = None
        self.log_listener = None
        self.current_output_dir: Optional[str] = None
        
    def setup_logging(self) -> None:
        """设置多进程安全日志系统"""
        analysis_results_dir = os.path.join(FileUtils.get_project_root(), "analysis_results")
        os.makedirs(analysis_results_dir, exist_ok=True)

        self.log_queue, self.log_listener = LoggerManager.create_multiprocess_logging_setup(
            output_dir=analysis_results_dir,
            log_filename="main.log",
            when="D",
            backup_count=14
        )

        self.log_listener.start()

        self.logger = LoggerManager.setup_worker_logger(
            name=__name__,
            queue=self.log_queue,
            level=logging.INFO,
            add_console=True
        )
    
    def cleanup_logging(self) -> None:
        """清理日志系统"""
        if hasattr(self, 'log_listener') and self.log_listener:
            try:
                LoggerManager.stop_listener(self.log_listener)
                if self.logger:
                    self.logger.info("日志监听器已停止")
            except (OSError, ValueError) as e:
                if self.logger:
                    self.logger.warning("停止日志监听器时出错: %s", e)
    
    def resolve_search_paths(self) -> List[str]:
        """解析搜索路径，支持通配符展开"""
        if not self.config.search_paths:
            return [os.path.abspath(os.path.join(os.getcwd(), '..'))]
        
        resolved_paths = []
        for path_pattern in self.config.search_paths:
            expanded = glob.glob(path_pattern, recursive=True)
            if expanded:
                expanded_dirs = [p for p in expanded if os.path.isdir(p)]
                resolved_paths.extend(expanded_dirs)
                if expanded_dirs:
                    self.logger.info("通配符 '%s' 展开为 %d 个目录", path_pattern, len(expanded_dirs))
            else:
                if os.path.isdir(path_pattern):
                    resolved_paths.append(path_pattern)
                else:
                    self.logger.warning("路径不存在或不是目录: %s", path_pattern)
        
        unique_paths = list(set(os.path.abspath(p) for p in resolved_paths))
        return unique_paths
    
    def setup_output_directory(self) -> Tuple[PathManager, str]:
        """设置输出目录和路径管理器"""
        current_analysis_params = {
            'sample_ratio': self.config.sample_ratio,
            'power_p': self.config.power_p,
            'pca_variance_ratio': self.config.pca_variance_ratio
        }

        path_manager = PathManager(self.config.output_dir)
        actual_output_dir = path_manager.set_output_dir_for_params(current_analysis_params)
        self.current_output_dir = actual_output_dir
        path_manager.output_dir = actual_output_dir  # 设置output_dir属性
        
        self.logger.info(f"使用参数专用目录: {actual_output_dir}")
        return path_manager, actual_output_dir
    
    def setup_analysis_targets(self, path_manager: PathManager, search_paths: List[str]) -> bool:
        """设置分析目标，返回是否可以使用现有结果"""
        current_analysis_params = {
            'sample_ratio': self.config.sample_ratio,
            'power_p': self.config.power_p,
            'pca_variance_ratio': self.config.pca_variance_ratio
        }
        
        # 检查是否已有完整结果（快速路径）
        if not self.config.force_recompute and path_manager.check_existing_complete_results():
            self.logger.info("发现完整的分析结果")
            path_manager.load_sampled_frames_from_csv()
            return True
        
        # 加载现有目标状态
        loaded_existing = path_manager.load_analysis_targets()
        if loaded_existing:
            self.logger.info("成功加载已有的分析目标状态")
        else:
            self.logger.info("未找到已有的分析目标文件，将创建新的")
        
        # 加载发现结果并去重
        path_manager.load_from_discovery(search_paths)
        path_manager.deduplicate_targets()
        
        # 参数兼容性检查
        params_compatible = False
        if loaded_existing:
            params_compatible = path_manager.check_params_compatibility(current_analysis_params)
        
        # 增量计算检查
        if not self.config.force_recompute and params_compatible:
            self.logger.info("检查已有分析结果以启用增量计算...")
            path_manager.check_existing_results()
        elif self.config.force_recompute:
            self.logger.info("强制重新计算模式：将重新计算所有输出")
        elif not params_compatible and loaded_existing:
            self.logger.info("参数不兼容：将重新计算所有输出")
        
        return False
    
    def determine_workers(self) -> int:
        """确定工作进程数"""
        if self.config.workers == -1:
            try:
                workers = int(os.environ.get('SLURM_CPUS_PER_TASK',
                           os.environ.get('SLURM_JOB_CPUS_PER_NODE', mp.cpu_count())))
            except (ValueError, TypeError) as e:
                self.logger.warning("Failed to determine optimal worker count from environment: %s", e)
                workers = max(1, mp.cpu_count())
        else:
            workers = max(1, self.config.workers)
        return workers
    
    def execute_analysis(self, path_manager: PathManager) -> List[tuple]:
        """执行分析（支持仅采样模式）"""
        # 创建分析器
        analyser = SystemAnalyser(
            include_hydrogen=False,
            sample_ratio=self.config.sample_ratio,
            power_p=self.config.power_p,
            pca_variance_ratio=self.config.pca_variance_ratio
        )
        
        # 获取全部分析目标
        all_targets = path_manager.get_all_targets()

        # 过滤掉已有分析结果的系统（增量计算）
        pending_targets = []
        skipped_count = 0
        for target in all_targets:
            if not self.config.force_recompute and ResultSaver.should_skip_analysis(path_manager.output_dir, target.system_name):
                self.logger.info(f"{target.system_name} 体系分析文件存在，跳过分析")
                skipped_count += 1
                continue
            pending_targets.append(target)
        
        if skipped_count > 0:
            self.logger.info(f"增量计算：跳过 {skipped_count} 个已有分析结果的体系")

        if not pending_targets:
            self.logger.info("没有需要处理的系统")
            return []

        # 采样复用判定
        reuse_map = {}
        if hasattr(path_manager, 'output_dir'):
            targets_file = os.path.join(path_manager.output_dir, "analysis_targets.json")
            if os.path.exists(targets_file):
                reuse_map = load_sampling_reuse_map(targets_file)
        self.logger.info(f"采样复用：待处理 {len(pending_targets)} 个体系，可复用 {len(reuse_map)} 个采样帧")

        analysis_targets = pending_targets
        system_paths = [t.system_path for t in analysis_targets]
        
        if not system_paths:
            self.logger.info("没有需要处理的系统")
            return []
        
        workers = self.determine_workers()
        
        self.logger.info(f"准备分析 {len(system_paths)} 个系统...")
        
        # 执行分析
        if workers > 1:
            return self._parallel_analysis(analyser, system_paths, path_manager, workers, reuse_map)
        else:
            return self._sequential_analysis(analyser, system_paths, path_manager, reuse_map)
    
    def _parallel_analysis(self, analyser: SystemAnalyser, system_paths: List[str], 
                          path_manager: PathManager, workers: int, reuse_map: Dict[str, List[int]]) -> List[tuple]:
        """并行分析系统"""
        from src.utils.common import run_parallel_tasks
        initializer_args = (analyser.sample_ratio, analyser.power_p, analyser.pca_variance_ratio, 
                           reuse_map, False, self.log_queue)
        
        # 调用通用并行工具
        results = run_parallel_tasks(
            tasks=system_paths,
            worker_fn=_child_worker,
            workers=workers,
            mode="process",
            initializer=_child_init,
            initargs=initializer_args,
            log_queue=self.log_queue,
            logger=self.logger,
            desc="体系分析"
        )
        
        # 过滤有效结果
        analysis_results = []
        for result in results:
            if result:
                analysis_results.append(result)
        
        return analysis_results
    
    def _sequential_analysis(self, analyser: SystemAnalyser, system_paths: List[str], 
                           path_manager: PathManager, reuse_map: dict = None) -> List[tuple]:
        """顺序分析系统"""
        reuse_map = reuse_map or {}
        path_to_presampled = {}
        for t in path_manager.targets:
            if t.system_path and t.system_name in reuse_map:
                path_to_presampled[t.system_path] = reuse_map[t.system_name]

        analysis_results = []
        
        for i, system_path in enumerate(system_paths):
            try:
                pre_frames_data = path_to_presampled.get(system_path)
                
                # 从字典中提取采样帧列表
                pre_frames = None
                if pre_frames_data and isinstance(pre_frames_data, dict):
                    pre_frames = pre_frames_data.get('sampled_frames')
                
                result = analyser.analyse_system_sampling_only(system_path, pre_sampled_frames=pre_frames)
                
                if result:
                    analysis_results.append(result)
                    system_name = getattr(result[0], 'system_name', os.path.basename(system_path.rstrip('/\\')))
                    
                    self.logger.info(f"({i+1}/{len(system_paths)}) {system_name} 体系分析完成")
                else:
                    self.logger.warning(f"分析失败 ({i+1}/{len(system_paths)}): {system_path}")
            except Exception as e:
                ErrorHandler.log_detailed_error(
                    self.logger, e, f"处理体系 {system_path} 时出错",
                    additional_info={
                        "当前索引": f"({i+1}/{len(system_paths)})",
                        "系统路径": system_path
                    }
                )
        return analysis_results
    
    def _export_sampled_frames(self, result: tuple, system_path: str, system_name: str) -> None:
        """导出采样帧为DeepMD格式（修正：frames为完整帧对象，sampled_frame_ids为帧号）"""
        if len(result) >= 2:
            try:
                sampled_frame_ids = result[1]
                out_root = os.path.join(self.current_output_dir, 'deepmd_npy_per_system')
                # 用dpdata加载完整帧对象
                import dpdata
                ls = dpdata.LabeledSystem(system_path, fmt="abacus/lcao/md")
                # dpdata.LabeledSystem支持下标访问，帧对象可直接传递
                ResultSaver.export_sampled_frames_per_system(
                    frames=ls,
                    sampled_frame_ids=sampled_frame_ids,
                    system_path=system_path,
                    output_root=out_root,
                    system_name=system_name,
                    logger=self.logger,
                    force=self.config.force_recompute
                )
            except Exception as e:
                self.logger.warning(f"DeepMD导出失败 {system_name}: {e}")
    
    def save_results(self, analysis_results: List[tuple], path_manager: PathManager) -> None:
        """保存分析结果"""
        if not analysis_results:
            self.logger.warning("没有分析结果需要保存")
            return

        # 同步采样帧到PathManager.targets
        path_manager.update_sampled_frames_from_results(analysis_results)
        
        # 保存analysis_targets.json
        try:
            current_analysis_params = {
                'sample_ratio': self.config.sample_ratio,
                'power_p': self.config.power_p,
                'pca_variance_ratio': self.config.pca_variance_ratio
            }
            path_manager.save_analysis_targets(current_analysis_params)
        except Exception as e:
            self.logger.warning(f"保存analysis_targets.json时出错: {e}")


def _worker_analyse_system(system_path: str, sample_ratio: float, power_p: float, 
                          pca_variance_ratio: float, pre_sampled_frames: Optional[List[int]] = None,
                          sampling_only: bool = False):
    """备用工作函数：独立创建分析器并执行"""
    analyser = SystemAnalyser(include_hydrogen=False,
                              sample_ratio=sample_ratio,
                              power_p=power_p,
                              pca_variance_ratio=pca_variance_ratio)
    
    if sampling_only:
        return analyser.analyse_system_sampling_only(system_path, pre_sampled_frames=pre_sampled_frames)
    else:
        return analyser.analyse_system(system_path, pre_sampled_frames=pre_sampled_frames)


def _child_init(sample_ratio: float, power_p: float, pca_variance_ratio: float,
                reuse_map: Dict[str, List[int]], sampling_only: bool = False, log_queue: mp.Queue = None):
    """工作进程初始化"""
    analyser = SystemAnalyser(include_hydrogen=False,
                              sample_ratio=sample_ratio,
                              power_p=power_p,
                              pca_variance_ratio=pca_variance_ratio)

    WorkerContext.initialize(analyser, reuse_map)
    WorkerContext.set_sampling_only(sampling_only)

    # Setup multiprocess logging for worker
    if log_queue is not None:
        LoggerManager.setup_worker_logger(
            name="WorkerProcess",
            queue=log_queue,
            level=logging.INFO,
            add_console=False
        )


def _child_worker(system_path: str):
    """工作进程执行函数（支持采样帧复用）"""
    analyser = WorkerContext.get_analyser()

    if analyser is None:
        # 备用情况：如果上下文未初始化，使用默认参数
        return _worker_analyse_system(system_path, 0.05, 0.5, 0.90,
                                    pre_sampled_frames=None, sampling_only=True)

    sys_name = os.path.basename(system_path.rstrip('/\\'))
    reuse_map = WorkerContext.get_reuse_map()
    pre_frames_data = reuse_map.get(sys_name)
    
    # 从字典中提取采样帧列表
    pre_frames = None
    if pre_frames_data and isinstance(pre_frames_data, dict):
        pre_frames = pre_frames_data.get('sampled_frames')

    # 步骤1只执行采样，不包含DeepMD导出
    return analyser.analyse_system_sampling_only(system_path, pre_sampled_frames=pre_frames)


