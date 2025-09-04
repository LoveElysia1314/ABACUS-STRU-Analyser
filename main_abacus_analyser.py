#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ABACUS主分析器程序 - 重构版本
功能：批量分析ABACUS分子动力学轨迹，专注于采样、体系分析和deepmd转化
"""

import os
import time
import argparse
import logging
import multiprocessing as mp
import glob
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# 导入自定义模块
from src.utils.logmanager import LoggerManager
from src.io.path_manager import PathManager, load_sampling_reuse_map, lightweight_discover_systems
from src.core.system_analyser import SystemAnalyser
from src.io.result_saver import ResultSaver
from src.utils.common import ErrorHandler
from src.utils.common import FileUtils


class AnalysisStep(Enum):
    """分析步骤枚举"""
    SAMPLING = 1         # 采样
    DEEPMD = 2          # DeepMD导出
    SAMPLING_COMPARE = 3 # 采样效果对比


class AnalysisMode(Enum):
    """分析模式枚举"""
    FULL_ANALYSIS = "full_analysis"  # 完整分析模式
    SAMPLING_ONLY = "sampling_only"  # 仅采样模式


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
    
    # 模式控制
    mode: AnalysisMode = AnalysisMode.FULL_ANALYSIS
    
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
            cls._analyser._sampling_only = sampling_only


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
    """工作进程执行函数（支持采样帧复用和仅采样模式）"""
    analyser = WorkerContext.get_analyser()

    if analyser is None:
        # 备用情况：如果上下文未初始化，使用默认参数
        return _worker_analyse_system(system_path, 0.05, 0.5, 0.90,
                                    pre_sampled_frames=None, sampling_only=False)

    sys_name = os.path.basename(system_path.rstrip('/\\'))
    reuse_map = WorkerContext.get_reuse_map()
    pre_frames = reuse_map.get(sys_name)

    # 检查是否为仅采样模式
    sampling_only = getattr(analyser, '_sampling_only', False)

    if sampling_only:
        return analyser.analyse_system_sampling_only(system_path, pre_sampled_frames=pre_frames)
    else:
        return analyser.analyse_system(system_path, pre_sampled_frames=pre_frames)


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
    
    def _sync_sampled_frames_to_targets(self, result: tuple, path_manager: PathManager) -> None:
        """同步采样帧到PathManager.targets"""
        if len(result) >= 2:
            metrics = result[0]
            system_name_curr = getattr(metrics, 'system_name', 'unknown')
            sampled_frames_curr = getattr(metrics, 'sampled_frames', [])
            for target in path_manager.targets:
                if target.system_name == system_name_curr:
                    target.sampled_frames = sampled_frames_curr
                    break
    
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
        path_manager.load_from_discovery(search_paths, preserve_existing=loaded_existing)
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

        # 检查已存在的分析结果，过滤需要跳过的体系
        pending_targets = []
        skipped_targets = []
        
        for target in all_targets:
            if not self.config.force_recompute:
                should_skip, reason = ResultSaver.should_skip_analysis(self.current_output_dir, target.system_name)
                if should_skip:
                    self.logger.info(f"{target.system_name} 体系{reason}，跳过分析")
                    skipped_targets.append(target)
                else:
                    pending_targets.append(target)
            else:
                pending_targets.append(target)

        if skipped_targets:
            self.logger.info(f"跳过 {len(skipped_targets)} 个体系")

        if not pending_targets:
            self.logger.info(f"所有 {len(all_targets)} 个体系均已处理，无需重复分析")
            return []

        # 采样复用判定仅对待处理体系执行
        reuse_map = path_manager.determine_sampling_reuse()
        self.logger.info(f"采样复用：待处理 {len(pending_targets)} 个体系，可复用 {len(reuse_map)} 个采样帧")

        analysis_targets = pending_targets
        system_paths = [t.system_path for t in analysis_targets]
        
        if not system_paths:
            self.logger.info("没有需要处理的系统")
            return []
        
        workers = self.determine_workers()
        
        # 记录模式信息
        mode_info = "仅采样模式" if self.config.mode == AnalysisMode.SAMPLING_ONLY else "完整分析模式"
        self.logger.info(f"准备分析 {len(system_paths)} 个系统 [{mode_info}]...")
        
        # 执行分析
        if workers > 1:
            return self._parallel_analysis(analyser, system_paths, path_manager, workers, reuse_map)
        else:
            return self._sequential_analysis(analyser, system_paths, path_manager, reuse_map)
    
    def _parallel_analysis(self, analyser: SystemAnalyser, system_paths: List[str], 
                          path_manager: PathManager, workers: int, reuse_map: Dict[str, List[int]]) -> List[tuple]:
        """并行分析系统"""
        from src.utils.parallel_utils import run_parallel_tasks
        sampling_only = self.config.mode == AnalysisMode.SAMPLING_ONLY
        initializer_args = (analyser.sample_ratio, analyser.power_p, analyser.pca_variance_ratio, 
                           reuse_map, sampling_only, self.log_queue)
        
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
        
        # 过滤有效结果并导出DeepMD数据
        analysis_results = []
        for result in results:
            if result:
                analysis_results.append(result)
                if len(result) >= 2:
                    try:
                        system_name = result[0].system_name
                        system_path = result[0].system_path
                        self._export_sampled_frames(result, system_path, system_name)
                    except (IndexError, AttributeError):
                        pass
        
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
        sampling_only = self.config.mode == AnalysisMode.SAMPLING_ONLY
        
        for i, system_path in enumerate(system_paths):
            try:
                pre_frames = path_to_presampled.get(system_path)
                
                if sampling_only:
                    result = analyser.analyse_system_sampling_only(system_path, pre_sampled_frames=pre_frames)
                else:
                    result = analyser.analyse_system(system_path, pre_sampled_frames=pre_frames)
                
                if result:
                    analysis_results.append(result)
                    system_name = getattr(result[0], 'system_name', os.path.basename(system_path.rstrip('/\\')))
                    mode_suffix = "[仅采样]" if sampling_only else ""
                    
                    self.logger.info(f"({i+1}/{len(system_paths)}) {system_name} 体系分析完成 {mode_suffix}")
                    
                    # 即时导出DeepMD数据
                    self._export_sampled_frames(result, system_path, system_name)
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
    
    def _handle_skipped_systems_deepmd_export(self, skipped_targets: List) -> None:
        """处理被跳过体系的DeepMD导出"""
        if not skipped_targets:
            return
            
        self.logger.info(f"检查 {len(skipped_targets)} 个跳过体系的DeepMD导出需求...")
        
        for target in skipped_targets:
            # 检查是否需要导出DeepMD数据
            should_skip, reason = ResultSaver.should_skip_analysis(self.current_output_dir, target.system_name)
            
            # 如果只有采样结果，需要导出DeepMD
            if reason == "采样结果已存在":
                try:
                    self.logger.info(f"{target.system_name} 需要补充DeepMD导出")
                    # 从analysis_targets.json中获取采样帧信息
                    targets_file = os.path.join(self.current_output_dir, "analysis_targets.json")
                    if os.path.exists(targets_file):
                        with open(targets_file, 'r', encoding='utf-8') as f:
                            targets_data = json.load(f)
                        
                        # 查找对应体系的采样帧
                        sampled_frames = None
                        for mol_name, mol_data in targets_data.get('molecules', {}).items():
                            for sys_name, sys_data in mol_data.get('systems', {}).items():
                                if sys_name == target.system_name:
                                    sampled_frames = sys_data.get('sampled_frames')
                                    break
                            if sampled_frames is not None:
                                break
                        
                        if sampled_frames:
                            # 执行DeepMD导出
                            out_root = os.path.join(self.current_output_dir, 'deepmd_npy_per_system')
                            ResultSaver.export_sampled_frames_per_system(
                                frames=[],  # 对于跳过的体系，我们没有frames数据
                                sampled_frame_ids=sampled_frames if isinstance(sampled_frames, list) else json.loads(sampled_frames),
                                system_path=target.system_path,
                                output_root=out_root,
                                system_name=target.system_name,
                                logger=self.logger,
                                force=False
                            )
                except Exception as e:
                    self.logger.warning(f"{target.system_name} DeepMD导出失败: {e}")


class WorkflowExecutor:
    """工作流执行器 - 提取公共方法降低冗余"""
    
    def __init__(self, orchestrator: AnalysisOrchestrator):
        self.orchestrator = orchestrator
    
    def execute_sampling_step(self, config: AnalysisConfig) -> Tuple[List[tuple], PathManager, str]:
        """执行采样步骤"""
        search_paths = self.orchestrator.resolve_search_paths()
        records = lightweight_discover_systems(search_paths, include_project=config.include_project)
        if not records:
            self.orchestrator.logger.error("轻量发现未找到体系")
            return [], None, ""
        
        # 输出目录
        path_manager, actual_output_dir = self.orchestrator.setup_output_directory()
        
        # 构建 mol_systems 用于 path_manager 兼容保存采样信息
        mol_systems: Dict[str, List[str]] = {}
        for rec in records:
            mol_systems.setdefault(rec.mol_id, []).append(rec.system_path)
        
        # 修复：确保analysis_targets.json被正确加载以启用复用
        self.orchestrator.setup_analysis_targets(path_manager, search_paths)
        
        # 采样复用 map
        reuse_map_raw = {}
        if path_manager.targets_file:
            reuse_map_raw = load_sampling_reuse_map(path_manager.targets_file)
        
        # 构建任务
        analyser_params = {
            'sample_ratio': config.sample_ratio,
            'power_p': config.power_p,
            'pca_variance_ratio': config.pca_variance_ratio
        }
        workers = self.orchestrator.determine_workers()
        
        analysis_results: List[tuple] = []
        # 迁移为通用任务准备与结果处理工具
        from src.utils.parallel_utils_ext import prepare_parallel_tasks, process_parallel_results
        tasks, stats = prepare_parallel_tasks(
            records, config, path_manager, ResultSaver, logger=self.orchestrator.logger
        )
        
        if stats.get('tasks', 0) > 0:
            self.orchestrator.logger.info(
                f"进程模式任务: {stats['tasks']} (复用采样 {stats['reused']}, 跳过 {stats['skipped']}, 仅导出deepmd {stats['deepmd_only']}, 复用采样补分析 {stats['reuse_sampling_analysis']})"
            )
        
        # 运行任务
        from src.core.process_scheduler import _worker
        from concurrent.futures import ProcessPoolExecutor, as_completed
        results = []
        failures = 0
        completed_count = 0
        total_count = len(tasks)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_task = {}
            for task in tasks:
                future = pool.submit(_worker, task, analyser_params)
                future_to_task[future] = task
            for future in as_completed(future_to_task):
                completed_count += 1
                task = future_to_task[future]
                try:
                    sys_name, result, dur = future.result()
                    if result is None or (isinstance(result, tuple) and result[0] is None):
                        failures += 1
                        self.orchestrator.logger.warning(f"({completed_count}/{total_count}) {sys_name} 体系分析失败 (用时 {dur:.2f}s)")
                    else:
                        results.append(result)
                        mode_suffix = "[仅采样]" if config.mode == AnalysisMode.SAMPLING_ONLY else ""
                        self.orchestrator.logger.info(f"({completed_count}/{total_count}) {sys_name} 体系分析完成 {mode_suffix}")
                except Exception as e:
                    failures += 1
                    self.orchestrator.logger.error(f"({completed_count}/{total_count}) {task.system_name} 体系分析异常: {e}")
        
        if total_count > 0:
            self.orchestrator.logger.info(f"进程调度完成: 成功 {len(results)} 失败 {failures} 总耗时 {time.time()-t0:.1f}s")
        
        analysis_results = [r for r in results if r]
        # 统一结果处理
        process_parallel_results(analysis_results, actual_output_dir, config, self.orchestrator.logger, ResultSaver)
        
        # 同步采样帧与保存 targets
        try:
            path_manager.update_sampled_frames_from_results(analysis_results)
            path_manager.save_analysis_targets({
                'sample_ratio': config.sample_ratio,
                'power_p': config.power_p,
                'pca_variance_ratio': config.pca_variance_ratio
            })
        except Exception as e:
            self.orchestrator.logger.warning(f"保存 targets 失败(忽略): {e}")
        
        return analysis_results, path_manager, actual_output_dir
    
    def execute_deepmd_step(self, analysis_results: List[tuple], path_manager: PathManager, actual_output_dir: str) -> None:
        """执行DeepMD导出步骤"""
        self.orchestrator.logger.info("开始独立DeepMD数据导出...")
        
        # 从已有的采样结果中导出DeepMD数据
        deepmd_count = 0
        for result in analysis_results:
            if result and len(result) >= 2:
                try:
                    system_name = result[0].system_name if hasattr(result[0], 'system_name') else str(result[0])
                    system_path = result[0].system_path if hasattr(result[0], 'system_path') else None
                    
                    # 调用已有的DeepMD导出方法
                    if system_path:
                        self.orchestrator._export_sampled_frames(result, system_path, system_name)
                        deepmd_count += 1
                        self.orchestrator.logger.debug(f"独立DeepMD导出完成: {system_name}")
                except Exception as e:
                    self.orchestrator.logger.error(f"独立DeepMD导出失败 {system_name}: {e}")
        
        self.orchestrator.logger.info(f"独立DeepMD导出完成，处理了 {deepmd_count} 个体系")
    
    def execute_sampling_compare_step(self, analysis_results: List[tuple], path_manager: PathManager, actual_output_dir: str) -> None:
        """执行采样效果对比步骤"""
        self.orchestrator.logger.info("开始采样效果对比分析...")
        
        try:
            # 导入采样对比模块
            from src.analysis.sampling_comparison import analyse_sampling_compare
            
            # 获取工作进程数
            workers = self.orchestrator.determine_workers()
            
            # 执行对比分析
            analyse_sampling_compare(result_dir=actual_output_dir, workers=workers)
            self.orchestrator.logger.info("采样效果对比分析完成")
            
        except ImportError as e:
            self.orchestrator.logger.error(f"无法导入采样对比模块: {e}")
        except Exception as e:
            self.orchestrator.logger.error(f"采样效果对比分析失败: {e}")
    
    def load_existing_results_for_deepmd(self, config: AnalysisConfig) -> Tuple[List[tuple], PathManager, str]:
        """为DeepMD导出加载已有的分析结果，确保正确解析 system_path 和 sampled_frames"""
        self.orchestrator.logger.info("加载已有结果用于DeepMD导出...")
        
        # 设置输出目录
        path_manager, actual_output_dir = self.orchestrator.setup_output_directory()
        
        # 查找analysis_targets.json文件
        targets_file = os.path.join(actual_output_dir, "analysis_targets.json")
        if not os.path.exists(targets_file):
            self.orchestrator.logger.error(f"未找到analysis_targets.json文件: {targets_file}")
            return [], path_manager, actual_output_dir
        
        # 读取采样数据
        try:
            with open(targets_file, 'r', encoding='utf-8') as f:
                targets_data = json.load(f)
            
            analysis_results = []
            molecules = targets_data.get('molecules', {})
            
            for mol_id, mol_data in molecules.items():
                systems = mol_data.get('systems', {})
                for system_name, system_data in systems.items():
                    system_path = system_data.get('system_path', '')
                    sampled_frames = []
                    # robustly parse sampled_frames (may be str or list)
                    sampled_frames_raw = system_data.get('sampled_frames', [])
                    if isinstance(sampled_frames_raw, str):
                        try:
                            sampled_frames = json.loads(sampled_frames_raw)
                        except Exception:
                            self.orchestrator.logger.warning(f"无法解析采样帧数据: {system_name}, 内容: {sampled_frames_raw}")
                            sampled_frames = []
                    elif isinstance(sampled_frames_raw, list):
                        sampled_frames = sampled_frames_raw
                    else:
                        sampled_frames = []
                    # 过滤非int类型
                    sampled_frames = [int(x) for x in sampled_frames if isinstance(x, int) or (isinstance(x, float) and x.is_integer())]
                    if not system_path or not sampled_frames:
                        self.orchestrator.logger.warning(f"系统数据不完整，跳过: {system_name}")
                        continue
                    # 解析系统信息
                    import re
                    match = re.match(r"struct_mol_(\d+)_conf_(\d+)_T(\d+)K", system_name)
                    if match:
                        mol_id_parsed = match.group(1)
                        conf = match.group(2)
                        temperature = match.group(3)
                    else:
                        self.orchestrator.logger.warning(f"系统名称格式不正确: {system_name}")
                        mol_id_parsed = mol_id
                        conf = "0"
                        temperature = "300"
                    # 创建TrajectoryMetrics对象
                    from src.core.metrics import TrajectoryMetrics
                    metrics = TrajectoryMetrics(
                        system_name=system_name,
                        mol_id=mol_id_parsed,
                        conf=conf,
                        temperature=temperature,
                        system_path=system_path
                    )
                    metrics.sampled_frames = sampled_frames
                    metrics.num_frames = len(sampled_frames)
                    # 构造分析结果元组（格式: (metrics, sampled_frames)）
                    result = (metrics, sampled_frames)
                    analysis_results.append(result)
                    self.orchestrator.logger.info(f"加载采样数据: {system_name}, 采样帧数: {len(sampled_frames)}")
            self.orchestrator.logger.info(f"成功加载 {len(analysis_results)} 个系统的采样数据")
        except Exception as e:
            self.orchestrator.logger.error(f"读取analysis_targets.json文件失败: {e}")
            return [], path_manager, actual_output_dir
        return analysis_results, path_manager, actual_output_dir
    
    def load_existing_results_for_compare(self, config: AnalysisConfig) -> Tuple[List[tuple], PathManager, str]:
        """为采样对比加载已有的分析结果"""
        self.orchestrator.logger.info("准备采样效果对比（从已有结果目录）...")
        
        # 设置输出目录
        path_manager, actual_output_dir = self.orchestrator.setup_output_directory()
        
        # 检查是否存在必要的文件
        targets_file = os.path.join(actual_output_dir, "analysis_targets.json")
        if not os.path.exists(targets_file):
            self.orchestrator.logger.warning(f"未找到analysis_targets.json文件: {targets_file}")
            self.orchestrator.logger.info("采样对比将尝试从结果目录中查找可用数据")
        
        # 对于采样对比，我们返回一个空的分析结果列表，
        # 因为analyse_sampling_compare函数会直接从结果目录工作
        analysis_results = [("placeholder",)]  # 非空列表表示有数据可用
        
        return analysis_results, path_manager, actual_output_dir
    
    def finalize_analysis(self, analysis_results: List[tuple], path_manager: PathManager, 
                         start_time: float, output_dir: str) -> None:
        """完成分析并输出统计信息"""
        # 保存分析目标状态
        try:
            current_analysis_params = {
                'sample_ratio': self.orchestrator.config.sample_ratio,
                'power_p': self.orchestrator.config.power_p,
                'pca_variance_ratio': self.orchestrator.config.pca_variance_ratio
            }
            path_manager.save_analysis_targets(current_analysis_params)
        except Exception as e:
            self.orchestrator.logger.error(f"保存分析目标失败: {str(e)}")

        # 输出最终统计
        elapsed = time.time() - start_time
        mode_map = {
            AnalysisMode.FULL_ANALYSIS: "完整分析模式",
            AnalysisMode.SAMPLING_ONLY: "仅采样模式", 
        }
        mode_info = mode_map.get(self.orchestrator.config.mode, "未知模式")
        
        total_targets = len(path_manager.targets)
        analysed_count = len(analysis_results)

        self.orchestrator.logger.info("=" * 60)
        self.orchestrator.logger.info(f"分析完成! [{mode_info}] 实际分析 {analysed_count}/{total_targets} 个体系")

        if analysis_results and self.orchestrator.config.mode == AnalysisMode.FULL_ANALYSIS:
            # 输出采样统计
            swap_counts = [result[2] for result in analysis_results if len(result) > 2]
            if swap_counts:
                import numpy as np
                self.orchestrator.logger.info("采样优化统计:")
                self.orchestrator.logger.info("  平均交换次数: %.2f", np.mean(swap_counts))
                self.orchestrator.logger.info(f"  总交换次数: {int(sum(swap_counts))}")

        self.orchestrator.logger.info(f"总耗时: {elapsed:.1f}s ({elapsed/60:.1f} 分钟)")
        self.orchestrator.logger.info(f"结果目录: {output_dir}")
        
        if self.orchestrator.config.mode == AnalysisMode.SAMPLING_ONLY:
            self.orchestrator.logger.info(f"DeepMD数据: {os.path.join(output_dir, 'deepmd_npy_per_system')}")
        
        self.orchestrator.logger.info("=" * 60)


class MainApp:
    """主应用程序类 - 重构版本"""
    
    def __init__(self):
        self.orchestrator = None
        self.workflow_executor = None
    
    def run(self) -> None:
        """运行主程序"""
        start_time = time.time()
        
        try:
            # 解析参数并创建配置
            config, args = self._parse_arguments_to_config()
            
            # 创建编排器
            self.orchestrator = AnalysisOrchestrator(config)
            
            # 创建工作流执行器
            self.workflow_executor = WorkflowExecutor(self.orchestrator)
            
            # 设置日志
            self.orchestrator.setup_logging()
            
            # 记录启动信息
            self._log_startup_info(config)
            
            # 执行主要分析流程
            self._execute_workflow(config, start_time)
            
        except Exception as e:
            if self.orchestrator and self.orchestrator.logger:
                self.orchestrator.logger.error(f"主程序执行出错: {str(e)}")
                import traceback
                self.orchestrator.logger.error(f"详细错误信息: {traceback.format_exc()}")
            else:
                import sys
                print(f"主程序执行出错: {str(e)}", file=sys.stderr)
                import traceback
                print(traceback.format_exc(), file=sys.stderr)
        finally:
            if self.orchestrator:
                self.orchestrator.cleanup_logging()
    
    def _parse_arguments_to_config(self) -> Tuple[AnalysisConfig, argparse.Namespace]:
        """解析命令行参数并创建配置对象"""
        parser = argparse.ArgumentParser(
            description='ABACUS分子动力学轨迹分析器 - 支持采样、DeepMD导出和采样对比',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
流程步骤说明:
  1 - 采样: 执行体系发现、采样、统计分析和DeepMD导出
  2 - 独立DeepMD导出: 仅将已有采样结果导出为DeepMD格式（当未执行步骤1时）
  3 - 采样效果对比: 执行不同采样方法的效果对比分析

使用示例:
  python main_abacus_analyser.py                    # 执行所有步骤 [1,2,3]
  python main_abacus_analyser.py --steps 1          # 仅执行采样（含DeepMD导出）
  python main_abacus_analyser.py --steps 2          # 仅执行独立DeepMD导出
  python main_abacus_analyser.py --steps 1 3        # 执行采样和采样对比
  python main_abacus_analyser.py --steps 2 3        # 执行独立DeepMD导出和采样对比
  python main_abacus_analyser.py --steps 1 2        # 执行采样（步骤2会被跳过）

注意: 当同时指定步骤1和2时，步骤2会被自动跳过，因为步骤1已包含DeepMD导出。
            """
        )

        # 核心参数
        parser.add_argument('-r', '--sample_ratio', type=float, default=0.1, help='采样比例')
        parser.add_argument('-p', '--power_p', type=float, default=-0.5, help='幂平均距离的p值')
        parser.add_argument('-v', '--pca_variance_ratio', type=float, default=0.95, help='PCA降维累计方差贡献率')

        # 运行配置
        parser.add_argument('-w', '--workers', type=int, default=-1, help='并行工作进程数')
        parser.add_argument('-o', '--output_dir', type=str, default='analysis_results', help='输出根目录')
        parser.add_argument('-s', '--search_path', nargs='*', default=None, help='递归搜索路径')
        parser.add_argument('-i', '--include_project', action='store_true', help='允许搜索项目自身目录')
        parser.add_argument('-f', '--force_recompute', action='store_true', help='强制重新计算')

        # 流程控制
        parser.add_argument('--steps', nargs='*', type=int, default=None, 
                          help='要执行的分析步骤：1=采样, 2=DeepMD导出, 3=采样效果对比。默认: [1,2,3]。示例: --steps 1 3')
        
        # 模式控制
        mode_group = parser.add_mutually_exclusive_group()
        mode_group.add_argument('--sampling_only', action='store_true', help='仅采样模式：只执行采样算法，不计算统计指标（等价于 --steps 1）')
        
        # 附加功能
        parser.add_argument('--sampling_compare', action='store_true', help='执行采样效果比较分析（等价于 --steps 3）')

        args = parser.parse_args()

        # 确定要执行的步骤
        steps_to_run = []
        
        if args.steps is not None:
            # 用户显式指定了步骤
            steps_to_run = args.steps
        elif args.sampling_only:
            # 向后兼容：仅采样模式
            steps_to_run = [1]
        elif args.sampling_compare:
            # 向后兼容：采样效果对比
            steps_to_run = [3]
        else:
            # 默认执行所有步骤
            steps_to_run = [1, 2, 3]
        
        # 验证步骤编号
        valid_steps = [1, 2, 3]
        invalid_steps = [s for s in steps_to_run if s not in valid_steps]
        if invalid_steps:
            raise ValueError(f"无效的步骤编号: {invalid_steps}。有效步骤: {valid_steps}")
        
        # 确定运行模式
        if 1 in steps_to_run and len(steps_to_run) == 1:
            run_mode = AnalysisMode.SAMPLING_ONLY
        else:
            run_mode = AnalysisMode.FULL_ANALYSIS

        config = AnalysisConfig(
            sample_ratio=args.sample_ratio,
            power_p=args.power_p,
            pca_variance_ratio=args.pca_variance_ratio,
            workers=args.workers,
            output_dir=args.output_dir,
            search_paths=args.search_path or [],
            include_project=args.include_project,
            force_recompute=args.force_recompute,
            steps=steps_to_run,
            mode=run_mode
        )

        return config, args

    def _log_startup_info(self, config: AnalysisConfig) -> None:
        """记录启动信息"""
        search_paths = self.orchestrator.resolve_search_paths()
        search_info = f"搜索路径: {search_paths if search_paths else '(当前目录的父目录)'}"

        # 步骤信息
        step_names = {1: "采样(含DeepMD)", 2: "独立DeepMD导出", 3: "采样效果对比"}
        steps_info = [f"{step}({step_names.get(step, '未知')})" for step in config.steps]
        steps_str = ",".join(steps_info)
        
        workers = self.orchestrator.determine_workers()

        self.orchestrator.logger.info(
            f"ABACUS主分析器启动 [执行步骤: {steps_str}] | 采样比例: {config.sample_ratio} | 工作进程: {workers}"
        )
        self.orchestrator.logger.info(search_info)
        self.orchestrator.logger.info(
            f"项目目录屏蔽: {'关闭' if config.include_project else '开启'}"
        )
        self.orchestrator.logger.info(
            f"强制重新计算: {'是' if config.force_recompute else '否'}"
        )

        if config.mode == AnalysisMode.SAMPLING_ONLY:
            self.orchestrator.logger.info("仅采样模式：将跳过统计指标计算")

    
    def _execute_workflow(self, config: AnalysisConfig, start_time: float) -> None:
        """执行主要工作流程（基于步骤列表）"""
        self.orchestrator.logger.info(f"开始执行分析流程，步骤: {config.steps}")
        
        # 初始化共享数据
        analysis_results = []
        path_manager = None
        actual_output_dir = None
        
        # 步骤1: 采样（包含DeepMD导出）
        if 1 in config.steps:
            self.orchestrator.logger.info("执行步骤1: 采样（包含DeepMD导出）")
            analysis_results, path_manager, actual_output_dir = self.workflow_executor.execute_sampling_step(config)
        
        # 步骤2: 独立DeepMD导出（仅当步骤1未执行时）
        if 2 in config.steps and 1 not in config.steps:
            self.orchestrator.logger.info("执行步骤2: 独立DeepMD导出")
            analysis_results, path_manager, actual_output_dir = self.workflow_executor.load_existing_results_for_deepmd(config)
            if analysis_results:
                self.workflow_executor.execute_deepmd_step(analysis_results, path_manager, actual_output_dir)
            else:
                self.orchestrator.logger.warning("步骤2: 未找到可用的采样数据，跳过DeepMD导出")
        elif 2 in config.steps and 1 in config.steps:
            self.orchestrator.logger.info("步骤2: 跳过独立DeepMD导出（已在步骤1中完成）")
        
        # 步骤3: 采样效果对比
        if 3 in config.steps:
            self.orchestrator.logger.info("执行步骤3: 采样效果对比")
            if analysis_results:  # 如果前面的步骤已执行
                self.workflow_executor.execute_sampling_compare_step(analysis_results, path_manager, actual_output_dir)
            else:  # 如果只执行步骤3，需要加载已有数据
                analysis_results, path_manager, actual_output_dir = self.workflow_executor.load_existing_results_for_compare(config)
                if analysis_results:
                    self.workflow_executor.execute_sampling_compare_step(analysis_results, path_manager, actual_output_dir)
                else:
                    self.orchestrator.logger.warning("步骤3: 未找到可用的分析数据，跳过采样效果对比")
        
        # 最终统计和清理
        if analysis_results and path_manager:
            self.workflow_executor.finalize_analysis(analysis_results, path_manager, start_time, actual_output_dir)


if __name__ == "__main__":
    app = MainApp()
    app.run()