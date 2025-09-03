#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ABACUS主分析器程序 - 重构版本
功能：批量分析ABACUS分子动力学轨迹，支持智能采样和相关性分析
新增：支持仅采样模式和完整分析模式
"""

import os
import time
import argparse
import logging
import multiprocessing as mp
import glob
import json
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

# 导入自定义模块
from src.utils.logmanager import LoggerManager
from src.io.path_manager import PathManager, load_sampling_reuse_map, lightweight_discover_systems
from src.core.system_analyser import SystemAnalyser
from src.io.result_saver import ResultSaver
from src.utils.common import ErrorHandler
from src.utils.common import FileUtils
from src.analysis.sampling_comparison.streaming_compare import StreamingSamplingComparisonManager
from src.core.process_scheduler import ProcessScheduler, ProcessAnalysisTask

# 采样效果评估（采样对比）
try:
    from src.analysis.sampling_comparison import analyse_sampling_compare as SamplingComparisonRunner
    SAMPLING_COMPARISON_AVAILABLE = True
except ImportError:
    SAMPLING_COMPARISON_AVAILABLE = False

try:
    from src.analysis.correlation_analyser import CorrelationAnalyser as ExternalCorrelationAnalyser
    CORRELATION_ANALYSER_AVAILABLE = True
except ImportError:
    CORRELATION_ANALYSER_AVAILABLE = False


class AnalysisMode(Enum):
    """分析模式枚举"""
    SAMPLING_ONLY = "sampling_only"  # 仅采样模式
    FULL_ANALYSIS = "full_analysis"  # 完整分析模式


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
    
    # 模式控制
    mode: AnalysisMode = AnalysisMode.FULL_ANALYSIS
    dry_run_reuse: bool = False
    enable_sampling_eval: bool = True
    # 调度器: legacy(旧路径) / process(进程池单体系单核) / thread(线程池轻量)
    scheduler: str = 'legacy'
    
    def __post_init__(self):
        if self.search_paths is None:
            self.search_paths = []


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
        # 流式输出控制
        self.streaming_enabled: bool = True
        self._stream_flush_counter: int = 0
        self._targets_flush_interval: int = 1  # 后续可参数化
        
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
                    self.logger.warning("通配符 '%s' 未匹配到任何目录", path_pattern)
            else:
                if os.path.isdir(path_pattern):
                    resolved_paths.append(path_pattern)
                else:
                    self.logger.warning("路径不存在或不是目录: %s", path_pattern)
        
        unique_paths = list(set(os.path.abspath(p) for p in resolved_paths))
        return unique_paths
    
    def discover_systems(self, search_paths: List[str]) -> dict:
        """从多个搜索路径发现ABACUS系统"""
        all_mol_systems = {}
        
        for search_path in search_paths:
            self.logger.info("搜索路径: %s", search_path)
            try:
                if self.config.include_project:
                    mol_systems = FileUtils.find_abacus_systems(search_path, include_project=True)
                else:
                    mol_systems = FileUtils.find_abacus_systems(search_path)
                
                for mol_key, system_paths in mol_systems.items():
                    if mol_key in all_mol_systems:
                        all_mol_systems[mol_key].extend(system_paths)
                    else:
                        all_mol_systems[mol_key] = system_paths
                
                self.logger.info("在 %s 中发现 %d 个分子类型", search_path, len(mol_systems))
            except (OSError, PermissionError) as e:
                self.logger.error("搜索路径 %s 时出错: %s", search_path, e)
        
        return all_mol_systems
    
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
        # 初始化流式采样比较管理器（仅完整模式 + 采样评估启用）
        if (self.streaming_enabled and self.config.mode == AnalysisMode.FULL_ANALYSIS \
            and self.config.enable_sampling_eval):
            try:
                self.sampling_stream_manager = StreamingSamplingComparisonManager(actual_output_dir, logger=self.logger)
            except (ImportError, AttributeError) as e:
                self.logger.warning("初始化流式采样比较失败，回退离线模式: %s", e)
                self.sampling_stream_manager = None
        
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

        # 已处理体系过滤（先过滤再做复用统计，使日志一致）
        processed_systems: Set[str] = set()
        if not self.config.force_recompute:
            progress_info = ResultSaver.load_progress(self.current_output_dir)
            processed_systems = set(progress_info.get('processed_systems', []))

        pending_targets = [t for t in all_targets if t.system_name not in processed_systems]

        if not pending_targets:
            # 全部已处理 / 无目标：精简输出
            self.logger.info(
                f"所有 {len(all_targets)} 个体系均已处理，无需重复分析 (force_recompute=False)。"
            )
            return []

        # 采样复用判定仅对待处理体系执行
        reuse_map = path_manager.determine_sampling_reuse()
        self.logger.info(
            f"采样复用：待处理 {len(pending_targets)} 个体系，可复用 {len(reuse_map)} 个采样帧 (全部: {len(all_targets)}, 已处理: {len(processed_systems)})"
        )

        # Dry-run检查
        if self.config.dry_run_reuse:
            self._output_reuse_plan(reuse_map, path_manager)
            return []

        analysis_targets = pending_targets
        system_paths = [t.system_path for t in analysis_targets]
        
        # 经过上方 pending 判定，这里 system_paths 一定非空；保留保护
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
    
    def _output_reuse_plan(self, reuse_map: dict, path_manager: PathManager) -> None:
        """输出采样复用计划"""
        plan = {
            "total_targets": len(path_manager.targets),
            "reused_sampling_systems": list(reuse_map.keys()),
            "reused_count": len(reuse_map),
            "resample_systems": [t.system_name for t in path_manager.targets if t.system_name not in reuse_map]
        }
        plan_path = os.path.join(self.current_output_dir, 'sampling_reuse_plan.json')
        with open(plan_path, 'w', encoding='utf-8') as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Dry-Run 完成：已生成采样复用计划 {plan_path} ，复用 {plan['reused_count']} 个，重采样 {len(plan['resample_systems'])} 个。")
    
    def _parallel_analysis(self, analyser: SystemAnalyser, system_paths: List[str], 
                          path_manager: PathManager, workers: int, reuse_map: Dict[str, List[int]]) -> List[tuple]:
        """并行分析系统"""
        analysis_results = []
        sampling_only = self.config.mode == AnalysisMode.SAMPLING_ONLY
        initializer_args = (analyser.sample_ratio, analyser.power_p, analyser.pca_variance_ratio, 
                           reuse_map, sampling_only, self.log_queue)
        chunksize = 1

        with mp.Pool(processes=workers, initializer=_child_init, initargs=initializer_args) as pool:
            try:
                for i, result in enumerate(pool.imap_unordered(_child_worker, system_paths, chunksize=chunksize)):
                    if result:
                        analysis_results.append(result)
                        
                        # 获取系统信息
                        try:
                            system_name = result[0].system_name
                            system_path = result[0].system_path
                        except (IndexError, AttributeError):
                            system_path = system_paths[i] if i < len(system_paths) else None
                            system_name = os.path.basename(system_path.rstrip('/\\')) if system_path else "未知"

                        mode_suffix = "[仅采样]" if sampling_only else ""
                        self.logger.info("分析完成 (已完成 %d/%d): %s %s", len(analysis_results), len(system_paths), system_name, mode_suffix)
                        
                        # 即时导出（如果有采样结果）
                        if system_path and len(result) >= 2:
                            self._export_sampled_frames(result, system_path, system_name)

                        # 流式保存单体系结果
                        if self.streaming_enabled:
                            try:
                                # 立即同步当前系统的采样帧到PathManager.targets
                                self._sync_sampled_frames_to_targets(result, path_manager)
                                
                                ResultSaver.save_single_system(
                                    output_dir=self.current_output_dir,
                                    result=result,
                                    sampling_only=sampling_only,
                                    flush_targets_hook=(lambda: path_manager.save_analysis_targets({
                                        'sample_ratio': self.config.sample_ratio,
                                        'power_p': self.config.power_p,
                                        'pca_variance_ratio': self.config.pca_variance_ratio
                                    })) if (self._targets_flush_interval == 1 or ((len(analysis_results) % self._targets_flush_interval) == 0)) else None
                                )
                            except (IOError, OSError) as se:
                                self.logger.warning("流式保存体系结果失败(忽略): %s", se)
                        # 流式采样比较
                        if (not sampling_only) and getattr(self, 'sampling_stream_manager', None) and len(result) >= 7:
                            try:
                                metrics = result[0]
                                frames = result[1]
                                pca_components_data = result[4]
                                # 构建 vectors: energy_standardized + PCs
                                # 建立 frame_id -> energy_standardized
                                id2energy = {f.frame_id: getattr(f, 'energy_standardized', None) for f in frames}
                                vectors_list = []
                                frame_ids = []
                                # 获取最大 PC 数
                                max_pc = 0
                                for item in pca_components_data:
                                    for k in item.keys():
                                        if k.startswith('PC'):
                                            idx_pc = int(k[2:])
                                            if idx_pc > max_pc:
                                                max_pc = idx_pc
                                for item in pca_components_data:
                                    fid = item.get('frame')
                                    if fid is None:
                                        continue
                                    energy_std = id2energy.get(fid)
                                    if energy_std is None:
                                        continue
                                    vec = [energy_std]
                                    for i_pc in range(1, max_pc+1):
                                        vec.append(item.get(f'PC{i_pc}', 0.0))
                                    vectors_list.append(vec)
                                    frame_ids.append(fid)
                                import numpy as _np
                                if vectors_list:
                                    vectors_arr = _np.array(vectors_list, dtype=float)
                                    frame_ids_arr = _np.array(frame_ids, dtype=int)
                                    sampled_set = set(getattr(metrics, 'sampled_frames', []))
                                    sampled_mask = _np.array([fid in sampled_set for fid in frame_ids_arr])
                                    self.sampling_stream_manager.update_per_system(
                                        system_name=metrics.system_name,
                                        system_path=system_path,
                                        vectors=vectors_arr,
                                        sampled_mask=sampled_mask,
                                        frame_ids=frame_ids_arr
                                    )
                            except (AttributeError, ValueError) as ce:
                                self.logger.warning("流式采样比较更新失败(忽略): %s", ce)
                    else:
                        self.logger.warning("并行分析返回空结果，标记为失败")
            except (RuntimeError, ValueError) as e:
                ErrorHandler.log_detailed_error(
                    self.logger, e, "并行处理出错",
                    additional_info={
                        "工作进程数": workers,
                        "系统路径数量": len(system_paths),
                        "已完成数量": len(analysis_results)
                    }
                )
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
                    mode = "复用采样" if pre_frames else "重新采样"
                    mode_suffix = "[仅采样]" if sampling_only else ""
                    system_name = getattr(result[0], 'system_name', os.path.basename(system_path.rstrip('/\\')))
                    self.logger.info(f"分析完成 ({i+1}/{len(system_paths)}): {system_name} [{mode}] {mode_suffix}")
                    
                    # 即时导出
                    self._export_sampled_frames(result, system_path, system_name)

                    # 流式保存
                    if self.streaming_enabled:
                        try:
                            # 立即同步当前系统的采样帧到PathManager.targets
                            self._sync_sampled_frames_to_targets(result, path_manager)
                            
                            ResultSaver.save_single_system(
                                output_dir=self.current_output_dir,
                                result=result,
                                sampling_only=sampling_only,
                                flush_targets_hook=(lambda: path_manager.save_analysis_targets({
                                    'sample_ratio': self.config.sample_ratio,
                                    'power_p': self.config.power_p,
                                    'pca_variance_ratio': self.config.pca_variance_ratio
                                })) if (self._targets_flush_interval == 1 or ((len(analysis_results) % self._targets_flush_interval) == 0)) else None
                            )
                        except Exception as se:
                            self.logger.warning(f"流式保存体系结果失败(忽略): {se}")
                    # 流式采样比较
                    if (not sampling_only) and getattr(self, 'sampling_stream_manager', None) and len(result) >= 7:
                        try:
                            metrics_obj = result[0]
                            frames = result[1]
                            pca_components_data = result[4]
                            id2energy = {f.frame_id: getattr(f, 'energy_standardized', None) for f in frames}
                            vectors_list = []
                            frame_ids = []
                            max_pc = 0
                            for item in pca_components_data:
                                for k in item.keys():
                                    if k.startswith('PC'):
                                        idx_pc = int(k[2:])
                                        if idx_pc > max_pc:
                                            max_pc = idx_pc
                            for item in pca_components_data:
                                fid = item.get('frame')
                                if fid is None:
                                    continue
                                energy_std = id2energy.get(fid)
                                if energy_std is None:
                                    continue
                                vec = [energy_std]
                                for i_pc in range(1, max_pc+1):
                                    vec.append(item.get(f'PC{i_pc}', 0.0))
                                vectors_list.append(vec)
                                frame_ids.append(fid)
                            import numpy as _np
                            if vectors_list:
                                vectors_arr = _np.array(vectors_list, dtype=float)
                                frame_ids_arr = _np.array(frame_ids, dtype=int)
                                sampled_set = set(getattr(metrics_obj, 'sampled_frames', []))
                                sampled_mask = _np.array([fid in sampled_set for fid in frame_ids_arr])
                                self.sampling_stream_manager.update_per_system(
                                    system_name=metrics_obj.system_name,
                                    system_path=system_path,
                                    vectors=vectors_arr,
                                    sampled_mask=sampled_mask,
                                    frame_ids=frame_ids_arr
                                )
                        except Exception as ce:
                            self.logger.warning(f"流式采样比较更新失败(忽略): {ce}")
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
        """导出采样帧到DeepMD格式"""
        try:
            if len(result) >= 2:
                metrics = result[0]
                frames = result[1]
                out_root = os.path.join(self.current_output_dir, 'deepmd_npy_per_system')
                ResultSaver.export_sampled_frames_per_system(
                    frames=frames,
                    sampled_frame_ids=getattr(metrics, 'sampled_frames', []) or [],
                    system_path=system_path,
                    output_root=out_root,
                    system_name=system_name,
                    logger=self.logger,
                    force=False
                )
        except Exception as de:
            self.logger.warning(f"体系 {system_path} deepmd 导出失败(忽略): {de}")
    
    def save_results(self, analysis_results: List[tuple], path_manager: PathManager) -> None:
        """保存分析结果"""
        if not analysis_results:
            self.logger.warning("没有分析结果需要保存")
            return

        # 流式模式下，集中保存退化为兜底（检查是否遗漏行）
        if self.streaming_enabled:
            self.logger.info("流式模式：跳过集中写入，执行兜底检查")
            try:
                # 读取 progress 已处理体系
                progress = ResultSaver.load_progress(self.current_output_dir)
                done = set(progress.get('processed_systems', []))
                missing = [r for r in analysis_results if getattr(r[0], 'system_name', None) not in done]
                if missing:
                    self.logger.info(f"检测到 {len(missing)} 个遗漏体系，补写...")
                    for r in missing:
                        try:
                            ResultSaver.save_single_system(self.current_output_dir, r, sampling_only=(self.config.mode==AnalysisMode.SAMPLING_ONLY))
                        except Exception:
                            pass
            except Exception as e:
                self.logger.warning(f"兜底检查失败(忽略): {e}")
            return
            
        # 仅采样模式的特殊处理
        if self.config.mode == AnalysisMode.SAMPLING_ONLY:
            self._save_sampling_only_results(analysis_results, path_manager)
        else:
            # 完整分析结果保存
            progress_info = ResultSaver.load_progress(self.current_output_dir)
            processed_systems = set(progress_info.get('processed_systems', []))
            is_incremental = len(processed_systems) > 0 and not self.config.force_recompute
            if is_incremental:
                self.logger.info(f"检测到已有进度（{len(processed_systems)} 个已处理系统），启用增量保存模式")
            else:
                self.logger.info("全新分析或强制重新计算，使用完整保存模式")
            ResultSaver.save_results(self.current_output_dir, analysis_results, incremental=is_incremental)
            # 强制同步采样帧到PathManager.targets，确保analysis_targets.json包含采样信息
            path_manager.update_sampled_frames_from_results(analysis_results)
            try:
                current_analysis_params = {
                    'sample_ratio': self.config.sample_ratio,
                    'power_p': self.config.power_p,
                    'pca_variance_ratio': self.config.pca_variance_ratio
                }
                path_manager.save_analysis_targets(current_analysis_params)
            except Exception as e:
                self.logger.warning(f"保存analysis_targets.json时出错: {e}")
    
    def _save_sampling_only_results(self, analysis_results: List[tuple], path_manager: PathManager) -> None:
        """仅采样模式：同步采样帧到PathManager.targets，并保存analysis_targets.json，不再生成sampling_summary.json"""
        # 使用统一的同步方法
        path_manager.update_sampled_frames_from_results(analysis_results)

        # 立即保存analysis_targets.json，确保采样帧写入
        try:
            current_analysis_params = {
                'sample_ratio': self.config.sample_ratio,
                'power_p': self.config.power_p,
                'pca_variance_ratio': self.config.pca_variance_ratio
            }
            path_manager.save_analysis_targets(current_analysis_params)
        except Exception as e:
            self.logger.warning(f"保存analysis_targets.json时出错: {e}")

        self.logger.info(f"仅采样模式结果已保存，DeepMD数据目录: {os.path.join(self.current_output_dir, 'deepmd_npy_per_system')}")
    
    def run_post_analysis(self) -> None:
        """运行后续分析（相关性分析和采样评估）"""
        if self.config.mode == AnalysisMode.SAMPLING_ONLY:
            self.logger.info("仅采样模式，跳过后续分析")
            return
            
        # 相关性分析
        self._run_correlation_analysis()
        
        # 采样效果评估
        if self.config.enable_sampling_eval:
            # 若存在流式管理器则直接 finalize，跳过离线全量遍历
            if getattr(self, 'sampling_stream_manager', None):
                try:
                    self.sampling_stream_manager.finalize()
                except Exception as fe:
                    self.logger.warning(f"流式采样汇总 finalize 失败: {fe}")
            else:
                self._run_sampling_evaluation()
    
    def _run_correlation_analysis(self) -> None:
        """运行相关性分析"""
        combined_csv_path = os.path.join(self.current_output_dir, "combined_analysis_results", "system_metrics_summary.csv")
        combined_output_dir = os.path.join(self.current_output_dir, "combined_analysis_results")

        if CORRELATION_ANALYSER_AVAILABLE and os.path.exists(combined_csv_path):
            analyser = None
            try:
                analyser = ExternalCorrelationAnalyser(logger=self.logger)
                analyser.analyse_correlations(combined_csv_path, combined_output_dir)
                self.logger.info("相关性分析完成")
            except Exception as e:
                self.logger.error(f"相关性分析失败: {str(e)}")
                import traceback
                self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            finally:
                if analyser and hasattr(analyser, 'cleanup'):
                    try:
                        analyser.cleanup()
                    except Exception as cleanup_error:
                        self.logger.warning(f"清理相关性分析器时出错: {str(cleanup_error)}")
        else:
            if not CORRELATION_ANALYSER_AVAILABLE:
                self.logger.warning("相关性分析模块不可用，跳过相关性分析")
            else:
                self.logger.warning(f"系统指标文件不存在，跳过相关性分析: {combined_csv_path}")

    def _run_sampling_evaluation(self) -> None:
        """运行采样效果评估"""
        if not SAMPLING_COMPARISON_AVAILABLE:
            self.logger.warning("采样效果评估模块不可用，跳过采样比较")
            return
        try:
            self.logger.info("开始采样效果评估 (采样 vs 随机 / 均匀)...")
            SamplingComparisonRunner(result_dir=self.current_output_dir)
            self.logger.info("采样效果评估完成: 生成 sampling_compare_enhanced.csv 与 sampling_methods_comparison.csv")
        except Exception as e:
            self.logger.error(f"采样效果评估失败: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())


class MainApp:
    """主应用程序类 - 重构版本"""
    
    def __init__(self):
        self.orchestrator = None
    
    def run(self) -> None:
        """运行主程序"""
        start_time = time.time()
        
        try:
            # 解析参数并创建配置
            config = self._parse_arguments_to_config()
            
            # 创建编排器
            self.orchestrator = AnalysisOrchestrator(config)
            
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
    
    def _parse_arguments_to_config(self) -> AnalysisConfig:
        """解析命令行参数并创建配置对象 (修正缩进)"""
        parser = argparse.ArgumentParser(description='ABACUS STRU轨迹主分析器 - 重构版本')

        # 核心参数
        parser.add_argument('-r', '--sample_ratio', type=float, default=0.1, help='采样比例')
        parser.add_argument('-p', '--power_p', type=float, default=-0.5, help='幂平均距离的p值')
        parser.add_argument('-v', '--pca_variance_ratio', type=float, default=0.90, help='PCA降维累计方差贡献率')

        # 运行配置
        parser.add_argument('-w', '--workers', type=int, default=-1, help='并行工作进程数')
        parser.add_argument('-o', '--output_dir', type=str, default='analysis_results', help='输出根目录')
        parser.add_argument('-s', '--search_path', nargs='*', default=None, help='递归搜索路径')
        parser.add_argument('-i', '--include_project', action='store_true', help='允许搜索项目自身目录')
        parser.add_argument('-f', '--force_recompute', action='store_true', help='强制重新计算')

        # 模式控制
        parser.add_argument('--sampling_only', action='store_true', help='仅采样模式：只执行采样算法，不计算统计指标')
        parser.add_argument('--dry_run_reuse', action='store_true', help='仅评估采样复用计划')
        parser.add_argument('--disable_sampling_eval', dest='enable_sampling_eval', action='store_false', help='禁用采样效果评估')
        parser.add_argument('--scheduler', choices=['legacy', 'process', 'thread'], default='process', help='调度器类型: legacy=旧逻辑, process=进程池(推荐), thread=线程池')

        args = parser.parse_args()

        return AnalysisConfig(
            sample_ratio=args.sample_ratio,
            power_p=args.power_p,
            pca_variance_ratio=args.pca_variance_ratio,
            workers=args.workers,
            output_dir=args.output_dir,
            search_paths=args.search_path or [],
            include_project=args.include_project,
            force_recompute=args.force_recompute,
            mode=AnalysisMode.SAMPLING_ONLY if args.sampling_only else AnalysisMode.FULL_ANALYSIS,
            dry_run_reuse=args.dry_run_reuse,
            enable_sampling_eval=args.enable_sampling_eval,
            scheduler=args.scheduler
        )

    def _log_startup_info(self, config: AnalysisConfig) -> None:
        """记录启动信息 (修正缩进)"""
        search_paths = self.orchestrator.resolve_search_paths()
        search_info = f"搜索路径: {search_paths if search_paths else '(当前目录的父目录)'}"

        mode_info = "仅采样模式" if config.mode == AnalysisMode.SAMPLING_ONLY else "完整分析模式"
        workers = self.orchestrator.determine_workers()

        self.orchestrator.logger.info(
            f"ABACUS主分析器启动 [{mode_info}] | 采样比例: {config.sample_ratio} | 工作进程: {workers}"
        )
        self.orchestrator.logger.info(search_info)
        self.orchestrator.logger.info(
            f"项目目录屏蔽: {'关闭' if config.include_project else '开启'}"
        )
        self.orchestrator.logger.info(
            f"强制重新计算: {'是' if config.force_recompute else '否'}"
        )
        self.orchestrator.logger.info(f"调度器: {config.scheduler}")

        if config.mode == AnalysisMode.SAMPLING_ONLY:
            self.orchestrator.logger.info("仅采样模式：将跳过统计指标计算和后续分析")
    
    def _execute_workflow(self, config: AnalysisConfig, start_time: float) -> None:
        """执行主要工作流程 (根据调度器分支)"""
        if config.scheduler == 'legacy':
            # 旧逻辑保留
            search_paths = self.orchestrator.resolve_search_paths()
            mol_systems = self.orchestrator.discover_systems(search_paths)
            if not mol_systems:
                self.orchestrator.logger.error("未找到符合格式的系统目录")
                return
            path_manager, actual_output_dir = self.orchestrator.setup_output_directory()
            has_existing_results = self.orchestrator.setup_analysis_targets(path_manager, search_paths)
            if has_existing_results and config.mode == AnalysisMode.FULL_ANALYSIS:
                self.orchestrator.logger.info("发现完整的分析结果，直接执行后续分析")
                self.orchestrator.run_post_analysis()
                return
            total_molecules = len(mol_systems)
            total_systems = sum(len(s) for s in mol_systems.values())
            final_targets = len(path_manager.targets)
            self.orchestrator.logger.info(f"发现 {total_molecules} 个分子，共 {total_systems} 个体系，去重后 {final_targets} 个目标")
            analysis_results = self.orchestrator.execute_analysis(path_manager)
            if analysis_results:
                self.orchestrator.save_results(analysis_results, path_manager)
                self.orchestrator.run_post_analysis()
            self._finalize_analysis(analysis_results, path_manager, start_time, actual_output_dir)
            return
        # 新轻量逻辑 (process / thread)
        search_paths = self.orchestrator.resolve_search_paths()
        t_discover_start = time.time()
        records = lightweight_discover_systems(search_paths, include_project=config.include_project)
        if not records:
            self.orchestrator.logger.error("轻量发现未找到体系")
            return
        self.orchestrator.logger.info(f"轻量发现完成: 体系 {len(records)} 耗时 {time.time()-t_discover_start:.1f}s")
        # 输出目录
        path_manager, actual_output_dir = self.orchestrator.setup_output_directory()
        # 构建 mol_systems 用于 path_manager 兼容保存采样信息
        mol_systems: Dict[str, List[str]] = {}
        for rec in records:
            mol_systems.setdefault(rec.mol_id, []).append(rec.system_path)
        path_manager.load_from_discovery(records, preserve_existing=False)
        path_manager.deduplicate_targets()
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
        self.orchestrator.logger.info(f"使用调度器: {config.scheduler} (workers={workers})")
        analysis_results: List[tuple] = []
        if config.scheduler == 'process':
            scheduler = ProcessScheduler(max_workers=workers, analyser_params=analyser_params)
            reused = 0
            for rec in records:
                pre = None
                meta = reuse_map_raw.get(rec.system_name)
                if meta and meta.get('source_hash') == rec.source_hash and meta.get('sampled_frames'):
                    pre = meta.get('sampled_frames')
                    reused += 1
                scheduler.add_task(ProcessAnalysisTask(
                    system_path=rec.system_path,
                    system_name=rec.system_name,
                    pre_sampled_frames=pre,
                    pre_stru_files=rec.selected_files,
                ))
            self.orchestrator.logger.info(f"进程模式任务: {len(records)} (复用 {reused})")
            raw_results = scheduler.run()
            # 过滤 None
            analysis_results = [r for r in raw_results if r]
        else:  # thread
            from src.core.analysis_orchestrator import AnalysisOrchestrator as CoreOrch
            core_orch = CoreOrch()
            # 初始化组件 (仅日志与 system_analyser 用)
            # 复用已建立日志
            core_orch.logger = self.orchestrator.logger
            core_orch.path_manager = path_manager
            core_orch.system_analyser = SystemAnalyser(
                include_hydrogen=False,
                sample_ratio=config.sample_ratio,
                power_p=config.power_p,
                pca_variance_ratio=config.pca_variance_ratio
            )
            analysis_results = core_orch.run_parallel_lightweight(search_paths, include_project=config.include_project, max_workers=workers)
        # 保存结果(逐体系流式写)
        for res in analysis_results:
            try:
                ResultSaver.save_single_system(actual_output_dir, res, sampling_only=(config.mode==AnalysisMode.SAMPLING_ONLY))
            except Exception as e:
                self.orchestrator.logger.warning(f"保存体系结果失败(忽略): {e}")
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
        # 后续分析
        if config.mode == AnalysisMode.FULL_ANALYSIS:
            self.orchestrator.run_post_analysis()
        self._finalize_analysis(analysis_results, path_manager, start_time, actual_output_dir)
    
    def _finalize_analysis(self, analysis_results: List[tuple], path_manager: PathManager, 
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

        # 流式模式：最终排序 system_metrics_summary.csv（不影响分析正确性）
        if self.orchestrator.streaming_enabled and self.orchestrator.config.mode == AnalysisMode.FULL_ANALYSIS:
            try:
                ResultSaver.reorder_system_summary(output_dir)
            except Exception as re:
                self.orchestrator.logger.warning(f"system_metrics_summary 排序失败(忽略): {re}")
        
        # 输出最终统计
        elapsed = time.time() - start_time
        mode_info = "仅采样模式" if self.orchestrator.config.mode == AnalysisMode.SAMPLING_ONLY else "完整分析模式"
        
        self.orchestrator.logger.info("=" * 60)
        self.orchestrator.logger.info(f"分析完成! [{mode_info}] 处理体系: {len(analysis_results)}/{len(path_manager.targets)}")

        if analysis_results and self.orchestrator.config.mode == AnalysisMode.FULL_ANALYSIS:
            # 仅在完整模式下输出采样统计
            swap_counts = [result[2] for result in analysis_results if len(result) > 2]
            if swap_counts:
                import numpy as np
                self.orchestrator.logger.info("采样优化统计:")
                self.orchestrator.logger.info("  平均交换次数: %.2f", np.mean(swap_counts))
                self.orchestrator.logger.info(f"  总交换次数: {int(sum(swap_counts))}")

        self.orchestrator.logger.info(f"总耗时: {elapsed:.1f}s ({elapsed/60:.1f} 分钟)")
        self.orchestrator.logger.info(f"结果目录: {output_dir}")
        
        if self.orchestrator.config.mode == AnalysisMode.SAMPLING_ONLY:
            self.orchestrator.logger.info(f"采样汇总: {os.path.join(output_dir, 'sampling_summary.json')}")
            self.orchestrator.logger.info(f"DeepMD数据: {os.path.join(output_dir, 'deepmd_npy_per_system')}")
        
        self.orchestrator.logger.info("=" * 60)


if __name__ == "__main__":
    MainApp().run()
