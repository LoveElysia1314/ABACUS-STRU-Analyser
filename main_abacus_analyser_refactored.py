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
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# 导入自定义模块
from src.utils import DirectoryDiscovery
from src.utils.logmanager import LoggerManager
from src.io.path_manager import PathManager
from src.core.system_analyser import SystemAnalyser, BatchAnalyser
from src.io.result_saver import ResultSaver
from src.utils.common import ErrorHandler
from src.utils.common import FileUtils
from src.utils.common import ErrorHandler as _Err

# 采样效果评估（采样对比）
try:
    from src.analysis.sampling_comparison import analyse_sampling_compare as SamplingComparisonRunner
    SAMPLING_COMPARISON_AVAILABLE = True
except Exception:
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
    
    def __post_init__(self):
        if self.search_paths is None:
            self.search_paths = []


# 多进程全局变量
_GLOBAL_ANALYSER = None
_GLOBAL_REUSE_MAP: Dict[str, List[int]] = {}


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
    global _GLOBAL_ANALYSER, _GLOBAL_REUSE_MAP

    _GLOBAL_ANALYSER = SystemAnalyser(include_hydrogen=False,
                                      sample_ratio=sample_ratio,
                                      power_p=power_p,
                                      pca_variance_ratio=pca_variance_ratio)
    _GLOBAL_REUSE_MAP = reuse_map or {}

    # 设置采样模式
    _GLOBAL_ANALYSER._sampling_only = sampling_only

    # Setup multiprocess logging for worker
    if log_queue is not None:
        worker_logger = LoggerManager.setup_worker_logger(
            name="WorkerProcess",
            queue=log_queue,
            level=logging.INFO,
            add_console=False
        )


def _child_worker(system_path: str):
    """工作进程执行函数（支持采样帧复用和仅采样模式）"""
    global _GLOBAL_ANALYSER, _GLOBAL_REUSE_MAP
    
    if _GLOBAL_ANALYSER is None:
        sampling_only = getattr(_GLOBAL_ANALYSER, '_sampling_only', False) if _GLOBAL_ANALYSER else False
        return _worker_analyse_system(system_path, 0.05, 0.5, 0.90, 
                                    pre_sampled_frames=None, sampling_only=sampling_only)
    
    sys_name = os.path.basename(system_path.rstrip('/\\'))
    pre_frames = _GLOBAL_REUSE_MAP.get(sys_name)
    
    # 检查是否为仅采样模式
    sampling_only = getattr(_GLOBAL_ANALYSER, '_sampling_only', False)
    
    if sampling_only:
        return _GLOBAL_ANALYSER.analyse_system_sampling_only(system_path, pre_sampled_frames=pre_frames)
    else:
        return _GLOBAL_ANALYSER.analyse_system(system_path, pre_sampled_frames=pre_frames)


class AnalysisOrchestrator:
    """分析流程编排器 - 核心逻辑提取"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = None
        self.log_queue = None
        self.log_listener = None
        self.current_output_dir = None
        
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
            except Exception as e:
                print(f"停止日志监听器时出错: {str(e)}")
    
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
                    self.logger.info(f"通配符 '{path_pattern}' 展开为 {len(expanded_dirs)} 个目录")
                else:
                    self.logger.warning(f"通配符 '{path_pattern}' 未匹配到任何目录")
            else:
                if os.path.isdir(path_pattern):
                    resolved_paths.append(path_pattern)
                else:
                    self.logger.warning(f"路径不存在或不是目录: {path_pattern}")
        
        unique_paths = list(set(os.path.abspath(p) for p in resolved_paths))
        return unique_paths
    
    def discover_systems(self, search_paths: List[str]) -> dict:
        """从多个搜索路径发现ABACUS系统"""
        all_mol_systems = {}
        
        for search_path in search_paths:
            self.logger.info(f"搜索路径: {search_path}")
            try:
                if self.config.include_project:
                    mol_systems = DirectoryDiscovery.find_abacus_systems(search_path, include_project=True)
                else:
                    mol_systems = DirectoryDiscovery.find_abacus_systems(search_path)
                
                for mol_key, system_paths in mol_systems.items():
                    if mol_key in all_mol_systems:
                        all_mol_systems[mol_key].extend(system_paths)
                    else:
                        all_mol_systems[mol_key] = system_paths
                
                self.logger.info(f"在 {search_path} 中发现 {len(mol_systems)} 个分子类型")
            except Exception as e:
                self.logger.error(f"搜索路径 {search_path} 时出错: {str(e)}")
        
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
        
        self.logger.info(f"使用参数专用目录: {actual_output_dir}")
        return path_manager, actual_output_dir
    
    def setup_analysis_targets(self, path_manager: PathManager, mol_systems: dict) -> bool:
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
        path_manager.load_from_discovery(mol_systems, preserve_existing=loaded_existing)
        path_manager.deduplicate_targets()
        
        # 参数兼容性检查
        params_compatible = False
        if loaded_existing:
            params_compatible = path_manager.check_params_compatibility(current_analysis_params)
        
        # 增量计算检查
        if not self.config.force_recompute and params_compatible:
            self.logger.info("检查已有分析结果以启用增量计算...")
            path_manager.check_existing_results()
        else:
            if self.config.force_recompute:
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
            except Exception as e:
                self.logger.warning(f"Failed to determine optimal worker count from environment: {e}")
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
        
        # 采样复用判定
        reuse_map = path_manager.determine_sampling_reuse()
        self.logger.info(f"采样复用判定完成：可复用 {len(reuse_map)} / {len(path_manager.targets)} 个系统的采样帧")

        # Dry-run检查
        if self.config.dry_run_reuse:
            self._output_reuse_plan(reuse_map, path_manager)
            return []

        # 获取分析目标
        analysis_targets = path_manager.get_all_targets()
        
        # 过滤已处理的系统（仅在非强制模式下）
        if not self.config.force_recompute:
            progress_info = ResultSaver.load_progress(self.current_output_dir)
            processed_systems = set(progress_info.get('processed_systems', []))
            
            if processed_systems:
                unprocessed_targets = [t for t in analysis_targets if t.system_name not in processed_systems]
                self.logger.info(f"检测到 {len(processed_systems)} 个已处理系统，跳过这些系统")
                analysis_targets = unprocessed_targets

        system_paths = [target.system_path for target in analysis_targets]
        
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
                        except Exception:
                            system_path = system_paths[i] if i < len(system_paths) else None
                            system_name = os.path.basename(system_path.rstrip('/\\')) if system_path else "未知"

                        mode_suffix = "[仅采样]" if sampling_only else ""
                        self.logger.info(f"分析完成 (已完成 {len(analysis_results)}/{len(system_paths)}): {system_name} {mode_suffix}")
                        
                        # 即时导出（如果有采样结果）
                        if system_path and len(result) >= 2:
                            self._export_sampled_frames(result, system_path, system_name)
                    else:
                        self.logger.warning("并行分析返回空结果，标记为失败")
            except Exception as e:
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
            path_manager.load_sampled_frames_from_csv()
    
    def _save_sampling_only_results(self, analysis_results: List[tuple], path_manager: PathManager) -> None:
        """仅采样模式：同步采样帧到PathManager.targets，并保存analysis_targets.json，不再生成sampling_summary.json"""
        # 建立 system_name -> sampled_frames 的映射
        sampled_frames_map = {}
        for result in analysis_results:
            if len(result) >= 2:
                metrics = result[0]
                system_name = getattr(metrics, 'system_name', 'unknown')
                sampled_frames = getattr(metrics, 'sampled_frames', [])
                sampled_frames_map[system_name] = sampled_frames

        # 同步采样帧到PathManager.targets
        for target in path_manager.targets:
            if target.system_name in sampled_frames_map:
                target.sampled_frames = sampled_frames_map[target.system_name]

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
                print(f"主程序执行出错: {str(e)}")
                import traceback
                print(traceback.format_exc())
        finally:
            if self.orchestrator:
                self.orchestrator.cleanup_logging()
    
    def _parse_arguments_to_config(self) -> AnalysisConfig:
        """解析命令行参数并创建配置对象"""
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
        
        args = parser.parse_args()
        
        # 创建配置对象
        config = AnalysisConfig(
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
            enable_sampling_eval=args.enable_sampling_eval
        )
        
        return config
    
    def _log_startup_info(self, config: AnalysisConfig) -> None:
        """记录启动信息"""
        search_paths = self.orchestrator.resolve_search_paths()
        search_info = f"搜索路径: {search_paths if search_paths else '(当前目录的父目录)'}"
        
        mode_info = "仅采样模式" if config.mode == AnalysisMode.SAMPLING_ONLY else "完整分析模式"
        workers = self.orchestrator.determine_workers()
        
        self.orchestrator.logger.info(f"ABACUS主分析器启动 [{mode_info}] | 采样比例: {config.sample_ratio} | 工作进程: {workers}")
        self.orchestrator.logger.info(search_info)
        self.orchestrator.logger.info(f"项目目录屏蔽: {'关闭' if config.include_project else '开启'}")
        self.orchestrator.logger.info(f"强制重新计算: {'是' if config.force_recompute else '否'}")
        
        if config.mode == AnalysisMode.SAMPLING_ONLY:
            self.orchestrator.logger.info("仅采样模式：将跳过统计指标计算和后续分析")
    
    def _execute_workflow(self, config: AnalysisConfig, start_time: float) -> None:
        """执行主要工作流程"""
        # 1. 系统发现
        search_paths = self.orchestrator.resolve_search_paths()
        mol_systems = self.orchestrator.discover_systems(search_paths)
        if not mol_systems:
            self.orchestrator.logger.error("未找到符合格式的系统目录")
            return
        
        # 2. 设置输出目录
        path_manager, actual_output_dir = self.orchestrator.setup_output_directory()
        
        # 3. 设置分析目标
        has_existing_results = self.orchestrator.setup_analysis_targets(path_manager, mol_systems)
        
        # 4. 快速路径检查（仅对完整分析有效）
        if has_existing_results and config.mode == AnalysisMode.FULL_ANALYSIS:
            self.orchestrator.logger.info("发现完整的分析结果，直接执行后续分析")
            self.orchestrator.run_post_analysis()
            return
        
        # 5. 输出系统统计
        total_molecules = len(mol_systems)
        total_systems = sum(len(s) for s in mol_systems.values())
        final_targets = len(path_manager.targets)
        self.orchestrator.logger.info(f"发现 {total_molecules} 个分子，共 {total_systems} 个体系，去重后 {final_targets} 个目标")
        
        # 6. 执行分析
        analysis_results = self.orchestrator.execute_analysis(path_manager)
        
        # 7. 保存结果
        if analysis_results:
            self.orchestrator.save_results(analysis_results, path_manager)
            
            # 8. 后续分析（仅完整模式）
            self.orchestrator.run_post_analysis()
        
        # 9. 保存最终状态并输出统计
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
                self.orchestrator.logger.info(f"采样优化统计:")
                self.orchestrator.logger.info(f"  平均交换次数: {np.mean(swap_counts):.2f}")
                self.orchestrator.logger.info(f"  总交换次数: {int(sum(swap_counts))}")

        self.orchestrator.logger.info(f"总耗时: {elapsed:.1f}s ({elapsed/60:.1f} 分钟)")
        self.orchestrator.logger.info(f"结果目录: {output_dir}")
        
        if self.orchestrator.config.mode == AnalysisMode.SAMPLING_ONLY:
            self.orchestrator.logger.info(f"采样汇总: {os.path.join(output_dir, 'sampling_summary.json')}")
            self.orchestrator.logger.info(f"DeepMD数据: {os.path.join(output_dir, 'deepmd_npy_per_system')}")
        
        self.orchestrator.logger.info("=" * 60)


if __name__ == "__main__":
    MainApp().run()
