#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ABACUS主分析器程序
功能：批量分析ABACUS分子动力学轨迹，支持智能采样和相关性分析
"""

import os
import sys
import time
import argparse
import logging
import multiprocessing as mp
import glob
from typing import List

# 导入自定义模块
from src.utils import DirectoryDiscovery, create_standard_logger
from src.io.path_manager import PathManager
from src.core.system_analyzer import SystemAnalyzer, BatchAnalyzer
from src.io.result_saver import ResultSaver

try:
    from src.analysis.correlation_analyzer import CorrelationAnalyzer as ExternalCorrelationAnalyzer
    CORRELATION_ANALYZER_AVAILABLE = True
except ImportError:
    CORRELATION_ANALYZER_AVAILABLE = False


def _worker_analyze_system(system_path: str, include_h: bool, sample_ratio: float, power_p: float):
    """Top-level worker function for multiprocessing.

    Creates a local SystemAnalyzer in the child process to avoid pickling
    the parent-process analyzer instance.
    """
    analyzer = SystemAnalyzer(
        include_hydrogen=include_h,
        sample_ratio=sample_ratio,
        power_p=power_p
    )
    return analyzer.analyze_system(system_path)


# Globals used when initializing analyzers inside worker processes
_GLOBAL_ANALYZER = None


def _child_init(include_h: bool, sample_ratio: float, power_p: float):
    """Initializer for worker processes: create one SystemAnalyzer per process."""
    global _GLOBAL_ANALYZER
    _GLOBAL_ANALYZER = SystemAnalyzer(
        include_hydrogen=include_h,
        sample_ratio=sample_ratio,
        power_p=power_p
    )


def _child_worker(system_path: str):
    """Worker that uses the per-process analyzer set by _child_init.

    Returns the same tuple result as SystemAnalyzer.analyze_system.
    """
    global _GLOBAL_ANALYZER
    if _GLOBAL_ANALYZER is None:
        # Fallback: construct a temporary analyzer (shouldn't happen when Pool initializer used)
        return _worker_analyze_system(system_path, False, 0.05, 0.5)
    return _GLOBAL_ANALYZER.analyze_system(system_path)


class MainApp:
    """主应用程序类"""
    
    def __init__(self):
        self.logger = None
    
    def run(self) -> None:
        """运行主程序"""
        start_time = time.time()
        
        # 解析命令行参数
        args = self._parse_arguments()
        
        # 配置日志 - 使用时间戳避免覆盖
        analysis_results_dir = os.path.join(os.getcwd(), "analysis_results")
        os.makedirs(analysis_results_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_filename = f"main_analysis_{timestamp}.log"
        self.logger = create_standard_logger(__name__, analysis_results_dir, log_filename)
        
        # 设置工作进程数
        workers = self._determine_workers(args.workers)
        
        # 记录启动信息
        search_paths = self._resolve_search_paths(args.search_path)
        search_info = f"搜索路径: {search_paths if search_paths else '(当前目录的父目录)'}"
        self.logger.info(f"ABACUS主分析器启动 | 采样比例: {args.sample_ratio} | 工作进程: {workers}")
        self.logger.info(search_info)
        self.logger.info(f"项目目录屏蔽: {'关闭' if args.include_project else '开启'}")
        self.logger.info(f"跳过单个结果: {'是' if args.skip_single_results else '否'}")
        self.logger.info(f"强制重新计算: {'是' if args.force_recompute else '否'}")
        self.logger.info(f"日志文件: analysis_results/{log_filename}")
        
        # 步骤1: 查找和管理系统目录
        mol_systems = self._discover_systems_from_paths(search_paths, args.include_project)
        if not mol_systems:
            self.logger.error("未找到符合格式的系统目录")
            return
        
        # 步骤2: 配置参数专用输出目录
        self.logger.info("配置分析参数专用目录...")
        current_analysis_params = {
            'include_h': args.include_h,
            'sample_ratio': args.sample_ratio,
            'power_p': args.power_p,
            'skip_single_results': args.skip_single_results
        }
        
        path_manager = PathManager(args.output_dir)
        actual_output_dir = path_manager.set_output_dir_for_params(current_analysis_params)
        self.logger.info(f"使用参数专用目录: {actual_output_dir}")
        
        # 检查是否已有完整结果（快速路径）
        if not args.force_recompute and path_manager.check_existing_complete_results():
            self.logger.info("发现完整的分析结果，直接执行相关性分析")
            self._run_correlation_analysis(actual_output_dir)
            return

        # 步骤3: 初始化目标管理
        self.logger.info("初始化分析目标管理...")
        
        # 尝试先加载已有的分析目标状态（从当前参数目录）
        loaded_existing = path_manager.load_analysis_targets()
        if loaded_existing:
            self.logger.info("成功加载已有的分析目标状态")
        else:
            self.logger.info("未找到已有的分析目标文件，将创建新的")
        
        self.logger.info("加载发现结果到路径管理器...")
        path_manager.load_from_discovery(mol_systems, preserve_existing=loaded_existing)
        
        # 去重处理（必须执行，无论是否跳过验证）
        self.logger.info("执行重复体系去重（基于系统名称，保留修改时间最晚的）...")
        path_manager.deduplicate_targets()
        
        # 参数一致性检查：检查分析参数是否与加载的参数兼容
        current_analysis_params = {
            'include_h': args.include_h,
            'sample_ratio': args.sample_ratio,
            'power_p': args.power_p,
            'skip_single_results': args.skip_single_results
        }
        
        params_compatible = False
        if loaded_existing:
            params_compatible = path_manager.check_params_compatibility(current_analysis_params)
        
        # 增量计算检查：检查已有的分析结果
        if not args.force_recompute and params_compatible:
            self.logger.info("检查已有分析结果以启用增量计算...")
            path_manager.check_existing_results(args.skip_single_results)
        else:
            if args.force_recompute:
                self.logger.info("强制重新计算模式：重置所有系统状态为待处理")
            elif not params_compatible and loaded_existing:
                self.logger.info("参数不兼容：重置所有系统状态为待处理")
            # 重置所有系统状态为pending
            for target in path_manager.targets:
                target.status = "pending"
        
        total_molecules = len(mol_systems)
        total_systems = sum(len(s) for s in mol_systems.values())
        final_targets = len(path_manager.targets)
        self.logger.info(f"发现 {total_molecules} 个分子，共 {total_systems} 个体系，去重后 {final_targets} 个目标")
        
        # 步骤3: 创建分析器并执行分析
        self.logger.info("创建系统分析器...")
        analyzer = SystemAnalyzer(
            include_hydrogen=args.include_h,
            sample_ratio=args.sample_ratio,
            power_p=args.power_p
        )
        
        analysis_targets = path_manager.get_targets_by_status("pending") + path_manager.get_targets_by_status("failed")
        system_paths = [target.system_path for target in analysis_targets]
        completed_count = len(path_manager.get_targets_by_status("completed"))
        
        self.logger.info(f"准备分析 {len(system_paths)} 个系统（已完成: {completed_count}，待处理: {len(system_paths)}）...")
        
        # 执行分析
        if workers > 1:
            self.logger.info(f"使用并行分析模式（{workers} 个工作进程）...")
            analysis_results = self._parallel_analysis(analyzer, system_paths, path_manager, workers)
        else:
            self.logger.info("使用顺序分析模式...")
            analysis_results = self._sequential_analysis(analyzer, system_paths, path_manager)
        
        # 步骤4: 保存结果
        if analysis_results:
            self.logger.info("保存分析结果...")
            
            # 判断是否有已完成的系统（增量计算模式）
            completed_systems = len(path_manager.get_targets_by_status("completed"))
            is_incremental = completed_systems > 0 and not args.force_recompute
            
            if is_incremental:
                self.logger.info(f"增量计算模式：合并 {len(analysis_results)} 个新结果与 {completed_systems} 个已有结果")
                
            if args.skip_single_results:
                if is_incremental:
                    self.logger.info("跳过单个系统详细结果，使用增量保存汇总...")
                    ResultSaver.save_summary_only_incremental(actual_output_dir, analysis_results)
                else:
                    self.logger.info("跳过单个系统详细结果，仅保存汇总...")
                    ResultSaver.save_summary_only(actual_output_dir, analysis_results)
            else:
                if is_incremental:
                    self.logger.info("使用增量保存模式...")
                    ResultSaver.save_all_results_incremental(actual_output_dir, analysis_results)
                else:
                    ResultSaver.save_all_results(actual_output_dir, analysis_results)
            
            # 执行相关性分析
            self._run_correlation_analysis(actual_output_dir)
        elif len(path_manager.get_targets_by_status("completed")) > 0:
            # 没有新的分析结果，但有已完成的系统
            self.logger.info("所有系统均已完成分析，跳过结果保存")
            # 仍然执行相关性分析
            self._run_correlation_analysis(actual_output_dir)
        
        # 步骤5: 保存最终状态并输出统计
        self.logger.info("保存分析目标状态...")
        try:
            path_manager.save_analysis_targets(current_analysis_params)
        except Exception as e:
            self.logger.error(f"保存分析目标失败: {str(e)}")
        
        self._output_final_statistics(analysis_results, start_time, actual_output_dir, path_manager)
    
    def _parse_arguments(self) -> argparse.Namespace:
        """解析命令行参数"""
        parser = argparse.ArgumentParser(description='ABACUS STRU轨迹主分析器')
        parser.add_argument('--include_h', action='store_true', help='包含氢原子')
        parser.add_argument('--sample_ratio', type=float, default=0.05, help='采样比例')
        parser.add_argument('--power_p', type=float, default=-0.5, help='幂平均距离的p值')
        parser.add_argument('--workers', type=int, default=-1, help='并行工作进程数')
        parser.add_argument('--output_dir', type=str, default="analysis_results", help='输出根目录')
        parser.add_argument('--search_path', nargs='*', default=None, 
                           help='递归搜索路径，支持多个路径和通配符 (默认为当前目录的父目录)')
        parser.add_argument('--include_project', action='store_true', 
                           help='允许搜索项目自身目录（默认屏蔽）')
        parser.add_argument('--skip_single_results', action='store_true',
                           help='跳过单个系统的详细结果输出(frame_metrics_*.csv)，仅保留汇总结果')
        parser.add_argument('--force_recompute', action='store_true',
                           help='强制重新计算所有系统，忽略已有的分析结果')
        return parser.parse_args()
    
    def _determine_workers(self, workers_arg: int) -> int:
        """确定工作进程数"""
        if workers_arg == -1:
            try:
                workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 
                           os.environ.get('SLURM_JOB_CPUS_PER_NODE', mp.cpu_count())))
            except:
                workers = max(1, mp.cpu_count())
        else:
            workers = max(1, workers_arg)
        return workers
    
    def _resolve_search_paths(self, search_path_args: List[str]) -> List[str]:
        """解析搜索路径，支持通配符展开"""
        if not search_path_args:
            # 默认使用当前目录的父目录
            return [os.path.abspath(os.path.join(os.getcwd(), '..'))]
        
        resolved_paths = []
        for path_pattern in search_path_args:
            # 展开通配符
            expanded = glob.glob(path_pattern, recursive=True)
            if expanded:
                # 只保留目录
                expanded_dirs = [p for p in expanded if os.path.isdir(p)]
                resolved_paths.extend(expanded_dirs)
                if expanded_dirs:
                    self.logger.info(f"通配符 '{path_pattern}' 展开为 {len(expanded_dirs)} 个目录")
                else:
                    self.logger.warning(f"通配符 '{path_pattern}' 未匹配到任何目录")
            else:
                # 没有匹配，检查是否为普通路径
                if os.path.isdir(path_pattern):
                    resolved_paths.append(path_pattern)
                else:
                    self.logger.warning(f"路径不存在或不是目录: {path_pattern}")
        
        # 去重并规范化路径
        unique_paths = list(set(os.path.abspath(p) for p in resolved_paths))
        return unique_paths
    
    def _discover_systems_from_paths(self, search_paths: List[str], include_project: bool = False) -> dict:
        """从多个搜索路径发现ABACUS系统"""
        all_mol_systems = {}
        
        for search_path in search_paths:
            self.logger.info(f"搜索路径: {search_path}")
            try:
                # 根据include_project标志调用相应的发现方法
                if include_project:
                    # 需要修改DirectoryDiscovery以支持include_project参数
                    mol_systems = DirectoryDiscovery.find_abacus_systems(search_path, include_project=True)
                else:
                    mol_systems = DirectoryDiscovery.find_abacus_systems(search_path)
                
                # 合并结果
                for mol_key, system_paths in mol_systems.items():
                    if mol_key in all_mol_systems:
                        all_mol_systems[mol_key].extend(system_paths)
                    else:
                        all_mol_systems[mol_key] = system_paths
                
                self.logger.info(f"在 {search_path} 中发现 {len(mol_systems)} 个分子类型")
            except Exception as e:
                self.logger.error(f"搜索路径 {search_path} 时出错: {str(e)}")
        
        return all_mol_systems
    
    def _parallel_analysis(self, analyzer: SystemAnalyzer, system_paths: List[str], 
                          path_manager: PathManager, workers: int) -> List[tuple]:
        """并行分析系统"""
        analysis_results = []
        # Use a Pool initializer to create one SystemAnalyzer per worker process and
        # use imap_unordered with a calculated chunksize for better throughput.
        initializer_args = (analyzer.include_hydrogen, analyzer.sample_ratio, analyzer.power_p)
        chunksize = max(1, len(system_paths) // (workers * 4)) if system_paths else 1

        with mp.Pool(processes=workers, initializer=_child_init, initargs=initializer_args) as pool:
            try:
                for i, result in enumerate(pool.imap_unordered(_child_worker, system_paths, chunksize=chunksize)):
                    # Note: imap_unordered returns results in completion order; map system path by index can't be used.
                    # We'll attempt to find the matching system path from result when possible.
                    if result:
                        # result expected to be tuple where first element has system_name and path
                        try:
                            system_name = result[0].system_name
                            system_path = result[0].system_path
                        except Exception:
                            # Fallback: use index-based mapping if result doesn't provide path
                            system_path = system_paths[i] if i < len(system_paths) else None

                        if system_path:
                            path_manager.update_target_status(system_path, "completed")
                        analysis_results.append(result)
                        self.logger.info(f"分析完成 (已完成 {len(analysis_results)}/{len(system_paths)}): {getattr(result[0], 'system_name', system_path)}")
                    else:
                        # Unknown which path failed; best-effort mark if we can
                        self.logger.warning("并行分析返回空结果，标记为失败")
            except Exception as e:
                self.logger.error(f"并行处理出错: {str(e)}")
        return analysis_results
    
    def _sequential_analysis(self, analyzer: SystemAnalyzer, system_paths: List[str], 
                           path_manager: PathManager) -> List[tuple]:
        """顺序分析系统"""
        analysis_results = []
        
        for i, system_path in enumerate(system_paths):
            try:
                path_manager.update_target_status(system_path, "processing")
                
                result = analyzer.analyze_system(system_path)
                if result:
                    analysis_results.append(result)
                    path_manager.update_target_status(system_path, "completed")
                    self.logger.info(f"分析完成 ({i+1}/{len(system_paths)}): {result[0].system_name}")
                else:
                    path_manager.update_target_status(system_path, "failed")
                    self.logger.warning(f"分析失败 ({i+1}/{len(system_paths)}): {system_path}")
                    
            except Exception as e:
                path_manager.update_target_status(system_path, "failed")
                self.logger.error(f"处理体系 {system_path} 时出错: {str(e)}")
        
        return analysis_results
    
    def _run_correlation_analysis(self, output_dir: str) -> None:
        """运行相关性分析"""
        combined_csv_path = os.path.join(output_dir, "combined_analysis_results", "system_metrics_summary.csv")
        combined_output_dir = os.path.join(output_dir, "combined_analysis_results")
        
        if CORRELATION_ANALYZER_AVAILABLE and os.path.exists(combined_csv_path):
            try:
                analyzer = ExternalCorrelationAnalyzer(logger=self.logger)
                analyzer.analyze_correlations(combined_csv_path, combined_output_dir)
                self.logger.info("相关性分析完成")
            except Exception as e:
                self.logger.error(f"相关性分析失败: {str(e)}")
        else:
            if not CORRELATION_ANALYZER_AVAILABLE:
                self.logger.warning("相关性分析模块不可用，跳过相关性分析")
            else:
                self.logger.warning(f"系统指标文件不存在，跳过相关性分析: {combined_csv_path}")
    
    def _output_final_statistics(self, analysis_results: List[tuple], start_time: float, 
                               output_dir: str, path_manager: PathManager) -> None:
        """输出最终统计信息"""
        elapsed = time.time() - start_time
        progress_summary = path_manager.get_progress_summary()
        
        # 计算采样统计
        swap_counts = [result[3] for result in analysis_results if len(result) > 3]
        improve_ratios = [result[4] for result in analysis_results if len(result) > 4]
        
        self.logger.info("=" * 60)
        self.logger.info(f"分析完成! 处理体系: {len(analysis_results)}/{progress_summary['total']}")
        self.logger.info(f"状态统计: 完成 {progress_summary['completed']}, "
                        f"失败 {progress_summary['failed']}, "
                        f"待处理 {progress_summary['pending']}")
        
        if swap_counts:
            import numpy as np
            self.logger.info(f"采样优化统计:")
            self.logger.info(f"  平均交换次数: {np.mean(swap_counts):.2f}")
            self.logger.info(f"  平均改善率: {np.mean(improve_ratios):.2%}")
            self.logger.info(f"  总交换次数: {sum(swap_counts)}")
        
        self.logger.info(f"总耗时: {elapsed:.1f}s ({elapsed/60:.1f} 分钟)")
        self.logger.info(f"结果目录: {output_dir}")
        self.logger.info(f"路径信息: {path_manager.targets_file}")
        self.logger.info(f"日志文件: analysis_results/main_analysis.log")
        self.logger.info("=" * 60)


if __name__ == "__main__":
    MainApp().run()