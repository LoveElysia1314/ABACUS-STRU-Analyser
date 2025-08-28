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
        
        # 配置日志 - 固定输出到analysis_results目录
        analysis_results_dir = os.path.join(os.getcwd(), "analysis_results")
        os.makedirs(analysis_results_dir, exist_ok=True)
        self.logger = create_standard_logger(__name__, analysis_results_dir, "main_analysis.log")
        
        # 设置工作进程数
        workers = self._determine_workers(args.workers)
        
        # 记录启动信息
        search_info = f"搜索路径: {args.search_path or '(当前目录的父目录)'}"
        self.logger.info(f"ABACUS主分析器启动 | 采样比例: {args.sample_ratio} | 工作进程: {workers}")
        self.logger.info(search_info)
        self.logger.info(f"日志文件: analysis_results/main_analysis.log")
        
        # 步骤1: 查找和管理系统目录
        mol_systems = DirectoryDiscovery.find_abacus_systems(args.search_path)
        if not mol_systems:
            self.logger.error("未找到符合格式的系统目录")
            return
        
        # 步骤2: 初始化路径管理器
        path_manager = PathManager(args.output_dir)
        path_manager.load_from_discovery(mol_systems)
        path_manager.save_targets()
        path_manager.save_summary()
        path_manager.export_target_paths()
        
        # 验证目标
        valid_count, invalid_count = path_manager.validate_targets()
        if valid_count == 0:
            self.logger.error("没有有效的分析目标")
            return
        
        total_molecules = len(mol_systems)
        total_systems = sum(len(s) for s in mol_systems.values())
        self.logger.info(f"发现 {total_molecules} 个分子，共 {total_systems} 个体系 (有效: {valid_count})")
        
        # 步骤3: 创建分析器并执行分析
        analyzer = SystemAnalyzer(
            include_hydrogen=args.include_h,
            sample_ratio=args.sample_ratio,
            power_p=args.power_p
        )
        
        analysis_targets = path_manager.get_targets_by_status("pending")
        system_paths = [target.system_path for target in analysis_targets]
        
        # 执行分析
        if workers > 1:
            analysis_results = self._parallel_analysis(analyzer, system_paths, path_manager, workers)
        else:
            analysis_results = self._sequential_analysis(analyzer, system_paths, path_manager)
        
        # 步骤4: 保存结果
        if analysis_results:
            ResultSaver.save_all_results(args.output_dir, analysis_results)
            
            # 执行相关性分析
            self._run_correlation_analysis(args.output_dir)
        
        # 步骤5: 保存最终状态并输出统计
        path_manager.save_targets()
        path_manager.save_summary()
        
        self._output_final_statistics(analysis_results, start_time, args.output_dir, path_manager)
    
    def _parse_arguments(self) -> argparse.Namespace:
        """解析命令行参数"""
        parser = argparse.ArgumentParser(description='ABACUS STRU轨迹主分析器')
        parser.add_argument('--include_h', action='store_true', help='包含氢原子')
        parser.add_argument('--sample_ratio', type=float, default=0.05, help='采样比例')
        parser.add_argument('--power_p', type=float, default=0.5, help='幂平均距离的p值')
        parser.add_argument('--workers', type=int, default=-1, help='并行工作进程数')
        parser.add_argument('--output_dir', type=str, default="analysis_results", help='输出根目录')
        parser.add_argument('--search_path', type=str, default=None, 
                           help='递归搜索路径 (默认为当前目录的父目录)')
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