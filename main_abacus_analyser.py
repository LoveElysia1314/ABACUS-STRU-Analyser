#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ABACUS主分析器程序 - 重构版本
功能：批量分析ABACUS分子动力学轨迹，专注于采样、体系分析和deepmd转化
"""

import os
import sys
import time
import argparse
import logging
import multiprocessing as mp
import glob
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入自定义模块
from src.utils.logmanager import LoggerManager
from src.io.path_manager import PathManager, load_sampling_reuse_map, lightweight_discover_systems
from src.core.system_analyser import SystemAnalyser
from src.io.result_saver import ResultSaver
from src.utils.common import ErrorHandler
from src.utils.common import FileUtils

from src.core.analysis_orchestrator import AnalysisOrchestrator, AnalysisConfig, WorkerContext


# 多进程工作上下文



def _deepmd_export_worker(task: tuple):
    """DeepMD导出并行工作函数

    参数
    ------
    task: tuple(system_path, system_name, sampled_frame_ids, output_root, force)
    返回
    ------
    str | None : 成功返回 system_name, 失败返回 None
    """
    try:
        system_path, system_name, sampled_frame_ids, output_root, force = task
        if not sampled_frame_ids:
            return (None, "No sampled_frame_ids")
        import os
        import dpdata
        from src.io.result_saver import ResultSaver  # 局部导入以兼容子进程
        ls = dpdata.LabeledSystem(system_path, fmt="abacus/lcao/md")
        ResultSaver.export_sampled_frames_per_system(
            frames=ls,
            sampled_frame_ids=sampled_frame_ids,
            system_path=system_path,
            output_root=output_root,
            system_name=system_name,
            logger=None,
            force=force
        )
        return (system_name, None)
    except Exception as e:
        return (None, f"{type(e).__name__}: {e}")


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
        
        # 直接使用orchestrator的execute_analysis方法
        analysis_results = self.orchestrator.execute_analysis(path_manager)
        
        # 保存结果
        self.orchestrator.save_results(analysis_results, path_manager)
        
        return analysis_results, path_manager, actual_output_dir
    
    def execute_deepmd_step(self, analysis_results: List[tuple], path_manager: PathManager, actual_output_dir: str) -> None:
        """执行DeepMD导出步骤（并行化实现）"""
        self.orchestrator.logger.info("开始独立DeepMD数据导出(支持并行)...")

        # 组装任务列表
        tasks = []
        out_root = os.path.join(actual_output_dir, 'deepmd_npy_per_system')
        os.makedirs(out_root, exist_ok=True)
        for result in analysis_results:
            if not result or len(result) < 2:
                continue
            metrics_obj = result[0]
            sampled_frame_ids = result[1]
            system_name = getattr(metrics_obj, 'system_name', None)
            system_path = getattr(metrics_obj, 'system_path', None)
            if system_name and system_path and sampled_frame_ids:
                tasks.append((system_path, system_name, sampled_frame_ids, out_root, self.orchestrator.config.force_recompute))

        if not tasks:
            self.orchestrator.logger.warning("没有可用于DeepMD导出的任务")
            return

        workers = self.orchestrator.determine_workers()
        if workers <= 1:
            # 回退顺序模式
            success = 0
            for t in tasks:
                name = _deepmd_export_worker(t)
                if name:
                    success += 1
                    self.orchestrator.logger.debug(f"DeepMD导出完成: {name}")
            self.orchestrator.logger.info(f"DeepMD导出完成: 成功 {success}/{len(tasks)} (顺序模式)")
            return

        from src.utils.common import run_parallel_tasks
        self.orchestrator.logger.info(f"DeepMD导出并行启动, 任务数 {len(tasks)}, workers={workers}")
        results = run_parallel_tasks(
            tasks=tasks,
            worker_fn=_deepmd_export_worker,
            workers=workers,
            mode="process",
            logger=None,  # 由下方统一输出日志
            desc="DeepMD导出"
        )
        success_count = 0
        for idx, (task, res) in enumerate(zip(tasks, results)):
            system_path, system_name, sampled_frame_ids, out_root, force = task
            name, err = res
            if name:
                success_count += 1
                self.orchestrator.logger.debug(f"DeepMD导出完成: {system_name}")
            else:
                self.orchestrator.logger.error(f"DeepMD导出失败: {system_name}, 错误: {err}")
        self.orchestrator.logger.info(f"DeepMD导出完成: 成功 {success_count}/{len(tasks)}")
    
    def execute_sampling_compare_step(self, analysis_results: List[tuple], path_manager: PathManager, actual_output_dir: str) -> None:
        """执行采样效果对比步骤"""
        self.orchestrator.logger.info("开始采样效果对比分析...")
        
        try:
            # 导入采样对比模块
            from src.analysis.sampling_comparison_analyser import analyse_sampling_compare
            
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
        
        total_targets = len(path_manager.targets)
        analysed_count = len(analysis_results)

        self.orchestrator.logger.info("=" * 60)
        self.orchestrator.logger.info(f"分析完成! 实际分析 {analysed_count}/{total_targets} 个体系")

        if analysis_results:
            # 输出采样统计
            swap_counts = [result[2] for result in analysis_results if len(result) > 2]
            if swap_counts:
                import numpy as np
                self.orchestrator.logger.info("采样优化统计:")
                self.orchestrator.logger.info("  平均交换次数: %.2f", np.mean(swap_counts))
                self.orchestrator.logger.info(f"  总交换次数: {int(sum(swap_counts))}")

        self.orchestrator.logger.info(f"总耗时: {elapsed:.1f}s ({elapsed/60:.1f} 分钟)")
        self.orchestrator.logger.info(f"结果目录: {output_dir}")
        
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
        
        config = AnalysisConfig(
            sample_ratio=args.sample_ratio,
            power_p=args.power_p,
            pca_variance_ratio=args.pca_variance_ratio,
            workers=args.workers,
            output_dir=args.output_dir,
            search_paths=args.search_path or [],
            include_project=args.include_project,
            force_recompute=args.force_recompute,
            steps=steps_to_run
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

    
    def _execute_workflow(self, config: AnalysisConfig, start_time: float) -> None:
        """执行主要工作流程（基于步骤列表）"""
        self.orchestrator.logger.info(f"开始执行分析流程，步骤: {config.steps}")
        
        # 初始化共享数据
        analysis_results = []
        path_manager = None
        actual_output_dir = None
        
        # 步骤1: 采样
        if 1 in config.steps:
            self.orchestrator.logger.info("执行步骤1: 采样")
            analysis_results, path_manager, actual_output_dir = self.workflow_executor.execute_sampling_step(config)
        
        # 步骤2: DeepMD导出
        if 2 in config.steps:
            self.orchestrator.logger.info("执行步骤2: DeepMD导出")
            if analysis_results:  # 如果步骤1已执行，使用其结果
                self.workflow_executor.execute_deepmd_step(analysis_results, path_manager, actual_output_dir)
            else:  # 如果只执行步骤2，加载已有数据
                analysis_results, path_manager, actual_output_dir = self.workflow_executor.load_existing_results_for_deepmd(config)
                if analysis_results:
                    self.workflow_executor.execute_deepmd_step(analysis_results, path_manager, actual_output_dir)
                else:
                    self.orchestrator.logger.warning("步骤2: 未找到可用的采样数据，跳过DeepMD导出")
        
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