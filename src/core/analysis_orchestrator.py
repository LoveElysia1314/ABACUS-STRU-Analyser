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
from ..utils.data_utils import ErrorHandler
from ..utils.file_utils import FileUtils
from ..analysis.correlation_analyser import CorrelationAnalyser


class AnalysisOrchestrator:
    """协调器类，用于组织和协调各个分析组件，降低耦合度。"""

    def __init__(self):
        self.logger = None
        self.path_manager = None
        self.system_analyser = None
        self.result_saver = None
        self.correlation_analyser = None

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
        self.correlation_analyser = CorrelationAnalyser(self.logger)

        return self.logger

    def discover_systems(self, search_paths: List[str], include_project: bool) -> List[str]:
        """发现系统目录"""
        from ..utils import DirectoryDiscovery
        return DirectoryDiscovery.discover_systems(search_paths, include_project)

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

    def run_correlation_analysis(self, output_dir: str):
        """运行相关性分析"""
        # 这里可以调用相关性分析模块
        self.logger.info("开始相关性分析...")
        # 具体实现可以后续添加

    def run_sampling_evaluation(self, output_dir: str):
        """运行采样效果评估"""
        try:
            from sampling_compare_demo import analyse_sampling_compare as SamplingComparisonRunner
            self.logger.info("开始采样效果评估...")
            SamplingComparisonRunner(output_dir)
        except Exception as e:
            self.logger.warning(f"采样效果评估失败: {e}")

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
