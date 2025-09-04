#!/usr/bin/env python
"""Analysis Orchestrator for coordinating analysis components."""

from typing import List


from ..utils.common import FileUtils

class AnalysisOrchestrator:
    """协调器类，用于组织和协调各个分析组件，降低耦合度。"""

    def __init__(self):
        self.logger = None
        self.path_manager = None
        self.system_analyser = None

    def discover_systems(self, search_paths: List[str], include_project: bool) -> List[str]:
        """发现系统目录"""
        return FileUtils.discover_systems(search_paths, include_project)

    def check_existing_results(self) -> bool:
        """检查是否已有完整结果"""
        return self.path_manager.check_existing_complete_results()
