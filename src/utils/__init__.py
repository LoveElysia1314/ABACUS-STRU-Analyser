#!/usr/bin/env python
"""
Refactored utilities package for ABACUS-STRU-Analyser v2.0

This package provides modular utility functions organized by functionality:
- data_utils: Data validation and processing
- file_utils: File and directory operations
- logging: Enhanced logging management (imported from separate package)
"""

from typing import Dict, List

# Import from new modular structure
# Import logging utilities from dedicated package
from .logmanager import LoggerManager, create_standard_logger

# Import core utilities
from .common import DataUtils, MathUtils, ValidationUtils
from .common import FileUtils

# Re-export commonly used classes at package level
__all__ = [
    # Core utility classes
    "ValidationUtils",
    "DataUtils",
    "MathUtils",
    "FileUtils",
    # Logging classes
    "LoggerManager",
    "create_standard_logger",
    # Constants
    "Constants",
    # Directory discovery
    "DirectoryDiscovery",
]


class Constants:
    """Application constants"""

    # Default values
    DEFAULT_SAMPLE_RATIO = 0.1
    DEFAULT_POWER_P = -0.5
    DEFAULT_MAX_WORKERS = -1

    # File patterns
    STRU_FILE_PATTERN = "STRU_MD_*"
    OUTPUT_DIR_PATTERN = "OUT.ABACUS"

    # Analysis parameters
    MIN_FRAMES_REQUIRED = 2
    MAX_FRAMES_WARNING = 10000

    # Numerical tolerances
    EPSILON = 1e-15
    ZERO_THRESHOLD = 1e-12

    # Logging formats
    DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# Constants and configuration


# Ensure package can be imported and used as before
if __name__ == "__main__":
    # Basic functionality test
    print("ABACUS-STRU-Analyser v2.0 Utils Package")
    print(f"Available utilities: {', '.join(__all__)}")


class DirectoryDiscovery:
    """目录发现工具类，用于查找ABACUS系统目录"""

    @staticmethod
    def find_abacus_systems(search_path: str, include_project: bool = False) -> Dict[str, List[str]]:
        """查找指定路径下的ABACUS系统目录

        Args:
            search_path: 搜索路径
            include_project: 是否包含项目级别的目录

        Returns:
            按分子ID分组的系统路径字典
        """
        import os
        import re

        mol_systems = {}

        if not os.path.exists(search_path):
            return mol_systems

        # 查找所有OUT.ABACUS目录
        for root, dirs, files in os.walk(search_path):
            if "OUT.ABACUS" in dirs:
                system_path = root
                out_abacus_path = os.path.join(system_path, "OUT.ABACUS")

                # 检查是否包含STRU文件
                if os.path.exists(out_abacus_path):
                    stru_files = [f for f in os.listdir(out_abacus_path)
                                if f.startswith("STRU_MD_") and f.endswith(".xyz")]
                    if stru_files:
                        # 提取分子ID
                        mol_id = "unknown"
                        for stru_file in stru_files:
                            match = re.search(r'STRU_MD_(\d+)', stru_file)
                            if match:
                                mol_id = match.group(1)
                                break

                        if mol_id not in mol_systems:
                            mol_systems[mol_id] = []
                        mol_systems[mol_id].append(system_path)

        return mol_systems
