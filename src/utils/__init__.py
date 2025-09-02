#!/usr/bin/env python
"""
Refactored utilities package for ABACUS-STRU-Analyser v2.0

This package provides modular utility functions organized by functionality:
- data_utils: Data validation and processing
- file_utils: File and directory operations
- logging: Enhanced logging management (imported from separate package)

For backward compatibility, commonly used classes are re-exported at package level.
"""

# Import from new modular structure
# Import logging utilities from dedicated package
from .logmanager import LoggerManager, create_standard_logger

# Legacy imports for backward compatibility
from .common import DataUtils, MathUtils, ValidationUtils
from .common import FileUtils

# Re-export commonly used classes at package level
__all__ = [
    # New modular classes
    "ValidationUtils",
    "DataUtils",
    "MathUtils",
    "FileUtils",
    # Logging classes
    "LoggerManager",
    "create_standard_logger",
    # Constants and legacy classes
    "Constants",
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


# Backward compatibility aliases
# These maintain the old interface while redirecting to new implementations


class DirectoryDiscovery:
    """Legacy compatibility class for directory discovery"""

    @staticmethod
    def find_structure_directories(search_path: str = ".") -> list:
        """Find directories containing ABACUS structure files

        Args:
            search_path: Path to search in

        Returns:
            List of directories containing structure files
        """
        import glob
        import os

        structure_dirs = []
        pattern = os.path.join(search_path, "**", Constants.OUTPUT_DIR_PATTERN, "STRU")

        for stru_dir in glob.glob(pattern, recursive=True):
            if os.path.isdir(stru_dir):
                # Check if contains STRU files
                stru_files = glob.glob(
                    os.path.join(stru_dir, Constants.STRU_FILE_PATTERN)
                )
                if stru_files:
                    # Return parent directory (system directory)
                    system_dir = os.path.dirname(os.path.dirname(stru_dir))
                    if system_dir not in structure_dirs:
                        structure_dirs.append(system_dir)

        return sorted(structure_dirs)

    # ---- Newly integrated advanced discovery helpers (moved from MeanStructureManager) ----
    @staticmethod
    def find_abacus_systems(search_path: str = ".", include_project: bool = False) -> dict:
        """Find ABACUS systems organized by molecule ID.

        Returns dict[molecule_id] -> list[system_dir].
        """
        import glob
        import os
        import re

        mol_systems = {}
        pattern = os.path.join(search_path, "**", Constants.OUTPUT_DIR_PATTERN, "STRU")
        for stru_dir in glob.glob(pattern, recursive=True):
            if os.path.isdir(stru_dir):
                stru_files = glob.glob(os.path.join(stru_dir, Constants.STRU_FILE_PATTERN))
                if not stru_files:
                    continue
                system_dir = os.path.dirname(os.path.dirname(stru_dir))
                if (not include_project) and DirectoryDiscovery._is_project_directory(system_dir):
                    continue
                mol_id = DirectoryDiscovery._extract_molecule_id(system_dir)
                mol_systems.setdefault(mol_id, []).append(system_dir)
        for mol_id in mol_systems:
            mol_systems[mol_id] = sorted(mol_systems[mol_id])
        return mol_systems

    @staticmethod
    def _extract_molecule_id(system_path: str) -> str:
        import re
        import os
        dirname = os.path.basename(system_path)
        match = re.search(r"mol_(\d+)", dirname)
        if match:
            return match.group(1)
        return dirname

    @staticmethod
    def _is_project_directory(system_path: str) -> bool:
        import os
        dirname = os.path.basename(system_path).lower()
        project_patterns = [
            'abacus-stru-analyser', 'analyser', 'analysis',
            'test', 'demo', 'example', 'sample'
        ]
        if dirname.startswith('struct_mol_'):
            return False
        return any(p in dirname for p in project_patterns)


class MeanStructureManager:
    """统一加载与访问均值构象数据的辅助类。

    目标：在分析阶段导出的 mean_structures/mean_structure_<system>.json 可被后续
    采样验证脚本、可视化、下游模型处理直接复用，而无需重新迭代对齐。
    """

    @staticmethod
    def load_mean_structure(run_dir: str, system_name: str):
        """加载指定体系的均值构象与元数据。

        Returns:
            (mean_structure_np, meta_dict) or (None, None)
        """
        import os, json, numpy as np
        path = os.path.join(run_dir, 'mean_structures', f'mean_structure_{system_name}.json')
        if not os.path.exists(path):
            return None, None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            ms = np.array(data.get('mean_structure', []), dtype=float)
            return ms, data
        except Exception:
            return None, None

    @staticmethod
    def list_available(run_dir: str):
        """列出当前 run 目录下可用的均值构象 system 名称列表。"""
        import os, re
        mean_dir = os.path.join(run_dir, 'mean_structures')
        if not os.path.isdir(mean_dir):
            return []
        names = []
        for fn in os.listdir(mean_dir):
            if fn.startswith('mean_structure_') and fn.endswith('.json'):
                sys_name = fn[len('mean_structure_'):-5]
                # 基本格式校验
                if re.match(r"struct_mol_\d+_conf_\d+_T\d+K", sys_name):
                    names.append(sys_name)
        return sorted(names)

    @staticmethod
    def ensure_loaded_or_compute(run_dir: str, system_dir: str, recompute_fn=None):
        """确保获得均值构象；若不存在且提供 recompute_fn 则调用生成。

        recompute_fn: callable -> (mean_structure: np.ndarray)
        返回 (mean_structure_np or None)
        """
        import os
        system_name = os.path.basename(system_dir.rstrip("/\\"))
        ms, _ = MeanStructureManager.load_mean_structure(run_dir, system_name)
        if ms is not None:
            return ms
        if recompute_fn is not None:
            try:
                ms = recompute_fn()
                return ms
            except Exception:
                return None
        return None

    # ---- Compatibility alias for discovery helpers ----
    @staticmethod
    def find_abacus_systems(search_path: str = ".", include_project: bool = False) -> dict:  # pragma: no cover - thin wrapper
        return DirectoryDiscovery.find_abacus_systems(search_path, include_project)


# For even better backward compatibility, create aliases for the old utils.py functions
def create_logger_legacy(*args, **kwargs):
    """Legacy wrapper for logger creation"""
    return LoggerManager.create_logger(*args, **kwargs)


# Add legacy aliases to __all__ for complete compatibility
__all__.extend(["DirectoryDiscovery", "create_logger_legacy"])


# Ensure package can be imported and used as before
if __name__ == "__main__":
    # Basic functionality test
    print("ABACUS-STRU-Analyser v2.0 Utils Package")
    print(f"Available utilities: {', '.join(__all__)}")
