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
from ..logmanager import LoggerManager, create_standard_logger

# Legacy imports for backward compatibility
from .data_utils import DataUtils, MathUtils, ValidationUtils, is_empty_unified
from .file_utils import FileUtils

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
    # Legacy compatibility
    "is_empty_unified",
    # Constants (if needed)
    "Constants",
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

    @staticmethod
    def find_abacus_systems(search_path: str = ".", include_project: bool = False) -> dict:
        """Find ABACUS systems organized by molecule ID
        
        Args:
            search_path: Path to search in
            include_project: Whether to include project directories
            
        Returns:
            Dictionary mapping molecule IDs to system paths
        """
        import glob
        import os
        import re

        mol_systems = {}
        
        # Search for ABACUS output directories with STRU subdirectories
        pattern = os.path.join(search_path, "**", Constants.OUTPUT_DIR_PATTERN, "STRU")
        
        for stru_dir in glob.glob(pattern, recursive=True):
            if os.path.isdir(stru_dir):
                # Check if contains STRU_MD_* files
                stru_files = glob.glob(
                    os.path.join(stru_dir, Constants.STRU_FILE_PATTERN)
                )
                if stru_files:
                    # Get system directory (grandparent of STRU)
                    system_dir = os.path.dirname(os.path.dirname(stru_dir))
                    
                    # Filter out project directories if needed
                    if not include_project and DirectoryDiscovery._is_project_directory(system_dir):
                        continue
                    
                    # Extract molecule ID from path
                    mol_id = DirectoryDiscovery._extract_molecule_id(system_dir)
                    
                    if mol_id not in mol_systems:
                        mol_systems[mol_id] = []
                    mol_systems[mol_id].append(system_dir)
        
        # Sort systems within each molecule
        for mol_id in mol_systems:
            mol_systems[mol_id] = sorted(mol_systems[mol_id])
            
        return mol_systems
    
    @staticmethod
    def _extract_molecule_id(system_path: str) -> str:
        """Extract molecule ID from system path"""
        import re
        import os
        
        # Try to extract from directory name patterns like struct_mol_XXX
        dirname = os.path.basename(system_path)
        match = re.search(r'mol_(\d+)', dirname)
        if match:
            return match.group(1)
        
        # Fallback: use directory name
        return dirname
    
    @staticmethod
    def _is_project_directory(system_path: str) -> bool:
        """Check if directory appears to be a project directory"""
        import os
        
        dirname = os.path.basename(system_path).lower()
        project_patterns = [
            'abacus-stru-analyser', 'analyser', 'analyser',
            'test', 'demo', 'example', 'sample'
        ]
        
        # Don't filter out struct_mol_XXX directories - these are actual data
        if dirname.startswith('struct_mol_'):
            return False
            
        return any(pattern in dirname for pattern in project_patterns)


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
