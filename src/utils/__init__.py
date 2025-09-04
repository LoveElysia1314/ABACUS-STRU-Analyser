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
from .common import Constants
from .common import run_parallel_tasks, setup_multiprocess_logging, stop_multiprocess_logging

# Re-export commonly used classes at package level
__all__ = [
    # Core utility classes
    "ValidationUtils",
    "DataUtils",
    "MathUtils",
    "FileUtils",
    "Constants",
    # Parallel execution utilities
    "run_parallel_tasks",
    "setup_multiprocess_logging",
    "stop_multiprocess_logging",
    # Logging classes
    "LoggerManager",
    "create_standard_logger",
]


# Constants and configuration


# Ensure package can be imported and used as before
if __name__ == "__main__":
    # Basic functionality test
    print("ABACUS-STRU-Analyser v2.0 Utils Package")
    print(f"Available utilities: {', '.join(__all__)}")
