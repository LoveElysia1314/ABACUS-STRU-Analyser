#!/usr/bin/env python
"""
Backward compatibility module for ABACUS-STRU-Analyser v2.0

This module maintains backward compatibility with v1.x imports while
redirecting to the new modular structure. It re-exports all commonly
used classes and functions from their new locations.

For new code, prefer importing directly from the specific modules:
- src.utils.data_utils
- src.utils.file_utils
- src.logging.manager
"""

# Import from new modular structure
from .logging.manager import LoggerManager, create_standard_logger

# Import legacy compatibility classes
from .utils import Constants, DirectoryDiscovery
from .utils.data_utils import DataUtils, MathUtils, ValidationUtils
from .utils.file_utils import FileUtils

# Re-export all for backward compatibility
__all__ = [
    # Data utilities
    "ValidationUtils",
    "DataUtils",
    "MathUtils",
    # File utilities
    "FileUtils",
    # Logging utilities
    "LoggerManager",
    "create_standard_logger",
    # Legacy compatibility
    "DirectoryDiscovery",
    "Constants",
]

# Deprecation notice for direct usage
import warnings


def _deprecated_import_warning():
    """Issue deprecation warning for direct utils.py imports"""
    warnings.warn(
        "Direct import from 'src.utils' is deprecated in v2.0. "
        "Please import from specific modules: "
        "'src.utils.data_utils', 'src.utils.file_utils', 'src.logging.manager'",
        DeprecationWarning,
        stacklevel=3,
    )


# Issue warning when this module is imported (except during testing)
import sys

if not any("test" in arg.lower() for arg in sys.argv):
    _deprecated_import_warning()
