#!/usr/bin/env python
"""Common utilities and helpers for the project."""

import os
import logging
import traceback
from typing import Any, List, Optional, Union

import numpy as np


class CommonUtils:
    """Common utility functions used across the project."""

    @staticmethod
    def ensure_directory(path: str) -> None:
        """Ensure directory exists, create if necessary."""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def get_project_root() -> str:
        """Get the project root directory."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate up to project root (assuming this is in src/utils/)
        project_root = os.path.dirname(os.path.dirname(current_dir))
        return project_root

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division that returns default value if denominator is zero."""
        if abs(denominator) < 1e-10:
            return default
        return numerator / denominator

    @staticmethod
    def nan_safe_mean(values: List[float]) -> float:
        """Calculate mean while safely handling NaN values."""
        if not values:
            return np.nan
        values_array = np.array(values)
        if np.all(np.isnan(values_array)):
            return np.nan
        return float(np.nanmean(values_array))

    @staticmethod
    def nan_safe_std(values: List[float], ddof: int = 1) -> float:
        """Calculate standard deviation while safely handling NaN values."""
        if not values or len(values) < 2:
            return np.nan
        values_array = np.array(values)
        if np.all(np.isnan(values_array)):
            return np.nan
        return float(np.nanstd(values_array, ddof=ddof))


class LoggingUtils:
    """Utilities for logging and error handling."""

    @staticmethod
    def setup_basic_logging(level: int = logging.INFO, log_file: Optional[str] = None):
        """Setup basic logging configuration."""
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                *(logging.FileHandler(log_file) for _ in [log_file] if log_file)
            ]
        )

    @staticmethod
    def log_exception(logger: logging.Logger, exception: Exception, context: str = ""):
        """Log exception with full traceback."""
        error_msg = f"{context}: {str(exception)}" if context else str(exception)
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")


# Re-exports for backward compatibility
ensure_directory = CommonUtils.ensure_directory
get_project_root = CommonUtils.get_project_root
safe_divide = CommonUtils.safe_divide
nan_safe_mean = CommonUtils.nan_safe_mean
nan_safe_std = CommonUtils.nan_safe_std
