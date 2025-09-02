#!/usr/bin/env python
"""Common utilities and helpers for the project."""

import os
import logging
import traceback
import csv
import glob
from pathlib import Path
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


# Re-exports for backward compatibility
ensure_directory = CommonUtils.ensure_directory
get_project_root = CommonUtils.get_project_root
safe_divide = CommonUtils.safe_divide
nan_safe_mean = CommonUtils.nan_safe_mean
nan_safe_std = CommonUtils.nan_safe_std


# ---- Data Processing Utilities (merged from data_utils.py) ----

class ValidationUtils:
    """Utilities for data validation and checking"""

    @staticmethod
    def is_empty(obj: Any) -> bool:
        """Unified empty check for various data types

        Args:
            obj: Object to check (None, list, numpy array, etc.)

        Returns:
            True if object is empty or None, False otherwise
        """
        if obj is None:
            return True

        # For numpy arrays, check size attribute first
        if hasattr(obj, "size"):
            return obj.size == 0

        # For containers with len() method
        if hasattr(obj, "__len__"):
            return len(obj) == 0

        # For other types, consider non-None as non-empty
        return False

    @staticmethod
    def validate_file_exists(file_path: str) -> bool:
        """Check if file exists"""
        import os
        return os.path.isfile(file_path)

    @staticmethod
    def validate_sample_size(data, min_size: int = 2) -> bool:
        """Check if data has minimum required size"""
        try:
            return len(data) >= min_size
        except (TypeError, AttributeError):
            return False


class ErrorHandler:
    """Utilities for enhanced error handling and logging"""

    @staticmethod
    def log_detailed_error(logger: logging.Logger, error: Exception, context: str = "",
                          additional_info: dict = None) -> None:
        """Log detailed error information including stack trace and context

        Args:
            logger: Logger instance to use for logging
            error: The exception that occurred
            context: Additional context information about where the error occurred
            additional_info: Dictionary of additional information to log
        """
        error_details = traceback.format_exc()

        if context:
            logger.error(f"{context}: {str(error)}")
        else:
            logger.error(f"错误: {str(error)}")

        logger.error(f"详细错误信息:\n{error_details}")

        # Log additional context information
        if additional_info:
            for key, value in additional_info.items():
                logger.error(f"{key}: {value}")

        # Log chained exceptions if present
        if hasattr(error, '__cause__') and error.__cause__:
            logger.error(f"根本原因: {error.__cause__}")
        if hasattr(error, '__context__') and error.__context__:
            logger.error(f"上下文: {error.__context__}")


class MathUtils:
    """Mathematical utility functions"""

    @staticmethod
    def normalize_array(arr: np.ndarray) -> np.ndarray:
        """Normalize array to zero mean and unit variance

        Args:
            arr: Input array

        Returns:
            Normalized array
        """
        if ValidationUtils.is_empty(arr):
            return arr

        mean_val = np.mean(arr)
        std_val = np.std(arr)

        if std_val < 1e-15:  # Constant array
            return np.zeros_like(arr)

        return (arr - mean_val) / std_val

    @staticmethod
    def safe_log(value: float, default: float = 0.0) -> float:
        """Safe logarithm calculation

        Args:
            value: Input value
            default: Value to return for invalid inputs

        Returns:
            Natural logarithm or default value
        """
        if value <= 0:
            return default
        return np.log(value)

    @staticmethod
    def safe_sqrt(value: Union[float, np.ndarray], default: Union[float, np.ndarray] = 0.0) -> Union[float, np.ndarray]:
        """Safe square root calculation

        Args:
            value: Input value (scalar or array)
            default: Value to return for invalid inputs

        Returns:
            Square root or default value
        """
        if isinstance(value, np.ndarray):
            # Handle array input
            result = np.where(value < 0, default, np.sqrt(value))
            return result
        else:
            # Handle scalar input
            if value < 0:
                return default
            return np.sqrt(value)

    @staticmethod
    def power_mean(values: np.ndarray, p: float, default: float = 0.0) -> float:
        """Calculate power mean of values

        Args:
            values: Input array of values
            p: Power parameter
            default: Default value to return for invalid inputs

        Returns:
            Power mean value
        """
        if len(values) == 0:
            return default

        if p == 0:
            # Geometric mean for p=0
            return np.exp(np.mean(np.log(values + 1e-15)))
        else:
            return np.power(np.mean(np.power(values, p)), 1.0/p)

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with default value for zero denominator"""
        if abs(denominator) < 1e-15:
            return default
        return numerator / denominator


class DataUtils:
    """Data processing and validation utilities"""

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with default value for zero denominator"""
        return MathUtils.safe_divide(numerator, denominator, default)

    @staticmethod
    def safe_mean(
        values: Optional[Union[List[float], np.ndarray]], default: float = 0.0
    ) -> float:
        """Safe mean calculation with empty data handling

        Args:
            values: Values to calculate mean for
            default: Value to return if values is empty or None

        Returns:
            Mean value or default
        """
        if ValidationUtils.is_empty(values):
            return default

        if isinstance(values, np.ndarray):
            return float(np.mean(values))
        else:
            return sum(values) / len(values)

    @staticmethod
    def safe_std(
        values: Optional[Union[List[float], np.ndarray]], default: float = 0.0
    ) -> float:
        """Safe standard deviation calculation

        Args:
            values: Values to calculate std for
            default: Value to return if values is empty or None

        Returns:
            Standard deviation or default
        """
        if ValidationUtils.is_empty(values):
            return default

        if isinstance(values, np.ndarray):
            return float(np.std(values))
        else:
            if len(values) == 1:
                return 0.0
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            return variance**0.5

    @staticmethod
    def to_python_types(obj):
        """Convert numpy types to Python native types"""
        try:
            if isinstance(obj, (list, tuple)):
                return [DataUtils.to_python_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        except ImportError:
            return obj

    @staticmethod
    def check_required_columns(df, required_columns: list) -> list:
        """Check if DataFrame has required columns"""
        try:
            missing = [col for col in required_columns if col not in df.columns]
            return missing
        except Exception:
            return required_columns

    @staticmethod
    def clean_dataframe(df, columns: list):
        """Clean DataFrame by removing rows with NaN in specified columns"""
        try:
            import pandas as pd
            return df.dropna(subset=columns)
        except Exception:
            return df

    @staticmethod
    def format_number(num, decimals: int = 3) -> str:
        """Format number with specified decimal places"""
        try:
            return f"{num:.{decimals}f}"
        except Exception:
            return str(num)


# ---- File Operation Utilities (merged from file_utils.py) ----

class FileUtils:
    """File and directory operation utilities"""

    @staticmethod
    def ensure_dir(path: str) -> None:
        """Ensure directory exists, create if necessary

        Args:
            path: Directory path to ensure
        """
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def find_files(
        pattern: str, search_dir: str = ".", recursive: bool = True
    ) -> List[str]:
        """Find files matching pattern

        Args:
            pattern: File pattern to match
            search_dir: Directory to search in
            recursive: Whether to search recursively

        Returns:
            List of matching file paths
        """
        if recursive:
            full_pattern = os.path.join(search_dir, "**", pattern)
            return glob.glob(full_pattern, recursive=True)
        else:
            full_pattern = os.path.join(search_dir, pattern)
            return glob.glob(full_pattern)

    @staticmethod
    def find_file_prioritized(
        filename: str,
        search_dir: str = ".",
        priority_subdirs: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Find file with priority search in specific subdirectories

        Args:
            filename: Name of file to find
            search_dir: Root directory to search
            priority_subdirs: Subdirectories to check first

        Returns:
            Path to found file or None
        """
        # Check current directory first
        current_path = os.path.join(search_dir, filename)
        if os.path.exists(current_path):
            return current_path

        # Check priority subdirectories
        if priority_subdirs:
            for subdir in priority_subdirs:
                priority_path = os.path.join(search_dir, subdir, filename)
                if os.path.exists(priority_path):
                    return priority_path

        # Fall back to recursive search
        files = FileUtils.find_files(filename, search_dir, recursive=True)
        return files[0] if files else None

    @staticmethod
    def safe_write_csv(
        filepath: str,
        data: List[List[Any]],
        headers: Optional[List[str]] = None,
        encoding: str = "utf-8-sig",
    ) -> bool:
        """Safely write data to CSV file

        Args:
            filepath: Path to CSV file
            data: Data rows to write
            headers: Optional header row
            encoding: File encoding

        Returns:
            True if successful, False otherwise
        """
        try:
            FileUtils.ensure_dir(os.path.dirname(filepath))
            with open(filepath, "w", newline="", encoding=encoding) as f:
                writer = csv.writer(f)
                if headers:
                    writer.writerow(headers)
                writer.writerows(data)
            return True
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to write CSV file {filepath}: {e}")
            return False

    @staticmethod
    def safe_read_csv(
        filepath: str, encoding: str = "utf-8"
    ) -> Optional[List[List[str]]]:
        """Safely read CSV file

        Args:
            filepath: Path to CSV file
            encoding: File encoding

        Returns:
            List of rows or None if failed
        """
        try:
            with open(filepath, encoding=encoding) as f:
                reader = csv.reader(f)
                return list(reader)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to read CSV file {filepath}: {e}")
            return None

    @staticmethod
    def safe_remove(filepath: str) -> bool:
        """Safely remove file

        Args:
            filepath: Path to file to remove

        Returns:
            True if successful or file doesn't exist, False otherwise
        """
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to remove file {filepath}: {e}")
            return False

    @staticmethod
    def get_file_size(filepath: str) -> int:
        """Get file size in bytes

        Args:
            filepath: Path to file

        Returns:
            File size in bytes, 0 if file doesn't exist or error
        """
        try:
            return os.path.getsize(filepath) if os.path.exists(filepath) else 0
        except Exception:
            return 0

    @staticmethod
    def is_file_newer(file1: str, file2: str) -> bool:
        """Check if file1 is newer than file2

        Args:
            file1: First file path
            file2: Second file path

        Returns:
            True if file1 is newer than file2
        """
        try:
            if not (os.path.exists(file1) and os.path.exists(file2)):
                return False
            return os.path.getmtime(file1) > os.path.getmtime(file2)
        except Exception:
            return False

    @staticmethod
    def get_project_root() -> str:
        """Get the project root directory

        Returns:
            Path to project root directory
        """
        # Get the directory containing this file
        current_file = os.path.abspath(__file__)
        # Go up to utils directory
        utils_dir = os.path.dirname(current_file)
        # Go up to src directory
        src_dir = os.path.dirname(utils_dir)
        # Go up one more level to project root
        project_root = os.path.dirname(src_dir)
        return project_root
