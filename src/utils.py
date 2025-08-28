#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具模块（从根目录的 utils.py 迁移）
"""

import os
import sys
import glob
import logging
import csv
from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd


class LoggerManager:
    @staticmethod
    def create_logger(name: str, level: int = logging.INFO, 
                     add_console: bool = True, 
                     log_file: Optional[str] = None,
                     log_format: str = '%(asctime)s - %(levelname)s - %(message)s',
                     date_format: str = '%H:%M:%S') -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers.clear()
        formatter = logging.Formatter(log_format, datefmt=date_format)
        if add_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        logger.propagate = False
        return logger

    @staticmethod
    def add_file_handler(logger: logging.Logger, log_file: str,
                        log_format: str = '%(asctime)s - %(levelname)s - %(message)s',
                        date_format: str = '%H:%M:%S') -> logging.FileHandler:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return file_handler

    @staticmethod
    def remove_handler(logger: logging.Logger, handler: logging.Handler) -> None:
        if handler in logger.handlers:
            logger.removeHandler(handler)
        handler.close()


class FileUtils:
    @staticmethod
    def ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def find_files(pattern: str, search_dir: str = ".", recursive: bool = True) -> List[str]:
        if recursive:
            full_pattern = os.path.join(search_dir, "**", pattern)
            return glob.glob(full_pattern, recursive=True)
        else:
            full_pattern = os.path.join(search_dir, pattern)
            return glob.glob(full_pattern)

    @staticmethod
    def find_file_prioritized(filename: str, search_dir: str = ".", 
                             priority_subdirs: Optional[List[str]] = None) -> Optional[str]:
        current_path = os.path.join(search_dir, filename)
        if os.path.exists(current_path):
            return current_path
        if priority_subdirs:
            for subdir in priority_subdirs:
                priority_path = os.path.join(search_dir, subdir, filename)
                if os.path.exists(priority_path):
                    return priority_path
        files = FileUtils.find_files(filename, search_dir, recursive=True)
        return files[0] if files else None

    @staticmethod
    def safe_write_csv(filepath: str, data: List[List[Any]], 
                      headers: Optional[List[str]] = None,
                      encoding: str = 'utf-8-sig') -> bool:
        try:
            FileUtils.ensure_dir(os.path.dirname(filepath))
            with open(filepath, 'w', newline='', encoding=encoding) as f:
                writer = csv.writer(f)
                if headers:
                    writer.writerow(headers)
                writer.writerows(data)
            return True
        except Exception:
            return False


class DataUtils:
    @staticmethod
    def to_python_types(data: Any) -> Any:
        if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
            result = []
            for item in data:
                if hasattr(item, 'item'):
                    result.append(item.item())
                elif isinstance(item, (np.integer, np.floating, np.complexfloating)):
                    result.append(item.item())
                elif str(item).isdigit():
                    result.append(int(item))
                else:
                    result.append(item)
            return result
        else:
            if hasattr(data, 'item'):
                return data.item()
            elif isinstance(data, (np.integer, np.floating, np.complexfloating)):
                return data.item()
            elif str(data).isdigit():
                return int(data)
            else:
                return data

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        return numerator / denominator if abs(denominator) > 1e-12 else default

    @staticmethod
    def format_number(value: float, precision: int = 6) -> str:
        return f"{value:.{precision}f}"

    @staticmethod
    def check_required_columns(df: pd.DataFrame, required_columns: List[str]) -> List[str]:
        return [col for col in required_columns if col not in df.columns]

    @staticmethod
    def clean_dataframe(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return df[columns].dropna()


class MathUtils:
    @staticmethod
    def safe_sqrt(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(value, np.ndarray):
            return np.sqrt(np.maximum(0, value))
        else:
            return np.sqrt(max(0, value))

    @staticmethod
    def safe_log(value: float, base: Optional[float] = None) -> float:
        value = max(1e-12, value)
        if base is None:
            return np.log(value)
        else:
            return np.log(value) / np.log(base)

    @staticmethod
    def normalize_vector(vector: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector / norm if norm > epsilon else vector

    @staticmethod
    def power_mean(arr: np.ndarray, p: float) -> float:
        arr = np.asarray(arr)
        arr = np.maximum(arr, 1e-12)
        if p == 0:
            return np.exp(np.mean(np.log(arr)))
        elif p == 1:
            return np.mean(arr)
        elif p == -1:
            return len(arr) / np.sum(1.0 / arr)
        else:
            return (np.mean(arr ** p)) ** (1.0 / p)

    @staticmethod
    def calculate_rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
        return np.sqrt(np.mean((predicted - actual) ** 2))

    @staticmethod
    def calculate_correlation_strength(abs_r: float) -> str:
        if abs_r < 0.3:
            return "弱相关"
        elif abs_r < 0.7:
            return "中等相关"
        else:
            return "强相关"

    @staticmethod
    def calculate_effect_size_interpretation(eta_squared: float) -> str:
        if eta_squared < 0.01:
            return "无效应"
        elif eta_squared < 0.06:
            return "小效应"
        elif eta_squared < 0.14:
            return "中等效应"
        else:
            return "大效应"


class ValidationUtils:
    @staticmethod
    def validate_file_exists(filepath: str) -> bool:
        return os.path.exists(filepath) and os.path.isfile(filepath)

    @staticmethod
    def validate_sample_size(data: Union[pd.DataFrame, np.ndarray, List], 
                           min_size: int = 2) -> bool:
        return len(data) >= min_size

    @staticmethod
    def validate_numeric_data(data: Union[pd.Series, np.ndarray, List]) -> bool:
        try:
            np.asarray(data, dtype=float)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_groups_for_anova(groups: List[np.ndarray], min_group_size: int = 2) -> bool:
        if len(groups) < 2:
            return False
        return all(len(group) >= min_group_size for group in groups)


class Constants:
    EPSILON = 1e-12
    SMALL_NUMBER = 1e-6
    ALPHA_0_05 = 0.05
    ALPHA_0_01 = 0.01
    # Default logging formatters used by various modules (fallbacks available)
    DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    DEFAULT_DATE_FORMAT = '%H:%M:%S'


class DirectoryDiscovery:
    """Minimal directory discovery utilities.

    This provides a lightweight implementation compatible with the
    original project's usage for finding ABACUS system directories.
    For the smoke test it will return an empty mapping when no
    search path is provided or when no matching directories are found.
    """

    @staticmethod
    def find_abacus_systems(search_path: Optional[str] = None) -> Dict[str, List[str]]:
        """Find ABACUS system directories.

        Args:
            search_path: root to search. If None, use parent of cwd.

        Returns:
            Mapping from molecule id (str) to list of system directory paths.
            Returns empty dict when nothing found.
        """
        # Determine root to search
        root = search_path
        if not root:
            root = os.path.abspath(os.path.join(os.getcwd(), '..'))
        root = os.path.abspath(root)

        # Compute this project's top-level directory and common dirs to ignore
        this_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        ignored_dirnames = {'__pycache__', '.git', 'analysis_results', 'venv', 'env', 'node_modules', '.venv'}

        found: Dict[str, List[str]] = {}
        try:
            # Walk top-down so we can prune directories we don't need to descend into
            for dirpath, dirnames, filenames in os.walk(root, topdown=True):
                # Normalize path for comparisons
                norm_dirpath = os.path.abspath(dirpath)

                # Skip scanning inside this project folder itself to avoid detecting test data or outputs
                if norm_dirpath.startswith(this_project_root):
                    # don't descend further from this directory
                    dirnames[:] = []
                    continue

                # Prune common ignored directories to speed up traversal
                if dirnames:
                    dirnames[:] = [d for d in dirnames if d not in ignored_dirnames]

                # look for the ABACUS STRU directory structure
                if 'OUT.ABACUS' in dirnames:
                    stru_path = os.path.join(dirpath, 'OUT.ABACUS', 'STRU')
                    if os.path.isdir(stru_path):
                        # Derive molecule id from directory name if possible
                        base = os.path.basename(dirpath)
                        mol_key = base
                        found.setdefault(mol_key, []).append(dirpath)
        except Exception:
            # On error, return empty mapping to allow graceful exit
            return {}

        return found


def create_standard_logger(name: str, output_dir: str, filename: str) -> logging.Logger:
    """Create a standard logger writing to output_dir/filename and console.

    Small wrapper around LoggerManager.create_logger to match previous
    project API expected by `abacus_main_analyzer`.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, filename)
    return LoggerManager.create_logger(name, level=logging.INFO, add_console=True, log_file=log_file)

