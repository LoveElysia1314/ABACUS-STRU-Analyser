#!/usr/bin/env python
"""
File operation utilities
"""

import csv
import glob
import logging
import os
from pathlib import Path
from typing import Any, List, Optional


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
