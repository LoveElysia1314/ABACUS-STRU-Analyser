"""Enhanced logging manager for ABACUS-STRU-Analyser v2.0"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


class LoggerManager:
    """Centralized logger management with consistent formatting and configuration"""

    DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    @staticmethod
    def create_logger(
        name: str,
        level: int = logging.INFO,
        add_console: bool = True,
        log_file: Optional[str] = None,
        log_format: str = DEFAULT_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT,
    ) -> logging.Logger:
        """Create a standardized logger instance

        Args:
            name: Logger name (typically __name__ from calling module)
            level: Logging level (default: INFO)
            add_console: Whether to add console handler
            log_file: Optional log file path
            log_format: Custom format string
            date_format: Custom date format string

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(log_format, datefmt=date_format)

        # Add console handler
        if add_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file:
            LoggerManager._add_file_handler_internal(logger, log_file, formatter)

        # Prevent propagation to root logger
        logger.propagate = False

        return logger

    @staticmethod
    def _add_file_handler_internal(
        logger: logging.Logger, log_file: str, formatter: logging.Formatter
    ) -> logging.FileHandler:
        """Internal method to add file handler"""
        # Ensure directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return file_handler

    @staticmethod
    def add_file_handler(
        logger: logging.Logger,
        log_file: str,
        log_format: str = DEFAULT_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT,
    ) -> logging.FileHandler:
        """Add file handler to existing logger

        Args:
            logger: Logger instance to modify
            log_file: Path to log file
            log_format: Format string for log messages
            date_format: Format string for timestamps

        Returns:
            Created file handler
        """
        formatter = logging.Formatter(log_format, datefmt=date_format)
        return LoggerManager._add_file_handler_internal(logger, log_file, formatter)

    @staticmethod
    def remove_handler(logger: logging.Logger, handler: logging.Handler) -> None:
        """Safely remove and close a handler

        Args:
            logger: Logger instance to modify
            handler: Handler to remove
        """
        if handler in logger.handlers:
            logger.removeHandler(handler)
        handler.close()

    @staticmethod
    def create_analysis_logger(
        name: str, output_dir: str, log_filename: str = "analysis.log"
    ) -> logging.Logger:
        """Create logger specifically for analysis operations

        Args:
            name: Logger name
            output_dir: Directory for log files
            log_filename: Name of log file

        Returns:
            Configured logger with both console and file output
        """
        log_path = os.path.join(output_dir, log_filename)
        return LoggerManager.create_logger(
            name=name,
            level=logging.INFO,
            add_console=True,
            log_file=log_path,
            log_format=LoggerManager.DEFAULT_FORMAT,
            date_format=LoggerManager.DEFAULT_DATE_FORMAT,
        )

    @staticmethod
    def setup_root_logger(level: int = logging.INFO) -> None:
        """Configure root logger for the application

        Args:
            level: Root logging level
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add console handler with standard format
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            LoggerManager.DEFAULT_FORMAT, datefmt=LoggerManager.DEFAULT_DATE_FORMAT
        )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)


def create_standard_logger(
    name: str, level: int = logging.INFO, log_file: Optional[str] = None
) -> logging.Logger:
    """Convenience function for creating standard loggers

    Small wrapper around LoggerManager.create_logger to match previous
    usage patterns while maintaining new functionality.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    return LoggerManager.create_logger(
        name=name, level=level, add_console=True, log_file=log_file
    )
