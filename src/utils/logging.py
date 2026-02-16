"""Logging Configuration Utility

This module provides centralized logging setup for the application.
It configures both console and file handlers with customizable log levels.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union
from zoneinfo import ZoneInfo


class TimezoneFormatter(logging.Formatter):
    """Custom formatter that supports timezone-aware timestamps.

    Converts log record timestamps to the specified timezone before formatting.
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        timezone: Optional[str] = None,
    ):
        """Initialize formatter with timezone support.

        Args:
            fmt: Log message format string
            datefmt: Timestamp format string
            timezone: Timezone string ('UTC', 'local', or IANA timezone like 'Asia/Tokyo')
        """
        super().__init__(fmt, datefmt)
        self.timezone = self._get_timezone(timezone)

    def _get_timezone(self, timezone: Optional[str]) -> Optional[ZoneInfo]:
        """Get ZoneInfo object from timezone string.

        Args:
            timezone: Timezone string ('UTC', 'local', or IANA timezone)

        Returns:
            ZoneInfo object or None for local time
        """
        if timezone is None or timezone.lower() == "local":
            return None  # Use local time
        elif timezone.upper() == "UTC":
            return ZoneInfo("UTC")
        else:
            return ZoneInfo(timezone)  # IANA timezone (e.g., 'Asia/Tokyo')

    def formatTime(
        self, record: logging.LogRecord, datefmt: Optional[str] = None
    ) -> str:
        """Format the log record timestamp with timezone conversion.

        Args:
            record: Log record to format
            datefmt: Optional date format override

        Returns:
            Formatted timestamp string
        """
        # Create datetime from record timestamp
        dt = datetime.fromtimestamp(record.created, tz=self.timezone)

        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.isoformat()


def setup_logging(
    log_file: Union[str, Path],
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    timezone: Optional[str] = None,
    module_levels: Optional[Dict[str, str]] = None,
) -> logging.Logger:
    """Configure application-wide logging with console and file handlers.

    Args:
        log_file: Path to log file
        console_level: Log level for console output (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file_level: Log level for file output
        log_format: Custom format string for log messages
        date_format: Custom format for timestamps
        timezone: Timezone for timestamps ('UTC', 'local', or IANA timezone like 'Asia/Tokyo')
        module_levels: Dict of module names to log levels for fine-grained control

    Returns:
        Root logger instance

    Example:
        >>> logger = setup_logging(
        ...     log_file="outputs/logs/train_20260216_143022.log",
        ...     console_level="INFO",
        ...     file_level="DEBUG",
        ...     timezone="UTC"
        ... )
        >>> logger.info("Logging initialized")
    """
    # Convert string path to Path object
    log_file = Path(log_file)

    # Create log directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Set default format if not provided
    if log_format is None:
        log_format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    # Set default date format if not provided
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    # Set default timezone if not provided
    if timezone is None:
        timezone = "UTC"

    # Create formatter with timezone support
    formatter = TimezoneFormatter(log_format, datefmt=date_format, timezone=timezone)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set to DEBUG to allow all messages through

    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(getattr(logging, file_level.upper()))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Configure module-specific log levels if provided
    if module_levels:
        for module_name, level in module_levels.items():
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(getattr(logging, level.upper()))

    # Suppress verbose third-party library logs
    # PIL/Pillow logs debug messages for every image chunk read
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

    return root_logger


def get_log_file_path(
    output_base_dir: Union[str, Path],
    log_subdir: str = "logs",
    timezone: Optional[str] = None,
) -> Path:
    """Generate timestamped log file path.

    Args:
        output_base_dir: Base output directory from config
        log_subdir: Subdirectory for logs
        timezone: Timezone for timestamp ('UTC', 'local', or IANA timezone like 'Asia/Tokyo')

    Returns:
        Path to log file with timestamp

    Example:
        >>> path = get_log_file_path("outputs/classifier-experiment", "logs", "UTC")
        >>> print(path)
        outputs/classifier-experiment/logs/log_20260216_143022.log
    """
    output_base_dir = Path(output_base_dir)

    # Get current time in the specified timezone
    if timezone is None or timezone.lower() == "local":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    elif timezone.upper() == "UTC":
        timestamp = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = datetime.now(ZoneInfo(timezone)).strftime("%Y%m%d_%H%M%S")

    log_filename = f"log_{timestamp}.log"
    return output_base_dir / log_subdir / log_filename


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.debug("This is a debug message")
    """
    return logging.getLogger(name)
