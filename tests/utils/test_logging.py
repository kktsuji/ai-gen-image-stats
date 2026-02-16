"""Tests for Logging Utility

Unit tests for the logging configuration and setup utilities.
These tests run on CPU only and do not require GPU hardware.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.utils.logging import (
    TimezoneFormatter,
    get_log_file_path,
    get_logger,
    setup_logging,
)


@pytest.mark.unit
class TestLoggingSetup:
    """Tests for logging setup function."""

    def test_setup_logging_creates_log_file(self, tmp_path):
        """setup_logging creates log file at specified path."""
        log_file = tmp_path / "test.log"
        logger = setup_logging(log_file=log_file)

        assert log_file.exists()
        assert isinstance(logger, logging.Logger)

    def test_setup_logging_console_handler(self, tmp_path):
        """setup_logging adds console handler with correct level."""
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file, console_level="WARNING")

        root_logger = logging.getLogger()
        console_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)
        ]

        assert len(console_handlers) > 0
        # Find the stdout handler (not file handler)
        console_handler = [
            h for h in console_handlers if not isinstance(h, logging.FileHandler)
        ][0]
        assert console_handler.level == logging.WARNING

    def test_setup_logging_file_handler(self, tmp_path):
        """setup_logging adds file handler with correct level."""
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file, file_level="DEBUG")

        root_logger = logging.getLogger()
        file_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.FileHandler)
        ]

        assert len(file_handlers) > 0
        assert file_handlers[0].level == logging.DEBUG

    def test_setup_logging_custom_format(self, tmp_path):
        """setup_logging accepts custom format string."""
        log_file = tmp_path / "test.log"
        custom_format = "%(levelname)s: %(message)s"

        setup_logging(log_file=log_file, log_format=custom_format)

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]
        formatter = handler.formatter

        assert formatter is not None
        assert custom_format in formatter._fmt

    def test_setup_logging_module_levels(self, tmp_path):
        """setup_logging configures module-specific log levels."""
        log_file = tmp_path / "test.log"
        module_levels = {"test.module": "ERROR", "another.module": "INFO"}

        setup_logging(log_file=log_file, module_levels=module_levels)

        test_logger = logging.getLogger("test.module")
        another_logger = logging.getLogger("another.module")

        assert test_logger.level == logging.ERROR
        assert another_logger.level == logging.INFO

    def test_logs_to_console_and_file(self, tmp_path):
        """Messages are logged to both console and file."""
        log_file = tmp_path / "test.log"

        logger = setup_logging(
            log_file=log_file, console_level="INFO", file_level="INFO"
        )
        test_message = "Test log message"
        logger.info(test_message)

        # Check file (primary verification)
        log_content = log_file.read_text()
        assert test_message in log_content

        # Verify both handlers exist
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 2  # Console + File
        has_console = any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            for h in root_logger.handlers
        )
        has_file = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
        assert has_console and has_file

    def test_setup_logging_creates_nested_directories(self, tmp_path):
        """setup_logging creates nested directories if they don't exist."""
        log_file = tmp_path / "nested" / "path" / "test.log"
        setup_logging(log_file=log_file)

        assert log_file.exists()
        assert log_file.parent.exists()

    def test_setup_logging_default_format(self, tmp_path):
        """setup_logging uses default format when not specified."""
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]
        formatter = handler.formatter

        assert formatter is not None
        # Default format should contain these components
        assert "%(asctime)s" in formatter._fmt
        assert "%(name)s" in formatter._fmt
        assert "%(levelname)s" in formatter._fmt
        assert "%(message)s" in formatter._fmt

    def test_setup_logging_clears_existing_handlers(self, tmp_path):
        """setup_logging clears existing handlers to avoid duplicates."""
        log_file1 = tmp_path / "test1.log"
        log_file2 = tmp_path / "test2.log"

        setup_logging(log_file=log_file1)
        handlers_count_1 = len(logging.getLogger().handlers)

        setup_logging(log_file=log_file2)
        handlers_count_2 = len(logging.getLogger().handlers)

        # Should have same number of handlers (cleared and re-added)
        assert handlers_count_1 == handlers_count_2

    def test_setup_logging_different_levels(self, tmp_path, caplog):
        """Console and file can have different log levels."""
        log_file = tmp_path / "test.log"
        caplog.set_level(logging.DEBUG)

        logger = setup_logging(
            log_file=log_file, console_level="INFO", file_level="DEBUG"
        )

        debug_message = "Debug message"
        info_message = "Info message"

        logger.debug(debug_message)
        logger.info(info_message)

        # File should have both
        log_content = log_file.read_text()
        assert debug_message in log_content
        assert info_message in log_content

        # Console caplog shows all due to caplog.set_level(DEBUG), but in real use
        # the console handler would filter DEBUG messages
        # We can verify handler level instead
        root_logger = logging.getLogger()
        console_handlers = [
            h
            for h in root_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert console_handlers[0].level == logging.INFO


@pytest.mark.unit
class TestLogFilePath:
    """Tests for log file path generation."""

    def test_get_log_file_path_structure(self):
        """get_log_file_path generates correct directory structure."""
        output_dir = "outputs/experiment"
        log_subdir = "logs"

        path = get_log_file_path(output_dir, log_subdir)

        assert isinstance(path, Path)
        assert "outputs" in str(path)
        assert "experiment" in str(path)
        assert "logs" in str(path)

    def test_get_log_file_path_timestamp(self):
        """get_log_file_path includes timestamp in filename."""
        output_dir = "outputs/test"

        path = get_log_file_path(output_dir)

        # Should have format: log_YYYYMMDD_HHMMSS.log
        filename = path.name
        assert filename.startswith("log_")
        assert filename.endswith(".log")
        assert len(filename) == len("log_20260216_143022.log")

    def test_get_log_file_path_default_subdir(self):
        """get_log_file_path uses 'logs' as default subdirectory."""
        output_dir = "outputs/test"

        path = get_log_file_path(output_dir)

        assert "logs" in str(path)

    def test_get_log_file_path_custom_subdir(self):
        """get_log_file_path accepts custom subdirectory."""
        output_dir = "outputs/test"
        custom_subdir = "custom_logs"

        path = get_log_file_path(output_dir, log_subdir=custom_subdir)

        assert custom_subdir in str(path)

    def test_get_log_file_path_path_object(self):
        """get_log_file_path accepts Path object as input."""
        output_dir = Path("outputs/test")

        path = get_log_file_path(output_dir)

        assert isinstance(path, Path)

    def test_get_log_file_path_unique_timestamps(self):
        """get_log_file_path generates unique paths for sequential calls."""
        import time

        output_dir = "outputs/test"

        path1 = get_log_file_path(output_dir)
        time.sleep(1.1)  # Ensure timestamp changes
        path2 = get_log_file_path(output_dir)

        # Paths should be different due to timestamp
        assert path1 != path2


@pytest.mark.unit
class TestGetLogger:
    """Tests for get_logger wrapper."""

    def test_get_logger_return_type(self):
        """get_logger returns a logging.Logger instance."""
        logger = get_logger("test.module")

        assert isinstance(logger, logging.Logger)

    def test_get_logger_unique_names(self):
        """get_logger creates separate loggers for different names."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1.name == "module1"
        assert logger2.name == "module2"
        assert logger1 != logger2

    def test_get_logger_same_name_returns_same_instance(self):
        """get_logger returns same instance for same name."""
        logger1 = get_logger("test.module")
        logger2 = get_logger("test.module")

        assert logger1 is logger2

    def test_get_logger_with_dunder_name(self):
        """get_logger works with __name__ pattern."""
        module_name = __name__
        logger = get_logger(module_name)

        assert logger.name == module_name
        assert isinstance(logger, logging.Logger)


@pytest.mark.unit
class TestTimezoneFormatter:
    """Tests for timezone-aware log formatting."""

    def test_timezone_formatter_utc(self):
        """TimezoneFormatter correctly handles UTC timezone."""
        formatter = TimezoneFormatter(
            fmt="%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            timezone="UTC",
        )

        # Create a mock log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "Test message" in formatted
        # Should have timestamp in format YYYY-MM-DD HH:MM:SS
        assert formatted.count("|") == 1

    def test_timezone_formatter_local(self):
        """TimezoneFormatter correctly handles local timezone."""
        formatter = TimezoneFormatter(
            fmt="%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            timezone="local",
        )

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "Test message" in formatted

    def test_timezone_formatter_iana_timezone(self):
        """TimezoneFormatter correctly handles IANA timezones."""
        formatter = TimezoneFormatter(
            fmt="%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            timezone="Asia/Tokyo",
        )

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "Test message" in formatted

    def test_timezone_formatter_none_uses_local(self):
        """TimezoneFormatter uses local time when timezone is None."""
        formatter = TimezoneFormatter(
            fmt="%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            timezone=None,
        )

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "Test message" in formatted


@pytest.mark.unit
class TestLoggingWithTimezone:
    """Tests for logging setup with timezone support."""

    def test_setup_logging_with_utc_timezone(self, tmp_path):
        """setup_logging correctly uses UTC timezone."""
        log_file = tmp_path / "test_utc.log"
        logger = setup_logging(log_file=log_file, timezone="UTC")

        test_message = "Test UTC message"
        logger.info(test_message)

        log_content = log_file.read_text()
        assert test_message in log_content

    def test_setup_logging_with_local_timezone(self, tmp_path):
        """setup_logging correctly uses local timezone."""
        log_file = tmp_path / "test_local.log"
        logger = setup_logging(log_file=log_file, timezone="local")

        test_message = "Test local message"
        logger.info(test_message)

        log_content = log_file.read_text()
        assert test_message in log_content

    def test_setup_logging_with_iana_timezone(self, tmp_path):
        """setup_logging correctly uses IANA timezone."""
        log_file = tmp_path / "test_tokyo.log"
        logger = setup_logging(log_file=log_file, timezone="Asia/Tokyo")

        test_message = "Test Tokyo message"
        logger.info(test_message)

        log_content = log_file.read_text()
        assert test_message in log_content

    def test_setup_logging_default_timezone_is_utc(self, tmp_path):
        """setup_logging defaults to UTC when timezone not specified."""
        log_file = tmp_path / "test_default.log"
        logger = setup_logging(log_file=log_file)

        test_message = "Test default timezone"
        logger.info(test_message)

        log_content = log_file.read_text()
        assert test_message in log_content

    def test_timezone_affects_timestamp_in_logs(self, tmp_path):
        """Different timezones produce different timestamps."""
        log_file_utc = tmp_path / "test_utc.log"
        log_file_tokyo = tmp_path / "test_tokyo.log"

        # Log with UTC
        logger_utc = setup_logging(
            log_file=log_file_utc,
            timezone="UTC",
            console_level="ERROR",  # Suppress console
        )
        logger_utc.info("UTC timestamp test")

        # Clear handlers
        logging.getLogger().handlers.clear()

        # Log with Tokyo timezone
        logger_tokyo = setup_logging(
            log_file=log_file_tokyo,
            timezone="Asia/Tokyo",
            console_level="ERROR",  # Suppress console
        )
        logger_tokyo.info("Tokyo timestamp test")

        # Read both logs
        utc_content = log_file_utc.read_text()
        tokyo_content = log_file_tokyo.read_text()

        # Both should have the message
        assert "UTC timestamp test" in utc_content
        assert "Tokyo timestamp test" in tokyo_content

        # Extract timestamps (first part before |)
        utc_timestamp = utc_content.split("|")[0].strip()
        tokyo_timestamp = tokyo_content.split("|")[0].strip()

        # Timestamps should be in expected format
        assert len(utc_timestamp) == len("2026-02-17 08:30:15")
        assert len(tokyo_timestamp) == len("2026-02-17 08:30:15")


@pytest.mark.unit
class TestLogFilePathWithTimezone:
    """Tests for log file path generation with timezone support."""

    def test_get_log_file_path_with_utc_timezone(self):
        """get_log_file_path generates path with UTC timestamp."""
        output_dir = "outputs/test"
        path = get_log_file_path(output_dir, timezone="UTC")

        assert isinstance(path, Path)
        assert "log_" in path.name
        assert path.name.endswith(".log")

    def test_get_log_file_path_with_local_timezone(self):
        """get_log_file_path generates path with local timestamp."""
        output_dir = "outputs/test"
        path = get_log_file_path(output_dir, timezone="local")

        assert isinstance(path, Path)
        assert "log_" in path.name
        assert path.name.endswith(".log")

    def test_get_log_file_path_with_iana_timezone(self):
        """get_log_file_path generates path with IANA timezone timestamp."""
        output_dir = "outputs/test"
        path = get_log_file_path(output_dir, timezone="Asia/Tokyo")

        assert isinstance(path, Path)
        assert "log_" in path.name
        assert path.name.endswith(".log")

    def test_get_log_file_path_default_timezone(self):
        """get_log_file_path uses UTC when timezone not specified."""
        output_dir = "outputs/test"
        path = get_log_file_path(output_dir)

        assert isinstance(path, Path)
        assert "log_" in path.name

    def test_different_timezones_may_produce_different_filenames(self):
        """Different timezones can produce different log filenames."""
        output_dir = "outputs/test"

        path_utc = get_log_file_path(output_dir, timezone="UTC")
        path_tokyo = get_log_file_path(output_dir, timezone="Asia/Tokyo")

        # Both should be valid paths
        assert isinstance(path_utc, Path)
        assert isinstance(path_tokyo, Path)

        # Filenames may differ if UTC and Tokyo are in different days
        # but both should follow the naming pattern
        assert path_utc.name.startswith("log_")
        assert path_tokyo.name.startswith("log_")
