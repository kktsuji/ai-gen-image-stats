"""Unit tests for Slack webhook notification utilities.

Tests cover:
- _post_slack() function
- _get_webhook_url() function
- _format_duration() function
- notify_success() function
- notify_error() function
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.utils.notification import (
    _format_duration,
    _get_webhook_url,
    _post_slack,
    notify_error,
    notify_success,
)


class TestPostSlack:
    """Tests for _post_slack() helper function."""

    def test_post_slack_sends_request(self):
        """Test that _post_slack sends a POST request with correct payload."""
        with patch("requests.post") as mock_post:
            _post_slack("Hello", "https://hooks.slack.com/test")

            mock_post.assert_called_once_with(
                "https://hooks.slack.com/test",
                data=json.dumps({"text": "Hello"}),
                timeout=5.0,
            )


class TestGetWebhookUrl:
    """Tests for _get_webhook_url() function."""

    def test_get_webhook_url_returns_url(self, monkeypatch):
        """Test that URL is returned when SLACK_WEBHOOK_URL is set."""
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        assert _get_webhook_url() == "https://hooks.slack.com/test"

    def test_get_webhook_url_returns_none_when_empty(self, monkeypatch):
        """Test that None is returned when SLACK_WEBHOOK_URL is empty."""
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "")
        assert _get_webhook_url() is None

    def test_get_webhook_url_returns_none_when_unset(self, monkeypatch):
        """Test that None is returned when SLACK_WEBHOOK_URL is not set."""
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
        assert _get_webhook_url() is None

    def test_get_webhook_url_strips_whitespace(self, monkeypatch):
        """Test that whitespace-only URL returns None."""
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "   ")
        assert _get_webhook_url() is None


class TestFormatDuration:
    """Tests for _format_duration() function."""

    def test_format_duration_seconds(self):
        """Test formatting of sub-minute durations."""
        assert _format_duration(45.2) == "45.2s"

    def test_format_duration_zero(self):
        """Test formatting of zero duration."""
        assert _format_duration(0.0) == "0.0s"

    def test_format_duration_minutes(self):
        """Test formatting of minute-range durations."""
        assert _format_duration(125) == "2m 5s"

    def test_format_duration_hours(self):
        """Test formatting of hour-range durations."""
        assert _format_duration(3725) == "1h 2m 5s"

    def test_format_duration_exact_minute(self):
        """Test formatting of exactly one minute."""
        assert _format_duration(60) == "1m 0s"


class TestNotifySuccess:
    """Tests for notify_success() function."""

    def test_notify_success_sends_message(self, monkeypatch):
        """Test that success notification sends a Slack message."""
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")

        config = {
            "experiment": "diffusion",
            "mode": "train",
            "output": {"base_dir": "outputs/test"},
        }

        with patch("src.utils.notification._post_slack") as mock_post:
            notify_success(config, 125.0)

            mock_post.assert_called_once()
            text = mock_post.call_args[0][0]
            assert "completed successfully" in text
            assert "diffusion" in text
            assert "outputs/test" in text
            assert "2m 5s" in text

    def test_notify_success_skips_when_no_url(self, monkeypatch):
        """Test that notification is skipped when no webhook URL is set."""
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)

        with patch("src.utils.notification._post_slack") as mock_post:
            notify_success({"experiment": "test"}, 10.0)
            mock_post.assert_not_called()

    def test_notify_success_includes_mode(self, monkeypatch):
        """Test that mode is included in the message when present."""
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")

        config = {
            "experiment": "diffusion",
            "mode": "generate",
            "output": {"base_dir": "outputs/test"},
        }

        with patch("src.utils.notification._post_slack") as mock_post:
            notify_success(config, 10.0)

            text = mock_post.call_args[0][0]
            assert "generate" in text

    def test_notify_success_omits_mode_when_absent(self, monkeypatch):
        """Test that mode line is omitted when mode is not in config."""
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")

        config = {
            "experiment": "data_preparation",
            "output": {"base_dir": "outputs/test"},
        }

        with patch("src.utils.notification._post_slack") as mock_post:
            notify_success(config, 10.0)

            text = mock_post.call_args[0][0]
            assert "Mode" not in text


class TestNotifyError:
    """Tests for notify_error() function."""

    def test_notify_error_sends_message(self, monkeypatch):
        """Test that error notification sends a Slack message."""
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")

        config = {
            "experiment": "classifier",
            "mode": "train",
            "output": {"base_dir": "outputs/test"},
        }
        error = RuntimeError("CUDA out of memory")

        with patch("src.utils.notification._post_slack") as mock_post:
            notify_error(config, error)

            mock_post.assert_called_once()
            text = mock_post.call_args[0][0]
            assert "failed" in text
            assert "classifier" in text
            assert "CUDA out of memory" in text

    def test_notify_error_truncates_long_message(self, monkeypatch):
        """Test that long error messages are truncated to 200 chars."""
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")

        config = {"experiment": "test", "output": {"base_dir": "outputs"}}
        long_msg = "x" * 300
        error = RuntimeError(long_msg)

        with patch("src.utils.notification._post_slack") as mock_post:
            notify_error(config, error)

            text = mock_post.call_args[0][0]
            # The error portion should be truncated to 200 chars
            assert "x" * 200 in text
            assert "x" * 201 not in text

    def test_notify_error_skips_when_no_url(self, monkeypatch):
        """Test that error notification is skipped when no webhook URL."""
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)

        with patch("src.utils.notification._post_slack") as mock_post:
            notify_error({"experiment": "test"}, RuntimeError("fail"))
            mock_post.assert_not_called()


class TestNotifyDoesNotRaise:
    """Tests that notification failures never propagate."""

    def test_notify_success_does_not_raise_on_request_failure(self, monkeypatch):
        """Test that _post_slack exceptions are caught in notify_success."""
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")

        config = {"experiment": "test", "output": {"base_dir": "outputs"}}

        with patch(
            "src.utils.notification._post_slack",
            side_effect=ConnectionError("network error"),
        ):
            # Should not raise
            notify_success(config, 10.0)

    def test_notify_error_does_not_raise_on_request_failure(self, monkeypatch):
        """Test that _post_slack exceptions are caught in notify_error."""
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")

        config = {"experiment": "test", "output": {"base_dir": "outputs"}}

        with patch(
            "src.utils.notification._post_slack",
            side_effect=ConnectionError("network error"),
        ):
            # Should not raise
            notify_error(config, RuntimeError("fail"))
