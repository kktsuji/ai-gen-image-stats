"""Slack webhook notification utilities.

Sends notifications to Slack when experiments complete or fail.
Requires SLACK_WEBHOOK_URL environment variable to be set (via .env file).
If the variable is unset or empty, notifications are silently skipped.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _post_slack(text: str, webhook_url: str) -> None:
    """Post a message to Slack via incoming webhook.

    Args:
        text: Message text to send.
        webhook_url: Slack incoming webhook URL.
    """
    import requests

    message = {"text": text}
    requests.post(webhook_url, data=json.dumps(message), timeout=5.0)


def _get_webhook_url() -> Optional[str]:
    """Return the Slack webhook URL from environment, or None if unset."""
    url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    return url if url else None


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like '1h 23m 45s' or '45.2s'.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def notify_success(config: Dict[str, Any], duration_seconds: float) -> None:
    """Send a Slack notification for successful experiment completion.

    Args:
        config: Experiment configuration dictionary.
        duration_seconds: Total elapsed time in seconds.
    """
    webhook_url = _get_webhook_url()
    if not webhook_url:
        return

    experiment = config.get("experiment", "unknown")
    mode = config.get("mode")
    base_dir = config.get("output", {}).get("base_dir", "N/A")
    duration_str = _format_duration(duration_seconds)

    parts = [f":white_check_mark: *Experiment completed successfully*"]
    parts.append(f"• Experiment: `{experiment}`")
    if mode:
        parts.append(f"• Mode: `{mode}`")
    parts.append(f"• Output: `{base_dir}`")
    parts.append(f"• Duration: {duration_str}")

    text = "\n".join(parts)

    try:
        _post_slack(text, webhook_url)
        logger.debug("Slack notification sent (success)")
    except Exception as e:
        logger.warning(f"Failed to send Slack notification: {e}")


def notify_error(config: Dict[str, Any], error: Exception) -> None:
    """Send a Slack notification when an experiment fails.

    Args:
        config: Experiment configuration dictionary.
        error: The exception that caused the failure.
    """
    webhook_url = _get_webhook_url()
    if not webhook_url:
        return

    experiment = config.get("experiment", "unknown")
    mode = config.get("mode")
    base_dir = config.get("output", {}).get("base_dir", "N/A")
    error_msg = str(error)[:200]  # Truncate long errors

    parts = [f":x: *Experiment failed*"]
    parts.append(f"• Experiment: `{experiment}`")
    if mode:
        parts.append(f"• Mode: `{mode}`")
    parts.append(f"• Output: `{base_dir}`")
    parts.append(f"• Error: {error_msg}")

    text = "\n".join(parts)

    try:
        _post_slack(text, webhook_url)
        logger.debug("Slack notification sent (error)")
    except Exception as e:
        logger.warning(f"Failed to send Slack notification: {e}")
