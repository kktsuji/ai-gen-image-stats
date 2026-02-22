# Slack Webhook Notification on Task Completion / Error

## Overview

Add a feature to notify a Slack channel via an incoming webhook when an experiment task completes successfully or fails with an error. This provides real-time awareness of long-running training jobs without needing to watch the terminal.

**Objective:** Post a Slack message containing key experiment info (experiment type, mode, output directory, duration or error cause) at the end of each experiment run in `src/main.py`.

**Design:**

- A new utility module `src/utils/notification.py` encapsulates all webhook logic.
- The Slack webhook URL is loaded from a `.env` file via `python-dotenv` (`load_dotenv()`). Both `requests` and `python-dotenv` are already listed in `requirements.txt`.
- The environment variable name is `SLACK_WEBHOOK_URL`.
- If the webhook URL is not set, notifications are silently skipped (no error).
- Notifications are fire-and-forget with a short timeout (5 s). A failure to send a notification must **never** cause the experiment itself to fail.
- The `main()` function in `src/main.py` is updated to:
  1. Call `load_dotenv()` at startup.
  2. Record `start_time` before dispatching to the experiment.
  3. On success: send a notification with experiment type, mode (if present), `output.base_dir`, and elapsed duration.
  4. On error: send a notification with experiment type and a short error description, then re-raise / exit as before.

**File changes:**

| File                               | Change                                                                 |
| ---------------------------------- | ---------------------------------------------------------------------- |
| `src/utils/notification.py`        | **New** — `notify_success()`, `notify_error()`, helper `_post_slack()` |
| `src/main.py`                      | Add `load_dotenv()`, timing, and notification calls in `main()`        |
| `.env.example`                     | **New** — template with `SLACK_WEBHOOK_URL=`                           |
| `tests/utils/test_notification.py` | **New** — unit tests for notification module                           |
| `tests/test_main.py`               | Add tests for notification integration in `main()`                     |

**Time estimate:** ~2–3 hours

## Implementation Checklist

- [ ] Phase 1: Create notification utility module
  - [ ] Task 1.1: Create `src/utils/notification.py` with `_post_slack()`, `notify_success()`, `notify_error()`
  - [ ] Task 1.2: Create `.env.example` with `SLACK_WEBHOOK_URL=`
- [ ] Phase 2: Integrate notifications into `main()`
  - [ ] Task 2.1: Add `load_dotenv()` call at the top of `main()`
  - [ ] Task 2.2: Record `start_time` before experiment dispatch
  - [ ] Task 2.3: Call `notify_success()` on successful completion
  - [ ] Task 2.4: Call `notify_error()` on exception, before `sys.exit(1)`
- [ ] Phase 3: Tests
  - [ ] Task 3.1: Unit tests for `src/utils/notification.py` (`tests/utils/test_notification.py`)
  - [ ] Task 3.2: Integration tests for notification calls in `main()` (`tests/test_main.py`)
  - [ ] Task 3.3: Run full test suite and confirm no regressions
- [ ] Phase 4: Documentation
  - [ ] Task 4.1: Update `docs/standards/architecture.md` — add `notification.py` to the utils listing
  - [ ] Task 4.2: Update `README.md` — mention `.env` setup for Slack notifications

## Phase Details

### Phase 1: Create notification utility module

#### Task 1.1: Create `src/utils/notification.py`

Create a new module with three functions:

```python
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
```

Key design decisions:

- `_post_slack()` uses a lazy `import requests` to match the user's preferred style and avoid import overhead when notifications are disabled.
- `_get_webhook_url()` reads from `os.environ` (populated by `load_dotenv()` in `main()`).
- `_format_duration()` produces human-friendly strings (e.g., `1h 23m 45s`).
- All notification functions catch and log exceptions internally — they never propagate errors upward.

#### Task 1.2: Create `.env.example`

```
# Slack incoming webhook URL for experiment notifications
# Obtain from: https://api.slack.com/messaging/webhooks
SLACK_WEBHOOK_URL=
```

Verify that `.env` is already in `.gitignore` (it is, at line 127).

---

### Phase 2: Integrate notifications into `main()`

#### Task 2.1–2.4: Modify `src/main.py`

Add to the imports section:

```python
import time
from dotenv import load_dotenv
from src.utils.notification import notify_success, notify_error
```

Modify the `main()` function body:

```python
def main(args: Optional[list] = None) -> None:
    # Load environment variables from .env file
    load_dotenv()

    # Parse arguments and load configuration
    config = parse_args(args)

    # Validate basic config structure
    validate_cli_config(config)

    # Get experiment type
    experiment = config["experiment"]

    # Record start time for duration tracking
    start_time = time.time()

    # Dispatch to experiment
    try:
        if experiment == "classifier":
            setup_experiment_classifier(config)
        elif experiment == "diffusion":
            setup_experiment_diffusion(config)
        elif experiment == "gan":
            setup_experiment_gan(config)
        elif experiment == "data_preparation":
            setup_experiment_data_preparation(config)
        else:
            raise ValueError(
                f"Unknown experiment type: {experiment}. "
                f"Supported experiments: classifier, diffusion, gan, data_preparation"
            )

        # Notify on success
        duration = time.time() - start_time
        notify_success(config, duration)

    except NotImplementedError as e:
        logger.error(f"Experiment '{experiment}' is not yet implemented: {e}")
        notify_error(config, e)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Experiment '{experiment}' failed with error: {e}")
        notify_error(config, e)
        sys.exit(1)
```

Note: `KeyboardInterrupt` is not caught here (it's caught inside each `setup_experiment_*` function), so Ctrl+C will not trigger a notification. This is intentional — user-initiated interrupts are not errors.

---

### Phase 3: Tests

#### Task 3.1: Unit tests — `tests/utils/test_notification.py`

Test cases:

1. **`test_post_slack_sends_request`** — Mock `requests.post`, verify it is called with correct URL, JSON payload, and timeout.
2. **`test_get_webhook_url_returns_url`** — Set `SLACK_WEBHOOK_URL` env var, verify it is returned.
3. **`test_get_webhook_url_returns_none_when_empty`** — Set env var to `""`, verify `None` returned.
4. **`test_get_webhook_url_returns_none_when_unset`** — Unset env var, verify `None` returned.
5. **`test_format_duration_seconds`** — e.g., `45.2` → `"45.2s"`.
6. **`test_format_duration_minutes`** — e.g., `125` → `"2m 5s"`.
7. **`test_format_duration_hours`** — e.g., `3725` → `"1h 2m 5s"`.
8. **`test_notify_success_sends_message`** — Mock `_post_slack`, verify message contents.
9. **`test_notify_success_skips_when_no_url`** — Verify `_post_slack` is not called.
10. **`test_notify_success_includes_mode`** — Verify mode is in message when present.
11. **`test_notify_error_sends_message`** — Mock `_post_slack`, verify error message contents.
12. **`test_notify_error_truncates_long_message`** — Error message > 200 chars is truncated.
13. **`test_notify_error_skips_when_no_url`** — Verify `_post_slack` is not called.
14. **`test_notify_does_not_raise_on_request_failure`** — Mock `_post_slack` to raise, verify no exception propagates.

#### Task 3.2: Integration tests — `tests/test_main.py`

Add test cases:

1. **`test_main_calls_notify_success_on_completion`** — Mock the experiment setup function and `notify_success`, assert it's called with config and duration > 0.
2. **`test_main_calls_notify_error_on_exception`** — Mock experiment to raise, assert `notify_error` is called.
3. **`test_main_loads_dotenv`** — Mock `load_dotenv`, assert it's called.

#### Task 3.3: Run full test suite

```bash
python -m pytest tests/ -v
```

Verify all existing and new tests pass with no regressions.

---

### Phase 4: Documentation

#### Task 4.1: Update `docs/standards/architecture.md`

Add `notification.py` to the utils directory listing:

```
│   ├── utils/
│   │   ├── ...
│   │   ├── logging.py                   # Logging configuration and setup
│   │   ├── metrics.py                   # Common metrics (FID, IS, PR-AUC, ROC-AUC)
│   │   ├── notification.py              # Slack webhook notifications
│   │   └── tensorboard.py               # TensorBoard utility functions (optional)
```

#### Task 4.2: Update `README.md`

Add a short section under setup or features:

```markdown
### Slack Notifications (Optional)

To receive Slack notifications when experiments complete or fail:

1. Create a Slack incoming webhook: https://api.slack.com/messaging/webhooks
2. Copy `.env.example` to `.env` and set `SLACK_WEBHOOK_URL`
3. Notifications are sent automatically — no config changes needed
```
