"""Git utilities for retrieving repository information.

This module provides functions to get Git metadata like commit hash and repository URL.
These are useful for experiment reproducibility and tracking.
"""

import subprocess
from typing import Dict, Optional


def get_git_commit_hash() -> Optional[str]:
    """Get the current Git commit hash.

    Returns:
        The short commit hash (7 characters) or None if not in a Git repository
        or if Git is not available.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return None


def get_git_repository_url() -> Optional[str]:
    """Get the Git repository URL.

    Returns:
        The remote origin URL or None if not in a Git repository,
        if Git is not available, or if no remote origin is configured.
    """
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return None


def get_git_info() -> Dict[str, Optional[str]]:
    """Get Git repository information.

    Returns:
        Dictionary containing:
        - commit_hash: Current commit hash (short)
        - repository_url: Remote origin URL
        Values are None if not available.
    """
    return {
        "commit_hash": get_git_commit_hash(),
        "repository_url": get_git_repository_url(),
    }
