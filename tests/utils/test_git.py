"""
Unit tests for Git utilities.

These tests verify Git repository information retrieval functionality,
including commit hash and repository URL extraction.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.utils.git import get_git_commit_hash, get_git_info, get_git_repository_url


@pytest.mark.unit
class TestGitCommitHash:
    """Tests for get_git_commit_hash function."""

    def test_get_commit_hash_success(self):
        """Test successful retrieval of commit hash."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = "abc1234\n"
            mock_run.return_value = mock_result

            result = get_git_commit_hash()

            assert result == "abc1234"
            mock_run.assert_called_once_with(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )

    def test_get_commit_hash_with_whitespace(self):
        """Test commit hash retrieval strips whitespace."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = "  def5678  \n"
            mock_run.return_value = mock_result

            result = get_git_commit_hash()

            assert result == "def5678"

    def test_get_commit_hash_not_git_repo(self):
        """Test behavior when not in a Git repository."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=128, cmd="git"
            )

            result = get_git_commit_hash()

            assert result is None

    def test_get_commit_hash_git_not_installed(self):
        """Test behavior when Git is not installed."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError

            result = get_git_commit_hash()

            assert result is None

    def test_get_commit_hash_timeout(self):
        """Test behavior when Git command times out."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=5)

            result = get_git_commit_hash()

            assert result is None

    def test_get_commit_hash_empty_output(self):
        """Test behavior with empty output."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = ""
            mock_run.return_value = mock_result

            result = get_git_commit_hash()

            assert result == ""


@pytest.mark.unit
class TestGitRepositoryUrl:
    """Tests for get_git_repository_url function."""

    def test_get_repository_url_success_https(self):
        """Test successful retrieval of HTTPS repository URL."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = "https://github.com/user/repo.git\n"
            mock_run.return_value = mock_result

            result = get_git_repository_url()

            assert result == "https://github.com/user/repo.git"
            mock_run.assert_called_once_with(
                ["git", "config", "--get", "remote.origin.url"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )

    def test_get_repository_url_success_ssh(self):
        """Test successful retrieval of SSH repository URL."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = "git@github.com:user/repo.git\n"
            mock_run.return_value = mock_result

            result = get_git_repository_url()

            assert result == "git@github.com:user/repo.git"

    def test_get_repository_url_with_whitespace(self):
        """Test repository URL retrieval strips whitespace."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = "  git@github.com:user/repo.git  \n"
            mock_run.return_value = mock_result

            result = get_git_repository_url()

            assert result == "git@github.com:user/repo.git"

    def test_get_repository_url_no_remote(self):
        """Test behavior when no remote origin is configured."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1, cmd="git"
            )

            result = get_git_repository_url()

            assert result is None

    def test_get_repository_url_not_git_repo(self):
        """Test behavior when not in a Git repository."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=128, cmd="git"
            )

            result = get_git_repository_url()

            assert result is None

    def test_get_repository_url_git_not_installed(self):
        """Test behavior when Git is not installed."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError

            result = get_git_repository_url()

            assert result is None

    def test_get_repository_url_timeout(self):
        """Test behavior when Git command times out."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=5)

            result = get_git_repository_url()

            assert result is None


@pytest.mark.unit
class TestGitInfo:
    """Tests for get_git_info function."""

    def test_get_git_info_success(self):
        """Test successful retrieval of all Git information."""
        with (
            patch("src.utils.git.get_git_commit_hash") as mock_hash,
            patch("src.utils.git.get_git_repository_url") as mock_url,
        ):
            mock_hash.return_value = "abc1234"
            mock_url.return_value = "git@github.com:user/repo.git"

            result = get_git_info()

            assert result == {
                "commit_hash": "abc1234",
                "repository_url": "git@github.com:user/repo.git",
            }

    def test_get_git_info_partial_success(self):
        """Test retrieval when only commit hash is available."""
        with (
            patch("src.utils.git.get_git_commit_hash") as mock_hash,
            patch("src.utils.git.get_git_repository_url") as mock_url,
        ):
            mock_hash.return_value = "abc1234"
            mock_url.return_value = None

            result = get_git_info()

            assert result == {
                "commit_hash": "abc1234",
                "repository_url": None,
            }

    def test_get_git_info_no_git(self):
        """Test behavior when Git is not available."""
        with (
            patch("src.utils.git.get_git_commit_hash") as mock_hash,
            patch("src.utils.git.get_git_repository_url") as mock_url,
        ):
            mock_hash.return_value = None
            mock_url.return_value = None

            result = get_git_info()

            assert result == {
                "commit_hash": None,
                "repository_url": None,
            }

    def test_get_git_info_returns_dict(self):
        """Test that get_git_info always returns a dictionary."""
        with (
            patch("src.utils.git.get_git_commit_hash") as mock_hash,
            patch("src.utils.git.get_git_repository_url") as mock_url,
        ):
            mock_hash.return_value = None
            mock_url.return_value = None

            result = get_git_info()

            assert isinstance(result, dict)
            assert "commit_hash" in result
            assert "repository_url" in result


@pytest.mark.unit
class TestGitIntegration:
    """Integration tests using actual Git commands (if available)."""

    def test_actual_git_commands_if_available(self):
        """Test with actual Git commands if in a repository."""
        # This test will naturally pass/fail based on actual Git status
        # It's useful for verifying the real integration works
        commit_hash = get_git_commit_hash()
        repo_url = get_git_repository_url()
        git_info = get_git_info()

        # If we're in a Git repo, these should be strings
        # If not, they should be None
        if commit_hash is not None:
            assert isinstance(commit_hash, str)
            assert len(commit_hash) > 0

        if repo_url is not None:
            assert isinstance(repo_url, str)
            assert len(repo_url) > 0

        # Info dict should always exist
        assert isinstance(git_info, dict)
        assert git_info["commit_hash"] == commit_hash
        assert git_info["repository_url"] == repo_url

    def test_commit_hash_format(self):
        """Test commit hash format if available."""
        commit_hash = get_git_commit_hash()

        if commit_hash is not None:
            # Short hash should be 7 characters (default for git rev-parse --short)
            assert len(commit_hash) == 7
            # Should be hexadecimal
            assert all(c in "0123456789abcdef" for c in commit_hash.lower())

    def test_repository_url_format(self):
        """Test repository URL format if available."""
        repo_url = get_git_repository_url()

        if repo_url is not None:
            # Should be either HTTPS or SSH format
            assert (
                repo_url.startswith("https://")
                or repo_url.startswith("http://")
                or repo_url.startswith("git@")
                or repo_url.startswith("/")  # Local path
            )
