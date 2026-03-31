"""Git status detection for execution tracking.

This module provides utilities to capture the Git state at execution time,
enabling reproducibility and version tracking for any project run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypedDict


class GitInfo(TypedDict, total=False):
    """Git status information for execution tracking.

    Attributes:
        version: Version string in format {sha} or {sha}+dirty
        commit_sha: Full commit SHA (40 characters)
        commit_sha_short: Short commit SHA (7 characters)
        branch: Current branch name (or 'HEAD' if detached)
        is_dirty: True if working directory has uncommitted changes
        is_git_repo: True if running inside a Git repository
    """
    version: str
    commit_sha: str
    commit_sha_short: str
    branch: str
    is_dirty: bool
    is_git_repo: bool


def get_git_info(repo_path: str | Path | None = None) -> GitInfo:
    """Capture current Git repository status for version tracking.

    This function generates a version string following PEP 440-inspired naming:
    - Clean state: `{commit_sha}`
    - Dirty state: `{commit_sha}+dirty`

    The dirty flag is set if:
    - Any tracked files are modified (staged or unstaged)
    - Any untracked files exist
    - Any files are staged for commit

    Args:
        repo_path: Path to the Git repository root. If None, uses current working directory.

    Returns:
        GitInfo dictionary containing:
        - version: Version string (e.g., "a1b2c3d" or "a1b2c3d+dirty")
        - commit_sha: Full 40-character commit hash
        - commit_sha_short: 7-character short hash
        - branch: Current branch name
        - is_dirty: Whether working directory has uncommitted changes
        - is_git_repo: Whether the path is inside a Git repository

    Example:
        >>> info = get_git_info()
        >>> print(info["version"])
        "a1b2c3d+dirty"
        >>> print(info["is_dirty"])
        True
    """
    cwd = Path(repo_path).resolve() if repo_path else Path.cwd()

    # Check if we're in a Git repository
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=cwd,
            capture_output=True,
            check=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return GitInfo(
            version="unknown",
            commit_sha="unknown",
            commit_sha_short="unknown",
            branch="unknown",
            is_dirty=False,
            is_git_repo=False,
        )

    # Get commit SHA (short and full)
    try:
        short_sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()

        full_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        # Repository exists but no commits yet
        return GitInfo(
            version="no-commits",
            commit_sha="no-commits",
            commit_sha_short="no-commits",
            branch="unknown",
            is_dirty=True,
            is_git_repo=True,
        )

    # Get current branch
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        branch = "HEAD"

    # Check for dirty state (any modifications, staged changes, or untracked files)
    try:
        status_output = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        is_dirty = len(status_output) > 0
    except subprocess.CalledProcessError:
        is_dirty = False

    # Generate version string
    version = f"{short_sha}+dirty" if is_dirty else short_sha

    return GitInfo(
        version=version,
        commit_sha=full_sha,
        commit_sha_short=short_sha,
        branch=branch,
        is_dirty=is_dirty,
        is_git_repo=True,
    )
