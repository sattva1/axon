"""Shared fixtures for the MCP test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from axon.mcp.repo_context import RepoContext


@pytest.fixture()
def make_ctx(tmp_path: Path):
    """Factory: build a RepoContext for a given storage backend.

    Defaults: slug='test-repo', is_local=True, repo_path=tmp_path,
    local_slug='test-repo'. Overrides accepted as kwargs.

    """
    def _make(storage: Any, **overrides: Any) -> RepoContext:
        defaults: dict[str, Any] = dict(
            storage=storage,
            slug='test-repo',
            is_local=True,
            repo_path=overrides.pop('repo_path', tmp_path),
            local_slug='test-repo',
        )
        defaults.update(overrides)

        return RepoContext(**defaults)
    return _make
