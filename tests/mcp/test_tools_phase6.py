"""Tests for Phase 6 freshness-warning integration in MCP tool handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from axon.core.meta import update_meta
from axon.mcp.tools import (
    handle_communities,
    handle_dead_code,
    handle_review_risk,
)


@pytest.fixture()
def mock_storage() -> MagicMock:
    """Minimal storage mock sufficient for handler calls under test."""
    storage = MagicMock()
    storage.execute_raw.return_value = []
    storage.iter_nodes.return_value = iter([])
    storage.get_process_memberships.return_value = {}
    return storage


@pytest.fixture()
def stale_repo(tmp_path: Path) -> Path:
    """Repo with stale dead_code (120s lag) and stale communities (120s lag)."""
    update_meta(
        tmp_path,
        last_incremental_at='2025-01-01T00:02:00+00:00',
        dead_code_last_refreshed_at='2025-01-01T00:00:00+00:00',
        communities_last_refreshed_at='2025-01-01T00:00:00+00:00',
    )
    return tmp_path


@pytest.fixture()
def fresh_repo(tmp_path: Path) -> Path:
    """Repo where all analyses are current (no lag)."""
    ts = '2025-01-01T00:02:00+00:00'
    update_meta(
        tmp_path,
        last_incremental_at=ts,
        dead_code_last_refreshed_at=ts,
        communities_last_refreshed_at=ts,
    )
    return tmp_path


class TestHandleDeadCode:
    def test_no_repo_path_no_warning(
        self, mock_storage: MagicMock, make_ctx: Any
    ) -> None:
        """handle_dead_code without repo_path never prefixes a warning."""
        result = handle_dead_code(make_ctx(mock_storage, repo_path=None))
        assert 'axon analyze' not in result

    def test_stale_repo_warning_present(
        self, mock_storage: MagicMock, stale_repo: Path, make_ctx: Any
    ) -> None:
        """Stale dead-code analysis causes warning to be prepended."""
        result = handle_dead_code(make_ctx(mock_storage, repo_path=stale_repo))
        assert 'axon analyze' in result

    def test_fresh_repo_no_warning(
        self, mock_storage: MagicMock, fresh_repo: Path, make_ctx: Any
    ) -> None:
        """Fresh meta produces no warning prefix."""
        result = handle_dead_code(make_ctx(mock_storage, repo_path=fresh_repo))
        assert 'axon analyze' not in result

    def test_missing_meta_no_warning(
        self, mock_storage: MagicMock, tmp_path: Path, make_ctx: Any
    ) -> None:
        """Missing meta.json (load_meta returns defaults) produces no warning."""
        result = handle_dead_code(make_ctx(mock_storage, repo_path=tmp_path))
        assert 'axon analyze' not in result


class TestHandleCommunities:
    def test_no_repo_path_no_warning(
        self, mock_storage: MagicMock, make_ctx: Any
    ) -> None:
        """handle_communities without repo_path never prefixes a warning."""
        result = handle_communities(make_ctx(mock_storage, repo_path=None))
        assert 'axon analyze' not in result

    def test_stale_repo_warning_present(
        self, mock_storage: MagicMock, stale_repo: Path, make_ctx: Any
    ) -> None:
        """Stale communities analysis causes warning to be prepended."""
        result = handle_communities(
            make_ctx(mock_storage, repo_path=stale_repo)
        )
        assert 'axon analyze' in result

    def test_fresh_repo_no_warning(
        self, mock_storage: MagicMock, fresh_repo: Path, make_ctx: Any
    ) -> None:
        """Fresh communities produce no warning."""
        result = handle_communities(
            make_ctx(mock_storage, repo_path=fresh_repo)
        )
        assert 'axon analyze' not in result

    def test_community_arg_honored_with_stale_repo(
        self, mock_storage: MagicMock, stale_repo: Path, make_ctx: Any
    ) -> None:
        """community and repo_path kwargs are both respected."""
        result = handle_communities(
            make_ctx(mock_storage, repo_path=stale_repo),
            community='my-community',
        )
        assert 'axon analyze' in result

    def test_uses_communities_field_not_dead_code(
        self, mock_storage: MagicMock, tmp_path: Path, make_ctx: Any
    ) -> None:
        """communities warning uses communities_last_refreshed_at, not dead_code field."""
        update_meta(
            tmp_path,
            last_incremental_at='2025-01-01T00:02:00+00:00',
            # dead_code is fresh; communities is stale
            dead_code_last_refreshed_at='2025-01-01T00:02:00+00:00',
            communities_last_refreshed_at='2025-01-01T00:00:00+00:00',
        )
        communities_result = handle_communities(
            make_ctx(mock_storage, repo_path=tmp_path)
        )
        dead_code_result = handle_dead_code(
            make_ctx(mock_storage, repo_path=tmp_path)
        )

        assert 'axon analyze' in communities_result
        assert 'axon analyze' not in dead_code_result


class TestHandleReviewRisk:
    def test_no_repo_path_no_warning(
        self, mock_storage: MagicMock, make_ctx: Any
    ) -> None:
        """handle_review_risk without repo_path never prefixes a warning."""
        result = handle_review_risk(make_ctx(mock_storage, repo_path=None), '')
        assert 'axon analyze' not in result

    def test_stale_repo_warning_present(
        self, mock_storage: MagicMock, stale_repo: Path, make_ctx: Any
    ) -> None:
        """Stale dead-code analysis in review_risk causes warning prefix."""
        result = handle_review_risk(
            make_ctx(mock_storage, repo_path=stale_repo), ''
        )
        assert 'axon analyze' in result

    def test_fresh_repo_no_warning(
        self, mock_storage: MagicMock, fresh_repo: Path, make_ctx: Any
    ) -> None:
        """Fresh repo produces no warning in review_risk."""
        result = handle_review_risk(
            make_ctx(mock_storage, repo_path=fresh_repo), ''
        )
        assert 'axon analyze' not in result

    def test_uses_dead_code_field_not_communities(
        self, mock_storage: MagicMock, tmp_path: Path, make_ctx: Any
    ) -> None:
        """review_risk warning uses dead_code_last_refreshed_at, not communities."""
        update_meta(
            tmp_path,
            last_incremental_at='2025-01-01T00:02:00+00:00',
            # dead_code is stale; communities is fresh
            dead_code_last_refreshed_at='2025-01-01T00:00:00+00:00',
            communities_last_refreshed_at='2025-01-01T00:02:00+00:00',
        )
        review_result = handle_review_risk(
            make_ctx(mock_storage, repo_path=tmp_path), ''
        )
        communities_result = handle_communities(
            make_ctx(mock_storage, repo_path=tmp_path)
        )

        assert 'axon analyze' in review_result
        assert 'axon analyze' not in communities_result


class TestDispatchToolRepoPath:
    """Verify repo_path flows from RepoContext into freshness-sensitive handlers.

    The Phase 3 refactor changed _dispatch_tool to accept ctx: RepoContext.
    These tests build a RepoContext via make_ctx and call _dispatch_tool directly
    to confirm the staleness warning reaches the output.
    """

    def test_repo_path_flows_to_dead_code(
        self, mock_storage: MagicMock, stale_repo: Path, make_ctx: Any
    ) -> None:
        """handle_dead_code receives repo_path via ctx."""
        from axon.mcp.server import _dispatch_tool

        result = _dispatch_tool(
            'axon_dead_code', {}, make_ctx(mock_storage, repo_path=stale_repo)
        )
        assert 'axon analyze' in result

    def test_no_repo_path_no_warning_via_dispatch(
        self, mock_storage: MagicMock, make_ctx: Any
    ) -> None:
        """_dispatch_tool with repo_path=None produces no warning for dead_code."""
        from axon.mcp.server import _dispatch_tool

        result = _dispatch_tool(
            'axon_dead_code', {}, make_ctx(mock_storage, repo_path=None)
        )
        assert 'axon analyze' not in result

    def test_repo_path_flows_to_communities(
        self, mock_storage: MagicMock, stale_repo: Path, make_ctx: Any
    ) -> None:
        """handle_communities receives repo_path via ctx."""
        from axon.mcp.server import _dispatch_tool

        result = _dispatch_tool(
            'axon_communities',
            {},
            make_ctx(mock_storage, repo_path=stale_repo),
        )
        assert 'axon analyze' in result

    def test_repo_path_flows_to_review_risk(
        self, mock_storage: MagicMock, stale_repo: Path, make_ctx: Any
    ) -> None:
        """handle_review_risk receives repo_path via ctx."""
        from axon.mcp.server import _dispatch_tool

        result = _dispatch_tool(
            'axon_review_risk',
            {'diff': ''},
            make_ctx(mock_storage, repo_path=stale_repo),
        )
        assert 'axon analyze' in result
