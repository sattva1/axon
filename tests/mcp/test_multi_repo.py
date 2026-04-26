"""Tests for multi-repo MCP dispatch layer (Phase 3, plan section 3.8).

Coverage:
- _ensure_multi_repo init flow and thread-safety
- _build_repo_context routing (repo arg, path, diff, STALE_MAJOR refusal)
- route_for_path and route_for_diff helpers
- Fan-out helpers: _foreign_symbol_matches, _foreign_query_hit_counts
- handle_context and handle_query with foreign repo footers
- handle_impact with explicit repo= arg
- Pool and drift filtering behaviour
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import axon.mcp.server as server_module
from axon.core.drift import DriftCache, DriftLevel, DriftReport
from axon.core.repos import (
    RepoPool,
    RepoResolver,
    RepoUnavailable,
    RegistryEntry,
)
from axon.core.storage.base import SearchResult
from axon.mcp.repo_context import RepoContext
from axon.mcp.repo_routing import RoutingError, route_for_diff, route_for_path
from axon.mcp.server import (
    _ServerState,
    _build_repo_context,
    _ensure_multi_repo,
)
from axon.mcp.tools import (
    _foreign_query_hit_counts,
    _foreign_symbol_matches,
    _MAX_FOREIGN_COUNT_REPOS,
    handle_context,
    handle_impact,
    handle_query,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_registry_entry(
    registry_dir: Path, slug: str, repo_path: Path
) -> None:
    """Write a minimal RegistryEntry meta.json for a given slug."""
    slot = registry_dir / slug
    slot.mkdir(parents=True, exist_ok=True)
    entry = RegistryEntry(
        name=repo_path.name,
        path=str(repo_path),
        slug=slug,
        last_indexed_at='2024-01-01T00:00:00',
        stats={},
        embedding_model='',
        embedding_dimensions=0,
    )
    (slot / 'meta.json').write_text(
        json.dumps(entry.to_json()), encoding='utf-8'
    )


def _make_repo_dir(tmp_path: Path, name: str) -> Path:
    """Create a minimal repo directory (no actual Kuzu DB)."""
    repo = tmp_path / name
    repo.mkdir(parents=True, exist_ok=True)
    return repo


def _make_indexed_repo(tmp_path: Path, name: str) -> Path:
    """Create a repo directory with an initialised Kuzu DB."""
    from axon.core.storage.kuzu_backend import KuzuBackend

    repo = tmp_path / name
    (repo / '.axon').mkdir(parents=True)
    backend = KuzuBackend()
    backend.initialize(repo / '.axon' / 'kuzu')
    backend.close()
    return repo


def _fresh_report() -> DriftReport:
    """Build a FRESH drift report for monkeypatching."""
    return DriftReport(
        level=DriftLevel.FRESH,
        reason='test',
        last_indexed_at='',
        head_sha=None,
        head_sha_at_index=None,
        files_changed_estimate=None,
        files_indexed_estimate=None,
        watcher_alive=False,
        tier_used=None,
    )


def _stale_major_report() -> DriftReport:
    """Build a STALE_MAJOR drift report for monkeypatching."""
    return DriftReport(
        level=DriftLevel.STALE_MAJOR,
        reason='test - forced stale',
        last_indexed_at='',
        head_sha=None,
        head_sha_at_index=None,
        files_changed_estimate=None,
        files_indexed_estimate=None,
        watcher_alive=False,
        tier_used=None,
    )


def _make_search_result(slug: str, name: str) -> SearchResult:
    """Build a minimal SearchResult for a given slug and symbol name."""
    return SearchResult(
        node_id=f'function:src/{name}.py:{name}',
        score=1.0,
        node_name=name,
        file_path=f'src/{name}.py',
        label='function',
    )


@pytest.fixture(autouse=True)
def reset_server_state() -> Any:
    """Reset _state before and after every test for isolation."""
    server_module._state = _ServerState()
    yield
    server_module._state = _ServerState()


# ---------------------------------------------------------------------------
# _ensure_multi_repo
# ---------------------------------------------------------------------------


class TestEnsureMultiRepo:
    """_ensure_multi_repo initialisation and idempotency."""

    @pytest.mark.asyncio
    async def test_dispatch_defaults_to_local_when_repo_absent(
        self, tmp_path: Path
    ) -> None:
        """Without an explicit repo= arg, _build_repo_context returns local ctx."""
        registry = tmp_path / 'registry'
        local_repo = _make_repo_dir(tmp_path, 'myrepo')
        _write_registry_entry(registry, 'myrepo', local_repo)

        server_module._state.repo_path = local_repo

        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        pool = RepoPool(resolver)
        drift_cache = DriftCache()
        server_module._state.resolver = resolver
        server_module._state.pool = pool
        server_module._state.drift_cache = drift_cache
        server_module._state.local_slug = resolver.local().slug  # type: ignore[union-attr]

        mock_storage = MagicMock()
        server_module._state.storage = mock_storage

        result = await _build_repo_context('axon_dead_code', {})

        assert isinstance(result, RepoContext)
        assert result.is_local is True
        assert result.storage is mock_storage

    @pytest.mark.asyncio
    async def test_dispatch_resolves_repo_arg_to_foreign_pool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When repo=<slug> is given, the foreign pool entry is returned as ctx."""
        registry = tmp_path / 'registry'
        local_repo = _make_indexed_repo(tmp_path, 'local')
        foreign_repo = _make_indexed_repo(tmp_path, 'foreign')
        _write_registry_entry(registry, 'local', local_repo)
        _write_registry_entry(registry, 'foreign', foreign_repo)

        server_module._state.repo_path = local_repo
        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        pool = RepoPool(resolver)
        drift_cache = DriftCache()
        server_module._state.resolver = resolver
        server_module._state.pool = pool
        server_module._state.drift_cache = drift_cache
        server_module._state.local_slug = resolver.local().slug  # type: ignore[union-attr]

        monkeypatch.setattr(
            drift_cache, 'get_or_probe', lambda _: _fresh_report()
        )

        result = await _build_repo_context(
            'axon_context', {'symbol': 'Foo', 'repo': 'foreign'}
        )

        assert isinstance(result, RepoContext)
        assert result.is_local is False
        assert result.slug == 'foreign'

    @pytest.mark.asyncio
    async def test_dispatch_init_is_thread_safe_under_concurrent_calls(
        self, tmp_path: Path
    ) -> None:
        """_ensure_multi_repo initialises exactly once under concurrent async calls."""
        registry = tmp_path / 'registry'
        local_repo = _make_repo_dir(tmp_path, 'myrepo')
        _write_registry_entry(registry, 'myrepo', local_repo)

        server_module._state.repo_path = local_repo

        init_count = 0
        original_init = RepoResolver.__init__

        def counting_init(self: RepoResolver, **kwargs: Any) -> None:
            nonlocal init_count
            init_count += 1
            original_init(self, **kwargs)

        with patch.object(RepoResolver, '__init__', counting_init):
            await asyncio.gather(*(_ensure_multi_repo() for _ in range(10)))

        assert init_count == 1, (
            f'RepoResolver.__init__ called {init_count} times; expected 1'
        )
        assert server_module._state.resolver is not None
        assert server_module._state.pool is not None


# ---------------------------------------------------------------------------
# route_for_path
# ---------------------------------------------------------------------------


class TestRouteForPath:
    """route_for_path routes a file path to its owning repo."""

    def test_route_for_path_routes_to_owning_repo(
        self, tmp_path: Path
    ) -> None:
        """A file under a registered repo path resolves to that repo."""
        registry = tmp_path / 'registry'
        repo_a = _make_repo_dir(tmp_path, 'repo_a')
        _write_registry_entry(registry, 'repo_a', repo_a)

        resolver = RepoResolver(registry_dir=registry, local_repo_path=repo_a)

        # A file inside repo_a's directory tree.
        file_path = str(repo_a / 'src' / 'module.py')
        entry = route_for_path(resolver, file_path, None)

        assert entry.slug == 'repo_a'

    def test_route_for_path_ambiguous_raises_with_candidates(
        self, tmp_path: Path
    ) -> None:
        """Ambiguous path (two repos both prefix-match) raises RoutingError."""
        registry = tmp_path / 'registry'
        # parent is both a repo and encloses child
        parent_repo = _make_repo_dir(tmp_path, 'parent')
        child_repo = _make_repo_dir(tmp_path, 'parent/child')
        _write_registry_entry(registry, 'parent', parent_repo)
        _write_registry_entry(registry, 'child', child_repo)

        resolver = RepoResolver(registry_dir=registry, local_repo_path=None)

        # A file inside child_repo is also inside parent_repo.
        ambiguous_file = str(child_repo / 'main.py')

        with pytest.raises(RoutingError) as exc_info:
            route_for_path(resolver, ambiguous_file, None)

        assert 'parent' in exc_info.value.candidates
        assert 'child' in exc_info.value.candidates


# ---------------------------------------------------------------------------
# route_for_diff
# ---------------------------------------------------------------------------


class TestRouteForDiff:
    """route_for_diff routes by diff file paths."""

    def test_route_for_diff_routes_when_all_paths_share_repo(
        self, tmp_path: Path
    ) -> None:
        """Diff with files all under one repo resolves to that repo."""
        registry = tmp_path / 'registry'
        repo_a = _make_repo_dir(tmp_path, 'repo_a')
        _write_registry_entry(registry, 'repo_a', repo_a)

        resolver = RepoResolver(registry_dir=registry, local_repo_path=repo_a)

        diff = (
            f'diff --git a/{repo_a}/src/a.py b/{repo_a}/src/a.py\n'
            f'diff --git a/{repo_a}/src/b.py b/{repo_a}/src/b.py\n'
        )
        entry = route_for_diff(resolver, diff, None)

        assert entry.slug == 'repo_a'

    def test_route_for_diff_split_raises(self, tmp_path: Path) -> None:
        """Diff spanning two distinct repos raises RoutingError."""
        registry = tmp_path / 'registry'
        repo_a = _make_repo_dir(tmp_path, 'repo_a')
        repo_b = _make_repo_dir(tmp_path, 'repo_b')
        _write_registry_entry(registry, 'repo_a', repo_a)
        _write_registry_entry(registry, 'repo_b', repo_b)

        # Set local to repo_a so the fallback path doesn't collapse everything.
        resolver = RepoResolver(registry_dir=registry, local_repo_path=repo_a)

        diff = (
            f'diff --git a/{repo_a}/src/a.py b/{repo_a}/src/a.py\n'
            f'diff --git a/{repo_b}/src/b.py b/{repo_b}/src/b.py\n'
        )

        with pytest.raises(RoutingError, match='multiple repos'):
            route_for_diff(resolver, diff, None)


# ---------------------------------------------------------------------------
# handle_context with local alternates and foreign footers
# ---------------------------------------------------------------------------


class TestHandleContextMultiRepo:
    """handle_context cross-repo footer behaviour."""

    def test_handle_context_local_first_with_alternates_footer(
        self, make_ctx: Any
    ) -> None:
        """When local returns multiple matches, primary shown + 'Also matches' footer."""
        storage = MagicMock()
        primary = _make_search_result('local', 'Foo')
        alternate = SearchResult(
            node_id='function:tests/test_foo.py:Foo',
            score=0.8,
            node_name='Foo',
            file_path='tests/test_foo.py',
            label='function',
        )
        # _resolve_symbol returns both.
        storage.exact_name_search.return_value = [primary, alternate]
        storage.fts_search.return_value = [primary, alternate]
        from axon.core.graph.model import GraphNode, NodeLabel

        node = GraphNode(
            id=primary.node_id,
            label=NodeLabel.FUNCTION,
            name='Foo',
            file_path=primary.file_path,
            start_line=1,
            end_line=10,
        )
        storage.get_node.return_value = node
        storage.get_callers_with_confidence.return_value = []
        storage.get_callees_with_confidence.return_value = []
        storage.get_type_refs.return_value = []
        storage.get_process_memberships.return_value = {}

        result = handle_context(make_ctx(storage), 'Foo')

        assert 'Foo' in result
        assert 'Also matches in this repo:' in result
        assert 'tests/test_foo.py' in result

    def test_handle_context_redirects_when_local_empty_foreign_present(
        self, make_ctx: Any
    ) -> None:
        """When local has no match but foreign does, redirect response is returned."""
        storage = MagicMock()
        storage.exact_name_search.return_value = []
        storage.fts_search.return_value = []

        foreign_match = _make_search_result('other-repo', 'Bar')
        foreign_matches = [('other-repo', [foreign_match])]

        result = handle_context(
            make_ctx(storage), 'Bar', foreign_matches=foreign_matches
        )

        assert "Symbol 'Bar' not found in this repo." in result
        assert 'other-repo' in result
        assert 'repo=<slug>' in result


# ---------------------------------------------------------------------------
# handle_call_path asymmetric top_k
# ---------------------------------------------------------------------------


class TestHandleCallPathTopK:
    """handle_call_path uses _LOCAL_MATCHES_TOP_K for from_symbol and 1 for to."""

    def test_handle_call_path_top_k_5_for_from_top_k_1_for_to(
        self, make_ctx: Any
    ) -> None:
        """from_symbol uses limit=5, to_symbol uses limit=1."""
        storage = MagicMock()
        from axon.core.graph.model import GraphNode, NodeLabel
        from axon.mcp.tools import handle_call_path

        from_result = _make_search_result('local', 'Foo')
        to_result = _make_search_result('local', 'Bar')
        storage.exact_name_search.return_value = []
        storage.fts_search.side_effect = [
            [from_result],  # from_symbol resolution
            [to_result],  # to_symbol resolution
        ]
        from_node = GraphNode(
            id=from_result.node_id,
            label=NodeLabel.FUNCTION,
            name='Foo',
            file_path=from_result.file_path,
            start_line=1,
            end_line=5,
        )
        to_node = GraphNode(
            id=to_result.node_id,
            label=NodeLabel.FUNCTION,
            name='Bar',
            file_path=to_result.file_path,
            start_line=10,
            end_line=20,
        )
        storage.get_node.side_effect = [from_node, to_node]
        storage.get_callees.return_value = []

        handle_call_path(make_ctx(storage), 'Foo', 'Bar')

        # fts_search calls: first with limit=5 (from), then with limit=1 (to).
        calls = storage.fts_search.call_args_list
        assert calls[0][1]['limit'] == 5 or calls[0][0][1] == 5
        assert calls[1][1]['limit'] == 1 or calls[1][0][1] == 1


# ---------------------------------------------------------------------------
# handle_query with foreign hit count footer
# ---------------------------------------------------------------------------


class TestHandleQueryMultiRepo:
    """handle_query cross-repo hit count footer."""

    def test_handle_query_local_only_with_foreign_hit_count_footer(
        self, make_ctx: Any
    ) -> None:
        """When foreign_hits provided, footer is appended to query results."""
        storage = MagicMock()
        local_result = _make_search_result('local', 'validate')
        storage.fts_search.return_value = [local_result]
        storage.get_process_memberships.return_value = {}

        foreign_hits = [('other-repo', 7)]
        result = handle_query(
            make_ctx(storage), 'validate', foreign_hits=foreign_hits
        )

        assert 'Hits also exist in:' in result
        assert 'other-repo' in result
        assert '7' in result

    def test_handle_query_foreign_count_capped_at_max(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_foreign_query_hit_counts caps foreign repos at _MAX_FOREIGN_COUNT_REPOS."""
        registry = tmp_path / 'registry'
        local_repo = _make_repo_dir(tmp_path, 'local')
        _write_registry_entry(registry, 'local', local_repo)

        # Register more foreign repos than the cap.
        foreign_repos = []
        for i in range(_MAX_FOREIGN_COUNT_REPOS + 3):
            repo = _make_repo_dir(tmp_path, f'foreign_{i}')
            _write_registry_entry(registry, f'foreign_{i}', repo)
            foreign_repos.append(repo)

        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        pool = MagicMock(spec=RepoPool)
        drift_cache = MagicMock(spec=DriftCache)

        # All foreign repos return fresh and have hits.
        drift_cache.get_or_probe.return_value = _fresh_report()
        mock_backend = MagicMock()
        mock_backend.fts_search.return_value = [
            _make_search_result('foreign', 'foo')
        ]
        pool.get.return_value = mock_backend

        results = _foreign_query_hit_counts(
            pool, resolver, drift_cache, 'foo', exclude_slug='local'
        )

        assert len(results) <= _MAX_FOREIGN_COUNT_REPOS


# ---------------------------------------------------------------------------
# handle_impact with explicit repo arg
# ---------------------------------------------------------------------------


class TestHandleImpactMultiRepo:
    """handle_impact works when ctx targets a foreign repo."""

    def test_handle_impact_with_repo_arg(self, make_ctx: Any) -> None:
        """handle_impact uses the storage in ctx (from foreign pool)."""
        from axon.core.graph.model import GraphNode, NodeLabel

        storage = MagicMock()
        result_sr = _make_search_result('foreign', 'Widget')
        storage.exact_name_search.return_value = [result_sr]
        storage.fts_search.return_value = [result_sr]
        node = GraphNode(
            id=result_sr.node_id,
            label=NodeLabel.FUNCTION,
            name='Widget',
            file_path=result_sr.file_path,
            start_line=1,
            end_line=10,
        )
        storage.get_node.return_value = node
        storage.traverse_with_depth.return_value = []
        storage.get_process_memberships.return_value = {}

        ctx = make_ctx(storage, slug='foreign', is_local=False)
        result = handle_impact(ctx, 'Widget')

        assert 'Widget' in result


# ---------------------------------------------------------------------------
# Pool fan-out: skips unavailable and STALE_MAJOR foreign repos
# ---------------------------------------------------------------------------


class TestPoolFanOut:
    """Fan-out helpers filter out unavailable and STALE_MAJOR repos silently."""

    def test_pool_skips_unavailable_foreign_repos_silently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_foreign_symbol_matches does not propagate RepoUnavailable errors."""
        registry = tmp_path / 'registry'
        local_repo = _make_repo_dir(tmp_path, 'local')
        foreign_repo = _make_repo_dir(tmp_path, 'foreign')
        _write_registry_entry(registry, 'local', local_repo)
        _write_registry_entry(registry, 'foreign', foreign_repo)

        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        pool = MagicMock(spec=RepoPool)
        drift_cache = MagicMock(spec=DriftCache)
        drift_cache.get_or_probe.return_value = _fresh_report()
        pool.get.side_effect = RepoUnavailable('foreign', 'test error')

        # Must not raise - silently returns empty.
        result = _foreign_symbol_matches(
            pool, resolver, drift_cache, 'Foo', exclude_slug='local'
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_pool_skips_stale_major_foreign_in_lookups_but_refuses_at_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """STALE_MAJOR foreign repos excluded from fan-out; refused when targeted."""
        registry = tmp_path / 'registry'
        local_repo = _make_indexed_repo(tmp_path, 'local')
        stale_repo = _make_indexed_repo(tmp_path, 'stale')
        _write_registry_entry(registry, 'local', local_repo)
        _write_registry_entry(registry, 'stale', stale_repo)

        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        pool = MagicMock(spec=RepoPool)
        drift_cache = MagicMock(spec=DriftCache)
        drift_cache.get_or_probe.return_value = _stale_major_report()

        # Fan-out: STALE_MAJOR silently excluded.
        result = _foreign_symbol_matches(
            pool, resolver, drift_cache, 'Foo', exclude_slug='local'
        )
        assert result == []
        # The pool was never asked for the stale repo.
        pool.get.assert_not_called()

        # Targeted: _build_repo_context must produce a refusal string.
        server_module._state.resolver = resolver
        server_module._state.pool = pool
        server_module._state.drift_cache = drift_cache
        server_module._state.local_slug = resolver.local().slug  # type: ignore[union-attr]
        server_module._state.storage = MagicMock()

        result_ctx = await _build_repo_context(
            'axon_context', {'symbol': 'Foo', 'repo': 'stale'}
        )
        assert isinstance(result_ctx, str)
        assert 'STALE_MAJOR' in result_ctx
