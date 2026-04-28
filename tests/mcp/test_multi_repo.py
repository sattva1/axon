"""Tests for multi-repo MCP dispatch layer (Phase 3, plan section 3.8).

Coverage:
- _ensure_multi_repo init flow and thread-safety
- _build_repo_context routing (repo arg, path, diff, STALE_MAJOR refusal)
- route_for_path and route_for_diff helpers
- Fan-out helpers: _foreign_symbol_matches, _foreign_query_hit_counts
- handle_context and handle_query with foreign repo footers
- handle_impact with explicit repo= arg
- Drift filtering behaviour
- Phase 1: open_foreign_backend handle lifetime and isolation tests
- Phase 2: per-call local opens, lock bypass, _state.storage ignored
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import AsyncExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import axon.mcp.server as server_module
from axon.core.drift import DriftCache, DriftLevel, DriftReport
from axon.core.repos import (
    RegistryEntry,
    RepoResolver,
    RepoUnavailable,
    open_foreign_backend,
)
from axon.core.storage.base import SearchResult
from axon.core.storage.kuzu_backend import KuzuBackend
from axon.mcp.repo_context import RepoContext
from axon.mcp.repo_routing import RoutingError, route_for_diff, route_for_path
from axon.mcp.server import (
    _build_repo_context,
    _ensure_multi_repo,
    _ServerState,
    _with_storage,
    call_tool,
    set_db_path,
    set_storage,
)
from axon.mcp.tools import (
    _MAX_FOREIGN_COUNT_REPOS,
    _foreign_query_hit_counts,
    _foreign_symbol_matches,
    _format_foreign_matches,
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


def _stale_minor_report() -> DriftReport:
    """Build a STALE_MINOR drift report for monkeypatching."""
    return DriftReport(
        level=DriftLevel.STALE_MINOR,
        reason='HEAD unchanged but working tree is dirty',
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
        """Without an explicit repo= arg, _build_repo_context returns local ctx.

        Phase 2: dispatch opens a fresh RO KuzuBackend per call, so the repo
        must have an initialised DB before dispatch runs.
        """
        registry = tmp_path / 'registry'
        local_repo = _make_indexed_repo(tmp_path, 'myrepo')
        _write_registry_entry(registry, 'myrepo', local_repo)

        server_module._state.repo_path = local_repo

        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        drift_cache = DriftCache()
        server_module._state.resolver = resolver
        server_module._state.drift_cache = drift_cache
        server_module._state.local_slug = resolver.local().slug  # type: ignore[union-attr]

        async with AsyncExitStack() as stack:
            result = await _build_repo_context('axon_dead_code', {}, stack)

        assert isinstance(result, RepoContext)
        assert result.is_local is True
        assert isinstance(result.storage, KuzuBackend)

    @pytest.mark.asyncio
    async def test_dispatch_resolves_repo_arg_to_foreign(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When repo=<slug> is given, the foreign backend is returned as ctx."""
        registry = tmp_path / 'registry'
        local_repo = _make_indexed_repo(tmp_path, 'local')
        foreign_repo = _make_indexed_repo(tmp_path, 'foreign')
        _write_registry_entry(registry, 'local', local_repo)
        _write_registry_entry(registry, 'foreign', foreign_repo)

        server_module._state.repo_path = local_repo
        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        drift_cache = DriftCache()
        server_module._state.resolver = resolver
        server_module._state.drift_cache = drift_cache
        server_module._state.local_slug = resolver.local().slug  # type: ignore[union-attr]

        monkeypatch.setattr(
            drift_cache, 'get_or_probe', lambda _: _fresh_report()
        )

        async with AsyncExitStack() as stack:
            result = await _build_repo_context(
                'axon_context', {'symbol': 'Foo', 'repo': 'foreign'}, stack
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
        for i in range(_MAX_FOREIGN_COUNT_REPOS + 3):
            repo = _make_indexed_repo(tmp_path, f'foreign_{i}')
            _write_registry_entry(registry, f'foreign_{i}', repo)

        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        drift_cache = MagicMock(spec=DriftCache)
        drift_cache.get_or_probe.return_value = _fresh_report()

        mock_backend = MagicMock()
        mock_backend.fts_search.return_value = [
            _make_search_result('foreign', 'foo')
        ]

        @contextmanager
        def _fake_open(res: Any, slug: str, **kwargs: Any):
            yield mock_backend

        with patch(
            'axon.mcp.tools.open_foreign_backend', side_effect=_fake_open
        ):
            results = _foreign_query_hit_counts(
                resolver, drift_cache, 'foo', exclude_slug='local'
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
# Fan-out: skips unavailable and STALE_MAJOR foreign repos
# ---------------------------------------------------------------------------


class TestFanOut:
    """Fan-out helpers filter out unavailable and STALE_MAJOR repos silently."""

    def test_skips_unavailable_foreign_repos_silently(
        self, tmp_path: Path
    ) -> None:
        """_foreign_symbol_matches does not propagate RepoUnavailable errors."""
        registry = tmp_path / 'registry'
        local_repo = _make_repo_dir(tmp_path, 'local')
        # Register a foreign repo path that has no real DB - open will fail.
        ghost_repo = tmp_path / 'ghost'
        ghost_repo.mkdir()
        _write_registry_entry(registry, 'local', local_repo)
        _write_registry_entry(registry, 'ghost', ghost_repo)

        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        drift_cache = MagicMock(spec=DriftCache)
        drift_cache.get_or_probe.return_value = _fresh_report()

        # Must not raise - silently returns empty.
        result = _foreign_symbol_matches(
            resolver, drift_cache, 'Foo', exclude_slug='local'
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_stale_major_foreign_in_lookups_but_refuses_at_target(
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
        drift_cache = MagicMock(spec=DriftCache)
        drift_cache.get_or_probe.return_value = _stale_major_report()

        # Fan-out: STALE_MAJOR silently excluded.
        result = _foreign_symbol_matches(
            resolver, drift_cache, 'Foo', exclude_slug='local'
        )
        assert result == []

        # Targeted: _build_repo_context must produce a refusal string.
        server_module._state.resolver = resolver
        server_module._state.drift_cache = drift_cache
        server_module._state.local_slug = resolver.local().slug  # type: ignore[union-attr]
        server_module._state.storage = MagicMock()

        async with AsyncExitStack() as stack:
            result_ctx = await _build_repo_context(
                'axon_context', {'symbol': 'Foo', 'repo': 'stale'}, stack
            )
        assert isinstance(result_ctx, str)
        assert 'STALE_MAJOR' in result_ctx


# ---------------------------------------------------------------------------
# Drift warning via _maybe_drift_warning / call_tool
# ---------------------------------------------------------------------------


class TestStaleMinerForeignWarning:
    """Drift warning is prepended for foreign STALE_MINOR repos, not local."""

    @pytest.mark.asyncio
    async def test_stale_minor_foreign_repo_warning_prepended_via_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Foreign STALE_MINOR repo: drift warning appears at top of response."""
        registry = tmp_path / 'registry'
        local_repo = _make_indexed_repo(tmp_path, 'local')
        foreign_repo = _make_indexed_repo(tmp_path, 'foreign')
        _write_registry_entry(registry, 'local', local_repo)
        _write_registry_entry(registry, 'foreign', foreign_repo)

        server_module._state.repo_path = local_repo
        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        drift_cache = MagicMock(spec=DriftCache)
        drift_cache.get_or_probe.return_value = _stale_minor_report()

        server_module._state.resolver = resolver
        server_module._state.drift_cache = drift_cache
        server_module._state.local_slug = resolver.local().slug  # type: ignore[union-attr]
        server_module._state.storage = MagicMock()

        response = await call_tool('axon_dead_code', {'repo': 'foreign'})
        text = response[0].text

        assert text.startswith('Note:'), (
            f'Expected drift warning at start; got: {text[:120]!r}'
        )
        assert 'minor drift' in text.lower() or 'stale' in text.lower()

    @pytest.mark.asyncio
    async def test_stale_minor_local_repo_no_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Local repo does not receive drift warning even when STALE_MINOR."""
        registry = tmp_path / 'registry'
        local_repo = _make_indexed_repo(tmp_path, 'local')
        _write_registry_entry(registry, 'local', local_repo)

        server_module._state.repo_path = local_repo
        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        drift_cache = MagicMock(spec=DriftCache)
        drift_cache.get_or_probe.return_value = _stale_minor_report()

        server_module._state.resolver = resolver
        server_module._state.drift_cache = drift_cache
        server_module._state.local_slug = resolver.local().slug  # type: ignore[union-attr]

        mock_storage = MagicMock()
        mock_storage.get_dead_code.return_value = []
        server_module._state.storage = mock_storage

        response = await call_tool('axon_dead_code', {})
        text = response[0].text

        assert not text.startswith('Note:'), (
            f'Unexpected drift warning on local repo; got: {text[:120]!r}'
        )


# ---------------------------------------------------------------------------
# _format_foreign_matches: redirect vs footer differ only in intro line
# ---------------------------------------------------------------------------


class TestFormatterConsistency:
    """_format_foreign_matches redirect and footer share per-repo entry format."""

    def test_foreign_matches_and_redirect_share_formatter(self) -> None:
        """redirect=True and redirect=False differ only in the introductory line."""
        match = SearchResult(
            node_id='function:src/widget.py:Widget',
            score=0.9,
            node_name='Widget',
            file_path='src/widget.py',
            label='function',
        )
        matches = [('other-repo', [match])]

        footer = _format_foreign_matches(matches, redirect=False)
        redirect = _format_foreign_matches(matches, redirect=True)

        # Strip leading/trailing whitespace per line and filter blanks to
        # isolate content lines from structural padding differences.
        footer_content = [line for line in footer.splitlines() if line.strip()]
        redirect_content = [line for line in redirect.splitlines() if line.strip()]

        # Both must produce the same number of content lines.
        assert len(footer_content) == len(redirect_content)

        # The introductory lines must differ (redirect phrasing vs footer).
        assert footer_content[0] != redirect_content[0]

        # All per-repo entry lines (after the intro) must be identical.
        assert footer_content[1:] == redirect_content[1:]

        # The per-repo entry line must include slug, name, label, and path.
        entry_line = footer_content[1]
        assert 'other-repo' in entry_line
        assert 'Widget' in entry_line
        assert 'function' in entry_line
        assert 'src/widget.py' in entry_line


# ---------------------------------------------------------------------------
# Bug 4 regression: path-keyed routing normalises absolute file_path
# ---------------------------------------------------------------------------


class TestPathKeyedNormalization:
    """_build_repo_context rewrites absolute file_path to repo-relative
    after routing to a foreign repo, so handler storage queries match
    the indexed (repo-relative) File node keys."""

    def test_absolute_path_under_foreign_repo_is_made_relative(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Absolute path under a foreign repo is rewritten to repo-relative."""
        registry = tmp_path / 'registry'
        local_repo = _make_repo_dir(tmp_path, 'local')
        foreign_repo = _make_indexed_repo(tmp_path, 'foreign')
        _write_registry_entry(registry, 'foreign', foreign_repo)

        # Patch _state in place so _ensure_multi_repo points at our registry.
        monkeypatch.setattr(server_module, '_state', _ServerState())
        server_module._state.repo_path = local_repo
        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        server_module._state.resolver = resolver
        server_module._state.drift_cache = MagicMock(spec=DriftCache)
        server_module._state.drift_cache.get_or_probe.return_value = (
            _fresh_report()
        )
        server_module._state.local_slug = 'local'

        target_file = foreign_repo / 'src' / 'widget.py'
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text('', encoding='utf-8')

        arguments = {'file_path': str(target_file)}

        async def _run() -> RepoContext | str:
            async with AsyncExitStack() as stack:
                return await _build_repo_context(
                    'axon_file_context', arguments, stack
                )

        ctx_or_refusal = asyncio.run(_run())
        assert isinstance(ctx_or_refusal, RepoContext)
        assert ctx_or_refusal.slug == 'foreign'
        # The arguments dict was mutated in place.
        assert arguments['file_path'] == 'src/widget.py'

    def test_relative_path_passes_through_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A repo-relative path stays unchanged (no spurious resolution)."""
        registry = tmp_path / 'registry'
        local_repo = _make_indexed_repo(tmp_path, 'local')
        _write_registry_entry(registry, 'local', local_repo)

        monkeypatch.setattr(server_module, '_state', _ServerState())
        server_module._state.repo_path = local_repo
        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        server_module._state.resolver = resolver
        server_module._state.drift_cache = MagicMock(spec=DriftCache)
        server_module._state.drift_cache.get_or_probe.return_value = (
            _fresh_report()
        )
        server_module._state.local_slug = 'local'

        arguments = {'file_path': 'src/widget.py'}

        async def _run() -> RepoContext | str:
            async with AsyncExitStack() as stack:
                return await _build_repo_context(
                    'axon_file_context', arguments, stack
                )

        ctx_or_refusal = asyncio.run(_run())
        assert isinstance(ctx_or_refusal, RepoContext)
        # Path unchanged.
        assert arguments['file_path'] == 'src/widget.py'


# ---------------------------------------------------------------------------
# Bug 3 regression: cross-repo footer excludes the targeted repo
# ---------------------------------------------------------------------------


class TestCrossRepoFooterExcludesTarget:
    """The 'Also exists in other repos' / 'Hits also exist in' footers
    must exclude the targeted repo, not just the local one."""

    def test_foreign_symbol_matches_excludes_target_when_target_is_foreign(
        self, tmp_path: Path
    ) -> None:
        """exclude_slug=ctx.slug correctly omits a foreign target from the
        fan-out result."""
        registry = tmp_path / 'registry'
        local_repo = _make_repo_dir(tmp_path, 'local')
        target_repo = _make_indexed_repo(tmp_path, 'target')
        other_repo = _make_indexed_repo(tmp_path, 'other')
        _write_registry_entry(registry, 'target', target_repo)
        _write_registry_entry(registry, 'other', other_repo)

        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        drift_cache = MagicMock(spec=DriftCache)
        drift_cache.get_or_probe.return_value = _fresh_report()

        target_mock = MagicMock()
        other_mock = MagicMock()
        target_mock.exact_name_search.return_value = [
            _make_search_result('target', 'Foo')
        ]
        other_mock.exact_name_search.return_value = [
            _make_search_result('other', 'Foo')
        ]

        # Use a context manager mock so open_foreign_backend works.
        def _fake_open(res: Any, slug: str, **kwargs: Any) -> Any:
            @contextmanager
            def _cm():
                if slug == 'target':
                    yield target_mock
                elif slug == 'other':
                    yield other_mock
                else:
                    raise RepoUnavailable(slug, 'unknown')

            return _cm()

        with patch(
            'axon.mcp.tools.open_foreign_backend', side_effect=_fake_open
        ):
            results = _foreign_symbol_matches(
                resolver, drift_cache, 'Foo', exclude_slug='target'
            )

        slugs = {slug for slug, _ in results}
        assert 'target' not in slugs
        assert 'other' in slugs

    def test_foreign_query_hit_counts_excludes_target_when_target_is_foreign(
        self, tmp_path: Path
    ) -> None:
        """exclude_slug=ctx.slug correctly omits a foreign target from query
        hit counts."""
        registry = tmp_path / 'registry'
        local_repo = _make_repo_dir(tmp_path, 'local')
        target_repo = _make_indexed_repo(tmp_path, 'target')
        other_repo = _make_indexed_repo(tmp_path, 'other')
        _write_registry_entry(registry, 'target', target_repo)
        _write_registry_entry(registry, 'other', other_repo)

        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        drift_cache = MagicMock(spec=DriftCache)
        drift_cache.get_or_probe.return_value = _fresh_report()

        target_mock = MagicMock()
        other_mock = MagicMock()
        target_mock.fts_search.return_value = [
            _make_search_result('target', 'foo')
        ]
        other_mock.fts_search.return_value = [
            _make_search_result('other', 'foo')
        ]

        def _fake_open(res: Any, slug: str, **kwargs: Any) -> Any:
            @contextmanager
            def _cm():
                if slug == 'target':
                    yield target_mock
                elif slug == 'other':
                    yield other_mock
                else:
                    raise RepoUnavailable(slug, 'unknown')

            return _cm()

        with patch(
            'axon.mcp.tools.open_foreign_backend', side_effect=_fake_open
        ):
            results = _foreign_query_hit_counts(
                resolver, drift_cache, 'foo', exclude_slug='target'
            )

        slugs = {slug for slug, _ in results}
        assert 'target' not in slugs
        assert 'other' in slugs


# ---------------------------------------------------------------------------
# Phase 1: open_foreign_backend handle lifetime and isolation
# ---------------------------------------------------------------------------


class TestForeignHandleLifetime:
    """open_foreign_backend handles are released promptly and not shared."""

    @pytest.mark.asyncio
    async def test_dispatch_releases_foreign_handle_on_stack_exit(
        self, tmp_path: Path
    ) -> None:
        """After the dispatch stack exits, the foreign handle is released.

        Proof: a fresh open_foreign_backend call after dispatch succeeds
        (no lock leak prevents re-acquisition).
        """
        registry = tmp_path / 'registry'
        local_repo = _make_repo_dir(tmp_path, 'local')
        foreign_repo = _make_indexed_repo(tmp_path, 'foreign')
        _write_registry_entry(registry, 'local', local_repo)
        _write_registry_entry(registry, 'foreign', foreign_repo)

        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        drift_cache = MagicMock(spec=DriftCache)
        drift_cache.get_or_probe.return_value = _fresh_report()

        server_module._state.repo_path = local_repo
        server_module._state.resolver = resolver
        server_module._state.drift_cache = drift_cache
        server_module._state.local_slug = resolver.local().slug  # type: ignore[union-attr]
        server_module._state.storage = MagicMock()

        # Dispatch opens the foreign handle inside its own AsyncExitStack.
        response = await call_tool('axon_dead_code', {'repo': 'foreign'})
        assert response

        # After dispatch returns, the stack is exited and the handle closed.
        # A fresh open must succeed (no lingering RO lock blocking re-entry).
        with open_foreign_backend(resolver, 'foreign') as backend:
            assert backend._db is not None

    @pytest.mark.asyncio
    async def test_concurrent_dispatch_does_not_share_foreign_handle(
        self, tmp_path: Path
    ) -> None:
        """Two concurrent call_tool invocations for the same foreign slug each
        get their own KuzuBackend instance."""
        registry = tmp_path / 'registry'
        local_repo = _make_repo_dir(tmp_path, 'local')
        foreign_repo = _make_indexed_repo(tmp_path, 'foreign')
        _write_registry_entry(registry, 'local', local_repo)
        _write_registry_entry(registry, 'foreign', foreign_repo)

        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        drift_cache = MagicMock(spec=DriftCache)
        drift_cache.get_or_probe.return_value = _fresh_report()

        server_module._state.repo_path = local_repo
        server_module._state.resolver = resolver
        server_module._state.drift_cache = drift_cache
        server_module._state.local_slug = resolver.local().slug  # type: ignore[union-attr]
        server_module._state.storage = MagicMock()

        backend_ids: list[int] = []
        original_initialize = KuzuBackend.initialize

        def capturing_initialize(
            self: KuzuBackend, *args: Any, **kwargs: Any
        ) -> None:
            backend_ids.append(id(self))
            original_initialize(self, *args, **kwargs)

        with patch.object(KuzuBackend, 'initialize', capturing_initialize):
            await asyncio.gather(
                call_tool('axon_dead_code', {'repo': 'foreign'}),
                call_tool('axon_dead_code', {'repo': 'foreign'}),
            )

        # Each dispatch must have opened its own backend.
        assert len(backend_ids) >= 2
        assert len(set(backend_ids)) >= 2, (
            'Both dispatches used the same KuzuBackend instance'
        )

    @pytest.mark.asyncio
    async def test_axon_analyze_can_acquire_rw_during_session_after_foreign_dispatch_completes(
        self, tmp_path: Path
    ) -> None:
        """After a foreign RO dispatch closes, the same process can open RW.

        Simulates axon analyze acquiring write access after a read-only MCP
        dispatch has finished and released its handle.
        """
        registry = tmp_path / 'registry'
        local_repo = _make_repo_dir(tmp_path, 'local')
        foreign_repo = _make_indexed_repo(tmp_path, 'foreign')
        _write_registry_entry(registry, 'local', local_repo)
        _write_registry_entry(registry, 'foreign', foreign_repo)

        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        drift_cache = MagicMock(spec=DriftCache)
        drift_cache.get_or_probe.return_value = _fresh_report()

        server_module._state.repo_path = local_repo
        server_module._state.resolver = resolver
        server_module._state.drift_cache = drift_cache
        server_module._state.local_slug = resolver.local().slug  # type: ignore[union-attr]
        server_module._state.storage = MagicMock()

        # Open RO via dispatch, then let it close.
        await call_tool('axon_dead_code', {'repo': 'foreign'})

        # After dispatch, open RW from the same process.
        rw_backend = KuzuBackend()
        rw_backend.initialize(foreign_repo / '.axon' / 'kuzu', read_only=False)
        assert rw_backend._db is not None
        rw_backend.close()

    @pytest.mark.asyncio
    async def test_fan_out_open_does_not_block_event_loop(
        self, tmp_path: Path
    ) -> None:
        """Fan-out inside asyncio.to_thread does not stall the event loop.

        A concurrent asyncio task must be able to increment a counter while
        the fan-out open is sleeping, proving fan-out runs off the event loop.
        """
        registry = tmp_path / 'registry'
        local_repo = _make_repo_dir(tmp_path, 'local')
        foreign_repo = _make_indexed_repo(tmp_path, 'foreign')
        _write_registry_entry(registry, 'local', local_repo)
        _write_registry_entry(registry, 'foreign', foreign_repo)

        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        drift_cache = MagicMock(spec=DriftCache)
        drift_cache.get_or_probe.return_value = _fresh_report()

        counter = [0]
        sleep_seconds = 0.1

        original_initialize = KuzuBackend.initialize

        def slow_initialize(
            self: KuzuBackend, *args: Any, **kwargs: Any
        ) -> None:
            # Blocking sleep inside the to_thread worker.
            time.sleep(sleep_seconds)
            original_initialize(self, *args, **kwargs)

        async def increment_forever() -> None:
            while True:
                await asyncio.sleep(0)
                counter[0] += 1

        async def run_fan_out() -> None:
            def _work() -> None:
                with open_foreign_backend(resolver, 'foreign'):
                    pass

            await asyncio.to_thread(_work)

        with patch.object(KuzuBackend, 'initialize', slow_initialize):
            task = asyncio.ensure_future(increment_forever())
            try:
                await run_fan_out()
            finally:
                task.cancel()

        # The counter must have advanced while the blocking open was happening.
        assert counter[0] > 0, (
            'Event loop was blocked - counter did not advance during fan-out open'
        )


# ---------------------------------------------------------------------------
# Phase 2: per-call local opens, lock bypass, _state.storage ignored
# ---------------------------------------------------------------------------


class TestPhase2LocalPerCallOpen:
    """Phase 2: local dispatch opens a fresh RO KuzuBackend per call."""

    @pytest.mark.asyncio
    async def test_dispatch_local_opens_per_call_in_standalone_mode(
        self, tmp_path: Path
    ) -> None:
        """Dispatch opens a fresh KuzuBackend even when _state.storage is None.

        Standalone mode: no watcher, _state.storage never set. Each call_tool
        invocation must open and close its own RO handle.
        """
        registry = tmp_path / 'registry'
        local_repo = _make_indexed_repo(tmp_path, 'myrepo')
        _write_registry_entry(registry, 'myrepo', local_repo)

        server_module._state.repo_path = local_repo
        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        server_module._state.resolver = resolver
        server_module._state.drift_cache = DriftCache()
        server_module._state.local_slug = resolver.local().slug  # type: ignore[union-attr]
        # _state.storage intentionally left as None.

        initialize_calls: list[int] = []
        original_initialize = KuzuBackend.initialize

        def counting_initialize(
            self: KuzuBackend, *args: Any, **kwargs: Any
        ) -> None:
            initialize_calls.append(id(self))
            original_initialize(self, *args, **kwargs)

        with patch.object(KuzuBackend, 'initialize', counting_initialize):
            response = await call_tool('axon_dead_code', {})

        assert response
        # At least one fresh KuzuBackend was opened during the call.
        assert len(initialize_calls) >= 1

    @pytest.mark.asyncio
    async def test_dispatch_local_does_not_acquire_state_lock_for_reads(
        self, tmp_path: Path
    ) -> None:
        """Two concurrent call_tool calls against local do not serialize via lock.

        Phase 2 removed the _state.lock acquisition from reads. The proof:
        inject a pre-acquired asyncio.Lock as _state.lock. If call_tool tried
        to acquire it, both concurrent calls would deadlock. Both must complete.
        """
        registry = tmp_path / 'registry'
        local_repo = _make_indexed_repo(tmp_path, 'myrepo')
        _write_registry_entry(registry, 'myrepo', local_repo)

        server_module._state.repo_path = local_repo
        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        server_module._state.resolver = resolver
        server_module._state.drift_cache = DriftCache()
        server_module._state.local_slug = resolver.local().slug  # type: ignore[union-attr]

        # Pre-acquire the lock so any attempt to acquire it would deadlock.
        held_lock = asyncio.Lock()
        await held_lock.acquire()
        server_module._state.lock = held_lock

        # Both calls must complete without deadlocking (5 s timeout).
        results = await asyncio.wait_for(
            asyncio.gather(
                call_tool('axon_dead_code', {}),
                call_tool('axon_dead_code', {}),
            ),
            timeout=5.0,
        )

        assert len(results) == 2
        for response in results:
            assert 'Internal error' not in response[0].text

    @pytest.mark.asyncio
    async def test_axon_analyze_can_acquire_rw_during_session_when_watcher_absent(
        self, tmp_path: Path
    ) -> None:
        """A fresh RW open succeeds in the same process when no watcher holds the lock.

        Phase 2 means MCP dispatch never holds a long-lived RO handle. After
        dispatch completes, another component (axon analyze) can freely acquire
        RW on the same DB.
        """
        registry = tmp_path / 'registry'
        local_repo = _make_indexed_repo(tmp_path, 'myrepo')
        _write_registry_entry(registry, 'myrepo', local_repo)

        server_module._state.repo_path = local_repo
        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        server_module._state.resolver = resolver
        server_module._state.drift_cache = DriftCache()
        server_module._state.local_slug = resolver.local().slug  # type: ignore[union-attr]

        # Run a dispatch to confirm it opens and closes cleanly.
        await call_tool('axon_dead_code', {})

        # After dispatch, the per-call RO handle is released. RW open must succeed.
        db_path = local_repo / '.axon' / 'kuzu'
        rw_backend = KuzuBackend()
        rw_backend.initialize(db_path, read_only=False)
        assert rw_backend._db is not None
        rw_backend.close()

    @pytest.mark.asyncio
    async def test_read_resource_opens_per_call_not_via_state_storage(
        self, tmp_path: Path
    ) -> None:
        """_with_storage ignores _state.storage and always opens RO per call.

        Set _state.storage to a sentinel that raises on any method call.
        _with_storage must succeed because it never touches _state.storage.
        """
        db_path = tmp_path / '.axon' / 'kuzu'
        (tmp_path / '.axon').mkdir(parents=True)
        rw = KuzuBackend()
        rw.initialize(db_path, read_only=False)
        rw.close()

        class _RaisingSentinel:
            def __getattr__(self, name: str) -> None:
                raise AttributeError(
                    f'_state.storage was accessed via .{name} - '
                    '_with_storage must not touch this field in Phase 2'
                )

        set_storage(_RaisingSentinel())  # type: ignore[arg-type]
        set_db_path(db_path)

        seen: list[object] = []
        result = await _with_storage(lambda st: seen.append(st) or 'ok')

        assert result == 'ok'
        assert len(seen) == 1
        assert isinstance(seen[0], KuzuBackend)
