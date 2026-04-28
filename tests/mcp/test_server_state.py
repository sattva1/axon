"""Tests for the _ServerState injection API in axon.mcp.server.

All tests treat the public API (set_repo_path / _resolve_db_path /
_with_storage) as the observable surface. Only the autouse reset fixture
touches _state directly, for isolation purposes.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import axon.mcp.server as server_module
import axon.mcp.tools as tools_module
from axon.core.drift import DriftCache
from axon.core.repos import RegistryEntry, RepoResolver
from axon.core.storage.kuzu_backend import KuzuBackend
from axon.mcp.server import (
    _dispatch_tool,
    _resolve_db_path,
    _ServerState,
    _with_storage,
    call_tool,
    set_repo_path,
)


@pytest.fixture(autouse=True)
def reset_state() -> None:
    """Reset module-level _state before each test for isolation."""
    server_module._state = _ServerState()
    yield
    server_module._state = _ServerState()


class TestWithStorage:
    async def test_with_storage_opens_per_call(self, tmp_path: Path) -> None:
        """_with_storage always opens a fresh RO connection per call."""
        db_path = tmp_path / '.axon' / 'kuzu'
        (tmp_path / '.axon').mkdir(parents=True)
        rw = KuzuBackend()
        rw.initialize(db_path, read_only=False)
        rw.close()

        set_repo_path(tmp_path)

        seen: list[object] = []
        await _with_storage(lambda st: seen.append(st) or 'ok')

        assert len(seen) == 1
        assert isinstance(seen[0], KuzuBackend)

    async def test_with_storage_concurrent_calls_are_not_serialized(
        self, tmp_path: Path
    ) -> None:
        """Phase 3: _with_storage does not serialize concurrent calls via a lock.

        Two concurrent _with_storage calls must be able to interleave --
        both start before either finishes. A real DB is needed because
        _with_storage always opens RO per call.
        """
        db_path = tmp_path / '.axon' / 'kuzu'
        (tmp_path / '.axon').mkdir(parents=True)
        rw = KuzuBackend()
        rw.initialize(db_path, read_only=False)
        rw.close()

        set_repo_path(tmp_path)

        timeline: list[str] = []

        async def _probe(label: str) -> str:
            def _fn(st: object) -> str:
                timeline.append(f'{label}:start')
                time.sleep(0.05)
                timeline.append(f'{label}:end')
                return label

            return await _with_storage(_fn)

        await asyncio.gather(_probe('A'), _probe('B'))

        assert timeline not in (
            ['A:start', 'A:end', 'B:start', 'B:end'],
            ['B:start', 'B:end', 'A:start', 'A:end'],
        ), 'Expected concurrent execution but calls were serialized'


class TestSetRepoPath:
    def test_set_repo_path_stores_it(self, tmp_path: Path) -> None:
        """set_repo_path stores the repo path on _state."""
        set_repo_path(tmp_path)
        assert server_module._state.repo_path == tmp_path

    def test_resolve_db_path_uses_set_repo_path(self, tmp_path: Path) -> None:
        """_resolve_db_path derives db_path from set repo_path."""
        set_repo_path(tmp_path)
        assert _resolve_db_path() == tmp_path / '.axon' / 'kuzu'

    def test_resolve_db_path_defaults_to_cwd_when_unset(self) -> None:
        """_resolve_db_path returns cwd/.axon/kuzu when no path was injected."""
        expected = Path.cwd() / '.axon' / 'kuzu'
        assert _resolve_db_path() == expected


class TestCallToolSanitization:
    """call_tool catch-all must not leak internal exception details."""

    async def test_call_tool_sanitizes_unhandled_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Exception detail is replaced with a ref-linked generic message.

        The exception is triggered by monkeypatching _build_repo_context to
        raise so the catch-all in call_tool is exercised without needing a
        real DB.



        """
        async def _exploding_build(
            tool_name: str, arguments: dict, stack: object
        ) -> object:
            raise RuntimeError('/etc/secret/internal-path')

        with (
            patch.object(
                server_module, '_build_repo_context', _exploding_build
            ),
            caplog.at_level('ERROR'),
        ):
            result = await call_tool('axon_query', {'query': 'x'})

        text = result[0].text
        assert '/etc/secret/internal-path' not in text
        assert 'Internal error' in text
        assert 'ref ' in text

        ref = text.split('ref ')[1].split(')')[0]
        matching = [
            r
            for r in caplog.records
            if r.exc_info and '/etc/secret/internal-path' in str(r.exc_info[1])
        ]
        assert matching, 'Full exception must appear in a log record'
        assert matching[0].ref == ref


class TestCypherReadOnlyEnforcement:
    """axon_cypher is routed through a fresh read-only KuzuDB connection.

    Even when WRITE_KEYWORDS is bypassed, the DB layer rejects writes because
    the connection is opened with read_only=True.
    """

    async def test_db_layer_rejects_write_even_with_rw_path(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """DB-layer read-only blocks writes even when rw DB exists."""
        db_path = tmp_path / '.axon' / 'kuzu'
        (tmp_path / '.axon').mkdir(parents=True)
        rw_storage = KuzuBackend()
        rw_storage.initialize(db_path, read_only=False)
        rw_storage.close()

        set_repo_path(tmp_path)

        monkeypatch.setattr(tools_module, 'WRITE_KEYWORDS', re.compile('$^'))

        with caplog.at_level('ERROR'):
            result = await call_tool(
                'axon_cypher',
                {'query': "CREATE (n:Function {id: 'should_not_write'})"},
            )
        text = result[0].text

        assert 'should_not_write' not in text
        assert 'Cypher query failed' in text or 'Internal error' in text
        assert 'ref ' in text

        check_storage = KuzuBackend()
        check_storage.initialize(db_path, read_only=True)
        try:
            rows = check_storage.execute_raw(
                "MATCH (n:Function {id: 'should_not_write'}) RETURN n.id"
            )
            assert rows == []
        finally:
            check_storage.close()


class TestDispatchToolSignature:
    """Regression tests for _dispatch_tool signature and repo_path plumbing."""

    def test_list_repos_works_without_repo_path(
        self, tmp_path: Path, make_ctx: Any
    ) -> None:
        """axon_list_repos dispatches via _state resolver/drift_cache."""
        mock_storage = MagicMock()
        registry = tmp_path / 'registry'
        registry.mkdir()
        local = tmp_path / 'local'
        local.mkdir()
        resolver = RepoResolver(registry_dir=registry, local_repo_path=local)
        server_module._state.resolver = resolver
        server_module._state.drift_cache = DriftCache()
        result = _dispatch_tool('axon_list_repos', {}, make_ctx(mock_storage))
        assert isinstance(result, str)

    def test_test_impact_receives_repo_path(self, make_ctx: Any) -> None:
        """axon_test_impact passes ctx.repo_path through to handle_test_impact."""
        mock_storage = MagicMock()
        mock_storage.execute_raw.return_value = []
        repo = Path('/tmp/repo')
        captured: list[Path | None] = []

        with patch.object(
            server_module,
            'handle_test_impact',
            side_effect=lambda ctx, **kw: (
                captured.append(ctx.repo_path) or 'ok'
            ),
        ):
            _dispatch_tool(
                'axon_test_impact',
                {'diff': 'diff --git a/x.py b/x.py\n'},
                make_ctx(mock_storage, repo_path=repo),
            )

        assert captured == [repo]

    def test_test_impact_default_repo_path_is_none(
        self, make_ctx: Any
    ) -> None:
        """axon_test_impact receives ctx.repo_path=None when context has no path."""
        mock_storage = MagicMock()
        mock_storage.execute_raw.return_value = []
        captured: list[Path | None] = []

        with patch.object(
            server_module,
            'handle_test_impact',
            side_effect=lambda ctx, **kw: (
                captured.append(ctx.repo_path) or 'ok'
            ),
        ):
            _dispatch_tool(
                'axon_test_impact',
                {'diff': 'diff --git a/x.py b/x.py\n'},
                make_ctx(mock_storage, repo_path=None),
            )

        assert captured == [None]


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


class TestStateStorageUnusedByDispatch:
    """_state.storage is no longer present; dispatch opens per-call (Phase 3)."""

    @pytest.mark.asyncio
    async def test_state_storage_is_no_longer_read_by_dispatch(
        self, tmp_path: Path
    ) -> None:
        """Dispatch succeeds with no storage field on _state (Phase 3 removed it).

        _ServerState no longer has a storage field. Dispatch always opens RO
        per call via _build_repo_context.
        """
        registry = tmp_path / 'registry'
        local_repo = tmp_path / 'myrepo'
        (local_repo / '.axon').mkdir(parents=True)
        rw = KuzuBackend()
        rw.initialize(local_repo / '.axon' / 'kuzu', read_only=False)
        rw.close()
        _write_registry_entry(registry, 'myrepo', local_repo)

        server_module._state.repo_path = local_repo
        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        server_module._state.resolver = resolver
        server_module._state.drift_cache = DriftCache()
        server_module._state.local_slug = resolver.local().slug  # type: ignore[union-attr]

        result = await call_tool('axon_dead_code', {})

        assert result
        assert 'Internal error' not in result[0].text
