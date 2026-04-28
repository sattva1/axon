"""Tests for the _ServerState injection API in axon.mcp.server.

All tests treat the public API (set_storage / set_lock / set_db_path /
_resolve_db_path / _with_storage) as the observable surface. Only the
autouse reset fixture touches _state directly, for isolation purposes.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import axon.mcp.server as server_module
import axon.mcp.tools as tools_module
from axon.core.drift import DriftCache
from axon.core.repos import RepoResolver
from axon.core.storage.kuzu_backend import KuzuBackend
from axon.mcp.server import (
    _dispatch_tool,
    _resolve_db_path,
    _ServerState,
    _with_storage,
    call_tool,
    set_db_path,
    set_lock,
    set_storage,
)


@pytest.fixture(autouse=True)
def reset_state() -> None:
    """Reset module-level _state before each test for isolation."""
    server_module._state = _ServerState()
    yield
    server_module._state = _ServerState()


class TestSetStorage:
    async def test_makes_handler_use_injected_backend(self) -> None:
        """set_storage causes _with_storage to pass the mock to the probe."""
        mock_storage = MagicMock()
        set_storage(mock_storage)

        seen: list[object] = []
        await _with_storage(lambda st: seen.append(st) or 'ok')

        assert len(seen) == 1
        assert seen[0] is mock_storage


class TestSetLock:
    async def test_serializes_concurrent_calls(self) -> None:
        """set_lock causes _with_storage calls to serialize through the lock.

        _with_storage calls fn synchronously inside asyncio.to_thread, so the
        probe must be a plain sync callable. A short sleep in the thread makes
        the interleaving observable: without the lock the two threads would run
        concurrently and the timeline would interleave; with the lock held
        across the thread call the first call completes before the second even
        starts (from the event-loop's perspective the lock is released only
        after to_thread returns).
        """
        lock = asyncio.Lock()
        set_storage(MagicMock())
        set_lock(lock)

        timeline: list[str] = []

        async def _probe(label: str) -> str:
            def _fn(st: object) -> str:
                timeline.append(f'{label}:start')
                time.sleep(0.05)
                timeline.append(f'{label}:end')
                return label

            return await _with_storage(_fn)

        await asyncio.gather(_probe('A'), _probe('B'))

        # With a lock the calls must not interleave: one must finish before the
        # other starts.
        assert timeline in (
            ['A:start', 'A:end', 'B:start', 'B:end'],
            ['B:start', 'B:end', 'A:start', 'A:end'],
        )


class TestSetDbPath:
    def test_overrides_default(self, tmp_path: Path) -> None:
        """set_db_path makes _resolve_db_path return the injected path."""
        custom = tmp_path / 'custom' / 'db'
        set_db_path(custom)

        assert _resolve_db_path() == custom

    def test_resolve_db_path_defaults_to_cwd_when_unset(self) -> None:
        """_resolve_db_path returns cwd/.axon/kuzu when no path was injected."""
        expected = Path.cwd() / '.axon' / 'kuzu'
        assert _resolve_db_path() == expected


class TestCallToolSanitization:
    """call_tool catch-all must not leak internal exception details."""

    async def test_call_tool_sanitizes_unhandled_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Exception detail is replaced with a ref-linked generic message."""
        mock_storage = MagicMock()
        mock_storage.fts_search.side_effect = RuntimeError(
            '/etc/secret/internal-path'
        )
        set_storage(mock_storage)

        with caplog.at_level('ERROR'):
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
    the connection is opened with read_only=True regardless of any injected
    read-write storage backend.
    """

    async def test_db_layer_rejects_write_even_if_rw_storage_injected(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """DB-layer read-only blocks writes even when rw storage is injected."""
        # Arrange: initialize a real KuzuDB with schema so _open_storage can
        # open a read-only connection against it. Do NOT pre-create the
        # directory - KuzuDB creates its own DB files at the given path.
        db_path = tmp_path / '.axon' / 'kuzu'
        (tmp_path / '.axon').mkdir(parents=True)
        rw_storage = KuzuBackend()
        rw_storage.initialize(db_path, read_only=False)

        # Inject the rw storage and point _open_storage at the same path.
        set_storage(rw_storage)
        set_db_path(db_path)

        # Bypass the user-friendly WRITE_KEYWORDS regex denylist so the query
        # reaches the DB layer. The DB layer is the definitive enforcement point.
        # Patch on tools_module because tools.py imports WRITE_KEYWORDS by name.
        monkeypatch.setattr(tools_module, 'WRITE_KEYWORDS', re.compile('$^'))

        # Act: submit a write query via the public call_tool interface.
        with caplog.at_level('ERROR'):
            result = await call_tool(
                'axon_cypher',
                {'query': "CREATE (n:Function {id: 'should_not_write'})"},
            )
        text = result[0].text

        # Assert: write was rejected - the error is surfaced as a sanitized
        # message with a ref id (either from handle_cypher or the catch-all).
        assert 'should_not_write' not in text
        assert 'Cypher query failed' in text or 'Internal error' in text
        assert 'ref ' in text

        # Verify the node was never persisted - re-open a fresh read-only
        # connection and confirm no matching node exists.
        check_storage = KuzuBackend()
        check_storage.initialize(db_path, read_only=True)
        try:
            rows = check_storage.execute_raw(
                "MATCH (n:Function {id: 'should_not_write'}) RETURN n.id"
            )
            assert rows == []
        finally:
            check_storage.close()
            rw_storage.close()


class TestDispatchToolSignature:
    """Regression tests for _dispatch_tool signature and repo_path plumbing.

    Phase 3 changed the signature from _dispatch_tool(name, args, storage,
    repo_path=...) to _dispatch_tool(name, args, ctx: RepoContext). These
    tests use make_ctx to build the context and verify the handler receives
    ctx.repo_path correctly.
    """

    def test_list_repos_works_without_repo_path(
        self, tmp_path: Path, make_ctx: Any
    ) -> None:
        """axon_list_repos dispatches via _state resolver/drift_cache."""
        mock_storage = MagicMock()
        # Inject the multi-repo state that _dispatch_tool reads directly.
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

        # Patch in server_module because that's where the name is bound after
        # the 'from axon.mcp.tools import handle_test_impact' import.
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


class TestSetStorageRepoPath:
    """set_storage repo_path plumbing into _ServerState."""

    def test_set_storage_without_repo_path_defaults_none(self) -> None:
        """Calling set_storage without repo_path leaves _state.repo_path as None."""
        mock_storage = MagicMock()
        set_storage(mock_storage)
        assert server_module._state.repo_path is None

    def test_set_storage_with_repo_path_stores_it(self) -> None:
        """Calling set_storage with repo_path stores it on _state."""
        mock_storage = MagicMock()
        repo = Path('/some/repo')
        set_storage(mock_storage, repo)
        assert server_module._state.repo_path == repo
