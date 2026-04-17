"""Tests for the _ServerState injection API in axon.mcp.server.

All tests treat the public API (set_storage / set_lock / set_db_path /
_resolve_db_path / _with_storage) as the observable surface. Only the
autouse reset fixture touches _state directly, for isolation purposes.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import axon.mcp.server as server_module
from axon.mcp.server import (
    _ServerState,
    _resolve_db_path,
    _with_storage,
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
