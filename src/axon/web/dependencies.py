"""FastAPI dependency providers for per-request KuzuBackend lifetimes.

All web route handlers that need storage should depend on storage_ro (for
reads) or storage_rw (for writes). This is the single source of truth for
how the web layer opens and closes KuzuBackend instances.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Request

from axon.core.repos import (
    _DISPATCH_OPEN_RETRIES,
    _DISPATCH_OPEN_RETRY_DELAY,
    _FLUSH_OPEN_RETRIES,
    _FLUSH_OPEN_RETRY_DELAY,
)
from axon.core.storage.kuzu_backend import KuzuBackend


def _open_blocking(
    db_path: Path,
    *,
    read_only: bool,
    max_retries: int,
    retry_delay: float,
) -> KuzuBackend:
    """Open a KuzuBackend synchronously.

    Args:
        db_path: Path to the KuzuDB directory.
        read_only: Whether to open in read-only mode.
        max_retries: Number of retries on lock contention.
        retry_delay: Seconds to wait between retries.

    Returns:
        Initialised KuzuBackend.
    """
    backend = KuzuBackend()
    backend.initialize(
        db_path,
        read_only=read_only,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    return backend


def _safe_close(backend: KuzuBackend | None) -> None:
    """Close a KuzuBackend, swallowing any exception.

    Args:
        backend: Backend to close, or None (no-op).
    """
    if backend is None:
        return
    try:
        backend.close()
    except Exception:
        pass


@asynccontextmanager
async def open_ro_backend(db_path: Path) -> AsyncIterator[KuzuBackend]:
    """Async context manager that opens a read-only backend and closes on exit.

    Args:
        db_path: Path to the KuzuDB directory.

    Yields:
        Initialised read-only KuzuBackend.
    """
    backend = await asyncio.to_thread(
        _open_blocking,
        db_path,
        read_only=True,
        max_retries=_DISPATCH_OPEN_RETRIES,
        retry_delay=_DISPATCH_OPEN_RETRY_DELAY,
    )
    try:
        yield backend
    finally:
        await asyncio.to_thread(_safe_close, backend)


async def storage_ro(request: Request) -> AsyncIterator[KuzuBackend]:
    """FastAPI dependency: open a read-only KuzuBackend for the current request.

    Args:
        request: The current FastAPI request (used to access app.state.db_path).

    Yields:
        Initialised read-only KuzuBackend.
    """
    db_path: Path = request.app.state.db_path
    async with open_ro_backend(db_path) as backend:
        yield backend


async def storage_rw(request: Request) -> AsyncIterator[KuzuBackend]:
    """FastAPI dependency: open a read-write KuzuBackend for the current request.

    Uses the patient retry policy because write opens compete with the watcher.

    Args:
        request: The current FastAPI request (used to access app.state.db_path).

    Yields:
        Initialised read-write KuzuBackend.
    """
    db_path: Path = request.app.state.db_path
    backend = await asyncio.to_thread(
        _open_blocking,
        db_path,
        read_only=False,
        max_retries=_FLUSH_OPEN_RETRIES,
        retry_delay=_FLUSH_OPEN_RETRY_DELAY,
    )
    try:
        yield backend
    finally:
        await asyncio.to_thread(_safe_close, backend)
