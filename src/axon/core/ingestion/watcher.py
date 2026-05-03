"""Watch mode for Axon -- re-indexes on file changes.

Uses ``watchfiles`` (Rust-backed) for efficient file system monitoring.
Changes accumulate in a ChangeQueue; a FlushCoordinator opens the RW
backend only for the duration of each flush, releasing the file lock
between bursts so concurrent readers (MCP, axon analyze) can acquire it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

import watchfiles
from watchfiles import Change

from axon.config.ignore import load_gitignore
from axon.core.ingestion.reindex import (
    run_full_global_phases,
)
from axon.core.ingestion.watcher_flush import (
    ChangeQueue,
    FlushCoordinator,
    FlushPolicy,
)
from axon.core.repos import _FLUSH_OPEN_RETRIES, _FLUSH_OPEN_RETRY_DELAY
from axon.core.storage.kuzu_backend import KuzuBackend

logger = logging.getLogger(__name__)


async def _watchfiles_producer(
    repo_path: Path, queue: ChangeQueue, stop_event: asyncio.Event | None
) -> None:
    """Forward watchfiles events into *queue*.

    Args:
        repo_path: Directory to watch recursively.
        queue: Shared ChangeQueue owned by the coordinator.
        stop_event: When set, watchfiles.awatch returns and this coroutine exits.
    """
    logger.info('Watching %s for changes...', repo_path)
    async for changes in watchfiles.awatch(
        repo_path,
        rust_timeout=500,
        yield_on_timeout=True,
        stop_event=stop_event,
    ):
        path_to_change: dict[str, Change] = {}
        for change_type, path_str in changes:
            path_to_change[path_str] = change_type
        for path_str, change_type in path_to_change.items():
            queue.push(change_type, Path(path_str))


async def _periodic_refresh(
    repo_path: Path,
    db_path: Path,
    interval_seconds: int,
    global_lock: asyncio.Lock,
    stop_event: asyncio.Event,
) -> None:
    """Periodically run full global phases regardless of file changes.

    Opens the RW backend for each refresh and closes it immediately
    after. Serialized with flush ticks via global_lock.

    Args:
        repo_path: Root of the repository.
        db_path: Path to the KuzuDB directory.
        interval_seconds: Sleep duration between refreshes.
        global_lock: asyncio.Lock shared with the FlushCoordinator.
        stop_event: When set, exits after the current sleep.
    """
    while not stop_event.is_set():
        await asyncio.sleep(interval_seconds)
        if stop_event.is_set():
            break
        async with global_lock:

            def _refresh_blocking() -> None:
                backend = KuzuBackend()
                backend.initialize(
                    db_path,
                    max_retries=_FLUSH_OPEN_RETRIES,
                    retry_delay=_FLUSH_OPEN_RETRY_DELAY,
                )
                try:
                    run_full_global_phases(repo_path, backend)
                finally:
                    backend.close()

            try:
                await asyncio.to_thread(_refresh_blocking)
            except Exception:
                logger.warning('Periodic global refresh failed', exc_info=True)


async def watch_repo(
    repo_path: Path,
    db_path: Path,
    *,
    stop_event: asyncio.Event | None = None,
    flush_policy: FlushPolicy | None = None,
    global_refresh_interval_seconds: int | None = None,
    on_commit_transition: Callable[[Path], None] | None = None,
) -> None:
    """Main watch loop -- monitor files and re-index on changes.

    File-system events accumulate in a ChangeQueue. The FlushCoordinator
    drains the queue on configurable triggers and opens the RW KuzuBackend
    only for the duration of each flush. This releases the file lock
    between bursts so axon analyze and MCP reads can proceed concurrently.

    Args:
        repo_path: Root of the repository to watch.
        db_path: Path to the KuzuDB directory.
        stop_event: Setting this event stops all tasks.
        flush_policy: Override default batching thresholds.
        global_refresh_interval_seconds: When set, a background task runs
            full global phases at this interval.
        on_commit_transition: Optional callback forwarded to FlushCoordinator.
            Called after each successful commit-transition flush.
    """
    policy = flush_policy or FlushPolicy()
    queue = ChangeQueue()
    global_lock = asyncio.Lock()
    _stop = stop_event or asyncio.Event()

    coordinator = FlushCoordinator(
        repo_path,
        db_path,
        queue,
        load_gitignore(repo_path),
        policy,
        global_lock,
        on_commit_transition,
    )

    async with asyncio.TaskGroup() as tg:
        tg.create_task(coordinator.run_until_stopped(_stop))
        tg.create_task(_watchfiles_producer(repo_path, queue, stop_event))
        if global_refresh_interval_seconds:
            tg.create_task(
                _periodic_refresh(
                    repo_path,
                    db_path,
                    global_refresh_interval_seconds,
                    global_lock,
                    _stop,
                )
            )

    logger.info('Watch stopped.')
