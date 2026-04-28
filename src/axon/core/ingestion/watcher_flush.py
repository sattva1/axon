"""Batched-flush coordinator for the file watcher.

Provides the producer-consumer infrastructure that lets the watcher release
the Kuzu write lock between change bursts.  The coordinator drains the
ChangeQueue on configurable triggers (interval, size, quiet period,
starvation) and opens the RW backend only for the duration of one flush.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from watchfiles import Change

from axon.core.drift import compute_drift_inputs
from axon.core.ingestion.reindex import (
    get_head_sha,
    reindex_files,
    run_incremental_global_phases,
)
from axon.core.meta import now_iso, update_meta
from axon.core.repos import _FLUSH_OPEN_RETRIES, _FLUSH_OPEN_RETRY_DELAY
from axon.core.storage.kuzu_backend import KuzuBackend

logger = logging.getLogger(__name__)

# Fairness bound: starvation trigger. SSOT is here (review N3).
MAX_DIRTY_AGE: Final[float] = 60.0


@dataclass(frozen=True)
class FlushPolicy:
    """Single source of truth for batch trigger thresholds.

    quiet_period_seconds doubles as the coordinator's poll cadence -
    finer debouncing requires faster polling. Intentional coupling.
    """

    interval_seconds: float = 5.0
    max_queue_size: int = 200
    quiet_period_seconds: float = 1.0


@dataclass
class PendingChange:
    """A single pending file-system event waiting to be flushed."""

    abs_path: Path
    change_type: Change
    queued_at_monotonic: float


class ChangeQueue:
    """Bounded coalescing queue of pending FS events.

    Coalescing rule: last event for a given path wins (push). For re-queue
    after a failed flush, push_if_absent preserves any newer event that
    arrived during the flush attempt.

    Thread-safety invariant: ALL mutations (push, push_if_absent, drain)
    must occur on the asyncio event loop. drain() is an atomic
    snapshot-and-clear; the returned list may be safely consumed inside
    asyncio.to_thread by the worker that received it.
    """

    def __init__(self) -> None:
        self._items: dict[Path, PendingChange] = {}
        self._first_added_at: float | None = None
        self._last_push_at: float | None = None

    def push(self, change: Change, path: Path) -> None:
        """Add or overwrite the event for *path* (last event wins).

        Args:
            change: The type of change that occurred.
            path: Absolute path of the changed file.
        """
        now = time.monotonic()
        if path not in self._items:
            if self._first_added_at is None:
                self._first_added_at = now
        self._items[path] = PendingChange(
            abs_path=path,
            change_type=change,
            queued_at_monotonic=now,
        )
        self._last_push_at = now

    def push_if_absent(self, change: PendingChange) -> bool:
        """Add *change* only when no event for that path is already queued.

        Preserves a newer event that arrived during a failed flush attempt.

        Args:
            change: PendingChange to conditionally re-queue.

        Returns:
            True when the item was inserted, False when a newer entry existed.
        """
        if change.abs_path in self._items:
            return False
        now = time.monotonic()
        if self._first_added_at is None:
            self._first_added_at = now
        self._items[change.abs_path] = change
        if self._last_push_at is None:
            self._last_push_at = now
        return True

    def __len__(self) -> int:
        return len(self._items)

    def drain(self) -> list[PendingChange]:
        """Atomically snapshot and clear the queue.

        Resets first_added_at so the next push starts a fresh age window.

        Returns:
            All pending changes at the time of the call.
        """
        items = list(self._items.values())
        self._items.clear()
        self._first_added_at = None
        self._last_push_at = None
        return items

    def first_added_age_seconds(self, now: float) -> float | None:
        """Return seconds since the oldest pending change was queued.

        Args:
            now: Current monotonic clock value.

        Returns:
            Age in seconds, or None when the queue is empty.
        """
        if self._first_added_at is None:
            return None
        return now - self._first_added_at

    @property
    def _last_push_age_seconds(self) -> float | None:
        """Seconds since the most recent push, or None when never pushed."""
        if self._last_push_at is None:
            return None
        return time.monotonic() - self._last_push_at


class FlushCoordinator:
    """Drives periodic batched flushes of accumulated FS events.

    Checks trigger conditions every policy.quiet_period_seconds and opens
    the RW KuzuBackend only for the duration of a flush. The asyncio
    global_lock serializes flush ticks vs. the optional periodic global
    refresh task.

    Args:
        repo_path: Root of the repository being watched.
        db_path: Path to the KuzuDB directory.
        queue: Shared change queue populated by the watchfiles producer.
        gitignore_patterns: Patterns loaded from .gitignore.
        policy: Trigger thresholds.
        global_lock: asyncio.Lock shared with _periodic_refresh.
    """

    def __init__(
        self,
        repo_path: Path,
        db_path: Path,
        queue: ChangeQueue,
        gitignore_patterns: list[str] | None,
        policy: FlushPolicy,
        global_lock: asyncio.Lock,
    ) -> None:
        self._repo_path = repo_path
        self._db_path = db_path
        self._queue = queue
        self._gitignore_patterns = gitignore_patterns
        self._policy = policy
        self._global_lock = global_lock
        self._last_flush_at: float = 0.0
        self._last_known_commit: str | None = get_head_sha(repo_path)

    async def run_until_stopped(self, stop_event: asyncio.Event) -> None:
        """Poll for trigger conditions until stop_event is set.

        Args:
            stop_event: Setting this event causes the coordinator to exit.
        """
        logger.info('Flush coordinator started for %s', self._repo_path)
        while not stop_event.is_set():
            await asyncio.sleep(self._policy.quiet_period_seconds)
            await self._maybe_flush()
        logger.info('Flush coordinator stopped.')

    async def _maybe_flush(self) -> None:
        """Check trigger conditions and flush when any fires.

        Holds the asyncio global_lock for the entire tick to serialize flush
        vs. periodic global-refresh task. The OS-level Kuzu file lock is
        released the moment backend.close() runs inside _flush_batch_blocking.
        The asyncio lock outlives the OS lock by microseconds - no other
        process is blocked once the OS lock is freed.
        """
        if len(self._queue) == 0:
            return

        now = time.monotonic()
        first_age = self._queue.first_added_age_seconds(now)
        if first_age is None:
            return

        quiet_age = self._queue._last_push_age_seconds
        size_trigger = len(self._queue) >= self._policy.max_queue_size
        interval_trigger = first_age >= self._policy.interval_seconds
        quiet_trigger = (
            quiet_age is not None
            and quiet_age >= self._policy.quiet_period_seconds
        )
        starvation_trigger = first_age >= MAX_DIRTY_AGE

        if not (
            size_trigger or interval_trigger or quiet_trigger or starvation_trigger
        ):
            return

        async with self._global_lock:
            batch = self._queue.drain()
            if not batch:
                return
            try:
                await asyncio.to_thread(self._flush_batch_blocking, batch)
            except Exception:
                logger.warning(
                    'Flush failed (%d events); re-queueing',
                    len(batch),
                    exc_info=True,
                )
                for item in batch:
                    # Don't overwrite a newer event that arrived during the flush.
                    self._queue.push_if_absent(item)

    def _flush_batch_blocking(self, batch: list[PendingChange]) -> None:
        """Open RW, reindex the batch, run global phases, then close.

        This is the only place that opens a RW KuzuBackend. It uses the
        patient retry policy so transient lock contention from concurrent
        reads is absorbed without propagating to the re-queue path.

        Args:
            batch: Drained list of pending changes to process.
        """
        changes = [
            (item.change_type, item.abs_path) for item in batch
        ]
        backend = KuzuBackend()
        backend.initialize(
            self._db_path,
            max_retries=_FLUSH_OPEN_RETRIES,
            retry_delay=_FLUSH_OPEN_RETRY_DELAY,
        )
        try:
            count, reindexed = reindex_files(
                changes,
                self._repo_path,
                backend,
                self._gitignore_patterns,
            )

            if not reindexed:
                return

            current_commit = get_head_sha(self._repo_path)
            commit_transition = current_commit != self._last_known_commit

            run_incremental_global_phases(
                backend,
                self._repo_path,
                reindexed,
                run_coupling=commit_transition,
            )

            if commit_transition:
                update_meta(
                    self._repo_path,
                    **compute_drift_inputs(
                        self._repo_path,
                        list(backend.get_file_index().keys()),
                    ),
                )
                self._last_known_commit = current_commit

            # Single authoritative write of last_incremental_at per flush (Major #1).
            update_meta(self._repo_path, last_incremental_at=now_iso())

            logger.info(
                'Flush complete: %d path(s), commit_transition=%s',
                count,
                commit_transition,
            )
        finally:
            backend.close()
