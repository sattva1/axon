"""Tests for FlushCoordinator and ChangeQueue from watcher_flush.py."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from watchfiles import Change

from axon.core.ingestion.watcher_flush import (
    ChangeQueue,
    FlushCoordinator,
    FlushPolicy,
    PendingChange,
)

# ---------------------------------------------------------------------------
# ChangeQueue tests
# ---------------------------------------------------------------------------


class TestChangeQueue:
    """ChangeQueue unit tests."""

    def test_coalesces_repeat_events_for_same_path(self) -> None:
        """Second push for the same path overwrites the first (last-event-wins)."""
        q = ChangeQueue()
        path = Path('/repo/src/foo.py')

        q.push(Change.modified, path)
        q.push(Change.deleted, path)

        assert len(q) == 1
        items = q.drain()
        assert items[0].change_type == Change.deleted

    def test_push_if_absent_preserves_newer_event(self) -> None:
        """push_if_absent is a no-op when a newer event already exists for the path."""
        q = ChangeQueue()
        path = Path('/repo/src/foo.py')

        q.push(Change.modified, path)
        existing = list(q._items.values())[0]

        stale = PendingChange(
            abs_path=path,
            change_type=Change.deleted,
            queued_at_monotonic=existing.queued_at_monotonic - 1.0,
        )
        result = q.push_if_absent(stale)

        assert result is False
        items = q.drain()
        assert items[0].change_type == Change.modified

    def test_first_added_age_tracks_oldest_pending(self) -> None:
        """first_added_age_seconds reflects the oldest queued event, not the newest."""
        q = ChangeQueue()
        path_a = Path('/repo/src/a.py')
        path_b = Path('/repo/src/b.py')

        q.push(Change.modified, path_a)
        first_age_after_a = q.first_added_age_seconds(time.monotonic())

        # Small sleep to ensure a measurable gap.
        time.sleep(0.05)
        q.push(Change.modified, path_b)
        age_after_both = q.first_added_age_seconds(time.monotonic())

        assert first_age_after_a is not None
        assert age_after_both is not None
        # The age after adding b should be >= age measured right after a.
        assert age_after_both >= first_age_after_a

    def test_drain_resets_first_added_at(self) -> None:
        """drain() clears the queue and resets _first_added_at to None."""
        q = ChangeQueue()
        q.push(Change.modified, Path('/repo/src/a.py'))

        assert q.first_added_age_seconds(time.monotonic()) is not None

        drained = q.drain()

        assert len(drained) == 1
        assert len(q) == 0
        assert q.first_added_age_seconds(time.monotonic()) is None
        assert q._last_push_at is None


# ---------------------------------------------------------------------------
# FlushCoordinator helpers
# ---------------------------------------------------------------------------


def _make_coordinator(
    tmp_path: Path, policy: FlushPolicy | None = None
) -> FlushCoordinator:
    """Build a FlushCoordinator with a fresh ChangeQueue and no-op git."""
    if policy is None:
        policy = FlushPolicy(
            interval_seconds=5.0, max_queue_size=200, quiet_period_seconds=1.0
        )
    queue = ChangeQueue()
    lock = asyncio.Lock()
    db_path = tmp_path / 'kuzu'
    with patch(
        'axon.core.ingestion.watcher_flush.get_head_sha', return_value=None
    ):
        coordinator = FlushCoordinator(
            repo_path=tmp_path,
            db_path=db_path,
            queue=queue,
            gitignore_patterns=None,
            policy=policy,
            global_lock=lock,
        )
    return coordinator


# ---------------------------------------------------------------------------
# FlushCoordinator trigger tests
# ---------------------------------------------------------------------------


class TestFlushCoordinatorTriggers:
    """Flush trigger condition tests for FlushCoordinator."""

    async def test_opens_rw_only_when_queue_nonempty(
        self, tmp_path: Path
    ) -> None:
        """Empty queue causes _maybe_flush to return without opening any backend."""
        coordinator = _make_coordinator(tmp_path)

        with patch.object(coordinator, '_flush_batch_blocking') as mock_flush:
            await coordinator._maybe_flush()

        mock_flush.assert_not_called()

    async def test_size_trigger_fires_at_max_queue_size(
        self, tmp_path: Path
    ) -> None:
        """Pushing max_queue_size distinct paths triggers a flush."""
        policy = FlushPolicy(
            interval_seconds=9999.0,
            max_queue_size=5,
            quiet_period_seconds=9999.0,
        )
        coordinator = _make_coordinator(tmp_path, policy)
        queue = coordinator._queue

        for i in range(5):
            queue.push(Change.modified, Path(f'/repo/src/file{i}.py'))

        with patch.object(coordinator, '_flush_batch_blocking') as mock_flush:
            await coordinator._maybe_flush()

        mock_flush.assert_called_once()

    async def test_interval_trigger_fires_after_interval(
        self, tmp_path: Path
    ) -> None:
        """After interval_seconds elapses since first push, flush fires."""
        policy = FlushPolicy(
            interval_seconds=0.05,
            max_queue_size=9999,
            quiet_period_seconds=9999.0,
        )
        coordinator = _make_coordinator(tmp_path, policy)
        coordinator._queue.push(Change.modified, Path('/repo/src/x.py'))

        time.sleep(0.1)

        with patch.object(coordinator, '_flush_batch_blocking') as mock_flush:
            await coordinator._maybe_flush()

        mock_flush.assert_called_once()

    async def test_quiet_trigger_fires_after_idle(
        self, tmp_path: Path
    ) -> None:
        """After quiet_period_seconds of no new pushes, flush fires."""
        policy = FlushPolicy(
            interval_seconds=9999.0,
            max_queue_size=9999,
            quiet_period_seconds=0.05,
        )
        coordinator = _make_coordinator(tmp_path, policy)
        coordinator._queue.push(Change.modified, Path('/repo/src/x.py'))

        time.sleep(0.1)

        with patch.object(coordinator, '_flush_batch_blocking') as mock_flush:
            await coordinator._maybe_flush()

        mock_flush.assert_called_once()

    async def test_starvation_trigger_at_max_dirty_age(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Starvation trigger fires when the queue age exceeds MAX_DIRTY_AGE."""
        low_age = 0.05
        monkeypatch.setattr(
            'axon.core.ingestion.watcher_flush.MAX_DIRTY_AGE', low_age
        )
        policy = FlushPolicy(
            interval_seconds=9999.0,
            max_queue_size=9999,
            quiet_period_seconds=9999.0,
        )
        coordinator = _make_coordinator(tmp_path, policy)
        coordinator._queue.push(Change.modified, Path('/repo/src/x.py'))

        time.sleep(low_age + 0.05)

        with patch.object(coordinator, '_flush_batch_blocking') as mock_flush:
            await coordinator._maybe_flush()

        mock_flush.assert_called_once()


# ---------------------------------------------------------------------------
# FlushCoordinator failure and re-queue tests
# ---------------------------------------------------------------------------


class TestFlushCoordinatorRequeue:
    """Flush failure and re-queue behaviour tests."""

    async def test_requeues_on_initialize_failure_with_push_if_absent(
        self, tmp_path: Path
    ) -> None:
        """When KuzuBackend.initialize raises, items go back via push_if_absent."""
        policy = FlushPolicy(
            interval_seconds=0.0, max_queue_size=1, quiet_period_seconds=0.0
        )
        coordinator = _make_coordinator(tmp_path, policy)
        path = Path('/repo/src/a.py')
        coordinator._queue.push(Change.modified, path)

        with patch(
            'axon.core.ingestion.watcher_flush.KuzuBackend'
        ) as mock_backend_cls:
            instance = mock_backend_cls.return_value
            instance.initialize.side_effect = RuntimeError('lock')

            push_if_absent_calls: list[PendingChange] = []

            def _track_push_if_absent(item: PendingChange) -> bool:
                push_if_absent_calls.append(item)
                return (
                    coordinator._queue._items.__setitem__(item.abs_path, item)
                    or True
                )

            # Patch push_if_absent on the queue instance directly.
            original_pia = coordinator._queue.push_if_absent
            coordinator._queue.push_if_absent = (  # type: ignore[method-assign]
                lambda item: _track_push_if_absent(item)
            )

            await coordinator._maybe_flush()

        assert len(push_if_absent_calls) == 1
        assert push_if_absent_calls[0].abs_path == path

        # Restore for cleanup safety.
        coordinator._queue.push_if_absent = original_pia  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# FlushCoordinator commit-transition tests
# ---------------------------------------------------------------------------


class TestFlushCoordinatorCommitTransition:
    """Coupling and drift-input update on commit-transition tests."""

    def _make_coordinator_with_mock_flush(
        self, tmp_path: Path, sha_sequence: list[str | None]
    ) -> tuple[FlushCoordinator, list[str]]:
        """Build a coordinator whose _flush_batch_blocking is real but patched.

        Returns the coordinator and an empty list that records coupling calls.
        """
        policy = FlushPolicy(
            interval_seconds=0.0, max_queue_size=1, quiet_period_seconds=0.0
        )
        queue = ChangeQueue()
        lock = asyncio.Lock()
        db_path = tmp_path / 'kuzu'
        sha_iter = iter(sha_sequence)
        with patch(
            'axon.core.ingestion.watcher_flush.get_head_sha',
            side_effect=sha_iter,
        ):
            coordinator = FlushCoordinator(
                repo_path=tmp_path,
                db_path=db_path,
                queue=queue,
                gitignore_patterns=None,
                policy=policy,
                global_lock=lock,
            )
        return coordinator

    async def test_runs_coupling_only_on_commit_transition(
        self, tmp_path: Path
    ) -> None:
        """Coupling fires only when HEAD SHA changes between flushes."""
        coupling_calls: list[bool] = []

        def fake_flush(batch: list[PendingChange]) -> None:
            coupling_calls.append(True)

        coordinator = self._make_coordinator_with_mock_flush(
            tmp_path, ['sha1', 'sha1']
        )
        coordinator._queue.push(Change.modified, Path('/repo/src/x.py'))

        # Patch get_head_sha to return the same SHA as _last_known_commit.
        with (
            patch(
                'axon.core.ingestion.watcher_flush.get_head_sha',
                return_value='sha1',
            ),
            patch(
                'axon.core.ingestion.watcher_flush.run_incremental_global_phases'
            ) as mock_inc,
            patch(
                'axon.core.ingestion.watcher_flush.reindex_files'
            ) as mock_ri,
            patch('axon.core.ingestion.watcher_flush.update_meta'),
            patch(
                'axon.core.ingestion.watcher_flush.KuzuBackend'
            ) as mock_backend_cls,
        ):
            mock_ri.return_value = (1, {'src/x.py'})
            instance = mock_backend_cls.return_value
            instance.get_file_index.return_value = {}

            await coordinator._maybe_flush()

        # same SHA: run_coupling=False
        mock_inc.assert_called_once()
        _, kwargs = mock_inc.call_args
        assert kwargs.get('run_coupling') is False

        # Now with a different SHA.
        coordinator._queue.push(Change.modified, Path('/repo/src/y.py'))
        coordinator._last_known_commit = 'sha1'

        with (
            patch(
                'axon.core.ingestion.watcher_flush.get_head_sha',
                return_value='sha2',
            ),
            patch(
                'axon.core.ingestion.watcher_flush.run_incremental_global_phases'
            ) as mock_inc2,
            patch(
                'axon.core.ingestion.watcher_flush.reindex_files'
            ) as mock_ri2,
            patch('axon.core.ingestion.watcher_flush.update_meta'),
            patch(
                'axon.core.ingestion.watcher_flush.compute_drift_inputs',
                return_value={},
            ),
            patch(
                'axon.core.ingestion.watcher_flush.KuzuBackend'
            ) as mock_backend_cls2,
        ):
            mock_ri2.return_value = (1, {'src/y.py'})
            instance2 = mock_backend_cls2.return_value
            instance2.get_file_index.return_value = {}

            await coordinator._maybe_flush()

        mock_inc2.assert_called_once()
        _, kwargs2 = mock_inc2.call_args
        assert kwargs2.get('run_coupling') is True

    async def test_updates_drift_inputs_only_on_commit_transition(
        self, tmp_path: Path
    ) -> None:
        """compute_drift_inputs is called only when the commit changes."""
        coordinator = self._make_coordinator_with_mock_flush(
            tmp_path, ['sha1', 'sha1']
        )
        coordinator._queue.push(Change.modified, Path('/repo/src/z.py'))
        coordinator._last_known_commit = 'sha1'

        with (
            patch(
                'axon.core.ingestion.watcher_flush.get_head_sha',
                return_value='sha1',
            ),
            patch(
                'axon.core.ingestion.watcher_flush.run_incremental_global_phases'
            ),
            patch(
                'axon.core.ingestion.watcher_flush.reindex_files'
            ) as mock_ri,
            patch('axon.core.ingestion.watcher_flush.update_meta'),
            patch(
                'axon.core.ingestion.watcher_flush.compute_drift_inputs'
            ) as mock_drift,
            patch(
                'axon.core.ingestion.watcher_flush.KuzuBackend'
            ) as mock_backend_cls,
        ):
            mock_ri.return_value = (1, {'src/z.py'})
            mock_backend_cls.return_value.get_file_index.return_value = {}

            await coordinator._maybe_flush()

        mock_drift.assert_not_called()

    async def test_writes_last_incremental_at_exactly_once_per_flush(
        self, tmp_path: Path
    ) -> None:
        """update_meta is called exactly once with last_incremental_at per flush."""
        coordinator = self._make_coordinator_with_mock_flush(
            tmp_path, ['sha1', 'sha1']
        )
        # Push several events (one distinct path each).
        for i in range(3):
            coordinator._queue.push(
                Change.modified, Path(f'/repo/src/f{i}.py')
            )
        coordinator._last_known_commit = 'sha1'

        update_meta_calls: list[dict] = []

        def tracking_update_meta(repo_root: Path, **fields) -> None:
            update_meta_calls.append(fields)

        with (
            patch(
                'axon.core.ingestion.watcher_flush.get_head_sha',
                return_value='sha1',
            ),
            patch(
                'axon.core.ingestion.watcher_flush.run_incremental_global_phases'
            ),
            patch(
                'axon.core.ingestion.watcher_flush.reindex_files'
            ) as mock_ri,
            patch(
                'axon.core.ingestion.watcher_flush.update_meta',
                side_effect=tracking_update_meta,
            ),
            patch(
                'axon.core.ingestion.watcher_flush.KuzuBackend'
            ) as mock_backend_cls,
        ):
            mock_ri.return_value = (3, {'src/f0.py', 'src/f1.py', 'src/f2.py'})
            mock_backend_cls.return_value.get_file_index.return_value = {}

            await coordinator._maybe_flush()

        incremental_writes = [
            c for c in update_meta_calls if 'last_incremental_at' in c
        ]
        assert len(incremental_writes) == 1


# ---------------------------------------------------------------------------
# FlushCoordinator on_commit_transition callback tests
# ---------------------------------------------------------------------------


class TestOnCommitTransitionCallback:
    """on_commit_transition callback wiring in _flush_batch_blocking."""

    def _make_coordinator_with_callback(
        self, tmp_path: Path, callback: object, initial_sha: str = 'sha1'
    ) -> FlushCoordinator:
        """Build a FlushCoordinator with a callback and a known initial commit."""
        policy = FlushPolicy(
            interval_seconds=0.0, max_queue_size=1, quiet_period_seconds=0.0
        )
        queue = ChangeQueue()
        lock = asyncio.Lock()
        db_path = tmp_path / 'kuzu'
        with patch(
            'axon.core.ingestion.watcher_flush.get_head_sha',
            return_value=initial_sha,
        ):
            coordinator = FlushCoordinator(
                repo_path=tmp_path,
                db_path=db_path,
                queue=queue,
                gitignore_patterns=None,
                policy=policy,
                global_lock=lock,
                on_commit_transition=callback,
            )
        return coordinator

    async def test_called_when_commit_transition_true(
        self, tmp_path: Path
    ) -> None:
        """Callback is invoked once with repo_path on a commit-transition flush."""
        callback = MagicMock()
        coordinator = self._make_coordinator_with_callback(
            tmp_path, callback, initial_sha='sha1'
        )
        coordinator._queue.push(Change.modified, Path('/repo/src/x.py'))
        # Ensure _last_known_commit differs from the current HEAD to force a
        # commit transition.
        coordinator._last_known_commit = 'sha1'

        with (
            patch(
                'axon.core.ingestion.watcher_flush.get_head_sha',
                return_value='sha2',
            ),
            patch(
                'axon.core.ingestion.watcher_flush.run_incremental_global_phases'
            ),
            patch(
                'axon.core.ingestion.watcher_flush.reindex_files'
            ) as mock_ri,
            patch('axon.core.ingestion.watcher_flush.update_meta'),
            patch(
                'axon.core.ingestion.watcher_flush.compute_drift_inputs',
                return_value={},
            ),
            patch(
                'axon.core.ingestion.watcher_flush.KuzuBackend'
            ) as mock_backend_cls,
        ):
            mock_ri.return_value = (1, {'src/x.py'})
            mock_backend_cls.return_value.get_file_index.return_value = {}

            await coordinator._maybe_flush()

        callback.assert_called_once_with(tmp_path)

    async def test_not_called_when_commit_transition_false(
        self, tmp_path: Path
    ) -> None:
        """Callback is not invoked when the HEAD sha is unchanged."""
        callback = MagicMock()
        coordinator = self._make_coordinator_with_callback(
            tmp_path, callback, initial_sha='sha1'
        )
        coordinator._queue.push(Change.modified, Path('/repo/src/x.py'))
        coordinator._last_known_commit = 'sha1'

        with (
            patch(
                'axon.core.ingestion.watcher_flush.get_head_sha',
                return_value='sha1',
            ),
            patch(
                'axon.core.ingestion.watcher_flush.run_incremental_global_phases'
            ),
            patch(
                'axon.core.ingestion.watcher_flush.reindex_files'
            ) as mock_ri,
            patch('axon.core.ingestion.watcher_flush.update_meta'),
            patch(
                'axon.core.ingestion.watcher_flush.KuzuBackend'
            ) as mock_backend_cls,
        ):
            mock_ri.return_value = (1, {'src/x.py'})
            mock_backend_cls.return_value.get_file_index.return_value = {}

            await coordinator._maybe_flush()

        callback.assert_not_called()

    async def test_exception_swallowed_and_update_meta_still_runs(
        self, tmp_path: Path
    ) -> None:
        """Callback exception is swallowed; last_incremental_at is still written."""
        callback = MagicMock(side_effect=RuntimeError('boom'))
        coordinator = self._make_coordinator_with_callback(
            tmp_path, callback, initial_sha='sha1'
        )
        coordinator._queue.push(Change.modified, Path('/repo/src/x.py'))
        coordinator._last_known_commit = 'sha1'

        update_meta_calls: list[dict] = []

        def tracking_update_meta(repo_root: Path, **fields) -> None:
            update_meta_calls.append(fields)

        with (
            patch(
                'axon.core.ingestion.watcher_flush.get_head_sha',
                return_value='sha2',
            ),
            patch(
                'axon.core.ingestion.watcher_flush.run_incremental_global_phases'
            ),
            patch(
                'axon.core.ingestion.watcher_flush.reindex_files'
            ) as mock_ri,
            patch(
                'axon.core.ingestion.watcher_flush.update_meta',
                side_effect=tracking_update_meta,
            ),
            patch(
                'axon.core.ingestion.watcher_flush.compute_drift_inputs',
                return_value={},
            ),
            patch(
                'axon.core.ingestion.watcher_flush.KuzuBackend'
            ) as mock_backend_cls,
        ):
            mock_ri.return_value = (1, {'src/x.py'})
            mock_backend_cls.return_value.get_file_index.return_value = {}

            # Must not raise even though callback raises RuntimeError.
            await coordinator._maybe_flush()

        incremental_writes = [
            c for c in update_meta_calls if 'last_incremental_at' in c
        ]
        assert len(incremental_writes) == 1
