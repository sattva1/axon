"""Tests for axon.core.drift - compute_drift_inputs, probe_drift, DriftCache."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import axon.core.drift as drift_module
from axon.core.drift import (
    DRIFT_CACHE_TTL_SECONDS,
    DRIFT_INDEXED_DIRS_CAP,
    LIVE_WATCHER_RECENCY_SECONDS,
    STALE_MAJOR_COMMIT_THRESHOLD,
    DriftCache,
    DriftLevel,
    compute_drift_inputs,
    probe_drift,
)
from axon.core.meta import SentinelEntry, load_meta, update_meta

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

_GIT_ENV = {
    'GIT_AUTHOR_NAME': 'Test',
    'GIT_AUTHOR_EMAIL': 'test@example.com',
    'GIT_COMMITTER_NAME': 'Test',
    'GIT_COMMITTER_EMAIL': 'test@example.com',
}
_GIT_COMMON = [
    '-c',
    'commit.gpgsign=false',
    '-c',
    'user.name=Test',
    '-c',
    'user.email=test@example.com',
]


def _git(
    args: list[str], cwd: Path, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a git command inside *cwd*."""
    env = {**os.environ, **_GIT_ENV}
    return subprocess.run(
        ['git'] + _GIT_COMMON + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
        check=check,
    )


def _init_git_repo(path: Path) -> str:
    """Init a git repo, add all files and commit. Return HEAD sha."""
    _git(['init'], path)
    _git(['add', '.'], path)
    _git(['commit', '-m', 'init'], path)
    return _git(['rev-parse', 'HEAD'], path).stdout.strip()


def _commit_file(repo: Path, rel_path: str, content: str) -> str:
    """Write *rel_path* and commit. Return new HEAD sha."""
    target = repo / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding='utf-8')
    _git(['add', rel_path], repo)
    _git(['commit', '-m', f'update {rel_path}'], repo)
    return _git(['rev-parse', 'HEAD'], repo).stdout.strip()


def _write_host_json(repo_path: Path, last_incremental_at: str) -> None:
    """Write a minimal host.json inside .axon/."""
    axon_dir = repo_path / '.axon'
    axon_dir.mkdir(parents=True, exist_ok=True)
    (axon_dir / 'host.json').write_text(
        json.dumps({'last_incremental_at': last_incremental_at}),
        encoding='utf-8',
    )


# ---------------------------------------------------------------------------
# compute_drift_inputs
# ---------------------------------------------------------------------------


class TestComputeDriftInputs:
    def test_records_all_fields(self, tmp_path: Path) -> None:
        """Result contains all expected top-level keys with correct types."""
        (tmp_path / 'a.py').write_text('x', encoding='utf-8')
        result = compute_drift_inputs(tmp_path, ['a.py'])

        assert 'head_sha_at_index' in result
        assert 'repo_root_mtime' in result
        assert 'indexed_file_count' in result
        assert 'sentinel_files' in result
        assert 'indexed_dirs' in result

        assert isinstance(result['head_sha_at_index'], str)
        assert isinstance(result['repo_root_mtime'], float)
        assert result['indexed_file_count'] == 1
        assert isinstance(result['sentinel_files'], list)
        assert isinstance(result['indexed_dirs'], list)

    def test_indexed_dirs_includes_leaf_dirs(self, tmp_path: Path) -> None:
        """indexed_dirs records every distinct parent dir, not just top-level."""
        (tmp_path / 'src' / 'axon' / 'core').mkdir(parents=True)
        (tmp_path / 'tests' / 'unit').mkdir(parents=True)
        (tmp_path / 'src' / 'axon' / 'core' / 'foo.py').write_text(
            '', encoding='utf-8'
        )
        (tmp_path / 'tests' / 'unit' / 'bar.py').write_text(
            '', encoding='utf-8'
        )

        result = compute_drift_inputs(
            tmp_path, ['src/axon/core/foo.py', 'tests/unit/bar.py']
        )

        dir_paths = {
            entry['path'] if isinstance(entry, dict) else entry.path
            for entry in result['indexed_dirs']
        }
        assert 'src/axon/core' in dir_paths
        assert 'tests/unit' in dir_paths
        # The top-level 'src' and 'tests' should NOT be in indexed_dirs unless
        # a file was directly under them.
        assert 'src' not in dir_paths
        assert 'tests' not in dir_paths

    def test_indexed_dirs_capped_at_max(self, tmp_path: Path) -> None:
        """When distinct parent dirs exceed cap, only the top-N are kept."""
        file_list: list[str] = []
        for i in range(DRIFT_INDEXED_DIRS_CAP + 200):
            rel = f'dir_{i}/file.py'
            (tmp_path / f'dir_{i}').mkdir(exist_ok=True)
            (tmp_path / rel).write_text('', encoding='utf-8')
            file_list.append(rel)

        result = compute_drift_inputs(tmp_path, file_list)
        assert len(result['indexed_dirs']) == DRIFT_INDEXED_DIRS_CAP

        counts = [
            (e['indexed_count'] if isinstance(e, dict) else e.indexed_count)
            for e in result['indexed_dirs']
        ]
        assert counts == sorted(counts, reverse=True)

    def test_coerces_none_head_sha_to_empty(self, tmp_path: Path) -> None:
        """Non-git directory produces head_sha_at_index == ''."""
        result = compute_drift_inputs(tmp_path, [])
        assert result['head_sha_at_index'] == ''

    def test_sentinel_sorted_ascending(self, tmp_path: Path) -> None:
        """Stored sentinel_files are in ascending mtime order."""
        files = [f'f{i}.py' for i in range(5)]
        for i, name in enumerate(files):
            p = tmp_path / name
            p.write_text('x', encoding='utf-8')
            os.utime(p, (1_000_000 + i * 10, 1_000_000 + i * 10))

        result = compute_drift_inputs(tmp_path, files)
        sentinels = result['sentinel_files']
        mtimes = [
            (s['mtime'] if isinstance(s, dict) else s.mtime) for s in sentinels
        ]
        assert mtimes == sorted(mtimes)


# ---------------------------------------------------------------------------
# probe_drift - Tier 0
# ---------------------------------------------------------------------------


class TestProbeTier0:
    def test_short_circuits_when_watcher_recent(self, tmp_path: Path) -> None:
        """FRESH result with tier_used=0 when host.json + recent last_incremental_at."""
        from datetime import datetime, timezone

        now_ts = datetime.now(timezone.utc).isoformat()
        _write_host_json(tmp_path, now_ts)
        update_meta(tmp_path, last_incremental_at=now_ts)

        report = probe_drift(tmp_path)
        assert report.level == DriftLevel.FRESH
        assert report.tier_used == 0
        assert report.watcher_alive is True

    def test_skipped_when_no_host_json(self, tmp_path: Path) -> None:
        """Falls through Tier 0 when host.json is absent."""
        from datetime import datetime, timezone

        now_ts = datetime.now(timezone.utc).isoformat()
        update_meta(tmp_path, last_incremental_at=now_ts)
        # No host.json written -> Tier 0 must not fire.

        report = probe_drift(tmp_path)
        assert report.tier_used != 0

    def test_skipped_when_last_incremental_old(self, tmp_path: Path) -> None:
        """Falls through Tier 0 when last_incremental_at is older than recency limit."""
        from datetime import datetime, timezone, timedelta

        old_ts = (
            datetime.now(timezone.utc)
            - timedelta(seconds=LIVE_WATCHER_RECENCY_SECONDS + 60)
        ).isoformat()
        _write_host_json(tmp_path, old_ts)
        update_meta(tmp_path, last_incremental_at=old_ts)

        report = probe_drift(tmp_path)
        assert report.tier_used != 0


# ---------------------------------------------------------------------------
# probe_drift - Tier 1
# ---------------------------------------------------------------------------


class TestProbeTier1:
    @pytest.fixture()
    def git_repo(self, tmp_path: Path) -> Path:
        """Minimal git repo with .axon/ gitignored and a single committed file."""
        (tmp_path / '.gitignore').write_text('.axon/\n', encoding='utf-8')
        (tmp_path / 'main.py').write_text('pass\n', encoding='utf-8')
        _init_git_repo(tmp_path)
        return tmp_path

    def test_fresh_when_head_unchanged_clean_tree(
        self, git_repo: Path
    ) -> None:
        """FRESH, tier_used=1 when HEAD matches and working tree is clean."""
        head = _git(['rev-parse', 'HEAD'], git_repo).stdout.strip()
        update_meta(git_repo, head_sha_at_index=head, indexed_file_count=1)

        report = probe_drift(git_repo)
        assert report.level == DriftLevel.FRESH
        assert report.tier_used == 1

    def test_minor_when_head_unchanged_dirty_tree(
        self, git_repo: Path
    ) -> None:
        """STALE_MINOR when HEAD matches but working tree has uncommitted changes."""
        head = _git(['rev-parse', 'HEAD'], git_repo).stdout.strip()
        update_meta(git_repo, head_sha_at_index=head, indexed_file_count=1)

        (git_repo / 'dirty.py').write_text('new', encoding='utf-8')

        report = probe_drift(git_repo)
        assert report.level == DriftLevel.STALE_MINOR
        assert report.tier_used == 1

    def test_minor_when_head_advanced_few_commits(
        self, git_repo: Path
    ) -> None:
        """STALE_MINOR when HEAD advanced by a small number of commits."""
        old_head = _git(['rev-parse', 'HEAD'], git_repo).stdout.strip()
        update_meta(
            git_repo, head_sha_at_index=old_head, indexed_file_count=10
        )

        for i in range(3):
            _commit_file(git_repo, f'file{i}.py', f'# {i}')

        report = probe_drift(git_repo)
        assert report.level == DriftLevel.STALE_MINOR
        assert report.tier_used == 1

    def test_major_when_head_advanced_many_commits(
        self, git_repo: Path
    ) -> None:
        """STALE_MAJOR when HEAD advanced by more than the threshold commits."""
        old_head = _git(['rev-parse', 'HEAD'], git_repo).stdout.strip()
        update_meta(
            git_repo, head_sha_at_index=old_head, indexed_file_count=200
        )

        for i in range(STALE_MAJOR_COMMIT_THRESHOLD + 1):
            _commit_file(git_repo, f'generated_{i}.py', f'# gen {i}')

        report = probe_drift(git_repo)
        assert report.level == DriftLevel.STALE_MAJOR
        assert report.tier_used == 1

    def test_major_when_majority_files_changed(self, git_repo: Path) -> None:
        """STALE_MAJOR when changed-file count exceeds 50% of indexed_file_count."""
        old_head = _git(['rev-parse', 'HEAD'], git_repo).stdout.strip()
        # Index claims 4 files; we will change 3 (>50%).
        update_meta(git_repo, head_sha_at_index=old_head, indexed_file_count=4)

        for i in range(3):
            _commit_file(git_repo, f'changed_{i}.py', f'# c{i}')

        report = probe_drift(git_repo)
        assert report.level == DriftLevel.STALE_MAJOR
        assert report.tier_used == 1


# ---------------------------------------------------------------------------
# probe_drift - Tier 2
# ---------------------------------------------------------------------------


class TestProbeTier2:
    def test_minor_when_sentinel_mtime_advanced(self, tmp_path: Path) -> None:
        """STALE_MINOR, tier_used=2 when a sentinel file's mtime advances."""
        sentinel_file = tmp_path / 'watch_me.py'
        sentinel_file.write_text('x', encoding='utf-8')
        old_mtime = 1_000_000.0
        os.utime(sentinel_file, (old_mtime, old_mtime))

        update_meta(
            tmp_path,
            head_sha_at_index='',
            sentinel_files=[
                SentinelEntry(path='watch_me.py', mtime=old_mtime)
            ],
            indexed_file_count=1,
        )

        new_mtime = old_mtime + 100
        os.utime(sentinel_file, (new_mtime, new_mtime))

        report = probe_drift(tmp_path)
        assert report.level == DriftLevel.STALE_MINOR
        assert report.tier_used == 2


# ---------------------------------------------------------------------------
# probe_drift - Tier 3
# ---------------------------------------------------------------------------


class TestProbeTier3:
    def test_minor_when_leaf_dir_mtime_advanced(self, tmp_path: Path) -> None:
        """STALE_MINOR, tier_used=3 when a leaf dir's mtime advances."""
        leaf_dir = tmp_path / 'src' / 'axon' / 'core'
        leaf_dir.mkdir(parents=True)
        (leaf_dir / 'foo.py').write_text('x', encoding='utf-8')

        old_dir_mtime = 1_000_000.0
        os.utime(leaf_dir, (old_dir_mtime, old_dir_mtime))

        from axon.core.meta import IndexedDirEntry

        update_meta(
            tmp_path,
            head_sha_at_index='',
            repo_root_mtime=old_dir_mtime,
            sentinel_files=[],
            indexed_dirs=[
                IndexedDirEntry(
                    path='src/axon/core', mtime=old_dir_mtime, indexed_count=1
                )
            ],
            indexed_file_count=1,
        )

        new_dir_mtime = old_dir_mtime + 100
        os.utime(leaf_dir, (new_dir_mtime, new_dir_mtime))

        report = probe_drift(tmp_path)
        assert report.level == DriftLevel.STALE_MINOR
        assert report.tier_used == 3

    def test_minor_when_new_top_level_dir_added(self, tmp_path: Path) -> None:
        """STALE_MINOR, tier_used=3 when repo_root_mtime advances."""
        old_root_mtime = 1_000_000.0
        os.utime(tmp_path, (old_root_mtime, old_root_mtime))

        update_meta(
            tmp_path,
            head_sha_at_index='',
            repo_root_mtime=old_root_mtime,
            sentinel_files=[],
            indexed_dirs=[],
            indexed_file_count=0,
        )

        new_root_mtime = old_root_mtime + 100
        os.utime(tmp_path, (new_root_mtime, new_root_mtime))

        report = probe_drift(tmp_path)
        assert report.level == DriftLevel.STALE_MINOR
        assert report.tier_used == 3


# ---------------------------------------------------------------------------
# probe_drift - edge cases
# ---------------------------------------------------------------------------


class TestProbeEdgeCases:
    def test_unknown_when_repo_path_missing(self, tmp_path: Path) -> None:
        """UNKNOWN level when repo_path does not exist."""
        missing = tmp_path / 'no_such_repo'
        report = probe_drift(missing)
        assert report.level == DriftLevel.UNKNOWN
        assert report.tier_used is None

    def test_stale_major_when_drift_fields_absent(
        self, tmp_path: Path
    ) -> None:
        """STALE_MAJOR when meta has last_indexed_at but no drift fields."""
        update_meta(tmp_path, last_indexed_at='2024-01-01T00:00:00+00:00')
        report = probe_drift(tmp_path)
        assert report.level == DriftLevel.STALE_MAJOR
        assert report.tier_used is None

    def test_watcher_alive_set_when_host_json_exists_in_tier1(
        self, tmp_path: Path
    ) -> None:
        """watcher_alive is True when host.json exists, regardless of which
        tier produced the freshness verdict.

        Regression test: previously watcher_alive was only set in Tier 0;
        when Tier 1 produced FRESH, the flag stayed False even with a live
        watcher's host.json present.
        """
        (tmp_path / '.gitignore').write_text('.axon/\n', encoding='utf-8')
        (tmp_path / 'main.py').write_text('pass\n', encoding='utf-8')
        _init_git_repo(tmp_path)

        head = _git(['rev-parse', 'HEAD'], tmp_path).stdout.strip()
        # Stale last_incremental_at so Tier 0 falls through to Tier 1.
        update_meta(
            tmp_path,
            head_sha_at_index=head,
            indexed_file_count=1,
            last_incremental_at='2024-01-01T00:00:00+00:00',
        )
        # Host.json exists - watcher is alive.
        _write_host_json(tmp_path, '2024-01-01T00:00:00+00:00')

        report = probe_drift(tmp_path)
        assert report.tier_used == 1
        assert report.level == DriftLevel.FRESH
        assert report.watcher_alive is True

    def test_watcher_alive_false_when_no_host_json(
        self, tmp_path: Path
    ) -> None:
        """watcher_alive is False when host.json is absent, even with a
        FRESH Tier 1 verdict."""
        (tmp_path / '.gitignore').write_text('.axon/\n', encoding='utf-8')
        (tmp_path / 'main.py').write_text('pass\n', encoding='utf-8')
        _init_git_repo(tmp_path)

        head = _git(['rev-parse', 'HEAD'], tmp_path).stdout.strip()
        update_meta(
            tmp_path, head_sha_at_index=head, indexed_file_count=1
        )

        report = probe_drift(tmp_path)
        assert report.tier_used == 1
        assert report.level == DriftLevel.FRESH
        assert report.watcher_alive is False


# ---------------------------------------------------------------------------
# DriftCache
# ---------------------------------------------------------------------------


class TestDriftCache:
    def test_ttl_caches_and_re_probes(self, tmp_path: Path) -> None:
        """First call probes; second (within TTL) is cached; after TTL re-probes."""
        call_count = 0

        def counting_probe(path: Path):
            nonlocal call_count
            call_count += 1

            from axon.core.drift import DriftReport, DriftLevel
            return DriftReport(
                level=DriftLevel.UNKNOWN,
                reason='test',
                last_indexed_at='',
                head_sha=None,
                head_sha_at_index=None,
                files_changed_estimate=None,
                files_indexed_estimate=None,
                watcher_alive=False,
                tier_used=None,
            )

        monotonic_calls: list[float] = []
        times = iter([0.0, 5.0, DRIFT_CACHE_TTL_SECONDS + 1])

        def fake_monotonic() -> float:
            val = next(times)
            monotonic_calls.append(val)
            return val

        cache = DriftCache()

        with patch.object(
            drift_module, 'probe_drift', side_effect=counting_probe
        ):
            with patch('axon.core.drift.time') as mock_time:
                mock_time.monotonic.side_effect = fake_monotonic
                mock_time.time = time.time

                r1 = cache.get_or_probe(tmp_path)
                assert call_count == 1

                r2 = cache.get_or_probe(tmp_path)
                assert call_count == 1
                assert r2 is r1

                cache.get_or_probe(tmp_path)
                assert call_count == 2

    def test_invalidate_drops_entry(self, tmp_path: Path) -> None:
        """invalidate(repo_path) forces a re-probe on next call."""
        probe_calls = 0

        def counting_probe(path: Path):
            nonlocal probe_calls
            probe_calls += 1

            from axon.core.drift import DriftReport, DriftLevel
            return DriftReport(
                level=DriftLevel.UNKNOWN,
                reason='test',
                last_indexed_at='',
                head_sha=None,
                head_sha_at_index=None,
                files_changed_estimate=None,
                files_indexed_estimate=None,
                watcher_alive=False,
                tier_used=None,
            )

        cache = DriftCache()

        with patch.object(
            drift_module, 'probe_drift', side_effect=counting_probe
        ):
            cache.get_or_probe(tmp_path)
            assert probe_calls == 1

            cache.invalidate(tmp_path)
            cache.get_or_probe(tmp_path)
            assert probe_calls == 2
