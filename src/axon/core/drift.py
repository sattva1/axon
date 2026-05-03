"""Four-tier drift detection for indexed repos.

Tiers run in execution order (0-3); each tier either returns a verdict or
falls through to the next:

- Tier 0: host.json + last_incremental_at recency. If a live watcher wrote
  meta.json within LIVE_WATCHER_RECENCY_SECONDS, the index is FRESH.
- Tier 1: git HEAD comparison. Compares the stored head_sha_at_index against
  the live HEAD; uses rev-list and diff counts to choose FRESH, STALE_MINOR,
  or STALE_MAJOR.
- Tier 2: sentinel mtime probe. Checks the stored top-N most-recently-
  modified indexed files. Sorted ascending so a changed file at the low end
  exits early.
- Tier 3: directory mtime probe. Checks repo_root_mtime and every distinct
  parent directory of indexed files captured at index time.

Any tier that produces a verdict short-circuits the remaining tiers. When
none of tiers 0-3 yields a verdict and the meta has no drift fields at all,
the result is STALE_MAJOR.

A session-level DriftCache (TTL = DRIFT_CACHE_TTL_SECONDS) avoids re-probing
the same repo more than once per cache window.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Iterable

from axon.core.host_meta import is_host_alive_fast, load_host_meta
from axon.core.meta import IndexedDirEntry, MetaFile, SentinelEntry, load_meta

logger = logging.getLogger(__name__)

DRIFT_SENTINEL_FILES = 64
DRIFT_INDEXED_DIRS_CAP = 5000
LIVE_WATCHER_RECENCY_SECONDS = 300
STALE_MAJOR_COMMIT_THRESHOLD = 50
STALE_MAJOR_FILE_PCT = 0.5
DRIFT_CACHE_TTL_SECONDS = 30


class DriftLevel(StrEnum):
    """Classification of how stale an indexed repo appears."""

    FRESH = auto()
    STALE_MINOR = auto()
    STALE_MAJOR = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class DriftReport:
    """Result of a drift probe for one repo.

    tier_used is the tier (0-3 in execution order) that produced the
    verdict, or None when level is UNKNOWN.

    slug is optionally populated by call sites that know the repo slug,
    allowing render_with_drift_warning to include it in the warning line.
    It is never set by probe_drift or DriftCache - those remain slug-agnostic.
    """

    level: DriftLevel
    reason: str
    last_indexed_at: str
    head_sha: str | None
    head_sha_at_index: str | None
    files_changed_estimate: int | None
    files_indexed_estimate: int | None
    watcher_alive: bool
    tier_used: int | None
    slug: str | None = None


def _get_head_sha(repo_path: Path) -> str | None:
    """Return the current git HEAD sha, or None when not in a git repo."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def compute_drift_inputs(
    repo_path: Path, indexed_files: Iterable[str]
) -> dict[str, Any]:
    """Collect filesystem stats for drift tracking and return update_meta kwargs.

    Stats every indexed file once. Builds:
    - sentinel_files: top-N by mtime descending, then stored sorted ascending
      so the probe can exit early on the first hit.
    - indexed_dirs: every distinct parent directory of an indexed file, each
      with its live mtime and indexed_count. Capped at DRIFT_INDEXED_DIRS_CAP
      by keeping the highest indexed_count dirs when the cap is hit.
    - repo_root_mtime: os.stat(repo_path).st_mtime.
    - head_sha_at_index: current HEAD sha, or '' when not in a git repo.
    - indexed_file_count: total number of indexed files.

    Root-level indexed files (no parent directory within the repo) use '.'
    as their directory path.
    """
    file_list = list(indexed_files)

    head_sha = _get_head_sha(repo_path)

    try:
        repo_root_mtime = os.stat(repo_path).st_mtime
    except OSError:
        repo_root_mtime = 0.0

    file_mtimes: list[tuple[float, str]] = []
    dir_counts: dict[str, int] = {}

    for rel_path in file_list:
        abs_path = repo_path / rel_path
        try:
            st = os.stat(abs_path)
            mtime = st.st_mtime
        except OSError:
            mtime = 0.0
        file_mtimes.append((mtime, rel_path))

        parent = str(Path(rel_path).parent)
        if parent == '.':
            dir_key = '.'
        else:
            dir_key = parent
        dir_counts[dir_key] = dir_counts.get(dir_key, 0) + 1

    file_mtimes.sort(key=lambda t: t[0], reverse=True)
    top_files = file_mtimes[:DRIFT_SENTINEL_FILES]
    top_files.sort(key=lambda t: t[0])
    sentinel_files = [
        SentinelEntry(path=rel, mtime=mtime) for mtime, rel in top_files
    ]

    indexed_dirs: list[IndexedDirEntry] = []
    if len(dir_counts) > DRIFT_INDEXED_DIRS_CAP:
        logger.debug(
            'indexed_dirs truncated: %d dirs -> cap %d',
            len(dir_counts),
            DRIFT_INDEXED_DIRS_CAP,
        )
        sorted_dirs = sorted(
            dir_counts.items(), key=lambda kv: kv[1], reverse=True
        )[:DRIFT_INDEXED_DIRS_CAP]
    else:
        sorted_dirs = list(dir_counts.items())

    for dir_path, count in sorted_dirs:
        abs_dir = repo_path / dir_path if dir_path != '.' else repo_path
        try:
            dir_mtime = os.stat(abs_dir).st_mtime
        except OSError:
            dir_mtime = 0.0
        indexed_dirs.append(
            IndexedDirEntry(
                path=dir_path, mtime=dir_mtime, indexed_count=count
            )
        )

    return {
        'head_sha_at_index': head_sha or '',
        'repo_root_mtime': repo_root_mtime,
        'indexed_file_count': len(file_list),
        'sentinel_files': sentinel_files,
        'indexed_dirs': indexed_dirs,
    }


def _parse_iso(ts: str) -> float | None:
    """Parse an ISO-8601 string to a UTC epoch float, or None on failure."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def _probe_tier0(
    repo_path: Path, meta: MetaFile, now: float
) -> DriftReport | None:
    """Tier 0: host.json existence + last_incremental_at recency + HEAD match.

    Guard ordering (deliberate):
    1. host-alive check first; on failure return None (fall through).
    2. last_incremental_at recency check; on failure return None.
    3. Call _get_head_sha. If None (not a git repo), return FRESH - non-git
       repos with an active watcher have no HEAD to compare and are treated
       as FRESH by policy.
    4. If _get_head_sha returns a sha and meta.head_sha_at_index is empty
       (legacy meta without drift tracking), return None so Tier 1/2/3
       can produce a verdict.
    5. If live sha != stored sha, return None - Tier 1 produces the actual
       STALE verdict. The watcher writes head_sha_at_index only after a
       successful commit_transition flush, so equality proves the watcher
       has caught up to the current HEAD.
    6. If live sha == stored sha, return FRESH.
    """
    if not is_host_alive_fast(repo_path):
        return None

    host = load_host_meta(repo_path)
    if host is None:
        return None

    incremental_ts = _parse_iso(meta.last_incremental_at)
    if incremental_ts is None:
        return None

    if (now - incremental_ts) > LIVE_WATCHER_RECENCY_SECONDS:
        return None

    live_sha = _get_head_sha(repo_path)
    if live_sha is None:
        # Non-git repo with active watcher - no HEAD to compare; policy is FRESH.
        return DriftReport(
            level=DriftLevel.FRESH,
            reason='live watcher updated index recently',
            last_indexed_at=meta.last_indexed_at,
            head_sha=None,
            head_sha_at_index=meta.head_sha_at_index or None,
            files_changed_estimate=None,
            files_indexed_estimate=meta.indexed_file_count or None,
            watcher_alive=True,
            tier_used=0,
        )

    stored_sha = meta.head_sha_at_index
    if not stored_sha:
        # Legacy meta without drift tracking - fall through to Tier 1+.
        return None

    if live_sha != stored_sha:
        # Watcher has not yet flushed the new HEAD; fall through to Tier 1.
        return None

    return DriftReport(
        level=DriftLevel.FRESH,
        reason='live watcher updated index recently',
        last_indexed_at=meta.last_indexed_at,
        head_sha=live_sha,
        head_sha_at_index=stored_sha,
        files_changed_estimate=None,
        files_indexed_estimate=meta.indexed_file_count or None,
        watcher_alive=True,
        tier_used=0,
    )


def _probe_tier1(repo_path: Path, meta: MetaFile) -> DriftReport | None:
    """Tier 1: git HEAD comparison."""
    stored_sha = meta.head_sha_at_index
    if not stored_sha:
        return None

    live_sha = _get_head_sha(repo_path)
    if live_sha is None:
        return None

    if live_sha == stored_sha:
        try:
            status_result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None

        if status_result.returncode != 0:
            return None

        if not status_result.stdout.strip():
            return DriftReport(
                level=DriftLevel.FRESH,
                reason='HEAD unchanged, working tree clean',
                last_indexed_at=meta.last_indexed_at,
                head_sha=live_sha,
                head_sha_at_index=stored_sha,
                files_changed_estimate=0,
                files_indexed_estimate=meta.indexed_file_count or None,
                watcher_alive=False,
                tier_used=1,
            )
        return DriftReport(
            level=DriftLevel.STALE_MINOR,
            reason='HEAD unchanged but working tree is dirty',
            last_indexed_at=meta.last_indexed_at,
            head_sha=live_sha,
            head_sha_at_index=stored_sha,
            files_changed_estimate=None,
            files_indexed_estimate=meta.indexed_file_count or None,
            watcher_alive=False,
            tier_used=1,
        )

    # Check whether stored_sha is an ancestor of live_sha. git reset --hard,
    # rebase, or diverged branch switches leave stored_sha unreachable from
    # live_sha, making rev-list --count return 0 (misleadingly small diff).
    try:
        ancestor_result = subprocess.run(
            ['git', 'merge-base', '--is-ancestor', stored_sha, live_sha],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        is_ancestor = ancestor_result.returncode == 0
    except Exception:
        is_ancestor = False

    if not is_ancestor:
        return DriftReport(
            level=DriftLevel.STALE_MAJOR,
            reason='HEAD diverged from indexed sha (non-ancestor)',
            last_indexed_at=meta.last_indexed_at,
            head_sha=live_sha,
            head_sha_at_index=stored_sha,
            files_changed_estimate=None,
            files_indexed_estimate=meta.indexed_file_count or None,
            watcher_alive=False,
            tier_used=1,
        )

    try:
        count_result = subprocess.run(
            ['git', 'rev-list', '--count', f'{stored_sha}..{live_sha}'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit_distance = (
            int(count_result.stdout.strip())
            if count_result.returncode == 0
            else 0
        )
    except Exception:
        commit_distance = 0

    try:
        diff_result = subprocess.run(
            ['git', 'diff', '--name-only', stored_sha, live_sha],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        changed_files = (
            len(
                [
                    line
                    for line in diff_result.stdout.splitlines()
                    if line.strip()
                ]
            )
            if diff_result.returncode == 0
            else 0
        )
    except Exception:
        changed_files = 0

    indexed_count = meta.indexed_file_count or 1
    is_major = (
        commit_distance > STALE_MAJOR_COMMIT_THRESHOLD
        or changed_files > STALE_MAJOR_FILE_PCT * indexed_count
    )

    level = DriftLevel.STALE_MAJOR if is_major else DriftLevel.STALE_MINOR
    return DriftReport(
        level=level,
        reason=(
            f'HEAD advanced by {commit_distance} commits, '
            f'{changed_files} files changed'
        ),
        last_indexed_at=meta.last_indexed_at,
        head_sha=live_sha,
        head_sha_at_index=stored_sha,
        files_changed_estimate=changed_files,
        files_indexed_estimate=meta.indexed_file_count or None,
        watcher_alive=False,
        tier_used=1,
    )


def _probe_tier2(repo_path: Path, meta: MetaFile) -> DriftReport | None:
    """Tier 2: sentinel mtime probe (stored ascending for early exit)."""
    if not meta.sentinel_files:
        return None

    for entry in meta.sentinel_files:
        abs_path = repo_path / entry.path
        try:
            live_mtime = os.stat(abs_path).st_mtime
        except OSError:
            continue
        if live_mtime > entry.mtime:
            return DriftReport(
                level=DriftLevel.STALE_MINOR,
                reason=f'sentinel file modified: {entry.path}',
                last_indexed_at=meta.last_indexed_at,
                head_sha=None,
                head_sha_at_index=meta.head_sha_at_index or None,
                files_changed_estimate=None,
                files_indexed_estimate=meta.indexed_file_count or None,
                watcher_alive=False,
                tier_used=2,
            )
    return None


def _probe_tier3(repo_path: Path, meta: MetaFile) -> DriftReport | None:
    """Tier 3: repo_root_mtime and indexed_dirs mtime probe."""
    if meta.repo_root_mtime > 0.0:
        try:
            live_root_mtime = os.stat(repo_path).st_mtime
        except OSError:
            live_root_mtime = 0.0
        if live_root_mtime > meta.repo_root_mtime:
            return DriftReport(
                level=DriftLevel.STALE_MINOR,
                reason='repo root directory mtime advanced (new top-level entry)',
                last_indexed_at=meta.last_indexed_at,
                head_sha=None,
                head_sha_at_index=meta.head_sha_at_index or None,
                files_changed_estimate=None,
                files_indexed_estimate=meta.indexed_file_count or None,
                watcher_alive=False,
                tier_used=3,
            )

    for entry in meta.indexed_dirs:
        abs_dir = repo_path / entry.path if entry.path != '.' else repo_path
        try:
            live_mtime = os.stat(abs_dir).st_mtime
        except OSError:
            continue
        if live_mtime > entry.mtime:
            return DriftReport(
                level=DriftLevel.STALE_MINOR,
                reason=f'indexed directory mtime advanced: {entry.path}',
                last_indexed_at=meta.last_indexed_at,
                head_sha=None,
                head_sha_at_index=meta.head_sha_at_index or None,
                files_changed_estimate=None,
                files_indexed_estimate=meta.indexed_file_count or None,
                watcher_alive=False,
                tier_used=3,
            )

    if meta.indexed_dirs or meta.repo_root_mtime > 0.0:
        return DriftReport(
            level=DriftLevel.FRESH,
            reason='all directory mtimes match index',
            last_indexed_at=meta.last_indexed_at,
            head_sha=None,
            head_sha_at_index=meta.head_sha_at_index or None,
            files_changed_estimate=0,
            files_indexed_estimate=meta.indexed_file_count or None,
            watcher_alive=False,
            tier_used=3,
        )
    return None


def probe_drift(repo_path: Path) -> DriftReport:
    """Run the four-tier drift cascade for *repo_path*.

    Returns a DriftReport with level UNKNOWN when no meta.json exists or
    when repo_path is not accessible. Falls through to STALE_MAJOR when
    meta.json exists but has no drift fields (pre-Phase-1 index).

    The ``watcher_alive`` flag is probed independently of which tier
    produces the verdict: a host.json file at *repo_path* always implies
    a live watcher, regardless of whether Tier 0 short-circuited the
    cascade or a later tier produced the freshness verdict.
    """
    if not repo_path.exists():
        return DriftReport(
            level=DriftLevel.UNKNOWN,
            reason='repo path does not exist',
            last_indexed_at='',
            head_sha=None,
            head_sha_at_index=None,
            files_changed_estimate=None,
            files_indexed_estimate=None,
            watcher_alive=False,
            tier_used=None,
        )

    meta = load_meta(repo_path)
    now = time.time()
    watcher_alive = is_host_alive_fast(repo_path)

    report = _probe_tier0(repo_path, meta, now)
    if report is None:
        report = _probe_tier1(repo_path, meta)
    if report is None:
        report = _probe_tier2(repo_path, meta)
    if report is None:
        report = _probe_tier3(repo_path, meta)

    if report is not None:
        if report.watcher_alive == watcher_alive:
            return report
        return dataclasses.replace(report, watcher_alive=watcher_alive)

    has_drift_fields = bool(
        meta.head_sha_at_index
        or meta.repo_root_mtime
        or meta.sentinel_files
        or meta.indexed_dirs
    )
    if not has_drift_fields:
        return DriftReport(
            level=DriftLevel.STALE_MAJOR,
            reason=(
                'drift fields missing - '
                'repo indexed before drift tracking landed'
            ),
            last_indexed_at=meta.last_indexed_at,
            head_sha=None,
            head_sha_at_index=None,
            files_changed_estimate=None,
            files_indexed_estimate=meta.indexed_file_count or None,
            watcher_alive=watcher_alive,
            tier_used=None,
        )

    return DriftReport(
        level=DriftLevel.UNKNOWN,
        reason='no tier produced a verdict',
        last_indexed_at=meta.last_indexed_at,
        head_sha=None,
        head_sha_at_index=meta.head_sha_at_index or None,
        files_changed_estimate=None,
        files_indexed_estimate=meta.indexed_file_count or None,
        watcher_alive=watcher_alive,
        tier_used=None,
    )


class DriftCache:
    """Per-process session-level cache of DriftReport keyed by repo path.

    get_or_probe returns a cached result if it is younger than
    DRIFT_CACHE_TTL_SECONDS, otherwise re-runs probe_drift.
    Thread-safe via an internal Lock.
    """

    def __init__(self) -> None:
        self._cache: dict[Path, tuple[DriftReport, float]] = {}
        self._lock = threading.Lock()

    def get_or_probe(self, repo_path: Path) -> DriftReport:
        """Return a cached DriftReport or run probe_drift and cache the result."""
        resolved = repo_path.resolve()
        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(resolved)
            if entry is not None:
                report, expires_at = entry
                if now < expires_at:
                    return report

        report = probe_drift(resolved)
        expires_at = now + DRIFT_CACHE_TTL_SECONDS

        with self._lock:
            self._cache[resolved] = (report, expires_at)

        return report

    def invalidate(self, repo_path: Path) -> None:
        """Remove a cached entry so the next call re-probes."""
        resolved = repo_path.resolve()
        with self._lock:
            self._cache.pop(resolved, None)
