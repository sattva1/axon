"""Single-source-of-truth reader/writer for .axon/meta.json.

Providers (pipeline/watcher) call update_meta() with the fields they
own; consumers (MCP handlers, CLI commands, status reports) call
load_meta(). Atomic writes via os.replace() prevent torn files under
crash or concurrent access.

Module placement: ``core/`` rather than ``core/ingestion/`` because
consumers span CLI, ingestion, and MCP layers; the subject matter is
repo-level metadata I/O, not ingestion-domain logic.
"""

from __future__ import annotations

import dataclasses
import fcntl
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dc_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_META_FILENAME = 'meta.json'
_META_DIRNAME = '.axon'
_LOCK_FILENAME = 'meta.json.lock'


@dataclass(frozen=True)
class SentinelEntry:
    """One of the most-recently-modified indexed files at index time.

    Stored sorted ascending by mtime so the drift probe can exit early on
    the first entry whose live mtime exceeds the recorded value.
    """

    path: str
    mtime: float


@dataclass(frozen=True)
class IndexedDirEntry:
    """A distinct parent directory of one or more indexed files.

    mtime is the directory's mtime at index time. indexed_count is the
    number of indexed files whose immediate parent is this directory.
    """

    path: str
    mtime: float
    indexed_count: int


@dataclass(frozen=True)
class MetaFile:
    """Immutable snapshot of .axon/meta.json.

    All fields are optional with defaults so missing-key tolerance is
    automatic; load_meta() fills gaps with defaults and never raises
    on malformed input.

    stats is dict[str, Any] rather than dict[str, int] so a single
    malformed value (e.g., float, string) doesn't cause the entire
    file to fall back to defaults on load. load_meta() coerces
    non-int values to 0 with a debug log.
    """

    version: str = ''
    name: str = ''
    path: str = ''
    embedding_model: str = ''
    embedding_dimensions: int = 0
    stats: dict[str, Any] = field(default_factory=dict)
    last_indexed_at: str = ''
    # Phase 6 freshness fields:
    last_incremental_at: str = ''
    dead_code_last_refreshed_at: str = ''
    communities_last_refreshed_at: str = ''
    # Phase 1 drift-detection fields:
    head_sha_at_index: str = ''
    repo_root_mtime: float = 0.0
    indexed_file_count: int = 0
    sentinel_files: list[SentinelEntry] = field(default_factory=list)
    indexed_dirs: list[IndexedDirEntry] = field(default_factory=list)


def meta_path(repo_root: Path) -> Path:
    """Canonical path - single source of truth for the location."""
    return repo_root / _META_DIRNAME / _META_FILENAME


def now_iso() -> str:
    """UTC ISO-8601 string. Centralised so tests can monkeypatch."""
    return datetime.now(timezone.utc).isoformat()


def _sanitize_stats(raw: Any) -> dict[str, int]:
    """Coerce stats to dict[str, int], dropping malformed values."""
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, int] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, bool):  # bool is int in Python; reject.
            continue
        if isinstance(v, int):
            cleaned[k] = v
        else:
            logger.debug("meta.json stats['%s']=%r coerced to 0", k, v)
            cleaned[k] = 0
    return cleaned


def _sanitize_sentinel_files(raw: Any) -> list[SentinelEntry]:
    """Reconstruct SentinelEntry list from JSON, dropping malformed entries."""
    if not isinstance(raw, list):
        return []
    result: list[SentinelEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            result.append(
                SentinelEntry(
                    path=str(item['path']), mtime=float(item['mtime'])
                )
            )
        except (KeyError, TypeError, ValueError):
            logger.debug('Dropping malformed sentinel_files entry: %r', item)
    return result


def _sanitize_indexed_dirs(raw: Any) -> list[IndexedDirEntry]:
    """Reconstruct IndexedDirEntry list from JSON, dropping malformed entries."""
    if not isinstance(raw, list):
        return []
    result: list[IndexedDirEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            result.append(
                IndexedDirEntry(
                    path=str(item['path']),
                    mtime=float(item['mtime']),
                    indexed_count=int(item['indexed_count']),
                )
            )
        except (KeyError, TypeError, ValueError):
            logger.debug('Dropping malformed indexed_dirs entry: %r', item)
    return result


def load_meta(repo_root: Path) -> MetaFile:
    """Read meta.json; defaults on any I/O or JSON error.

    Torn reads (partial JSON from a crashed write) return defaults
    too; update_meta() will repair on the next write. Unknown keys
    in the JSON are silently dropped (forward-compat: older tools
    reading files written by newer ones).
    """
    path = meta_path(repo_root)
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return MetaFile()
    if not isinstance(raw, dict):
        return MetaFile()
    known = {f.name for f in dc_fields(MetaFile)}
    filtered = {k: v for k, v in raw.items() if k in known}
    if 'stats' in filtered:
        filtered['stats'] = _sanitize_stats(filtered['stats'])
    if 'sentinel_files' in filtered:
        filtered['sentinel_files'] = _sanitize_sentinel_files(
            filtered['sentinel_files']
        )
    if 'indexed_dirs' in filtered:
        filtered['indexed_dirs'] = _sanitize_indexed_dirs(
            filtered['indexed_dirs']
        )
    try:
        return MetaFile(**filtered)
    except TypeError:
        logger.debug('meta.json has incompatible types; returning defaults')
        return MetaFile()


def update_meta(repo_root: Path, **fields: Any) -> None:
    """Atomically merge *fields* into existing meta.json.

    Semantics: load existing, apply *fields* as a shallow override,
    with special-case MERGE for the nested ``stats`` dict (so partial
    stats updates don't clobber siblings). Writes to tmp + os.replace
    for atomicity. Creates .axon/ if missing.

    Concurrent writers are serialized via an exclusive POSIX flock on
    a sibling lock file (``.axon/meta.json.lock``). The lock is held
    only during the load-modify-write cycle; the os.replace swap is
    inside the critical section. The lock file itself is never renamed.
    """
    path = meta_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / _LOCK_FILENAME
    with open(lock_path, 'a', encoding='utf-8') as lock_fh:
        os.chmod(lock_path, 0o644)
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            current = load_meta(repo_root)
            merged = asdict(current)
            for key, value in fields.items():
                if key == 'stats' and isinstance(value, dict):
                    merged_stats = dict(merged.get('stats') or {})
                    merged_stats.update(value)
                    merged['stats'] = merged_stats
                elif (
                    isinstance(value, list)
                    and value
                    and dataclasses.is_dataclass(value[0])
                ):
                    merged[key] = [asdict(item) for item in value]
                else:
                    merged[key] = value
            tmp = path.with_suffix(path.suffix + '.tmp')
            tmp.write_text(json.dumps(merged, indent=2), encoding='utf-8')
            os.replace(tmp, path)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
