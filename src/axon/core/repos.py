"""Registry schema, slug allocation, repo resolver, pool, and qualified IDs.

``RegistryEntry`` is the single source of truth for the schema of
``~/.axon/repos/{slug}/meta.json``.  ``allocate_slug`` centralises the
slug-allocation logic used during ``axon analyze`` and by the resolver when
a local repo has not yet been registered.  ``RepoResolver`` enumerates
known repos and resolves identifiers (slug, absolute or relative path).
``RepoPool`` is a foreign-only, lazy, session-lifetime read-only
``KuzuBackend`` pool consumed by the MCP dispatch layer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axon.core.storage.kuzu_backend import KuzuBackend

logger = logging.getLogger(__name__)


_QUALIFIER_SEPARATOR = '::'

def default_registry_dir() -> Path:
    """Return the per-user registry directory path.

    Implemented as a function so callers and tests that monkeypatch
    ``Path.home`` see the patched value at call time.
    """
    return Path.home() / '.axon' / 'repos'


# ---------------------------------------------------------------------------
# Registry schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistryEntry:
    """Single source of truth for the schema of ~/.axon/repos/{slug}/meta.json.

    Provides ``to_json`` / ``from_json`` so callers never hand-roll the
    serialisation of individual fields.
    """

    name: str
    path: str
    slug: str
    last_indexed_at: str
    stats: dict[str, int]
    embedding_model: str
    embedding_dimensions: int

    def to_json(self) -> dict[str, Any]:
        """Serialise the entry to a plain dict suitable for json.dumps."""
        return {
            'name': self.name,
            'path': self.path,
            'slug': self.slug,
            'last_indexed_at': self.last_indexed_at,
            'stats': dict(self.stats),
            'embedding_model': self.embedding_model,
            'embedding_dimensions': self.embedding_dimensions,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> 'RegistryEntry | None':
        """Deserialise from a dict.

        Returns None when required fields are missing or have the wrong type
        so callers can treat malformed registry slots gracefully.
        """
        try:
            return cls(
                name=str(data['name']),
                path=str(data['path']),
                slug=str(data['slug']),
                last_indexed_at=str(data.get('last_indexed_at', '')),
                stats={
                    k: int(v)
                    for k, v in data.get('stats', {}).items()
                    if isinstance(k, str)
                },
                embedding_model=str(data.get('embedding_model', '')),
                embedding_dimensions=int(data.get('embedding_dimensions', 0)),
            )
        except (KeyError, TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Resolved repo entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoEntry:
    """A resolved repo with all paths pre-computed.

    ``is_local`` is True only for the entry that corresponds to the repo the
    current MCP session is attached to.  Foreign entries always have
    ``is_local=False``.
    """

    slug: str
    path: Path
    db_path: Path
    meta_path: Path
    is_local: bool = False


# ---------------------------------------------------------------------------
# Slug allocation
# ---------------------------------------------------------------------------


def allocate_slug(repo_path: Path, registry_dir: Path | None = None) -> str:
    """Allocate the canonical slug for *repo_path*.

    The slug is the repo name (last path segment) when that slot is free or
    already maps to *repo_path*.  On collision with a different path the slug
    is ``{repo_name}-{sha256(path)[:8]}``.  Any stale slots that point to
    *repo_path* under a different slug are cleaned up as a side-effect.
    """
    registry_dir = (
        registry_dir if registry_dir is not None else default_registry_dir()
    )
    repo_name = repo_path.name
    candidate_dir = registry_dir / repo_name
    slug = repo_name

    if candidate_dir.exists():
        existing_meta = candidate_dir / 'meta.json'
        try:
            existing = json.loads(existing_meta.read_text(encoding='utf-8'))
            if existing.get('path') != str(repo_path):
                short_hash = hashlib.sha256(
                    str(repo_path).encode()
                ).hexdigest()[:8]
                slug = f'{repo_name}-{short_hash}'
        except (json.JSONDecodeError, OSError):
            # Corrupt slot - clean it before claiming.
            shutil.rmtree(candidate_dir, ignore_errors=True)

    # Remove any stale entry for the same repo_path under a different slug.
    if registry_dir.exists():
        for old_dir in registry_dir.iterdir():
            if not old_dir.is_dir() or old_dir.name == slug:
                continue
            old_meta = old_dir / 'meta.json'
            try:
                old_data = json.loads(old_meta.read_text(encoding='utf-8'))
                if old_data.get('path') == str(repo_path):
                    shutil.rmtree(old_dir, ignore_errors=True)
            except (json.JSONDecodeError, OSError):
                continue

    return slug


# ---------------------------------------------------------------------------
# Resolver exceptions
# ---------------------------------------------------------------------------


class RepoNotFound(Exception):
    """Raised by ``RepoResolver.resolve_strict`` when no entry matches.

    ``candidates`` carries the slugs of all currently known repos so the
    caller can surface a helpful error message.
    """

    def __init__(self, identifier: str, candidates: list[str]) -> None:
        self.identifier = identifier
        self.candidates = candidates
        super().__init__(
            f'Repo not found: {identifier!r}. Known slugs: {candidates}'
        )


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def _build_repo_entry_from_slug_dir(slug_dir: Path) -> RepoEntry | None:
    """Parse a registry slot directory and return a RepoEntry.

    Returns None when the slot is missing, malformed, or the recorded path no
    longer exists on disk.
    """
    meta_file = slug_dir / 'meta.json'
    try:
        data = json.loads(meta_file.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return None

    entry = RegistryEntry.from_json(data)
    if entry is None:
        return None

    repo_path = Path(entry.path)
    return RepoEntry(
        slug=entry.slug,
        path=repo_path,
        db_path=repo_path / '.axon' / 'kuzu',
        meta_path=repo_path / '.axon' / 'meta.json',
    )


class RepoResolver:
    """Enumerate and resolve registered repos from the global registry.

    ``local_repo_path`` identifies the repo the current session is attached
    to.  That entry gets ``is_local=True`` and is excluded from
    ``list_foreign()``.  When the local repo has not yet been registered
    (e.g., before the first ``axon analyze``), a ``RepoEntry`` is still
    returned by ``local()`` - its slug is allocated via ``allocate_slug`` but
    nothing is written to the registry (that is ``analyze``'s job).
    """

    def __init__(
        self,
        registry_dir: Path | None = None,
        local_repo_path: Path | None = None,
    ) -> None:
        self._registry_dir = (
            registry_dir
            if registry_dir is not None
            else default_registry_dir()
        )
        self._local_repo_path = (
            local_repo_path.resolve() if local_repo_path is not None else None
        )

    # ------------------------------------------------------------------
    # Registry enumeration
    # ------------------------------------------------------------------

    def _read_registry(self) -> list[RepoEntry]:
        """Return all valid entries from the registry directory."""
        if not self._registry_dir.exists():
            return []
        entries: list[RepoEntry] = []
        for slug_dir in sorted(self._registry_dir.iterdir()):
            if not slug_dir.is_dir():
                continue
            entry = _build_repo_entry_from_slug_dir(slug_dir)
            if entry is not None:
                entries.append(entry)
        return entries

    def list_known(self) -> list[RepoEntry]:
        """All registered repos plus the local entry.

        The local entry is marked ``is_local=True``.  If the local repo is
        also in the registry it appears exactly once (marked local).
        """
        registered = self._read_registry()
        local_entry = self.local()
        if local_entry is None:
            return registered

        # Replace the registry entry for the local path with the local-marked
        # version, or append if not yet registered.
        result: list[RepoEntry] = []
        local_path = local_entry.path
        replaced = False
        for entry in registered:
            if entry.path.resolve() == local_path:
                result.append(local_entry)
                replaced = True
            else:
                result.append(entry)
        if not replaced:
            result.append(local_entry)
        return result

    def list_foreign(self) -> list[RepoEntry]:
        """All registered repos except the local entry."""
        return [e for e in self.list_known() if not e.is_local]

    def local(self) -> RepoEntry | None:
        """The local repo entry with ``is_local=True``.

        Returns None when no local repo path was provided at construction.
        When the local path is not yet in the registry, a RepoEntry is still
        synthesised (slug allocated but not written).
        """
        if self._local_repo_path is None:
            return None

        repo_path = self._local_repo_path
        # Try to find it in the registry by path.
        for entry in self._read_registry():
            if entry.path.resolve() == repo_path:
                return RepoEntry(
                    slug=entry.slug,
                    path=entry.path,
                    db_path=entry.db_path,
                    meta_path=entry.meta_path,
                    is_local=True,
                )

        # Not yet registered - synthesise an entry without writing.
        slug = allocate_slug(repo_path, self._registry_dir)
        return RepoEntry(
            slug=slug,
            path=repo_path,
            db_path=repo_path / '.axon' / 'kuzu',
            meta_path=repo_path / '.axon' / 'meta.json',
            is_local=True,
        )

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, identifier: str) -> RepoEntry | None:
        """Resolve *identifier* to a RepoEntry.

        Resolution order:
        1. Exact slug match in the registry.
        2. ``Path(identifier).resolve()`` matches an entry's path.
        3. ``(Path.cwd() / identifier).resolve()`` matches an entry's path.
        Returns None when no match is found.
        """
        known = self.list_known()

        # 1. Slug match.
        for entry in known:
            if entry.slug == identifier:
                return entry

        # 2. Absolute or relative path match.
        try:
            abs_id = Path(identifier).resolve()
        except (OSError, ValueError):
            return None

        for entry in known:
            if entry.path.resolve() == abs_id:
                return entry

        # 3. cwd-relative path match.
        try:
            rel_id = (Path.cwd() / identifier).resolve()
        except (OSError, ValueError):
            return None

        for entry in known:
            if entry.path.resolve() == rel_id:
                return entry

        return None

    def resolve_strict(self, identifier: str) -> RepoEntry:
        """Like ``resolve`` but raises ``RepoNotFound`` instead of returning None."""
        entry = self.resolve(identifier)
        if entry is not None:
            return entry
        candidates = [e.slug for e in self.list_known()]
        raise RepoNotFound(identifier, candidates)


# ---------------------------------------------------------------------------
# Pool exceptions
# ---------------------------------------------------------------------------


class RepoUnavailable(Exception):
    """Raised by ``RepoPool.get`` when a foreign backend cannot be opened.

    ``reason`` is a human-readable string suitable for inclusion in an MCP
    tool response.
    """

    def __init__(self, slug: str, reason: str) -> None:
        self.slug = slug
        self.reason = reason
        super().__init__(f'Repo {slug!r} unavailable: {reason}')


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------


class RepoPool:
    """Foreign-only, lazy, session-lifetime read-only KuzuBackend pool.

    Each foreign repo is opened at most once per session.  Failures are cached
    so repeated requests for an unavailable repo do not retry the expensive
    ``KuzuBackend.initialize`` path.

    Invariant: ``get`` MUST be called from inside ``asyncio.to_thread`` in
    async contexts to keep the blocking native DB open off the event loop.
    The Phase 3 dispatch layer is the sole caller and upholds this invariant.

    The local repo is excluded - callers should access the local writer or
    standalone fallback path directly.
    """

    def __init__(self, resolver: RepoResolver) -> None:
        self._resolver = resolver
        self._backends: dict[str, KuzuBackend] = {}
        self._failures: dict[str, RepoUnavailable] = {}
        self._lock = threading.Lock()

    def get(self, slug: str) -> KuzuBackend:
        """Return the cached backend for *slug*, opening it on first access.

        Raises ``RepoUnavailable`` when the repo cannot be resolved or the
        database cannot be opened (``RuntimeError`` or ``OSError`` from
        ``KuzuBackend.initialize`` are both wrapped).
        """
        with self._lock:
            # Reject local slug with a clear message.
            local = self._resolver.local()
            if local is not None and slug == local.slug:
                raise RepoUnavailable(
                    slug,
                    'local repo must be accessed via the session writer, '
                    'not the foreign pool',
                )

            if slug in self._backends:
                return self._backends[slug]

            if slug in self._failures:
                raise self._failures[slug]

            try:
                entry = self._resolver.resolve_strict(slug)
            except RepoNotFound:
                exc = RepoUnavailable(slug, 'not registered')
                self._failures[slug] = exc
                raise exc

            backend = KuzuBackend()
            try:
                backend.initialize(
                    entry.db_path,
                    read_only=True,
                    max_retries=3,
                    retry_delay=0.3,
                )
            except (RuntimeError, OSError) as e:
                exc = RepoUnavailable(slug, str(e))
                self._failures[slug] = exc
                raise exc

            self._backends[slug] = backend
            return backend

    def known_foreign_slugs(self) -> list[str]:
        """Slugs of all foreign repos currently visible in the registry."""
        return [e.slug for e in self._resolver.list_foreign()]

    def close_all(self) -> None:
        """Close every cached backend and clear both the cache and failures map."""
        with self._lock:
            for backend in self._backends.values():
                try:
                    backend.close()
                except Exception:
                    logger.debug(
                        'Error closing backend during pool teardown',
                        exc_info=True,
                    )
            self._backends.clear()
            self._failures.clear()


# ---------------------------------------------------------------------------
# Qualified node IDs (display-only)
# ---------------------------------------------------------------------------


def qualify_node_id(slug: str, raw_id: str) -> str:
    """Render ``{slug}::{raw_id}`` for cross-repo display only.

    Raw IDs inside any single Kuzu DB are never modified.  Qualified form
    appears only in cross-repo response sections produced by the MCP layer.
    """
    return f'{slug}{_QUALIFIER_SEPARATOR}{raw_id}'


def parse_qualified_id(qid: str) -> tuple[str | None, str]:
    """Inverse of ``qualify_node_id``.

    Returns ``(slug, raw_id)`` when *qid* contains the qualifier separator, or
    ``(None, qid)`` when it does not.
    """
    if _QUALIFIER_SEPARATOR not in qid:
        return None, qid
    slug, _, raw = qid.partition(_QUALIFIER_SEPARATOR)
    return slug, raw
