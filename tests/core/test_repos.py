"""Tests for repos.py: RegistryEntry, RepoEntry, allocate_slug, RepoResolver,
RepoPool, qualify_node_id, and parse_qualified_id (Phase 2, multi-repo MCP).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from axon.core.repos import (
    RegistryEntry,
    RepoNotFound,
    RepoPool,
    RepoResolver,
    RepoUnavailable,
    allocate_slug,
    parse_qualified_id,
    qualify_node_id,
)
from axon.core.storage.kuzu_backend import KuzuBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_indexed_repo(tmp_path: Path, name: str) -> Path:
    """Create a minimal indexed repo directory at tmp_path/name.

    Initialises a KuzuBackend so the DB lock can be acquired by the pool.
    The .axon parent is created but kuzu is left to KuzuBackend to create.
    """
    repo = tmp_path / name
    (repo / '.axon').mkdir(parents=True)
    backend = KuzuBackend()
    backend.initialize(repo / '.axon' / 'kuzu')
    backend.close()
    return repo


def _write_registry_entry(
    registry_dir: Path, slug: str, repo_path: Path
) -> None:
    """Write a minimal RegistryEntry meta.json for a given slug.

    Used to simulate a registered repo without running axon analyze.
    """
    slot = registry_dir / slug
    slot.mkdir(parents=True, exist_ok=True)
    entry = RegistryEntry(
        name=repo_path.name,
        path=str(repo_path),
        slug=slug,
        last_indexed_at='2024-01-01T00:00:00',
        stats={},
        embedding_model='',
        embedding_dimensions=0,
    )
    (slot / 'meta.json').write_text(
        json.dumps(entry.to_json()), encoding='utf-8'
    )


# ---------------------------------------------------------------------------
# RegistryEntry
# ---------------------------------------------------------------------------


class TestRegistryEntry:
    """RegistryEntry serialisation and deserialisation."""

    def test_roundtrip(self) -> None:
        """to_json / from_json preserves all fields."""
        original = RegistryEntry(
            name='myrepo',
            path='/home/user/myrepo',
            slug='myrepo',
            last_indexed_at='2024-06-15T12:00:00',
            stats={'nodes': 42, 'edges': 7},
            embedding_model='sentence-transformers/all-MiniLM-L6-v2',
            embedding_dimensions=384,
        )
        restored = RegistryEntry.from_json(original.to_json())
        assert restored == original

    def test_from_json_returns_none_on_missing_field(self) -> None:
        """Missing required field 'path' causes from_json to return None."""
        data: dict = {
            'name': 'myrepo',
            'slug': 'myrepo',
            'last_indexed_at': '2024-01-01T00:00:00',
            'stats': {},
            'embedding_model': '',
            'embedding_dimensions': 0,
        }
        assert RegistryEntry.from_json(data) is None

    def test_from_json_returns_none_on_wrong_type(self) -> None:
        """embedding_dimensions as a non-castable string returns None."""
        data: dict = {
            'name': 'myrepo',
            'path': '/tmp/repo',
            'slug': 'myrepo',
            'last_indexed_at': '',
            'stats': {},
            'embedding_model': '',
            'embedding_dimensions': 'not-a-number',
        }
        assert RegistryEntry.from_json(data) is None

    def test_from_json_returns_none_on_garbage(self) -> None:
        """Completely garbage input returns None without raising."""
        assert RegistryEntry.from_json({'x': object()}) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# allocate_slug
# ---------------------------------------------------------------------------


class TestAllocateSlug:
    """allocate_slug slug-allocation logic."""

    def test_uses_repo_name_when_free(self, tmp_path: Path) -> None:
        """Fresh registry gives slug equal to the repo name."""
        registry = tmp_path / 'registry'
        repo = tmp_path / 'myproject'
        slug = allocate_slug(repo, registry)
        assert slug == 'myproject'

    def test_returns_existing_slug_when_path_already_registered(
        self, tmp_path: Path
    ) -> None:
        """Second call for the same repo_path under an existing slot returns the same slug."""
        registry = tmp_path / 'registry'
        repo = tmp_path / 'myproject'
        _write_registry_entry(registry, 'myproject', repo)
        slug = allocate_slug(repo, registry)
        assert slug == 'myproject'

    def test_collision_disambiguates_with_path_hash(
        self, tmp_path: Path
    ) -> None:
        """Different repo at the same name gets slug {name}-{sha256(path)[:8]}."""
        registry = tmp_path / 'registry'
        other_repo = tmp_path / 'other' / 'myproject'
        new_repo = tmp_path / 'new' / 'myproject'
        # Register the other repo under the plain slug.
        _write_registry_entry(registry, 'myproject', other_repo)
        slug = allocate_slug(new_repo, registry)
        expected_hash = hashlib.sha256(str(new_repo).encode()).hexdigest()[:8]
        assert slug == f'myproject-{expected_hash}'

    def test_cleans_corrupt_slot_before_claiming(self, tmp_path: Path) -> None:
        """Pre-existing slot with garbage meta.json is cleaned up before slug is claimed."""
        registry = tmp_path / 'registry'
        repo = tmp_path / 'myproject'
        slot = registry / 'myproject'
        slot.mkdir(parents=True)
        (slot / 'meta.json').write_text('not json at all', encoding='utf-8')
        slug = allocate_slug(repo, registry)
        assert slug == 'myproject'
        assert not slot.exists()

    def test_removes_stale_entries_for_same_repo_under_other_slugs(
        self, tmp_path: Path
    ) -> None:
        """Stale slot pointing to repo_path under a different slug is removed."""
        registry = tmp_path / 'registry'
        repo = tmp_path / 'myproject'
        old_hash = hashlib.sha256(str(repo).encode()).hexdigest()[:8]
        old_slug = f'myproject-{old_hash}'
        _write_registry_entry(registry, old_slug, repo)
        stale_dir = registry / old_slug
        assert stale_dir.exists()
        # Now allocate with the canonical name free - stale slot should disappear.
        slug = allocate_slug(repo, registry)
        assert slug == 'myproject'
        assert not stale_dir.exists()


# ---------------------------------------------------------------------------
# RepoResolver
# ---------------------------------------------------------------------------


class TestResolverLocal:
    """RepoResolver.local() behaviour."""

    def test_returns_synthesised_entry_when_not_registered(
        self, tmp_path: Path
    ) -> None:
        """local() synthesises a RepoEntry even before analyze has run."""
        registry = tmp_path / 'registry'
        repo = tmp_path / 'myrepo'
        resolver = RepoResolver(registry_dir=registry, local_repo_path=repo)
        entry = resolver.local()
        assert entry is not None
        assert entry.is_local is True
        assert entry.slug == 'myrepo'
        assert entry.path == repo.resolve()

    def test_returns_registered_entry_when_present(
        self, tmp_path: Path
    ) -> None:
        """local() returns the registered entry with is_local=True."""
        registry = tmp_path / 'registry'
        repo = tmp_path / 'myrepo'
        _write_registry_entry(registry, 'myrepo', repo)
        resolver = RepoResolver(registry_dir=registry, local_repo_path=repo)
        entry = resolver.local()
        assert entry is not None
        assert entry.is_local is True
        assert entry.slug == 'myrepo'


class TestResolverResolve:
    """RepoResolver.resolve() resolution strategies."""

    def test_resolves_by_slug(self, tmp_path: Path) -> None:
        """resolve('foo') finds the foo entry by exact slug match."""
        registry = tmp_path / 'registry'
        repo = tmp_path / 'foo'
        _write_registry_entry(registry, 'foo', repo)
        resolver = RepoResolver(registry_dir=registry)
        entry = resolver.resolve('foo')
        assert entry is not None
        assert entry.slug == 'foo'

    def test_resolves_by_absolute_path(self, tmp_path: Path) -> None:
        """resolve('/abs/path/to/repo') finds the entry by path."""
        registry = tmp_path / 'registry'
        repo = tmp_path / 'foo'
        repo.mkdir(parents=True)
        _write_registry_entry(registry, 'foo', repo)
        resolver = RepoResolver(registry_dir=registry)
        entry = resolver.resolve(str(repo))
        assert entry is not None
        assert entry.slug == 'foo'

    def test_resolves_by_cwd_relative_path(self, tmp_path: Path) -> None:
        """resolve('subdir') works when cwd is parent of the registered repo."""
        registry = tmp_path / 'registry'
        repo = tmp_path / 'subdir-repo'
        repo.mkdir(parents=True)
        _write_registry_entry(registry, 'subdir-repo', repo)
        resolver = RepoResolver(registry_dir=registry)
        # Resolve relative from tmp_path.
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            entry = resolver.resolve('subdir-repo')
        finally:
            os.chdir(original_cwd)
        assert entry is not None
        assert entry.slug == 'subdir-repo'

    def test_unknown_returns_none(self, tmp_path: Path) -> None:
        """resolve('nope') returns None when not registered."""
        registry = tmp_path / 'registry'
        resolver = RepoResolver(registry_dir=registry)
        assert resolver.resolve('nope') is None

    def test_strict_raises_repo_not_found_with_candidates(
        self, tmp_path: Path
    ) -> None:
        """resolve_strict('nope') raises RepoNotFound with the candidate list."""
        registry = tmp_path / 'registry'
        repo_a = tmp_path / 'alpha'
        _write_registry_entry(registry, 'alpha', repo_a)
        resolver = RepoResolver(registry_dir=registry)
        with pytest.raises(RepoNotFound) as exc_info:
            resolver.resolve_strict('nope')
        exc = exc_info.value
        assert exc.identifier == 'nope'
        assert 'alpha' in exc.candidates


class TestResolverList:
    """RepoResolver.list_known() and list_foreign() enumeration."""

    def test_list_known_includes_local_marked(self, tmp_path: Path) -> None:
        """list_known() includes the local entry exactly once with is_local=True."""
        registry = tmp_path / 'registry'
        repo = tmp_path / 'localrepo'
        _write_registry_entry(registry, 'localrepo', repo)
        resolver = RepoResolver(registry_dir=registry, local_repo_path=repo)
        known = resolver.list_known()
        local_entries = [e for e in known if e.is_local]
        assert len(local_entries) == 1
        assert local_entries[0].slug == 'localrepo'
        # No duplicate for the same path.
        assert len(known) == len({e.path for e in known})

    def test_list_foreign_excludes_local(self, tmp_path: Path) -> None:
        """list_foreign() does not include the local entry."""
        registry = tmp_path / 'registry'
        local_repo = tmp_path / 'localrepo'
        foreign_repo = tmp_path / 'foreignrepo'
        _write_registry_entry(registry, 'localrepo', local_repo)
        _write_registry_entry(registry, 'foreignrepo', foreign_repo)
        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        foreign = resolver.list_foreign()
        slugs = [e.slug for e in foreign]
        assert 'localrepo' not in slugs
        assert 'foreignrepo' in slugs


# ---------------------------------------------------------------------------
# RepoPool
# ---------------------------------------------------------------------------


class TestRepoPool:
    """RepoPool connection management and error handling."""

    def test_get_local_slug_raises_repo_unavailable(
        self, tmp_path: Path
    ) -> None:
        """pool.get(local_slug) raises RepoUnavailable with 'must use writer' message."""
        registry = tmp_path / 'registry'
        repo = _make_indexed_repo(tmp_path, 'local')
        _write_registry_entry(registry, 'local', repo)
        resolver = RepoResolver(registry_dir=registry, local_repo_path=repo)
        pool = RepoPool(resolver)
        with pytest.raises(RepoUnavailable, match='writer'):
            pool.get('local')
        pool.close_all()

    def test_get_opens_read_only_and_caches(self, tmp_path: Path) -> None:
        """First get opens the backend; second get returns the same instance."""
        registry = tmp_path / 'registry'
        foreign = _make_indexed_repo(tmp_path, 'foreign')
        _write_registry_entry(registry, 'foreign', foreign)
        resolver = RepoResolver(registry_dir=registry)
        pool = RepoPool(resolver)
        b1 = pool.get('foreign')
        b2 = pool.get('foreign')
        assert b1 is b2
        pool.close_all()

    def test_get_raises_repo_unavailable_on_unknown_slug(
        self, tmp_path: Path
    ) -> None:
        """pool.get('never-registered') raises RepoUnavailable('not registered')."""
        registry = tmp_path / 'registry'
        resolver = RepoResolver(registry_dir=registry)
        pool = RepoPool(resolver)
        with pytest.raises(RepoUnavailable, match='not registered'):
            pool.get('never-registered')

    def test_get_raises_repo_unavailable_on_axon_dir_deleted(
        self, tmp_path: Path
    ) -> None:
        """Deleting .axon/kuzu after registration causes RepoUnavailable on get."""
        registry = tmp_path / 'registry'
        foreign = _make_indexed_repo(tmp_path, 'foreign')
        _write_registry_entry(registry, 'foreign', foreign)
        # Kuzu creates a single DB file (not a directory); remove it.
        (foreign / '.axon' / 'kuzu').unlink(missing_ok=True)
        resolver = RepoResolver(registry_dir=registry)
        pool = RepoPool(resolver)
        with pytest.raises(RepoUnavailable) as exc_info:
            pool.get('foreign')
        assert exc_info.value.slug == 'foreign'
        # Error message should surface the underlying OS/RuntimeError text.
        assert exc_info.value.reason

    def test_get_caches_failures(self, tmp_path: Path) -> None:
        """Second pool.get for an unavailable repo returns the cached exception."""
        registry = tmp_path / 'registry'
        resolver = RepoResolver(registry_dir=registry)
        pool = RepoPool(resolver)
        try:
            pool.get('ghost')
        except RepoUnavailable:
            pass
        # Second call must raise the same exception object.
        with pytest.raises(RepoUnavailable) as exc_info:
            pool.get('ghost')
        assert exc_info.value is pool._failures['ghost']

    def test_close_all_releases_locks(self, tmp_path: Path) -> None:
        """After close_all(), a fresh writer can open the same DB."""
        registry = tmp_path / 'registry'
        foreign = _make_indexed_repo(tmp_path, 'foreign')
        _write_registry_entry(registry, 'foreign', foreign)
        resolver = RepoResolver(registry_dir=registry)
        pool = RepoPool(resolver)
        pool.get('foreign')
        pool.close_all()
        # Should be able to open in write mode after the read-only handle is released.
        writer = KuzuBackend()
        writer.initialize(foreign / '.axon' / 'kuzu')
        writer.close()

    def test_idle_handle_evicted_on_next_get_of_other_slug(
        self, tmp_path: Path
    ) -> None:
        """A handle idle longer than idle_ttl_seconds is closed on the next
        ``get`` for any slug, releasing its read-only lock so an external
        writer can acquire the DB.
        """
        registry = tmp_path / 'registry'
        idle = _make_indexed_repo(tmp_path, 'idle')
        active = _make_indexed_repo(tmp_path, 'active')
        _write_registry_entry(registry, 'idle', idle)
        _write_registry_entry(registry, 'active', active)
        resolver = RepoResolver(registry_dir=registry)
        pool = RepoPool(resolver, idle_ttl_seconds=0.05)

        idle_backend = pool.get('idle')
        assert 'idle' in pool._backends

        # Wait past the TTL, then touch a different slug.
        time.sleep(0.1)
        pool.get('active')

        # The idle handle should have been evicted by the lazy sweep.
        assert 'idle' not in pool._backends
        # And its lock should now be released - a writer can open the DB.
        writer = KuzuBackend()
        writer.initialize(idle / '.axon' / 'kuzu')
        writer.close()
        # idle_backend is now closed; explicitly call close() to confirm
        # double-close is safe (no exception).
        idle_backend.close()
        pool.close_all()

    def test_close_idle_releases_handles_explicitly(
        self, tmp_path: Path
    ) -> None:
        """close_idle() can be called externally to release stale handles
        without waiting for the next get."""
        registry = tmp_path / 'registry'
        foreign = _make_indexed_repo(tmp_path, 'foreign')
        _write_registry_entry(registry, 'foreign', foreign)
        resolver = RepoResolver(registry_dir=registry)
        pool = RepoPool(resolver, idle_ttl_seconds=0.05)

        pool.get('foreign')
        assert 'foreign' in pool._backends

        time.sleep(0.1)
        pool.close_idle()

        assert 'foreign' not in pool._backends
        # Writer can now claim the DB.
        writer = KuzuBackend()
        writer.initialize(foreign / '.axon' / 'kuzu')
        writer.close()
        pool.close_all()

    def test_get_within_ttl_keeps_handle_warm(self, tmp_path: Path) -> None:
        """A handle re-fetched within the TTL stays open and returns the
        same instance."""
        registry = tmp_path / 'registry'
        foreign = _make_indexed_repo(tmp_path, 'foreign')
        _write_registry_entry(registry, 'foreign', foreign)
        resolver = RepoResolver(registry_dir=registry)
        pool = RepoPool(resolver, idle_ttl_seconds=10.0)

        b1 = pool.get('foreign')
        b2 = pool.get('foreign')
        assert b1 is b2
        pool.close_all()


# ---------------------------------------------------------------------------
# qualify_node_id / parse_qualified_id
# ---------------------------------------------------------------------------


class TestQualifiedIds:
    """qualify_node_id and parse_qualified_id round-trip and edge cases."""

    def test_qualify_node_id_format(self) -> None:
        """qualify_node_id('foo', 'function:src/x.py:bar') -> 'foo::function:src/x.py:bar'."""
        result = qualify_node_id('foo', 'function:src/x.py:bar')
        assert result == 'foo::function:src/x.py:bar'

    def test_parse_qualified_id_with_separator(self) -> None:
        """parse_qualified_id round-trips a qualified ID."""
        qid = qualify_node_id('foo', 'function:src/x.py:bar')
        slug, raw = parse_qualified_id(qid)
        assert slug == 'foo'
        assert raw == 'function:src/x.py:bar'

    def test_parse_qualified_id_without_separator_returns_none_slug(
        self,
    ) -> None:
        """Bare raw IDs come back with slug=None and raw unchanged."""
        slug, raw = parse_qualified_id('function:src/x.py:bar')
        assert slug is None
        assert raw == 'function:src/x.py:bar'
