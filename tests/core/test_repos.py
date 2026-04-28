"""Tests for repos.py: RegistryEntry, RepoEntry, allocate_slug, RepoResolver,
open_foreign_backend, qualify_node_id, and parse_qualified_id.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from axon.core.repos import (
    _DISPATCH_OPEN_RETRIES,
    _DISPATCH_OPEN_RETRY_DELAY,
    _FLUSH_OPEN_RETRIES,
    _FLUSH_OPEN_RETRY_DELAY,
    RegistryEntry,
    RepoNotFound,
    RepoResolver,
    RepoUnavailable,
    allocate_slug,
    open_foreign_backend,
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
# open_foreign_backend
# ---------------------------------------------------------------------------


class TestOpenForeignBackend:
    """open_foreign_backend on-demand foreign backend open and error handling."""

    def test_open_foreign_yields_initialized_backend_and_closes(
        self, tmp_path: Path
    ) -> None:
        """Context manager yields an open backend; _db and _conn are None after exit."""
        registry = tmp_path / 'registry'
        foreign = _make_indexed_repo(tmp_path, 'foreign')
        _write_registry_entry(registry, 'foreign', foreign)
        resolver = RepoResolver(registry_dir=registry)

        with open_foreign_backend(resolver, 'foreign') as backend:
            assert backend._db is not None
            assert backend._conn is not None

        assert backend._db is None
        assert backend._conn is None

    def test_open_foreign_local_slug_raises_repo_unavailable(
        self, tmp_path: Path
    ) -> None:
        """Passing the local slug raises RepoUnavailable about the local read path."""
        registry = tmp_path / 'registry'
        repo = _make_indexed_repo(tmp_path, 'local')
        _write_registry_entry(registry, 'local', repo)
        resolver = RepoResolver(registry_dir=registry, local_repo_path=repo)

        with pytest.raises(
            RepoUnavailable,
            match='local repo must be accessed via the local read path',
        ):
            with open_foreign_backend(resolver, 'local'):
                pass

    def test_open_foreign_unknown_slug_raises_repo_unavailable(
        self, tmp_path: Path
    ) -> None:
        """Unregistered slug raises RepoUnavailable with reason 'not registered'."""
        registry = tmp_path / 'registry'
        resolver = RepoResolver(registry_dir=registry)

        with pytest.raises(RepoUnavailable, match='not registered'):
            with open_foreign_backend(resolver, 'ghost'):
                pass

    def test_open_foreign_axon_dir_deleted_raises_repo_unavailable(
        self, tmp_path: Path
    ) -> None:
        """Delete the .axon dir post-register; open_foreign_backend raises RepoUnavailable."""
        registry = tmp_path / 'registry'
        foreign = _make_indexed_repo(tmp_path, 'foreign')
        _write_registry_entry(registry, 'foreign', foreign)
        # Remove the DB so initialize fails.
        shutil.rmtree(foreign / '.axon', ignore_errors=True)
        resolver = RepoResolver(registry_dir=registry)

        with pytest.raises(RepoUnavailable) as exc_info:
            with open_foreign_backend(resolver, 'foreign'):
                pass

        assert exc_info.value.slug == 'foreign'

    def test_open_foreign_does_not_cache_handles_across_calls(
        self, tmp_path: Path
    ) -> None:
        """Two successive opens for the same slug yield different KuzuBackend instances."""
        registry = tmp_path / 'registry'
        foreign = _make_indexed_repo(tmp_path, 'foreign')
        _write_registry_entry(registry, 'foreign', foreign)
        resolver = RepoResolver(registry_dir=registry)

        with open_foreign_backend(resolver, 'foreign') as b1:
            id1 = id(b1)

        with open_foreign_backend(resolver, 'foreign') as b2:
            id2 = id(b2)

        assert id1 != id2

    def test_open_foreign_does_not_cache_failures_across_calls(
        self, tmp_path: Path
    ) -> None:
        """After a failed open, re-registering the repo makes the next call succeed."""
        registry = tmp_path / 'registry'
        # First attempt: no DB exists.
        ghost = tmp_path / 'ghost'
        ghost.mkdir()
        (ghost / '.axon').mkdir()
        _write_registry_entry(registry, 'ghost', ghost)
        resolver = RepoResolver(registry_dir=registry)

        with pytest.raises(RepoUnavailable):
            with open_foreign_backend(resolver, 'ghost'):
                pass

        # Fix: create a real DB at the path.
        real = _make_indexed_repo(tmp_path, 'real-ghost')
        # Re-write registry entry to point to a real DB.
        _write_registry_entry(registry, 'ghost', real)

        # Second call must succeed (no sticky failure).
        with open_foreign_backend(resolver, 'ghost') as backend:
            assert backend._db is not None

    def test_open_foreign_propagates_oserror_as_repo_unavailable(
        self, tmp_path: Path
    ) -> None:
        """OSError from KuzuBackend.initialize surfaces as RepoUnavailable."""
        registry = tmp_path / 'registry'
        foreign = _make_indexed_repo(tmp_path, 'foreign')
        _write_registry_entry(registry, 'foreign', foreign)
        resolver = RepoResolver(registry_dir=registry)

        with patch.object(
            KuzuBackend, 'initialize', side_effect=OSError('disk error')
        ):
            with pytest.raises(RepoUnavailable, match='disk error'):
                with open_foreign_backend(resolver, 'foreign'):
                    pass

    def test_open_foreign_dispatch_retry_policy_fails_fast(
        self, tmp_path: Path
    ) -> None:
        """Default max_retries and retry_delay match _DISPATCH constants."""
        registry = tmp_path / 'registry'
        foreign = _make_indexed_repo(tmp_path, 'foreign')
        _write_registry_entry(registry, 'foreign', foreign)
        resolver = RepoResolver(registry_dir=registry)

        captured: dict = {}

        original_initialize = KuzuBackend.initialize

        def spy_initialize(
            self: KuzuBackend, path: Path, **kwargs: object
        ) -> None:
            captured['max_retries'] = kwargs.get('max_retries')
            captured['retry_delay'] = kwargs.get('retry_delay')
            original_initialize(self, path, **kwargs)

        with patch.object(KuzuBackend, 'initialize', spy_initialize):
            with open_foreign_backend(resolver, 'foreign'):
                pass

        assert captured['max_retries'] == _DISPATCH_OPEN_RETRIES
        assert captured['retry_delay'] == _DISPATCH_OPEN_RETRY_DELAY

    def test_open_foreign_flush_retry_policy_used_with_explicit_kwargs(
        self, tmp_path: Path
    ) -> None:
        """Explicit flush-policy kwargs reach KuzuBackend.initialize."""
        registry = tmp_path / 'registry'
        foreign = _make_indexed_repo(tmp_path, 'foreign')
        _write_registry_entry(registry, 'foreign', foreign)
        resolver = RepoResolver(registry_dir=registry)

        captured: dict = {}

        original_initialize = KuzuBackend.initialize

        def spy_initialize(
            self: KuzuBackend, path: Path, **kwargs: object
        ) -> None:
            captured['max_retries'] = kwargs.get('max_retries')
            captured['retry_delay'] = kwargs.get('retry_delay')
            original_initialize(self, path, **kwargs)

        with patch.object(KuzuBackend, 'initialize', spy_initialize):
            with open_foreign_backend(
                resolver,
                'foreign',
                max_retries=_FLUSH_OPEN_RETRIES,
                retry_delay=_FLUSH_OPEN_RETRY_DELAY,
            ):
                pass

        assert captured['max_retries'] == _FLUSH_OPEN_RETRIES
        assert captured['retry_delay'] == _FLUSH_OPEN_RETRY_DELAY


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
