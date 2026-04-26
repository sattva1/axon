"""Tests for the Phase 4 handle_list_repos upgrade.

Verifies that handle_list_repos uses RepoResolver rather than Path.glob,
produces per-entry freshness/reachability output, marks the local entry,
appends a usage-hint footer, and correctly surfaces unreachable foreign repos.

These tests spin up real KuzuBackend instances where the pool needs to probe
reachability, keeping them as lightweight as possible.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from axon.core.drift import DriftCache
from axon.core.repos import RepoPool, RepoResolver, RegistryEntry
from axon.core.storage.kuzu_backend import KuzuBackend
from axon.mcp.tools import handle_list_repos


# ---------------------------------------------------------------------------
# Helpers (mirrors the pattern from tests/core/test_repos.py)
# ---------------------------------------------------------------------------


def _make_indexed_repo(tmp_path: Path, name: str) -> Path:
    """Create a minimal indexed repo with a real Kuzu DB under .axon/kuzu."""
    repo = tmp_path / name
    (repo / '.axon').mkdir(parents=True)
    backend = KuzuBackend()
    backend.initialize(repo / '.axon' / 'kuzu')
    backend.close()
    return repo


def _write_registry_entry(
    registry_dir: Path, slug: str, repo_path: Path
) -> None:
    """Write a minimal RegistryEntry into the registry for a given slug."""
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


@pytest.fixture()
def two_repo_setup(tmp_path: Path):
    """Create a registry with one local and one foreign indexed repo.

    Returns (registry, local_repo, foreign_repo, resolver, pool, drift_cache).
    """
    registry = tmp_path / 'registry'
    local_repo = _make_indexed_repo(tmp_path, 'local-repo')
    foreign_repo = _make_indexed_repo(tmp_path, 'foreign-repo')
    _write_registry_entry(registry, 'local-repo', local_repo)
    _write_registry_entry(registry, 'foreign-repo', foreign_repo)
    resolver = RepoResolver(registry_dir=registry, local_repo_path=local_repo)
    pool = RepoPool(resolver)
    drift_cache = DriftCache()
    return registry, local_repo, foreign_repo, resolver, pool, drift_cache


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestListReposUsesResolver:
    """handle_list_repos enumerates via RepoResolver, not Path.glob."""

    def test_list_known_is_called(self, two_repo_setup: tuple) -> None:
        """RepoResolver.list_known is called; Path.glob is not."""
        _, _, _, resolver, pool, drift_cache = two_repo_setup
        list_known_calls: list[int] = []
        original_list_known = resolver.list_known

        def tracked_list_known() -> list:
            result = original_list_known()
            list_known_calls.append(1)
            return result

        with (
            patch.object(
                resolver, 'list_known', side_effect=tracked_list_known
            ),
            patch.object(
                Path,
                'glob',
                side_effect=AssertionError('Path.glob must not be called'),
            ),
        ):
            handle_list_repos(
                resolver=resolver, pool=pool, drift_cache=drift_cache
            )

        assert list_known_calls, 'resolver.list_known was not called'


class TestListReposOutput:
    """Per-entry fields appear in the formatted output."""

    def test_freshness_per_entry(self, two_repo_setup: tuple) -> None:
        """Output contains a Freshness line for each entry."""
        _, _, _, resolver, pool, drift_cache = two_repo_setup
        result = handle_list_repos(
            resolver=resolver,
            pool=pool,
            drift_cache=drift_cache,
            local_slug='local-repo',
        )
        assert result.count('Freshness:') == 2

    def test_local_entry_marked(self, two_repo_setup: tuple) -> None:
        """The local repo entry carries the (LOCAL) marker."""
        _, _, _, resolver, pool, drift_cache = two_repo_setup
        result = handle_list_repos(
            resolver=resolver,
            pool=pool,
            drift_cache=drift_cache,
            local_slug='local-repo',
        )
        assert '(LOCAL)' in result

    def test_usage_hint_footer(self, two_repo_setup: tuple) -> None:
        """Output ends with the repo=<slug> usage hint."""
        _, _, _, resolver, pool, drift_cache = two_repo_setup
        result = handle_list_repos(
            resolver=resolver, pool=pool, drift_cache=drift_cache
        )
        assert 'repo=<slug>' in result


class TestListReposReachability:
    """Reachability is surfaced per entry."""

    def test_reachable_yes_for_local(self, two_repo_setup: tuple) -> None:
        """Local entry shows Reachable: yes."""
        _, local_repo, _, resolver, pool, drift_cache = two_repo_setup
        result = handle_list_repos(
            resolver=resolver,
            pool=pool,
            drift_cache=drift_cache,
            local_slug='local-repo',
        )
        # The local entry block appears before foreign entries; check presence.
        assert 'Reachable: yes' in result

    def test_reachable_no_for_unavailable_foreign(
        self, tmp_path: Path
    ) -> None:
        """Foreign repo without a Kuzu DB shows Reachable: no."""
        registry = tmp_path / 'registry'
        local_repo = _make_indexed_repo(tmp_path, 'local-repo')
        _write_registry_entry(registry, 'local-repo', local_repo)

        # Foreign repo has the directory but no .axon/kuzu (DB missing).
        foreign_repo = tmp_path / 'ghost-repo'
        foreign_repo.mkdir()
        (foreign_repo / '.axon').mkdir()
        _write_registry_entry(registry, 'ghost-repo', foreign_repo)

        resolver = RepoResolver(
            registry_dir=registry, local_repo_path=local_repo
        )
        pool = RepoPool(resolver)
        drift_cache = DriftCache()
        result = handle_list_repos(
            resolver=resolver,
            pool=pool,
            drift_cache=drift_cache,
            local_slug='local-repo',
        )
        assert 'Reachable: no' in result
