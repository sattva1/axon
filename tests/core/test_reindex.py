"""Tests for the public reindex helpers in axon.core.ingestion.reindex."""

from __future__ import annotations

from pathlib import Path

import pytest
from watchfiles import Change

from axon.core.ingestion.reindex import reindex_files
from axon.core.meta import load_meta
from axon.core.storage.kuzu_backend import KuzuBackend


@pytest.fixture()
def indexed_repo(tmp_path: Path, kuzu_backend: KuzuBackend) -> Path:
    """Minimal Python repo with one indexed file."""
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'app.py').write_text(
        'def hello():\n    return "hello"\n', encoding='utf-8'
    )
    return tmp_path


class TestReindexFiles:
    """reindex_files helper tests."""

    def test_no_longer_writes_last_incremental_at(
        self, indexed_repo: Path, kuzu_backend: KuzuBackend
    ) -> None:
        """reindex_files must NOT write last_incremental_at (Major #1 fix)."""
        target = indexed_repo / 'src' / 'app.py'
        changes: list[tuple[Change, Path]] = [(Change.modified, target)]

        meta_before = load_meta(indexed_repo)

        reindex_files(changes, indexed_repo, kuzu_backend)

        meta_after = load_meta(indexed_repo)
        assert (
            meta_after.last_incremental_at == meta_before.last_incremental_at
        )

    def test_returns_set_of_reindexed_paths(
        self, indexed_repo: Path, kuzu_backend: KuzuBackend
    ) -> None:
        """reindex_files returns (count, set_of_relative_paths)."""
        target = indexed_repo / 'src' / 'app.py'
        changes: list[tuple[Change, Path]] = [(Change.modified, target)]

        count, paths = reindex_files(changes, indexed_repo, kuzu_backend)

        assert count == 1
        assert paths == {'src/app.py'}
