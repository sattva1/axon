"""Tests for watcher staleness timestamp behavior.

Phase 3: reindex helpers now live in axon.core.ingestion.reindex.
Major #1: reindex_files no longer writes last_incremental_at;
the flush coordinator owns the single authoritative write per flush.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from watchfiles import Change

from axon.core.embeddings.embedder import _DEFAULT_MODEL
from axon.core.graph.graph import KnowledgeGraph
from axon.core.ingestion.reindex import (
    ensure_current_embeddings,
    reindex_files as _reindex_files,
    run_full_global_phases as _run_full_global_phases,
    run_incremental_global_phases as _run_incremental_global_phases,
)
from axon.core.meta import load_meta, update_meta
from axon.core.storage.base import EMBEDDING_DIMENSIONS


def _make_mock_storage() -> MagicMock:
    """Build a MagicMock that satisfies all StorageBackend calls made by reindex."""
    storage = MagicMock()
    storage.load_graph.return_value = KnowledgeGraph()
    storage.get_nodes_by_label.return_value = []
    storage.get_relationships_by_type.return_value = []
    return storage


class TestReindexFilesTimestampBump:
    def test_no_timestamp_written_by_reindex_files(
        self, tmp_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """reindex_files does NOT write last_incremental_at (Major #1).

        The flush coordinator owns the single authoritative write per flush.
        """
        update_meta_calls: list[dict] = []

        def fake_update_meta(repo: Path, **fields: object) -> None:
            update_meta_calls.append(dict(fields))

        monkeypatch.setattr(
            'axon.core.ingestion.reindex._pipeline_reindex_files',
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.update_meta', fake_update_meta
        )

        storage = _make_mock_storage()
        app_path = tmp_repo / 'src' / 'app.py'

        _reindex_files([(Change.modified, app_path)], tmp_repo, storage)

        for call in update_meta_calls:
            assert 'last_incremental_at' not in call, (
                'reindex_files must not write last_incremental_at (Major #1)'
            )

    def test_no_update_meta_when_no_entries(
        self, tmp_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """reindex_files does not call update_meta when nothing was reindexed."""
        calls: list[str] = []

        monkeypatch.setattr(
            'axon.core.ingestion.reindex.update_meta',
            lambda *a, **kw: calls.append('update_meta'),
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex._pipeline_reindex_files',
            lambda *a, **kw: None,
        )

        storage = _make_mock_storage()
        # Use an unsupported file type so no entries are built.
        readme = tmp_repo / 'README.md'
        readme.write_text('# hello', encoding='utf-8')

        _reindex_files([(Change.modified, readme)], tmp_repo, storage)

        assert 'update_meta' not in calls


class TestRunIncrementalGlobalPhasesSmallChange:
    def test_writes_dead_code_timestamp_only(
        self, tmp_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Small change (<3 files) writes dead_code_last_refreshed_at, not communities."""
        updated_fields: list[dict] = []

        def fake_update_meta(repo: Path, **fields: object) -> None:
            updated_fields.append(dict(fields))

        monkeypatch.setattr(
            'axon.core.ingestion.reindex.update_meta', fake_update_meta
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.process_dead_code', lambda g: 0
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.process_communities', lambda g: 0
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.process_processes', lambda g: 0
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.ensure_current_embeddings',
            lambda st, rp: False,
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.embed_nodes', lambda g, ids: {}
        )

        storage = _make_mock_storage()
        dirty = {'src/app.py'}  # 1 file - small change

        _run_incremental_global_phases(storage, tmp_repo, dirty)

        written_keys = {k for d in updated_fields for k in d}
        assert 'dead_code_last_refreshed_at' in written_keys
        assert 'communities_last_refreshed_at' not in written_keys

    def test_skips_process_communities(
        self, tmp_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Small change branch skips process_communities entirely."""
        communities_called = []

        monkeypatch.setattr(
            'axon.core.ingestion.reindex.update_meta', lambda *a, **kw: None
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.process_dead_code', lambda g: 0
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.process_communities',
            lambda g: communities_called.append(True) or 0,
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.process_processes', lambda g: 0
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.ensure_current_embeddings',
            lambda st, rp: False,
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.embed_nodes', lambda g, ids: {}
        )

        storage = _make_mock_storage()
        _run_incremental_global_phases(storage, tmp_repo, {'src/one.py'})

        assert communities_called == []


class TestRunIncrementalGlobalPhasesFullBranch:
    def test_delegates_to_run_full_global_phases(
        self, tmp_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Change with >=3 dirty files delegates to run_full_global_phases."""
        full_called = []

        def fake_full(repo: Path, storage: object) -> None:
            full_called.append(True)

        monkeypatch.setattr(
            'axon.core.ingestion.reindex.run_full_global_phases', fake_full
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.ensure_current_embeddings',
            lambda st, rp: False,
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.embed_nodes', lambda g, ids: {}
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.update_meta', lambda *a, **kw: None
        )

        storage = _make_mock_storage()
        dirty = {'a.py', 'b.py', 'c.py'}  # 3 files - full branch

        _run_incremental_global_phases(storage, tmp_repo, dirty)

        assert full_called == [True]


class TestRunFullGlobalPhases:
    def test_writes_communities_timestamp_after_process_communities(
        self, tmp_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_full_global_phases writes communities_last_refreshed_at after communities."""
        call_log: list[str] = []

        def fake_process_communities(graph: object) -> int:
            call_log.append('process_communities')
            return 0

        def fake_update_meta(repo: Path, **fields: object) -> None:
            call_log.append(
                'update_meta:'
                + ','.join(sorted(str(k) for k in fields.keys()))
            )

        monkeypatch.setattr(
            'axon.core.ingestion.reindex.process_communities',
            fake_process_communities,
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.process_processes', lambda g: 0
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.process_dead_code', lambda g: 0
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.update_meta', fake_update_meta
        )

        storage = _make_mock_storage()
        _run_full_global_phases(tmp_repo, storage)

        communities_pos = call_log.index('process_communities')
        communities_update_pos = next(
            i
            for i, e in enumerate(call_log)
            if 'communities_last_refreshed_at' in e
        )
        assert communities_pos < communities_update_pos

    def test_writes_dead_code_timestamp_after_process_dead_code(
        self, tmp_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_full_global_phases writes dead_code_last_refreshed_at after dead code."""
        call_log: list[str] = []

        def fake_process_dead_code(graph: object) -> int:
            call_log.append('process_dead_code')
            return 0

        def fake_update_meta(repo: Path, **fields: object) -> None:
            call_log.append(
                'update_meta:'
                + ','.join(sorted(str(k) for k in fields.keys()))
            )

        monkeypatch.setattr(
            'axon.core.ingestion.reindex.process_communities', lambda g: 0
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.process_processes', lambda g: 0
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.process_dead_code',
            fake_process_dead_code,
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.update_meta', fake_update_meta
        )

        storage = _make_mock_storage()
        _run_full_global_phases(tmp_repo, storage)

        dead_code_pos = call_log.index('process_dead_code')
        dead_code_update_pos = next(
            i
            for i, e in enumerate(call_log)
            if 'dead_code_last_refreshed_at' in e
        )
        assert dead_code_pos < dead_code_update_pos

    def test_writes_both_timestamps(
        self, tmp_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_full_global_phases writes both communities and dead_code timestamps."""
        updated_keys: list[str] = []

        def fake_update_meta(repo: Path, **fields: object) -> None:
            updated_keys.extend(str(k) for k in fields.keys())

        monkeypatch.setattr(
            'axon.core.ingestion.reindex.process_communities', lambda g: 0
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.process_processes', lambda g: 0
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.process_dead_code', lambda g: 0
        )
        monkeypatch.setattr(
            'axon.core.ingestion.reindex.update_meta', fake_update_meta
        )

        storage = _make_mock_storage()
        _run_full_global_phases(tmp_repo, storage)

        assert 'communities_last_refreshed_at' in updated_keys
        assert 'dead_code_last_refreshed_at' in updated_keys


class TestEnsureCurrentEmbeddings:
    def test_model_mismatch_updates_meta(self, tmp_path: Path) -> None:
        """Model mismatch triggers re-embed and writes new model to meta.json."""
        update_meta(
            tmp_path, version='1.0', name='repo', embedding_model='old-model'
        )

        storage = MagicMock()
        storage.load_graph.return_value = KnowledgeGraph()

        with patch(
            'axon.core.ingestion.reindex.embed_graph',
            return_value={'n1': [0.1, 0.2]},
        ):
            result = ensure_current_embeddings(storage, tmp_path)

        assert result is True
        meta = load_meta(tmp_path)
        assert meta.embedding_model == _DEFAULT_MODEL
        assert meta.embedding_dimensions == EMBEDDING_DIMENSIONS

    def test_model_mismatch_preserves_other_fields(
        self, tmp_path: Path
    ) -> None:
        """Re-embed does not clobber sibling meta fields."""
        update_meta(
            tmp_path,
            version='2.0',
            name='myrepo',
            embedding_model='old-model',
            stats={'files': 50},
        )

        storage = MagicMock()
        storage.load_graph.return_value = KnowledgeGraph()

        with patch('axon.core.ingestion.reindex.embed_graph', return_value={}):
            ensure_current_embeddings(storage, tmp_path)

        meta = load_meta(tmp_path)
        assert meta.version == '2.0'
        assert meta.name == 'myrepo'
        assert meta.stats['files'] == 50

    def test_missing_meta_returns_false_no_write(self, tmp_path: Path) -> None:
        """No meta.json -> embedding_model is empty -> returns False, no write."""
        storage = MagicMock()

        result = ensure_current_embeddings(storage, tmp_path)

        assert result is False
        storage.load_graph.assert_not_called()
        # No meta.json should have been created.
        assert not (tmp_path / '.axon' / 'meta.json').exists()

    def test_model_matches_returns_false(self, tmp_path: Path) -> None:
        """Matching model returns False without calling load_graph."""
        update_meta(tmp_path, embedding_model=_DEFAULT_MODEL)

        storage = MagicMock()

        result = ensure_current_embeddings(storage, tmp_path)

        assert result is False
        storage.load_graph.assert_not_called()
