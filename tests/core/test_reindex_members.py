"""Tests for incremental reindex of module constants (Phase 7, SS8.3).

Verifies that reindex_files correctly resolves MODULE_CONSTANT ACCESSES
edges after renaming a constant, both for same-file and cross-file access.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from axon.core.graph.model import NodeLabel, generate_id
from axon.core.ingestion.members import build_module_constant_index
from axon.core.ingestion.pipeline import reindex_files, run_pipeline
from axon.core.ingestion.walker import FileEntry
from axon.core.storage.kuzu_backend import KuzuBackend


@pytest.fixture()
def repo_with_constant(tmp_path: Path) -> Path:
    """Fixture repo with a module constant, a same-file consumer, and a cross-file importer.

    Layout::

        tmp_repo/
        +-- src/
            +-- cfg.py      (defines FOO = 1, function read_foo reads FOO)
            +-- consumer.py (imports FOO from cfg, function use_it reads FOO)
    """
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'cfg.py').write_text(
        'FOO = 1\n\ndef read_foo():\n    return FOO\n', encoding='utf-8'
    )
    (src / 'consumer.py').write_text(
        'from .cfg import FOO\n\ndef use_it():\n    return FOO * 2\n',
        encoding='utf-8',
    )
    return tmp_path


class TestReindexModuleConstantRename:
    """reindex_files correctly updates MODULE_CONSTANT nodes and edges."""

    def test_new_constant_node_present_after_rename(
        self, repo_with_constant: Path, kuzu_backend: KuzuBackend
    ) -> None:
        """After renaming FOO -> BAR in cfg.py, BAR node exists in storage."""
        run_pipeline(repo_with_constant, kuzu_backend, embeddings=False)

        # Rename the constant.
        cfg = repo_with_constant / 'src' / 'cfg.py'
        new_content = 'BAR = 1\n\ndef read_foo():\n    return BAR\n'
        cfg.write_text(new_content, encoding='utf-8')

        entry = FileEntry(
            path='src/cfg.py', content=new_content, language='python'
        )
        reindex_files(
            [entry], repo_with_constant, kuzu_backend, rebuild_fts=False
        )

        bar_id = generate_id(NodeLabel.MODULE_CONSTANT, 'src/cfg.py', 'BAR')
        node = kuzu_backend.get_node(bar_id)
        assert node is not None
        assert node.name == 'BAR'

    def test_old_constant_node_absent_after_rename(
        self, repo_with_constant: Path, kuzu_backend: KuzuBackend
    ) -> None:
        """After renaming FOO -> BAR, the old FOO node is gone from storage."""
        run_pipeline(repo_with_constant, kuzu_backend, embeddings=False)

        cfg = repo_with_constant / 'src' / 'cfg.py'
        new_content = 'BAR = 1\n\ndef read_foo():\n    return BAR\n'
        cfg.write_text(new_content, encoding='utf-8')

        entry = FileEntry(
            path='src/cfg.py', content=new_content, language='python'
        )
        reindex_files(
            [entry], repo_with_constant, kuzu_backend, rebuild_fts=False
        )

        foo_id = generate_id(NodeLabel.MODULE_CONSTANT, 'src/cfg.py', 'FOO')
        node = kuzu_backend.get_node(foo_id)
        assert node is None

    def test_same_file_accesses_resolve_after_rename(
        self, repo_with_constant: Path, kuzu_backend: KuzuBackend
    ) -> None:
        """Same-file function reading BAR gets an ACCESSES edge after rename."""
        run_pipeline(repo_with_constant, kuzu_backend, embeddings=False)

        cfg = repo_with_constant / 'src' / 'cfg.py'
        new_content = 'BAR = 1\n\ndef read_foo():\n    return BAR\n'
        cfg.write_text(new_content, encoding='utf-8')

        entry = FileEntry(
            path='src/cfg.py', content=new_content, language='python'
        )
        reindex_files(
            [entry], repo_with_constant, kuzu_backend, rebuild_fts=False
        )

        bar_id = generate_id(NodeLabel.MODULE_CONSTANT, 'src/cfg.py', 'BAR')
        rows = kuzu_backend.get_accessors(bar_id)
        accessor_names = {node.name for node, _, _ in rows}
        assert 'read_foo' in accessor_names

    def test_index_built_after_new_nodes_invariant(
        self, repo_with_constant: Path, kuzu_backend: KuzuBackend
    ) -> None:
        """The in-memory graph returned by reindex_files contains BAR.

        This verifies the invariant from SS5.3: process_parsing writes new
        member nodes into the in-memory graph before the index builds run,
        so same-file accesses resolve correctly.
        """
        run_pipeline(repo_with_constant, kuzu_backend, embeddings=False)

        cfg = repo_with_constant / 'src' / 'cfg.py'
        new_content = 'BAR = 1\n\ndef read_foo():\n    return BAR\n'
        cfg.write_text(new_content, encoding='utf-8')

        entry = FileEntry(
            path='src/cfg.py', content=new_content, language='python'
        )
        graph = reindex_files(
            [entry], repo_with_constant, kuzu_backend, rebuild_fts=False
        )

        const_idx = build_module_constant_index(graph)
        assert 'BAR' in const_idx.get('src/cfg.py', {})
