from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from axon.core.graph.model import (
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.ingestion.pipeline import (
    PipelineResult,
    reindex_files,
    run_pipeline,
)
from axon.core.ingestion.walker import FileEntry
from axon.core.storage.kuzu_backend import KuzuBackend


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """Create a small Python repository under a temporary directory.

    Layout::

        tmp_repo/
        +-- src/
            +-- main.py    (imports validate from auth, calls it)
            +-- auth.py    (imports helper from utils, calls it)
            +-- utils.py   (standalone helper function)
    """
    src = tmp_path / "src"
    src.mkdir()

    (src / "main.py").write_text(
        "from .auth import validate\n"
        "\n"
        "def main():\n"
        "    validate()\n",
        encoding="utf-8",
    )

    (src / "auth.py").write_text(
        "from .utils import helper\n"
        "\n"
        "def validate():\n"
        "    helper()\n",
        encoding="utf-8",
    )

    (src / "utils.py").write_text(
        "def helper():\n"
        "    pass\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture()
def storage(tmp_path: Path) -> KuzuBackend:
    """Provide an initialised KuzuBackend for testing."""
    db_path = tmp_path / "test_db"
    backend = KuzuBackend()
    backend.initialize(db_path)
    yield backend
    backend.close()


class TestRunPipelineBasic:
    def test_run_pipeline_basic(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        _, result = run_pipeline(tmp_repo, storage)

        assert isinstance(result, PipelineResult)
        assert result.duration_seconds > 0.0


class TestRunPipelineFileCount:
    def test_run_pipeline_file_count(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        _, result = run_pipeline(tmp_repo, storage)

        assert result.files == 3


class TestRunPipelineFindsSymbols:
    def test_run_pipeline_finds_symbols(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        _, result = run_pipeline(tmp_repo, storage)

        assert result.symbols >= 3


class TestRunPipelineFindsRelationships:
    def test_run_pipeline_finds_relationships(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        _, result = run_pipeline(tmp_repo, storage)

        assert result.relationships > 0


class TestRunPipelineProgressCallback:
    def test_run_pipeline_progress_callback(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        calls: list[tuple[str, float]] = []

        def callback(phase: str, pct: float) -> None:
            calls.append((phase, pct))

        run_pipeline(tmp_repo, storage, progress_callback=callback)

        # At minimum, every phase should report start (0.0) and end (1.0).
        assert len(calls) >= 2

        phase_names = {name for name, _ in calls}
        assert "Walking files" in phase_names
        assert "Processing structure" in phase_names
        assert "Parsing code" in phase_names
        assert "Resolving imports" in phase_names
        assert "Resolving relationships" in phase_names
        assert "Loading to storage" in phase_names


class TestRunPipelineLoadsToStorage:
    def test_run_pipeline_loads_to_storage(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        run_pipeline(tmp_repo, storage)

        # File nodes should be stored. The walker produces paths relative to
        # repo root, so "src/main.py" should exist as a File node.
        node = storage.get_node("file:src/main.py:")
        assert node is not None
        assert node.name == "main.py"


@pytest.fixture()
def rich_repo(tmp_path: Path) -> Path:
    """Create a repository with classes and type annotations for phases 7-11.

    Layout::

        rich_repo/
        +-- src/
            +-- models.py   (User class)
            +-- auth.py     (validate function using User type, calls check)
            +-- check.py    (check function, calls verify)
            +-- verify.py   (verify function -- standalone, no callers)
            +-- unused.py   (orphan function -- dead code candidate)
    """
    src = tmp_path / "src"
    src.mkdir()

    (src / "models.py").write_text(
        "class User:\n"
        "    def __init__(self, name: str):\n"
        "        self.name = name\n",
        encoding="utf-8",
    )

    (src / "auth.py").write_text(
        "from .models import User\n"
        "from .check import check\n"
        "\n"
        "def validate(user: User) -> bool:\n"
        "    return check(user)\n",
        encoding="utf-8",
    )

    (src / "check.py").write_text(
        "from .verify import verify\n"
        "\n"
        "def check(obj) -> bool:\n"
        "    return verify(obj)\n",
        encoding="utf-8",
    )

    (src / "verify.py").write_text(
        "def verify(obj) -> bool:\n"
        "    return obj is not None\n",
        encoding="utf-8",
    )

    (src / "unused.py").write_text(
        "def orphan_func():\n"
        "    pass\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture()
def rich_storage(tmp_path: Path) -> KuzuBackend:
    """Provide an initialised KuzuBackend for the rich repo tests."""
    db_path = tmp_path / "rich_db"
    backend = KuzuBackend()
    backend.initialize(db_path)
    yield backend
    backend.close()


class TestRunPipelineFullPhases:
    def test_run_pipeline_full_phases(
        self, rich_repo: Path, rich_storage: KuzuBackend
    ) -> None:
        _, result = run_pipeline(rich_repo, rich_storage)

        # Basic sanity checks.
        assert isinstance(result, PipelineResult)
        assert result.files == 5
        assert result.symbols >= 5  # User, __init__, validate, check, verify, orphan_func
        assert result.relationships > 0
        assert result.duration_seconds > 0.0

        # Phase 8 (communities) and Phase 9 (processes) return ints >= 0.
        # The exact count depends on the graph structure, but they must be
        # non-negative integers.
        assert isinstance(result.clusters, int)
        assert result.clusters >= 0

        assert isinstance(result.processes, int)
        assert result.processes >= 0

        # Phase 10 (dead code): orphan_func has no callers and is not a
        # constructor, test function, or dunder -- it should be flagged.
        assert isinstance(result.dead_code, int)
        assert result.dead_code >= 1

        # Phase 11 (coupling): no git repo, so coupling should be 0.
        assert isinstance(result.coupled_pairs, int)
        assert result.coupled_pairs == 0


class TestRunPipelineProgressIncludesNewPhases:
    def test_run_pipeline_progress_includes_new_phases(
        self, rich_repo: Path, rich_storage: KuzuBackend
    ) -> None:
        calls: list[tuple[str, float]] = []

        def callback(phase: str, pct: float) -> None:
            calls.append((phase, pct))

        run_pipeline(rich_repo, rich_storage, progress_callback=callback)

        phase_names = {name for name, _ in calls}

        # Phases 1-4 (sequential).
        assert "Walking files" in phase_names
        assert "Processing structure" in phase_names
        assert "Parsing code" in phase_names
        assert "Resolving imports" in phase_names

        # Phases 5-7 (concurrent calls/heritage/types).
        assert "Resolving relationships" in phase_names

        # Phases 8-11 (global).
        assert "Detecting communities" in phase_names
        assert "Detecting execution flows" in phase_names
        assert "Finding dead code" in phase_names
        assert "Analyzing git history" in phase_names

        # Storage loading (always present).
        assert 'Loading to storage' in phase_names

        # Progress values should be monotonically non-decreasing.
        pcts = [pct for _, pct in calls]
        assert all(a <= b + 1e-9 for a, b in zip(pcts, pcts[1:]))
        # Overall progress should reach near 1.0.
        assert pcts[-1] >= 0.95
        # All values in [0, 1].
        assert all(0.0 <= p <= 1.0 + 1e-9 for p in pcts)


class TestRunPipelineEmbeddings:
    def test_embedding_phase_in_progress(
        self, rich_repo: Path, rich_storage: KuzuBackend
    ) -> None:
        calls: list[tuple[str, float]] = []

        def callback(phase: str, pct: float) -> None:
            calls.append((phase, pct))

        run_pipeline(rich_repo, rich_storage, progress_callback=callback)

        phase_names = {name for name, _ in calls}
        assert "Generating embeddings" in phase_names

    def test_result_symbols_set_even_if_embed_fails(
        self, rich_repo: Path, rich_storage: KuzuBackend
    ) -> None:
        with patch(
            "axon.core.ingestion.pipeline.embed_graph",
            side_effect=RuntimeError("model not found"),
        ):
            _, result = run_pipeline(rich_repo, rich_storage)

        # symbols and relationships are computed before the embedding step
        assert result.symbols >= 5
        assert result.relationships > 0
        assert result.embeddings == 0

    def test_no_storage_skips_embedding(self, rich_repo: Path) -> None:
        calls: list[tuple[str, float]] = []

        def callback(phase: str, pct: float) -> None:
            calls.append((phase, pct))

        _, result = run_pipeline(
            rich_repo, storage=None, progress_callback=callback
        )

        phase_names = {name for name, _ in calls}
        assert 'Generating embeddings' not in phase_names
        assert result.embeddings == 0


# ---------------------------------------------------------------------------
# Phase 1 - reindex_files preserves metadata_json
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_file_repo(tmp_path: Path) -> Path:
    """Two-file repo: file_a.py calls file_b.py."""
    src = tmp_path / 'src'
    src.mkdir()

    (src / 'file_a.py').write_text(
        'from .file_b import target\n\ndef caller():\n    target()\n',
        encoding='utf-8',
    )
    (src / 'file_b.py').write_text(
        'def target():\n    pass\n', encoding='utf-8'
    )
    return tmp_path


@pytest.fixture()
def two_file_storage(tmp_path: Path) -> KuzuBackend:
    """Initialised KuzuBackend for two_file_repo tests."""
    db_path = tmp_path / 'two_file_db'
    backend = KuzuBackend()
    backend.initialize(db_path)
    yield backend
    backend.close()


class TestReindexFilesPreservesMetadataJson:
    """T5 - reindex_files does not drop metadata_json on cross-file edges."""

    def test_metadata_json_survives_reindex(
        self, two_file_repo: Path, two_file_storage: KuzuBackend
    ) -> None:
        """Cross-file edge extra props survive a reindex of the target file.

        Bulk-loads a graph with an A->B edge carrying dispatch_kind and
        confidence. Re-indexes file B (simulating a content change). After
        reindex, the A->B edge must still carry the original extra props,
        because reindex_files saves inbound cross-file edges via
        get_inbound_cross_file_edges before clearing the changed file, and
        A->B is inbound to B with source A not in the changed set.
        """
        run_pipeline(two_file_repo, two_file_storage, embeddings=False)

        # Locate the caller (in src/file_a.py) and target (in src/file_b.py).
        caller_id = generate_id(NodeLabel.FUNCTION, 'src/file_a.py', 'caller')
        target_id = generate_id(NodeLabel.FUNCTION, 'src/file_b.py', 'target')

        assert two_file_storage.get_node(caller_id) is not None
        assert two_file_storage.get_node(target_id) is not None

        # Replace the pipeline-generated edge with one carrying extra metadata.
        two_file_storage.execute_raw('MATCH ()-[r:CodeRelation]->() DELETE r')
        meta_rel = GraphRelationship(
            id=f'CALLS:{caller_id}->{target_id}',
            type=RelType.CALLS,
            source=caller_id,
            target=target_id,
            properties={'dispatch_kind': 'test_marker', 'confidence': 0.75},
        )
        two_file_storage.add_relationships([meta_rel])

        # Modify file_b.py (add a comment) and reindex it. The A->B edge is
        # inbound to B, so get_inbound_cross_file_edges('src/file_b.py',
        # exclude={'src/file_b.py'}) will find it (source A is not excluded)
        # and preserve its metadata across the reindex.
        new_content = '# updated\ndef target():\n    pass\n'
        (two_file_repo / 'src' / 'file_b.py').write_text(
            new_content, encoding='utf-8'
        )
        entry = FileEntry(
            path='src/file_b.py', content=new_content, language='python'
        )
        reindex_files(
            [entry], two_file_repo, two_file_storage, rebuild_fts=False
        )

        # The A->B edge with metadata must still be present.
        edges = two_file_storage.get_inbound_cross_file_edges('src/file_b.py')
        assert edges, 'No cross-file edges found after reindex'

        meta_edge = next(
            (
                e
                for e in edges
                if e.source == caller_id and e.target == target_id
            ),
            None,
        )
        assert meta_edge is not None, (
            'Cross-file edge from caller to target not found after reindex'
        )
        props = meta_edge.properties or {}
        assert props.get('dispatch_kind') == 'test_marker'
