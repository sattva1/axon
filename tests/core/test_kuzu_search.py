from __future__ import annotations

import logging
from pathlib import Path

import pytest

from axon.core.graph.model import GraphNode, NodeLabel, generate_id
from axon.core.storage.base import NodeEmbedding
from axon.core.storage.kuzu_backend import KuzuBackend


@pytest.fixture()
def backend(tmp_path: Path) -> KuzuBackend:
    """Return a KuzuBackend initialised in a temporary directory."""
    db_path = tmp_path / "search_test_db"
    b = KuzuBackend()
    b.initialize(db_path)
    yield b
    b.close()


def _make_node(
    label: NodeLabel = NodeLabel.FUNCTION,
    file_path: str = "src/app.py",
    name: str = "my_func",
    content: str = "",
    signature: str = "",
) -> GraphNode:
    """Helper to build a GraphNode with a deterministic id."""
    return GraphNode(
        id=generate_id(label, file_path, name),
        label=label,
        name=name,
        file_path=file_path,
        content=content,
        signature=signature,
    )


def _make_method(
    file_path: str, name: str, class_name: str, content: str = ''
) -> GraphNode:
    """Build a METHOD GraphNode with class_name set.

    Args:
        file_path: Source file path for the method.
        name: Method name.
        class_name: Name of the enclosing class.
        content: Optional body content.

    Returns:
        A GraphNode with label METHOD and class_name populated.
    """
    return GraphNode(
        id=generate_id(NodeLabel.METHOD, file_path, name),
        label=NodeLabel.METHOD,
        name=name,
        file_path=file_path,
        class_name=class_name,
        content=content,
    )


@pytest.fixture()
def dotted_backend(tmp_path: Path) -> KuzuBackend:
    """KuzuBackend seeded with a small class/method graph for dotted-path tests.

    Seed graph:
    - src/a.py: METHOD bar in class Foo (source file, score 2.0).
    - src/b.py: METHOD bar in class Foo (source file, score 2.0).
    - src/c.py: METHOD missing_method in class Foo (source file, score 2.0).
    - src/c.py: FUNCTION bar, no class_name (source file, score 2.0).
    - src/tests/test_a.py: METHOD bar in class Foo (test file, score 1.0).
    """
    db_path = tmp_path / 'dotted_test_db'
    b = KuzuBackend()
    b.initialize(db_path)

    nodes = [
        _make_method('src/a.py', 'bar', 'Foo', 'method body a'),
        _make_method('src/b.py', 'bar', 'Foo', 'method body b'),
        _make_method('src/c.py', 'missing_method', 'Foo', ''),
        _make_node(
            label=NodeLabel.FUNCTION,
            file_path='src/c.py',
            name='bar',
            content='plain function',
        ),
        _make_method('src/tests/test_a.py', 'bar', 'Foo', 'test method body'),
    ]
    b.add_nodes(nodes)
    yield b
    b.close()


class TestFtsSearch:
    def test_exact_name_match(self, backend: KuzuBackend) -> None:
        node = _make_node(name="process_data", content="does stuff")
        backend.add_nodes([node])
        backend.rebuild_fts_indexes()

        results = backend.fts_search("process_data", limit=10)
        assert len(results) >= 1
        top = results[0]
        assert top.node_id == node.id
        assert top.score > 0
        assert top.node_name == "process_data"

    def test_partial_name_match(self, backend: KuzuBackend) -> None:
        node = _make_node(name="process_data_pipeline", content="")
        backend.add_nodes([node])
        backend.rebuild_fts_indexes()

        results = backend.fts_search("process_data", limit=10)
        assert len(results) >= 1
        top = results[0]
        assert top.node_id == node.id
        assert top.score > 0

    def test_content_match(self, backend: KuzuBackend) -> None:
        node = _make_node(name="unrelated_name", content="this calls process_data inside")
        backend.add_nodes([node])
        backend.rebuild_fts_indexes()

        results = backend.fts_search("process_data", limit=10)
        assert len(results) >= 1
        top = results[0]
        assert top.node_id == node.id
        assert top.score > 0

    def test_no_match(self, backend: KuzuBackend) -> None:
        node = _make_node(name="hello", content="world")
        backend.add_nodes([node])
        backend.rebuild_fts_indexes()

        results = backend.fts_search("nonexistent_symbol", limit=10)
        assert results == []

    def test_limit_respected(self, backend: KuzuBackend) -> None:
        nodes = [
            _make_node(name=f"func_{i}", file_path=f"src/f{i}.py", content="common_term")
            for i in range(5)
        ]
        backend.add_nodes(nodes)
        backend.rebuild_fts_indexes()

        results = backend.fts_search("common_term", limit=3)
        assert len(results) == 3

    def test_case_insensitive(self, backend: KuzuBackend) -> None:
        node = _make_node(name="ProcessData", content="")
        backend.add_nodes([node])
        backend.rebuild_fts_indexes()

        results = backend.fts_search("processdata", limit=10)
        assert len(results) >= 1
        assert results[0].node_id == node.id

    def test_score_ordering(self, backend: KuzuBackend) -> None:
        name_match = _make_node(name="target", file_path="src/a.py", content="")
        content_only = _make_node(
            name="unrelated", file_path="src/c.py", content="has target in body"
        )
        backend.add_nodes([name_match, content_only])
        backend.rebuild_fts_indexes()

        results = backend.fts_search("target", limit=10)
        assert len(results) >= 2
        # Name match should score higher than content-only
        assert results[0].node_id == name_match.id
        assert results[0].score >= results[1].score

    def test_result_fields_populated(self, backend: KuzuBackend) -> None:
        node = _make_node(
            label=NodeLabel.CLASS,
            name="MyClass",
            file_path="src/models.py",
            content="class body here",
        )
        backend.add_nodes([node])
        backend.rebuild_fts_indexes()

        results = backend.fts_search("MyClass", limit=10)
        assert len(results) >= 1
        r = results[0]
        assert r.node_name == "MyClass"
        assert r.file_path == "src/models.py"
        assert r.label == "class"
        assert r.snippet != ""

    def test_signature_match(self, backend: KuzuBackend) -> None:
        node = _make_node(
            name="unrelated",
            content="",
            signature="def special_function(x: int) -> str",
        )
        backend.add_nodes([node])
        backend.rebuild_fts_indexes()

        results = backend.fts_search("special_function", limit=10)
        assert len(results) >= 1
        assert results[0].node_id == node.id
        assert results[0].score > 0


class TestEmbeddingsAndVectorSearch:
    def test_store_and_retrieve_by_vector(self, backend: KuzuBackend) -> None:
        # Insert a node so we can populate SearchResult fields.
        node = _make_node(name="embed_func", file_path="src/embed.py", content="body")
        backend.add_nodes([node])

        # Store an embedding for that node.
        vec = [1.0] + [0.0] * 383
        emb = NodeEmbedding(node_id=node.id, embedding=vec)
        backend.store_embeddings([emb])

        # Search with the same vector -- cosine similarity should be 1.0.
        results = backend.vector_search(vec, limit=5)
        assert len(results) >= 1
        top = results[0]
        assert top.node_id == node.id
        assert top.score == pytest.approx(1.0, abs=1e-6)
        assert top.node_name == "embed_func"

    def test_vector_search_empty(self, backend: KuzuBackend) -> None:
        results = backend.vector_search([1.0] + [0.0] * 383, limit=5)
        assert results == []

    def test_vector_search_ranking(self, backend: KuzuBackend) -> None:
        n1 = _make_node(name="close_func", file_path="src/a.py")
        n2 = _make_node(name="far_func", file_path="src/b.py")
        backend.add_nodes([n1, n2])

        # close_func embedding is close to query, far_func is orthogonal.
        close_vec = [0.9, 0.1] + [0.0] * 382
        far_vec = [0.0, 0.0, 1.0] + [0.0] * 381
        backend.store_embeddings([
            NodeEmbedding(node_id=n1.id, embedding=close_vec),
            NodeEmbedding(node_id=n2.id, embedding=far_vec),
        ])

        query_vec = [1.0] + [0.0] * 383
        results = backend.vector_search(query_vec, limit=5)
        assert len(results) == 2
        assert results[0].node_id == n1.id
        assert results[0].score > results[1].score

    def test_vector_search_limit(self, backend: KuzuBackend) -> None:
        nodes = []
        embeddings = []
        for i in range(5):
            n = _make_node(name=f"vfunc_{i}", file_path=f"src/v{i}.py")
            nodes.append(n)
            # All somewhat similar embeddings — one-hot in first 5 dims, rest zeros.
            vec = [0.0] * 384
            vec[i] = 1.0
            embeddings.append(NodeEmbedding(node_id=n.id, embedding=vec))
        backend.add_nodes(nodes)
        backend.store_embeddings(embeddings)

        query_vec = [1.0, 0.5, 0.3, 0.1, 0.0] + [0.0] * 379
        results = backend.vector_search(query_vec, limit=2)
        assert len(results) == 2

    def test_store_embeddings_upsert(self, backend: KuzuBackend) -> None:
        node = _make_node(name="upsert_func", file_path="src/u.py")
        backend.add_nodes([node])

        emb1 = NodeEmbedding(node_id=node.id, embedding=[1.0] + [0.0] * 383)
        backend.store_embeddings([emb1])

        emb2 = NodeEmbedding(node_id=node.id, embedding=[0.0, 1.0] + [0.0] * 382)
        backend.store_embeddings([emb2])

        # Search with [0, 1, 0...] should find it with high similarity.
        query_vec = [0.0, 1.0] + [0.0] * 382
        results = backend.vector_search(query_vec, limit=5)
        assert len(results) == 1
        assert results[0].score == pytest.approx(1.0, abs=1e-6)


class TestFuzzySearch:
    def test_exact_name_returns_result(self, backend: KuzuBackend) -> None:
        node = _make_node(name="validate_user", content="validates user")
        backend.add_nodes([node])

        results = backend.fuzzy_search("validate_user", limit=10)
        assert len(results) >= 1
        assert results[0].node_id == node.id
        assert results[0].score == 1.0  # distance 0 -> score 1.0

    def test_typo_within_distance(self, backend: KuzuBackend) -> None:
        node = _make_node(name="validate_user", content="validates user")
        backend.add_nodes([node])

        # "validte_user" is 1 edit away from "validate_user"
        results = backend.fuzzy_search("validte_user", limit=10, max_distance=2)
        assert len(results) >= 1
        assert results[0].node_id == node.id
        assert results[0].score < 1.0  # distance > 0

    def test_typo_beyond_distance(self, backend: KuzuBackend) -> None:
        node = _make_node(name="validate_user", content="validates user")
        backend.add_nodes([node])

        # "xyz_abc" is many edits away
        results = backend.fuzzy_search("xyz_abc", limit=10, max_distance=2)
        assert len(results) == 0

    def test_fuzzy_score_decreases_with_distance(self, backend: KuzuBackend) -> None:
        node = _make_node(name="process", content="")
        backend.add_nodes([node])

        exact = backend.fuzzy_search("process", limit=10)
        one_off = backend.fuzzy_search("procss", limit=10)  # 1 edit: missing 'e'
        assert len(exact) >= 1
        assert len(one_off) >= 1
        assert exact[0].score > one_off[0].score

    def test_fuzzy_limit(self, backend: KuzuBackend) -> None:
        nodes = [
            _make_node(name=f"func_{i}", file_path=f"src/f{i}.py")
            for i in range(5)
        ]
        backend.add_nodes(nodes)

        results = backend.fuzzy_search("func_0", limit=2, max_distance=2)
        assert len(results) <= 2

    def test_fuzzy_result_fields(self, backend: KuzuBackend) -> None:
        node = _make_node(
            name="my_handler", file_path="src/handlers.py", content="handler body"
        )
        backend.add_nodes([node])

        results = backend.fuzzy_search("my_handler", limit=10)
        assert len(results) >= 1
        r = results[0]
        assert r.node_name == 'my_handler'
        assert r.file_path == 'src/handlers.py'
        assert r.snippet != ''


class TestExactNameSearchDottedPath:
    """Dotted-path (Class.method) resolution via exact_name_search."""

    def test_dotted_method_resolution(
        self, dotted_backend: KuzuBackend
    ) -> None:
        """Foo.bar returns METHOD nodes from both source files and the test file."""
        results = dotted_backend.exact_name_search('Foo.bar', limit=5)
        node_ids = {r.node_id for r in results}
        assert generate_id(NodeLabel.METHOD, 'src/a.py', 'bar') in node_ids
        assert generate_id(NodeLabel.METHOD, 'src/b.py', 'bar') in node_ids
        assert (
            generate_id(NodeLabel.METHOD, 'src/tests/test_a.py', 'bar')
            in node_ids
        )
        # Plain function from src/c.py must NOT appear (no class_name match).
        assert (
            generate_id(NodeLabel.FUNCTION, 'src/c.py', 'bar') not in node_ids
        )

    def test_dotted_no_plain_function_included(
        self, dotted_backend: KuzuBackend
    ) -> None:
        """Dotted search excludes plain functions that have no class_name."""
        results = dotted_backend.exact_name_search('Foo.bar', limit=5)
        plain_fn_id = generate_id(NodeLabel.FUNCTION, 'src/c.py', 'bar')
        assert all(r.node_id != plain_fn_id for r in results)

    def test_unknown_member(self, dotted_backend: KuzuBackend) -> None:
        """Foo.missing returns empty when no method named missing exists in Foo."""
        results = dotted_backend.exact_name_search('Foo.missing', limit=5)
        assert results == []

    def test_unknown_class(self, dotted_backend: KuzuBackend) -> None:
        """Missing.bar returns empty when no class named Missing exists."""
        results = dotted_backend.exact_name_search('Missing.bar', limit=5)
        assert results == []

    def test_plain_name_unchanged(self, dotted_backend: KuzuBackend) -> None:
        """Plain name 'bar' returns all nodes named bar regardless of class_name."""
        results = dotted_backend.exact_name_search('bar', limit=10)
        node_ids = {r.node_id for r in results}
        assert generate_id(NodeLabel.METHOD, 'src/a.py', 'bar') in node_ids
        assert generate_id(NodeLabel.METHOD, 'src/b.py', 'bar') in node_ids
        assert (
            generate_id(NodeLabel.METHOD, 'src/tests/test_a.py', 'bar')
            in node_ids
        )
        # Plain function also returned by plain search.
        assert generate_id(NodeLabel.FUNCTION, 'src/c.py', 'bar') in node_ids

    def test_multi_dot_falls_back_to_last_segment(
        self, dotted_backend: KuzuBackend
    ) -> None:
        """module.Foo.bar falls back to plain search on 'bar'."""
        results_multi = dotted_backend.exact_name_search(
            'module.Foo.bar', limit=10
        )
        results_plain = dotted_backend.exact_name_search('bar', limit=10)
        assert {r.node_id for r in results_multi} == {
            r.node_id for r in results_plain
        }

    def test_multi_dot_logged(
        self, dotted_backend: KuzuBackend, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Multi-dot input triggers a debug log entry."""
        with caplog.at_level(
            logging.DEBUG, logger='axon.core.storage.kuzu_backend'
        ):
            dotted_backend.exact_name_search('module.Foo.bar', limit=5)
        assert any('more than one dot' in r.message for r in caplog.records)

    def test_empty_left_half_falls_through_to_plain(
        self, dotted_backend: KuzuBackend
    ) -> None:
        """'.bar' is treated as a plain-name search on the literal string."""
        results = dotted_backend.exact_name_search('.bar', limit=5)
        # No node has name == ".bar", so result is empty.
        assert results == []

    def test_empty_right_half_falls_through_to_plain(
        self, dotted_backend: KuzuBackend
    ) -> None:
        """'Foo.' is treated as a plain-name search on the literal string."""
        results = dotted_backend.exact_name_search('Foo.', limit=5)
        # No node has name == "Foo.", so result is empty.
        assert results == []

    def test_ordering_source_before_test(
        self, dotted_backend: KuzuBackend
    ) -> None:
        """Source-file candidates (score 2.0) appear before test candidates (1.0)."""
        results = dotted_backend.exact_name_search('Foo.bar', limit=5)
        assert len(results) == 3

        test_file_id = generate_id(
            NodeLabel.METHOD, 'src/tests/test_a.py', 'bar'
        )
        # Last result must be the test-file candidate.
        assert results[-1].node_id == test_file_id
        # All preceding results must have higher scores.
        assert all(r.score > results[-1].score for r in results[:-1])

    def test_ordering_lexicographic_tiebreak(
        self, dotted_backend: KuzuBackend
    ) -> None:
        """Among source-file candidates, node_id lexicographic order is applied."""
        results = dotted_backend.exact_name_search('Foo.bar', limit=5)
        src_results = [r for r in results if r.score == 2.0]
        src_ids = [r.node_id for r in src_results]
        assert src_ids == sorted(src_ids)

    def test_limit_honored(self, dotted_backend: KuzuBackend) -> None:
        """limit=1 returns exactly the single highest-ranked source candidate."""
        results = dotted_backend.exact_name_search('Foo.bar', limit=1)
        assert len(results) == 1
        # The highest-ranked is the lexicographically first source-file method.
        expected_id = generate_id(NodeLabel.METHOD, 'src/a.py', 'bar')
        assert results[0].node_id == expected_id
        assert results[0].score == 2.0
