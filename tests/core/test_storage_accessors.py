"""Tests for KuzuBackend.get_accessors (Phase 5 ACCESSES edge support)."""

from __future__ import annotations

from pathlib import Path

import pytest

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.storage.kuzu_backend import KuzuBackend


@pytest.fixture()
def backend(tmp_path: Path) -> KuzuBackend:
    """KuzuBackend in a temporary directory."""
    b = KuzuBackend()
    b.initialize(tmp_path / 'test_db')
    yield b
    b.close()


def _enum_member_node(
    name: str,
    parent: str,
    file_path: str = 'src/status.py',
    line: int = 5,
) -> GraphNode:
    """Build an ENUM_MEMBER GraphNode."""
    return GraphNode(
        id=generate_id(NodeLabel.ENUM_MEMBER, file_path, f'{parent}.{name}'),
        label=NodeLabel.ENUM_MEMBER,
        name=name,
        file_path=file_path,
        start_line=line,
        end_line=line,
        class_name=parent,
    )


def _function_node(
    name: str,
    file_path: str = 'src/worker.py',
    start_line: int = 1,
    end_line: int = 20,
) -> GraphNode:
    """Build a FUNCTION GraphNode."""
    return GraphNode(
        id=generate_id(NodeLabel.FUNCTION, file_path, name),
        label=NodeLabel.FUNCTION,
        name=name,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
    )


def _accesses_rel(
    source_id: str,
    target_id: str,
    mode: str = 'read',
    confidence: float = 1.0,
) -> GraphRelationship:
    """Build an ACCESSES GraphRelationship."""
    return GraphRelationship(
        id=f'{source_id}->accesses->{target_id}@{mode}',
        type=RelType.ACCESSES,
        source=source_id,
        target=target_id,
        properties={'access_mode': mode, 'confidence': confidence},
    )


def _build_graph_with_accesses(
    modes: list[str],
) -> tuple[KnowledgeGraph, GraphNode, list[GraphNode]]:
    """Build a graph with one ENUM_MEMBER and N accessor functions."""
    graph = KnowledgeGraph()
    member = _enum_member_node('PENDING', 'Status')
    graph.add_node(member)

    accessors = []
    for i, mode in enumerate(modes):
        fn = _function_node(f'worker_{i}', start_line=i * 10 + 1, end_line=i * 10 + 9)
        graph.add_node(fn)
        graph.add_relationship(_accesses_rel(fn.id, member.id, mode=mode))
        accessors.append(fn)

    return graph, member, accessors


class TestGetAccessors:
    """KuzuBackend.get_accessors returns correct (node, mode, confidence) triples."""

    def test_round_trip_all_modes(self, backend: KuzuBackend) -> None:
        """Stored ACCESSES edges with varied modes are all returned."""
        graph, member, _ = _build_graph_with_accesses(['read', 'write', 'both'])
        backend.bulk_load(graph)

        rows = backend.get_accessors(member.id)
        assert len(rows) == 3
        modes = {mode for _, mode, _ in rows}
        assert modes == {'read', 'write', 'both'}

    def test_filter_by_mode_read(self, backend: KuzuBackend) -> None:
        """get_accessors(mode='read') returns only read-mode edges."""
        graph, member, _ = _build_graph_with_accesses(['read', 'write', 'both'])
        backend.bulk_load(graph)

        rows = backend.get_accessors(member.id, mode='read')
        assert len(rows) == 1
        _, mode, _ = rows[0]
        assert mode == 'read'

    def test_filter_by_mode_write(self, backend: KuzuBackend) -> None:
        """get_accessors(mode='write') returns only write-mode edges."""
        graph, member, _ = _build_graph_with_accesses(['read', 'write'])
        backend.bulk_load(graph)

        rows = backend.get_accessors(member.id, mode='write')
        assert len(rows) == 1
        _, mode, _ = rows[0]
        assert mode == 'write'

    def test_non_existent_node_returns_empty(self, backend: KuzuBackend) -> None:
        """get_accessors on an unknown node ID returns empty list."""
        graph = KnowledgeGraph()
        backend.bulk_load(graph)

        rows = backend.get_accessors('enum_member:src/fake.py:X.Y')
        assert rows == []

    def test_node_with_no_accesses_returns_empty(
        self, backend: KuzuBackend
    ) -> None:
        """ENUM_MEMBER with no ACCESSES edges returns empty list."""
        graph = KnowledgeGraph()
        member = _enum_member_node('ACTIVE', 'Status')
        graph.add_node(member)
        backend.bulk_load(graph)

        rows = backend.get_accessors(member.id)
        assert rows == []

    def test_accessor_node_is_returned(self, backend: KuzuBackend) -> None:
        """Returned GraphNode is the accessor (source), not the member."""
        fn = _function_node('my_handler')
        member = _enum_member_node('DONE', 'Status')
        graph = KnowledgeGraph()
        graph.add_node(fn)
        graph.add_node(member)
        graph.add_relationship(_accesses_rel(fn.id, member.id, mode='read'))
        backend.bulk_load(graph)

        rows = backend.get_accessors(member.id)
        assert len(rows) == 1
        node, _, _ = rows[0]
        assert node.name == 'my_handler'

    def test_confidence_preserved(self, backend: KuzuBackend) -> None:
        """Confidence value from the ACCESSES edge is returned correctly."""
        fn = _function_node('handler')
        member = _enum_member_node('PENDING', 'Status')
        graph = KnowledgeGraph()
        graph.add_node(fn)
        graph.add_node(member)
        graph.add_relationship(
            _accesses_rel(fn.id, member.id, mode='read', confidence=0.8)
        )
        backend.bulk_load(graph)

        rows = backend.get_accessors(member.id)
        _, _, conf = rows[0]
        assert conf == pytest.approx(0.8)

    def test_load_graph_round_trips_access_mode(self, tmp_path: Path) -> None:
        """Close and reopen the backend; ACCESSES edges retain access_mode."""
        db_path = tmp_path / 'round_trip_db'

        # Write phase.
        b1 = KuzuBackend()
        b1.initialize(db_path)
        fn = _function_node('processor')
        member = _enum_member_node('QUEUED', 'Job')
        graph = KnowledgeGraph()
        graph.add_node(fn)
        graph.add_node(member)
        graph.add_relationship(_accesses_rel(fn.id, member.id, mode='write'))
        b1.bulk_load(graph)
        b1.close()

        # Read phase.
        b2 = KuzuBackend()
        b2.initialize(db_path, read_only=True)
        loaded = b2.load_graph()
        b2.close()

        rels = loaded.get_relationships_by_type(RelType.ACCESSES)
        assert len(rels) == 1
        assert rels[0].properties.get('access_mode') == 'write'


# ---------------------------------------------------------------------------
# Phase 7 additions: CLASS_ATTRIBUTE and MODULE_CONSTANT accessor tests
# ---------------------------------------------------------------------------


def _class_attribute_node(
    name: str, parent: str, file_path: str = 'src/models.py', line: int = 5
) -> GraphNode:
    """Build a CLASS_ATTRIBUTE GraphNode."""
    return GraphNode(
        id=generate_id(
            NodeLabel.CLASS_ATTRIBUTE, file_path, f'{parent}.{name}'
        ),
        label=NodeLabel.CLASS_ATTRIBUTE,
        name=name,
        file_path=file_path,
        start_line=line,
        end_line=line,
        class_name=parent,
    )


def _module_constant_node(
    name: str, file_path: str = 'src/constants.py', line: int = 3
) -> GraphNode:
    """Build a MODULE_CONSTANT GraphNode."""
    return GraphNode(
        id=generate_id(NodeLabel.MODULE_CONSTANT, file_path, name),
        label=NodeLabel.MODULE_CONSTANT,
        name=name,
        file_path=file_path,
        start_line=line,
        end_line=line,
        class_name='',
    )


class TestGetAccessorsClassAttribute:
    """get_accessors works correctly with CLASS_ATTRIBUTE target nodes."""

    def test_class_attribute_accessors_returned(
        self, backend: KuzuBackend
    ) -> None:
        """ACCESSES edges pointing to a CLASS_ATTRIBUTE are returned."""
        fn = _function_node('handler')
        attr = _class_attribute_node('timeout', 'Config')
        graph = KnowledgeGraph()
        graph.add_node(fn)
        graph.add_node(attr)
        graph.add_relationship(_accesses_rel(fn.id, attr.id, mode='read'))
        backend.bulk_load(graph)

        rows = backend.get_accessors(attr.id)
        assert len(rows) == 1
        node, mode, conf = rows[0]
        assert node.name == 'handler'
        assert mode == 'read'

    def test_class_attribute_mode_filter(self, backend: KuzuBackend) -> None:
        """mode='write' filter works on CLASS_ATTRIBUTE accessor edges."""
        fn_r = _function_node('reader', start_line=1, end_line=5)
        fn_w = _function_node('writer', start_line=10, end_line=15)
        attr = _class_attribute_node('count', 'Counter')
        graph = KnowledgeGraph()
        graph.add_node(fn_r)
        graph.add_node(fn_w)
        graph.add_node(attr)
        graph.add_relationship(_accesses_rel(fn_r.id, attr.id, mode='read'))
        graph.add_relationship(_accesses_rel(fn_w.id, attr.id, mode='write'))
        backend.bulk_load(graph)

        write_rows = backend.get_accessors(attr.id, mode='write')
        assert len(write_rows) == 1
        node, mode, _ = write_rows[0]
        assert node.name == 'writer'
        assert mode == 'write'

    def test_class_attribute_no_accessors(self, backend: KuzuBackend) -> None:
        """CLASS_ATTRIBUTE with no ACCESSES edges returns empty list."""
        graph = KnowledgeGraph()
        attr = _class_attribute_node('field', 'Foo')
        graph.add_node(attr)
        backend.bulk_load(graph)

        rows = backend.get_accessors(attr.id)
        assert rows == []


class TestGetAccessorsModuleConstant:
    """get_accessors works correctly with MODULE_CONSTANT target nodes."""

    def test_module_constant_accessors_returned(
        self, backend: KuzuBackend
    ) -> None:
        """ACCESSES edges pointing to a MODULE_CONSTANT are returned."""
        fn = _function_node('compute')
        const = _module_constant_node('MAX_SIZE')
        graph = KnowledgeGraph()
        graph.add_node(fn)
        graph.add_node(const)
        graph.add_relationship(_accesses_rel(fn.id, const.id, mode='read'))
        backend.bulk_load(graph)

        rows = backend.get_accessors(const.id)
        assert len(rows) == 1
        node, mode, _ = rows[0]
        assert node.name == 'compute'
        assert mode == 'read'

    def test_module_constant_mode_filter(self, backend: KuzuBackend) -> None:
        """mode='read' filter returns only read-mode accesses for MODULE_CONSTANT."""
        fn_r = _function_node('reader', start_line=1, end_line=5)
        fn_b = _function_node('both_user', start_line=10, end_line=15)
        const = _module_constant_node('LIMIT')
        graph = KnowledgeGraph()
        graph.add_node(fn_r)
        graph.add_node(fn_b)
        graph.add_node(const)
        graph.add_relationship(_accesses_rel(fn_r.id, const.id, mode='read'))
        graph.add_relationship(_accesses_rel(fn_b.id, const.id, mode='both'))
        backend.bulk_load(graph)

        read_rows = backend.get_accessors(const.id, mode='read')
        assert len(read_rows) == 1
        assert read_rows[0][1] == 'read'

    def test_module_constant_no_accessors(self, backend: KuzuBackend) -> None:
        """MODULE_CONSTANT with no ACCESSES edges returns empty list."""
        graph = KnowledgeGraph()
        const = _module_constant_node('SENTINEL')
        graph.add_node(const)
        backend.bulk_load(graph)

        rows = backend.get_accessors(const.id)
        assert rows == []

    def test_all_three_member_kinds_mixed(self, backend: KuzuBackend) -> None:
        """Accessors can be retrieved for ENUM_MEMBER, CLASS_ATTRIBUTE, and MODULE_CONSTANT."""
        fn = _function_node('user')
        enum_m = _enum_member_node('RED', 'Color')
        cls_attr = _class_attribute_node('size', 'Widget')
        mod_const = _module_constant_node('TIMEOUT')

        graph = KnowledgeGraph()
        for n in [fn, enum_m, cls_attr, mod_const]:
            graph.add_node(n)
        graph.add_relationship(_accesses_rel(fn.id, enum_m.id, mode='read'))
        graph.add_relationship(_accesses_rel(fn.id, cls_attr.id, mode='read'))
        graph.add_relationship(_accesses_rel(fn.id, mod_const.id, mode='read'))
        backend.bulk_load(graph)

        for target in [enum_m, cls_attr, mod_const]:
            rows = backend.get_accessors(target.id)
            assert len(rows) == 1, f'Expected 1 accessor for {target.label}'
            assert rows[0][0].name == 'user'
