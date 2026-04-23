"""Tests for Phase 5 enum index building and member access processing."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.ingestion.members import (
    EnumIndex,
    build_enum_index,
    process_member_accesses,
)
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.parsers.base import MemberAccess, ParseResult


def _make_enum_member(
    name: str,
    parent: str,
    file_path: str = 'src/status.py',
    line: int = 5,
) -> GraphNode:
    """Build an ENUM_MEMBER GraphNode."""
    node_id = generate_id(
        NodeLabel.ENUM_MEMBER, file_path, f'{parent}.{name}'
    )
    return GraphNode(
        id=node_id,
        label=NodeLabel.ENUM_MEMBER,
        name=name,
        file_path=file_path,
        start_line=line,
        end_line=line,
        class_name=parent,
    )


def _make_function(
    name: str,
    file_path: str = 'src/worker.py',
    start_line: int = 1,
    end_line: int = 10,
) -> GraphNode:
    """Build a FUNCTION GraphNode with correct id convention."""
    return GraphNode(
        id=generate_id(NodeLabel.FUNCTION, file_path, name),
        label=NodeLabel.FUNCTION,
        name=name,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
    )


def _make_parse_data(
    file_path: str,
    accesses: list[MemberAccess] | None = None,
) -> FileParseData:
    """Build FileParseData carrying the given member accesses."""
    result = ParseResult()
    result.member_accesses = accesses or []
    return FileParseData(
        file_path=file_path,
        language='python',
        parse_result=result,
    )


class TestBuildEnumIndex:
    """build_enum_index produces the correct nested dict structure."""

    def test_empty_graph_returns_empty_index(self) -> None:
        """No ENUM_MEMBER nodes produces an empty index."""
        graph = KnowledgeGraph()
        idx = build_enum_index(graph)
        assert idx == {}

    def test_single_enum_three_members(self) -> None:
        """Three members from one enum are accessible by parent + name."""
        graph = KnowledgeGraph()
        members = [
            _make_enum_member('RED', 'Color'),
            _make_enum_member('GREEN', 'Color'),
            _make_enum_member('BLUE', 'Color'),
        ]
        for m in members:
            graph.add_node(m)

        idx = build_enum_index(graph)
        assert 'Color' in idx
        for name in ('RED', 'GREEN', 'BLUE'):
            assert name in idx['Color']
            assert len(idx['Color'][name]) == 1

    def test_same_parent_name_across_files_merged(self) -> None:
        """Two enums named Color in different files share a parent key."""
        graph = KnowledgeGraph()
        m1 = _make_enum_member('RED', 'Color', file_path='src/a.py')
        m2 = _make_enum_member('RED', 'Color', file_path='src/b.py')
        graph.add_node(m1)
        graph.add_node(m2)

        idx = build_enum_index(graph)
        assert len(idx['Color']['RED']) == 2

    def test_different_parents_separated(self) -> None:
        """Members from two different enums are indexed under separate keys."""
        graph = KnowledgeGraph()
        graph.add_node(_make_enum_member('NORTH', 'Direction'))
        graph.add_node(_make_enum_member('ACTIVE', 'Status'))

        idx = build_enum_index(graph)
        assert 'NORTH' in idx.get('Direction', {})
        assert 'ACTIVE' in idx.get('Status', {})
        assert 'ACTIVE' not in idx.get('Direction', {})


class TestProcessMemberAccesses:
    """process_member_accesses emits correct ACCESSES edges and logs."""

    def _graph_with_member_and_caller(
        self,
        member_file: str = 'src/status.py',
        caller_file: str = 'src/worker.py',
    ) -> tuple[KnowledgeGraph, EnumIndex, str, str]:
        """Return (graph, enum_index, member_id, caller_id)."""
        graph = KnowledgeGraph()
        member = _make_enum_member(
            'PENDING', 'Status', file_path=member_file, line=5
        )
        graph.add_node(member)

        caller = _make_function(
            'process_job', file_path=caller_file, start_line=1, end_line=20
        )
        graph.add_node(caller)

        idx = build_enum_index(graph)
        return graph, idx, member.id, caller.id

    def test_known_enum_access_emits_edge(self) -> None:
        """Resolved access produces one ACCESSES relationship."""
        graph, idx, member_id, caller_id = (
            self._graph_with_member_and_caller()
        )
        access = MemberAccess(
            parent='Status', name='PENDING', line=10, mode='read'
        )
        parse_data = [_make_parse_data('src/worker.py', [access])]

        emitted = process_member_accesses(parse_data, graph, idx)
        assert emitted == 1
        rels = graph.get_relationships_by_type(RelType.ACCESSES)
        assert len(rels) == 1
        assert rels[0].source == caller_id
        assert rels[0].target == member_id

    def test_edge_carries_access_mode_and_confidence(self) -> None:
        """Emitted edge has access_mode and confidence in properties."""
        graph, idx, _, _ = self._graph_with_member_and_caller()
        access = MemberAccess(
            parent='Status', name='PENDING', line=10, mode='write'
        )
        parse_data = [_make_parse_data('src/worker.py', [access])]

        process_member_accesses(parse_data, graph, idx)
        rels = graph.get_relationships_by_type(RelType.ACCESSES)
        assert rels[0].properties['access_mode'] == 'write'
        assert 'confidence' in rels[0].properties

    def test_same_file_confidence_is_one(self) -> None:
        """Access to a member in the same file gets confidence=1.0."""
        graph, idx, _, _ = self._graph_with_member_and_caller(
            member_file='src/status.py', caller_file='src/status.py'
        )
        access = MemberAccess(
            parent='Status', name='PENDING', line=10, mode='read'
        )
        parse_data = [_make_parse_data('src/status.py', [access])]

        process_member_accesses(parse_data, graph, idx)
        rels = graph.get_relationships_by_type(RelType.ACCESSES)
        assert rels[0].properties['confidence'] == 1.0

    def test_cross_file_confidence_is_point_eight(self) -> None:
        """Access to a member in a different file gets confidence=0.8."""
        graph, idx, _, _ = self._graph_with_member_and_caller(
            member_file='src/status.py', caller_file='src/worker.py'
        )
        access = MemberAccess(
            parent='Status', name='PENDING', line=10, mode='read'
        )
        parse_data = [_make_parse_data('src/worker.py', [access])]

        process_member_accesses(parse_data, graph, idx)
        rels = graph.get_relationships_by_type(RelType.ACCESSES)
        assert rels[0].properties['confidence'] == pytest.approx(0.8)

    def test_unknown_parent_no_edge(self) -> None:
        """Access to unknown parent emits no edge but is counted as considered."""
        graph = KnowledgeGraph()
        caller = _make_function('worker')
        graph.add_node(caller)
        idx: EnumIndex = {}

        access = MemberAccess(
            parent='Unknown', name='PENDING', line=5, mode='read'
        )
        parse_data = [_make_parse_data('src/worker.py', [access])]

        emitted = process_member_accesses(parse_data, graph, idx)
        assert emitted == 0
        assert not graph.get_relationships_by_type(RelType.ACCESSES)

    def test_access_in_function_uses_function_as_caller(self) -> None:
        """Caller node is the function containing the access line."""
        graph, idx, member_id, caller_id = (
            self._graph_with_member_and_caller()
        )
        access = MemberAccess(
            parent='Status', name='PENDING', line=10, mode='read'
        )
        parse_data = [_make_parse_data('src/worker.py', [access])]

        process_member_accesses(parse_data, graph, idx)
        rels = graph.get_relationships_by_type(RelType.ACCESSES)
        assert rels[0].source == caller_id

    def test_access_at_top_level_dropped(self) -> None:
        """Access at module scope (no containing symbol) is silently dropped."""
        graph, idx, _, _ = self._graph_with_member_and_caller()
        # Line 100 is outside any function's span (1-20).
        access = MemberAccess(
            parent='Status', name='PENDING', line=100, mode='read'
        )
        parse_data = [_make_parse_data('src/worker.py', [access])]

        emitted = process_member_accesses(parse_data, graph, idx)
        assert emitted == 0

    def test_log_reports_considered_and_emitted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Logger emits considered and emitted counts after processing."""
        graph, idx, _, _ = self._graph_with_member_and_caller()
        access = MemberAccess(
            parent='Status', name='PENDING', line=10, mode='read'
        )
        parse_data = [_make_parse_data('src/worker.py', [access])]

        with caplog.at_level(logging.INFO, logger='axon.core.ingestion.members'):
            process_member_accesses(parse_data, graph, idx)

        combined = ' '.join(caplog.messages)
        assert 'considered' in combined
        assert 'emitted' in combined

    def test_find_containing_symbol_called_with_correct_args(self) -> None:
        """find_containing_symbol is invoked with (line, file_path, index)."""
        graph, idx, _, _ = self._graph_with_member_and_caller()
        access = MemberAccess(
            parent='Status', name='PENDING', line=10, mode='read'
        )
        parse_data = [_make_parse_data('src/worker.py', [access])]

        calls_received: list[tuple] = []
        original = (
            'axon.core.ingestion.members.find_containing_symbol'
        )
        with patch(original, wraps=lambda *a, **kw: None) as spy:
            process_member_accesses(parse_data, graph, idx)
            for call_args in spy.call_args_list:
                calls_received.append(call_args.args)

        assert calls_received
        line_arg, file_arg, _ = calls_received[0]
        assert line_arg == 10
        assert file_arg == 'src/worker.py'
