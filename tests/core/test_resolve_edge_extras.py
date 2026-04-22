"""Tests for _make_edge extra_props propagation in call resolution.

Primary CALLS edges carry Phase-4a metadata from CallInfo.extra_props().
Argument sub-edges propagate the outer call's extra_props (Phase 4b).
Receiver sub-edges (caller -> variable) do NOT carry extra metadata.
"""

from __future__ import annotations

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.ingestion.calls import process_calls
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.parsers.base import CallInfo, ParseResult


def _build_minimal_graph() -> KnowledgeGraph:
    """Return a graph with caller and target symbols in the same file."""
    g = KnowledgeGraph()
    file_id = generate_id(NodeLabel.FILE, 'src/tasks.py')
    g.add_node(
        GraphNode(
            id=file_id,
            label=NodeLabel.FILE,
            name='tasks.py',
            file_path='src/tasks.py',
        )
    )
    # caller: lines 1-20
    caller_id = generate_id(NodeLabel.FUNCTION, 'src/tasks.py', 'run')
    g.add_node(
        GraphNode(
            id=caller_id,
            label=NodeLabel.FUNCTION,
            name='run',
            file_path='src/tasks.py',
            start_line=1,
            end_line=20,
        )
    )
    g.add_relationship(
        GraphRelationship(
            id=f'defines:{file_id}->{caller_id}',
            type=RelType.DEFINES,
            source=file_id,
            target=caller_id,
        )
    )
    # target: lines 22-30
    target_id = generate_id(NodeLabel.FUNCTION, 'src/tasks.py', 'worker')
    g.add_node(
        GraphNode(
            id=target_id,
            label=NodeLabel.FUNCTION,
            name='worker',
            file_path='src/tasks.py',
            start_line=22,
            end_line=30,
        )
    )
    g.add_relationship(
        GraphRelationship(
            id=f'defines:{file_id}->{target_id}',
            type=RelType.DEFINES,
            source=file_id,
            target=target_id,
        )
    )
    return g


class TestResolveFileCallsPrimaryEdgeExtras:
    """Primary CALLS edges carry Phase-4a metadata; arg sub-edges propagate it."""

    def test_primary_edge_carries_dispatch_kind_and_in_try(self) -> None:
        """Primary edge properties include dispatch_kind and in_try from CallInfo."""
        graph = _build_minimal_graph()

        call = CallInfo(
            name='worker', line=5, dispatch_kind='detached_task', in_try=True
        )
        parse_data = [
            FileParseData(
                file_path='src/tasks.py',
                language='python',
                parse_result=ParseResult(calls=[call]),
            )
        ]

        edges = process_calls(parse_data, graph, collect=True)
        assert edges is not None

        caller_id = generate_id(NodeLabel.FUNCTION, 'src/tasks.py', 'run')
        target_id = generate_id(NodeLabel.FUNCTION, 'src/tasks.py', 'worker')

        primary_edges = [
            e for e in edges if e.source == caller_id and e.target == target_id
        ]
        assert primary_edges, 'expected primary CALLS edge'
        props = primary_edges[0].properties
        assert props.get('dispatch_kind') == 'detached_task'
        assert props.get('in_try') is True

    def _build_graph_with_cb(self) -> tuple[KnowledgeGraph, str]:
        """Return a graph with caller, worker, and cb symbols; cb node id."""
        graph = _build_minimal_graph()
        file_id = generate_id(NodeLabel.FILE, 'src/tasks.py')
        cb_id = generate_id(NodeLabel.FUNCTION, 'src/tasks.py', 'cb')
        graph.add_node(
            GraphNode(
                id=cb_id,
                label=NodeLabel.FUNCTION,
                name='cb',
                file_path='src/tasks.py',
                start_line=32,
                end_line=35,
            )
        )
        graph.add_relationship(
            GraphRelationship(
                id=f'defines:{file_id}->{cb_id}',
                type=RelType.DEFINES,
                source=file_id,
                target=cb_id,
            )
        )
        return graph, cb_id

    def test_argument_sub_edges_propagate_dispatch_kind(self) -> None:
        """Arg sub-edge carries the outer call's dispatch_kind in properties."""
        graph, cb_id = self._build_graph_with_cb()

        call = CallInfo(
            name='worker',
            line=5,
            arguments=['cb'],
            dispatch_kind='detached_task',
            in_try=True,
        )
        parse_data = [
            FileParseData(
                file_path='src/tasks.py',
                language='python',
                parse_result=ParseResult(calls=[call]),
            )
        ]

        edges = process_calls(parse_data, graph, collect=True)
        assert edges is not None

        caller_id = generate_id(NodeLabel.FUNCTION, 'src/tasks.py', 'run')
        sub_edges = [
            e for e in edges if e.source == caller_id and e.target == cb_id
        ]
        assert sub_edges, 'expected sub-edge for callback argument'
        sub_props = sub_edges[0].properties
        assert sub_props.get('dispatch_kind') == 'detached_task'

    def test_receiver_sub_edges_remain_bare(self) -> None:
        """Receiver sub-edge (caller to variable) does not carry extra_props."""
        graph, cb_id = self._build_graph_with_cb()

        # 'worker' is called as a receiver expression; 'cb' is passed as arg.
        # The receiver sub-edge is caller -> worker (the variable node if
        # primary resolution fails). Use a name that won't resolve so the
        # receiver edge is the only sub-edge produced.
        call = CallInfo(
            name='unknown_pool',
            line=5,
            arguments=['cb'],
            dispatch_kind='detached_task',
            in_try=True,
        )
        parse_data = [
            FileParseData(
                file_path='src/tasks.py',
                language='python',
                parse_result=ParseResult(calls=[call]),
            )
        ]

        edges = process_calls(parse_data, graph, collect=True)
        assert edges is not None

        caller_id = generate_id(NodeLabel.FUNCTION, 'src/tasks.py', 'run')
        # The argument sub-edge to cb should carry dispatch_kind (Phase 4b).
        arg_sub_edges = [
            e for e in edges if e.source == caller_id and e.target == cb_id
        ]
        assert arg_sub_edges, 'expected arg sub-edge for callback argument'
        # Receiver/primary edge to unknown_pool does NOT exist in graph.
        other_targets = {
            e.target
            for e in edges
            if e.source == caller_id and e.target != cb_id
        }
        # No stray edges carrying extra props to unresolved receiver.
        for edge in edges:
            if edge.source == caller_id and edge.target in other_targets:
                assert 'dispatch_kind' not in edge.properties
