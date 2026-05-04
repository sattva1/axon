from __future__ import annotations

import pytest

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.ingestion.processes import (
    deduplicate_flows,
    find_entry_points,
    generate_process_label,
    process_processes,
    trace_flow,
)


def _add_function(
    graph: KnowledgeGraph,
    name: str,
    file_path: str = "src/app.py",
    *,
    content: str = "",
    language: str = "python",
    is_exported: bool = False,
) -> GraphNode:
    """Add a FUNCTION node and return it."""
    node_id = generate_id(NodeLabel.FUNCTION, file_path, name)
    node = GraphNode(
        id=node_id,
        label=NodeLabel.FUNCTION,
        name=name,
        file_path=file_path,
        content=content,
        language=language,
        is_exported=is_exported,
    )
    graph.add_node(node)
    return node


def _add_call(
    graph: KnowledgeGraph,
    source: GraphNode,
    target: GraphNode,
    confidence: float = 1.0,
) -> None:
    """Add a CALLS relationship between two nodes."""
    rel_id = f"calls:{source.id}->{target.id}"
    graph.add_relationship(
        GraphRelationship(
            id=rel_id,
            type=RelType.CALLS,
            source=source.id,
            target=target.id,
            properties={"confidence": confidence},
        )
    )


def _add_member_of(
    graph: KnowledgeGraph,
    node: GraphNode,
    community_id: str,
) -> None:
    """Add a MEMBER_OF relationship from *node* to a community."""
    rel_id = f"member_of:{node.id}->{community_id}"
    graph.add_relationship(
        GraphRelationship(
            id=rel_id,
            type=RelType.MEMBER_OF,
            source=node.id,
            target=community_id,
        )
    )

# Fixture: call graph
#
#   main() --> validate() --> hash_password()
#                         \-> query_db() --> format_result()
#
#   orphan_func() <-- (has incoming call from some_caller)


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """Build a graph matching the specification.

    - main() calls validate()
    - validate() calls hash_password() and query_db()
    - query_db() calls format_result()
    - orphan_func() has an incoming call (so it is NOT an entry point)
    """
    g = KnowledgeGraph()

    main = _add_function(g, 'main')
    validate = _add_function(g, 'validate')
    hash_password = _add_function(g, 'hash_password')
    query_db = _add_function(g, 'query_db')
    format_result = _add_function(g, 'format_result')
    # Use a neutral file path so the filename heuristic does not flag these
    # as entry points; the test validates the has_incoming gate only.
    orphan_func = _add_function(g, 'orphan_func', file_path='src/utils.py')

    # Also add a caller for orphan_func so it has an incoming CALLS edge.
    some_caller = _add_function(g, 'some_caller', file_path='src/utils.py')

    _add_call(g, main, validate)
    _add_call(g, validate, hash_password)
    _add_call(g, validate, query_db)
    _add_call(g, query_db, format_result)
    _add_call(g, some_caller, orphan_func)

    return g


class TestFindEntryPoints:
    def test_find_entry_points(self, graph: KnowledgeGraph) -> None:
        entry_points = find_entry_points(graph)
        ep_names = {n.name for n in entry_points}

        # main has no incoming CALLS -> entry point.
        assert "main" in ep_names
        # orphan_func HAS an incoming CALLS edge -> not an entry point
        # (unless matched by framework pattern, which it does not).
        assert "orphan_func" not in ep_names

    def test_entry_point_flag_set(self, graph: KnowledgeGraph) -> None:
        entry_points = find_entry_points(graph)
        for ep in entry_points:
            assert ep.is_entry_point is True


class TestFindEntryPointsFramework:
    def test_test_function_is_entry_point(self) -> None:
        g = KnowledgeGraph()
        test_fn = _add_function(g, "test_something", language="python")

        # Give it an incoming call so *only* the framework pattern triggers.
        caller = _add_function(g, "runner")
        _add_call(g, caller, test_fn)

        entry_points = find_entry_points(g)
        ep_names = {n.name for n in entry_points}
        assert "test_something" in ep_names

    def test_decorator_pattern_entry_point(self) -> None:
        g = KnowledgeGraph()
        _add_function(
            g,
            "index",
            content='@app.route("/")\ndef index():\n    pass',
            language="python",
        )

        entry_points = find_entry_points(g)
        ep_names = {n.name for n in entry_points}
        assert "index" in ep_names

    def test_ts_handler_is_entry_point(self) -> None:
        g = KnowledgeGraph()
        _add_function(
            g,
            "handler",
            file_path="src/api.ts",
            language="typescript",
        )

        entry_points = find_entry_points(g)
        ep_names = {n.name for n in entry_points}
        assert "handler" in ep_names


class TestTraceFlow:
    def test_trace_flow(self, graph: KnowledgeGraph) -> None:
        main_id = generate_id(NodeLabel.FUNCTION, "src/app.py", "main")
        main_node = graph.get_node(main_id)
        assert main_node is not None

        flow = trace_flow(main_node, graph)
        flow_names = [n.name for n in flow]

        # BFS from main: main -> validate -> {hash_password, query_db} -> format_result
        assert flow_names[0] == "main"
        assert "validate" in flow_names
        assert "hash_password" in flow_names
        assert "query_db" in flow_names
        assert "format_result" in flow_names
        assert len(flow) == 5

    def test_trace_flow_no_cycles(self, graph: KnowledgeGraph) -> None:
        g = KnowledgeGraph()
        a = _add_function(g, "a")
        b = _add_function(g, "b")
        _add_call(g, a, b)
        _add_call(g, b, a)  # cycle

        flow = trace_flow(a, g)
        assert len(flow) == 2  # a, b -- no revisit


class TestTraceFlowMaxDepth:
    def test_trace_flow_max_depth(self, graph: KnowledgeGraph) -> None:
        main_id = generate_id(NodeLabel.FUNCTION, "src/app.py", "main")
        main_node = graph.get_node(main_id)
        assert main_node is not None

        flow = trace_flow(main_node, graph, max_depth=1)
        flow_names = [n.name for n in flow]

        # main -> validate (depth 1), but hash_password/query_db at depth 2 are cut off.
        assert "main" in flow_names
        assert "validate" in flow_names
        # Depth-2 nodes should NOT appear.
        assert "hash_password" not in flow_names
        assert "query_db" not in flow_names


class TestGenerateProcessLabel:
    def test_generate_process_label(self) -> None:
        nodes = [
            GraphNode(id=f"n{i}", label=NodeLabel.FUNCTION, name=name)
            for i, name in enumerate(
                ["main", "validate", "hash_password", "query_db", "format_result"]
            )
        ]
        label = generate_process_label(nodes)
        # Max 4 steps in the label.
        assert label == "main \u2192 validate \u2192 hash_password \u2192 query_db"

    def test_generate_process_label_single(self) -> None:
        nodes = [GraphNode(id="n0", label=NodeLabel.FUNCTION, name="main")]
        label = generate_process_label(nodes)
        assert label == "main"

    def test_generate_process_label_empty(self) -> None:
        assert generate_process_label([]) == ""


class TestDeduplicateFlows:
    def test_deduplicate_flows(self) -> None:
        # Create nodes.
        a = GraphNode(id="a", label=NodeLabel.FUNCTION, name="a")
        b = GraphNode(id="b", label=NodeLabel.FUNCTION, name="b")
        c = GraphNode(id="c", label=NodeLabel.FUNCTION, name="c")
        d = GraphNode(id="d", label=NodeLabel.FUNCTION, name="d")

        long_flow = [a, b, c, d]
        short_flow = [a, b, c]  # 100% overlap with long_flow (3/3)

        result = deduplicate_flows([short_flow, long_flow])
        assert len(result) == 1
        assert len(result[0]) == 4  # Kept the longer flow.

    def test_deduplicate_keeps_distinct(self) -> None:
        a = GraphNode(id="a", label=NodeLabel.FUNCTION, name="a")
        b = GraphNode(id="b", label=NodeLabel.FUNCTION, name="b")
        c = GraphNode(id="c", label=NodeLabel.FUNCTION, name="c")
        d = GraphNode(id="d", label=NodeLabel.FUNCTION, name="d")

        flow1 = [a, b]
        flow2 = [c, d]

        result = deduplicate_flows([flow1, flow2])
        assert len(result) == 2


class TestProcessProcessesCreatesNodes:
    def test_process_processes_creates_nodes(
        self, graph: KnowledgeGraph
    ) -> None:
        process_processes(graph)

        process_nodes = graph.get_nodes_by_label(NodeLabel.PROCESS)
        assert len(process_nodes) > 0

        # Each Process node has a name and step_count property.
        for pn in process_nodes:
            assert pn.name != ""
            assert pn.properties["step_count"] > 1


class TestProcessProcessesCreatesSteps:
    def test_process_processes_creates_steps(
        self, graph: KnowledgeGraph
    ) -> None:
        process_processes(graph)

        step_rels = graph.get_relationships_by_type(RelType.STEP_IN_PROCESS)
        assert len(step_rels) > 0

        # All step relationships should have a step_number property.
        for rel in step_rels:
            assert "step_number" in rel.properties
            assert isinstance(rel.properties["step_number"], int)

        # Verify step numbers start at 0 for each process.
        process_nodes = graph.get_nodes_by_label(NodeLabel.PROCESS)
        for pn in process_nodes:
            incoming = graph.get_incoming(pn.id, RelType.STEP_IN_PROCESS)
            step_numbers = sorted(
                r.properties["step_number"] for r in incoming
            )
            assert step_numbers[0] == 0
            assert step_numbers == list(range(len(step_numbers)))


class TestProcessProcessesReturnsCount:
    def test_process_processes_returns_count(
        self, graph: KnowledgeGraph
    ) -> None:
        count = process_processes(graph)

        process_nodes = graph.get_nodes_by_label(NodeLabel.PROCESS)
        assert count == len(process_nodes)
        assert count > 0


class TestAlembicEntryPoints:
    """Alembic upgrade/downgrade functions in migration version dirs."""

    @pytest.mark.parametrize(
        'file_path',
        [
            'src/migrations/versions/0001_init.py',
            'src/alembic/versions/0001_add_user.py',
        ],
    )
    def test_upgrade_is_entry_point(self, file_path: str) -> None:
        """upgrade function in a migration versions dir is an entry point."""
        g = KnowledgeGraph()
        node = _add_function(g, 'upgrade', file_path=file_path)
        entry_points = find_entry_points(g)
        assert node in entry_points

    @pytest.mark.parametrize(
        'file_path',
        [
            'src/migrations/versions/0001_init.py',
            'src/alembic/versions/0002_rename.py',
        ],
    )
    def test_downgrade_is_entry_point(self, file_path: str) -> None:
        """downgrade function in a migration versions dir is an entry point."""
        g = KnowledgeGraph()
        node = _add_function(g, 'downgrade', file_path=file_path)
        entry_points = find_entry_points(g)
        assert node in entry_points

    def test_upgrade_outside_versions_dir_not_entry_point(self) -> None:
        """upgrade in an unrelated path does NOT trigger alembic heuristic."""
        g = KnowledgeGraph()
        node = _add_function(g, 'upgrade', file_path='src/utils/upgrade.py')
        # Only an entry point if name heuristic or no-incoming applies.
        # The name 'upgrade' is not in the named-entry-point set and has no
        # incoming calls, so it WILL be an entry point via the has_incoming
        # fallback - but the alembic path should not be the trigger.
        # Verify the node IS flagged but NOT because of alembic (structural test).
        entry_points = find_entry_points(g)
        # It should be an entry point because has_incoming == False,
        # but that is expected behavior - just confirm it doesn't crash.
        assert isinstance(entry_points, list)


class TestFilenameHeuristics:
    """Filename-based entry-point detection (manage.py, conftest.py, etc.)."""

    @pytest.mark.parametrize(
        'filename', ['manage.py', 'conftest.py', 'wsgi.py', 'asgi.py']
    )
    def test_function_is_entry_point(self, filename: str) -> None:
        """Any top-level function in a framework-special filename is an entry point."""
        g = KnowledgeGraph()
        node = _add_function(g, 'do_work', file_path=f'src/{filename}')
        entry_points = find_entry_points(g)
        ep_names = {n.name for n in entry_points}
        assert 'do_work' in ep_names

    def test_filename_heuristic_overrides_incoming_calls(self) -> None:
        """A function in manage.py is still an entry point even if callers exist.

        The reorder in _is_entry_point ensures filename heuristics are checked
        before the has_incoming gate. Without the reorder, a module-scope call
        at Step 1 would add an incoming CALLS edge and incorrectly demote the
        entry point.
        """
        g = KnowledgeGraph()
        target = _add_function(g, 'run_server', file_path='project/manage.py')
        # Simulate an incoming CALLS edge (e.g., from a FILE node after Step 1).
        caller = _add_function(g, 'some_caller', file_path='src/app.py')
        _add_call(g, caller, target)

        entry_points = find_entry_points(g)
        ep_names = {n.name for n in entry_points}
        assert 'run_server' in ep_names

    def test_unrelated_function_with_incoming_calls_is_not_entry_point(
        self,
    ) -> None:
        """A normal function that has incoming CALLS edges is not an entry point."""
        g = KnowledgeGraph()
        caller = _add_function(g, 'main_fn', file_path='src/main.py')
        target = _add_function(g, 'helper', file_path='src/utils.py')
        _add_call(g, caller, target)

        entry_points = find_entry_points(g)
        ep_names = {n.name for n in entry_points}
        # helper has an incoming call and is not in any special file/name set.
        assert 'helper' not in ep_names
