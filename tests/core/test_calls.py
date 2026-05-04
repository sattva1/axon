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
from axon.core.ingestion.calls import (
    _CALL_BLOCKLIST,
    process_calls,
    resolve_call,
)
from axon.core.ingestion.dead_code import process_dead_code
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.ingestion.symbol_lookup import build_name_index
from axon.core.parsers.base import CallInfo, ParseResult

_CALLABLE_LABELS = (NodeLabel.FUNCTION, NodeLabel.METHOD, NodeLabel.CLASS)


def _add_file_node(graph: KnowledgeGraph, path: str) -> str:
    """Add a File node and return its ID."""
    node_id = generate_id(NodeLabel.FILE, path)
    graph.add_node(
        GraphNode(
            id=node_id,
            label=NodeLabel.FILE,
            name=path.rsplit("/", 1)[-1],
            file_path=path,
        )
    )
    return node_id


def _add_symbol_node(
    graph: KnowledgeGraph,
    label: NodeLabel,
    file_path: str,
    name: str,
    start_line: int,
    end_line: int,
    class_name: str = "",
) -> str:
    """Add a symbol node with a DEFINES relationship from the file node."""
    symbol_name = (
        f"{class_name}.{name}" if label == NodeLabel.METHOD and class_name else name
    )
    node_id = generate_id(label, file_path, symbol_name)
    graph.add_node(
        GraphNode(
            id=node_id,
            label=label,
            name=name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            class_name=class_name,
        )
    )
    file_id = generate_id(NodeLabel.FILE, file_path)
    graph.add_relationship(
        GraphRelationship(
            id=f"defines:{file_id}->{node_id}",
            type=RelType.DEFINES,
            source=file_id,
            target=node_id,
        )
    )
    return node_id


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """Build a graph matching the test fixture specification.

    File: src/auth.py
        Function: validate (lines 1-10)
        Function: hash_password (lines 12-20)

    File: src/app.py
        Function: login (lines 1-15)

    File: src/utils.py
        Function: helper (lines 1-5)
    """
    g = KnowledgeGraph()

    # Files
    _add_file_node(g, "src/auth.py")
    _add_file_node(g, "src/app.py")
    _add_file_node(g, "src/utils.py")

    # Symbols in src/auth.py
    _add_symbol_node(g, NodeLabel.FUNCTION, "src/auth.py", "validate", 1, 10)
    _add_symbol_node(g, NodeLabel.FUNCTION, "src/auth.py", "hash_password", 12, 20)

    # Symbols in src/app.py
    _add_symbol_node(g, NodeLabel.FUNCTION, "src/app.py", "login", 1, 15)

    # Symbols in src/utils.py
    _add_symbol_node(g, NodeLabel.FUNCTION, "src/utils.py", "helper", 1, 5)

    return g


@pytest.fixture()
def parse_data() -> list[FileParseData]:
    """Parse data with calls matching the fixture specification.

    src/auth.py: hash_password() at line 5 (inside validate)
    src/app.py: validate() at line 8 (inside login)
    """
    return [
        FileParseData(
            file_path="src/auth.py",
            language="python",
            parse_result=ParseResult(
                calls=[CallInfo(name="hash_password", line=5)],
            ),
        ),
        FileParseData(
            file_path="src/app.py",
            language="python",
            parse_result=ParseResult(
                calls=[CallInfo(name="validate", line=8)],
            ),
        ),
    ]


class TestBuildCallIndex:
    def test_build_call_index(self, graph: KnowledgeGraph) -> None:
        index = build_name_index(graph, _CALLABLE_LABELS)

        # All four functions should appear.
        assert "validate" in index
        assert "hash_password" in index
        assert "login" in index
        assert "helper" in index

        # Each name maps to exactly one node ID.
        assert len(index["validate"]) == 1
        assert len(index["hash_password"]) == 1

        # IDs match expected generate_id output.
        expected_validate = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "validate"
        )
        assert index["validate"] == [expected_validate]

    def test_build_call_index_includes_classes(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/models.py")
        _add_symbol_node(g, NodeLabel.CLASS, "src/models.py", "User", 1, 20)

        index = build_name_index(g, _CALLABLE_LABELS)
        assert "User" in index
        assert len(index["User"]) == 1

    def test_build_call_index_multiple_same_name(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/a.py")
        _add_file_node(g, "src/b.py")
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/a.py", "init", 1, 5)
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/b.py", "init", 1, 5)

        index = build_name_index(g, _CALLABLE_LABELS)
        assert 'init' in index
        assert len(index['init']) == 2


class TestResolveCallSameFile:
    def test_resolve_call_same_file(self, graph: KnowledgeGraph) -> None:
        index = build_name_index(graph, _CALLABLE_LABELS)
        call = CallInfo(name="hash_password", line=5)

        target_id, confidence = resolve_call(
            call, "src/auth.py", index, graph
        )

        expected_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "hash_password"
        )
        assert target_id == expected_id
        assert confidence == 1.0


class TestResolveCallGlobal:
    def test_resolve_call_global(self, graph: KnowledgeGraph) -> None:
        index = build_name_index(graph, _CALLABLE_LABELS)
        call = CallInfo(name="validate", line=8)

        target_id, confidence = resolve_call(
            call, "src/app.py", index, graph
        )

        expected_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "validate"
        )
        assert target_id == expected_id
        assert confidence == 0.5


class TestResolveCallUnresolved:
    def test_resolve_call_unresolved(self, graph: KnowledgeGraph) -> None:
        index = build_name_index(graph, _CALLABLE_LABELS)
        call = CallInfo(name="nonexistent_function", line=3)

        target_id, confidence = resolve_call(
            call, "src/auth.py", index, graph
        )

        assert target_id is None
        assert confidence == 0.0


class TestProcessCallsCreatesRelationships:
    def test_process_calls_creates_relationships(
        self,
        graph: KnowledgeGraph,
        parse_data: list[FileParseData],
    ) -> None:
        process_calls(parse_data, graph)

        calls_rels = graph.get_relationships_by_type(RelType.CALLS)
        assert len(calls_rels) == 2

        # Collect source->target pairs.
        pairs = {(r.source, r.target) for r in calls_rels}

        validate_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "validate"
        )
        hash_pw_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "hash_password"
        )
        login_id = generate_id(NodeLabel.FUNCTION, "src/app.py", "login")

        # validate -> hash_password (same-file call at line 5 inside validate)
        assert (validate_id, hash_pw_id) in pairs
        # login -> validate (cross-file call at line 8 inside login)
        assert (login_id, validate_id) in pairs


class TestProcessCallsConfidence:
    def test_process_calls_confidence(
        self,
        graph: KnowledgeGraph,
        parse_data: list[FileParseData],
    ) -> None:
        process_calls(parse_data, graph)

        calls_rels = graph.get_relationships_by_type(RelType.CALLS)

        validate_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "validate"
        )
        hash_pw_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "hash_password"
        )
        login_id = generate_id(NodeLabel.FUNCTION, "src/app.py", "login")

        confidences = {(r.source, r.target): r.properties["confidence"] for r in calls_rels}

        # Same-file call: confidence 1.0
        assert confidences[(validate_id, hash_pw_id)] == 1.0
        # Cross-file global match: confidence 0.5
        assert confidences[(login_id, validate_id)] == 0.5


class TestProcessCallsNoDuplicates:
    def test_process_calls_no_duplicates(self, graph: KnowledgeGraph) -> None:
        # Two identical calls to hash_password inside validate.
        duplicate_parse_data = [
            FileParseData(
                file_path="src/auth.py",
                language="python",
                parse_result=ParseResult(
                    calls=[
                        CallInfo(name="hash_password", line=5),
                        CallInfo(name="hash_password", line=7),
                    ],
                ),
            ),
        ]

        process_calls(duplicate_parse_data, graph)

        calls_rels = graph.get_relationships_by_type(RelType.CALLS)
        # Both calls resolve to validate -> hash_password, but only one
        # relationship should exist.
        assert len(calls_rels) == 1


class TestResolveMethodCallSelf:
    def test_resolve_method_call_self(self) -> None:
        g = KnowledgeGraph()

        _add_file_node(g, "src/service.py")
        _add_symbol_node(
            g,
            NodeLabel.CLASS,
            "src/service.py",
            "AuthService",
            1,
            30,
        )
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/service.py",
            "login",
            3,
            15,
            class_name="AuthService",
        )
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/service.py",
            "check_token",
            17,
            28,
            class_name="AuthService",
        )

        index = build_name_index(g, _CALLABLE_LABELS)
        call = CallInfo(name="check_token", line=10, receiver="self")

        target_id, confidence = resolve_call(
            call, "src/service.py", index, g
        )

        expected_id = generate_id(
            NodeLabel.METHOD, "src/service.py", "AuthService.check_token"
        )
        assert target_id == expected_id
        assert confidence == 1.0

    def test_resolve_method_call_this(self) -> None:
        g = KnowledgeGraph()

        _add_file_node(g, "src/service.ts")
        _add_symbol_node(
            g,
            NodeLabel.CLASS,
            "src/service.ts",
            "AuthService",
            1,
            30,
        )
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/service.ts",
            "checkToken",
            17,
            28,
            class_name="AuthService",
        )

        index = build_name_index(g, _CALLABLE_LABELS)
        call = CallInfo(name="checkToken", line=10, receiver="this")

        target_id, confidence = resolve_call(
            call, "src/service.ts", index, g
        )

        expected_id = generate_id(
            NodeLabel.METHOD, "src/service.ts", "AuthService.checkToken"
        )
        assert target_id == expected_id
        assert confidence == 1.0


class TestResolveCallImportResolved:
    def test_resolve_call_import_resolved(self) -> None:
        g = KnowledgeGraph()

        # Two files: app.py imports validate from auth.py.
        _add_file_node(g, "src/auth.py")
        _add_file_node(g, "src/app.py")

        _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/auth.py", "validate", 1, 10
        )
        _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/app.py", "login", 1, 15
        )

        # IMPORTS relationship: app.py -> auth.py with symbol "validate"
        app_file_id = generate_id(NodeLabel.FILE, "src/app.py")
        auth_file_id = generate_id(NodeLabel.FILE, "src/auth.py")
        g.add_relationship(
            GraphRelationship(
                id=f"imports:{app_file_id}->{auth_file_id}",
                type=RelType.IMPORTS,
                source=app_file_id,
                target=auth_file_id,
                properties={"symbols": "validate"},
            )
        )

        index = build_name_index(g, _CALLABLE_LABELS)
        call = CallInfo(name="validate", line=8)

        target_id, confidence = resolve_call(
            call, "src/app.py", index, g
        )

        expected_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "validate"
        )
        assert target_id == expected_id
        assert confidence == 1.0


class TestCallBlocklist:
    def test_blocklist_is_frozenset(self) -> None:
        assert isinstance(_CALL_BLOCKLIST, frozenset)

    def test_python_builtins_in_blocklist(self) -> None:
        for name in ("print", "len", "range", "isinstance", "super"):
            assert name in _CALL_BLOCKLIST

    def test_js_globals_in_blocklist(self) -> None:
        for name in ("console", "setTimeout", "fetch", "JSON", "Promise"):
            assert name in _CALL_BLOCKLIST

    def test_react_hooks_in_blocklist(self) -> None:
        for name in ("useState", "useEffect", "useCallback", "useMemo"):
            assert name in _CALL_BLOCKLIST

    def test_blocklisted_call_creates_no_edge(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/main.py")
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/main.py", "do_work", 1, 10)

        parse_data = [
            FileParseData(
                file_path="src/main.py",
                language="python",
                parse_result=ParseResult(
                    calls=[CallInfo(name="print", line=5)],
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        assert len(calls_rels) == 0

    def test_blocklisted_argument_creates_no_edge(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/main.py")
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/main.py", "do_work", 1, 10)

        parse_data = [
            FileParseData(
                file_path="src/main.py",
                language="python",
                parse_result=ParseResult(
                    calls=[
                        CallInfo(name="apply_func", line=5, arguments=["str"]),
                    ],
                ),
            ),
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        # apply_func is not in the graph so no edge for it; 'str' is blocklisted.
        assert len(calls_rels) == 0

    def test_non_blocklisted_call_still_resolves(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, 'src/main.py')
        _add_symbol_node(g, NodeLabel.FUNCTION, 'src/main.py', 'caller', 1, 10)
        _add_symbol_node(
            g, NodeLabel.FUNCTION, 'src/main.py', 'my_helper', 12, 20
        )

        parse_data = [
            FileParseData(
                file_path='src/main.py',
                language='python',
                parse_result=ParseResult(
                    calls=[CallInfo(name='my_helper', line=5)]
                ),
            )
        ]

        process_calls(parse_data, g)
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        assert len(calls_rels) == 1


class TestArgumentSubEdgeExtraProps:
    """Argument sub-edge extra_props propagation (Phase 4b)."""

    def test_argument_sub_edge_carries_dispatch_kind(self) -> None:
        """Arg sub-edge from a thread_executor call carries dispatch_kind."""
        g = KnowledgeGraph()
        _add_file_node(g, 'src/runner.py')
        # runner is the symbol that contains the call expression.
        _add_symbol_node(
            g, NodeLabel.FUNCTION, 'src/runner.py', 'runner', 1, 20
        )
        # callback is the argument identifier that resolves to a symbol.
        _add_symbol_node(
            g, NodeLabel.FUNCTION, 'src/runner.py', 'callback', 22, 35
        )

        parse_data = [
            FileParseData(
                file_path='src/runner.py',
                language='python',
                parse_result=ParseResult(
                    calls=[
                        CallInfo(
                            name='submit',
                            line=10,
                            arguments=['callback'],
                            dispatch_kind='thread_executor',
                        )
                    ]
                ),
            )
        ]

        edges = process_calls(parse_data, g, collect=True)
        assert edges is not None

        callback_id = generate_id(
            NodeLabel.FUNCTION, 'src/runner.py', 'callback'
        )
        arg_edges = [e for e in edges if e.target == callback_id]
        assert arg_edges, 'Expected at least one argument sub-edge to callback'
        arg_edge = arg_edges[0]
        assert arg_edge.properties.get('dispatch_kind') == 'thread_executor'

    def test_argument_sub_edge_default_direct_has_no_extras(self) -> None:
        """Direct-dispatch calls produce arg sub-edges without dispatch_kind."""
        g = KnowledgeGraph()
        _add_file_node(g, 'src/runner.py')
        _add_symbol_node(
            g, NodeLabel.FUNCTION, 'src/runner.py', 'orchestrate', 1, 20
        )
        _add_symbol_node(
            g, NodeLabel.FUNCTION, 'src/runner.py', 'handler', 22, 35
        )

        parse_data = [
            FileParseData(
                file_path='src/runner.py',
                language='python',
                parse_result=ParseResult(
                    calls=[
                        CallInfo(
                            name='apply',
                            line=10,
                            arguments=['handler'],
                            # dispatch_kind defaults to 'direct'
                        )
                    ]
                ),
            )
        ]

        edges = process_calls(parse_data, g, collect=True)
        assert edges is not None

        handler_id = generate_id(
            NodeLabel.FUNCTION, 'src/runner.py', 'handler'
        )
        arg_edges = [e for e in edges if e.target == handler_id]
        assert arg_edges, 'Expected argument sub-edge to handler'
        arg_edge = arg_edges[0]
        # Sparse encoding: direct dispatch is the default, so the key must be absent.
        assert 'dispatch_kind' not in arg_edge.properties


class TestModuleScopeCallAttributesToFileNode:
    """Step 1 - module-scope calls attribute to the FILE node."""

    def test_attributes_to_file_node(self) -> None:
        """Top-level register(handler) produces a CALLS edge from FILE to handler."""
        g = KnowledgeGraph()
        _add_file_node(g, 'src/setup.py')
        _add_symbol_node(
            g, NodeLabel.FUNCTION, 'src/setup.py', 'handler', 5, 15
        )

        # Call at line 3 is outside any symbol (module scope).
        parse_data = [
            FileParseData(
                file_path='src/setup.py',
                language='python',
                parse_result=ParseResult(
                    calls=[CallInfo(name='handler', line=3)]
                ),
            )
        ]

        process_calls(parse_data, g)

        file_id = generate_id(NodeLabel.FILE, 'src/setup.py')
        handler_id = generate_id(NodeLabel.FUNCTION, 'src/setup.py', 'handler')
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        sources = {r.source for r in calls_rels}
        targets = {r.target for r in calls_rels}
        assert file_id in sources
        assert handler_id in targets

    def test_keeps_target_alive_in_dead_code(self) -> None:
        """A function called only at module scope is NOT flagged dead."""
        g = KnowledgeGraph()
        _add_file_node(g, 'src/app.py')
        _add_symbol_node(
            g, NodeLabel.FUNCTION, 'src/app.py', 'register', 5, 15
        )

        # register() call at module scope (line 2 is before any function).
        parse_data = [
            FileParseData(
                file_path='src/app.py',
                language='python',
                parse_result=ParseResult(
                    calls=[CallInfo(name='register', line=2)]
                ),
            )
        ]

        process_calls(parse_data, g)
        process_dead_code(g)

        register_id = generate_id(NodeLabel.FUNCTION, 'src/app.py', 'register')
        node = g.get_node(register_id)
        assert node is not None
        assert node.is_dead is False


class TestReceiverTypeResolution:
    """Step 3 - type-aware receiver resolution via receiver_type."""

    def test_local_binding_receiver_resolves_to_class_method(self) -> None:
        """user = User(); user.save() resolves to User.save, not literal 'user'."""
        g = KnowledgeGraph()
        _add_file_node(g, 'src/service.py')
        _add_symbol_node(g, NodeLabel.CLASS, 'src/service.py', 'User', 1, 30)
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            'src/service.py',
            'save',
            5,
            20,
            class_name='User',
        )
        _add_symbol_node(
            g, NodeLabel.FUNCTION, 'src/service.py', 'create_user', 32, 50
        )

        parse_data = [
            FileParseData(
                file_path='src/service.py',
                language='python',
                parse_result=ParseResult(
                    calls=[
                        # Line 40 is inside create_user; receiver_type resolved
                        # by parser from the 'u = User()' binding.
                        CallInfo(
                            name='save',
                            line=40,
                            receiver='u',
                            receiver_type='User',
                        )
                    ]
                ),
            )
        ]

        edges = process_calls(parse_data, g, collect=True)
        assert edges is not None

        save_id = generate_id(NodeLabel.METHOD, 'src/service.py', 'User.save')
        targets = {e.target for e in edges}
        assert save_id in targets

    def test_with_as_binding_receiver_resolves(self) -> None:
        """with Conn() as c: c.commit() resolves to Conn.commit."""
        g = KnowledgeGraph()
        _add_file_node(g, 'src/db.py')
        _add_symbol_node(g, NodeLabel.CLASS, 'src/db.py', 'Conn', 1, 25)
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            'src/db.py',
            'commit',
            5,
            15,
            class_name='Conn',
        )
        _add_symbol_node(
            g, NodeLabel.FUNCTION, 'src/db.py', 'run_query', 30, 50
        )

        parse_data = [
            FileParseData(
                file_path='src/db.py',
                language='python',
                parse_result=ParseResult(
                    calls=[
                        CallInfo(
                            name='commit',
                            line=40,
                            receiver='c',
                            receiver_type='Conn',
                        )
                    ]
                ),
            )
        ]

        edges = process_calls(parse_data, g, collect=True)
        assert edges is not None

        commit_id = generate_id(NodeLabel.METHOD, 'src/db.py', 'Conn.commit')
        targets = {e.target for e in edges}
        assert commit_id in targets

    def test_static_style_receiver_still_works(self) -> None:
        """Foo.bar() with empty receiver_type falls back to literal class match."""
        g = KnowledgeGraph()
        _add_file_node(g, 'src/utils.py')
        _add_symbol_node(g, NodeLabel.CLASS, 'src/utils.py', 'Foo', 1, 20)
        _add_symbol_node(
            g, NodeLabel.METHOD, 'src/utils.py', 'bar', 3, 15, class_name='Foo'
        )
        _add_symbol_node(
            g, NodeLabel.FUNCTION, 'src/utils.py', 'caller', 22, 40
        )

        parse_data = [
            FileParseData(
                file_path='src/utils.py',
                language='python',
                parse_result=ParseResult(
                    calls=[
                        # Static-style: receiver is the class name; receiver_type empty.
                        CallInfo(
                            name='bar',
                            line=30,
                            receiver='Foo',
                            receiver_type='',
                        )
                    ]
                ),
            )
        ]

        edges = process_calls(parse_data, g, collect=True)
        assert edges is not None

        bar_id = generate_id(NodeLabel.METHOD, 'src/utils.py', 'Foo.bar')
        targets = {e.target for e in edges}
        assert bar_id in targets

    def test_self_receiver_unchanged(self) -> None:
        """self.save() uses _resolve_self_method (receiver_type stays empty)."""
        g = KnowledgeGraph()
        _add_file_node(g, 'src/models.py')
        _add_symbol_node(g, NodeLabel.CLASS, 'src/models.py', 'Record', 1, 40)
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            'src/models.py',
            'save',
            3,
            15,
            class_name='Record',
        )
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            'src/models.py',
            'persist',
            17,
            35,
            class_name='Record',
        )

        parse_data = [
            FileParseData(
                file_path='src/models.py',
                language='python',
                parse_result=ParseResult(
                    calls=[
                        # receiver='self', receiver_type='' -> _resolve_self_method
                        CallInfo(
                            name='save',
                            line=25,
                            receiver='self',
                            receiver_type='',
                        )
                    ]
                ),
            )
        ]

        edges = process_calls(parse_data, g, collect=True)
        assert edges is not None

        save_id = generate_id(NodeLabel.METHOD, 'src/models.py', 'Record.save')
        targets = {e.target for e in edges}
        assert save_id in targets


class TestFunctionReferences:
    """Step 4 - first-class function references produce CALLS edges."""

    def test_assignment_alias_creates_reference_edge(self) -> None:
        """handler = my_func produces a CALLS edge from caller to my_func."""
        g = KnowledgeGraph()
        _add_file_node(g, 'src/app.py')
        _add_symbol_node(g, NodeLabel.FUNCTION, 'src/app.py', 'setup', 1, 10)
        _add_symbol_node(
            g, NodeLabel.FUNCTION, 'src/app.py', 'my_func', 12, 20
        )

        parse_data = [
            FileParseData(
                file_path='src/app.py',
                language='python',
                parse_result=ParseResult(
                    calls=[CallInfo(name='my_func', line=5, is_reference=True)]
                ),
            )
        ]

        edges = process_calls(parse_data, g, collect=True)
        assert edges is not None

        my_func_id = generate_id(NodeLabel.FUNCTION, 'src/app.py', 'my_func')
        targets = {e.target for e in edges}
        assert my_func_id in targets

    def test_list_literal_creates_reference_edges(self) -> None:
        """routes = [a, b] produces CALLS edges to both a and b."""
        g = KnowledgeGraph()
        _add_file_node(g, 'src/router.py')
        _add_symbol_node(g, NodeLabel.FUNCTION, 'src/router.py', 'a', 1, 5)
        _add_symbol_node(g, NodeLabel.FUNCTION, 'src/router.py', 'b', 7, 12)

        parse_data = [
            FileParseData(
                file_path='src/router.py',
                language='python',
                parse_result=ParseResult(
                    calls=[
                        CallInfo(name='a', line=15, is_reference=True),
                        CallInfo(name='b', line=15, is_reference=True),
                    ]
                ),
            )
        ]

        edges = process_calls(parse_data, g, collect=True)
        assert edges is not None

        a_id = generate_id(NodeLabel.FUNCTION, 'src/router.py', 'a')
        b_id = generate_id(NodeLabel.FUNCTION, 'src/router.py', 'b')
        targets = {e.target for e in edges}
        assert a_id in targets
        assert b_id in targets

    def test_dict_value_creates_reference_edges(self) -> None:
        """dispatch = {'x': a} produces an edge to a."""
        g = KnowledgeGraph()
        _add_file_node(g, 'src/dispatch.py')
        _add_symbol_node(g, NodeLabel.FUNCTION, 'src/dispatch.py', 'a', 1, 5)

        parse_data = [
            FileParseData(
                file_path='src/dispatch.py',
                language='python',
                parse_result=ParseResult(
                    calls=[CallInfo(name='a', line=10, is_reference=True)]
                ),
            )
        ]

        edges = process_calls(parse_data, g, collect=True)
        assert edges is not None

        a_id = generate_id(NodeLabel.FUNCTION, 'src/dispatch.py', 'a')
        targets = {e.target for e in edges}
        assert a_id in targets

    def test_reference_via_property_set_on_edge(self) -> None:
        """Reference edge carries via='reference' in its properties."""
        g = KnowledgeGraph()
        _add_file_node(g, 'src/app.py')
        _add_symbol_node(g, NodeLabel.FUNCTION, 'src/app.py', 'setup', 1, 10)
        _add_symbol_node(
            g, NodeLabel.FUNCTION, 'src/app.py', 'my_func', 12, 20
        )

        parse_data = [
            FileParseData(
                file_path='src/app.py',
                language='python',
                parse_result=ParseResult(
                    calls=[CallInfo(name='my_func', line=5, is_reference=True)]
                ),
            )
        ]

        edges = process_calls(parse_data, g, collect=True)
        assert edges is not None

        my_func_id = generate_id(NodeLabel.FUNCTION, 'src/app.py', 'my_func')
        ref_edges = [e for e in edges if e.target == my_func_id]
        assert ref_edges
        assert ref_edges[0].properties.get('via') == 'reference'

    def test_reference_skipped_for_all_caps_constant(self) -> None:
        """config = SETTINGS does NOT produce a reference edge (ALL_CAPS filter)."""
        g = KnowledgeGraph()
        _add_file_node(g, 'src/config.py')
        _add_symbol_node(
            g, NodeLabel.FUNCTION, 'src/config.py', 'setup', 1, 10
        )
        # SETTINGS as a function would exist in the graph, but should not be
        # emitted as a reference because parser filters ALL_CAPS identifiers.
        _add_symbol_node(
            g, NodeLabel.FUNCTION, 'src/config.py', 'SETTINGS', 12, 20
        )

        # Simulate what the parser produces: ALL_CAPS RHS is NOT emitted.
        # The parse result has NO is_reference=True call for SETTINGS.
        parse_data = [
            FileParseData(
                file_path='src/config.py',
                language='python',
                parse_result=ParseResult(calls=[]),
            )
        ]

        edges = process_calls(parse_data, g, collect=True)
        assert edges is not None

        settings_id = generate_id(
            NodeLabel.FUNCTION, 'src/config.py', 'SETTINGS'
        )
        targets = {e.target for e in edges}
        assert settings_id not in targets

    def test_reference_skipped_for_blocklist_name(self) -> None:
        """x = list does NOT produce a reference edge (blocklist filter)."""
        g = KnowledgeGraph()
        _add_file_node(g, 'src/app.py')
        _add_symbol_node(g, NodeLabel.FUNCTION, 'src/app.py', 'caller', 1, 20)

        # Parser would not emit 'list' as is_reference due to blocklist.
        # Confirm that even if it did, the ingestion blocklist also stops it.
        parse_data = [
            FileParseData(
                file_path='src/app.py',
                language='python',
                parse_result=ParseResult(
                    calls=[CallInfo(name='list', line=5, is_reference=True)]
                ),
            )
        ]

        edges = process_calls(parse_data, g, collect=True)
        assert edges is not None
        # 'list' is in _CALL_BLOCKLIST; the call is skipped before resolution.
        assert all(e.properties.get('via') != 'reference' for e in edges)

    def test_fuzzy_match_does_not_emit_reference_edge(self) -> None:
        """Reference branch requires conf >= 1.0; global fuzzy (0.5) is dropped."""
        g = KnowledgeGraph()
        # my_func defined in a DIFFERENT file than the reference call.
        _add_file_node(g, 'src/handlers.py')
        _add_file_node(g, 'src/app.py')
        _add_symbol_node(
            g, NodeLabel.FUNCTION, 'src/handlers.py', 'my_func', 1, 10
        )
        _add_symbol_node(g, NodeLabel.FUNCTION, 'src/app.py', 'setup', 1, 10)

        # Reference call comes from src/app.py; my_func is in src/handlers.py.
        # No IMPORTS edge -> resolve_call returns confidence 0.5 (fuzzy global).
        parse_data = [
            FileParseData(
                file_path='src/app.py',
                language='python',
                parse_result=ParseResult(
                    calls=[CallInfo(name='my_func', line=5, is_reference=True)]
                ),
            )
        ]

        edges = process_calls(parse_data, g, collect=True)
        assert edges is not None

        my_func_id = generate_id(
            NodeLabel.FUNCTION, 'src/handlers.py', 'my_func'
        )
        ref_edges = [
            e
            for e in edges
            if e.target == my_func_id
            and e.properties.get('via') == 'reference'
        ]
        assert not ref_edges, (
            'Fuzzy-matched reference should be dropped (conf < 1.0)'
        )

    def test_call_and_reference_to_same_target_keeps_call_priority(
        self,
    ) -> None:
        """When same target is both called (conf 1.0) and referenced, call wins."""
        g = KnowledgeGraph()
        _add_file_node(g, 'src/app.py')
        _add_symbol_node(g, NodeLabel.FUNCTION, 'src/app.py', 'setup', 1, 15)
        _add_symbol_node(
            g, NodeLabel.FUNCTION, 'src/app.py', 'my_func', 17, 30
        )

        # setup calls and also "references" my_func. Both are inside setup
        # (lines 5 and 8 are within 1-15 range).
        parse_data = [
            FileParseData(
                file_path='src/app.py',
                language='python',
                parse_result=ParseResult(
                    calls=[
                        # Real call - processed first by loop order.
                        CallInfo(name='my_func', line=5),
                        # Reference to same target - processed second.
                        CallInfo(name='my_func', line=8, is_reference=True),
                    ]
                ),
            )
        ]

        edges = process_calls(parse_data, g, collect=True)
        assert edges is not None

        my_func_id = generate_id(NodeLabel.FUNCTION, 'src/app.py', 'my_func')
        target_edges = [e for e in edges if e.target == my_func_id]
        # Only one edge should exist (deduplication).
        assert len(target_edges) == 1
        # The surviving edge must be the call (no 'via' property, conf 1.0).
        surviving = target_edges[0]
        assert 'via' not in surviving.properties
        assert surviving.properties.get('confidence') == 1.0

    def test_module_scope_reference_attributes_to_file_node(self) -> None:
        """handler = my_func at module scope produces edge from FILE to my_func."""
        g = KnowledgeGraph()
        _add_file_node(g, 'src/setup.py')
        _add_symbol_node(
            g, NodeLabel.FUNCTION, 'src/setup.py', 'my_func', 5, 15
        )

        # Reference call at line 2 is module scope (before any symbol).
        parse_data = [
            FileParseData(
                file_path='src/setup.py',
                language='python',
                parse_result=ParseResult(
                    calls=[CallInfo(name='my_func', line=2, is_reference=True)]
                ),
            )
        ]

        # my_func is defined in the same file -> resolve_call returns conf 1.0
        process_calls(parse_data, g)

        file_id = generate_id(NodeLabel.FILE, 'src/setup.py')
        my_func_id = generate_id(NodeLabel.FUNCTION, 'src/setup.py', 'my_func')
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        pairs = {(r.source, r.target) for r in calls_rels}
        assert (file_id, my_func_id) in pairs
