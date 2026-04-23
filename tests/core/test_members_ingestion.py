"""Tests for Phase 5/7 member index building and member access processing."""

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
    build_imported_names,
    build_module_constant_index,
    build_parent_qualified_member_index,
    process_member_accesses,
)
from axon.core.ingestion.parser_phase import (
    _MEMBER_KIND_TO_LABEL,
    FileParseData,
)
from axon.core.parsers.base import ImportInfo, MemberAccess, ParseResult


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


# ---------------------------------------------------------------------------
# Phase 7 additions
# ---------------------------------------------------------------------------


def _make_class_attribute(
    name: str, parent: str, file_path: str = 'src/models.py', line: int = 5
) -> GraphNode:
    """Build a CLASS_ATTRIBUTE GraphNode."""
    node_id = generate_id(
        NodeLabel.CLASS_ATTRIBUTE, file_path, f'{parent}.{name}'
    )
    return GraphNode(
        id=node_id,
        label=NodeLabel.CLASS_ATTRIBUTE,
        name=name,
        file_path=file_path,
        start_line=line,
        end_line=line,
        class_name=parent,
    )


def _make_module_constant(
    name: str, file_path: str = 'src/constants.py', line: int = 3
) -> GraphNode:
    """Build a MODULE_CONSTANT GraphNode."""
    node_id = generate_id(NodeLabel.MODULE_CONSTANT, file_path, name)
    return GraphNode(
        id=node_id,
        label=NodeLabel.MODULE_CONSTANT,
        name=name,
        file_path=file_path,
        start_line=line,
        end_line=line,
        class_name='',
    )


def _make_file_node(file_path: str) -> GraphNode:
    """Build a FILE GraphNode."""
    return GraphNode(
        id=generate_id(NodeLabel.FILE, file_path, ''),
        label=NodeLabel.FILE,
        name=file_path,
        file_path=file_path,
    )


def _make_parse_data_with_imports(
    file_path: str,
    accesses: list[MemberAccess] | None = None,
    imports: list[ImportInfo] | None = None,
) -> FileParseData:
    """Build FileParseData with optional imports and member accesses."""
    result = ParseResult()
    result.member_accesses = accesses or []
    result.imports = imports or []
    return FileParseData(
        file_path=file_path, language='python', parse_result=result
    )


class TestBuildParentQualifiedMemberIndex:
    """build_parent_qualified_member_index covers ENUM_MEMBER + CLASS_ATTRIBUTE."""

    def test_empty_graph_returns_empty(self) -> None:
        """Empty graph produces an empty index."""
        idx = build_parent_qualified_member_index(KnowledgeGraph())
        assert idx == {}

    def test_enum_member_indexed(self) -> None:
        """ENUM_MEMBER nodes appear in the index."""
        graph = KnowledgeGraph()
        graph.add_node(_make_enum_member('RED', 'Color'))
        idx = build_parent_qualified_member_index(graph)
        assert 'Color' in idx
        assert 'RED' in idx['Color']

    def test_class_attribute_indexed(self) -> None:
        """CLASS_ATTRIBUTE nodes appear in the index."""
        graph = KnowledgeGraph()
        graph.add_node(_make_class_attribute('host', 'Config'))
        idx = build_parent_qualified_member_index(graph)
        assert 'Config' in idx
        assert 'host' in idx['Config']

    def test_module_constant_absent(self) -> None:
        """MODULE_CONSTANT nodes are NOT in the parent-qualified index."""
        graph = KnowledgeGraph()
        graph.add_node(_make_module_constant('MAX_RETRIES'))
        idx = build_parent_qualified_member_index(graph)
        # Module constants have parent='', so they'd land under '' key only
        # if the index included them. They must not appear here.
        assert not any('MAX_RETRIES' in names for names in idx.values())

    def test_both_kinds_coexist(self) -> None:
        """ENUM_MEMBER and CLASS_ATTRIBUTE from different parents both indexed."""
        graph = KnowledgeGraph()
        graph.add_node(_make_enum_member('ACTIVE', 'Status'))
        graph.add_node(_make_class_attribute('timeout', 'Config'))
        idx = build_parent_qualified_member_index(graph)
        assert 'ACTIVE' in idx.get('Status', {})
        assert 'timeout' in idx.get('Config', {})


class TestBuildModuleConstantIndex:
    """build_module_constant_index covers MODULE_CONSTANT only."""

    def test_empty_graph_returns_empty(self) -> None:
        """Empty graph produces empty index."""
        idx = build_module_constant_index(KnowledgeGraph())
        assert idx == {}

    def test_constant_indexed_by_file_and_name(self) -> None:
        """MODULE_CONSTANT node is keyed by (file_path, name)."""
        graph = KnowledgeGraph()
        node = _make_module_constant('MAX_SIZE', file_path='src/config.py')
        graph.add_node(node)
        idx = build_module_constant_index(graph)
        assert 'src/config.py' in idx
        assert 'MAX_SIZE' in idx['src/config.py']
        assert idx['src/config.py']['MAX_SIZE'] == node.id

    def test_multiple_constants_same_file(self) -> None:
        """Multiple constants from the same file all appear under that key."""
        graph = KnowledgeGraph()
        graph.add_node(
            _make_module_constant('A', file_path='src/cfg.py', line=1)
        )
        graph.add_node(
            _make_module_constant('B', file_path='src/cfg.py', line=2)
        )
        idx = build_module_constant_index(graph)
        assert set(idx.get('src/cfg.py', {}).keys()) == {'A', 'B'}

    def test_class_attribute_not_present(self) -> None:
        """CLASS_ATTRIBUTE nodes do not appear in the module constant index."""
        graph = KnowledgeGraph()
        graph.add_node(_make_class_attribute('x', 'Foo'))
        idx = build_module_constant_index(graph)
        assert idx == {}

    def test_enum_member_not_present(self) -> None:
        """ENUM_MEMBER nodes do not appear in the module constant index."""
        graph = KnowledgeGraph()
        graph.add_node(_make_enum_member('RED', 'Color'))
        idx = build_module_constant_index(graph)
        assert idx == {}


class TestBuildImportedNames:
    """build_imported_names resolves imports to target file paths."""

    def _file_index_for(self, *file_paths: str) -> dict[str, str]:
        """Build a minimal file_index from a list of file paths."""
        return {fp: generate_id(NodeLabel.FILE, fp, '') for fp in file_paths}

    def test_plain_import_resolved(self) -> None:
        """from mymod import MY_CONST resolves to target file path."""
        file_index = self._file_index_for('src/consumer.py', 'src/mymod.py')
        imp = ImportInfo(module='mymod', names=['MY_CONST'])
        fpd = _make_parse_data_with_imports('src/consumer.py', imports=[imp])
        result = build_imported_names([fpd], file_index, source_roots={'src'})
        assert 'src/consumer.py' in result
        assert 'MY_CONST' in result['src/consumer.py']
        assert result['src/consumer.py']['MY_CONST'] == 'src/mymod.py'

    def test_alias_import_uses_alias_key(self) -> None:
        """from mymod import MY_CONST as MC uses alias 'MC' as the key."""
        file_index = self._file_index_for('src/consumer.py', 'src/mymod.py')
        imp = ImportInfo(
            module='mymod', names=['MY_CONST'], aliases={'MC': 'MY_CONST'}
        )
        fpd = _make_parse_data_with_imports('src/consumer.py', imports=[imp])
        result = build_imported_names([fpd], file_index, source_roots={'src'})
        consumer = result.get('src/consumer.py', {})
        assert 'MC' in consumer
        assert consumer['MC'] == 'src/mymod.py'
        # Original name should NOT appear (it is aliased)
        assert 'MY_CONST' not in consumer

    def test_unresolved_import_absent(self) -> None:
        """Import to an unknown module produces no entry."""
        file_index = self._file_index_for('src/consumer.py')
        imp = ImportInfo(module='third_party_lib', names=['SOMETHING'])
        fpd = _make_parse_data_with_imports('src/consumer.py', imports=[imp])
        result = build_imported_names([fpd], file_index, source_roots={'src'})
        assert 'src/consumer.py' not in result

    def test_non_python_file_skipped(self) -> None:
        """TypeScript files are skipped."""
        file_index = self._file_index_for('src/app.ts')
        result_obj = ParseResult()
        fpd = FileParseData(
            file_path='src/app.ts',
            language='typescript',
            parse_result=result_obj,
        )
        result = build_imported_names([fpd], file_index, source_roots=set())
        assert result == {}


class TestProcessMemberAccessesPhase7:
    """process_member_accesses extended paths for CLASS_ATTRIBUTE and bare IDs."""

    def test_class_attribute_access_emits_edge(self) -> None:
        """Access to Capital.attr (CLASS_ATTRIBUTE) produces ACCESSES edge."""
        graph = KnowledgeGraph()
        attr_node = _make_class_attribute(
            'host', 'Config', file_path='src/config.py', line=3
        )
        graph.add_node(attr_node)
        caller = _make_function('connect', file_path='src/app.py')
        graph.add_node(caller)

        parent_idx = build_parent_qualified_member_index(graph)
        access = MemberAccess(
            parent='Config', name='host', line=5, mode='read'
        )
        parse_data = [_make_parse_data('src/app.py', [access])]

        emitted = process_member_accesses(
            parse_data, graph, parent_idx, {}, {}
        )
        assert emitted == 1
        rels = graph.get_relationships_by_type(RelType.ACCESSES)
        assert rels[0].target == attr_node.id

    def test_self_attr_resolved_class_access_emits_edge(self) -> None:
        """Access with parent='Foo' (self.field resolved by parser) emits edge."""
        graph = KnowledgeGraph()
        attr_node = _make_class_attribute('x', 'Foo', file_path='src/foo.py')
        graph.add_node(attr_node)
        method = GraphNode(
            id=generate_id(NodeLabel.METHOD, 'src/foo.py', 'Foo.bar'),
            label=NodeLabel.METHOD,
            name='bar',
            file_path='src/foo.py',
            start_line=5,
            end_line=15,
        )
        graph.add_node(method)

        parent_idx = build_parent_qualified_member_index(graph)
        # Parser already resolved 'self' to 'Foo' before emitting this access
        access = MemberAccess(parent='Foo', name='x', line=8, mode='write')
        parse_data = [_make_parse_data('src/foo.py', [access])]

        emitted = process_member_accesses(
            parse_data, graph, parent_idx, {}, {}
        )
        assert emitted == 1

    def test_bare_id_same_file_constant_resolved(self) -> None:
        """Bare-identifier access to a same-file constant emits ACCESSES edge."""
        graph = KnowledgeGraph()
        const_node = _make_module_constant(
            'MAX', file_path='src/utils.py', line=1
        )
        graph.add_node(const_node)
        fn = _make_function('compute', file_path='src/utils.py')
        graph.add_node(fn)

        mod_idx = build_module_constant_index(graph)
        access = MemberAccess(parent='', name='MAX', line=5, mode='read')
        parse_data = [_make_parse_data('src/utils.py', [access])]

        emitted = process_member_accesses(parse_data, graph, {}, mod_idx, {})
        assert emitted == 1
        rels = graph.get_relationships_by_type(RelType.ACCESSES)
        assert rels[0].target == const_node.id

    def test_bare_id_cross_file_constant_resolved(self) -> None:
        """Cross-file bare-ID access resolved via imported_names."""
        graph = KnowledgeGraph()
        const_node = _make_module_constant(
            'LIMIT', file_path='src/cfg.py', line=1
        )
        graph.add_node(const_node)
        fn = _make_function('run', file_path='src/worker.py')
        graph.add_node(fn)

        mod_idx = build_module_constant_index(graph)
        # Worker imported LIMIT from cfg.py
        imported = {'src/worker.py': {'LIMIT': 'src/cfg.py'}}
        access = MemberAccess(parent='', name='LIMIT', line=5, mode='read')
        parse_data = [_make_parse_data('src/worker.py', [access])]

        emitted = process_member_accesses(
            parse_data, graph, {}, mod_idx, imported
        )
        assert emitted == 1
        rels = graph.get_relationships_by_type(RelType.ACCESSES)
        assert rels[0].target == const_node.id

    def test_bare_id_no_match_dropped(self) -> None:
        """Bare-identifier access with no matching constant is silently dropped."""
        graph = KnowledgeGraph()
        fn = _make_function('run')
        graph.add_node(fn)

        access = MemberAccess(
            parent='', name='UNKNOWN_CONST', line=5, mode='read'
        )
        parse_data = [_make_parse_data('src/worker.py', [access])]

        emitted = process_member_accesses(parse_data, graph, {}, {}, {})
        assert emitted == 0

    def test_cross_file_constant_confidence_is_point_eight(self) -> None:
        """Cross-file module constant access gets confidence=0.8."""
        graph = KnowledgeGraph()
        const_node = _make_module_constant('LIMIT', file_path='src/cfg.py')
        graph.add_node(const_node)
        fn = _make_function('run', file_path='src/worker.py')
        graph.add_node(fn)

        mod_idx = build_module_constant_index(graph)
        imported = {'src/worker.py': {'LIMIT': 'src/cfg.py'}}
        access = MemberAccess(parent='', name='LIMIT', line=5, mode='read')
        parse_data = [_make_parse_data('src/worker.py', [access])]

        process_member_accesses(parse_data, graph, {}, mod_idx, imported)
        rels = graph.get_relationships_by_type(RelType.ACCESSES)
        assert rels[0].properties['confidence'] == pytest.approx(0.8)

    def test_same_file_constant_confidence_is_one(self) -> None:
        """Same-file module constant access gets confidence=1.0."""
        graph = KnowledgeGraph()
        const_node = _make_module_constant('LIMIT', file_path='src/utils.py')
        graph.add_node(const_node)
        fn = _make_function('run', file_path='src/utils.py')
        graph.add_node(fn)

        mod_idx = build_module_constant_index(graph)
        access = MemberAccess(parent='', name='LIMIT', line=5, mode='read')
        parse_data = [_make_parse_data('src/utils.py', [access])]

        process_member_accesses(parse_data, graph, {}, mod_idx, {})
        rels = graph.get_relationships_by_type(RelType.ACCESSES)
        assert rels[0].properties['confidence'] == pytest.approx(1.0)


class TestMemberKindToLabelCoverage:
    """_MEMBER_KIND_TO_LABEL covers exactly the expected set of member kinds."""

    def test_key_set_matches_expected(self) -> None:
        """Key set must be exactly {enum_member, class_attribute, module_constant}."""
        expected = {'enum_member', 'class_attribute', 'module_constant'}
        assert set(_MEMBER_KIND_TO_LABEL.keys()) == expected

    def test_values_are_correct_nodelabels(self) -> None:
        """Each key maps to the corresponding NodeLabel member."""
        assert _MEMBER_KIND_TO_LABEL['enum_member'] == NodeLabel.ENUM_MEMBER
        assert (
            _MEMBER_KIND_TO_LABEL['class_attribute']
            == NodeLabel.CLASS_ATTRIBUTE
        )
        assert (
            _MEMBER_KIND_TO_LABEL['module_constant']
            == NodeLabel.MODULE_CONSTANT
        )
