"""Tests for Phase 7 CLASS_ATTRIBUTE and MODULE_CONSTANT branches in MCP tools."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from axon.core.graph.model import GraphNode, NodeLabel
from axon.core.storage.base import SearchResult
from axon.mcp.tools import (
    handle_context,
    handle_explain,
    handle_file_context,
    handle_impact,
)


def _make_class_attr_node(
    name: str = 'timeout',
    parent: str = 'Config',
    file_path: str = 'src/config.py',
    start_line: int = 5,
) -> GraphNode:
    """Build a CLASS_ATTRIBUTE GraphNode."""
    return GraphNode(
        id=f'class_attribute:{file_path}:{parent}.{name}',
        label=NodeLabel.CLASS_ATTRIBUTE,
        name=name,
        file_path=file_path,
        start_line=start_line,
        class_name=parent,
    )


def _make_module_const_node(
    name: str = 'MAX_RETRIES',
    file_path: str = 'src/constants.py',
    start_line: int = 3,
) -> GraphNode:
    """Build a MODULE_CONSTANT GraphNode."""
    return GraphNode(
        id=f'module_constant:{file_path}:{name}',
        label=NodeLabel.MODULE_CONSTANT,
        name=name,
        file_path=file_path,
        start_line=start_line,
        class_name='',
    )


def _make_accessor_node(
    name: str, file_path: str = 'src/worker.py', start_line: int = 10
) -> GraphNode:
    """Build a FUNCTION GraphNode acting as an accessor."""
    return GraphNode(
        id=f'function:{file_path}:{name}',
        label=NodeLabel.FUNCTION,
        name=name,
        file_path=file_path,
        start_line=start_line,
    )


@pytest.fixture()
def mock_storage() -> MagicMock:
    """Mock storage returning a generic FUNCTION node by default."""
    storage = MagicMock()
    storage.fts_search.return_value = [
        SearchResult(
            node_id='function:src/app.py:run',
            score=1.0,
            node_name='run',
            file_path='src/app.py',
            label='function',
        )
    ]
    storage.get_node.return_value = GraphNode(
        id='function:src/app.py:run',
        label=NodeLabel.FUNCTION,
        name='run',
        file_path='src/app.py',
        start_line=1,
        end_line=10,
    )
    storage.get_callers.return_value = []
    storage.get_callees.return_value = []
    storage.get_type_refs.return_value = []
    storage.get_callers_with_confidence.return_value = []
    storage.get_callees_with_confidence.return_value = []
    storage.get_process_memberships.return_value = {}
    storage.traverse_with_depth.return_value = []
    storage.execute_raw.return_value = []
    storage.get_accessors.return_value = []
    return storage


@pytest.fixture()
def class_attr_storage(mock_storage: MagicMock) -> MagicMock:
    """Storage mock configured to return a CLASS_ATTRIBUTE node."""
    node = _make_class_attr_node()
    mock_storage.fts_search.return_value = [
        SearchResult(
            node_id=node.id,
            score=1.0,
            node_name=node.name,
            file_path=node.file_path,
            label='class_attribute',
        )
    ]
    mock_storage.get_node.return_value = node
    return mock_storage


@pytest.fixture()
def module_const_storage(mock_storage: MagicMock) -> MagicMock:
    """Storage mock configured to return a MODULE_CONSTANT node."""
    node = _make_module_const_node()
    mock_storage.fts_search.return_value = [
        SearchResult(
            node_id=node.id,
            score=1.0,
            node_name=node.name,
            file_path=node.file_path,
            label='module_constant',
        )
    ]
    mock_storage.get_node.return_value = node
    return mock_storage


class TestHandleContextClassAttribute:
    """handle_context renders CLASS_ATTRIBUTE nodes correctly."""

    def test_class_attribute_header_present(
        self, class_attr_storage: MagicMock, make_ctx: Any
    ) -> None:
        """Result contains the 'Class Attribute:' header line."""
        result = handle_context(make_ctx(class_attr_storage), 'Config.timeout')
        assert 'Class Attribute:' in result

    def test_class_attribute_parent_line(
        self, class_attr_storage: MagicMock, make_ctx: Any
    ) -> None:
        """Result contains a 'Parent:' line with the class name."""
        result = handle_context(make_ctx(class_attr_storage), 'Config.timeout')
        assert 'Parent: Config' in result

    def test_class_attribute_no_accessors(
        self, class_attr_storage: MagicMock, make_ctx: Any
    ) -> None:
        """When no accessors exist, result says 'Accessors: none'."""
        class_attr_storage.get_accessors.return_value = []
        result = handle_context(make_ctx(class_attr_storage), 'Config.timeout')
        assert 'Accessors: none' in result

    def test_class_attribute_accessors_grouped_by_mode(
        self, class_attr_storage: MagicMock, make_ctx: Any
    ) -> None:
        """Accessors are rendered in mode-grouped sections."""
        read_fn = _make_accessor_node('read_config')
        write_fn = _make_accessor_node('set_timeout', start_line=20)
        class_attr_storage.get_accessors.return_value = [
            (read_fn, 'read', 1.0),
            (write_fn, 'write', 1.0),
        ]
        result = handle_context(make_ctx(class_attr_storage), 'Config.timeout')
        assert 'Accessors (read)' in result
        assert 'Accessors (write)' in result
        assert 'read_config' in result
        assert 'set_timeout' in result


class TestHandleContextModuleConstant:
    """handle_context renders MODULE_CONSTANT nodes correctly."""

    def test_module_constant_header_present(
        self, module_const_storage: MagicMock, make_ctx: Any
    ) -> None:
        """Result contains the 'Module Constant:' header line."""
        result = handle_context(make_ctx(module_const_storage), 'MAX_RETRIES')
        assert 'Module Constant:' in result

    def test_module_constant_no_parent_class_in_header(
        self, module_const_storage: MagicMock, make_ctx: Any
    ) -> None:
        """Module constant header does not include a class name."""
        result = handle_context(make_ctx(module_const_storage), 'MAX_RETRIES')
        # The header should NOT contain a 'Parent:' line with a class name
        assert 'Module:' in result

    def test_module_constant_module_line(
        self, module_const_storage: MagicMock, make_ctx: Any
    ) -> None:
        """Result contains 'Module: src/constants.py' instead of 'Parent:'."""
        result = handle_context(make_ctx(module_const_storage), 'MAX_RETRIES')
        assert 'Module: src/constants.py' in result

    def test_module_constant_no_accessors(
        self, module_const_storage: MagicMock, make_ctx: Any
    ) -> None:
        """Empty accessor list yields 'Accessors: none'."""
        module_const_storage.get_accessors.return_value = []
        result = handle_context(make_ctx(module_const_storage), 'MAX_RETRIES')
        assert 'Accessors: none' in result

    def test_module_constant_accessor_listed(
        self, module_const_storage: MagicMock, make_ctx: Any
    ) -> None:
        """Accessor function names appear in the output."""
        fn = _make_accessor_node('retry_loop')
        module_const_storage.get_accessors.return_value = [(fn, 'read', 1.0)]
        result = handle_context(make_ctx(module_const_storage), 'MAX_RETRIES')
        assert 'retry_loop' in result


class TestHandleImpactClassAttribute:
    """handle_impact on CLASS_ATTRIBUTE uses get_accessors path."""

    def test_class_attr_uses_accessors_not_traverse(
        self, class_attr_storage: MagicMock, make_ctx: Any
    ) -> None:
        """get_accessors is called; traverse_with_depth is not called."""
        fn = _make_accessor_node('handler')
        class_attr_storage.get_accessors.return_value = [(fn, 'read', 1.0)]

        handle_impact(make_ctx(class_attr_storage), 'Config.timeout')

        class_attr_storage.get_accessors.assert_called_once()
        class_attr_storage.traverse_with_depth.assert_not_called()

    def test_class_attr_impact_renders_flat_list(
        self, class_attr_storage: MagicMock, make_ctx: Any
    ) -> None:
        """Impact output includes accessor count and kind label."""
        fn = _make_accessor_node('handler')
        class_attr_storage.get_accessors.return_value = [(fn, 'read', 1.0)]

        result = handle_impact(make_ctx(class_attr_storage), 'Config.timeout')
        assert 'handler' in result
        assert 'Class_Attribute' in result

    def test_class_attr_no_accessors_message(
        self, class_attr_storage: MagicMock, make_ctx: Any
    ) -> None:
        """Impact output says 'No accessors found' when empty."""
        class_attr_storage.get_accessors.return_value = []
        result = handle_impact(make_ctx(class_attr_storage), 'Config.timeout')
        assert 'No accessors found' in result


class TestHandleImpactModuleConstant:
    """handle_impact on MODULE_CONSTANT uses get_accessors path."""

    def test_module_const_uses_accessors_not_traverse(
        self, module_const_storage: MagicMock, make_ctx: Any
    ) -> None:
        """get_accessors is called; traverse_with_depth is not called."""
        fn = _make_accessor_node('compute')
        module_const_storage.get_accessors.return_value = [(fn, 'read', 1.0)]

        handle_impact(make_ctx(module_const_storage), 'MAX_RETRIES')

        module_const_storage.get_accessors.assert_called_once()
        module_const_storage.traverse_with_depth.assert_not_called()

    def test_module_const_impact_renders_flat_list(
        self, module_const_storage: MagicMock, make_ctx: Any
    ) -> None:
        """Impact output includes accessor count and Module_Constant label."""
        fn = _make_accessor_node('compute')
        module_const_storage.get_accessors.return_value = [(fn, 'read', 1.0)]

        result = handle_impact(make_ctx(module_const_storage), 'MAX_RETRIES')
        assert 'compute' in result
        assert 'Module_Constant' in result

    def test_module_const_no_accessors_message(
        self, module_const_storage: MagicMock, make_ctx: Any
    ) -> None:
        """Impact output says 'No accessors found' when empty."""
        module_const_storage.get_accessors.return_value = []
        result = handle_impact(make_ctx(module_const_storage), 'MAX_RETRIES')
        assert 'No accessors found' in result


class TestHandleExplainClassAttribute:
    """handle_explain renders CLASS_ATTRIBUTE with accessor count."""

    def test_class_attr_explain_kind_label(
        self, class_attr_storage: MagicMock, make_ctx: Any
    ) -> None:
        """Explanation header shows 'Class_Attribute' kind label."""
        result = handle_explain(make_ctx(class_attr_storage), 'Config.timeout')
        assert 'Class_Attribute' in result

    def test_class_attr_explain_parent_reference(
        self, class_attr_storage: MagicMock, make_ctx: Any
    ) -> None:
        """Explanation contains 'Class attribute of ``Config``.' text."""
        result = handle_explain(make_ctx(class_attr_storage), 'Config.timeout')
        assert 'Class attribute of' in result
        assert 'Config' in result

    def test_class_attr_explain_accessor_count(
        self, class_attr_storage: MagicMock, make_ctx: Any
    ) -> None:
        """Explanation reports number of accessing symbols."""
        fn = _make_accessor_node('reader')
        class_attr_storage.get_accessors.return_value = [(fn, 'read', 1.0)]
        result = handle_explain(make_ctx(class_attr_storage), 'Config.timeout')
        assert '1 symbol' in result


class TestHandleExplainModuleConstant:
    """handle_explain renders MODULE_CONSTANT with accessor count."""

    def test_module_const_explain_kind_label(
        self, module_const_storage: MagicMock, make_ctx: Any
    ) -> None:
        """Explanation header shows 'Module_Constant' kind label."""
        result = handle_explain(make_ctx(module_const_storage), 'MAX_RETRIES')
        assert 'Module_Constant' in result

    def test_module_const_explain_module_reference(
        self, module_const_storage: MagicMock, make_ctx: Any
    ) -> None:
        """Explanation contains 'Module constant in' text with the file path."""
        result = handle_explain(make_ctx(module_const_storage), 'MAX_RETRIES')
        assert 'Module constant in' in result
        assert 'src/constants.py' in result

    def test_module_const_explain_zero_accessors(
        self, module_const_storage: MagicMock, make_ctx: Any
    ) -> None:
        """Explanation reports 0 symbols when no accessors exist."""
        module_const_storage.get_accessors.return_value = []
        result = handle_explain(make_ctx(module_const_storage), 'MAX_RETRIES')
        assert '0 symbol' in result


class TestHandleFileContextMemberSections:
    """handle_file_context shows class attribute and module constant sections."""

    def _build_execute_raw_side_effects(
        self,
        enum_rows: list[list],
        cls_attr_rows: list[list],
        mod_const_rows: list[list],
    ) -> list[list]:
        """Build 9-call execute_raw side effects for handle_file_context."""
        return [
            [['my_func', 'Function', 1, False, False, False]],  # sym_rows
            [],  # imports_out
            [],  # imports_in
            [],  # coupling
            [],  # dead
            [],  # communities
            enum_rows,  # enum_rows
            cls_attr_rows,  # cls_attr_rows
            mod_const_rows,  # mod_const_rows
        ]

    def test_class_attributes_section_shown(
        self, mock_storage: MagicMock, make_ctx: Any
    ) -> None:
        """'Class attributes:' line appears when cls_attr_rows is non-empty."""
        mock_storage.execute_raw.side_effect = (
            self._build_execute_raw_side_effects(
                enum_rows=[],
                cls_attr_rows=[['Config', 2, 3]],
                mod_const_rows=[],
            )
        )
        result = handle_file_context(make_ctx(mock_storage), 'src/config.py')
        assert 'Class attributes:' in result
        assert 'Config' in result
        assert '2 attrs' in result
        assert '3 accessors' in result

    def test_class_attributes_section_absent_when_empty(
        self, mock_storage: MagicMock, make_ctx: Any
    ) -> None:
        """'Class attributes:' line is absent when no attrs exist."""
        mock_storage.execute_raw.side_effect = (
            self._build_execute_raw_side_effects(
                enum_rows=[],
                cls_attr_rows=[['Config', 0, 0]],
                mod_const_rows=[],
            )
        )
        result = handle_file_context(make_ctx(mock_storage), 'src/config.py')
        assert 'Class attributes:' not in result

    def test_module_constants_section_shown(
        self, mock_storage: MagicMock, make_ctx: Any
    ) -> None:
        """'Module constants:' line appears when mod_const_rows has a non-zero count."""
        mock_storage.execute_raw.side_effect = (
            self._build_execute_raw_side_effects(
                enum_rows=[], cls_attr_rows=[], mod_const_rows=[[4, 7]]
            )
        )
        result = handle_file_context(
            make_ctx(mock_storage), 'src/constants.py'
        )
        assert 'Module constants:' in result
        assert '4' in result
        assert '7' in result

    def test_module_constants_section_absent_when_zero(
        self, mock_storage: MagicMock, make_ctx: Any
    ) -> None:
        """'Module constants:' line is absent when count is 0."""
        mock_storage.execute_raw.side_effect = (
            self._build_execute_raw_side_effects(
                enum_rows=[], cls_attr_rows=[], mod_const_rows=[[0, 0]]
            )
        )
        result = handle_file_context(make_ctx(mock_storage), 'src/app.py')
        assert 'Module constants:' not in result

    def test_all_three_member_sections_present(
        self, mock_storage: MagicMock, make_ctx: Any
    ) -> None:
        """File with enums, class attrs, and module constants shows all three sections."""
        mock_storage.execute_raw.side_effect = (
            self._build_execute_raw_side_effects(
                enum_rows=[['Status', 2, 4]],
                cls_attr_rows=[['Config', 3, 1]],
                mod_const_rows=[[5, 8]],
            )
        )
        result = handle_file_context(make_ctx(mock_storage), 'src/mixed.py')
        assert 'Enums:' in result
        assert 'Class attributes:' in result
        assert 'Module constants:' in result

    def test_no_member_sections_when_all_empty(
        self, mock_storage: MagicMock, make_ctx: Any
    ) -> None:
        """When all member rows are empty, no member sections appear."""
        mock_storage.execute_raw.side_effect = (
            self._build_execute_raw_side_effects(
                enum_rows=[], cls_attr_rows=[], mod_const_rows=[]
            )
        )
        result = handle_file_context(make_ctx(mock_storage), 'src/plain.py')
        assert 'Enums:' not in result
        assert 'Class attributes:' not in result
        assert 'Module constants:' not in result
