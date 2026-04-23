"""Tests for Phase 5 ENUM_MEMBER branches in MCP tool handlers."""

from __future__ import annotations

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


def _make_enum_member_node(
    name: str = 'PENDING',
    parent: str = 'Status',
    file_path: str = 'src/status.py',
    start_line: int = 5,
) -> GraphNode:
    """Build an ENUM_MEMBER GraphNode."""
    return GraphNode(
        id=f'enum_member:{file_path}:{parent}.{name}',
        label=NodeLabel.ENUM_MEMBER,
        name=name,
        file_path=file_path,
        start_line=start_line,
        class_name=parent,
    )


def _make_accessor_node(
    name: str,
    file_path: str = 'src/worker.py',
    start_line: int = 10,
) -> GraphNode:
    """Build a FUNCTION GraphNode that acts as an accessor."""
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
            node_id='function:src/auth.py:validate',
            score=1.0,
            node_name='validate',
            file_path='src/auth.py',
            label='function',
        ),
    ]
    storage.get_node.return_value = GraphNode(
        id='function:src/auth.py:validate',
        label=NodeLabel.FUNCTION,
        name='validate',
        file_path='src/auth.py',
        start_line=10,
        end_line=30,
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
def enum_member_storage(mock_storage: MagicMock) -> MagicMock:
    """Storage mock configured to return an ENUM_MEMBER node."""
    node = _make_enum_member_node()
    mock_storage.fts_search.return_value = [
        SearchResult(
            node_id=node.id,
            score=1.0,
            node_name=node.name,
            file_path=node.file_path,
            label='enum_member',
        ),
    ]
    mock_storage.get_node.return_value = node
    return mock_storage


class TestHandleContextEnumMember:
    """handle_context renders ENUM_MEMBER nodes correctly."""

    def test_enum_member_header_present(
        self, enum_member_storage: MagicMock
    ) -> None:
        """Result contains the 'Enum Member:' header line."""
        result = handle_context(enum_member_storage, 'Status.PENDING')
        assert 'Enum Member:' in result

    def test_enum_member_parent_line(
        self, enum_member_storage: MagicMock
    ) -> None:
        """Result contains a 'Parent:' line with the class name."""
        result = handle_context(enum_member_storage, 'Status.PENDING')
        assert 'Parent: Status' in result

    def test_enum_member_no_accessors_message(
        self, enum_member_storage: MagicMock
    ) -> None:
        """When no accessors exist, result says 'Accessors: none'."""
        enum_member_storage.get_accessors.return_value = []
        result = handle_context(enum_member_storage, 'Status.PENDING')
        assert 'Accessors: none' in result

    def test_enum_member_accessors_grouped_by_mode(
        self, enum_member_storage: MagicMock
    ) -> None:
        """Accessors are rendered in mode-grouped sections."""
        read_fn = _make_accessor_node('process_item')
        write_fn = _make_accessor_node('reset_item', start_line=20)
        both_fn = _make_accessor_node('toggle_item', start_line=30)
        enum_member_storage.get_accessors.return_value = [
            (read_fn, 'read', 1.0),
            (write_fn, 'write', 0.8),
            (both_fn, 'both', 1.0),
        ]
        result = handle_context(enum_member_storage, 'Status.PENDING')
        assert 'Accessors (read)' in result
        assert 'Accessors (write)' in result
        assert 'Accessors (both)' in result
        assert 'process_item' in result
        assert 'reset_item' in result
        assert 'toggle_item' in result

    def test_non_enum_node_existing_rendering_unchanged(
        self, mock_storage: MagicMock
    ) -> None:
        """Non-ENUM_MEMBER node falls through to standard context rendering."""
        result = handle_context(mock_storage, 'validate')
        assert 'Symbol: validate (Function)' in result
        assert 'Enum Member:' not in result


class TestHandleImpactEnumMember:
    """handle_impact on ENUM_MEMBER uses get_accessors path."""

    def test_enum_member_uses_accessors_not_traverse(
        self, enum_member_storage: MagicMock
    ) -> None:
        """get_accessors is called; traverse_with_depth is not called."""
        accessor = _make_accessor_node('handler')
        enum_member_storage.get_accessors.return_value = [(accessor, 'read', 1.0)]

        handle_impact(enum_member_storage, 'Status.PENDING')

        enum_member_storage.get_accessors.assert_called_once()
        enum_member_storage.traverse_with_depth.assert_not_called()

    def test_enum_member_impact_renders_flat_list(
        self, enum_member_storage: MagicMock
    ) -> None:
        """Impact output includes total accessor count."""
        accessor = _make_accessor_node('handler')
        enum_member_storage.get_accessors.return_value = [(accessor, 'read', 1.0)]

        result = handle_impact(enum_member_storage, 'Status.PENDING')
        assert 'handler' in result
        assert 'Total:' in result or '1 accessor' in result

    def test_enum_member_no_accessors_message(
        self, enum_member_storage: MagicMock
    ) -> None:
        """Impact output says 'No accessors found' when empty."""
        enum_member_storage.get_accessors.return_value = []

        result = handle_impact(enum_member_storage, 'Status.PENDING')
        assert 'No accessors found' in result

    def test_non_enum_impact_uses_traverse(
        self, mock_storage: MagicMock
    ) -> None:
        """Standard function uses traverse_with_depth, not get_accessors."""
        mock_storage.traverse_with_depth.return_value = []

        handle_impact(mock_storage, 'validate')

        mock_storage.traverse_with_depth.assert_called()


class TestHandleExplainEnumMember:
    """handle_explain renders ENUM_MEMBER with accessor count."""

    def test_enum_member_explain_header(
        self, enum_member_storage: MagicMock
    ) -> None:
        """Explanation header shows label 'Enum_Member'."""
        result = handle_explain(enum_member_storage, 'Status.PENDING')
        assert 'Enum_Member' in result

    def test_enum_member_explain_parent_reference(
        self, enum_member_storage: MagicMock
    ) -> None:
        """Explanation contains 'Enum member of ``Status``.' text."""
        result = handle_explain(enum_member_storage, 'Status.PENDING')
        assert 'Enum member of' in result
        assert 'Status' in result

    def test_enum_member_explain_accessor_count(
        self, enum_member_storage: MagicMock
    ) -> None:
        """Explanation reports the number of accessing symbols."""
        accessor = _make_accessor_node('handler')
        enum_member_storage.get_accessors.return_value = [
            (accessor, 'read', 1.0),
        ]
        result = handle_explain(enum_member_storage, 'Status.PENDING')
        assert '1 symbol' in result

    def test_non_enum_explain_unchanged(self, mock_storage: MagicMock) -> None:
        """Non-ENUM_MEMBER explain uses the standard rendering path."""
        mock_storage.execute_raw.return_value = []
        result = handle_explain(mock_storage, 'validate')
        assert 'Explanation: validate' in result
        assert 'Enum member of' not in result


class TestHandleFileContextEnumSection:
    """handle_file_context includes or omits the Enums section."""

    def _storage_with_enum_rows(
        self,
        mock_storage: MagicMock,
        enum_rows: list[list],
    ) -> MagicMock:
        """Wire execute_raw side effects with required minimum rows."""
        # handle_file_context calls execute_raw 7 times (sym, imports_out,
        # imports_in, coupling, dead, comm, enum). We make sym_rows non-empty
        # so the function doesn't return early with 'No data found'.
        side_effects: list[list] = [
            [['my_func', 'Function', 1, False, False, False]],  # sym_rows
            [],   # imports_out
            [],   # imports_in
            [],   # coupling
            [],   # dead
            [],   # communities
            enum_rows,  # enum_rows
        ]
        mock_storage.execute_raw.side_effect = side_effects
        return mock_storage

    def test_file_with_enums_shows_enum_section(
        self, mock_storage: MagicMock
    ) -> None:
        """'Enums:' line appears when enum_rows is non-empty."""
        self._storage_with_enum_rows(
            mock_storage,
            [['Status', 3, 5]],
        )
        result = handle_file_context(mock_storage, 'src/status.py')
        assert 'Enums:' in result
        assert 'Status' in result
        assert '3 members' in result
        assert '5 accessors' in result

    def test_file_without_enums_no_enum_section(
        self, mock_storage: MagicMock
    ) -> None:
        """'Enums:' line is absent when enum_rows is empty."""
        self._storage_with_enum_rows(mock_storage, [])
        result = handle_file_context(mock_storage, 'src/worker.py')
        assert 'Enums:' not in result
