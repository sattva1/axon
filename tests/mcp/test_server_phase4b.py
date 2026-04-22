"""Phase 4b MCP server surface tests.

Covers:
- axon_concurrent_with in the TOOLS list.
- _dispatch_tool routing for axon_concurrent_with.
- axon_impact schema includes propagate_through parameter.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import axon.mcp.server as server_module
from axon.mcp.server import TOOLS, _dispatch_tool


class TestListToolsPhase4b:
    """TOOLS list includes the new Phase 4b tools."""

    def test_list_tools_includes_axon_concurrent_with(self) -> None:
        """axon_concurrent_with is present in the static TOOLS list."""
        names = [t.name for t in TOOLS]
        assert 'axon_concurrent_with' in names

    def test_axon_impact_schema_includes_propagate_through(self) -> None:
        """axon_impact tool schema exposes the propagate_through parameter."""
        impact_tool = next(t for t in TOOLS if t.name == 'axon_impact')
        props = impact_tool.inputSchema.get('properties', {})
        assert 'propagate_through' in props

    def test_axon_concurrent_with_requires_symbol(self) -> None:
        """axon_concurrent_with requires the symbol parameter."""
        concurrent_tool = next(
            t for t in TOOLS if t.name == 'axon_concurrent_with'
        )
        assert 'symbol' in concurrent_tool.inputSchema.get('required', [])


class TestDispatchToolPhase4b:
    """_dispatch_tool routes Phase 4b tool names to the correct handlers."""

    def test_dispatch_axon_concurrent_with_routes_to_handler(self) -> None:
        """axon_concurrent_with is dispatched to handle_concurrent_with."""
        mock_storage = MagicMock()
        captured: list[tuple] = []

        with patch.object(
            server_module,
            'handle_concurrent_with',
            side_effect=lambda storage, symbol, depth=3: (
                captured.append((symbol, depth)) or 'ok'
            ),
        ):
            result = _dispatch_tool(
                'axon_concurrent_with',
                {'symbol': 'runner', 'depth': 2},
                mock_storage,
            )

        assert result == 'ok'
        assert captured == [('runner', 2)]

    def test_dispatch_axon_impact_passes_propagate_through(self) -> None:
        """axon_impact dispatch passes propagate_through to handle_impact."""
        mock_storage = MagicMock()
        captured: list[list | None] = []

        with patch.object(
            server_module,
            'handle_impact',
            side_effect=lambda storage, symbol, depth=3, propagate_through=None: (
                captured.append(propagate_through) or 'ok'
            ),
        ):
            _dispatch_tool(
                'axon_impact',
                {'symbol': 'validate', 'propagate_through': ['direct']},
                mock_storage,
            )

        assert captured == [['direct']]

    def test_dispatch_axon_impact_propagate_through_defaults_none(
        self,
    ) -> None:
        """axon_impact dispatch passes None when propagate_through is absent."""
        mock_storage = MagicMock()
        captured: list[list | None] = []

        with patch.object(
            server_module,
            'handle_impact',
            side_effect=lambda storage, symbol, depth=3, propagate_through=None: (
                captured.append(propagate_through) or 'ok'
            ),
        ):
            _dispatch_tool('axon_impact', {'symbol': 'validate'}, mock_storage)

        assert captured == [None]
