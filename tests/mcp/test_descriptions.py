"""Tests for TOOLS list: annotations, descriptions, and schema contracts.

Phase 4 requirements: every tool must carry readOnlyHint=True, every
description must contain a "When to use" or "instead of Grep" phrase,
multi-repo tools must expose a 'repo' input property, axon_list_repos must
not, and total description length must stay within budget.
"""

from __future__ import annotations

from axon.mcp.server import (
    TOOLS,
    _DIFF_KEYED_TOOLS,
    _PATH_KEYED_TOOLS,
    _SYMBOL_KEYED_TOOLS,
)

# Tools that accept an explicit repo selector in their input schema.
_MULTI_REPO_TOOL_NAMES: frozenset[str] = (
    _PATH_KEYED_TOOLS
    | _DIFF_KEYED_TOOLS
    | _SYMBOL_KEYED_TOOLS
    | frozenset(
        {
            'axon_query',
            'axon_cypher',
            'axon_communities',
            'axon_dead_code',
            'axon_cycles',
        }
    )
)

_TOOLS_BY_NAME = {t.name: t for t in TOOLS}


class TestAnnotations:
    """readOnlyHint annotation is present on every tool."""

    def test_read_only_hint_true(self) -> None:
        """Every tool carries annotations.readOnlyHint == True."""
        for tool in TOOLS:
            assert tool.annotations is not None, (
                f'{tool.name}: annotations is None'
            )
            assert tool.annotations.readOnlyHint is True, (
                f'{tool.name}: readOnlyHint is not True'
            )


class TestDescriptions:
    """Each tool description satisfies the Phase 4 content requirements."""

    def test_each_description_has_usage_guidance(self) -> None:
        """Every description contains 'When to use' or 'instead of Grep'."""
        for tool in TOOLS:
            desc_lower = tool.description.lower()
            has_when = 'when to use' in desc_lower
            has_grep = 'instead of grep' in desc_lower
            assert has_when or has_grep, (
                f'{tool.name}: description missing usage guidance'
            )

    def test_descriptions_within_token_budget(self) -> None:
        """Total description character count stays below 25 000."""
        total = sum(len(t.description) for t in TOOLS)
        assert total < 25_000, (
            f'Total description length {total} exceeds 25 000 character budget'
        )


class TestRepoParam:
    """Schema 'repo' property is present exactly where expected."""

    def test_repo_param_present_on_multi_repo_tools(self) -> None:
        """All multi-repo tools expose a 'repo' property in inputSchema."""
        for name in _MULTI_REPO_TOOL_NAMES:
            tool = _TOOLS_BY_NAME[name]
            props = tool.inputSchema.get('properties') or {}
            assert 'repo' in props, (
                f'{name}: missing repo property in inputSchema'
            )

    def test_repo_param_absent_on_axon_list_repos(self) -> None:
        """axon_list_repos takes no arguments including 'repo'."""
        tool = _TOOLS_BY_NAME['axon_list_repos']
        props = tool.inputSchema.get('properties') or {}
        assert 'repo' not in props
