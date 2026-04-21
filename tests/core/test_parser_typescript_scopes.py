"""Phase 4a: TypeScript scope-aware parser tests.

Covers the _TsScopeStack integration: in_try and awaited fields only.
"""

from __future__ import annotations

import pytest

from axon.core.parsers.base import CallInfo
from axon.core.parsers.typescript import TypeScriptParser


@pytest.fixture
def ts_parser() -> TypeScriptParser:
    """Shared TypeScriptParser instance."""
    return TypeScriptParser(dialect='typescript')


def _calls_by_name(
    parser: TypeScriptParser, code: str, name: str
) -> list[CallInfo]:
    """Parse *code* and return all CallInfos whose name equals *name*."""
    result = parser.parse(code, 'test.ts')
    return [c for c in result.calls if c.name == name]


class TestTsScopeSubset:
    """TypeScript scope tracking: try_depth and awaited_depth only."""

    def test_in_try_body(self, ts_parser: TypeScriptParser) -> None:
        """Call inside try { } body has in_try=True."""
        code = 'try {\n    foo();\n} catch (e) {}\n'
        calls = _calls_by_name(ts_parser, code, 'foo')
        assert calls, 'expected foo call'
        assert calls[0].in_try is True

    def test_await_sets_awaited(self, ts_parser: TypeScriptParser) -> None:
        """Call inside await expression has awaited=True."""
        code = 'async function run() {\n    await foo();\n}\n'
        calls = _calls_by_name(ts_parser, code, 'foo')
        assert calls, 'expected foo call'
        assert calls[0].awaited is True

    def test_module_level_all_false(self, ts_parser: TypeScriptParser) -> None:
        """Call at module level has in_try=False and awaited=False."""
        code = 'foo();\n'
        calls = _calls_by_name(ts_parser, code, 'foo')
        assert calls, 'expected foo call'
        assert calls[0].in_try is False
        assert calls[0].awaited is False

    def test_nested_try_inner_body_in_try(
        self, ts_parser: TypeScriptParser
    ) -> None:
        """Call in inner try body inside outer try body: in_try=True."""
        code = (
            'try {\n'
            '    try {\n'
            '        call();\n'
            '    } catch (e) {}\n'
            '} catch (e) {}\n'
        )
        calls = _calls_by_name(ts_parser, code, 'call')
        assert calls, 'expected call expression'
        assert calls[0].in_try is True
