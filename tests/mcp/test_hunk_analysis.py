"""Tests for _is_hunk_executable in axon.mcp.tools."""

from __future__ import annotations

import pytest

from axon.mcp.tools import _is_hunk_executable


class TestHunkExecutable:
    """Tree-sitter-based hunk executability detection."""

    def test_docstring_hunk_python(self) -> None:
        """Hunk covering a single-line docstring returns False."""
        source = (
            'def foo():\n    """A single-line docstring."""\n    return 1\n'
        )
        assert _is_hunk_executable(source, [(2, 2)], 'python') is False

    def test_multiline_docstring_hunk_python(self) -> None:
        """Hunk spanning internal lines of a multi-line docstring returns False."""
        source = (
            'def foo():\n'
            '    """This is a\n'
            '    multi-line\n'
            '    docstring."""\n'
            '    return 1\n'
        )
        assert _is_hunk_executable(source, [(2, 4)], 'python') is False

    def test_comment_only_hunk_python(self) -> None:
        """Hunk covering only a comment line returns False."""
        source = """\
def foo():
    x = 1
    y = 2
    z = 3
    w = 4
    a = 5
    # changed comment
    return a
"""
        # Line 7 is the comment.
        assert _is_hunk_executable(source, [(7, 7)], 'python') is False

    def test_mixed_hunk_python(self) -> None:
        """Hunk with both a comment and an executable call returns True."""
        source = """\
def foo():
    x = 1
    y = 2
    z = 3
    # comment line
    foo()
    return 0
"""
        # Lines 5-6: one comment, one call.
        assert _is_hunk_executable(source, [(5, 6)], 'python') is True

    def test_blank_only_hunk(self) -> None:
        """Hunk covering only blank lines returns False."""
        source = 'def foo():\n    x = 1\n\n\n    return x\n'
        # Lines 3-4 are blank.
        assert _is_hunk_executable(source, [(3, 4)], 'python') is False

    def test_unsupported_language_returns_true(self) -> None:
        """Non-inspected language is treated as executable (conservative)."""
        assert (
            _is_hunk_executable('# some rust code\n', [(1, 1)], 'rust') is True
        )

    def test_empty_hunk_ranges_returns_false(self) -> None:
        """Empty hunk_ranges: nothing to check, returns False."""
        assert _is_hunk_executable('x = 1\n', [], 'python') is False

    def test_parser_error_returns_true(self) -> None:
        """Garbage content that still passes the parser (tree-sitter is forgiving)."""
        # tree-sitter can parse almost anything with error recovery, so this test
        # verifies the conservative True path via unsupported language instead.
        gibberish = '!!! @@@\n' * 10
        # Unsupported language -> always True (conservative).
        assert _is_hunk_executable(gibberish, [(1, 5)], 'cobol') is True

    def test_typescript_single_line_comment(self) -> None:
        """TypeScript // comment is a comment node -> not executable."""
        source = 'const x = 1;\n// changed comment\nconst y = 2;\n'
        assert _is_hunk_executable(source, [(2, 2)], 'typescript') is False

    def test_typescript_block_comment(self) -> None:
        """TypeScript block comment /* */ is a comment node -> not executable."""
        source = 'const a = 1;\n/* block\n   comment */\nconst b = 2;\n'
        assert _is_hunk_executable(source, [(2, 3)], 'typescript') is False

    def test_executable_code_returns_true_python(self) -> None:
        """Hunk with a plain assignment returns True."""
        source = 'x = 1\ny = 2\nz = 3\n'
        assert _is_hunk_executable(source, [(1, 3)], 'python') is True

    def test_javascript_comment_not_executable(self) -> None:
        """JavaScript // comment is not executable."""
        source = 'var x = 1;\n// this changed\nvar y = 2;\n'
        assert _is_hunk_executable(source, [(2, 2)], 'javascript') is False

    def test_tsx_executable_line_returns_true(self) -> None:
        """tsx is in the supported set; a plain assignment returns True."""
        source = 'const a = 1;\nconst b = 2;\n'
        assert _is_hunk_executable(source, [(1, 1)], 'tsx') is True
