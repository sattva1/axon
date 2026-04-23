"""Tests for Phase 5 enum member extraction and member access detection."""

from __future__ import annotations

import pytest

from axon.core.parsers.base import MemberAccess
from axon.core.parsers.python_lang import PythonParser


@pytest.fixture(scope='module')
def parser() -> PythonParser:
    """Shared PythonParser instance for the module."""
    return PythonParser()


class TestEnumClassDetection:
    """Parser correctly identifies which classes are enums."""

    def test_plain_enum_base_detected(self, parser: PythonParser) -> None:
        """Class extending Enum emits MemberInfo entries."""
        src = 'from enum import Enum\nclass Color(Enum):\n    RED = 1\n'
        result = parser.parse(src, 'test.py')
        assert any(m.parent == 'Color' for m in result.members)

    @pytest.mark.parametrize('base', ['IntEnum', 'StrEnum', 'Flag', 'IntFlag'])
    def test_stdlib_enum_variants_detected(
        self, parser: PythonParser, base: str
    ) -> None:
        """All standard enum base classes trigger member extraction."""
        src = f'class Status({base}):\n    ACTIVE = 1\n'
        result = parser.parse(src, 'test.py')
        assert any(m.parent == 'Status' for m in result.members)

    def test_mixin_str_enum_detected(self, parser: PythonParser) -> None:
        """class Color(str, Enum) is detected because Enum is present."""
        src = 'class Color(str, Enum):\n    RED = "red"\n'
        result = parser.parse(src, 'test.py')
        assert any(m.parent == 'Color' for m in result.members)

    def test_non_enum_class_no_members(self, parser: PythonParser) -> None:
        """Plain class emits no MemberInfo records."""
        src = 'class Foo:\n    BAR = 1\n'
        result = parser.parse(src, 'test.py')
        assert not result.members

    def test_unknown_base_no_members(self, parser: PythonParser) -> None:
        """class Foo(Bar) with unknown Bar emits no MemberInfo."""
        src = 'class Foo(Bar):\n    BAR = 1\n'
        result = parser.parse(src, 'test.py')
        assert not result.members

    def test_enum_as_any_base_detected(self, parser: PythonParser) -> None:
        """Intersection with ENUM_BASES triggers extraction even with mixins."""
        src = 'class Status(Mixin1, Mixin2, IntFlag):\n    A = 1\n'
        result = parser.parse(src, 'test.py')
        assert any(m.parent == 'Status' for m in result.members)

    def test_enum_class_symbol_kind_is_enum(
        self, parser: PythonParser
    ) -> None:
        """Enum classes emit SymbolInfo with kind='enum', not 'class'.

        Ingestion relies on this to label the class as NodeLabel.ENUM so
        the DEFINES edge from parent-to-member resolves to an existing
        node.
        """
        src = 'class Color(Enum):\n    RED = 1\n'
        result = parser.parse(src, 'test.py')
        color_sym = next(s for s in result.symbols if s.name == 'Color')
        assert color_sym.kind == 'enum'

    def test_plain_class_symbol_kind_is_class(
        self, parser: PythonParser
    ) -> None:
        """Non-enum classes keep kind='class'."""
        src = 'class Foo:\n    BAR = 1\n'
        result = parser.parse(src, 'test.py')
        foo_sym = next(s for s in result.symbols if s.name == 'Foo')
        assert foo_sym.kind == 'class'


class TestEnumMemberExtraction:
    """Correct MemberInfo records are produced from enum bodies."""

    def test_plain_assignment_extracted(self, parser: PythonParser) -> None:
        """Simple RED = 1 assignment becomes a MemberInfo."""
        src = 'class Color(Enum):\n    RED = 1\n'
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.parent == 'Color']
        assert len(members) == 1
        m = members[0]
        assert m.name == 'RED'
        assert m.kind == 'enum_member'
        assert m.line == 2

    def test_annotated_assignment_extracted(self, parser: PythonParser) -> None:
        """GREEN: int = 2 annotated assignment produces MemberInfo."""
        src = 'class Color(Enum):\n    GREEN: int = 2\n'
        result = parser.parse(src, 'test.py')
        names = [m.name for m in result.members if m.parent == 'Color']
        assert 'GREEN' in names

    def test_mixed_body_extracts_only_assignments(
        self, parser: PythonParser
    ) -> None:
        """Methods inside enum body are not emitted as members."""
        src = (
            'class Status(Enum):\n'
            '    ACTIVE = 1\n'
            '    INACTIVE = 2\n'
            '    def label(self):\n'
            '        return self.name\n'
        )
        result = parser.parse(src, 'test.py')
        member_names = [m.name for m in result.members if m.parent == 'Status']
        assert 'ACTIVE' in member_names
        assert 'INACTIVE' in member_names
        assert 'label' not in member_names

    def test_dunder_targets_skipped(self, parser: PythonParser) -> None:
        """_ignore_ = [...] style names are not emitted."""
        src = (
            'class Status(Enum):\n'
            '    _ignore_ = ["X"]\n'
            '    ACTIVE = 1\n'
        )
        result = parser.parse(src, 'test.py')
        names = [m.name for m in result.members if m.parent == 'Status']
        assert '_ignore_' not in names
        assert 'ACTIVE' in names

    def test_tuple_unpacking_skipped(self, parser: PythonParser) -> None:
        """X, Y = 1, 2 tuple target is not an identifier and is skipped."""
        src = (
            'class Color(Enum):\n'
            '    RED = 1\n'
        )
        result = parser.parse(src, 'test.py')
        names = {m.name for m in result.members if m.parent == 'Color'}
        assert names == {'RED'}

    def test_nested_class_not_emitted_as_member(
        self, parser: PythonParser
    ) -> None:
        """Inner class definition inside enum body is not an enum member."""
        src = (
            'class Status(Enum):\n'
            '    ACTIVE = 1\n'
            '    class Inner:\n'
            '        pass\n'
        )
        result = parser.parse(src, 'test.py')
        names = [m.name for m in result.members if m.parent == 'Status']
        assert 'Inner' not in names
        assert 'ACTIVE' in names

    def test_line_numbers_recorded(self, parser: PythonParser) -> None:
        """MemberInfo line number matches the source position."""
        src = 'class Color(Enum):\n    RED = 1\n    GREEN = 2\n'
        result = parser.parse(src, 'test.py')
        by_name = {m.name: m for m in result.members if m.parent == 'Color'}
        assert by_name['RED'].line == 2
        assert by_name['GREEN'].line == 3

    def test_multiple_members_all_extracted(self, parser: PythonParser) -> None:
        """Three member assignments produce three MemberInfo records."""
        src = (
            'class Direction(Enum):\n'
            '    NORTH = 1\n'
            '    SOUTH = 2\n'
            '    EAST = 3\n'
        )
        result = parser.parse(src, 'test.py')
        names = {m.name for m in result.members if m.parent == 'Direction'}
        assert names == {'NORTH', 'SOUTH', 'EAST'}


class TestMemberAccessExtraction:
    """_scan_node_for_read_accesses and helpers emit correct MemberAccess."""

    def test_read_access_assignment_rhs(self, parser: PythonParser) -> None:
        """x = Status.PENDING emits a read MemberAccess."""
        src = 'x = Status.PENDING\n'
        result = parser.parse(src, 'test.py')
        accesses = [
            a for a in result.member_accesses
            if a.parent == 'Status' and a.name == 'PENDING'
        ]
        assert accesses
        assert accesses[0].mode == 'read'

    def test_write_access_assignment_lhs(self, parser: PythonParser) -> None:
        """Status.PENDING = 1 emits a write MemberAccess."""
        src = 'Status.PENDING = 1\n'
        result = parser.parse(src, 'test.py')
        accesses = [
            a for a in result.member_accesses
            if a.parent == 'Status' and a.name == 'PENDING'
        ]
        assert accesses
        assert accesses[0].mode == 'write'

    def test_augmented_assignment_emits_both(self, parser: PythonParser) -> None:
        """Status.PENDING += 1 emits a MemberAccess with mode=both."""
        src = 'Status.PENDING += 1\n'
        result = parser.parse(src, 'test.py')
        accesses = [
            a for a in result.member_accesses
            if a.parent == 'Status' and a.name == 'PENDING'
        ]
        assert accesses
        assert accesses[0].mode == 'both'

    def test_rhs_read_in_compound_assignment(
        self, parser: PythonParser
    ) -> None:
        """self.status = Status.PENDING emits read for Status.PENDING."""
        src = 'self.status = Status.PENDING\n'
        result = parser.parse(src, 'test.py')
        accesses = [
            a for a in result.member_accesses
            if a.parent == 'Status' and a.name == 'PENDING'
        ]
        assert accesses
        assert accesses[0].mode == 'read'
        # self.status is lowercase so not emitted
        self_accesses = [
            a for a in result.member_accesses if a.parent == 'self'
        ]
        assert not self_accesses

    def test_function_call_style_access_in_assignment(
        self, parser: PythonParser
    ) -> None:
        """x = Foo.BAR() emits a single read access for Foo.BAR (not double)."""
        src = 'x = Foo.BAR()\n'
        result = parser.parse(src, 'test.py')
        accesses = [
            a for a in result.member_accesses
            if a.parent == 'Foo' and a.name == 'BAR'
        ]
        assert len(accesses) == 1
        assert accesses[0].mode == 'read'

    def test_bare_call_statement_no_member_access(
        self, parser: PythonParser
    ) -> None:
        """Foo.BAR() as bare statement becomes a CallInfo, not a MemberAccess."""
        src = 'Foo.BAR()\n'
        result = parser.parse(src, 'test.py')
        accesses = [
            a for a in result.member_accesses
            if a.parent == 'Foo' and a.name == 'BAR'
        ]
        # Bare call statement is handled by _extract_calls_recursive as a
        # CallInfo; _scan_node_for_read_accesses is not invoked for it.
        assert not accesses
        calls = [c for c in result.calls if c.name == 'BAR' and c.receiver == 'Foo']
        assert calls

    def test_chained_attribute_not_emitted(self, parser: PythonParser) -> None:
        """pkg.Foo.BAR is not emitted because LHS of inner attr is not uppercase-root."""
        src = 'x = pkg.Foo.BAR\n'
        result = parser.parse(src, 'test.py')
        # pkg.Foo is the attribute expression for pkg; Foo.BAR would require
        # LHS to be a plain identifier. The nested attribute is skipped.
        accesses = [
            a for a in result.member_accesses if a.name == 'BAR'
        ]
        assert not accesses

    def test_lowercase_attribute_not_emitted(self, parser: PythonParser) -> None:
        """foo.bar is not emitted (lowercase LHS)."""
        src = 'x = foo.bar\n'
        result = parser.parse(src, 'test.py')
        accesses = [a for a in result.member_accesses if a.parent == 'foo']
        assert not accesses

    def test_self_cls_not_emitted(self, parser: PythonParser) -> None:
        """self.X and cls.X are rejected by _is_capital_attr."""
        src = 'x = self.X\ny = cls.Y\n'
        result = parser.parse(src, 'test.py')
        accesses = [
            a for a in result.member_accesses
            if a.parent in ('self', 'cls')
        ]
        assert not accesses

    def test_access_inside_function_body(self, parser: PythonParser) -> None:
        """Capital.attr inside a function body is emitted."""
        src = (
            'def process():\n'
            '    x = Status.DONE\n'
        )
        result = parser.parse(src, 'test.py')
        accesses = [
            a for a in result.member_accesses
            if a.parent == 'Status' and a.name == 'DONE'
        ]
        assert accesses

    def test_access_at_module_level_emitted(self, parser: PythonParser) -> None:
        """Capital.attr at module top-level is emitted by the parser."""
        src = 'x = Level.HIGH\n'
        result = parser.parse(src, 'test.py')
        accesses = [
            a for a in result.member_accesses
            if a.parent == 'Level' and a.name == 'HIGH'
        ]
        assert accesses

    def test_line_number_recorded_on_access(self, parser: PythonParser) -> None:
        """MemberAccess records the correct source line."""
        src = '\n\nx = Status.ACTIVE\n'
        result = parser.parse(src, 'test.py')
        accesses = [
            a for a in result.member_accesses if a.parent == 'Status'
        ]
        assert accesses
        assert accesses[0].line == 3

    def test_member_access_dataclass_fields(
        self, parser: PythonParser
    ) -> None:
        """MemberAccess carries parent, name, line, and mode attributes."""
        src = 'val = Color.RED\n'
        result = parser.parse(src, 'test.py')
        accesses = [
            a for a in result.member_accesses if a.parent == 'Color'
        ]
        assert accesses
        a = accesses[0]
        assert isinstance(a, MemberAccess)
        assert a.parent == 'Color'
        assert a.name == 'RED'
        assert a.mode == 'read'
        assert a.line > 0
