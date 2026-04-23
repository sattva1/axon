"""Tests for Phase 7 class attribute, module constant,
and self-attribution extraction in PythonParser."""

from __future__ import annotations

import pytest

from axon.core.parsers.python_lang import PythonParser


@pytest.fixture(scope='module')
def parser() -> PythonParser:
    """Shared PythonParser instance for the module."""
    return PythonParser()


class TestClassAttributeExtraction:
    """_extract_class_attributes emits MemberInfo for the right patterns."""

    def test_annotated_field_emitted(self, parser: PythonParser) -> None:
        """field: int = 5 on a plain class emits a class_attribute member."""
        src = 'class Foo:\n    field: int = 5\n'
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.parent == 'Foo']
        assert len(members) == 1
        m = members[0]
        assert m.name == 'field'
        assert m.kind == 'class_attribute'

    def test_classvar_emitted(self, parser: PythonParser) -> None:
        """field: ClassVar[int] = 5 emits a class_attribute member."""
        src = 'class Foo:\n    field: ClassVar[int] = 5\n'
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.parent == 'Foo']
        assert any(m.name == 'field' for m in members)

    def test_plain_assignment_no_decorator_not_emitted(
        self, parser: PythonParser
    ) -> None:
        """Plain x = 5 on a non-framework class is NOT emitted."""
        src = 'class Foo:\n    x = 5\n'
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.parent == 'Foo']
        assert not members

    def test_dunder_not_emitted(self, parser: PythonParser) -> None:
        """__name__-style fields are skipped."""
        src = 'class Foo:\n    __slots__: list = []\n'
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.parent == 'Foo']
        assert not members

    def test_pydantic_basemodel_annotated_field(
        self, parser: PythonParser
    ) -> None:
        """Annotated field on BaseModel subclass is emitted."""
        src = (
            'from pydantic import BaseModel, Field\n'
            'class M(BaseModel):\n'
            '    field: int = Field(default=5)\n'
        )
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.parent == 'M']
        assert any(m.name == 'field' for m in members)

    def test_pydantic_basemodel_plain_annotation_emitted(
        self, parser: PythonParser
    ) -> None:
        """Simple annotated field without Field(...) on BaseModel is emitted."""
        src = 'class M(BaseModel):\n    value: int\n'
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.parent == 'M']
        assert any(m.name == 'value' for m in members)

    def test_pydantic_rootmodel_annotated_field(
        self, parser: PythonParser
    ) -> None:
        """Annotated field on RootModel subclass is emitted."""
        src = 'class R(RootModel):\n    root: list[int]\n'
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.parent == 'R']
        assert any(m.name == 'root' for m in members)

    def test_dataclass_annotated_field_emitted(
        self, parser: PythonParser
    ) -> None:
        """@dataclass class M: field: int = 5 annotated field emitted."""
        src = (
            'from dataclasses import dataclass\n'
            '@dataclass\n'
            'class M:\n'
            '    field: int = 5\n'
        )
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.parent == 'M']
        assert any(m.name == 'field' for m in members)

    def test_dataclass_field_call_rhs_emitted(
        self, parser: PythonParser
    ) -> None:
        """@dataclass class: x = field(default=...) emitted via _is_field_call_rhs."""
        src = (
            'from dataclasses import dataclass, field\n'
            '@dataclass\n'
            'class M:\n'
            '    x = field(default=0)\n'
        )
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.parent == 'M']
        assert any(m.name == 'x' for m in members)

    def test_dataclasses_dotted_decorator_emitted(
        self, parser: PythonParser
    ) -> None:
        """@dataclasses.dataclass is recognized via exact DATACLASS_DECORATORS match."""
        src = (
            'import dataclasses\n'
            '@dataclasses.dataclass\n'
            'class M:\n'
            '    field: int = 0\n'
        )
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.parent == 'M']
        assert any(m.name == 'field' for m in members)

    def test_attrs_define_annotated_field_emitted(
        self, parser: PythonParser
    ) -> None:
        """@attrs.define class M: field: int = 5 annotated field emitted."""
        src = 'import attrs\n@attrs.define\nclass M:\n    field: int = 5\n'
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.parent == 'M']
        assert any(m.name == 'field' for m in members)

    def test_attrs_define_field_call_rhs_emitted(
        self, parser: PythonParser
    ) -> None:
        """@attrs.define class M: x = field(default=...) emitted."""
        src = '@attrs.define\nclass M:\n    x = field(default=0)\n'
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.parent == 'M']
        assert any(m.name == 'x' for m in members)

    def test_attr_s_decorator_is_dataclass(self, parser: PythonParser) -> None:
        """@attr.s decorator triggers is_dataclass=True for field extraction."""
        src = '@attr.s\nclass M:\n    field: int = 0\n'
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.parent == 'M']
        assert any(m.name == 'field' for m in members)

    def test_plain_class_no_decorator_plain_assign_not_emitted(
        self, parser: PythonParser
    ) -> None:
        """Plain class with no decorator: plain assignment not emitted."""
        src = 'class Plain:\n    x = 5\n    y = "hello"\n'
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.parent == 'Plain']
        assert not members

    def test_multiple_annotated_attrs_all_emitted(
        self, parser: PythonParser
    ) -> None:
        """Multiple annotated fields on a plain class are all emitted."""
        src = (
            'class Config:\n'
            '    host: str = "localhost"\n'
            '    port: int = 8080\n'
            '    debug: bool = False\n'
        )
        result = parser.parse(src, 'test.py')
        members = {m.name for m in result.members if m.parent == 'Config'}
        assert members == {'host', 'port', 'debug'}

    def test_member_kind_is_class_attribute(
        self, parser: PythonParser
    ) -> None:
        """Emitted MemberInfo.kind is 'class_attribute', not 'enum_member'."""
        src = 'class Foo:\n    x: int = 1\n'
        result = parser.parse(src, 'test.py')
        m = next(m for m in result.members if m.parent == 'Foo')
        assert m.kind == 'class_attribute'

    def test_line_number_recorded(self, parser: PythonParser) -> None:
        """MemberInfo line number matches source position."""
        src = 'class Foo:\n    x: int = 1\n    y: str = ""\n'
        result = parser.parse(src, 'test.py')
        by_name = {m.name: m for m in result.members if m.parent == 'Foo'}
        assert by_name['x'].line == 2
        assert by_name['y'].line == 3


class TestModuleConstantExtraction:
    """_extract_module_constants emits MemberInfo for qualifying statements."""

    def test_plain_allcaps_assignment_emitted(
        self, parser: PythonParser
    ) -> None:
        """MY_CONST = 5 is emitted as a module_constant."""
        src = 'MY_CONST = 5\n'
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.kind == 'module_constant']
        assert any(m.name == 'MY_CONST' for m in members)

    def test_annotated_int_constant_emitted(
        self, parser: PythonParser
    ) -> None:
        """MY_CONST: int = 5 annotated assignment emitted."""
        src = 'MY_CONST: int = 5\n'
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.kind == 'module_constant']
        assert any(m.name == 'MY_CONST' for m in members)

    def test_final_annotation_emitted(self, parser: PythonParser) -> None:
        """MY_CONST: Final[int] = 5 with Final annotation emitted."""
        src = 'MY_CONST: Final[int] = 5\n'
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.kind == 'module_constant']
        assert any(m.name == 'MY_CONST' for m in members)

    def test_bare_final_annotation_emitted(self, parser: PythonParser) -> None:
        """MY_CONST: Final = 5 with bare Final annotation emitted."""
        src = 'MY_CONST: Final = 5\n'
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.kind == 'module_constant']
        assert any(m.name == 'MY_CONST' for m in members)

    @pytest.mark.parametrize(
        'rhs,label',
        [
            ('[1, 2, 3]', 'list'),
            ('(1, 2, 3)', 'tuple'),
            ("{'a': 1}", 'dict'),
            ('{1, 2, 3}', 'set'),
            ('"hello"', 'string'),
            ('True', 'bool true'),
            ('False', 'bool false'),
            ('None', 'none literal'),
            ('-5', 'unary minus'),
        ],
    )
    def test_literal_rhs_emitted(
        self, parser: PythonParser, rhs: str, label: str
    ) -> None:
        """Module constant with literal RHS is emitted."""
        src = f'CONST = {rhs}\n'
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.kind == 'module_constant']
        assert any(m.name == 'CONST' for m in members), (
            f'Expected CONST to be emitted for {label} rhs={rhs!r}'
        )

    def test_call_rhs_not_emitted(self, parser: PythonParser) -> None:
        """logger = logging.getLogger(...) is NOT emitted (call RHS)."""
        src = 'import logging\nlogger = logging.getLogger(__name__)\n'
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.kind == 'module_constant']
        assert not any(m.name == 'logger' for m in members)

    def test_inside_function_not_emitted(self, parser: PythonParser) -> None:
        """Assignment inside a function body is NOT emitted as module constant."""
        src = 'def foo():\n    MY_CONST = 5\n'
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.kind == 'module_constant']
        assert not any(m.name == 'MY_CONST' for m in members)

    def test_dunder_not_emitted(self, parser: PythonParser) -> None:
        """__version__ = '1.0' is NOT emitted (dunder filter)."""
        src = "__version__ = '1.0'\n"
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.kind == 'module_constant']
        assert not any(m.name == '__version__' for m in members)

    def test_private_sentinel_emitted(self, parser: PythonParser) -> None:
        """_SENTINEL = 1 (single-underscore private) IS emitted."""
        src = '_SENTINEL = 1\n'
        result = parser.parse(src, 'test.py')
        members = [m for m in result.members if m.kind == 'module_constant']
        assert any(m.name == '_SENTINEL' for m in members)

    def test_kind_is_module_constant(self, parser: PythonParser) -> None:
        """Emitted MemberInfo.kind is 'module_constant'."""
        src = 'MAX = 100\n'
        result = parser.parse(src, 'test.py')
        m = next((m for m in result.members if m.name == 'MAX'), None)
        assert m is not None
        assert m.kind == 'module_constant'

    def test_parent_is_empty(self, parser: PythonParser) -> None:
        """Module constants have parent='' (no parent class)."""
        src = 'MAX = 100\n'
        result = parser.parse(src, 'test.py')
        m = next((m for m in result.members if m.name == 'MAX'), None)
        assert m is not None
        assert m.parent == ''

    def test_line_number_recorded(self, parser: PythonParser) -> None:
        """MemberInfo line number matches source position."""
        src = '\nMAX = 100\n'
        result = parser.parse(src, 'test.py')
        m = next(m for m in result.members if m.name == 'MAX')
        assert m.line == 2


class TestSelfAttributionExtraction:
    """Self-attribute assignments/reads emit MemberAccess with correct parent."""

    def test_write_assignment_emitted(self, parser: PythonParser) -> None:
        """self.x = 1 inside method emits MemberAccess(parent='Foo', mode='write')."""
        src = 'class Foo:\n    def bar(self):\n        self.x = 1\n'
        result = parser.parse(src, 'test.py')
        accesses = [
            a
            for a in result.member_accesses
            if a.parent == 'Foo' and a.name == 'x'
        ]
        assert accesses
        assert accesses[0].mode == 'write'

    def test_augmented_assign_mode_both(self, parser: PythonParser) -> None:
        """self.x += 1 emits MemberAccess with mode='both'."""
        src = 'class Foo:\n    def bar(self):\n        self.x += 1\n'
        result = parser.parse(src, 'test.py')
        accesses = [
            a
            for a in result.member_accesses
            if a.parent == 'Foo' and a.name == 'x'
        ]
        assert accesses
        assert accesses[0].mode == 'both'

    def test_read_in_return_emitted(self, parser: PythonParser) -> None:
        """return self.x emits MemberAccess with mode='read'."""
        src = 'class Foo:\n    def bar(self):\n        return self.x\n'
        result = parser.parse(src, 'test.py')
        accesses = [
            a
            for a in result.member_accesses
            if a.parent == 'Foo' and a.name == 'x'
        ]
        assert accesses
        assert accesses[0].mode == 'read'

    def test_classmethod_cls_write_emitted(self, parser: PythonParser) -> None:
        """cls.x = 1 in classmethod emits MemberAccess(parent='Foo')."""
        src = (
            'class Foo:\n'
            '    @classmethod\n'
            '    def bar(cls):\n'
            '        cls.x = 1\n'
        )
        result = parser.parse(src, 'test.py')
        accesses = [
            a
            for a in result.member_accesses
            if a.parent == 'Foo' and a.name == 'x'
        ]
        assert accesses
        assert accesses[0].mode == 'write'

    def test_staticmethod_no_attribution(self, parser: PythonParser) -> None:
        """@staticmethod def bar(x): x.attr = 1 must NOT emit MemberAccess(parent='Bar').

        A @staticmethod's first parameter is not an instance receiver, so no
        attribution frame should be pushed for it.
        """
        src = (
            'class Bar:\n'
            '    @staticmethod\n'
            '    def foo(x):\n'
            '        x.attr = 1\n'
        )
        result = parser.parse(src, 'test.py')
        bar_accesses = [a for a in result.member_accesses if a.parent == 'Bar']
        assert not bar_accesses

    def test_varargs_method_no_attribution_no_error(
        self, parser: PythonParser
    ) -> None:
        """def bar(*args) inside a class does not crash and emits nothing."""
        src = 'class Foo:\n    def bar(*args):\n        pass\n'
        # Should not raise; *args has no self attribution
        result = parser.parse(src, 'test.py')
        accesses = [a for a in result.member_accesses if a.parent == 'Foo']
        assert not accesses

    def test_chained_attr_no_false_emission(
        self, parser: PythonParser
    ) -> None:
        """self.pool.shutdown() must NOT emit MemberAccess(name='shutdown')."""
        src = 'class Foo:\n    def stop(self):\n        self.pool.shutdown()\n'
        result = parser.parse(src, 'test.py')
        # The only acceptable accesses here: self.pool (if emitted) at most.
        # 'shutdown' must NOT be attributed to Foo.
        shutdown_accesses = [
            a
            for a in result.member_accesses
            if a.name == 'shutdown' and a.parent == 'Foo'
        ]
        assert not shutdown_accesses

    def test_nested_inner_class_attribution(
        self, parser: PythonParser
    ) -> None:
        """Method inside inner class resolves to inner class, not outer."""
        src = (
            'class Outer:\n'
            '    class Inner:\n'
            '        def method(self):\n'
            '            self.field = 1\n'
        )
        result = parser.parse(src, 'test.py')
        inner_accesses = [
            a
            for a in result.member_accesses
            if a.parent == 'Inner' and a.name == 'field'
        ]
        outer_accesses = [
            a
            for a in result.member_accesses
            if a.parent == 'Outer' and a.name == 'field'
        ]
        assert inner_accesses
        assert not outer_accesses

    def test_parent_is_class_name(self, parser: PythonParser) -> None:
        """MemberAccess.parent is the class name, not 'self'."""
        src = (
            'class MyClass:\n'
            '    def do_thing(self):\n'
            '        self.value = 42\n'
        )
        result = parser.parse(src, 'test.py')
        accesses = [a for a in result.member_accesses if a.name == 'value']
        assert accesses
        assert accesses[0].parent == 'MyClass'
        assert accesses[0].parent != 'self'


class TestAllCapsReadAccesses:
    """_scan_node_for_read_accesses emits ALL_CAPS bare identifier reads."""

    def test_allcaps_return_emitted(self, parser: PythonParser) -> None:
        """return MY_CONSTANT * 2 emits a MemberAccess for MY_CONSTANT."""
        src = 'def f():\n    return MY_CONSTANT * 2\n'
        result = parser.parse(src, 'test.py')
        accesses = [
            a
            for a in result.member_accesses
            if a.name == 'MY_CONSTANT' and a.parent == ''
        ]
        assert accesses
        assert accesses[0].mode == 'read'

    def test_lowercase_not_emitted(self, parser: PythonParser) -> None:
        """return logger does NOT emit a MemberAccess (not ALL_CAPS)."""
        src = 'def f():\n    return logger\n'
        result = parser.parse(src, 'test.py')
        accesses = [
            a
            for a in result.member_accesses
            if a.name == 'logger' and a.parent == ''
        ]
        assert not accesses

    def test_underscore_loop_var_not_emitted(
        self, parser: PythonParser
    ) -> None:
        """for _ in range(3) does NOT emit (regex requires leading letter)."""
        src = 'def f():\n    for _ in range(3):\n        pass\n'
        result = parser.parse(src, 'test.py')
        accesses = [
            a
            for a in result.member_accesses
            if a.name == '_' and a.parent == ''
        ]
        assert not accesses

    def test_allcaps_assigned_from_read_emitted(
        self, parser: PythonParser
    ) -> None:
        """x = MAX_SIZE emits MemberAccess for MAX_SIZE."""
        src = 'def f():\n    x = MAX_SIZE\n'
        result = parser.parse(src, 'test.py')
        accesses = [
            a
            for a in result.member_accesses
            if a.name == 'MAX_SIZE' and a.parent == ''
        ]
        assert accesses

    def test_mixed_case_not_emitted(self, parser: PythonParser) -> None:
        """MixedCase is not ALL_CAPS - not emitted as a module constant read."""
        src = 'def f():\n    return SomeClass\n'
        result = parser.parse(src, 'test.py')
        accesses = [
            a
            for a in result.member_accesses
            if a.name == 'SomeClass' and a.parent == ''
        ]
        assert not accesses
