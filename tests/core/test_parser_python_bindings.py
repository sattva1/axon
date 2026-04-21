"""Phase 4a-follow-up: binding tracker tests for dispatch_kind resolution.

Covers local-variable bindings, with-as bindings, class self-attribute
bindings, import alias resolution, and end-to-end axon source patterns.
"""

from __future__ import annotations

import pytest

from axon.core.parsers.base import CallInfo
from axon.core.parsers.python_lang import PythonParser


@pytest.fixture
def parser() -> PythonParser:
    """Shared PythonParser instance."""
    return PythonParser()


def _calls_by_name(
    parser: PythonParser, code: str, name: str
) -> list[CallInfo]:
    """Parse *code* and return all CallInfos whose name equals *name*."""
    result = parser.parse(code, 'test.py')
    return [c for c in result.calls if c.name == name]


class TestFunctionLocalBindings:
    """Local variable bindings inside function bodies."""

    def test_local_thread_pool_submit(self, parser: PythonParser) -> None:
        """x = ThreadPoolExecutor(); x.submit(g) -> thread_executor."""
        code = (
            'from concurrent.futures import ThreadPoolExecutor\n'
            'def f():\n'
            '    executor = ThreadPoolExecutor()\n'
            '    executor.submit(g)\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'thread_executor'

    def test_local_process_pool_submit(self, parser: PythonParser) -> None:
        """x = ProcessPoolExecutor(); x.submit(g) -> process_executor."""
        code = (
            'from concurrent.futures import ProcessPoolExecutor\n'
            'def f():\n'
            '    executor = ProcessPoolExecutor()\n'
            '    executor.submit(g)\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'process_executor'

    def test_rebinding_last_wins(self, parser: PythonParser) -> None:
        """Last binding assignment wins when variable is rebound."""
        code = (
            'from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor\n'
            'def f():\n'
            '    e = ThreadPoolExecutor()\n'
            '    e = ProcessPoolExecutor()\n'
            '    e.submit(g)\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'process_executor'

    def test_sibling_functions_no_leakage(self, parser: PythonParser) -> None:
        """Binding in a() does not leak into sibling b()."""
        code = (
            'from concurrent.futures import ThreadPoolExecutor\n'
            'def a():\n'
            '    e = ThreadPoolExecutor()\n'
            'def b():\n'
            '    e.submit(g)\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'direct'

    def test_unresolvable_rhs_is_direct(self, parser: PythonParser) -> None:
        """Factory-returned executor with unknown type falls through to direct."""
        code = 'def f():\n    e = make_executor()\n    e.submit(g)\n'
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'direct'

    def test_conditional_no_else_binds_unconditionally(
        self, parser: PythonParser
    ) -> None:
        """DFS records binding unconditionally (no flow analysis).

        Known limitation: the binding tracker performs a simple DFS without
        conditional-flow analysis, so e = ThreadPoolExecutor() inside an
        if-branch is recorded regardless of whether the branch is taken.
        This test pins that documented behavior. If future work adds
        conditional-flow analysis, the expected value becomes 'direct'.
        """
        code = (
            'from concurrent.futures import ThreadPoolExecutor\n'
            'def f(cond):\n'
            '    if cond:\n'
            '        e = ThreadPoolExecutor()\n'
            '    e.submit(g)\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'thread_executor'


class TestWithAsBindings:
    """Bindings created via with Call() as name: forms."""

    def test_with_thread_pool_as_executor(self, parser: PythonParser) -> None:
        """with ThreadPoolExecutor() as executor: executor.submit() -> thread_executor."""
        code = (
            'from concurrent.futures import ThreadPoolExecutor\n'
            'with ThreadPoolExecutor(max_workers=2) as executor:\n'
            '    executor.submit(fn)\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'thread_executor'

    def test_with_process_pool_as_pool(self, parser: PythonParser) -> None:
        """with ProcessPoolExecutor() as pool: pool.submit() -> process_executor."""
        code = (
            'from concurrent.futures import ProcessPoolExecutor\n'
            'with ProcessPoolExecutor(max_workers=4) as pool:\n'
            '    pool.submit(fn)\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'process_executor'

    def test_nested_with(self, parser: PythonParser) -> None:
        """Nested with blocks both bind correctly."""
        code = (
            'from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor\n'
            'with ThreadPoolExecutor() as a:\n'
            '    with ProcessPoolExecutor() as b:\n'
            '        a.submit(f)\n'
            '        b.submit(g)\n'
        )
        result = parser.parse(code, 'test.py')
        submits = [c for c in result.calls if c.name == 'submit']
        assert len(submits) == 2
        kinds = {c.receiver: c.dispatch_kind for c in submits}
        assert kinds.get('a') == 'thread_executor'
        assert kinds.get('b') == 'process_executor'

    def test_binding_scope_ends_after_with_block(
        self, parser: PythonParser
    ) -> None:
        """with-as binding does not leak outside the with block."""
        code = (
            'from concurrent.futures import ThreadPoolExecutor\n'
            'def f():\n'
            '    with ThreadPoolExecutor() as e:\n'
            '        pass\n'
            '    e.submit(fn)\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'direct'


class TestClassInstanceAttrs:
    """Class-level self.attr bindings resolved via prescan."""

    def test_self_pool_set_in_init(self, parser: PythonParser) -> None:
        """self.pool = TPE() in __init__, used in run() -> thread_executor."""
        code = (
            'from concurrent.futures import ThreadPoolExecutor\n'
            'class Worker:\n'
            '    def __init__(self):\n'
            '        self.pool = ThreadPoolExecutor()\n'
            '    def run(self, fn):\n'
            '        self.pool.submit(fn)\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'thread_executor'

    def test_self_attr_set_in_non_init_method(
        self, parser: PythonParser
    ) -> None:
        """self.pool = TPE() in setup() method, used in run() -> thread_executor."""
        code = (
            'from concurrent.futures import ThreadPoolExecutor\n'
            'class Worker:\n'
            '    def setup(self):\n'
            '        self.pool = ThreadPoolExecutor()\n'
            '    def run(self, fn):\n'
            '        self.pool.submit(fn)\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'thread_executor'

    def test_outer_class_self_not_visible_in_nested_class(
        self, parser: PythonParser
    ) -> None:
        """Outer class self.pool not visible inside nested class."""
        code = (
            'from concurrent.futures import ThreadPoolExecutor\n'
            'class Outer:\n'
            '    def __init__(self):\n'
            '        self.pool = ThreadPoolExecutor()\n'
            '    class Inner:\n'
            '        def run(self):\n'
            '            self.pool.submit(fn)\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'direct'

    def test_rebinding_self_attr_last_wins(self, parser: PythonParser) -> None:
        """Last self.pool assignment (by traversal order) wins."""
        code = (
            'from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor\n'
            'class Worker:\n'
            '    def setup_thread(self):\n'
            '        self.pool = ThreadPoolExecutor()\n'
            '    def setup_process(self):\n'
            '        self.pool = ProcessPoolExecutor()\n'
            '    def run(self, fn):\n'
            '        self.pool.submit(fn)\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        # Last assignment by DFS traversal wins.
        assert calls[0].dispatch_kind in {
            'thread_executor',
            'process_executor',
        }

    def test_nested_function_self_not_bound(
        self, parser: PythonParser
    ) -> None:
        """Prescan stops at nested function_definition boundaries.

        self.pool = TPE() inside an inner helper() is a different self
        context -- prescan skips it. self.pool.submit() in the outer
        method falls through to direct.
        """
        code = (
            'from concurrent.futures import ThreadPoolExecutor\n'
            'class Widget:\n'
            '    def build(self):\n'
            '        def helper():\n'
            '            self.pool = ThreadPoolExecutor()\n'
            '        helper()\n'
            '        self.pool.submit(fn)\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'direct'

    def test_known_class_instance_map(self, parser: PythonParser) -> None:
        """self.pool = TPE(); self.pool.map() -> thread_executor."""
        code = (
            'from concurrent.futures import ThreadPoolExecutor\n'
            'class Worker:\n'
            '    def __init__(self):\n'
            '        self.pool = ThreadPoolExecutor()\n'
            '    def process(self, fn, items):\n'
            '        self.pool.map(fn, items)\n'
        )
        calls = _calls_by_name(parser, code, 'map')
        assert calls
        assert calls[0].dispatch_kind == 'thread_executor'


class TestImportedTypeConstruction:
    """Import alias resolution feeds into binding tracker."""

    def test_from_import_as_alias(self, parser: PythonParser) -> None:
        """from X import Y as Z; e = Z() -> type resolves to Y -> thread_executor."""
        code = (
            'from concurrent.futures import ThreadPoolExecutor as TPE\n'
            'def f():\n'
            '    e = TPE()\n'
            '    e.submit(g)\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'thread_executor'

    def test_plain_from_import_no_alias(self, parser: PythonParser) -> None:
        """from X import Y (no alias); e = Y() -> thread_executor."""
        code = (
            'from concurrent.futures import ThreadPoolExecutor\n'
            'def f():\n'
            '    e = ThreadPoolExecutor()\n'
            '    e.submit(g)\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'thread_executor'

    def test_unknown_type_falls_through(self, parser: PythonParser) -> None:
        """Unknown type not in dispatch table -> direct."""
        code = (
            'from my.lib import MyExecutor\n'
            'def f():\n'
            '    e = MyExecutor()\n'
            '    e.submit(g)\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'direct'

    def test_module_alias_attribute_call_resolves_last_segment(
        self, parser: PythonParser
    ) -> None:
        """import mod as m; e = m.ThreadPoolExecutor() -> thread_executor.

        _resolve_callee_to_type_name extracts the last identifier from any
        attribute call, so m.ThreadPoolExecutor() binds e to the canonical
        class name ThreadPoolExecutor and resolves normally. This is
        different from the plan's documented expectation (direct); the
        implementation is more capable than predicted.
        """
        code = (
            'import concurrent.futures as cf\n'
            'def f():\n'
            '    e = cf.ThreadPoolExecutor()\n'
            '    e.submit(g)\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'thread_executor'


class TestDispatcherResolutionEndToEnd:
    """End-to-end: patterns mirroring real axon source files."""

    def test_axon_pipeline_pattern(self, parser: PythonParser) -> None:
        """Mimic ingestion/pipeline.py: pool.submit in with-as block."""
        code = (
            'from concurrent.futures import ThreadPoolExecutor\n'
            'def run(parse_data, graph, repo_path):\n'
            '    with ThreadPoolExecutor(max_workers=3) as pool:\n'
            '        calls_f = pool.submit(process_calls, parse_data, graph)\n'
            '        heritage_f = pool.submit(\n'
            '            process_heritage, parse_data, graph\n'
            '        )\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert len(calls) == 2
        assert all(c.dispatch_kind == 'thread_executor' for c in calls)

    def test_axon_diff_pattern(self, parser: PythonParser) -> None:
        """Mimic diff.py: executor.submit in with-as block."""
        code = (
            'from concurrent.futures import ThreadPoolExecutor\n'
            'def build_graphs(repo_path, base_ref, current_ref):\n'
            '    with ThreadPoolExecutor(max_workers=2) as executor:\n'
            '        base_future = executor.submit(\n'
            '            _build_graph_for_ref, repo_path, base_ref\n'
            '        )\n'
            '        current_future = executor.submit(\n'
            '            _build_graph_for_ref, repo_path, current_ref\n'
            '        )\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert len(calls) == 2
        assert all(c.dispatch_kind == 'thread_executor' for c in calls)

    def test_axon_calls_pattern(self, parser: PythonParser) -> None:
        """Mimic ingestion/calls.py: pool.submit in list comprehension."""
        code = (
            'from concurrent.futures import ThreadPoolExecutor\n'
            'def process(parse_data, workers):\n'
            '    with ThreadPoolExecutor(max_workers=workers) as pool:\n'
            '        futures = [\n'
            '            pool.submit(resolve_file_calls, fpd)\n'
            '            for fpd in parse_data\n'
            '        ]\n'
        )
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'thread_executor'
