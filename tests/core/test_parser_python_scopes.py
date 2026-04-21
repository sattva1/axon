"""Phase 4a: scope-aware parser walk and CallInfo enrichment tests.

Covers _ScopeStack integration, dispatch-kind classification, return-consumption
classification, exception-type pseudo-CallInfos, and context-manager tracking.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from axon.core.parsers.base import CallInfo
from axon.core.parsers.python_lang import PythonParser


def _make_callinfo(**kwargs: object) -> CallInfo:
    """Return a CallInfo with all Phase-4a fields at their defaults.

    Keyword args override individual fields, making equality assertions concise
    without repeating every default value.
    """
    defaults: dict[str, object] = dict(
        name='',
        line=0,
        dispatch_kind='direct',
        in_try=False,
        in_except=False,
        in_finally=False,
        in_loop=False,
        awaited=False,
        context_managers=(),
        return_consumption='stored',
    )
    defaults.update(kwargs)
    return CallInfo(**defaults)  # type: ignore[arg-type]


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


class TestScopeBooleans:
    """ScopeStack booleans are correctly set on emitted CallInfos."""

    def test_module_level_all_false(self, parser: PythonParser) -> None:
        """Call at module level has all scope booleans False."""
        code = 'foo()\n'
        calls = _calls_by_name(parser, code, 'foo')
        assert calls, 'expected at least one foo call'
        c = calls[0]
        assert c.in_try is False
        assert c.in_except is False
        assert c.in_finally is False
        assert c.in_loop is False
        assert c.awaited is False

    def test_inside_try_body(self, parser: PythonParser) -> None:
        """Call inside try: body has in_try=True only."""
        code = 'try:\n    foo()\nexcept Exception:\n    pass\n'
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        c = calls[0]
        assert c.in_try is True
        assert c.in_except is False
        assert c.in_finally is False

    def test_inside_except_body(self, parser: PythonParser) -> None:
        """Call inside except: clause body has in_except=True only."""
        code = 'try:\n    pass\nexcept Exception:\n    foo()\n'
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        c = calls[0]
        assert c.in_try is False
        assert c.in_except is True
        assert c.in_finally is False

    def test_inside_finally_body(self, parser: PythonParser) -> None:
        """Call inside finally: clause has in_finally=True only."""
        code = (
            'try:\n'
            '    pass\n'
            'except Exception:\n'
            '    pass\n'
            'finally:\n'
            '    foo()\n'
        )
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        c = calls[0]
        assert c.in_try is False
        assert c.in_except is False
        assert c.in_finally is True

    def test_nested_try_inner_body(self, parser: PythonParser) -> None:
        """Call in inner try body inside outer try body: in_try=True."""
        code = (
            'try:\n'
            '    try:\n'
            '        foo()\n'
            '    except Exception:\n'
            '        pass\n'
            'except Exception:\n'
            '    pass\n'
        )
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        c = calls[0]
        assert c.in_try is True

    def test_except_inside_outer_try(self, parser: PythonParser) -> None:
        """Except body of inner try inside outer try body: counter semantics right.

        The inner except clause is inside the outer try body, so the outer
        try_depth=1 is still active. The call gets both in_except=True (inner
        except depth) and in_try=True (outer try depth). This exercises that the
        counter-based stack composes correctly across nesting levels.
        """
        code = (
            'try:\n'
            '    try:\n'
            '        pass\n'
            '    except Exception:\n'
            '        foo()\n'
            'except Exception:\n'
            '    pass\n'
        )
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        c = calls[0]
        # Inner except depth > 0 because we are inside the inner except clause.
        assert c.in_except is True
        # Outer try depth > 0 because the inner except runs inside the outer try.
        assert c.in_try is True

    def test_for_loop(self, parser: PythonParser) -> None:
        """Call inside for loop body has in_loop=True."""
        code = 'for i in range(10):\n    foo()\n'
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        assert calls[0].in_loop is True

    def test_nested_for_loops(self, parser: PythonParser) -> None:
        """Call inside nested for loops still has in_loop=True."""
        code = 'for i in range(3):\n    for j in range(3):\n        foo()\n'
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        assert calls[0].in_loop is True

    def test_while_loop(self, parser: PythonParser) -> None:
        """Call inside while loop body has in_loop=True."""
        code = 'while True:\n    foo()\n    break\n'
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        assert calls[0].in_loop is True

    def test_async_with_single_manager(self, parser: PythonParser) -> None:
        """async with: populates context_managers with the expression text."""
        code = 'async def run():\n    async with sem:\n        foo()\n'
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        assert calls[0].context_managers == ('sem',)

    def test_nested_async_with_outer_first(self, parser: PythonParser) -> None:
        """Nested async with: outer manager listed before inner."""
        code = (
            'async def run():\n'
            '    async with sem:\n'
            '        async with lock:\n'
            '            foo()\n'
        )
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        assert calls[0].context_managers == ('sem', 'lock')

    def test_multi_item_with(self, parser: PythonParser) -> None:
        """Multi-item with A, B: produces (A, B) in source order."""
        code = "with open('a') as fa, open('b') as fb:\n    foo()\n"
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        cms = calls[0].context_managers
        assert len(cms) == 2
        assert "open('a')" in cms[0]
        assert "open('b')" in cms[1]

    def test_context_managers_truncated_at_80_chars(
        self, parser: PythonParser
    ) -> None:
        """Context manager expression longer than 80 chars is truncated."""
        long_expr = 'some_long_function_name(' + 'x' * 100 + ')'
        code = f'with {long_expr}:\n    foo()\n'
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        cms = calls[0].context_managers
        assert len(cms) == 1
        assert len(cms[0]) <= 80

    def test_await_expression_sets_awaited(self, parser: PythonParser) -> None:
        """Call under await expression has awaited=True."""
        code = 'async def run():\n    await foo()\n'
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        assert calls[0].awaited is True

    def test_nested_await_both_awaited(self, parser: PythonParser) -> None:
        """In nested await asyncio.gather(await foo()), both calls awaited=True."""
        code = (
            'import asyncio\n'
            'async def run():\n'
            '    await asyncio.gather(await foo())\n'
        )
        result = parser.parse(code, 'test.py')
        gather_calls = [c for c in result.calls if c.name == 'gather']
        foo_calls = [c for c in result.calls if c.name == 'foo']
        assert gather_calls, 'expected gather call'
        assert foo_calls, 'expected foo call'
        assert gather_calls[0].awaited is True
        assert foo_calls[0].awaited is True


class TestReturnConsumption:
    """return_consumption field classification tests."""

    def test_bare_call_ignored(self, parser: PythonParser) -> None:
        """Standalone expression-statement call classified as 'ignored'."""
        code = 'foo()\n'
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        assert calls[0].return_consumption == 'ignored'

    def test_assignment_stored(self, parser: PythonParser) -> None:
        """Assignment target classified as 'stored'."""
        code = 'x = foo()\n'
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        assert calls[0].return_consumption == 'stored'

    def test_return_statement_passed_through(
        self, parser: PythonParser
    ) -> None:
        """Call inside return statement classified as 'passed_through'."""
        code = 'def bar():\n    return foo()\n'
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        assert calls[0].return_consumption == 'passed_through'

    def test_await_classified_awaited(self, parser: PythonParser) -> None:
        """Call under await expression classified as 'awaited'."""
        code = 'async def run():\n    await foo()\n'
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        assert calls[0].return_consumption == 'awaited'

    def test_argument_position_stored(self, parser: PythonParser) -> None:
        """Call in argument position classified as 'stored' in Phase-4a.

        Phase-4a deviation: argument-position uses 'stored' rather than a
        dedicated 'argument' value (spec note: may be refined in a later phase).
        """
        code = 'process(fetch())\n'
        calls = _calls_by_name(parser, code, 'fetch')
        assert calls
        assert calls[0].return_consumption == 'stored'


class TestDispatcherRecognisers:
    """_classify_dispatch_kind recognises asyncio, executor, and Celery patterns."""

    def test_asyncio_create_task(self, parser: PythonParser) -> None:
        """asyncio.create_task() -> detached_task."""
        code = 'asyncio.create_task(coro())\n'
        calls = _calls_by_name(parser, code, 'create_task')
        assert calls
        assert calls[0].dispatch_kind == 'detached_task'

    def test_asyncio_ensure_future(self, parser: PythonParser) -> None:
        """asyncio.ensure_future() -> detached_task."""
        code = 'asyncio.ensure_future(coro())\n'
        calls = _calls_by_name(parser, code, 'ensure_future')
        assert calls
        assert calls[0].dispatch_kind == 'detached_task'

    def test_non_asyncio_create_task(self, parser: PythonParser) -> None:
        """tg.create_task() -> detached_task regardless of receiver."""
        code = 'tg.create_task(coro())\n'
        calls = _calls_by_name(parser, code, 'create_task')
        assert calls
        assert calls[0].dispatch_kind == 'detached_task'

    def test_loop_call_soon(self, parser: PythonParser) -> None:
        """loop.call_soon() -> detached_task."""
        code = 'loop.call_soon(cb)\n'
        calls = _calls_by_name(parser, code, 'call_soon')
        assert calls
        assert calls[0].dispatch_kind == 'detached_task'

    def test_loop_call_later(self, parser: PythonParser) -> None:
        """loop.call_later() -> detached_task."""
        code = 'loop.call_later(1, cb)\n'
        calls = _calls_by_name(parser, code, 'call_later')
        assert calls
        assert calls[0].dispatch_kind == 'detached_task'

    def test_loop_call_soon_threadsafe(self, parser: PythonParser) -> None:
        """loop.call_soon_threadsafe() -> detached_task."""
        code = 'loop.call_soon_threadsafe(cb)\n'
        calls = _calls_by_name(parser, code, 'call_soon_threadsafe')
        assert calls
        assert calls[0].dispatch_kind == 'detached_task'

    def test_run_in_executor_none_executor(self, parser: PythonParser) -> None:
        """loop.run_in_executor(None, fn) -> detached_task."""
        code = 'loop.run_in_executor(None, fn)\n'
        calls = _calls_by_name(parser, code, 'run_in_executor')
        assert calls
        assert calls[0].dispatch_kind == 'detached_task'

    def test_run_in_executor_pool_executor(self, parser: PythonParser) -> None:
        """loop.run_in_executor(pool, fn) -> thread_executor."""
        code = 'loop.run_in_executor(pool, fn)\n'
        calls = _calls_by_name(parser, code, 'run_in_executor')
        assert calls
        assert calls[0].dispatch_kind == 'thread_executor'

    def test_run_in_executor_no_args(self, parser: PythonParser) -> None:
        """loop.run_in_executor() with no args -> thread_executor.

        The implementation checks whether the first arg is None; with no args
        the check finds no named child, first_arg_is_none stays False, so it
        falls through to 'thread_executor'. The spec note about treating empty
        arg-lists like None is NOT implemented in Phase-4a.
        """
        code = 'loop.run_in_executor()\n'
        calls = _calls_by_name(parser, code, 'run_in_executor')
        assert calls
        assert calls[0].dispatch_kind == 'thread_executor'

    def test_thread_pool_executor_submit(self, parser: PythonParser) -> None:
        """ThreadPoolExecutor.submit() -> thread_executor."""
        code = 'ThreadPoolExecutor.submit(fn)\n'
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'thread_executor'

    def test_process_pool_executor_submit(self, parser: PythonParser) -> None:
        """ProcessPoolExecutor.submit() -> process_executor."""
        code = 'ProcessPoolExecutor.submit(fn)\n'
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'process_executor'

    def test_generic_executor_submit_fallback(
        self, parser: PythonParser
    ) -> None:
        """Unknown receiver ending in 'Executor' .submit() -> direct.

        Substring heuristics were removed in Phase 4a-follow-up.
        Unresolvable receiver types fall through to 'direct'.
        """
        code = 'MyCustomExecutor.submit(fn)\n'
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'direct'

    def test_celery_shared_task_apply_async(
        self, parser: PythonParser
    ) -> None:
        """@shared_task decorated function .apply_async() -> enqueued_job."""
        code = (
            'from celery import shared_task\n'
            '\n'
            '@shared_task\n'
            'def my_task():\n'
            '    pass\n'
            '\n'
            'my_task.apply_async(args=[1])\n'
        )
        calls = _calls_by_name(parser, code, 'apply_async')
        assert calls
        assert calls[0].dispatch_kind == 'enqueued_job'

    def test_celery_app_task_delay(self, parser: PythonParser) -> None:
        """@app.task decorated function .delay() -> enqueued_job."""
        code = '@app.task\ndef send_email():\n    pass\n\nsend_email.delay()\n'
        calls = _calls_by_name(parser, code, 'delay')
        assert calls
        assert calls[0].dispatch_kind == 'enqueued_job'

    def test_celery_undecorated_apply_async_is_direct(
        self, parser: PythonParser
    ) -> None:
        """Undecorated function .apply_async() -> direct (no decorator match)."""
        code = 'def plain_func():\n    pass\n\nplain_func.apply_async()\n'
        calls = _calls_by_name(parser, code, 'apply_async')
        assert calls
        assert calls[0].dispatch_kind == 'direct'

    def test_bare_create_task_detached(self, parser: PythonParser) -> None:
        """Bare create_task() name -> detached_task (known false-positive risk)."""
        code = 'create_task(coro())\n'
        calls = _calls_by_name(parser, code, 'create_task')
        assert calls
        assert calls[0].dispatch_kind == 'detached_task'

    def test_known_class_static_call_resolves(
        self, parser: PythonParser
    ) -> None:
        """ThreadPoolExecutor.submit(fn) at module scope -> thread_executor.

        No binding context; the fast path recognises the receiver as a known
        class name via _DISPATCH_KNOWN_CLASSES.
        """
        code = 'ThreadPoolExecutor.submit(fn)\n'
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'thread_executor'

    def test_unknown_class_static_call_is_direct(
        self, parser: PythonParser
    ) -> None:
        """MyCustomThing.submit(fn) at module scope -> direct.

        Only classes in _DISPATCH_KNOWN_CLASSES get the static-style fast
        path. Unknown classes fall through to direct.
        """
        code = 'MyCustomThing.submit(fn)\n'
        calls = _calls_by_name(parser, code, 'submit')
        assert calls
        assert calls[0].dispatch_kind == 'direct'

    def test_plain_call_direct(self, parser: PythonParser) -> None:
        """Ordinary function call -> direct."""
        code = 'foo()\n'
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        assert calls[0].dispatch_kind == 'direct'


class TestExceptionTypeReferences:
    """Exception-type pseudo-CallInfos carry zeroed scope fields."""

    def test_except_type_zeroed_scope(self, parser: PythonParser) -> None:
        """except ValueError: emits zeroed scope fields on the type reference.

        Guards against Major #5 from the review committee: type references
        must not inherit the stack snapshot (they are not call sites).
        """
        code = 'try:\n    pass\nexcept ValueError:\n    pass\n'
        result = parser.parse(code, 'test.py')
        val_err_refs = [c for c in result.calls if c.name == 'ValueError']
        assert val_err_refs, 'expected ValueError type reference'
        c = val_err_refs[0]
        assert c.in_try is False
        assert c.in_except is False
        assert c.in_finally is False

    def test_raise_type_zeroed_scope(self, parser: PythonParser) -> None:
        """raise ValueError inside try body: type reference has zeroed scope.

        The raise_statement arm in _extract_calls_recursive uses
        _make_type_reference_callinfo which ignores the current stack.
        """
        code = 'try:\n    raise ValueError\nexcept Exception:\n    pass\n'
        result = parser.parse(code, 'test.py')
        val_err_refs = [c for c in result.calls if c.name == 'ValueError']
        assert val_err_refs, 'expected ValueError type reference'
        c = val_err_refs[0]
        assert c.in_try is False
        assert c.in_except is False
        assert c.in_finally is False


class TestExtractContextManagerStrings:
    """_extract_context_manager_strings helper produces correct text."""

    def test_single_with_open(self, parser: PythonParser) -> None:
        """Single with open(f) as fh -> ['open(f)']."""
        code = "with open('x') as fh:\n    foo()\n"
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        cms = list(calls[0].context_managers)
        assert len(cms) == 1
        assert 'open' in cms[0]

    def test_multi_item_with_two_opens(self, parser: PythonParser) -> None:
        """with open(a), open(b): -> two entries."""
        code = "with open('a') as fa, open('b') as fb:\n    foo()\n"
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        cms = calls[0].context_managers
        assert len(cms) == 2

    def test_truncated_at_80_chars(self, parser: PythonParser) -> None:
        """Expression longer than 80 chars is truncated at exactly 80 chars."""
        long_var = 'a' * 100
        code = f'with {long_var}:\n    foo()\n'
        calls = _calls_by_name(parser, code, 'foo')
        assert calls
        cms = calls[0].context_managers
        assert cms, 'expected at least one context manager string'
        assert len(cms[0]) == 80


class TestAxonSourceMostlyDirect:
    """False-positive benchmark: axon source should be >= 95 percent direct calls."""

    def test_dispatch_kind_ratio(self) -> None:
        """Parse all .py files under src/axon/ and check direct ratio >= 0.95."""
        src_root = Path(__file__).resolve().parents[2] / 'src' / 'axon'
        py_files = list(src_root.rglob('*.py'))
        assert py_files, 'no .py files found under src/axon/'

        parser_inst = PythonParser()
        total = 0
        direct_count = 0

        for py_file in py_files:
            content = py_file.read_text(encoding='utf-8')
            if not content.strip():
                continue
            result = parser_inst.parse(content, str(py_file))
            for call in result.calls:
                total += 1
                if call.dispatch_kind == 'direct':
                    direct_count += 1

        assert total > 0, 'no calls collected from axon source'
        ratio = direct_count / total
        assert ratio >= 0.95, (
            f'direct ratio {ratio:.3f} < 0.95 '
            f'(direct={direct_count}, total={total})'
        )

        thread_executor_submit_map = sum(
            1
            for py_file in py_files
            for call in parser_inst.parse(
                py_file.read_text(encoding='utf-8'), str(py_file)
            ).calls
            if call.dispatch_kind == 'thread_executor'
            and call.name in {'submit', 'map'}
        )
        assert thread_executor_submit_map >= 7, (
            f'expected >= 7 thread_executor submit/map calls in axon source, '
            f'got {thread_executor_submit_map}'
        )
