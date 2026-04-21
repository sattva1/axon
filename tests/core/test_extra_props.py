"""Tests for CallInfo.extra_props() sparse serialization."""

from __future__ import annotations

from axon.core.parsers.base import CallInfo


def _make_callinfo(**kwargs: object) -> CallInfo:
    """Return a CallInfo with all Phase-4a fields at their defaults.

    Keyword args override individual fields. Used for concise equality
    assertions without repeating every default.
    """
    defaults: dict[str, object] = dict(
        name='x',
        line=1,
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


class TestCallInfoExtraProps:
    """extra_props() sparse encoding tests."""

    def test_all_defaults_returns_empty(self) -> None:
        """Default CallInfo produces empty extra_props."""
        c = CallInfo(name='x', line=1)
        assert c.extra_props() == {}

    def test_dispatch_kind_non_default(self) -> None:
        """Non-default dispatch_kind is included."""
        c = CallInfo(name='x', line=1, dispatch_kind='detached_task')
        assert c.extra_props() == {'dispatch_kind': 'detached_task'}

    def test_bool_flags_in_try_awaited(self) -> None:
        """Multiple non-default bool flags are included together."""
        c = CallInfo(name='x', line=1, in_try=True, awaited=True)
        assert c.extra_props() == {'in_try': True, 'awaited': True}

    def test_context_managers_serialized_as_list(self) -> None:
        """Non-empty context_managers serialised as list, not tuple."""
        c = CallInfo(name='x', line=1, context_managers=('sem',))
        props = c.extra_props()
        assert props == {'context_managers': ['sem']}
        assert isinstance(props['context_managers'], list)

    def test_return_consumption_non_default(self) -> None:
        """Non-default return_consumption is included."""
        c = CallInfo(name='x', line=1, return_consumption='awaited')
        assert c.extra_props() == {'return_consumption': 'awaited'}

    def test_empty_context_managers_not_emitted(self) -> None:
        """Empty context_managers tuple does NOT appear in extra_props."""
        c = CallInfo(name='x', line=1, context_managers=())
        assert 'context_managers' not in c.extra_props()

    def test_original_fields_excluded(self) -> None:
        """name, line, receiver, arguments are never in extra_props."""
        c = CallInfo(name='foo', line=5, receiver='self', arguments=['cb'])
        keys = set(c.extra_props().keys())
        assert not keys.intersection({'name', 'line', 'receiver', 'arguments'})

    def test_multiple_non_defaults(self) -> None:
        """All non-default Phase-4a fields appear simultaneously."""
        c = _make_callinfo(
            dispatch_kind='enqueued_job',
            in_try=True,
            in_loop=True,
            context_managers=('lock', 'conn'),
            return_consumption='ignored',
        )
        props = c.extra_props()
        assert props['dispatch_kind'] == 'enqueued_job'
        assert props['in_try'] is True
        assert props['in_loop'] is True
        assert props['context_managers'] == ['lock', 'conn']
        assert props['return_consumption'] == 'ignored'
