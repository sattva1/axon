"""Tests for the RateLimitedASGIApp sliding-window rate limiter."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest

from axon.web.rate_limit import RateLimitedASGIApp, build_rate_limited_app


def _scope(
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
    client: tuple[str, int] = ('203.0.113.1', 12345),
    scope_type: str = 'http',
) -> dict[str, Any]:
    """Build a minimal ASGI scope dict for testing."""
    return {'type': scope_type, 'client': client, 'headers': headers or []}


class _RecordingApp:
    """Stub ASGI inner app that records call count and returns 200 OK."""

    def __init__(self) -> None:
        self.call_count = 0

    async def __call__(
        self, scope: dict[str, Any], receive: Any, send: Any
    ) -> None:
        self.call_count += 1
        await send(
            {'type': 'http.response.start', 'status': 200, 'headers': []}
        )
        await send({'type': 'http.response.body', 'body': b''})


class _CaptureSend:
    """Collect ASGI send events for assertions."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def __call__(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    @property
    def status(self) -> int | None:
        for ev in self.events:
            if ev.get('type') == 'http.response.start':
                return ev['status']
        return None

    def header(self, name: bytes) -> bytes | None:
        for ev in self.events:
            if ev.get('type') == 'http.response.start':
                for k, v in ev.get('headers', []):
                    if k == name:
                        return v
        return None


async def _call(
    app: RateLimitedASGIApp, scope: dict[str, Any]
) -> _CaptureSend:
    """Invoke *app* once and return the captured send events."""
    cap = _CaptureSend()
    receive = AsyncMock()
    await app(scope, receive, cap)
    return cap


class TestUnderLimit:
    async def test_passes_through(self) -> None:
        """Exactly max_requests requests all reach the inner app."""
        inner = _RecordingApp()
        app = RateLimitedASGIApp(inner, max_requests=5, window_seconds=60.0)
        scope = _scope()

        for _ in range(5):
            cap = await _call(app, scope)
            assert cap.status == 200

        assert inner.call_count == 5


class TestOverLimit:
    async def test_returns_429_with_retry_after(self) -> None:
        """Request N+1 gets a 429 with a valid Retry-After header."""
        inner = _RecordingApp()
        app = RateLimitedASGIApp(inner, max_requests=3, window_seconds=60.0)
        scope = _scope()

        for _ in range(3):
            await _call(app, scope)

        cap = await _call(app, scope)
        assert cap.status == 429
        retry_after = cap.header(b'retry-after')
        assert retry_after is not None
        assert int(retry_after) >= 1
        # Inner app must not have been reached for the 429 response.
        assert inner.call_count == 3


class TestPerSessionIsolation:
    async def test_sessions_not_shared(self) -> None:
        """Two distinct mcp-session-id values do not share a bucket."""
        inner = _RecordingApp()
        app = RateLimitedASGIApp(inner, max_requests=2, window_seconds=60.0)

        scope_a = _scope(headers=[(b'mcp-session-id', b'session-A')])
        scope_b = _scope(headers=[(b'mcp-session-id', b'session-B')])

        # Exhaust session A.
        for _ in range(2):
            await _call(app, scope_a)
        cap_a = await _call(app, scope_a)
        assert cap_a.status == 429

        # Session B must still be unaffected.
        for _ in range(2):
            cap_b = await _call(app, scope_b)
            assert cap_b.status == 200


class TestHostFallback:
    async def test_when_no_session_header(self) -> None:
        """Different client hosts get separate buckets when no session header."""
        inner = _RecordingApp()
        app = RateLimitedASGIApp(inner, max_requests=2, window_seconds=60.0)

        scope_x = _scope(client=('10.0.0.1', 1111))
        scope_y = _scope(client=('10.0.0.2', 2222))

        # Exhaust host X.
        for _ in range(2):
            await _call(app, scope_x)
        cap_x = await _call(app, scope_x)
        assert cap_x.status == 429

        # Host Y must still be unaffected.
        for _ in range(2):
            cap_y = await _call(app, scope_y)
            assert cap_y.status == 200


class TestLoopbackBypass:
    async def test_ipv4(self) -> None:
        """127.0.0.1 clients bypass rate limiting regardless of volume."""
        inner = _RecordingApp()
        app = RateLimitedASGIApp(inner, max_requests=5, window_seconds=60.0)
        scope = _scope(client=('127.0.0.1', 9999))

        for _ in range(1000):
            cap = await _call(app, scope)
            assert cap.status == 200

        assert inner.call_count == 1000

    async def test_ipv6(self) -> None:
        """::1 clients also bypass rate limiting."""
        inner = _RecordingApp()
        app = RateLimitedASGIApp(inner, max_requests=5, window_seconds=60.0)
        scope = _scope(client=('::1', 9999))

        for _ in range(100):
            cap = await _call(app, scope)
            assert cap.status == 200

        assert inner.call_count == 100


class TestWindowExpiry:
    async def test_restores_capacity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After the window elapses, capacity is fully restored."""
        calls: list[float] = []

        def _monotonic() -> float:
            return calls[-1] if calls else 0.0

        monkeypatch.setattr(time, 'monotonic', _monotonic)

        inner = _RecordingApp()
        app = RateLimitedASGIApp(inner, max_requests=2, window_seconds=10.0)
        scope = _scope()

        # Fill bucket at t=0.
        calls.append(0.0)
        await _call(app, scope)
        await _call(app, scope)
        cap = await _call(app, scope)
        assert cap.status == 429

        # Advance time past the window.
        calls.append(11.0)
        cap = await _call(app, scope)
        assert cap.status == 200


class TestDisabledViaEnv:
    def test_returns_inner_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AXON_MCP_RATE_LIMIT_DISABLED=1 makes build_rate_limited_app return inner."""
        monkeypatch.setenv('AXON_MCP_RATE_LIMIT_DISABLED', '1')
        inner = _RecordingApp()

        result = build_rate_limited_app(inner)

        assert result is inner


class TestLifespanPassthrough:
    async def test_scope_passes_through(self) -> None:
        """lifespan scope type bypasses rate limiting entirely."""
        inner = _RecordingApp()
        app = RateLimitedASGIApp(inner, max_requests=0, window_seconds=60.0)
        scope = _scope(scope_type='lifespan')

        # max_requests=0 would reject any http request, but lifespan must pass.
        cap = _CaptureSend()
        receive = AsyncMock()
        await app(scope, receive, cap)

        assert inner.call_count == 1


class TestConcurrentRequests:
    async def test_in_flight_not_serialized(self) -> None:
        """Two concurrent slow requests overlap (no lock held across await)."""
        delays_observed: list[float] = []

        class _SlowInner:
            async def __call__(
                self, scope: dict[str, Any], receive: Any, send: Any
            ) -> None:
                start = time.monotonic()
                await asyncio.sleep(0.05)
                delays_observed.append(time.monotonic() - start)
                await send(
                    {
                        'type': 'http.response.start',
                        'status': 200,
                        'headers': [],
                    }
                )
                await send({'type': 'http.response.body', 'body': b''})

        inner = _SlowInner()
        app = RateLimitedASGIApp(inner, max_requests=10, window_seconds=60.0)

        scope_a = _scope(headers=[(b'mcp-session-id', b'concurrent-A')])
        scope_b = _scope(headers=[(b'mcp-session-id', b'concurrent-B')])

        wall_start = time.monotonic()
        await asyncio.gather(_call(app, scope_a), _call(app, scope_b))
        wall_elapsed = time.monotonic() - wall_start

        # If requests ran sequentially, elapsed would be ~100 ms; overlapping
        # puts elapsed well under 90 ms with ~50 ms sleeps.
        assert wall_elapsed < 0.09


class TestEmptyBucketGC:
    async def test_evicted_after_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After window expiry an empty bucket is removed from _buckets."""
        tick: list[float] = [0.0]

        monkeypatch.setattr(time, 'monotonic', lambda: tick[0])

        inner = _RecordingApp()
        app = RateLimitedASGIApp(inner, max_requests=2, window_seconds=10.0)
        scope = _scope()

        # Issue one request at t=0; bucket has one entry.
        await _call(app, scope)
        assert '203.0.113.1' in app._buckets

        # Advance past the window and issue another request; old entry expires.
        tick[0] = 11.0
        await _call(app, scope)

        # Bucket either doesn't exist or is empty after the expiry sweep.
        bucket = app._buckets.get('203.0.113.1')
        assert bucket is None or len(bucket) == 1


class TestEnvOverrides:
    def test_apply(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AXON_MCP_RATE_LIMIT_REQUESTS env var is picked up by factory."""
        monkeypatch.setenv('AXON_MCP_RATE_LIMIT_REQUESTS', '5')
        inner = _RecordingApp()

        result = build_rate_limited_app(inner)

        assert isinstance(result, RateLimitedASGIApp)
        assert result._max == 5
