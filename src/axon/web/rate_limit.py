"""Sliding-window rate limiter for the MCP HTTP transport.

Wraps an inner ASGI app and rejects requests that exceed
AXON_MCP_RATE_LIMIT_REQUESTS per AXON_MCP_RATE_LIMIT_WINDOW_SECONDS,
keyed per session/host. Loopback traffic bypasses the limit so local
axon serve --watch proxies are unaffected.

Concurrency note: all MCP HTTP traffic flows through one uvicorn worker.
The bucket check-and-append block contains no await points, so it is
atomic from the perspective of other coroutines on the same event loop.
No lock is needed. Multi-worker deployments would require a shared store
(e.g. Redis) and are out of scope (YAGNI).
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from typing import Any

DEFAULT_MAX_REQUESTS = 200
DEFAULT_WINDOW_SECONDS = 60.0
SESSION_HEADER = b'mcp-session-id'
LOOPBACK_HOSTS = frozenset({'127.0.0.1', '::1'})

_429_BODY = (
    b'{"jsonrpc":"2.0","id":null,'
    b'"error":{"code":-32000,"message":"Rate limit exceeded"}}'
)


class RateLimitedASGIApp:
    """Sliding-window rate limiter wrapping an inner ASGI application."""

    def __init__(
        self, app: Any, *, max_requests: int, window_seconds: float
    ) -> None:
        """Store inner app and rate-limit parameters.

        Args:
            app: Inner ASGI application to wrap.
            max_requests: Maximum requests allowed per window per key.
            window_seconds: Length of the sliding window in seconds.
        """
        self._inner = app
        self._max = max_requests
        self._window = window_seconds
        self._buckets: dict[str, deque[float]] = {}

    async def __call__(
        self, scope: dict[str, Any], receive: Any, send: Any
    ) -> None:
        """Process an ASGI request, applying rate limiting for HTTP scopes."""
        if scope['type'] != 'http':
            await self._inner(scope, receive, send)
            return

        client: tuple[str, int] | None = scope.get('client')
        if client and client[0] in LOOPBACK_HOSTS:
            await self._inner(scope, receive, send)
            return

        headers: dict[bytes, bytes] = {
            name: value for name, value in scope.get('headers', [])
        }
        session_id = headers.get(SESSION_HEADER)
        if session_id:
            key = session_id.decode()
        elif client:
            key = client[0]
        else:
            key = '<anonymous>'

        now = time.monotonic()
        bucket = self._buckets.setdefault(key, deque())
        while bucket and now - bucket[0] >= self._window:
            bucket.popleft()
        if not bucket:
            self._buckets.pop(key, None)
        if len(bucket) >= self._max:
            oldest = bucket[0]
            retry_after = max(1, math.ceil(self._window - (now - oldest)))
            await self._send_429(send, retry_after)
            return
        # Re-fetch in case GC above removed the bucket entry.
        self._buckets.setdefault(key, deque()).append(now)

        await self._inner(scope, receive, send)

    async def _send_429(self, send: Any, retry_after: int) -> None:
        """Send a 429 Too Many Requests response with a JSON-RPC error body.

        Args:
            send: ASGI send callable.
            retry_after: Seconds the client should wait before retrying.
        """
        await send(
            {
                'type': 'http.response.start',
                'status': 429,
                'headers': [
                    (b'content-type', b'application/json'),
                    (b'retry-after', str(retry_after).encode()),
                ],
            }
        )
        await send({'type': 'http.response.body', 'body': _429_BODY})


def build_rate_limited_app(inner: Any) -> Any:
    """Return a rate-limited wrapper around the inner ASGI app.

    Reads configuration from environment variables each call so tests can
    monkeypatch os.environ without module-reload tricks.

    - AXON_MCP_RATE_LIMIT_DISABLED: when set to 1/true/yes (case-insensitive),
      returns inner unchanged.
    - AXON_MCP_RATE_LIMIT_REQUESTS: max requests per window (default 200).
    - AXON_MCP_RATE_LIMIT_WINDOW_SECONDS: window length in seconds (default 60).

    Args:
        inner: The inner ASGI application to wrap.

    Returns:
        Either inner unchanged (when disabled) or a RateLimitedASGIApp.
    """
    disabled = os.environ.get('AXON_MCP_RATE_LIMIT_DISABLED', '').lower()
    if disabled in ('1', 'true', 'yes'):
        return inner

    try:
        max_requests = int(
            os.environ.get(
                'AXON_MCP_RATE_LIMIT_REQUESTS', DEFAULT_MAX_REQUESTS
            )
        )
    except ValueError:
        max_requests = DEFAULT_MAX_REQUESTS

    try:
        window_seconds = float(
            os.environ.get(
                'AXON_MCP_RATE_LIMIT_WINDOW_SECONDS', DEFAULT_WINDOW_SECONDS
            )
        )
    except ValueError:
        window_seconds = DEFAULT_WINDOW_SECONDS

    return RateLimitedASGIApp(
        inner, max_requests=max_requests, window_seconds=window_seconds
    )
