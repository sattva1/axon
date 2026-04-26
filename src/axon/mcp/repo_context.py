"""Per-tool-call context for multi-repo MCP dispatch.

RepoContext is intentionally narrow - it carries only the information that
handlers need to operate on one repo. The pool, resolver, and drift cache
stay in the dispatch layer (mcp/server.py); fan-out helpers receive them
as explicit arguments. This keeps handlers from bypassing freshness filters
or the stale-major refusal guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from axon.core.storage.base import StorageBackend


@dataclass(frozen=True)
class RepoContext:
    """Immutable per-call context passed to every tool handler.

    Handlers read storage and repo metadata through this object only.
    The pool, resolver, and drift_cache live in the dispatch layer and
    are passed explicitly to fan-out helpers so handlers cannot bypass
    the freshness filter or stale-major refusal guard.
    """

    storage: StorageBackend
    slug: str
    is_local: bool
    repo_path: Path | None
    local_slug: str | None
