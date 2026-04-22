"""Staleness detection for MCP tool output.

Global analyses (dead code, communities) run either at full-index time
or during watcher global-phase passes. A handful of small-change
watcher invocations can advance last_incremental_at without
re-running those globals; this module renders the gap visible.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from axon.core.meta import MetaFile, load_meta

_STALENESS_THRESHOLD_SECONDS = 60


def _parse_iso(value: str) -> datetime | None:
    """Parse an ISO-8601 string, coercing naive results to UTC.

    Axon's own writers emit offset-aware timestamps via now_iso(), but
    legacy or hand-edited meta.json values may be naive. Subtracting
    a naive datetime from an aware one raises TypeError - which must
    never happen on a staleness-warning code path. This function
    normalises any naive parse result to UTC so the subtraction is
    always defined.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _warning_line(refreshed: datetime, delta_seconds: float) -> str:
    """Format a one-line staleness warning string."""
    return (
        f'Note: global analysis last refreshed at '
        f'{refreshed.isoformat()}; {int(delta_seconds)}s of incremental '
        f'reindexes since. Run `axon analyze` for fresh results.\n\n'
    )


def staleness_warning_dead_code(meta: MetaFile) -> str:
    """Return warning if dead_code analysis lags last_incremental_at
    by > 60 s; empty string otherwise."""
    refreshed = _parse_iso(meta.dead_code_last_refreshed_at)
    last_incremental = _parse_iso(meta.last_incremental_at)
    if refreshed is None or last_incremental is None:
        return ''
    delta = (last_incremental - refreshed).total_seconds()
    if delta <= _STALENESS_THRESHOLD_SECONDS:
        return ''
    return _warning_line(refreshed, delta)


def staleness_warning_communities(meta: MetaFile) -> str:
    """Return warning if communities analysis lags last_incremental_at
    by > 60 s; empty string otherwise."""
    refreshed = _parse_iso(meta.communities_last_refreshed_at)
    last_incremental = _parse_iso(meta.last_incremental_at)
    if refreshed is None or last_incremental is None:
        return ''
    delta = (last_incremental - refreshed).total_seconds()
    if delta <= _STALENESS_THRESHOLD_SECONDS:
        return ''
    return _warning_line(refreshed, delta)


def render_with_dead_code_warning(repo_path: Path | None, body: str) -> str:
    """Prepend a dead-code staleness warning to body if applicable."""
    if repo_path is None:
        return body
    return staleness_warning_dead_code(load_meta(repo_path)) + body


def render_with_communities_warning(repo_path: Path | None, body: str) -> str:
    """Prepend a communities staleness warning to body if applicable."""
    if repo_path is None:
        return body
    return staleness_warning_communities(load_meta(repo_path)) + body
