"""Path and diff routing helpers for multi-repo MCP dispatch.

``route_for_path`` and ``route_for_diff`` determine which registered repo
owns a given file path or diff, so the dispatch layer can route the call
to the right storage backend without requiring the caller to pass an
explicit ``repo`` argument.

Both functions raise ``RoutingError`` on ambiguity or unresolvable input.
The dispatch layer catches that and renders it as a tool-result string.
"""

from __future__ import annotations

import re
from pathlib import Path

from axon.core.repos import RepoEntry, RepoNotFound, RepoResolver

_DIFF_FILE_PATTERN = re.compile(r'^diff --git a/(.+?) b/(.+?)$', re.MULTILINE)


class RoutingError(Exception):
    """Raised when path or diff routing cannot determine a unique repo.

    ``candidates`` carries the slugs of all repos considered so the caller
    can surface a helpful disambiguation message.
    """

    def __init__(self, message: str, candidates: list[str]) -> None:
        self.candidates = candidates
        super().__init__(message)


def route_for_path(
    resolver: RepoResolver, file_path: str, explicit_repo: str | None
) -> RepoEntry:
    """Resolve the repo that owns *file_path*.

    When *explicit_repo* is provided it is resolved directly via
    ``resolver.resolve_strict`` and returned - no path matching is
    attempted.

    Without an explicit repo the resolved absolute path is tested against
    every known repo's path prefix:
    - Exactly one match: returned.
    - Multiple matches: ``RoutingError`` (ambiguous).
    - Zero matches: falls back to the local repo; raises if no local repo.

    Args:
        resolver: RepoResolver with the current session's registry.
        file_path: File path string to route (may be relative or absolute).
        explicit_repo: Slug, absolute path, or relative path of the target
            repo, or None to auto-detect.

    Returns:
        The RepoEntry for the owning repo.

    Raises:
        RoutingError: When the explicit repo is not found, or when the path
            is ambiguous between multiple repos, or when no local repo is
            available as fallback.
    """
    if explicit_repo is not None:
        try:
            return resolver.resolve_strict(explicit_repo)
        except RepoNotFound as exc:
            raise RoutingError(
                f"repo '{explicit_repo}' not found", candidates=exc.candidates
            ) from exc

    abs_path = _safe_resolve(file_path)

    matches: list[RepoEntry] = []
    if abs_path is not None:
        for entry in resolver.list_known():
            try:
                entry_resolved = entry.path.resolve()
            except (OSError, ValueError):
                continue
            if abs_path == entry_resolved or _is_relative_to(
                abs_path, entry_resolved
            ):
                matches.append(entry)

    if len(matches) > 1:
        candidates = [e.slug for e in matches]
        raise RoutingError(
            f"path '{file_path}' is ambiguous between repos: "
            f'{", ".join(candidates)}',
            candidates=candidates,
        )

    if len(matches) == 1:
        return matches[0]

    # Zero matches - fall back to local.
    local = resolver.local()
    if local is not None:
        return local

    known_slugs = [e.slug for e in resolver.list_known()]
    raise RoutingError(
        f"path '{file_path}' could not be routed to any known repo "
        f'and no local repo is configured',
        candidates=known_slugs,
    )


def route_for_diff(
    resolver: RepoResolver, diff: str, explicit_repo: str | None
) -> RepoEntry:
    """Resolve the repo that owns all files in *diff*.

    When *explicit_repo* is provided, routing is delegated to
    ``route_for_path`` with the explicit value (no diff parsing).

    Without an explicit repo the diff is parsed to extract file paths;
    each path is routed individually.  When all paths map to the same
    repo that repo is returned.  A cross-repo diff raises ``RoutingError``.
    An empty or unparse-able diff falls back to the local repo.

    Args:
        resolver: RepoResolver with the current session's registry.
        diff: Raw git diff string.
        explicit_repo: Slug, absolute path, or relative path of the target
            repo, or None to auto-detect from diff contents.

    Returns:
        The single RepoEntry owning all diff paths.

    Raises:
        RoutingError: When the explicit repo is not found, when diff paths
            span multiple repos, or when no local repo is available as
            fallback.
    """
    if explicit_repo is not None:
        # Delegate to route_for_path for consistent resolution + error
        # message (path arg is unused when explicit_repo is set).
        return route_for_path(resolver, '', explicit_repo)

    file_paths = _parse_diff_file_paths(diff)

    if not file_paths:
        local = resolver.local()
        if local is not None:
            return local
        known_slugs = [e.slug for e in resolver.list_known()]
        raise RoutingError(
            'diff is empty and no local repo is configured',
            candidates=known_slugs,
        )

    seen_slugs: dict[str, RepoEntry] = {}
    for path in file_paths:
        try:
            entry = route_for_path(resolver, path, None)
        except RoutingError:
            # Unresolvable path - treat as local (conservative fallback).
            local = resolver.local()
            if local is None:
                continue
            entry = local
        seen_slugs[entry.slug] = entry

    if len(seen_slugs) == 0:
        local = resolver.local()
        if local is not None:
            return local
        known_slugs = [e.slug for e in resolver.list_known()]
        raise RoutingError(
            'could not route any diff file path to a known repo',
            candidates=known_slugs,
        )

    if len(seen_slugs) == 1:
        return next(iter(seen_slugs.values()))

    # Multiple repos - this diff spans a cross-repo change.
    candidates = sorted(seen_slugs.keys())
    raise RoutingError(
        f'diff spans multiple repos: {", ".join(candidates)}',
        candidates=candidates,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_resolve(path_str: str) -> Path | None:
    """Resolve *path_str* to an absolute Path, returning None on error."""
    try:
        return Path(path_str).resolve()
    except (OSError, ValueError):
        return None


def _is_relative_to(child: Path, parent: Path) -> bool:
    """Return True when *child* is under *parent*.

    Wraps ``Path.is_relative_to`` (Python 3.9+) with a try/except fallback
    for unusual runtime environments.
    """
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _parse_diff_file_paths(diff: str) -> list[str]:
    """Extract the set of unique file paths from a git diff string."""
    paths: list[str] = []
    seen: set[str] = set()
    for m in _DIFF_FILE_PATTERN.finditer(diff):
        path = m.group(2)
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths
