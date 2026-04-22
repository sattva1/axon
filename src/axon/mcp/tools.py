"""MCP tool handler implementations for Axon.

Each function accepts a storage backend and the tool-specific arguments,
performs the appropriate query, and returns a human-readable string suitable
for inclusion in an MCP ``TextContent`` response.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from axon.core.cypher_guard import WRITE_KEYWORDS, sanitize_cypher
from axon.core.embeddings.embedder import embed_query
from axon.core.ingestion.community import export_to_igraph
from axon.core.ingestion.parser_phase import get_parser
from axon.core.ingestion.test_classifier import (
    PytestConfig,
    is_test_file,
    load_pytest_config,
)
from axon.core.search.hybrid import hybrid_search
from axon.core.storage.base import StorageBackend
from axon.core.storage.kuzu_backend import escape_cypher as _escape_cypher
from axon.mcp.resources import get_dead_code_list

logger = logging.getLogger(__name__)

MAX_TRAVERSE_DEPTH = 10
MAX_CYPHER_LENGTH = 100_000  # Cap on raw Cypher query length (characters).
MAX_DIFF_LENGTH = 100_000  # Cap on raw diff input length (characters).

_SAFE_PATH = re.compile(r'^[a-zA-Z0-9._/\-\s]+$')


def _new_ref_id() -> str:
    """Generate an 8-hex-char correlation id for log/client pairing.

    32 bits of entropy; birthday-collision probability ~50% at ~65k concurrent
    entries - acceptable for log correlation at expected MCP traffic. If error
    volume ever exceeds that, bump to 12 chars (48 bits).
    """
    return uuid.uuid4().hex[:8]


def _confidence_tag(confidence: float) -> str:
    """Return a visual confidence indicator for edge display."""
    if confidence >= 0.9:
        return ""
    if confidence >= 0.5:
        return " (~)"
    return " (?)"


def _resolve_symbol(storage: StorageBackend, symbol: str) -> list:
    """Resolve a symbol name to search results, preferring exact name matches."""
    if hasattr(storage, "exact_name_search"):
        results = storage.exact_name_search(symbol, limit=1)
        if results:
            return results
    return storage.fts_search(symbol, limit=1)


def handle_list_repos(registry_dir: Path | None = None) -> str:
    """List indexed repositories by scanning for .axon directories.

    Scans the global registry directory (defaults to ``~/.axon/repos``) for
    project metadata files and returns a formatted summary.

    Args:
        registry_dir: Directory containing repo metadata. If ``None``,
            defaults to ``~/.axon/repos``.

    Returns:
        Formatted list of indexed repositories with stats, or a message
        indicating none were found.
    """
    use_cwd_fallback = registry_dir is None
    if registry_dir is None:
        registry_dir = Path.home() / ".axon" / "repos"

    repos: list[dict[str, Any]] = []

    if registry_dir.exists():
        for meta_file in registry_dir.glob("*/meta.json"):
            try:
                data = json.loads(meta_file.read_text())
                repos.append(data)
            except (json.JSONDecodeError, OSError):
                continue

    if not repos and use_cwd_fallback:
        cwd_axon = Path.cwd() / ".axon" / "meta.json"
        if cwd_axon.exists():
            try:
                data = json.loads(cwd_axon.read_text())
                repos.append(data)
            except (json.JSONDecodeError, OSError):
                pass

    if not repos:
        return "No indexed repositories found. Run `axon index` on a project first."

    lines = [f"Indexed repositories ({len(repos)}):"]
    lines.append("")
    for i, repo in enumerate(repos, 1):
        name = repo.get("name", "unknown")
        path = repo.get("path", "")
        stats = repo.get("stats", {})
        files = stats.get("files", "?")
        symbols = stats.get("symbols", "?")
        relationships = stats.get("relationships", "?")
        lines.append(f"  {i}. {name}")
        lines.append(f"     Path: {path}")
        lines.append(f"     Files: {files}  Symbols: {symbols}  Relationships: {relationships}")
        lines.append("")

    return "\n".join(lines)


def _group_by_process(
    results: list,
    storage: StorageBackend,
) -> dict[str, list]:
    """Map search results to their parent execution processes."""
    if not results:
        return {}

    node_ids = [r.node_id for r in results]
    try:
        node_to_process = storage.get_process_memberships(node_ids)
    except AttributeError:
        return {}

    groups: dict[str, list] = {}
    for r in results:
        pname = node_to_process.get(r.node_id)
        if pname:
            groups.setdefault(pname, []).append(r)

    return groups


def _format_query_results(results: list, groups: dict[str, list]) -> str:
    """Format search results with process grouping.

    Results belonging to a process appear under a labelled section.
    Remaining results appear in an "Other results" section.
    """
    grouped_ids: set[str] = {r.node_id for group in groups.values() for r in group}
    ungrouped = [r for r in results if r.node_id not in grouped_ids]

    lines: list[str] = []
    counter = 1

    for process_name, proc_results in groups.items():
        lines.append(f"=== {process_name} ===")
        for r in proc_results:
            label = r.label.title() if r.label else "Unknown"
            lines.append(f"{counter}. {r.node_name} ({label}) -- {r.file_path}")
            if r.snippet:
                snippet = r.snippet[:200].replace("\n", " ").strip()
                lines.append(f"   {snippet}")
            counter += 1
        lines.append("")

    if ungrouped:
        if groups:
            lines.append("=== Other results ===")
        for r in ungrouped:
            label = r.label.title() if r.label else "Unknown"
            lines.append(f"{counter}. {r.node_name} ({label}) -- {r.file_path}")
            if r.snippet:
                snippet = r.snippet[:200].replace("\n", " ").strip()
                lines.append(f"   {snippet}")
            counter += 1
        lines.append("")

    lines.append("Next: Use context() on a specific symbol for the full picture.")
    return "\n".join(lines)


def handle_query(storage: StorageBackend, query: str, limit: int = 20) -> str:
    """Execute hybrid search and format results, grouped by execution process.

    Args:
        storage: The storage backend to search against.
        query: Text search query.
        limit: Maximum number of results (default 20, capped at 100).

    Returns:
        Formatted search results grouped by process, with file, name, label,
        and snippet for each result.
    """
    limit = max(1, min(limit, 100))

    query_embedding = embed_query(query)
    if query_embedding is None:
        logger.warning("embed_query returned None; falling back to FTS-only search")

    results = hybrid_search(query, storage, query_embedding=query_embedding, limit=limit)
    if not results:
        return f"No results found for '{query}'."

    groups = _group_by_process(results, storage)
    return _format_query_results(results, groups)


def handle_context(storage: StorageBackend, symbol: str) -> str:
    """Provide a 360-degree view of a symbol.

    Looks up the symbol by name via full-text search, then retrieves its
    callers, callees, and type references.

    Args:
        storage: The storage backend.
        symbol: Plain name (``foo``) or dotted path (``Class.method``).
            Multi-dot paths (e.g. ``module.Class.method``) fall back to
            the last segment. When multiple symbols match (e.g., a class
            named ``Foo`` exists in two different files), the best match
            is chosen by score (source files over tests) then lexicographic
            node ID.

    Returns:
        Formatted view including callers, callees, type refs, and guidance.
    """
    if not symbol or not symbol.strip():
        return "Error: 'symbol' parameter is required and cannot be empty."

    results = _resolve_symbol(storage, symbol)
    if not results:
        return f"Symbol '{symbol}' not found."

    node = storage.get_node(results[0].node_id)
    if not node:
        return f"Symbol '{symbol}' not found."

    label_display = node.label.value.title() if node.label else "Unknown"
    lines = [f"Symbol: {node.name} ({label_display})"]
    lines.append(f"File: {node.file_path}:{node.start_line}-{node.end_line}")

    if node.signature:
        lines.append(f"Signature: {node.signature}")

    if node.is_dead:
        lines.append("Status: DEAD CODE (unreachable)")

    try:
        callers_meta = storage.get_callers_with_metadata(node.id)
        callers_raw = None
    except (AttributeError, TypeError):
        callers_meta = None
        try:
            callers_raw = storage.get_callers_with_confidence(node.id)
        except (AttributeError, TypeError):
            callers_raw = [(c, 1.0) for c in storage.get_callers(node.id)]

    if callers_meta is not None:
        if callers_meta:
            lines.append(f'\nCallers ({len(callers_meta)}):')
        for c, conf, meta in callers_meta:
            tag = _confidence_tag(conf)
            dispatch = meta.get('dispatch_kind', 'direct')
            extra_tags = ''
            if dispatch != 'direct':
                extra_tags += f'  [{dispatch}]'
            ret_consumption = meta.get('return_consumption')
            if ret_consumption and ret_consumption != 'ignored':
                extra_tags += f'  [return: {ret_consumption}]'
            lines.append(
                f'  -> {c.name}  {c.file_path}:{c.start_line}{extra_tags}{tag}'
            )
    elif callers_raw:
        lines.append(f'\nCallers ({len(callers_raw)}):')
        for c, conf in callers_raw:
            tag = _confidence_tag(conf)
            lines.append(f"  -> {c.name}  {c.file_path}:{c.start_line}{tag}")

    try:
        callees_meta = storage.get_callees_with_metadata(node.id)
        callees_raw = None
    except (AttributeError, TypeError):
        callees_meta = None
        try:
            callees_raw = storage.get_callees_with_confidence(node.id)
        except (AttributeError, TypeError):
            callees_raw = [(c, 1.0) for c in storage.get_callees(node.id)]

    if callees_meta is not None:
        if callees_meta:
            lines.append(f'\nCallees ({len(callees_meta)}):')
        for c, conf, meta in callees_meta:
            tag = _confidence_tag(conf)
            dispatch = meta.get('dispatch_kind', 'direct')
            extra_tags = ''
            if dispatch != 'direct':
                extra_tags += f'  [{dispatch}]'
            if meta.get('awaited'):
                extra_tags += '  [awaited]'
            if meta.get('in_try'):
                extra_tags += '  [in_try]'
            lines.append(
                f'  -> {c.name}  {c.file_path}:{c.start_line}{extra_tags}{tag}'
            )
    elif callees_raw:
        lines.append(f'\nCallees ({len(callees_raw)}):')
        for c, conf in callees_raw:
            tag = _confidence_tag(conf)
            lines.append(f"  -> {c.name}  {c.file_path}:{c.start_line}{tag}")

    type_refs = storage.get_type_refs(node.id)
    if type_refs:
        lines.append(f"\nType references ({len(type_refs)}):")
        for t in type_refs:
            lines.append(f"  -> {t.name}  {t.file_path}")

    escaped_id = _escape_cypher(node.id)
    heritage_rows = storage.execute_raw(
        f"MATCH (n)-[r:CodeRelation]->(parent) "
        f"WHERE n.id = '{escaped_id}' "
        f"AND r.rel_type IN ['extends', 'implements'] "
        f"RETURN parent.name, parent.file_path, r.rel_type"
    ) or []
    if heritage_rows:
        lines.append(f"\nHeritage ({len(heritage_rows)}):")
        for row in heritage_rows:
            parent_name = row[0] or "?"
            parent_file = row[1] or "?"
            rel = row[2] or "?"
            lines.append(f"  -> {rel}: {parent_name}  {parent_file}")

    if node.file_path:
        escaped_fp = _escape_cypher(node.file_path)
        import_rows = storage.execute_raw(
            f"MATCH (a:File)-[r:CodeRelation]->(b:File) "
            f"WHERE b.file_path = '{escaped_fp}' "
            f"AND r.rel_type = 'imports' "
            f"RETURN a.file_path ORDER BY a.file_path"
        ) or []
        if import_rows:
            importers = [r[0] for r in import_rows if r[0]]
            lines.append(f'\nImported by ({len(importers)}):')
            for imp in importers:
                lines.append(f'  -> {imp}')

    lines.append('')
    lines.append('Next: Use impact() if planning changes to this symbol.')
    return '\n'.join(lines)


_DISPATCH_KINDS: frozenset[str] = frozenset(
    {
        'direct',
        'thread_executor',
        'process_executor',
        'detached_task',
        'enqueued_job',
        'callback_registry',
    }
)

_ASYNC_DISPATCH_KINDS: frozenset[str] = frozenset(
    {
        'thread_executor',
        'process_executor',
        'detached_task',
        'enqueued_job',
        'callback_registry',
    }
)

_DEPTH_LABELS: dict[int, str] = {
    1: 'Direct callers (will break)',
    2: 'Indirect (may break)',
}


def _bfs_with_dispatch_filter(
    storage: StorageBackend,
    start_id: str,
    depth: int,
    direction: str,
    dispatch_filter: set[str] | None,
) -> list[tuple[Any, int, str]]:
    """BFS through CALLS edges, optionally filtered by edge dispatch_kind.

    When ``dispatch_filter`` is None, follow every edge (equivalent to
    traversal with no filter). When set, follow an edge only when the
    edge's ``metadata_json.dispatch_kind`` is in the filter set. The edge's
    effective dispatch_kind defaults to ``"direct"`` when the metadata has
    no explicit key (sparse-encoding default).

    Args:
        storage: The storage backend.
        start_id: Node ID to start BFS from.
        depth: Maximum traversal depth (clamped to MAX_TRAVERSE_DEPTH).
        direction: ``"callers"`` or ``"callees"``.
        dispatch_filter: Set of dispatch_kind values to follow, or None to
            follow all edges.

    Returns:
        List of (node, hop_depth, edge_dispatch_kind) tuples. The third
        element is the dispatch_kind of the edge by which the node was
        first reached.
    """
    depth = max(1, min(depth, MAX_TRAVERSE_DEPTH))
    visited: set[str] = set()
    result: list[tuple[Any, int, str]] = []
    queue: deque[tuple[str, int]] = deque([(start_id, 0)])
    visited.add(start_id)

    while queue:
        current_id, current_depth = queue.popleft()
        if current_depth >= depth:
            continue

        try:
            edges = (
                storage.get_callers_with_metadata(current_id)
                if direction == 'callers'
                else storage.get_callees_with_metadata(current_id)
            )
        except (AttributeError, TypeError):
            edges = []

        for node, _confidence, meta in edges:
            if node.id in visited:
                continue
            edge_kind = meta.get('dispatch_kind', 'direct')
            if (
                dispatch_filter is not None
                and edge_kind not in dispatch_filter
            ):
                continue
            visited.add(node.id)
            result.append((node, current_depth + 1, edge_kind))
            queue.append((node.id, current_depth + 1))

    return result


def handle_impact(
    storage: StorageBackend,
    symbol: str,
    depth: int = 3,
    propagate_through: list[str] | None = None,
) -> str:
    """Analyse the blast radius of changing a symbol, grouped by hop depth.

    Uses BFS traversal through CALLS edges to find all affected symbols
    up to the specified depth, then groups results by distance.

    When ``propagate_through`` is set, only CALLS edges whose
    ``dispatch_kind`` is in that set are followed. Unknown kinds are
    accepted silently (logged at DEBUG level) but have no effect if they
    match no edges.

    Args:
        storage: The storage backend.
        symbol: Plain name (``foo``) or dotted path (``Class.method``).
            Multi-dot paths (e.g. ``module.Class.method``) fall back to
            the last segment. When multiple symbols match (e.g., a class
            named ``Foo`` exists in two different files), the best match
            is chosen by score (source files over tests) then lexicographic
            node ID.
        depth: Maximum traversal depth (default 3).
        propagate_through: Optional list of dispatch_kind values to follow.
            When None, all edges are traversed (default behavior).

    Returns:
        Formatted impact analysis with depth-grouped sections.
    """
    if not symbol or not symbol.strip():
        return "Error: 'symbol' parameter is required and cannot be empty."

    depth = max(1, min(depth, MAX_TRAVERSE_DEPTH))

    results = _resolve_symbol(storage, symbol)
    if not results:
        return f"Symbol '{symbol}' not found."

    start_node = storage.get_node(results[0].node_id)
    if not start_node:
        return f"Symbol '{symbol}' not found."

    label_display = start_node.label.value.title()

    if propagate_through is not None:
        unknown = set(propagate_through) - _DISPATCH_KINDS
        if unknown:
            logger.debug(
                'handle_impact: unknown dispatch_kind values in propagate_through: %s',
                unknown,
            )
        dispatch_filter = set(propagate_through)
        reached = _bfs_with_dispatch_filter(
            storage, start_node.id, depth, 'callers', dispatch_filter
        )
        if not reached:
            return (
                f'No upstream callers via propagate_through='
                f"{sorted(dispatch_filter)} for '{symbol}'."
            )

        by_depth: dict[int, list] = {}
        for node, d, _kind in reached:
            by_depth.setdefault(d, []).append(node)

        total = len(reached)
        lines = [f'Impact analysis for: {start_node.name} ({label_display})']
        lines.append(
            f'Depth: {depth} | Total: {total} symbols '
            f'| filter: {sorted(dispatch_filter)}'
        )

        counter = 1
        for d in sorted(by_depth.keys()):
            depth_label = _DEPTH_LABELS.get(d, 'Transitive (review)')
            lines.append(f'\nDepth {d} - {depth_label}:')
            for node in by_depth[d]:
                label = node.label.value.title() if node.label else 'Unknown'
                lines.append(
                    f'  {counter}. {node.name} ({label}) -- '
                    f'{node.file_path}:{node.start_line}'
                )
                counter += 1

        lines.append('')
        lines.append('Tip: Review each affected symbol before making changes.')
        return '\n'.join(lines)

    affected_with_depth = storage.traverse_with_depth(
        start_node.id, depth, direction='callers'
    )
    if not affected_with_depth:
        return f"No upstream callers found for '{symbol}'."

    by_depth_plain: dict[int, list] = {}
    for node, d in affected_with_depth:
        by_depth_plain.setdefault(d, []).append(node)

    total = len(affected_with_depth)
    lines = [f'Impact analysis for: {start_node.name} ({label_display})']
    lines.append(f'Depth: {depth} | Total: {total} symbols')

    conf_lookup = {
        node.id: conf for node, conf in storage.get_callers_with_confidence(start_node.id)
    }

    counter = 1
    for d in sorted(by_depth_plain.keys()):
        depth_label = _DEPTH_LABELS.get(d, 'Transitive (review)')
        lines.append(f'\nDepth {d} - {depth_label}:')
        for node in by_depth_plain[d]:
            label = node.label.value.title() if node.label else 'Unknown'
            conf = conf_lookup.get(node.id)
            tag = f'  (confidence: {conf:.2f})' if conf is not None else ''
            lines.append(
                f'  {counter}. {node.name} ({label}) -- '
                f'{node.file_path}:{node.start_line}{tag}'
            )
            counter += 1

    lines.append("")
    lines.append("Tip: Review each affected symbol before making changes.")
    return "\n".join(lines)


def handle_concurrent_with(
    storage: StorageBackend, symbol: str, depth: int = 3
) -> str:
    """Find symbols that may run concurrently with the given symbol.

    Traces outgoing CALLS edges whose dispatch_kind is any non-direct value
    (thread_executor, process_executor, detached_task, enqueued_job,
    callback_registry) and reports the set of reachable symbols grouped by
    dispatch kind.

    Args:
        storage: The storage backend.
        symbol: Plain name or dotted path. Resolution follows the same
            dotted-path rules as handle_context and handle_impact.
        depth: Maximum traversal depth (default 3).

    Returns:
        Formatted list of concurrent-reachable symbols grouped by dispatch kind.
    """
    if not symbol or not symbol.strip():
        return "Error: 'symbol' parameter is required and cannot be empty."

    depth = max(1, min(depth, MAX_TRAVERSE_DEPTH))

    results = _resolve_symbol(storage, symbol)
    if not results:
        return f"Symbol '{symbol}' not found."

    start_node = storage.get_node(results[0].node_id)
    if not start_node:
        return f"Symbol '{symbol}' not found."

    reached = _bfs_with_dispatch_filter(
        storage,
        start_node.id,
        depth,
        'callees',
        dispatch_filter=set(_ASYNC_DISPATCH_KINDS),
    )

    if not reached:
        label_display = start_node.label.value.title()
        return (
            f'No concurrently-dispatched callees found for '
            f"'{start_node.name}' ({label_display}) within depth {depth}."
        )

    by_kind: dict[str, list[tuple[Any, int]]] = {}
    for node, hop_depth, edge_kind in reached:
        by_kind.setdefault(edge_kind, []).append((node, hop_depth))

    label_display = start_node.label.value.title()
    lines = [
        f'Concurrent callees of: {start_node.name} ({label_display})',
        f'Depth: {depth} | Total: {len(reached)} symbols',
    ]

    for kind in sorted(by_kind.keys()):
        nodes_at_kind = sorted(by_kind[kind], key=lambda t: t[0].name)
        lines.append(f'\n[{kind}] ({len(nodes_at_kind)} symbol(s)):')
        for node, hop_depth in nodes_at_kind:
            node_label = node.label.value.title() if node.label else 'Unknown'
            lines.append(
                f'  -> {node.name} ({node_label})'
                f'  {node.file_path}:{node.start_line}'
                f'  (depth {hop_depth})'
            )

    return '\n'.join(lines)


def handle_dead_code(storage: StorageBackend) -> str:
    """List all symbols marked as dead code.

    Delegates to :func:`~axon.mcp.resources.get_dead_code_list` for the
    shared query and formatting.

    Args:
        storage: The storage backend.

    Returns:
        Formatted list of dead code symbols grouped by file.
    """
    return get_dead_code_list(storage)

_DIFF_FILE_PATTERN = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)
_DIFF_HUNK_PATTERN = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


def _parse_diff_files(diff: str) -> dict[str, list[tuple[int, int]]]:
    """Parse a git diff and return {file_path: [(start, end), ...]}."""
    changed_files: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None

    for line in diff.split("\n"):
        file_match = _DIFF_FILE_PATTERN.match(line)
        if file_match:
            current_file = file_match.group(2)
            if current_file not in changed_files:
                changed_files[current_file] = []
            continue

        hunk_match = _DIFF_HUNK_PATTERN.match(line)
        if hunk_match and current_file is not None:
            start = int(hunk_match.group(1))
            count = max(1, int(hunk_match.group(2) or "1"))
            changed_files[current_file].append((start, start + count - 1))

    return changed_files


def handle_detect_changes(storage: StorageBackend, diff: str) -> str:
    """Map git diff output to affected symbols.

    Parses the diff to find changed files and line ranges, then queries
    the storage backend to identify which symbols those lines belong to.

    Args:
        storage: The storage backend.
        diff: Raw git diff output string.

    Returns:
        Formatted list of affected symbols per changed file.
    """
    if not diff.strip():
        return 'Empty diff provided.'

    if len(diff) > MAX_DIFF_LENGTH:
        return (
            f'Diff rejected: exceeds maximum length of {MAX_DIFF_LENGTH:,} '
            f'characters (got {len(diff):,}).'
        )

    changed_files = _parse_diff_files(diff)

    if not changed_files:
        return "Could not parse any changed files from the diff."

    lines = [f"Changed files: {len(changed_files)}"]
    lines.append("")
    total_affected = 0

    for file_path, ranges in changed_files.items():
        affected_symbols = []
        if not _SAFE_PATH.match(file_path):
            logger.warning("Skipping unsafe file path in diff: %r", file_path)
            lines.append(f"  {file_path}:")
            lines.append("    (skipped: path contains unsafe characters)")
            lines.append("")
            continue

        rows = storage.execute_raw(
            f"MATCH (n) WHERE n.file_path = '{_escape_cypher(file_path)}' "
            f"AND n.start_line > 0 "
            f"RETURN n.id, n.name, n.file_path, n.start_line, n.end_line"
        ) or []
        for row in rows:
            node_id = row[0] or ""
            name = row[1] or ""
            start_line = row[3] or 0
            end_line = row[4] or 0
            label_prefix = node_id.split(":", 1)[0] if node_id else ""
            for start, end in ranges:
                if start_line <= end and end_line >= start:
                    affected_symbols.append((name, label_prefix.title(), start_line, end_line))
                    break

        lines.append(f"  {file_path}:")
        if affected_symbols:
            for sym_name, label, s_line, e_line in affected_symbols:
                lines.append(f"    - {sym_name} ({label}) lines {s_line}-{e_line}")
                total_affected += 1
        else:
            lines.append("    (no indexed symbols in changed lines)")
        lines.append("")

    lines.append(f"Total affected symbols: {total_affected}")
    lines.append("")
    lines.append("Next: Use impact() on affected symbols to see downstream effects.")
    return "\n".join(lines)


def handle_cypher(storage: StorageBackend, query: str) -> str:
    """Execute a raw Cypher query and return formatted results.

    Only read-only queries are allowed. Queries containing write keywords
    (DELETE, DROP, CREATE, SET, etc.) are rejected.

    The caller is expected to pass a storage backend opened in read-only
    mode (see server._with_readonly_storage). The WRITE_KEYWORDS regex
    check here remains as defense-in-depth and to return a friendlier
    error message than a raw KuzuDB read-only violation.

    Args:
        storage: The storage backend, expected to be opened read-only.
        query: The Cypher query string.

    Returns:
        Formatted query results, or an error message if execution fails.
    """
    if len(query) > MAX_CYPHER_LENGTH:
        return (
            f'Query rejected: exceeds maximum length of {MAX_CYPHER_LENGTH:,} '
            f'characters (got {len(query):,}).'
        )

    # Strip comments so write keywords hidden inside comment blocks are detected.
    cleaned_query = sanitize_cypher(query)
    if WRITE_KEYWORDS.search(cleaned_query):
        return (
            "Query rejected: only read-only queries (MATCH/RETURN) are allowed. "
            "Write operations (DELETE, DROP, CREATE, SET, MERGE) are not permitted."
        )

    try:
        rows = storage.execute_raw(query)
    except Exception:
        ref = _new_ref_id()
        logger.exception(
            'handle_cypher execute_raw failed', extra={'ref': ref}
        )
        return f'Cypher query failed (ref {ref}); see server logs.'

    if not rows:
        return "Query returned no results."

    lines = [f"Results ({len(rows)} rows):"]
    lines.append("")
    for i, row in enumerate(rows, 1):
        formatted_values = [str(v) for v in row]
        lines.append(f"  {i}. {' | '.join(formatted_values)}")

    return "\n".join(lines)


def handle_coupling(
    storage: StorageBackend, file_path: str, min_strength: float = 0.3
) -> str:
    """Query temporal coupling for a file and flag hidden dependencies."""
    if not file_path or not file_path.strip():
        return "Error: 'file_path' parameter is required and cannot be empty."

    file_path = file_path.strip()
    if not _SAFE_PATH.match(file_path):
        return "Error: file path contains unsafe characters."

    escaped = _escape_cypher(file_path)
    rows = (
        storage.execute_raw(
            f'MATCH (a:File)-[r:CodeRelation]-(b:File) '
            f"WHERE a.file_path = '{escaped}' AND r.rel_type = 'coupled_with' "
            f'RETURN b.file_path, r.strength, r.co_changes '
            f'ORDER BY r.strength DESC'
        )
        or []
    )

    rows = [r for r in rows if (r[1] or 0) >= min_strength]

    if not rows:
        return f"No temporal coupling found for '{file_path}' (min strength: {min_strength})."

    import_rows = storage.execute_raw(
        f"MATCH (a:File)-[r:CodeRelation]->(b:File) "
        f"WHERE a.file_path = '{escaped}' AND r.rel_type = 'imports' "
        f"RETURN b.file_path"
    ) or []
    imported_files = {r[0] for r in import_rows}

    lines = [f"Temporal coupling for: {file_path}"]
    lines.append("=" * 48)
    lines.append("")

    for i, row in enumerate(rows, 1):
        coupled_path = row[0] or "?"
        strength = row[1] or 0.0
        co_changes = row[2] or 0
        has_import = coupled_path in imported_files
        import_flag = "imports: yes" if has_import else "imports: no \u26a0\ufe0f"
        lines.append(
            f"  {i}. {coupled_path}  strength: {strength:.2f}  "
            f"co_changes: {co_changes}  ({import_flag})"
        )

    lines.append("")
    hidden = [r[0] for r in rows if r[0] not in imported_files]
    if hidden:
        lines.append(
            f"\u26a0\ufe0f {len(hidden)} file(s) have hidden dependencies (no static import)."
        )
    return "\n".join(lines)


def handle_call_path(
    storage: StorageBackend,
    from_symbol: str,
    to_symbol: str,
    max_depth: int = 10,
) -> str:
    """Find the shortest call chain between two symbols via BFS.

    Args:
        storage: The storage backend.
        from_symbol: Plain name (``foo``) or dotted path (``Class.method``).
            Multi-dot paths (e.g. ``module.Class.method``) fall back to
            the last segment. When multiple symbols match (e.g., a class
            named ``Foo`` exists in two different files), the best match
            is chosen by score (source files over tests) then lexicographic
            node ID.
        to_symbol: Plain name (``foo``) or dotted path (``Class.method``).
            Multi-dot paths (e.g. ``module.Class.method``) fall back to
            the last segment. When multiple symbols match (e.g., a class
            named ``Foo`` exists in two different files), the best match
            is chosen by score (source files over tests) then lexicographic
            node ID.
        max_depth: Maximum BFS depth (default 10).

    Returns:
        Formatted call chain or a message when no path exists.
    """
    if not from_symbol or not from_symbol.strip():
        return (
            "Error: 'from_symbol' parameter is required and cannot be empty."
        )
    if not to_symbol or not to_symbol.strip():
        return "Error: 'to_symbol' parameter is required and cannot be empty."

    max_depth = max(1, min(max_depth, MAX_TRAVERSE_DEPTH))

    from_results = _resolve_symbol(storage, from_symbol)
    if not from_results:
        return f"Source symbol '{from_symbol}' not found."

    to_results = _resolve_symbol(storage, to_symbol)
    if not to_results:
        return f"Target symbol '{to_symbol}' not found."

    src_node = storage.get_node(from_results[0].node_id)
    tgt_node = storage.get_node(to_results[0].node_id)
    if not src_node or not tgt_node:
        return "Could not resolve one or both symbols."

    if src_node.id == tgt_node.id:
        return f"Source and target are the same symbol: {src_node.name}"

    parent: dict[str, str] = {}
    queue: deque[tuple[str, int]] = deque([(src_node.id, 0)])
    visited: set[str] = {src_node.id}

    found = False
    while queue:
        current_id, depth = queue.popleft()
        if depth >= max_depth:
            continue

        for callee in storage.get_callees(current_id):
            if callee.id in visited:
                continue
            visited.add(callee.id)
            parent[callee.id] = current_id

            if callee.id == tgt_node.id:
                found = True
                break

            queue.append((callee.id, depth + 1))

        if found:
            break

    if not found:
        return (
            f"No call path found from '{src_node.name}' to '{tgt_node.name}' "
            f"within {max_depth} hops."
        )

    path_ids: list[str] = []
    node_id = tgt_node.id
    while node_id is not None:
        path_ids.append(node_id)
        node_id = parent.get(node_id)
    path_ids.reverse()

    # Build an edge-metadata index for the hops in this path keyed by
    # (source_id, target_id) so we can annotate each line with dispatch_kind.
    edge_meta_index: dict[tuple[str, str], dict[str, Any]] = {}
    for prev_id, curr_id in zip(path_ids, path_ids[1:]):
        try:
            callees_meta = storage.get_callees_with_metadata(prev_id)
            for callee_node, _conf, meta in callees_meta:
                if callee_node.id == curr_id:
                    edge_meta_index[(prev_id, curr_id)] = meta
                    break
        except (AttributeError, TypeError):
            pass

    hop_count = len(path_ids) - 1
    path_names = []
    lines = []
    for i, nid in enumerate(path_ids, 1):
        node = storage.get_node(nid)
        dispatch_annotation = ''
        if i > 1:
            prev_nid = path_ids[i - 2]
            meta = edge_meta_index.get((prev_nid, nid), {})
            kind = meta.get('dispatch_kind', 'direct')
            if kind != 'direct':
                dispatch_annotation = f'  [{kind}]'
        if node:
            label = node.label.value.title() if node.label else 'Unknown'
            path_names.append(node.name)
            lines.append(
                f'  {i}. {node.name} ({label})'
                f'{dispatch_annotation}'
                f' - {node.file_path}:{node.start_line}'
            )
        else:
            path_names.append(nid)
            lines.append(f'  {i}. {nid}{dispatch_annotation}')

    header = (
        f'Call path: {" -> ".join(path_names)}'
        f' ({hop_count} hop{"s" if hop_count != 1 else ""})'
    )
    return header + '\n\n' + '\n'.join(lines)


def handle_communities(
    storage: StorageBackend, community: str | None = None
) -> str:
    """List communities or drill into a specific one."""
    if community:
        escaped = _escape_cypher(community)
        rows = (
            storage.execute_raw(
                f'MATCH (n)-[r:CodeRelation]->(c:Community) '
                f"WHERE c.name = '{escaped}' AND r.rel_type = 'member_of' "
                f'RETURN n.name, label(n), n.file_path, n.start_line, '
                f'n.is_entry_point, n.is_exported '
                f'ORDER BY n.file_path, n.start_line'
            )
            or []
        )

        if not rows:
            return f"Community '{community}' not found or has no members."

        lines = [f"Community: {community}"]
        lines.append(f"Members ({len(rows)}):")
        lines.append("")
        for row in rows:
            name = row[0] or "?"
            label = row[1] or "Unknown"
            file_path = row[2] or "?"
            start_line = row[3] or 0
            is_entry = row[4] if len(row) > 4 else False
            is_exported = row[5] if len(row) > 5 else False
            tags = []
            if is_entry:
                tags.append("entry point")
            if is_exported:
                tags.append("exported")
            tag_str = f"  [{', '.join(tags)}]" if tags else ""
            lines.append(f"  - {name} ({label}) — {file_path}:{start_line}{tag_str}")

        return "\n".join(lines)

    rows = storage.execute_raw(
        "MATCH (c:Community) "
        "RETURN c.name, c.cohesion, c.properties_json "
        "ORDER BY c.cohesion DESC"
    ) or []

    if not rows:
        return "No communities detected. Run indexing with community detection enabled."

    lines = [f"Communities ({len(rows)} detected):"]
    lines.append("")
    for i, row in enumerate(rows, 1):
        name = row[0] or "?"
        cohesion = row[1] or 0.0
        props_raw = row[2] or "{}"
        try:
            props = json.loads(props_raw) if isinstance(props_raw, str) else props_raw
        except (json.JSONDecodeError, TypeError):
            props = {}
        symbol_count = props.get('symbol_count', '?')
        lines.append(
            f'  {i}. {name}  (cohesion: {cohesion:.2f}, {symbol_count} symbols)'
        )

    cross_procs = (
        storage.execute_raw(
            'MATCH (n)-[r1:CodeRelation]->(p:Process), (n)-[r2:CodeRelation]->(c:Community) '
            "WHERE r1.rel_type = 'step_in_process' AND r2.rel_type = 'member_of' "
            'WITH p.name AS proc, collect(DISTINCT c.name) AS comms '
            'WHERE size(comms) > 1 '
            'RETURN proc, comms'
        )
        or []
    )

    if cross_procs:
        lines.append("")
        lines.append("Cross-community processes:")
        for row in cross_procs:
            proc_name = row[0] or "?"
            comms = row[1] if len(row) > 1 else []
            comm_str = " → ".join(comms) if isinstance(comms, list) else str(comms)
            lines.append(f"  - {proc_name} ({comm_str})")

    return "\n".join(lines)


def handle_explain(storage: StorageBackend, symbol: str) -> str:
    """Produce a narrative explanation of a symbol.

    Args:
        storage: The storage backend.
        symbol: Plain name (``foo``) or dotted path (``Class.method``).
            Multi-dot paths (e.g. ``module.Class.method``) fall back to
            the last segment. When multiple symbols match (e.g., a class
            named ``Foo`` exists in two different files), the best match
            is chosen by score (source files over tests) then lexicographic
            node ID.

    Returns:
        Narrative explanation of the symbol's role and relationships.
    """
    if not symbol or not symbol.strip():
        return "Error: 'symbol' parameter is required and cannot be empty."

    results = _resolve_symbol(storage, symbol)
    if not results:
        return f"Symbol '{symbol}' not found."

    node = storage.get_node(results[0].node_id)
    if not node:
        return f"Symbol '{symbol}' not found."

    label_display = node.label.value.title() if node.label else "Unknown"
    lines = [f"Explanation: {node.name} ({label_display})"]
    lines.append("=" * 48)
    lines.append("")

    roles = []
    if node.is_entry_point:
        roles.append("Entry point")
    if node.is_exported:
        roles.append("Exported")
    if node.is_dead:
        roles.append("Dead code (unreachable)")
    if roles:
        lines.append(f"Role: {', '.join(roles)}")

    lines.append(f"Location: {node.file_path}:{node.start_line}-{node.end_line}")

    if node.signature:
        lines.append(f"Signature: {node.signature}")

    escaped_id = _escape_cypher(node.id)
    comm_rows = (
        storage.execute_raw(
            f'MATCH (n)-[r:CodeRelation]->(c:Community) '
            f"WHERE n.id = '{escaped_id}' AND r.rel_type = 'member_of' "
            f'RETURN c.name'
        )
        or []
    )
    if comm_rows:
        comm_name = comm_rows[0][0] or '?'
        lines.append(f'Community: {comm_name}')

    lines.append('')

    callers = storage.get_callers_with_confidence(node.id)
    callees = storage.get_callees_with_confidence(node.id)

    if callers:
        caller_names = ", ".join(c.name for c, _ in callers[:5])
        suffix = f" (+{len(callers) - 5} more)" if len(callers) > 5 else ""
        lines.append(f"Called by {len(callers)}: {caller_names}{suffix}")
    else:
        lines.append("Called by: nothing (root or dead)")

    if callees:
        callee_names = ', '.join(c.name for c, _ in callees[:5])
        suffix = f' (+{len(callees) - 5} more)' if len(callees) > 5 else ''
        lines.append(f'Calls {len(callees)}: {callee_names}{suffix}')
    else:
        lines.append('Calls: nothing (leaf)')

    proc_rows = (
        storage.execute_raw(
            f'MATCH (n)-[r:CodeRelation]->(p:Process) '
            f"WHERE n.id = '{escaped_id}' AND r.rel_type = 'step_in_process' "
            f'RETURN p.name'
        )
        or []
    )
    if proc_rows:
        lines.append('')
        lines.append('Process flows through this symbol:')
        for row in proc_rows:
            proc_name = row[0] or "?"
            lines.append(f"  - {proc_name}")

    return "\n".join(lines)


def handle_review_risk(storage: StorageBackend, diff: str) -> str:
    """Assess PR risk by synthesizing multiple graph signals."""
    if not diff.strip():
        return 'Empty diff provided.'

    if len(diff) > MAX_DIFF_LENGTH:
        return (
            f'Diff rejected: exceeds maximum length of {MAX_DIFF_LENGTH:,} '
            f'characters (got {len(diff):,}).'
        )

    changed_files = _parse_diff_files(diff)
    if not changed_files:
        return "Could not parse any changed files from the diff."

    changed_file_set = set(changed_files.keys())
    all_affected_symbols: list[tuple[str, str, str, int]] = []
    entry_points_hit = 0
    total_dependents = 0

    for file_path, ranges in changed_files.items():
        if not _SAFE_PATH.match(file_path):
            continue
        escaped = _escape_cypher(file_path)
        rows = storage.execute_raw(
            f"MATCH (n) WHERE n.file_path = '{escaped}' "
            f"AND n.start_line > 0 "
            f"RETURN n.id, n.name, n.file_path, n.start_line, n.end_line"
        ) or []

        for row in rows:
            node_id = row[0] or ""
            name = row[1] or ""
            start_line = row[3] or 0
            end_line = row[4] or 0
            label_prefix = node_id.split(":", 1)[0].title() if node_id else ""

            hit = any(start_line <= end and end_line >= start for start, end in ranges)
            if not hit:
                continue

            node = storage.get_node(node_id)
            dep_count = 0
            if node:
                deps = storage.traverse_with_depth(node.id, 2, direction="callers")
                dep_count = len(deps)
                if node.is_entry_point:
                    entry_points_hit += 1

            total_dependents += dep_count
            all_affected_symbols.append((name, label_prefix, file_path, dep_count))

    missing_cochange: list[tuple[str, str, float]] = []
    for file_path in changed_files:
        if not _SAFE_PATH.match(file_path):
            continue
        escaped = _escape_cypher(file_path)
        coupling_rows = (
            storage.execute_raw(
                f'MATCH (a:File)-[r:CodeRelation]-(b:File) '
                f"WHERE a.file_path = '{escaped}' "
                f"AND r.rel_type = 'coupled_with' AND r.strength >= 0.5 "
                f'RETURN b.file_path, r.strength'
            )
            or []
        )
        for row in coupling_rows:
            coupled_file = row[0] or ''
            strength = row[1] or 0.0
            if coupled_file not in changed_file_set:
                missing_cochange.append((coupled_file, file_path, strength))

    communities_touched: set[str] = set()
    for name, label, file_path, _ in all_affected_symbols:
        escaped = _escape_cypher(f'{label.lower()}:{file_path}:{name}')
        comm_rows = (
            storage.execute_raw(
                f'MATCH (n)-[r:CodeRelation]->(c:Community) '
                f"WHERE n.id = '{escaped}' AND r.rel_type = 'member_of' RETURN c.name"
            )
            or []
        )
        for row in comm_rows:
            if row[0]:
                communities_touched.add(row[0])

    score = entry_points_hit + len(missing_cochange) + total_dependents // 10
    if len(communities_touched) > 1:
        score += 2
    score = min(score, 10)

    if score <= 3:
        level = "LOW"
    elif score <= 6:
        level = "MEDIUM"
    else:
        level = "HIGH"

    lines = ["PR Risk Assessment"]
    lines.append("=" * 48)
    lines.append(f"Risk: {level} (score: {score}/10)")
    lines.append("")

    if all_affected_symbols:
        lines.append(f"Changed symbols ({len(all_affected_symbols)}):")
        for name, label, fp, deps in all_affected_symbols:
            tags = []
            if deps > 0:
                tags.append(f"{deps} downstream dependents")
            tag_str = f"  [{', '.join(tags)}]" if tags else ""
            lines.append(f"  - {name} ({label}) — {fp}{tag_str}")
    else:
        lines.append("No indexed symbols in changed lines.")

    if missing_cochange:
        lines.append("")
        lines.append("⚠️ Missing co-change files (usually change together):")
        for missing, coupled_with, strength in missing_cochange:
            lines.append(f"  - {missing} (strength: {strength:.2f} with {coupled_with})")

    if len(communities_touched) > 1:
        lines.append("")
        lines.append(f"Community boundary crossings: {len(communities_touched)}")
        lines.append(f"  Spans: {', '.join(sorted(communities_touched))}")

    return "\n".join(lines)


def handle_file_context(storage: StorageBackend, file_path: str) -> str:
    """Provide comprehensive context for a single file."""
    if not file_path or not file_path.strip():
        return "Error: 'file_path' parameter is required and cannot be empty."

    file_path = file_path.strip()
    if not _SAFE_PATH.match(file_path):
        return "Error: file path contains unsafe characters."

    escaped = _escape_cypher(file_path)

    sym_rows = (
        storage.execute_raw(
            f"MATCH (n) WHERE n.file_path = '{escaped}' AND n.start_line > 0 "
            f'RETURN n.name, label(n), n.start_line, n.is_dead, n.is_entry_point, n.is_exported '
            f'ORDER BY n.start_line'
        )
        or []
    )

    imports_out = (
        storage.execute_raw(
            f'MATCH (a:File)-[r:CodeRelation]->(b:File) '
            f"WHERE a.file_path = '{escaped}' AND r.rel_type = 'imports' "
            f'RETURN b.file_path ORDER BY b.file_path'
        )
        or []
    )

    imports_in = (
        storage.execute_raw(
            f'MATCH (a:File)-[r:CodeRelation]->(b:File) '
            f"WHERE b.file_path = '{escaped}' AND r.rel_type = 'imports' "
            f'RETURN a.file_path ORDER BY a.file_path'
        )
        or []
    )

    coupling_rows = (
        storage.execute_raw(
            f'MATCH (a:File)-[r:CodeRelation]-(b:File) '
            f"WHERE a.file_path = '{escaped}' AND r.rel_type = 'coupled_with' "
            f'RETURN b.file_path, r.strength, r.co_changes '
            f'ORDER BY r.strength DESC LIMIT 5'
        )
        or []
    )

    dead_rows = (
        storage.execute_raw(
            f"MATCH (n) WHERE n.is_dead = true AND n.file_path = '{escaped}' "
            f'RETURN n.name, n.start_line, label(n)'
        )
        or []
    )

    comm_rows = (
        storage.execute_raw(
            f'MATCH (n)-[r:CodeRelation]->(c:Community) '
            f"WHERE n.file_path = '{escaped}' AND r.rel_type = 'member_of' "
            f'RETURN c.name, count(n) ORDER BY count(n) DESC'
        )
        or []
    )

    if not sym_rows and not imports_out and not imports_in:
        return f"No data found for file '{file_path}'. Is it indexed?"

    lines = [f"File: {file_path}"]
    lines.append("=" * 48)

    if sym_rows:
        lines.append("")
        lines.append(f"Symbols ({len(sym_rows)}):")
        for row in sym_rows:
            name = row[0] or "?"
            label = row[1] or "Unknown"
            start_line = row[2] or 0
            is_dead = row[3] if len(row) > 3 else False
            is_entry = row[4] if len(row) > 4 else False
            is_exported = row[5] if len(row) > 5 else False
            tags = []
            if is_entry:
                tags.append("entry point")
            if is_exported:
                tags.append("exported")
            if is_dead:
                tags.append("dead")
            tag_str = f"  [{', '.join(tags)}]" if tags else ""
            lines.append(f"  - {name} ({label}) line {start_line}{tag_str}")

    if imports_out:
        out_paths = [r[0] for r in imports_out if r[0]]
        lines.append("")
        lines.append(f"Imports ({len(out_paths)}): {', '.join(out_paths)}")

    if imports_in:
        in_paths = [r[0] for r in imports_in if r[0]]
        lines.append(f"Imported by ({len(in_paths)}): {', '.join(in_paths)}")

    if coupling_rows:
        lines.append("")
        lines.append(f"Coupled files ({len(coupling_rows)}):")
        for row in coupling_rows:
            coupled_path = row[0] or "?"
            strength = row[1] or 0.0
            co_changes = row[2] or 0
            lines.append(f"  - {coupled_path}  strength: {strength:.2f}  co_changes: {co_changes}")

    if dead_rows:
        lines.append("")
        lines.append(f"Dead code ({len(dead_rows)}):")
        for row in dead_rows:
            name = row[0] or "?"
            start_line = row[1] or 0
            label = row[2] or "Unknown"
            lines.append(f"  - {name} ({label}) line {start_line}")

    if comm_rows:
        lines.append("")
        comm_parts = [f"{r[0]} ({r[1]} symbols)" for r in comm_rows if r[0]]
        lines.append(f"Communities: {', '.join(comm_parts)}")

    return "\n".join(lines)


def handle_cycles(storage: StorageBackend, min_size: int = 2) -> str:
    """Detect circular dependencies using strongly connected components."""
    min_size = max(2, min_size)

    try:
        graph = storage.load_graph()
    except Exception:
        ref = _new_ref_id()
        logger.exception('handle_cycles load_graph failed', extra={'ref': ref})
        return f'Error loading graph (ref {ref}); see server logs.'

    ig_graph, index_to_node_id = export_to_igraph(graph)

    if ig_graph.vcount() == 0:
        return "No symbols in the graph to analyze."

    sccs = ig_graph.connected_components(mode="strong")

    cycles = [
        list(component)
        for component in sccs
        if len(component) >= min_size
    ]

    if not cycles:
        return "No circular dependencies detected."

    cycles.sort(key=len, reverse=True)

    lines = [f"Circular Dependencies ({len(cycles)} groups)"]
    lines.append("=" * 48)

    for i, component in enumerate(cycles, 1):
        node_ids = [index_to_node_id[idx] for idx in component if idx in index_to_node_id]
        nodes = [graph.get_node(nid) for nid in node_ids]
        nodes = [n for n in nodes if n is not None]

        severity = "CRITICAL" if len(nodes) >= 5 else ""
        size_label = f" — {severity}" if severity else ""
        lines.append(f"\nCycle {i} ({len(nodes)} symbols){size_label}:")
        for node in nodes:
            label = node.label.value.title() if node.label else "Unknown"
            lines.append(
                f"  - {node.name} ({label}) — "
                f"{node.file_path}:{node.start_line}"
            )

    return "\n".join(lines)


def _build_warnings(
    non_executable_files: list[str], config_excluded_test_files: list[str]
) -> str:
    """Build the optional Warnings section for test impact output.

    Returns an empty string when both lists are empty so the caller can
    skip the section entirely (keeping existing output identical).

    Args:
        non_executable_files: Files skipped because all hunks are
            comments or docstrings.
        config_excluded_test_files: Test files excluded by pytest config
            (testpaths, norecursedirs, or collect_ignore).

    Returns:
        Formatted warnings block, or empty string.
    """
    if not non_executable_files and not config_excluded_test_files:
        return ''

    parts: list[str] = ['Warnings:']

    if non_executable_files:
        parts.append('  Ignored (docstring/comment-only changes):')
        for f in sorted(set(non_executable_files)):
            parts.append(f'    - {f}')

    if config_excluded_test_files:
        parts.append(
            '  Excluded by pytest config (testpaths/norecursedirs/collect_ignore):'
        )
        for f in sorted(set(config_excluded_test_files)):
            parts.append(f'    - {f}')

    return '\n'.join(parts)


_HUNK_EXECUTABLE_LANGUAGES: frozenset[str] = frozenset(
    {'python', 'typescript', 'tsx', 'javascript'}
)

_EXT_TO_LANGUAGE: dict[str, str] = {
    '.py': 'python',
    '.ts': 'typescript',
    '.tsx': 'tsx',
    '.js': 'javascript',
    '.jsx': 'javascript',
}

_STATEMENT_CONTAINERS: frozenset[str] = frozenset(
    {
        'module',
        'block',
        'function_definition',
        'class_definition',
        'method_definition',
    }
)


def _is_non_executable_statement(node: Any) -> bool:
    """Return True when *node* contains no executable code.

    Handles:
    - comment nodes (Python # comments, JS // and block comments).
    - expression_statement whose sole child is a string node (docstring).

    Args:
        node: A tree-sitter Node.

    Returns:
        True when the node is entirely non-executable.
    """
    if node.type == 'comment':
        return True
    # Docstring pattern: expression_statement -> single string child.
    if (
        node.type == 'expression_statement'
        and len(node.children) == 1
        and node.children[0].type == 'string'
    ):
        return True
    return False


def _is_container_node(node: Any) -> bool:
    """Return True when *node* is a structural container for other statements.

    Container nodes are recursed into rather than collected as leaf statements.
    A compound statement (e.g. if_statement) that contains a block child is
    also treated as a container so the inner statements are inspected directly.

    Args:
        node: A tree-sitter Node.

    Returns:
        True when the node should be recursed into.
    """
    if node.type in _STATEMENT_CONTAINERS:
        return True
    # Compound statements that wrap a block (if, for, while, try, with, ...)
    # should be transparent so inner leaf statements are collected directly.
    return any(c.type == 'block' for c in node.children)


def _collect_statement_nodes_for_row(
    node: Any, row: int, result: list[Any]
) -> None:
    """Collect leaf statement-level nodes that overlap *row* into *result*.

    Recurses through container nodes (module, block, function/class definitions,
    compound statements) and collects the innermost statement-level or comment
    node that covers the row.

    Args:
        node: Current tree-sitter Node being visited.
        row: 0-indexed row number to match.
        result: Accumulator list; matching nodes are appended here.
    """
    if node.start_point[0] > row or node.end_point[0] < row:
        return
    if node.type == 'comment':
        result.append(node)
        return
    if _is_container_node(node):
        for child in node.children:
            _collect_statement_nodes_for_row(child, row, result)
        return
    # Leaf statement node (expression_statement, return_statement, etc.)
    if node.type.endswith('_statement'):
        result.append(node)
        return
    # Recurse into anything else that might contain statements.
    for child in node.children:
        _collect_statement_nodes_for_row(child, row, result)


def _is_hunk_executable(
    file_content: str, hunk_ranges: list[tuple[int, int]], language: str
) -> bool:
    """Return False when every changed line is a comment, docstring, or blank.

    Uses tree-sitter to walk the AST and inspect node types. Hunk ranges
    are 1-indexed (as in git diff output); they are converted to 0-indexed
    row numbers internally.

    Only ``python``, ``typescript``, ``tsx``, and ``javascript`` are
    inspected. Any other language returns True (conservative - treat as
    executable). On any parser error also returns True.

    Args:
        file_content: Full source code of the file.
        hunk_ranges: List of (start_line, end_line) ranges, 1-indexed inclusive.
        language: Source language identifier.

    Returns:
        True if any changed line contains executable code, False if every
        changed line is a comment, docstring, or blank.
    """
    if language not in _HUNK_EXECUTABLE_LANGUAGES:
        return True

    if not hunk_ranges:
        return False

    try:
        lang_parser = get_parser(language)
    except (ValueError, KeyError):
        return True

    try:
        # Access the underlying tree-sitter Parser stored as _parser.
        ts_parser = lang_parser._parser  # type: ignore[attr-defined]
        tree = ts_parser.parse(file_content.encode('utf-8'))
    except Exception:
        return True

    lines = file_content.splitlines()

    # Convert 1-indexed ranges to 0-indexed row sets.
    changed_rows: set[int] = set()
    for start, end in hunk_ranges:
        for row in range(start - 1, end):
            changed_rows.add(row)

    for row in changed_rows:
        # Blank lines are non-executable; skip them.
        if row < len(lines) and not lines[row].strip():
            continue

        # Collect all statement-level nodes overlapping this row.
        overlapping: list[Any] = []
        _collect_statement_nodes_for_row(tree.root_node, row, overlapping)

        # If no statement node covers the row, treat conservatively as
        # executable (e.g. parser error recovery nodes).
        if not overlapping:
            return True

        # Row is executable if any overlapping statement is executable.
        if not all(_is_non_executable_statement(n) for n in overlapping):
            return True

    return False


def handle_test_impact(
    storage: StorageBackend,
    diff: str = '',
    symbols: list[str] | None = None,
    repo_path: Path | None = None,
) -> str:
    """Find tests likely affected by code changes.

    When *repo_path* is provided, the function also:
    - loads pytest config to narrow test-file classification,
    - skips changed files whose hunks are entirely docstrings or comments.

    When *repo_path* is None the function operates in config-unaware mode:
    default heuristic only, no hunk-executable filtering.

    Args:
        storage: Storage backend for graph queries.
        diff: Raw git diff output.
        symbols: List of symbol names to check instead of a diff. Each
            item can be a plain name (``foo``) or a dotted path
            (``Class.method``). Multi-dot paths (e.g.
            ``module.Class.method``) fall back to the last segment. When
            multiple symbols match (e.g., a class named ``Foo`` exists in
            two different files), the best match is chosen by score
            (source files over tests) then lexicographic node ID.
        repo_path: Absolute path to the repository root, or None.

    Returns:
        Formatted impact report string.
    """
    changed_symbol_ids: list[tuple[str, str]] = []
    non_executable_files: list[str] = []

    # Load pytest config when the repo path is known.
    pytest_config: PytestConfig | None = None
    if repo_path is not None:
        pytest_config = load_pytest_config(repo_path)

    if diff and diff.strip():
        if len(diff) > MAX_DIFF_LENGTH:
            return (
                f'Diff rejected: exceeds maximum length of {MAX_DIFF_LENGTH:,} '
                f'characters (got {len(diff):,}).'
            )
        changed_files = _parse_diff_files(diff)
        for file_path, ranges in changed_files.items():
            if not _SAFE_PATH.match(file_path):
                continue

            # Skip files whose hunks are entirely non-executable.
            if repo_path is not None:
                ext = Path(file_path).suffix.lower()
                language = _EXT_TO_LANGUAGE.get(ext, '')
                if language in _HUNK_EXECUTABLE_LANGUAGES:
                    abs_path = repo_path / file_path
                    try:
                        file_content = abs_path.read_text(encoding='utf-8')
                    except OSError:
                        file_content = None

                    if file_content is not None and not _is_hunk_executable(
                        file_content, ranges, language
                    ):
                        non_executable_files.append(file_path)
                        continue

            escaped = _escape_cypher(file_path)
            rows = (
                storage.execute_raw(
                    f"MATCH (n) WHERE n.file_path = '{escaped}' "
                    f'AND n.start_line > 0 '
                    f'RETURN n.id, n.name, n.start_line, n.end_line'
                )
                or []
            )
            for row in rows:
                node_id = row[0] or ''
                name = row[1] or ''
                start_line = row[2] or 0
                end_line = row[3] or 0
                hit = any(
                    start_line <= end and end_line >= start
                    for start, end in ranges
                )
                if hit:
                    changed_symbol_ids.append((node_id, name))

    elif symbols:
        for sym_name in symbols:
            results = _resolve_symbol(storage, sym_name)
            if results:
                node = storage.get_node(results[0].node_id)
                if node:
                    changed_symbol_ids.append((node.id, node.name))

    else:
        return "Error: provide either 'diff' or 'symbols' parameter."

    if not changed_symbol_ids:
        if non_executable_files:
            warning_lines = [
                'No changed symbols found.',
                '',
                'Warnings:',
                '  Ignored (docstring/comment-only changes):',
            ]
            for f in sorted(non_executable_files):
                warning_lines.append(f'    - {f}')
            return '\n'.join(warning_lines)
        return 'No changed symbols found.'

    test_hits: dict[str, list[tuple[str, str, int]]] = {}
    config_excluded_test_files: list[str] = []

    for sym_id, sym_name in changed_symbol_ids:
        for caller, depth in storage.traverse_with_depth(
            sym_id, 4, direction='callers'
        ):
            if is_test_file(caller.file_path, pytest_config):
                test_hits.setdefault(caller.file_path, []).append(
                    (caller.name, sym_name, depth)
                )
            elif pytest_config is not None and is_test_file(
                caller.file_path, None
            ):
                # Default heuristic says test, but config excludes it.
                config_excluded_test_files.append(caller.file_path)

    if not test_hits:
        msg = (
            f'No test files found in the call graph of {len(changed_symbol_ids)} '
            f'changed symbol(s). Tests may not directly call these symbols.'
        )
        warnings = _build_warnings(
            non_executable_files, config_excluded_test_files
        )
        if warnings:
            return msg + '\n\n' + warnings
        return msg

    lines = ['Test Impact Analysis']
    lines.append('=' * 48)
    lines.append(f'Changed symbols: {len(changed_symbol_ids)}')
    lines.append('')

    direct_files: dict[str, list[tuple[str, str, int]]] = {}
    transitive_files: dict[str, list[tuple[str, str, int]]] = {}

    for test_file, hits in sorted(test_hits.items()):
        for test_name, source_sym, depth in hits:
            if depth <= 2:
                direct_files.setdefault(test_file, []).append(
                    (test_name, source_sym, depth)
                )
            else:
                transitive_files.setdefault(test_file, []).append(
                    (test_name, source_sym, depth)
                )

    total_tests = sum(len(v) for v in test_hits.values())

    if direct_files:
        lines.append(f'Affected tests ({total_tests}):')
        for test_file, hits in sorted(direct_files.items()):
            lines.append(f'  {test_file}:')
            seen: set[tuple[str, str]] = set()
            for test_name, source_sym, depth in hits:
                key = (test_name, source_sym)
                if key not in seen:
                    seen.add(key)
                    lines.append(f"    - {test_name} (calls: {source_sym})")
        lines.append("")

    if transitive_files:
        lines.append("Tests with indirect coverage (depth 3+):")
        for test_file, hits in sorted(transitive_files.items()):
            lines.append(f"  {test_file}:")
            seen = set()
            for test_name, source_sym, depth in hits:
                key = (test_name, source_sym)
                if key not in seen:
                    seen.add(key)
                    lines.append(
                        f'    - {test_name} (transitive via: {source_sym})'
                    )

    warnings = _build_warnings(
        non_executable_files, config_excluded_test_files
    )
    if warnings:
        lines.append('')
        lines.append(warnings)

    return '\n'.join(lines)
