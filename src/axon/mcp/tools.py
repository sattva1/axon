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
from axon.core.drift import DriftCache, DriftLevel
from axon.core.embeddings.embedder import embed_query
from axon.core.graph.model import GraphNode, NodeLabel
from axon.core.ingestion.community import export_to_igraph
from axon.core.ingestion.parser_phase import get_parser
from axon.core.ingestion.test_classifier import (
    PytestConfig,
    is_test_file,
    load_pytest_config,
)
from axon.core.meta import load_meta
from axon.core.repos import RepoPool, RepoResolver, RepoUnavailable
from axon.core.search.hybrid import hybrid_search
from axon.core.storage.base import SearchResult, StorageBackend
from axon.core.storage.kuzu_backend import escape_cypher as _escape_cypher
from axon.mcp.freshness import (
    render_with_communities_warning,
    render_with_dead_code_warning,
)
from axon.mcp.repo_context import RepoContext
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


def _resolve_symbol(
    storage: StorageBackend, symbol: str, *, top_k: int = 1
) -> list[SearchResult]:
    """Resolve *symbol* to up to *top_k* SearchResults, preferring exact name.

    Args:
        storage: The storage backend to search against.
        symbol: Plain name or dotted path.
        top_k: Maximum number of results to return.

    Returns:
        List of SearchResult, at most *top_k* items.
    """
    if hasattr(storage, 'exact_name_search'):
        results = storage.exact_name_search(symbol, limit=top_k)
        if results:
            return results
    return storage.fts_search(symbol, limit=top_k)


_LOCAL_MATCHES_TOP_K = 5

_MAX_FOREIGN_COUNT_REPOS = 5


def handle_list_repos(
    *,
    resolver: RepoResolver,
    pool: RepoPool,
    drift_cache: DriftCache,
    local_slug: str | None = None,
) -> str:
    """List indexed repositories using the resolver/pool/drift_cache.

    For each entry returned by ``resolver.list_known()`` the function reads
    per-repo stats from ``.axon/meta.json`` via ``load_meta``, probes drift
    via ``drift_cache.get_or_probe``, and checks reachability via ``pool.get``
    for foreign repos.

    Args:
        resolver: Resolver that enumerates known repos.
        pool: Connection pool used to probe foreign-repo reachability.
        drift_cache: Cache of drift reports keyed by repo path.
        local_slug: Slug of the local repo - used for the (LOCAL) marker.
            When None, no entry is marked local.

    Returns:
        Formatted list of indexed repositories with freshness, watcher status,
        and reachability per entry, or a message when the registry is empty.
    """
    entries = resolver.list_known()
    if not entries:
        return (
            'No indexed repositories found. '
            'Run `axon analyze` on a project first.'
        )

    lines = [f'Indexed repositories ({len(entries)}):']
    lines.append('')

    for i, entry in enumerate(entries, 1):
        is_local = entry.is_local or entry.slug == local_slug
        local_marker = ' (LOCAL)' if is_local else ''

        # Stats from .axon/meta.json
        try:
            meta = load_meta(entry.path)
            stats = meta.stats
            files = stats.get('files', '?')
            symbols = stats.get('symbols', '?')
            relationships = stats.get('relationships', '?')
        except Exception:
            files, symbols, relationships = '?', '?', '?'

        # Drift level and watcher status.
        try:
            report = drift_cache.get_or_probe(entry.path)
            freshness = str(report.level)
            watcher = 'alive' if report.watcher_alive else 'dead'
        except Exception:
            freshness = 'unknown'
            watcher = 'unknown'

        # Reachability: local is always reachable; foreign via pool.
        if is_local:
            reachable = 'yes'
        else:
            try:
                pool.get(entry.slug)
                reachable = 'yes'
            except (RepoUnavailable, Exception):
                reachable = 'no'

        lines.append(f'  {i}. {entry.slug}{local_marker}')
        lines.append(f'     Path: {entry.path}')
        lines.append(
            f'     Files: {files}  Symbols: {symbols}'
            f'  Relationships: {relationships}'
        )
        lines.append(
            f'     Freshness: {freshness}'
            f'  Watcher: {watcher}  Reachable: {reachable}'
        )
        lines.append('')

    lines.append(
        'To query a specific repo, pass repo=<slug> to any multi-repo tool'
        ' (e.g. axon_context, axon_query).'
    )
    return '\n'.join(lines)


def _foreign_symbol_matches(
    pool: RepoPool,
    resolver: RepoResolver,
    drift_cache: DriftCache,
    symbol: str,
    *,
    per_repo_limit: int = 3,
    exclude_slug: str | None = None,
) -> list[tuple[str, list[SearchResult]]]:
    """Look up *symbol* in every accessible foreign repo.

    Repos that cannot be opened (``RepoUnavailable``) and repos whose drift
    level is ``STALE_MAJOR`` are skipped silently.  The local repo is
    excluded via *exclude_slug*.

    Args:
        pool: Foreign-repo connection pool.
        resolver: Resolver used to enumerate foreign repos.
        drift_cache: Cache of drift reports keyed by repo path.
        symbol: Symbol name to search for.
        per_repo_limit: Maximum results to return per repo.
        exclude_slug: Slug of the repo to exclude (typically the local repo).

    Returns:
        List of (slug, results) pairs, one per repo that has matches.
        Ordered by resolver.list_foreign() order.
    """
    output: list[tuple[str, list[SearchResult]]] = []
    for entry in resolver.list_foreign():
        if entry.slug == exclude_slug:
            continue
        try:
            report = drift_cache.get_or_probe(entry.path)
        except Exception:
            continue
        if report.level == DriftLevel.STALE_MAJOR:
            continue
        try:
            backend = pool.get(entry.slug)
        except RepoUnavailable:
            continue
        try:
            results = _resolve_symbol(backend, symbol, top_k=per_repo_limit)
        except Exception:
            continue
        if results:
            output.append((entry.slug, results))
    return output


def _foreign_query_hit_counts(
    pool: RepoPool,
    resolver: RepoResolver,
    drift_cache: DriftCache,
    query: str,
    *,
    exclude_slug: str | None = None,
) -> list[tuple[str, int]]:
    """Return FTS hit counts for *query* across foreign repos.

    At most ``_MAX_FOREIGN_COUNT_REPOS`` foreign repos are queried.  Repos
    are selected from ``resolver.list_foreign()`` in their natural order;
    repos beyond the cap, repos that cannot be opened, and STALE_MAJOR
    repos are skipped.  Only repos with >0 hits are included in the result.

    Each foreign repo is probed synchronously.  The caller may wrap this
    in ``asyncio.to_thread`` if the event loop must remain unblocked.

    Args:
        pool: Foreign-repo connection pool.
        resolver: Resolver used to enumerate foreign repos.
        drift_cache: Cache of drift reports keyed by repo path.
        query: FTS query string.
        exclude_slug: Slug of the repo to exclude (typically the local repo).

    Returns:
        List of (slug, hit_count) pairs for repos with >0 hits.
    """
    output: list[tuple[str, int]] = []
    checked = 0
    for entry in resolver.list_foreign():
        if checked >= _MAX_FOREIGN_COUNT_REPOS:
            break
        if entry.slug == exclude_slug:
            continue
        try:
            report = drift_cache.get_or_probe(entry.path)
        except Exception:
            continue
        if report.level == DriftLevel.STALE_MAJOR:
            continue
        try:
            backend = pool.get(entry.slug)
        except RepoUnavailable:
            continue
        checked += 1
        try:
            # Two-pass: check existence cheaply, then get rough count.
            quick = backend.fts_search(query, limit=1)
            if not quick:
                continue
            results_wide = backend.fts_search(query, limit=50)
            count = len(results_wide)
        except Exception:
            continue
        if count > 0:
            output.append((entry.slug, count))
    return output


def _format_alternates(alternates: list[SearchResult]) -> str:
    """Format a list of alternate local symbol matches as a footer section.

    Args:
        alternates: SearchResult items beyond the first (primary) result.

    Returns:
        Formatted multi-line string, or empty string when list is empty.
    """
    if not alternates:
        return ''
    lines = ['', 'Also matches in this repo:']
    for r in alternates:
        lines.append(f'  - {r.node_name} ({r.label}) {r.file_path or "?"}')
    return '\n'.join(lines)


def _format_foreign_matches(
    matches: list[tuple[str, list[SearchResult]]], *, redirect: bool = False
) -> str:
    """Format a cross-repo symbol match list as a footer or redirect section.

    When *redirect* is True the output is styled as a "Symbol not found
    locally - consider these foreign repos" redirect response.

    Args:
        matches: List of (slug, results) pairs from ``_foreign_symbol_matches``.
        redirect: When True, use redirect-style wording instead of footer.

    Returns:
        Formatted multi-line string, or empty string when list is empty.
    """
    if not matches:
        return ''
    if redirect:
        lines = ['Also exists in other repos (pass repo=<slug> to query):']
    else:
        lines = ['', 'Also exists in other repos:']
    for slug, results in matches:
        for r in results:
            lines.append(
                f'  - {slug}: {r.node_name} ({r.label}) {r.file_path or "?"}'
            )
    return '\n'.join(lines)


def _format_query_hit_counts(counts: list[tuple[str, int]]) -> str:
    """Format a cross-repo FTS hit-count footer for axon_query responses.

    Args:
        counts: List of (slug, hit_count) pairs from
            ``_foreign_query_hit_counts``.

    Returns:
        Single-line footer string, or empty string when list is empty.
    """
    if not counts:
        return ''
    parts = [f'{slug} ({n})' for slug, n in counts]
    return f'Hits also exist in: {", ".join(parts)} - pass repo=<slug> to see them.'


def _group_by_process(
    results: list, storage: StorageBackend
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


def handle_query(
    ctx: RepoContext,
    query: str,
    limit: int = 20,
    *,
    foreign_hits: list[tuple[str, int]] | None = None,
) -> str:
    """Execute hybrid search and format results, grouped by execution process.

    Args:
        ctx: Per-call repo context.
        query: Text search query.
        limit: Maximum number of results (default 20, capped at 100).
        foreign_hits: Optional pre-computed hit counts from foreign repos,
            appended as a footer when non-empty.

    Returns:
        Formatted search results grouped by process, with file, name, label,
        and snippet for each result.
    """
    storage = ctx.storage
    limit = max(1, min(limit, 100))

    query_embedding = embed_query(query)
    if query_embedding is None:
        logger.warning(
            'embed_query returned None; falling back to FTS-only search'
        )

    results = hybrid_search(
        query, storage, query_embedding=query_embedding, limit=limit
    )
    if not results:
        base = f"No results found for '{query}'."
        footer = _format_query_hit_counts(foreign_hits or [])
        return base + ('\n' + footer if footer else '')

    groups = _group_by_process(results, storage)
    body = _format_query_results(results, groups)
    footer = _format_query_hit_counts(foreign_hits or [])
    if footer:
        body = body + '\n' + footer
    return body


_MEMBER_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.ENUM_MEMBER,
        NodeLabel.CLASS_ATTRIBUTE,
        NodeLabel.MODULE_CONSTANT,
    }
)


def _render_member_context(node: GraphNode, storage: StorageBackend) -> str:
    """Render a 360-degree view for a member node."""
    if node.label == NodeLabel.ENUM_MEMBER:
        header = (
            f'Enum Member: {node.class_name}.{node.name}'
            f' ({node.file_path}:{node.start_line})'
        )
        parent_line = f'Parent: {node.class_name}'
    elif node.label == NodeLabel.CLASS_ATTRIBUTE:
        header = (
            f'Class Attribute: {node.class_name}.{node.name}'
            f' ({node.file_path}:{node.start_line})'
        )
        parent_line = f'Parent: {node.class_name}'
    else:  # MODULE_CONSTANT
        header = (
            f'Module Constant: {node.name}'
            f' ({node.file_path}:{node.start_line})'
        )
        parent_line = f'Module: {node.file_path}'

    lines = [header, parent_line, '']

    accessors = storage.get_accessors(node.id)
    if not accessors:
        lines.append('Accessors: none')
        return '\n'.join(lines)

    by_mode: dict[str, list[GraphNode]] = {}
    for acc_node, acc_mode, _ in accessors:
        by_mode.setdefault(acc_mode or 'read', []).append(acc_node)

    for mode in ('read', 'write', 'both'):
        nodes = by_mode.get(mode, [])
        if not nodes:
            continue
        lines.append(f'Accessors ({mode}): {len(nodes)}')
        for n in nodes:
            lines.append(f'  - {n.name}  {n.file_path}:{n.start_line}')

    return '\n'.join(lines)


# Back-compat alias used by existing tests that reference the old name.
_render_enum_member_context = _render_member_context


def _render_member_accessors_flat(
    node: GraphNode, accessors: list[tuple[GraphNode, str, float]]
) -> str:
    """Render a flat accessor list for handle_impact."""
    if node.label == NodeLabel.MODULE_CONSTANT:
        subject = node.name
        kind_label = 'Module_Constant'
    elif node.label == NodeLabel.CLASS_ATTRIBUTE:
        subject = f'{node.class_name}.{node.name}'
        kind_label = 'Class_Attribute'
    else:
        subject = f'{node.class_name}.{node.name}'
        kind_label = 'Enum_Member'
    lines = [f'Impact analysis for: {subject} ({kind_label})']
    if not accessors:
        lines.append('No accessors found.')
        return '\n'.join(lines)
    lines.append(f'Total: {len(accessors)} accessor(s)')
    lines.append('')
    for i, (acc_node, acc_mode, conf) in enumerate(accessors, 1):
        label = acc_node.label.value.title() if acc_node.label else 'Unknown'
        tag = _confidence_tag(conf)
        lines.append(
            f'  {i}. {acc_node.name} ({label}) -- '
            f'{acc_node.file_path}:{acc_node.start_line}'
            f'  [mode: {acc_mode or "read"}]{tag}'
        )
    lines.append('')
    lines.append('Tip: Review each accessor before changing this member.')
    return '\n'.join(lines)


# Back-compat alias.
_render_enum_accessors_flat = _render_member_accessors_flat


def _render_member_explain(
    node: GraphNode, accessors: list[tuple[GraphNode, str, float]]
) -> str:
    """Render a narrative explanation for a member node."""
    if node.label == NodeLabel.MODULE_CONSTANT:
        subject = node.name
        kind_label = 'Module_Constant'
        detail = f'Module constant in ``{node.file_path}``.'
    elif node.label == NodeLabel.CLASS_ATTRIBUTE:
        subject = f'{node.class_name}.{node.name}'
        kind_label = 'Class_Attribute'
        detail = f'Class attribute of ``{node.class_name}``.'
    else:
        subject = f'{node.class_name}.{node.name}'
        kind_label = 'Enum_Member'
        detail = f'Enum member of ``{node.class_name}``.'
    lines = [f'Explanation: {subject} ({kind_label})']
    lines.append('=' * 48)
    lines.append('')
    lines.append(detail)
    lines.append(f'Location: {node.file_path}:{node.start_line}')
    accessor_files = {n.file_path for n, _, _ in accessors}
    lines.append(
        f'Accessed by {len(accessors)} symbol(s) across '
        f'{len(accessor_files)} file(s).'
    )
    return '\n'.join(lines)


# Back-compat alias.
_render_enum_member_explain = _render_member_explain


def handle_context(
    ctx: RepoContext,
    symbol: str,
    *,
    foreign_matches: list[tuple[str, list[SearchResult]]] | None = None,
) -> str:
    """Provide a 360-degree view of a symbol.

    Looks up the symbol by name via full-text search, then retrieves its
    callers, callees, and type references.

    When *foreign_matches* is provided and the local lookup fails, a
    redirect-style response is returned pointing at the foreign repos.

    Args:
        ctx: Per-call repo context.
        symbol: Plain name (``foo``) or dotted path (``Class.method``).
            Multi-dot paths (e.g. ``module.Class.method``) fall back to
            the last segment. When multiple symbols match (e.g., a class
            named ``Foo`` exists in two different files), the best match
            is chosen by score (source files over tests) then lexicographic
            node ID.
        foreign_matches: Optional pre-computed foreign-repo symbol matches.
            When provided and non-empty, appended as a cross-repo footer.

    Returns:
        Formatted view including callers, callees, type refs, and guidance.
    """
    storage = ctx.storage
    if not symbol or not symbol.strip():
        return "Error: 'symbol' parameter is required and cannot be empty."

    results = _resolve_symbol(storage, symbol, top_k=_LOCAL_MATCHES_TOP_K)
    if not results:
        if foreign_matches:
            redirect = _format_foreign_matches(foreign_matches, redirect=True)
            return f"Symbol '{symbol}' not found in this repo.\n\n" + redirect
        return f"Symbol '{symbol}' not found."

    node = storage.get_node(results[0].node_id)
    if not node:
        if foreign_matches:
            redirect = _format_foreign_matches(foreign_matches, redirect=True)
            return f"Symbol '{symbol}' not found in this repo.\n\n" + redirect
        return f"Symbol '{symbol}' not found."

    if node.label in _MEMBER_LABELS:
        return _render_member_context(node, storage)

    label_display = node.label.value.title() if node.label else 'Unknown'
    lines = [f'Symbol: {node.name} ({label_display})']
    lines.append(f'File: {node.file_path}:{node.start_line}-{node.end_line}')

    if node.signature:
        lines.append(f'Signature: {node.signature}')

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

    # Cross-repo footers.
    alternates_footer = _format_alternates(results[1:])
    if alternates_footer:
        lines.append(alternates_footer)

    foreign_footer = _format_foreign_matches(foreign_matches or [])
    if foreign_footer:
        lines.append(foreign_footer)

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
    ctx: RepoContext,
    symbol: str,
    depth: int = 3,
    propagate_through: list[str] | None = None,
    *,
    foreign_matches: list[tuple[str, list[SearchResult]]] | None = None,
) -> str:
    """Analyse the blast radius of changing a symbol, grouped by hop depth.

    Uses BFS traversal through CALLS edges to find all affected symbols
    up to the specified depth, then groups results by distance.

    When ``propagate_through`` is set, only CALLS edges whose
    ``dispatch_kind`` is in that set are followed. Unknown kinds are
    accepted silently (logged at DEBUG level) but have no effect if they
    match no edges.

    Args:
        ctx: Per-call repo context.
        symbol: Plain name (``foo``) or dotted path (``Class.method``).
            Multi-dot paths (e.g. ``module.Class.method``) fall back to
            the last segment. When multiple symbols match (e.g., a class
            named ``Foo`` exists in two different files), the best match
            is chosen by score (source files over tests) then lexicographic
            node ID.
        depth: Maximum traversal depth (default 3).
        propagate_through: Optional list of dispatch_kind values to follow.
            When None, all edges are traversed (default behavior).
        foreign_matches: Optional pre-computed foreign-repo symbol matches,
            appended as a cross-repo footer when non-empty.

    Returns:
        Formatted impact analysis with depth-grouped sections.
    """
    storage = ctx.storage
    if not symbol or not symbol.strip():
        return "Error: 'symbol' parameter is required and cannot be empty."

    depth = max(1, min(depth, MAX_TRAVERSE_DEPTH))

    results = _resolve_symbol(storage, symbol, top_k=_LOCAL_MATCHES_TOP_K)
    if not results:
        if foreign_matches:
            redirect = _format_foreign_matches(foreign_matches, redirect=True)
            return f"Symbol '{symbol}' not found in this repo.\n\n" + redirect
        return f"Symbol '{symbol}' not found."

    start_node = storage.get_node(results[0].node_id)
    if not start_node:
        if foreign_matches:
            redirect = _format_foreign_matches(foreign_matches, redirect=True)
            return f"Symbol '{symbol}' not found in this repo.\n\n" + redirect
        return f"Symbol '{symbol}' not found."

    if start_node.label in _MEMBER_LABELS:
        accessors = storage.get_accessors(start_node.id)
        return _render_member_accessors_flat(start_node, accessors)

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

    foreign_footer = _format_foreign_matches(foreign_matches or [])
    if foreign_footer:
        lines.append(foreign_footer)

    lines.append('')
    lines.append('Tip: Review each affected symbol before making changes.')
    return '\n'.join(lines)


def handle_concurrent_with(
    ctx: RepoContext,
    symbol: str,
    depth: int = 3,
    *,
    foreign_matches: list[tuple[str, list[SearchResult]]] | None = None,
) -> str:
    """Find symbols that may run concurrently with the given symbol.

    Traces outgoing CALLS edges whose dispatch_kind is any non-direct value
    (thread_executor, process_executor, detached_task, enqueued_job,
    callback_registry) and reports the set of reachable symbols grouped by
    dispatch kind.

    Args:
        ctx: Per-call repo context.
        symbol: Plain name or dotted path. Resolution follows the same
            dotted-path rules as handle_context and handle_impact.
        depth: Maximum traversal depth (default 3).
        foreign_matches: Optional pre-computed foreign-repo symbol matches,
            appended as a cross-repo footer when non-empty.

    Returns:
        Formatted list of concurrent-reachable symbols grouped by dispatch kind.
    """
    storage = ctx.storage
    if not symbol or not symbol.strip():
        return "Error: 'symbol' parameter is required and cannot be empty."

    depth = max(1, min(depth, MAX_TRAVERSE_DEPTH))

    results = _resolve_symbol(storage, symbol, top_k=_LOCAL_MATCHES_TOP_K)
    if not results:
        if foreign_matches:
            redirect = _format_foreign_matches(foreign_matches, redirect=True)
            return f"Symbol '{symbol}' not found in this repo.\n\n" + redirect
        return f"Symbol '{symbol}' not found."

    start_node = storage.get_node(results[0].node_id)
    if not start_node:
        if foreign_matches:
            redirect = _format_foreign_matches(foreign_matches, redirect=True)
            return f"Symbol '{symbol}' not found in this repo.\n\n" + redirect
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

    foreign_footer = _format_foreign_matches(foreign_matches or [])
    if foreign_footer:
        lines.append(foreign_footer)

    return '\n'.join(lines)


def handle_dead_code(ctx: RepoContext) -> str:
    """List all symbols marked as dead code.

    Delegates to :func:`~axon.mcp.resources.get_dead_code_list` for the
    shared query and formatting.

    Args:
        ctx: Per-call repo context.

    Returns:
        Formatted list of dead code symbols grouped by file.
    """
    body = get_dead_code_list(ctx.storage)
    return render_with_dead_code_warning(ctx.repo_path, body)


_DIFF_FILE_PATTERN = re.compile(r'^diff --git a/(.+?) b/(.+?)$', re.MULTILINE)
_DIFF_HUNK_PATTERN = re.compile(
    r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', re.MULTILINE
)


def _parse_diff_files(diff: str) -> dict[str, list[tuple[int, int]]]:
    """Parse a git diff and return {file_path: [(start, end), ...]}."""
    changed_files: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None

    for line in diff.split('\n'):
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


def handle_detect_changes(ctx: RepoContext, diff: str) -> str:
    """Map git diff output to affected symbols.

    Parses the diff to find changed files and line ranges, then queries
    the storage backend to identify which symbols those lines belong to.

    Args:
        ctx: Per-call repo context.
        diff: Raw git diff output string.

    Returns:
        Formatted list of affected symbols per changed file.
    """
    storage = ctx.storage
    if not diff.strip():
        return 'Empty diff provided.'

    if len(diff) > MAX_DIFF_LENGTH:
        return (
            f'Diff rejected: exceeds maximum length of {MAX_DIFF_LENGTH:,} '
            f'characters (got {len(diff):,}).'
        )

    changed_files = _parse_diff_files(diff)

    if not changed_files:
        return 'Could not parse any changed files from the diff.'

    lines = [f'Changed files: {len(changed_files)}']
    lines.append('')
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


def handle_cypher(ctx: RepoContext, query: str) -> str:
    """Execute a raw Cypher query and return formatted results.

    Only read-only queries are allowed. Queries containing write keywords
    (DELETE, DROP, CREATE, SET, etc.) are rejected.

    The caller is expected to pass a storage backend opened in read-only
    mode (see server._with_readonly_storage). The WRITE_KEYWORDS regex
    check here remains as defense-in-depth and to return a friendlier
    error message than a raw KuzuDB read-only violation.

    Args:
        ctx: Per-call repo context (storage expected to be opened read-only).
        query: The Cypher query string.

    Returns:
        Formatted query results, or an error message if execution fails.
    """
    storage = ctx.storage
    if len(query) > MAX_CYPHER_LENGTH:
        return (
            f'Query rejected: exceeds maximum length of {MAX_CYPHER_LENGTH:,} '
            f'characters (got {len(query):,}).'
        )

    # Strip comments so write keywords hidden inside comment blocks are detected.
    cleaned_query = sanitize_cypher(query)
    if WRITE_KEYWORDS.search(cleaned_query):
        return (
            'Query rejected: only read-only queries (MATCH/RETURN) are allowed. '
            'Write operations (DELETE, DROP, CREATE, SET, MERGE) are not permitted.'
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
    ctx: RepoContext, file_path: str, min_strength: float = 0.3
) -> str:
    """Query temporal coupling for a file and flag hidden dependencies."""
    storage = ctx.storage
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
    ctx: RepoContext,
    from_symbol: str,
    to_symbol: str,
    max_depth: int = 10,
    *,
    foreign_matches: list[tuple[str, list[SearchResult]]] | None = None,
) -> str:
    """Find the shortest call chain between two symbols via BFS.

    Args:
        ctx: Per-call repo context.
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
        foreign_matches: Optional pre-computed foreign-repo symbol matches
            for *from_symbol*, appended as a cross-repo footer when
            non-empty and local lookup fails.

    Returns:
        Formatted call chain or a message when no path exists.
    """
    storage = ctx.storage
    if not from_symbol or not from_symbol.strip():
        return (
            "Error: 'from_symbol' parameter is required and cannot be empty."
        )
    if not to_symbol or not to_symbol.strip():
        return "Error: 'to_symbol' parameter is required and cannot be empty."

    max_depth = max(1, min(max_depth, MAX_TRAVERSE_DEPTH))

    from_results = _resolve_symbol(
        storage, from_symbol, top_k=_LOCAL_MATCHES_TOP_K
    )
    if not from_results:
        if foreign_matches:
            redirect = _format_foreign_matches(foreign_matches, redirect=True)
            return (
                f"Source symbol '{from_symbol}' not found in this repo.\n\n"
                + redirect
            )
        return f"Source symbol '{from_symbol}' not found."

    to_results = _resolve_symbol(storage, to_symbol, top_k=1)
    if not to_results:
        return f"Target symbol '{to_symbol}' not found."

    src_node = storage.get_node(from_results[0].node_id)
    tgt_node = storage.get_node(to_results[0].node_id)
    if not src_node or not tgt_node:
        return 'Could not resolve one or both symbols.'

    if src_node.id == tgt_node.id:
        return f'Source and target are the same symbol: {src_node.name}'

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


def _build_communities(
    storage: StorageBackend, community: str | None = None
) -> str:
    """Build the communities output body without staleness warnings."""
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
        lines.append('')
        lines.append('Cross-community processes:')
        for row in cross_procs:
            proc_name = row[0] or '?'
            comms = row[1] if len(row) > 1 else []
            comm_str = (
                ' → '.join(comms) if isinstance(comms, list) else str(comms)
            )
            lines.append(f'  - {proc_name} ({comm_str})')

    return '\n'.join(lines)


def handle_communities(ctx: RepoContext, community: str | None = None) -> str:
    """List communities or drill into a specific one."""
    body = _build_communities(ctx.storage, community)
    return render_with_communities_warning(ctx.repo_path, body)


def handle_explain(
    ctx: RepoContext,
    symbol: str,
    *,
    foreign_matches: list[tuple[str, list[SearchResult]]] | None = None,
) -> str:
    """Produce a narrative explanation of a symbol.

    Args:
        ctx: Per-call repo context.
        symbol: Plain name (``foo``) or dotted path (``Class.method``).
            Multi-dot paths (e.g. ``module.Class.method``) fall back to
            the last segment. When multiple symbols match (e.g., a class
            named ``Foo`` exists in two different files), the best match
            is chosen by score (source files over tests) then lexicographic
            node ID.
        foreign_matches: Optional pre-computed foreign-repo symbol matches,
            appended as a cross-repo footer when non-empty.

    Returns:
        Narrative explanation of the symbol's role and relationships.
    """
    storage = ctx.storage
    if not symbol or not symbol.strip():
        return "Error: 'symbol' parameter is required and cannot be empty."

    results = _resolve_symbol(storage, symbol, top_k=_LOCAL_MATCHES_TOP_K)
    if not results:
        if foreign_matches:
            redirect = _format_foreign_matches(foreign_matches, redirect=True)
            return f"Symbol '{symbol}' not found in this repo.\n\n" + redirect
        return f"Symbol '{symbol}' not found."

    node = storage.get_node(results[0].node_id)
    if not node:
        if foreign_matches:
            redirect = _format_foreign_matches(foreign_matches, redirect=True)
            return f"Symbol '{symbol}' not found in this repo.\n\n" + redirect
        return f"Symbol '{symbol}' not found."

    if node.label in _MEMBER_LABELS:
        accessors = storage.get_accessors(node.id)
        return _render_member_explain(node, accessors)

    label_display = node.label.value.title() if node.label else 'Unknown'
    lines = [f'Explanation: {node.name} ({label_display})']
    lines.append('=' * 48)
    lines.append('')

    roles = []
    if node.is_entry_point:
        roles.append('Entry point')
    if node.is_exported:
        roles.append('Exported')
    if node.is_dead:
        roles.append('Dead code (unreachable)')
    if roles:
        lines.append(f'Role: {", ".join(roles)}')

    lines.append(
        f'Location: {node.file_path}:{node.start_line}-{node.end_line}'
    )

    if node.signature:
        lines.append(f'Signature: {node.signature}')

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
            proc_name = row[0] or '?'
            lines.append(f'  - {proc_name}')

    foreign_footer = _format_foreign_matches(foreign_matches or [])
    if foreign_footer:
        lines.append(foreign_footer)

    return '\n'.join(lines)


def _build_review_risk(storage: StorageBackend, diff: str) -> str:
    """Build the PR risk assessment body without staleness warnings."""
    if not diff.strip():
        return 'Empty diff provided.'

    if len(diff) > MAX_DIFF_LENGTH:
        return (
            f'Diff rejected: exceeds maximum length of {MAX_DIFF_LENGTH:,} '
            f'characters (got {len(diff):,}).'
        )

    changed_files = _parse_diff_files(diff)
    if not changed_files:
        return 'Could not parse any changed files from the diff.'

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


def handle_review_risk(ctx: RepoContext, diff: str) -> str:
    """Assess PR risk by synthesizing multiple graph signals."""
    body = _build_review_risk(ctx.storage, diff)
    return render_with_dead_code_warning(ctx.repo_path, body)


def handle_file_context(ctx: RepoContext, file_path: str) -> str:
    """Provide comprehensive context for a single file."""
    storage = ctx.storage
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

    # Enum summary: for each enum in the file, count members and accessors.
    enum_rows = (
        storage.execute_raw(
            f"MATCH (e:Enum) WHERE e.file_path = '{escaped}' "
            f'OPTIONAL MATCH (e)-[d:CodeRelation]->(m:EnumMember) '
            f"WHERE d.rel_type = 'defines' "
            f'OPTIONAL MATCH (acc)-[a:CodeRelation]->(m) '
            f"WHERE a.rel_type = 'accesses' "
            f'RETURN e.name, count(DISTINCT m), count(DISTINCT acc) '
            f'ORDER BY e.name'
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
        lines.append('')
        comm_parts = [f'{r[0]} ({r[1]} symbols)' for r in comm_rows if r[0]]
        lines.append(f'Communities: {", ".join(comm_parts)}')

    if enum_rows:
        enum_parts = []
        for row in enum_rows:
            enum_name = row[0] or '?'
            member_count = int(row[1]) if row[1] is not None else 0
            accessor_count = int(row[2]) if row[2] is not None else 0
            enum_parts.append(
                f'{enum_name} ({member_count} members, '
                f'{accessor_count} accessors)'
            )
        if enum_parts:
            lines.append('')
            lines.append(f'Enums: {"; ".join(enum_parts)}')

    # Class attribute summary.
    cls_attr_rows = (
        storage.execute_raw(
            f"MATCH (c:Class) WHERE c.file_path = '{escaped}' "
            f'OPTIONAL MATCH (c)-[d:CodeRelation]->(a:Classattribute) '
            f"WHERE d.rel_type = 'defines' "
            f'OPTIONAL MATCH (acc)-[r:CodeRelation]->(a) '
            f"WHERE r.rel_type = 'accesses' "
            f'RETURN c.name, count(DISTINCT a), count(DISTINCT acc) '
            f'ORDER BY c.name'
        )
        or []
    )
    cls_attr_parts = []
    for row in cls_attr_rows:
        cls_name = row[0] or '?'
        attr_count = int(row[1]) if row[1] is not None else 0
        acc_count = int(row[2]) if row[2] is not None else 0
        if attr_count > 0:
            cls_attr_parts.append(
                f'{cls_name} ({attr_count} attrs, {acc_count} accessors)'
            )
    if cls_attr_parts:
        lines.append('')
        lines.append(f'Class attributes: {"; ".join(cls_attr_parts)}')

    # Module constant summary.
    mod_const_rows = (
        storage.execute_raw(
            f'MATCH (f:File)-[d:CodeRelation]->(m:Moduleconstant) '
            f"WHERE f.file_path = '{escaped}' AND d.rel_type = 'defines' "
            f'OPTIONAL MATCH (acc)-[r:CodeRelation]->(m) '
            f"WHERE r.rel_type = 'accesses' "
            f'RETURN count(DISTINCT m), count(DISTINCT acc)'
        )
        or []
    )
    if mod_const_rows:
        row = mod_const_rows[0]
        const_count = int(row[0]) if row[0] is not None else 0
        acc_count = int(row[1]) if row[1] is not None else 0
        if const_count > 0:
            lines.append('')
            lines.append(
                f'Module constants: {const_count}'
                f' ({acc_count} accessor references)'
            )

    return '\n'.join(lines)


def handle_cycles(ctx: RepoContext, min_size: int = 2) -> str:
    """Detect circular dependencies using strongly connected components."""
    storage = ctx.storage
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
        list(component) for component in sccs if len(component) >= min_size
    ]

    if not cycles:
        return 'No circular dependencies detected.'

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
    ctx: RepoContext, diff: str = '', symbols: list[str] | None = None
) -> str:
    """Find tests likely affected by code changes.

    When ``ctx.repo_path`` is set, the function also:
    - loads pytest config to narrow test-file classification,
    - skips changed files whose hunks are entirely docstrings or comments.

    When ``ctx.repo_path`` is None the function operates in config-unaware
    mode: default heuristic only, no hunk-executable filtering.

    Args:
        ctx: Per-call repo context.
        diff: Raw git diff output.
        symbols: List of symbol names to check instead of a diff. Each
            item can be a plain name (``foo``) or a dotted path
            (``Class.method``). Multi-dot paths (e.g.
            ``module.Class.method``) fall back to the last segment. When
            multiple symbols match (e.g., a class named ``Foo`` exists in
            two different files), the best match is chosen by score
            (source files over tests) then lexicographic node ID.

    Returns:
        Formatted impact report string.
    """
    storage = ctx.storage
    repo_path = ctx.repo_path
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
            results = _resolve_symbol(storage, sym_name, top_k=1)
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
