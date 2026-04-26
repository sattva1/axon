"""MCP server for Axon — exposes code intelligence tools over stdio and HTTP.

Registers fifteen tools and three resources that give AI agents and MCP clients
access to the Axon knowledge graph.  The server lazily initialises a
:class:`KuzuBackend` from the ``.axon/kuzu`` directory in the current
working directory.

Usage::

    # MCP server only
    axon mcp

    # MCP server with live file watching (recommended)
    axon serve --watch
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from mcp.server import Server
from mcp.server.fastmcp.server import StreamableHTTPASGIApp
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import Resource, TextContent, Tool, ToolAnnotations

from axon.core.drift import DriftCache, DriftLevel
from axon.core.repos import RepoNotFound, RepoPool, RepoResolver
from axon.core.storage.kuzu_backend import KuzuBackend
from axon.mcp.freshness import render_with_drift_warning
from axon.mcp.repo_context import RepoContext
from axon.mcp.repo_routing import RoutingError, route_for_diff, route_for_path
from axon.mcp.resources import get_dead_code_list, get_overview, get_schema
from axon.mcp.tools import (
    MAX_TRAVERSE_DEPTH,
    _foreign_query_hit_counts,
    _foreign_symbol_matches,
    _new_ref_id,
    handle_call_path,
    handle_communities,
    handle_concurrent_with,
    handle_context,
    handle_coupling,
    handle_cycles,
    handle_cypher,
    handle_dead_code,
    handle_detect_changes,
    handle_explain,
    handle_file_context,
    handle_impact,
    handle_list_repos,
    handle_query,
    handle_review_risk,
    handle_test_impact,
)

logger = logging.getLogger(__name__)

server = Server('axon')


@dataclass(slots=True)
class _ServerState:
    storage: KuzuBackend | None = None
    lock: asyncio.Lock | None = None
    db_path: Path | None = None
    repo_path: Path | None = None
    # Phase 3 multi-repo additions:
    resolver: RepoResolver | None = None
    pool: RepoPool | None = None
    drift_cache: DriftCache | None = None
    local_slug: str | None = None
    # Initialised lazily on first _dispatch_tool call, never at import time.
    multi_repo_init_lock: asyncio.Lock | None = None


_state = _ServerState()


def _resolve_db_path() -> Path:
    if _state.db_path is None:
        _state.db_path = Path.cwd() / '.axon' / 'kuzu'
    return _state.db_path


def set_storage(storage: KuzuBackend, repo_path: Path | None = None) -> None:
    """Inject a pre-initialised storage backend (e.g. from ``axon serve --watch``).

    Args:
        storage: Initialised storage backend to inject.
        repo_path: Absolute path to the repository root. When provided,
            freshness checks and hunk-executable filtering are enabled.
            Existing callers that omit this argument keep working unchanged.
    """
    _state.storage = storage
    _state.repo_path = repo_path


def set_lock(lock: asyncio.Lock) -> None:
    """Inject a shared lock for coordinating storage access with the file watcher."""
    _state.lock = lock


def set_db_path(path: Path) -> None:
    """Inject a custom database path for standalone MCP server mode.

    Must be called before the server handles any tool requests (i.e., before
    entering the event loop in ``main()``).
    """
    _state.db_path = path


@contextmanager
def _open_storage() -> Iterator[KuzuBackend]:
    """Open a short-lived read-only connection for a single tool/resource call.

    Used when no persistent storage was injected (read-only fallback mode).
    Each call gets a fresh connection that sees the latest on-disk data and
    releases the file lock immediately after the query completes.
    """
    db_path = _resolve_db_path()
    if not db_path.exists():
        raise FileNotFoundError(f"No .axon/kuzu directory in {db_path.parent.parent}")
    storage = KuzuBackend()
    storage.initialize(db_path, read_only=True, max_retries=3, retry_delay=0.3)
    try:
        yield storage
    finally:
        storage.close()


async def _with_readonly_storage(fn: Callable[[KuzuBackend], str]) -> str:
    """Run *fn* against a fresh read-only storage connection.

    Used for user-submitted Cypher. Enforces read-only at the DB layer,
    regardless of whether a read-write backend is injected via set_storage().

    """
    def _run() -> str:
        with _open_storage() as st:
            return fn(st)

    return await asyncio.to_thread(_run)


async def _with_storage(fn: Callable[[KuzuBackend], str]) -> str:
    """Run *fn* against the appropriate storage backend.

    Uses the injected persistent backend when available (with optional
    async lock), otherwise opens a short-lived read-only connection.
    """
    if _state.storage is not None:
        if _state.lock is not None:
            async with _state.lock:
                return await asyncio.to_thread(fn, _state.storage)
        return await asyncio.to_thread(fn, _state.storage)

    def _run() -> str:
        with _open_storage() as st:
            return fn(st)

    return await asyncio.to_thread(_run)


# ---------------------------------------------------------------------------
# Multi-repo constants
# ---------------------------------------------------------------------------

# Tools that should be routed by their `file_path` argument.
_PATH_KEYED_TOOLS: frozenset[str] = frozenset(
    {'axon_coupling', 'axon_file_context'}
)

# Tools that should be routed by their `diff` argument.
_DIFF_KEYED_TOOLS: frozenset[str] = frozenset(
    {'axon_review_risk', 'axon_test_impact', 'axon_detect_changes'}
)

# Symbol-keyed tools receive cross-repo fan-out footers.
_SYMBOL_KEYED_TOOLS: frozenset[str] = frozenset(
    {
        'axon_context',
        'axon_explain',
        'axon_call_path',
        'axon_impact',
        'axon_concurrent_with',
    }
)

# Description added to `repo` property on all multi-repo-relevant tools.
_REPO_PARAM_DESC = (
    'Optional repo identifier - slug, absolute path, or relative path. '
    'Defaults to the local repo. Use axon_list_repos to discover available slugs.'
)


# ---------------------------------------------------------------------------
# Multi-repo lazy init
# ---------------------------------------------------------------------------


async def _ensure_multi_repo() -> None:
    """Idempotent. ONLY called from inside _dispatch_tool / call_tool.

    Never called at module import time. Never called from the stdio-to-HTTP
    proxy in cli/main.py:_proxy_stdio_to_http_mcp - that proxy only forwards
    bytes between two transports and never invokes _dispatch_tool.
    """
    if _state.multi_repo_init_lock is None:
        _state.multi_repo_init_lock = asyncio.Lock()
    async with _state.multi_repo_init_lock:
        if _state.resolver is not None:
            return
        local_repo_path = _state.repo_path or Path.cwd()
        resolver = RepoResolver(local_repo_path=local_repo_path)
        local_entry = resolver.local()
        _state.resolver = resolver
        _state.pool = RepoPool(resolver)
        _state.drift_cache = DriftCache()
        _state.local_slug = (
            local_entry.slug if local_entry is not None else None
        )


async def _build_repo_context(
    tool_name: str, arguments: dict
) -> RepoContext | str:
    """Resolve the target repo and return a RepoContext for handler dispatch.

    Returns a string refusal message instead of a RepoContext when:
    - A routing error occurs (ambiguous path, missing repo, cross-repo diff).
    - The resolved foreign repo is STALE_MAJOR (re-index hint included).

    Args:
        tool_name: Name of the MCP tool being dispatched.
        arguments: Raw tool arguments dict.

    Returns:
        RepoContext on success, or a string body for early refusal.
    """
    await _ensure_multi_repo()

    resolver = _state.resolver
    pool = _state.pool
    drift_cache = _state.drift_cache
    local_slug = _state.local_slug
    assert (
        resolver is not None and pool is not None and drift_cache is not None
    )

    # Resolve the target repo entry.
    explicit_repo: str | None = arguments.get('repo')
    try:
        if tool_name in _PATH_KEYED_TOOLS:
            entry = route_for_path(
                resolver, arguments.get('file_path', ''), explicit_repo
            )
        elif tool_name in _DIFF_KEYED_TOOLS:
            entry = route_for_diff(
                resolver, arguments.get('diff', ''), explicit_repo
            )
        else:
            if explicit_repo:
                entry = resolver.resolve_strict(explicit_repo)
            else:
                local = resolver.local()
                if local is None:
                    return (
                        'No local repo configured. '
                        'Pass repo=<slug> to query a specific repo.'
                    )
                entry = local
    except RoutingError as exc:
        candidates_hint = (
            f' Known repos: {", ".join(exc.candidates)}.'
            if exc.candidates
            else ''
        )
        return f'Routing error: {exc}.{candidates_hint}'
    except RepoNotFound as exc:
        candidates_hint = (
            f' Known repos: {", ".join(exc.candidates)}.'
            if exc.candidates
            else ''
        )
        return f"Repo '{exc.identifier}' not found.{candidates_hint}"

    # Stale-major check for foreign repos.
    is_local = entry.slug == local_slug
    if not is_local:
        try:
            report = drift_cache.get_or_probe(entry.path)
            if report.level == DriftLevel.STALE_MAJOR:
                return (
                    f"Repo '{entry.slug}' index is significantly out of date "
                    f'(STALE_MAJOR: {report.reason}). '
                    f'Re-run `axon analyze` in that repo and try again.'
                )
        except Exception:
            pass  # If drift probe fails, proceed optimistically.

    # Build storage.
    if is_local:
        if _state.storage is not None:
            storage = _state.storage
        else:
            # Standalone axon mcp mode - open a fresh read-only connection.
            # The handler must be called within a _open_storage() context;
            # we return None here and handle the open in call_tool.
            # To avoid complexity we return a sentinel instead.
            storage = None  # type: ignore[assignment]
    else:
        try:
            storage = await asyncio.to_thread(pool.get, entry.slug)
        except Exception as exc:
            return f"Repo '{entry.slug}' unavailable: {exc}"

    return RepoContext(
        storage=storage,
        slug=entry.slug,
        is_local=is_local,
        repo_path=entry.path if not is_local else _state.repo_path,
        local_slug=local_slug,
    )


_TOOL_ANNOTATIONS = ToolAnnotations(readOnlyHint=True, idempotentHint=True)

TOOLS: list[Tool] = [
    Tool(
        name='axon_list_repos',
        description=(
            'List all indexed repositories the current MCP session can reach,'
            ' with freshness, watcher status, and reachability per entry.\n\n'
            'When to use this instead of Grep/Read: use this as the first'
            ' step before any cross-repo query to discover available slugs'
            ' and assess whether each repo is fresh enough to trust.\n\n'
            'Parameters:\n'
            '  (none) - this tool takes no arguments.\n\n'
            'Returns: a numbered list of repos. Each entry shows slug, path,'
            ' file/symbol/relationship counts, freshness level'
            ' (fresh/stale_minor/stale_major/unknown), watcher alive state,'
            ' and whether the repo is reachable from this session. The local'
            ' repo is marked (LOCAL). A usage hint footer explains how to'
            ' target a specific repo with `repo=<slug>`.\n\n'
            'Multi-repo posture: enumerates every entry registered in'
            ' ~/.axon/repos. Stale-major repos are listed but queries against'
            ' them will be refused - re-run `axon analyze` to refresh.'
        ),
        inputSchema={'type': 'object', 'properties': {}},
        annotations=_TOOL_ANNOTATIONS,
    ),
    Tool(
        name='axon_query',
        description=(
            'Search the knowledge graph using hybrid keyword-plus-vector'
            ' search and return ranked symbols matching the query.\n\n'
            'When to use this instead of Grep/Read: prefer this over grep'
            ' when the question is conceptual - "auth handlers",'
            ' "things that compute totals", "classes that validate input" -'
            ' rather than a literal string. Returns ranked symbols grouped by'
            ' execution flow with file, label, and snippet per result.\n\n'
            'Parameters:\n'
            '  - `query`: natural-language or keyword search text.\n'
            '  - `limit`: maximum results to return (default 20, cap 100).\n'
            '  - `repo` (optional): slug, absolute path, or relative path of'
            ' the repo to query. Defaults to the local repo. Use'
            ' `axon_list_repos` to discover slugs.\n\n'
            'Returns: results numbered and grouped by execution-process'
            ' section when process membership is detected, otherwise a flat'
            ' ranked list. Each entry shows name, label, file path, and a'
            ' short snippet. A cross-repo footer reports hit counts in other'
            ' repos so you know where to look next.\n\n'
            'Multi-repo posture: when `repo` is omitted, queries the local'
            ' repo and appends a footer with hit counts from up to 5 foreign'
            ' repos (excluding stale-major ones). Pass `repo=<slug>` to'
            ' search a specific foreign repo directly.'
        ),
        inputSchema={
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Search query text.',
                },
                'limit': {
                    'type': 'integer',
                    'description': 'Maximum number of results (default 20).',
                    'default': 20,
                },
                'repo': {'type': 'string', 'description': _REPO_PARAM_DESC},
            },
            'required': ['query'],
        },
        annotations=_TOOL_ANNOTATIONS,
    ),
    Tool(
        name='axon_context',
        description=(
            'Get a 360-degree structural view of a named symbol: callers,'
            ' callees, type references, heritage, importers, and community'
            ' membership in one call.\n\n'
            'When to use this instead of Grep/Read: prefer this over Read'
            ' and Grep when you want the full neighborhood of a symbol -'
            ' who calls it, what it calls, what types it references, which'
            ' files import it, and what community it belongs to - without'
            ' manually chasing definitions across files.\n\n'
            'Parameters:\n'
            '  - `symbol`: plain name (`foo`) or dotted path (`Class.method`).'
            ' Multi-dot paths fall back to the last segment.\n'
            '  - `repo` (optional): target repo slug, absolute path, or'
            ' relative path. Defaults to the local repo.\n\n'
            'Returns: symbol header (label, file, line range, signature),'
            ' callers with dispatch kind and confidence, callees with await'
            ' and try annotations, type references, heritage (extends/'
            'implements), importers, alternate local matches, and a'
            ' cross-repo "also exists in" footer.\n\n'
            'Multi-repo posture: when the symbol is not found locally and'
            ' foreign repos have matches, returns a redirect response listing'
            ' those repos with `repo=<slug>` hint. Cross-repo footers are'
            ' appended for symbols that also exist in accessible foreign repos.'
        ),
        inputSchema={
            'type': 'object',
            'properties': {
                'symbol': {
                    'type': 'string',
                    'description': 'Name of the symbol to look up.',
                },
                'repo': {'type': 'string', 'description': _REPO_PARAM_DESC},
            },
            'required': ['symbol'],
        },
        annotations=_TOOL_ANNOTATIONS,
    ),
    Tool(
        name='axon_impact',
        description=(
            'Blast-radius analysis: find all symbols affected by changing'
            ' a given symbol, grouped by hop depth from the change point.\n\n'
            'When to use this instead of Grep/Read: prefer this over grep'
            ' when you are about to change a symbol and want to know what'
            ' breaks. Grep finds some callers in the same file or module;'
            ' this traces the full upstream caller graph with depth labels'
            ' and confidence scores, including indirect dependents that grep'
            ' will miss.\n\n'
            'Parameters:\n'
            '  - `symbol`: name of the symbol being changed.\n'
            '  - `depth`: maximum traversal depth (default 3, max'
            f' {MAX_TRAVERSE_DEPTH}).\n'
            '  - `propagate_through`: optional list of dispatch_kind values'
            ' to follow (direct, thread_executor, process_executor,'
            ' detached_task, enqueued_job, callback_registry). When omitted,'
            ' all edges are traversed.\n'
            '  - `repo` (optional): target repo slug or path.\n\n'
            'Returns: impact header with total count, then depth-grouped'
            ' sections (Depth 1 = direct callers, Depth 2 = indirect, etc.).'
            ' Each entry shows name, label, file, and line. Confidence scores'
            ' appear for depth-1 callers.\n\n'
            'Multi-repo posture: appends a cross-repo footer when the symbol'
            ' also exists in accessible foreign repos. Stale-major foreign'
            ' repos are excluded from footers.'
        ),
        inputSchema={
            'type': 'object',
            'properties': {
                'symbol': {
                    'type': 'string',
                    'description': 'Name of the symbol to analyse.',
                },
                'depth': {
                    'type': 'integer',
                    'description': (
                        f'Maximum traversal depth (default 3, max'
                        f' {MAX_TRAVERSE_DEPTH}).'
                    ),
                    'default': 3,
                    'minimum': 1,
                    'maximum': MAX_TRAVERSE_DEPTH,
                },
                'propagate_through': {
                    'type': 'array',
                    'items': {
                        'type': 'string',
                        'enum': [
                            'direct',
                            'thread_executor',
                            'process_executor',
                            'detached_task',
                            'enqueued_job',
                            'callback_registry',
                        ],
                    },
                    'description': (
                        'Optional. When set, only traverse CALLS edges whose '
                        'dispatch_kind is in this set. Default: traverse all edges.'
                    ),
                },
                'repo': {'type': 'string', 'description': _REPO_PARAM_DESC},
            },
            'required': ['symbol'],
        },
        annotations=_TOOL_ANNOTATIONS,
    ),
    Tool(
        name='axon_dead_code',
        description=(
            'List all symbols flagged as dead (unreachable) code in the'
            ' indexed graph, grouped by file.\n\n'
            'When to use this instead of Grep/Read: use this instead of'
            ' manually scanning for unused definitions. Axon computes'
            ' reachability over the full call graph at index time; this'
            ' tool surfaces the result without requiring any query text.\n\n'
            'Parameters:\n'
            '  - `repo` (optional): target repo slug or path. Defaults to'
            ' the local repo.\n\n'
            'Returns: symbols grouped by file, each entry showing name,'
            ' label, and line number. A freshness warning is prepended when'
            ' dead-code data predates the last index run.\n\n'
            'Multi-repo posture: targets a single repo per call. Pass'
            ' `repo=<slug>` to inspect a foreign repo.'
            ' Stale-major foreign repos are refused.'
        ),
        inputSchema={
            'type': 'object',
            'properties': {
                'repo': {'type': 'string', 'description': _REPO_PARAM_DESC}
            },
        },
        annotations=_TOOL_ANNOTATIONS,
    ),
    Tool(
        name='axon_detect_changes',
        description=(
            'Parse a raw git diff and map changed lines to the indexed'
            ' symbols that overlap those lines in each changed file.\n\n'
            'When to use this instead of Grep/Read: use this as the first'
            ' step when reviewing a patch - it tells you which named symbols'
            ' are touched by each hunk, so you can then call `axon_impact`'
            ' on the affected symbols to see downstream effects without'
            ' manually correlating line numbers to function boundaries.\n\n'
            'Parameters:\n'
            '  - `diff`: raw `git diff` output (max 100 000 chars).\n'
            '  - `repo` (optional): target repo slug or path. When omitted,'
            ' the diff is routed to the repo that owns the majority of the'
            ' changed files.\n\n'
            'Returns: a section per changed file listing affected symbol'
            ' names, labels, and line ranges. Files with no indexed symbols'
            ' in the changed lines are noted explicitly. A total count'
            ' closes the output.\n\n'
            'Multi-repo posture: routes to the repo that owns the changed'
            ' files. *Cross-repo diffs (files from different repos in one'
            ' patch) are refused* with a routing error listing the repos'
            ' detected.'
        ),
        inputSchema={
            'type': 'object',
            'properties': {
                'diff': {
                    'type': 'string',
                    'description': 'Raw git diff output.',
                    'maxLength': 100000,
                },
                'repo': {'type': 'string', 'description': _REPO_PARAM_DESC},
            },
            'required': ['diff'],
        },
        annotations=_TOOL_ANNOTATIONS,
    ),
    Tool(
        name='axon_cypher',
        description=(
            'Execute a read-only Cypher query directly against the Axon'
            ' knowledge graph and return raw tabular results.\n\n'
            'When to use this instead of Grep/Read: use this for ad-hoc'
            ' structural questions that no other tool answers - e.g., "all'
            ' nodes with more than 20 callers", "files that import X and'
            ' belong to community Y". Requires familiarity with the Axon'
            ' graph schema (call `axon://schema` resource for the layout).\n\n'
            'Parameters:\n'
            '  - `query`: Cypher MATCH/RETURN query (max 100 000 chars).'
            ' *Write operations (CREATE, DELETE, SET, MERGE, DROP) are'
            ' rejected.*\n'
            '  - `repo` (optional): target repo slug or path.\n\n'
            'Returns: tabular results as "N rows:" followed by one'
            ' pipe-delimited row per line, or "Query returned no results."\n\n'
            'Multi-repo posture: each call targets a single repo. The query'
            ' runs against a read-only connection regardless of whether the'
            ' local injected storage is read-write.'
        ),
        inputSchema={
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Cypher query string.',
                    'maxLength': 100000,
                },
                'repo': {'type': 'string', 'description': _REPO_PARAM_DESC},
            },
            'required': ['query'],
        },
        annotations=_TOOL_ANNOTATIONS,
    ),
    Tool(
        name='axon_coupling',
        description=(
            'Show files that are temporally coupled with a given file,'
            ' revealing hidden dependencies inferred from git co-change'
            ' history.\n\n'
            'When to use this instead of Grep/Read: use this when you want'
            ' to know which files tend to change together with a given file'
            ' - even when there is no static import between them. Grep'
            ' cannot surface this relationship; it is computed from commit'
            ' history during indexing.\n\n'
            'Parameters:\n'
            '  - `file_path`: path to the file to analyse (relative or'
            ' absolute).\n'
            '  - `min_strength`: minimum coupling strength (0.0-1.0,'
            ' default 0.3). Lower values return more pairs.\n'
            '  - `repo` (optional): target repo slug or path. When omitted,'
            ' the file path is used to auto-route to the owning repo.\n\n'
            'Returns: coupled files ordered by strength descending. Each'
            ' entry shows path, strength, co-change count, and whether a'
            ' static import already covers the dependency. Files with no'
            ' static import are flagged as hidden dependencies.\n\n'
            'Multi-repo posture: auto-routes to the repo that owns'
            ' `file_path`. Pass `repo=<slug>` to override.'
        ),
        inputSchema={
            'type': 'object',
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'Path to the file to analyze coupling for.',
                },
                'min_strength': {
                    'type': 'number',
                    'description': (
                        'Minimum coupling strength threshold (default 0.3).'
                    ),
                    'default': 0.3,
                },
                'repo': {'type': 'string', 'description': _REPO_PARAM_DESC},
            },
            'required': ['file_path'],
        },
        annotations=_TOOL_ANNOTATIONS,
    ),
    Tool(
        name='axon_communities',
        description=(
            'List detected code communities (Leiden clusters over the call'
            ' graph) or drill into a specific community to see its members.\n\n'
            'When to use this instead of Grep/Read: use this to understand'
            ' module-level boundaries that were not explicitly designed -'
            ' which functions cluster together by call patterns, and which'
            ' processes cross community boundaries. Read cannot surface this;\n'
            ' it must be computed over the full graph.\n\n'
            'Parameters:\n'
            '  - `community`: optional community name to drill into.'
            ' When omitted, returns a summary list of all communities with'
            ' cohesion score and symbol count.\n'
            '  - `repo` (optional): target repo slug or path.\n\n'
            'Returns (list mode): communities ranked by cohesion descending,'
            ' each with name, cohesion score, and symbol count. A'
            ' cross-community processes section lists processes that span'
            ' multiple communities. Returns (drill mode): all members of the'
            ' named community with name, label, file, line, entry-point,'
            ' and exported flags.\n\n'
            'Multi-repo posture: targets a single repo per call.'
            ' A freshness warning is prepended when community data is stale.'
        ),
        inputSchema={
            'type': 'object',
            'properties': {
                'community': {
                    'type': 'string',
                    'description': (
                        'Optional community name to drill into. Omit to list all.'
                    ),
                },
                'repo': {'type': 'string', 'description': _REPO_PARAM_DESC},
            },
        },
        annotations=_TOOL_ANNOTATIONS,
    ),
    Tool(
        name='axon_explain',
        description=(
            'Get a narrative explanation of a symbol: its role, community'
            ' membership, process flows it participates in, caller count,'
            ' and callee summary - formatted for onboarding and code review.\n\n'
            'When to use this instead of Grep/Read: use this when you need'
            ' a plain-language description of what a symbol does and where'
            ' it fits in the system, rather than its raw call graph.'
            ' Combines information from multiple graph queries into a single'
            ' readable summary.\n\n'
            'Parameters:\n'
            '  - `symbol`: plain name or dotted path of the symbol.\n'
            '  - `repo` (optional): target repo slug or path.\n\n'
            'Returns: explanation header, role flags (entry point, exported,'
            ' dead code), location, signature, community, caller and callee'
            ' summaries (top 5 with overflow count), and process flows the'
            ' symbol participates in. A cross-repo footer lists other repos'
            ' that contain the symbol.\n\n'
            'Multi-repo posture: appends a cross-repo footer from accessible'
            ' foreign repos. When the symbol is absent locally and present'
            ' in a foreign repo, returns a redirect with `repo=<slug>` hint.'
        ),
        inputSchema={
            'type': 'object',
            'properties': {
                'symbol': {
                    'type': 'string',
                    'description': 'Name of the symbol to explain.',
                },
                'repo': {'type': 'string', 'description': _REPO_PARAM_DESC},
            },
            'required': ['symbol'],
        },
        annotations=_TOOL_ANNOTATIONS,
    ),
    Tool(
        name='axon_review_risk',
        description=(
            'PR risk assessment: synthesises multiple graph signals from a'
            ' git diff to produce a risk score (LOW/MEDIUM/HIGH out of 10)'
            ' with supporting evidence.\n\n'
            'When to use this instead of Grep/Read: use this before merging'
            ' a PR to quantify risk from blast radius, missing co-change'
            ' files, and community boundary crossings. Grep can find changed'
            ' symbols; this computes downstream dependents, hidden coupling'
            ' violations, and cross-community span in one call.\n\n'
            'Parameters:\n'
            '  - `diff`: raw `git diff` output (max 100 000 chars).\n'
            '  - `repo` (optional): target repo slug or path. Auto-routes'
            ' to the repo that owns the changed files when omitted.\n\n'
            'Returns: risk level and numeric score, changed symbols with'
            ' downstream-dependent counts, missing co-change files (coupled'
            ' files not present in the diff), and community boundary'
            ' crossings. Score components: entry points hit, missing'
            ' co-change files, dependents / 10, +2 for multi-community span.\n\n'
            'Multi-repo posture: routes to the repo that owns the diff files.'
            ' *Cross-repo diffs are refused.* Stale-major foreign repos'
            ' are refused with a re-index hint.'
        ),
        inputSchema={
            'type': 'object',
            'properties': {
                'diff': {
                    'type': 'string',
                    'description': 'Raw git diff output.',
                    'maxLength': 100000,
                },
                'repo': {'type': 'string', 'description': _REPO_PARAM_DESC},
            },
            'required': ['diff'],
        },
        annotations=_TOOL_ANNOTATIONS,
    ),
    Tool(
        name='axon_call_path',
        description=(
            'Find the shortest call chain between two named symbols using'
            ' BFS over CALLS edges in the knowledge graph.\n\n'
            'When to use this instead of Grep/Read: use this instead of'
            ' manual grep-tracing when you need the shortest execution path'
            ' between two functions. Grep requires chasing each callee by'
            ' hand; this returns the full path in one call, annotated with'
            ' dispatch kinds (thread, async, etc.) where applicable.\n\n'
            'Parameters:\n'
            '  - `from_symbol`: source symbol name or dotted path. Resolved'
            ' with top_k=5 so alternate matches surface if the primary is'
            ' wrong.\n'
            '  - `to_symbol`: target symbol name or dotted path. Resolved'
            ' with top_k=1 (binary success/fail).\n'
            '  - `max_depth`: maximum BFS hops (default 10, max'
            f' {MAX_TRAVERSE_DEPTH}).\n'
            '  - `repo` (optional): target repo slug or path.\n\n'
            'Returns: a header with the path as "A -> B -> C (N hops)"'
            ' followed by a numbered list of each hop with label, file,'
            ' line, and dispatch-kind annotation when non-direct.\n\n'
            'Multi-repo posture: operates within a single repo. A cross-repo'
            ' footer is appended when `from_symbol` also exists in foreign'
            ' repos. Stale-major foreign repos are excluded from footers.'
        ),
        inputSchema={
            'type': 'object',
            'properties': {
                'from_symbol': {
                    'type': 'string',
                    'description': 'Name of the source symbol.',
                },
                'to_symbol': {
                    'type': 'string',
                    'description': 'Name of the target symbol.',
                },
                'max_depth': {
                    'type': 'integer',
                    'description': (
                        f'Maximum hops (default 10, max {MAX_TRAVERSE_DEPTH}).'
                    ),
                    'default': 10,
                    'minimum': 1,
                    'maximum': MAX_TRAVERSE_DEPTH,
                },
                'repo': {'type': 'string', 'description': _REPO_PARAM_DESC},
            },
            'required': ['from_symbol', 'to_symbol'],
        },
        annotations=_TOOL_ANNOTATIONS,
    ),
    Tool(
        name='axon_file_context',
        description=(
            'Get comprehensive context for a single file: all its indexed'
            ' symbols, import graph (in and out), temporal coupling, dead'
            ' code, community membership, enum summaries, and class-attribute'
            ' summaries in one call.\n\n'
            'When to use this instead of Grep/Read: use this instead of'
            ' Read + grep when you need a structural overview of an entire'
            ' file before editing it. Read gives you source text; this gives'
            ' you the graph neighborhood - what the file imports, what imports'
            ' it, which symbols are dead, and what it couples to.\n\n'
            'Parameters:\n'
            '  - `file_path`: path to the file (relative or absolute).\n'
            '  - `repo` (optional): target repo slug or path. Auto-routes'
            ' to the owning repo from `file_path` when omitted.\n\n'
            'Returns: symbols table (name, label, line, entry-point/exported/'
            'dead flags), outbound imports, inbound importers, top-5 coupled'
            ' files by strength, dead-code entries, community memberships,'
            ' enum summaries with member and accessor counts, class-attribute'
            ' summaries, and module-constant counts.\n\n'
            'Multi-repo posture: auto-routes to the repo that owns the file'
            ' path. Pass `repo=<slug>` to override.'
        ),
        inputSchema={
            'type': 'object',
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'Path to the file to analyze.',
                },
                'repo': {'type': 'string', 'description': _REPO_PARAM_DESC},
            },
            'required': ['file_path'],
        },
        annotations=_TOOL_ANNOTATIONS,
    ),
    Tool(
        name='axon_test_impact',
        description=(
            'Find test files likely affected by code changes by tracing'
            ' callers from changed symbols up to test-file boundaries.\n\n'
            'When to use this instead of Grep/Read: use this before running'
            ' CI to select the minimal relevant test subset, or after a'
            ' refactor to identify which tests need updating. Grep can find'
            ' literal usages; this traces indirect callers through the full'
            ' call graph to surface tests that exercise changed code without'
            ' directly naming it.\n\n'
            'Parameters:\n'
            '  - `diff`: raw `git diff` output (max 100 000 chars). Supply'
            ' this *or* `symbols` - at least one is required.\n'
            '  - `symbols`: explicit list of symbol names to trace from.'
            ' Used when you already know which symbols changed.\n'
            '  - `repo` (optional): target repo slug or path. Diff input'
            ' auto-routes to the owning repo.\n\n'
            'Returns: directly impacted test files, indirectly impacted files'
            ' (via caller graph), and a warnings section listing'
            ' docstring/comment-only hunks that were skipped and test files'
            ' excluded by pytest config.\n\n'
            'Multi-repo posture: routes to the repo that owns the diff files.'
            ' *Cross-repo diffs are refused.*'
        ),
        inputSchema={
            'type': 'object',
            'properties': {
                'diff': {
                    'type': 'string',
                    'description': 'Raw git diff output.',
                    'maxLength': 100000,
                },
                'symbols': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'List of symbol names to check.',
                },
                'repo': {'type': 'string', 'description': _REPO_PARAM_DESC},
            },
        },
        annotations=_TOOL_ANNOTATIONS,
    ),
    Tool(
        name='axon_cycles',
        description=(
            'Detect circular dependencies in the knowledge graph using'
            ' strongly connected component (SCC) analysis. Returns cycle'
            ' groups sorted by size descending.\n\n'
            'When to use this instead of Grep/Read: use this to find import'
            ' or call cycles that prevent clean module separation. Grep'
            ' cannot detect cycles; it sees individual edges, not paths.'
            ' This runs over the entire graph in one call.\n\n'
            'Parameters:\n'
            '  - `min_size`: minimum number of nodes in a cycle to report'
            ' (default 2, minimum 2). Raise to filter noise.\n'
            '  - `repo` (optional): target repo slug or path.\n\n'
            'Returns: cycle groups numbered and sorted by size, each listing'
            ' member symbols with name, label, file, and line. Groups of 5+'
            ' nodes are flagged CRITICAL.\n\n'
            'Multi-repo posture: operates within a single repo per call.'
            ' Stale-major repos are refused.'
        ),
        inputSchema={
            'type': 'object',
            'properties': {
                'min_size': {
                    'type': 'integer',
                    'description': 'Minimum cycle size to report (default 2).',
                    'default': 2,
                    'minimum': 2,
                },
                'repo': {'type': 'string', 'description': _REPO_PARAM_DESC},
            },
        },
        annotations=_TOOL_ANNOTATIONS,
    ),
    Tool(
        name='axon_concurrent_with',
        description=(
            'Find symbols that may execute concurrently with a given symbol'
            ' by tracing outgoing CALLS edges whose dispatch_kind is'
            ' non-direct (thread_executor, process_executor, detached_task,'
            ' enqueued_job, callback_registry).\n\n'
            'When to use this instead of Grep/Read: use this when auditing'
            ' thread safety or async correctness. Grep finds`executor.submit`'
            ' call sites; this returns the actual callees reachable through'
            ' those dispatch edges, grouped by dispatch mechanism, so you'
            ' know which symbols share mutable state and may race.\n\n'
            'Parameters:\n'
            '  - `symbol`: name of the symbol to analyse.\n'
            '  - `depth`: maximum traversal depth (default 3, max'
            f' {MAX_TRAVERSE_DEPTH}).\n'
            '  - `repo` (optional): target repo slug or path.\n\n'
            'Returns: concurrent callees grouped by dispatch kind,'
            ' each entry showing name, label, file, line, and hop depth.'
            ' When no concurrently dispatched callees are found, says so'
            ' explicitly.\n\n'
            'Multi-repo posture: appends a cross-repo footer when the'
            ' symbol also exists in accessible foreign repos.'
            ' Stale-major foreign repos are excluded from footers.'
        ),
        inputSchema={
            'type': 'object',
            'properties': {
                'symbol': {
                    'type': 'string',
                    'description': 'Name of the symbol to analyse.',
                },
                'depth': {
                    'type': 'integer',
                    'description': (
                        f'Maximum traversal depth '
                        f'(default 3, max {MAX_TRAVERSE_DEPTH}).'
                    ),
                    'default': 3,
                    'minimum': 1,
                    'maximum': MAX_TRAVERSE_DEPTH,
                },
                'repo': {'type': 'string', 'description': _REPO_PARAM_DESC},
            },
            'required': ['symbol'],
        },
        annotations=_TOOL_ANNOTATIONS,
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of available Axon tools."""
    return TOOLS


def _dispatch_tool(name: str, arguments: dict, ctx: RepoContext) -> str:
    """Dispatch a tool call to the appropriate handler using *ctx*.

    Args:
        name: MCP tool name.
        arguments: Raw tool arguments dict.
        ctx: Per-call repo context with resolved storage and metadata.

    Returns:
        Formatted handler result string.
    """
    if name == 'axon_list_repos':
        return handle_list_repos(
            resolver=_state.resolver,  # type: ignore[arg-type]
            pool=_state.pool,  # type: ignore[arg-type]
            drift_cache=_state.drift_cache,  # type: ignore[arg-type]
            local_slug=_state.local_slug,
        )
    elif name == 'axon_query':
        return handle_query(
            ctx, arguments.get('query', ''), limit=arguments.get('limit', 20)
        )
    elif name == 'axon_context':
        return handle_context(ctx, arguments.get('symbol', ''))
    elif name == 'axon_impact':
        return handle_impact(
            ctx,
            arguments.get('symbol', ''),
            depth=arguments.get('depth', 3),
            propagate_through=arguments.get('propagate_through'),
        )
    elif name == 'axon_dead_code':
        return handle_dead_code(ctx)
    elif name == 'axon_detect_changes':
        return handle_detect_changes(ctx, arguments.get('diff', ''))
    elif name == 'axon_cypher':
        return handle_cypher(ctx, arguments.get('query', ''))
    elif name == 'axon_coupling':
        return handle_coupling(
            ctx,
            arguments.get('file_path', ''),
            min_strength=arguments.get('min_strength', 0.3),
        )
    elif name == 'axon_communities':
        return handle_communities(ctx, community=arguments.get('community'))
    elif name == 'axon_explain':
        return handle_explain(ctx, arguments.get('symbol', ''))
    elif name == 'axon_review_risk':
        return handle_review_risk(ctx, arguments.get('diff', ''))
    elif name == 'axon_call_path':
        return handle_call_path(
            ctx,
            arguments.get('from_symbol', ''),
            arguments.get('to_symbol', ''),
            max_depth=arguments.get('max_depth', 10),
        )
    elif name == 'axon_file_context':
        return handle_file_context(ctx, arguments.get('file_path', ''))
    elif name == 'axon_test_impact':
        return handle_test_impact(
            ctx,
            diff=arguments.get('diff', ''),
            symbols=arguments.get('symbols'),
        )
    elif name == 'axon_cycles':
        return handle_cycles(ctx, min_size=arguments.get('min_size', 2))
    elif name == 'axon_concurrent_with':
        return handle_concurrent_with(
            ctx, arguments.get('symbol', ''), depth=arguments.get('depth', 3)
        )
    else:
        return f'Unknown tool: {name}'


def _maybe_drift_warning(result: str, ctx: RepoContext) -> str:
    """Prepend a stale-minor drift warning when ctx is a foreign repo.

    No-op for the local repo (the watcher handles staleness there via the
    existing render_with_dead_code_warning/render_with_communities_warning
    mechanisms). No-op when drift_cache is uninitialised or the probe fails.

    Args:
        result: Handler response string.
        ctx: Per-call repo context.

    Returns:
        result unchanged for local repos or non-STALE_MINOR drift levels.
        For foreign STALE_MINOR repos, a warning line is prepended.
    """
    if ctx.is_local or ctx.repo_path is None:
        return result
    drift_cache = _state.drift_cache
    if drift_cache is None:
        return result
    try:
        report = drift_cache.get_or_probe(ctx.repo_path)
        if report.level == DriftLevel.STALE_MINOR:
            decorated = report.__class__(
                level=report.level,
                reason=report.reason,
                last_indexed_at=report.last_indexed_at,
                head_sha=report.head_sha,
                head_sha_at_index=report.head_sha_at_index,
                files_changed_estimate=report.files_changed_estimate,
                files_indexed_estimate=report.files_indexed_estimate,
                watcher_alive=report.watcher_alive,
                tier_used=report.tier_used,
                slug=ctx.slug,
            )
            return render_with_drift_warning(decorated, result)
    except Exception:
        pass
    return result


async def _run_tool_with_fan_out(
    name: str, arguments: dict, ctx: RepoContext
) -> str:
    """Run a tool handler, optionally injecting cross-repo fan-out data.

    For symbol-keyed tools, computes foreign-repo symbol matches before
    invoking the handler and passes them as ``foreign_matches``.  For
    ``axon_query``, computes foreign hit counts similarly.

    All fan-out I/O happens via ``asyncio.to_thread`` to keep blocking
    pool operations off the event loop.

    For foreign repos that are STALE_MINOR, a drift warning is prepended to
    the result by ``_maybe_drift_warning``.

    Args:
        name: MCP tool name.
        arguments: Raw tool arguments dict.
        ctx: Resolved per-call repo context.

    Returns:
        Handler result string with cross-repo footers appended where applicable.
    """
    resolver = _state.resolver
    pool = _state.pool
    drift_cache = _state.drift_cache
    local_slug = _state.local_slug

    if (
        name in _SYMBOL_KEYED_TOOLS
        and resolver is not None
        and pool is not None
        and drift_cache is not None
    ):
        symbol = arguments.get('symbol') or arguments.get('from_symbol', '')

        def _get_foreign_matches() -> list:
            return _foreign_symbol_matches(
                pool, resolver, drift_cache, symbol, exclude_slug=local_slug
            )

        foreign_matches = await asyncio.to_thread(_get_foreign_matches)

        if name == 'axon_context':
            result = handle_context(
                ctx,
                arguments.get('symbol', ''),
                foreign_matches=foreign_matches,
            )
        elif name == 'axon_impact':
            result = handle_impact(
                ctx,
                arguments.get('symbol', ''),
                depth=arguments.get('depth', 3),
                propagate_through=arguments.get('propagate_through'),
                foreign_matches=foreign_matches,
            )
        elif name == 'axon_explain':
            result = handle_explain(
                ctx,
                arguments.get('symbol', ''),
                foreign_matches=foreign_matches,
            )
        elif name == 'axon_call_path':
            result = handle_call_path(
                ctx,
                arguments.get('from_symbol', ''),
                arguments.get('to_symbol', ''),
                max_depth=arguments.get('max_depth', 10),
                foreign_matches=foreign_matches,
            )
        elif name == 'axon_concurrent_with':
            result = handle_concurrent_with(
                ctx,
                arguments.get('symbol', ''),
                depth=arguments.get('depth', 3),
                foreign_matches=foreign_matches,
            )
        else:
            result = _dispatch_tool(name, arguments, ctx)
        return _maybe_drift_warning(result, ctx)

    if (
        name == 'axon_query'
        and resolver is not None
        and pool is not None
        and drift_cache is not None
    ):
        query = arguments.get('query', '')

        def _get_foreign_hits() -> list:
            return _foreign_query_hit_counts(
                pool, resolver, drift_cache, query, exclude_slug=local_slug
            )

        foreign_hits = await asyncio.to_thread(_get_foreign_hits)
        result = handle_query(
            ctx,
            query,
            limit=arguments.get('limit', 20),
            foreign_hits=foreign_hits,
        )
        return _maybe_drift_warning(result, ctx)

    result = _dispatch_tool(name, arguments, ctx)
    return _maybe_drift_warning(result, ctx)


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch a tool call to the appropriate handler."""
    try:
        if name == 'axon_list_repos':
            await _ensure_multi_repo()
            result = await asyncio.to_thread(
                handle_list_repos,
                resolver=_state.resolver,  # type: ignore[arg-type]
                pool=_state.pool,  # type: ignore[arg-type]
                drift_cache=_state.drift_cache,  # type: ignore[arg-type]
                local_slug=_state.local_slug,
            )
            return [TextContent(type='text', text=result)]

        # Build repo context via multi-repo aware resolver.
        ctx_or_refusal = await _build_repo_context(name, arguments)
        if isinstance(ctx_or_refusal, str):
            return [TextContent(type='text', text=ctx_or_refusal)]

        ctx: RepoContext = ctx_or_refusal

        if ctx.storage is None:
            # Standalone axon mcp mode - open a fresh read-only connection
            # for the local repo and re-wrap in a RepoContext.

            if name == 'axon_cypher':
                def _run_readonly() -> str:
                    with _open_storage() as st:
                        real_ctx = RepoContext(
                            storage=st,
                            slug=ctx.slug,
                            is_local=ctx.is_local,
                            repo_path=ctx.repo_path,
                            local_slug=ctx.local_slug,
                        )
                        return _dispatch_tool(name, arguments, real_ctx)

                result = await asyncio.to_thread(_run_readonly)

            else:
                def _run_rw() -> str:
                    with _open_storage() as st:
                        real_ctx = RepoContext(
                            storage=st,
                            slug=ctx.slug,
                            is_local=ctx.is_local,
                            repo_path=ctx.repo_path,
                            local_slug=ctx.local_slug,
                        )
                        return _dispatch_tool(name, arguments, real_ctx)

                result = await asyncio.to_thread(_run_rw)
        elif name == 'axon_cypher':
            # Cypher always runs against a read-only connection for the local
            # repo. For foreign repos the pool already opens read-only; for
            # the injected writer we open a fresh read-only connection.

            if ctx.is_local and _state.storage is not None:
                def _run_cypher_local() -> str:
                    with _open_storage() as st:
                        real_ctx = RepoContext(
                            storage=st,
                            slug=ctx.slug,
                            is_local=ctx.is_local,
                            repo_path=ctx.repo_path,
                            local_slug=ctx.local_slug,
                        )
                        return handle_cypher(
                            real_ctx, arguments.get('query', '')
                        )

                result = await asyncio.to_thread(_run_cypher_local)
            else:
                if ctx.is_local and _state.lock is not None:
                    async with _state.lock:
                        result = await asyncio.to_thread(
                            _dispatch_tool, name, arguments, ctx
                        )
                else:
                    result = await asyncio.to_thread(
                        _dispatch_tool, name, arguments, ctx
                    )
        else:
            if ctx.is_local and _state.lock is not None:
                async with _state.lock:
                    result = await _run_tool_with_fan_out(name, arguments, ctx)
            else:
                result = await _run_tool_with_fan_out(name, arguments, ctx)

    except Exception:
        ref = _new_ref_id()
        logger.exception(
            'Tool %s raised an unhandled exception', name, extra={'ref': ref}
        )
        result = f'Internal error (ref {ref}); see server logs.'

    return [TextContent(type='text', text=result)]


@server.list_resources()
async def list_resources() -> list[Resource]:
    """Return the list of available Axon resources."""
    return [
        Resource(
            uri="axon://overview",
            name="Codebase Overview",
            description="High-level statistics about the indexed codebase.",
            mimeType="text/plain",
        ),
        Resource(
            uri="axon://dead-code",
            name="Dead Code Report",
            description="List of all symbols flagged as unreachable.",
            mimeType="text/plain",
        ),
        Resource(
            uri="axon://schema",
            name="Graph Schema",
            description="Description of the Axon knowledge graph schema.",
            mimeType="text/plain",
        ),
    ]


def _dispatch_resource(uri_str: str, storage: KuzuBackend) -> str:
    if uri_str == "axon://overview":
        return get_overview(storage)
    if uri_str == "axon://dead-code":
        return get_dead_code_list(storage)
    if uri_str == "axon://schema":
        return get_schema()
    return f"Unknown resource: {uri_str}"


@server.read_resource()
async def read_resource(uri) -> str:
    """Read the contents of an Axon resource."""
    uri_str = str(uri)
    return await _with_storage(lambda st: _dispatch_resource(uri_str, st))


async def main() -> None:
    """Run the Axon MCP server over stdio transport."""
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def create_streamable_http_app() -> tuple[StreamableHTTPSessionManager, StreamableHTTPASGIApp]:
    """Create a streamable HTTP transport for the existing MCP server."""
    session_manager = StreamableHTTPSessionManager(app=server)
    return session_manager, StreamableHTTPASGIApp(session_manager)


if __name__ == "__main__":
    asyncio.run(main())
