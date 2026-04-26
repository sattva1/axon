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
from mcp.types import Resource, TextContent, Tool

from axon.core.drift import DriftCache, DriftLevel
from axon.core.repos import RepoNotFound, RepoPool, RepoResolver
from axon.core.storage.kuzu_backend import KuzuBackend
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


TOOLS: list[Tool] = [
    Tool(
        name='axon_list_repos',
        description='List all indexed repositories with their stats.',
        inputSchema={'type': 'object', 'properties': {}},
    ),
    Tool(
        name='axon_query',
        description=(
            'Search the knowledge graph using hybrid (keyword + vector) search. '
            'Returns ranked symbols matching the query.'
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
    ),
    Tool(
        name='axon_context',
        description=(
            'Get a 360-degree view of a symbol: callers, callees, type references, '
            'and community membership.'
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
    ),
    Tool(
        name='axon_impact',
        description=(
            'Blast radius analysis: find all symbols affected by changing a given symbol.'
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
                        f'Maximum traversal depth (default 3, max {MAX_TRAVERSE_DEPTH}).'
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
    ),
    Tool(
        name='axon_dead_code',
        description='List all symbols detected as dead (unreachable) code.',
        inputSchema={
            'type': 'object',
            'properties': {
                'repo': {'type': 'string', 'description': _REPO_PARAM_DESC}
            },
        },
    ),
    Tool(
        name='axon_detect_changes',
        description=(
            'Parse a git diff and map changed files/lines to affected symbols '
            'in the knowledge graph.'
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
    ),
    Tool(
        name='axon_cypher',
        description='Execute a raw Cypher query against the knowledge graph.',
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
    ),
    Tool(
        name='axon_coupling',
        description=(
            'Show files temporally coupled with a given file. '
            'Reveals hidden dependencies from git co-change patterns.'
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
                    'description': 'Minimum coupling strength threshold (default 0.3).',
                    'default': 0.3,
                },
                'repo': {'type': 'string', 'description': _REPO_PARAM_DESC},
            },
            'required': ['file_path'],
        },
    ),
    Tool(
        name='axon_communities',
        description=(
            'List detected code communities (Leiden clusters) or drill into '
            'a specific community to see its members.'
        ),
        inputSchema={
            'type': 'object',
            'properties': {
                'community': {
                    'type': 'string',
                    'description': 'Optional community name to drill into. Omit to list all.',
                },
                'repo': {'type': 'string', 'description': _REPO_PARAM_DESC},
            },
        },
    ),
    Tool(
        name='axon_explain',
        description=(
            'Get a narrative explanation of a symbol: its role, community, '
            'process flows, and relationships summarized for onboarding.'
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
    ),
    Tool(
        name='axon_review_risk',
        description=(
            'PR risk assessment: analyzes a git diff to find affected symbols, '
            'missing co-change files, community boundary crossings, and '
            'downstream blast radius. Returns a risk score.'
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
    ),
    Tool(
        name='axon_call_path',
        description=(
            'Find the shortest call chain between two symbols. '
            'Uses BFS over CALLS edges.'
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
                    'description': f'Maximum hops (default 10, max {MAX_TRAVERSE_DEPTH}).',
                    'default': 10,
                    'minimum': 1,
                    'maximum': MAX_TRAVERSE_DEPTH,
                },
                'repo': {'type': 'string', 'description': _REPO_PARAM_DESC},
            },
            'required': ['from_symbol', 'to_symbol'],
        },
    ),
    Tool(
        name='axon_file_context',
        description=(
            'Get comprehensive context for a file: symbols, imports, '
            'coupling, dead code, and community membership in one call.'
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
    ),
    Tool(
        name='axon_test_impact',
        description=(
            'Find tests likely affected by code changes. Accepts a git diff '
            'or symbol names, traces callers to find test files.'
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
    ),
    Tool(
        name='axon_cycles',
        description=(
            'Detect circular dependencies using strongly connected '
            'component analysis. Returns cycle groups sorted by size.'
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
    ),
    Tool(
        name='axon_concurrent_with',
        description=(
            'Find symbols that may run concurrently with the given symbol. '
            'Traces dispatch edges (thread/process executors, asyncio tasks, '
            'Celery enqueued jobs, etc.) and returns reachable callbacks.'
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
        return handle_list_repos()
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


async def _run_tool_with_fan_out(
    name: str, arguments: dict, ctx: RepoContext
) -> str:
    """Run a tool handler, optionally injecting cross-repo fan-out data.

    For symbol-keyed tools, computes foreign-repo symbol matches before
    invoking the handler and passes them as ``foreign_matches``.  For
    ``axon_query``, computes foreign hit counts similarly.

    All fan-out I/O happens via ``asyncio.to_thread`` to keep blocking
    pool operations off the event loop.

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
            return handle_context(
                ctx,
                arguments.get('symbol', ''),
                foreign_matches=foreign_matches,
            )
        elif name == 'axon_impact':
            return handle_impact(
                ctx,
                arguments.get('symbol', ''),
                depth=arguments.get('depth', 3),
                propagate_through=arguments.get('propagate_through'),
                foreign_matches=foreign_matches,
            )
        elif name == 'axon_explain':
            return handle_explain(
                ctx,
                arguments.get('symbol', ''),
                foreign_matches=foreign_matches,
            )
        elif name == 'axon_call_path':
            return handle_call_path(
                ctx,
                arguments.get('from_symbol', ''),
                arguments.get('to_symbol', ''),
                max_depth=arguments.get('max_depth', 10),
                foreign_matches=foreign_matches,
            )
        elif name == 'axon_concurrent_with':
            return handle_concurrent_with(
                ctx,
                arguments.get('symbol', ''),
                depth=arguments.get('depth', 3),
                foreign_matches=foreign_matches,
            )

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
        return handle_query(
            ctx,
            query,
            limit=arguments.get('limit', 20),
            foreign_hits=foreign_hits,
        )

    return _dispatch_tool(name, arguments, ctx)


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch a tool call to the appropriate handler."""
    try:
        if name == 'axon_list_repos':
            result = handle_list_repos()
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
