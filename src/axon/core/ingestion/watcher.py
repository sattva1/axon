"""Watch mode for Axon — re-indexes on file changes.

Uses ``watchfiles`` (Rust-backed) for efficient file system monitoring with
native debouncing.  Changes are processed in tiers:

- **File-local** (immediate): Phases 2-7 on changed files only.
- **Global** (after quiet period): Hydrate graph from storage, run
  communities/processes/dead-code on the full graph.
- **Embeddings** (with global): Re-embed dirty nodes + CALLS neighbors.
- **Coupling** (commit-triggered): Re-run git coupling when HEAD changes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import watchfiles
from watchfiles import Change

from axon.config.ignore import load_gitignore, should_ignore
from axon.config.languages import is_supported
from axon.core.drift import compute_drift_inputs, _get_head_sha
from axon.core.embeddings.embedder import (
    _DEFAULT_MODEL,
    embed_graph,
    embed_nodes,
)
from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import NodeLabel, RelType
from axon.core.ingestion.community import process_communities
from axon.core.ingestion.coupling import process_coupling
from axon.core.ingestion.dead_code import process_dead_code
from axon.core.ingestion.pipeline import reindex_files
from axon.core.ingestion.processes import process_processes
from axon.core.ingestion.walker import FileEntry, read_file
from axon.core.meta import load_meta, now_iso, update_meta
from axon.core.storage.base import EMBEDDING_DIMENSIONS, StorageBackend

logger = logging.getLogger(__name__)

# Debounce: global phases fire after this many seconds of no file changes.
QUIET_PERIOD = 5.0

# Maximum time dirty files can accumulate before forcing a global phase,
# even if changes keep arriving (prevents starvation under continuous writes).
MAX_DIRTY_AGE = 60.0

# How often watchfiles yields (controls quiet-period check granularity).
_POLL_INTERVAL_MS = 500


def ensure_current_embeddings(
    storage: StorageBackend, repo_path: Path
) -> bool:
    """Re-embed the full graph when the stored embedding model is outdated."""
    meta = load_meta(repo_path)
    if not meta.embedding_model:
        return False

    if meta.embedding_model == _DEFAULT_MODEL:
        return False

    logger.info(
        'Embedding model changed from %s to %s, re-embedding all symbols',
        meta.embedding_model,
        _DEFAULT_MODEL,
    )

    try:
        graph = storage.load_graph()
        embeddings = embed_graph(graph)
        if embeddings:
            storage.store_embeddings(embeddings)

        update_meta(
            repo_path,
            embedding_model=_DEFAULT_MODEL,
            embedding_dimensions=EMBEDDING_DIMENSIONS,
        )
        return True
    except Exception:
        logger.warning('Full re-embedding failed', exc_info=True)
        return False


def _reindex_files(
    changes: list[tuple[Change, Path]],
    repo_path: Path,
    storage: StorageBackend,
    gitignore_patterns: list[str] | None = None,
) -> tuple[int, set[str]]:
    """Re-index changed files through file-local phases.

    Returns (count_reindexed, set_of_relative_file_paths_reindexed).
    """
    entries: list[FileEntry] = []
    reindexed_paths: set[str] = set()

    for change_type, abs_path in changes:
        if change_type == Change.deleted and not abs_path.is_file():
            try:
                relative = str(abs_path.relative_to(repo_path))
                storage.remove_nodes_by_file(relative)
                reindexed_paths.add(relative)
            except (ValueError, OSError):
                pass
            continue

        if not abs_path.is_file():
            # Transient atomic-save window: path temporarily absent for
            # non-deletion events. The next watchfiles cycle resolves it.
            logger.debug('Skipping temporarily absent path: %s', abs_path)
            continue

        try:
            relative = str(abs_path.relative_to(repo_path))
        except ValueError:
            continue

        if should_ignore(relative, gitignore_patterns):
            continue

        if not is_supported(abs_path):
            continue

        entry = read_file(repo_path, abs_path)
        if entry is not None:
            entries.append(entry)
            reindexed_paths.add(relative)

    if entries:
        reindex_files(entries, repo_path, storage, rebuild_fts=False)
        update_meta(repo_path, last_incremental_at=now_iso())

    return len(reindexed_paths), reindexed_paths


def _compute_dirty_node_ids(graph: KnowledgeGraph, dirty_files: set[str]) -> set[str]:
    """Find all node IDs in dirty files + their immediate CALLS neighbors."""
    if not dirty_files:
        return set()

    dirty_node_ids = {n.id for n in graph.iter_nodes() if n.file_path in dirty_files}

    neighbor_ids: set[str] = set()
    for node_id in dirty_node_ids:
        for rel in graph.get_outgoing(node_id, RelType.CALLS):
            neighbor_ids.add(rel.target)
        for rel in graph.get_incoming(node_id, RelType.CALLS):
            neighbor_ids.add(rel.source)

    return dirty_node_ids | neighbor_ids


_SMALL_CHANGE_THRESHOLD = 3


def _run_full_global_phases(repo_path: Path, storage: StorageBackend) -> None:
    """Run the full set of global phases (communities, processes, dead code).

    Called by both the full-global branch of _run_incremental_global_phases
    and the periodic-refresh scheduler inside watch_repo. Callers are
    responsible for calling storage.rebuild_fts_indexes() afterwards so that
    any post-call work (e.g., coupling) is also covered by the rebuild.
    """
    storage.delete_synthetic_nodes()

    logger.info('Hydrating graph from storage...')
    graph = storage.load_graph()

    num_communities = process_communities(graph)
    logger.info('Communities: %d', num_communities)
    update_meta(repo_path, communities_last_refreshed_at=now_iso())

    num_processes = process_processes(graph)
    logger.info('Processes: %d', num_processes)

    num_dead = process_dead_code(graph)
    logger.info('Dead code: %d', num_dead)
    update_meta(repo_path, dead_code_last_refreshed_at=now_iso())

    new_nodes = list(graph.get_nodes_by_label(NodeLabel.COMMUNITY)) + list(
        graph.get_nodes_by_label(NodeLabel.PROCESS)
    )
    new_rels = list(graph.get_relationships_by_type(RelType.MEMBER_OF)) + list(
        graph.get_relationships_by_type(RelType.STEP_IN_PROCESS)
    )
    if new_nodes:
        storage.add_nodes(new_nodes)
    if new_rels:
        storage.add_relationships(new_rels)

    dead_ids = {n.id for n in graph.iter_nodes() if n.is_dead}
    alive_ids = {n.id for n in graph.iter_nodes() if not n.is_dead}
    storage.update_dead_flags(dead_ids, alive_ids)


def _run_incremental_global_phases(
    storage: StorageBackend,
    repo_path: Path,
    dirty_files: set[str],
    run_coupling: bool = False,
) -> None:
    """Run global phases incrementally using graph hydrated from storage.

    When the change is small (< _SMALL_CHANGE_THRESHOLD files), only dead code
    detection runs (communities and processes are expensive and unlikely to
    shift from a 1-2 file change).
    """
    small_change = len(dirty_files) < _SMALL_CHANGE_THRESHOLD

    if not small_change:
        _run_full_global_phases(repo_path, storage)
    else:
        logger.info(
            'Small change (%d files) — skipping communities/processes',
            len(dirty_files),
        )

        logger.info('Hydrating graph from storage...')
        graph = storage.load_graph()

        num_dead = process_dead_code(graph)
        logger.info('Dead code: %d', num_dead)
        update_meta(repo_path, dead_code_last_refreshed_at=now_iso())

        dead_ids = {n.id for n in graph.iter_nodes() if n.is_dead}
        alive_ids = {n.id for n in graph.iter_nodes() if not n.is_dead}
        storage.update_dead_flags(dead_ids, alive_ids)

    if run_coupling:
        graph = storage.load_graph()
        storage.remove_relationships_by_type(RelType.COUPLED_WITH)
        num_coupled = process_coupling(graph, repo_path)
        coupled_rels = list(graph.get_relationships_by_type(RelType.COUPLED_WITH))
        if coupled_rels:
            storage.add_relationships(coupled_rels)
        logger.info("Coupling: %d pairs", num_coupled)

    if not ensure_current_embeddings(storage, repo_path):
        graph = storage.load_graph()
        dirty_node_ids = _compute_dirty_node_ids(graph, dirty_files)
        if dirty_node_ids:
            logger.info("Re-embedding %d nodes...", len(dirty_node_ids))
            try:
                embeddings = embed_nodes(graph, dirty_node_ids)
                if embeddings:
                    storage.upsert_embeddings(embeddings)
            except Exception:
                logger.warning("Incremental embedding failed", exc_info=True)

    storage.rebuild_fts_indexes()

    logger.info("Incremental global phases complete.")


async def watch_repo(
    repo_path: Path,
    storage: StorageBackend,
    *,
    stop_event: asyncio.Event | None = None,
    lock: asyncio.Lock | None = None,
    global_refresh_interval_seconds: int | None = None,
) -> None:
    """Main watch loop — monitor files and re-index on changes.

    File-local reindex runs immediately on every change. Global phases
    (communities, processes, dead code, embeddings) run after a quiet
    period of QUIET_PERIOD seconds with no new changes. Coupling runs
    only when new git commits are detected.

    When global_refresh_interval_seconds is set, a background task
    periodically runs the full global phases regardless of file changes.
    """
    async def _run_sync(fn, *args):
        if lock is not None:
            async with lock:
                return await asyncio.to_thread(fn, *args)
        return await asyncio.to_thread(fn, *args)

    gitignore = load_gitignore(repo_path)
    dirty_files: set[str] = set()
    last_change_time: float = 0.0
    first_dirty_time: float = 0.0
    global_lock = asyncio.Lock()
    last_known_commit = _get_head_sha(repo_path)

    refresh_task: asyncio.Task[None] | None = None

    if global_refresh_interval_seconds and global_refresh_interval_seconds > 0:
        async def _periodic_refresh() -> None:
            while True:
                await asyncio.sleep(global_refresh_interval_seconds)
                async with global_lock:
                    await asyncio.to_thread(
                        _run_full_global_phases, repo_path, storage
                    )

        refresh_task = asyncio.create_task(_periodic_refresh())

    logger.info('Watching %s for changes...', repo_path)

    try:
        async for changes in watchfiles.awatch(
            repo_path,
            rust_timeout=_POLL_INTERVAL_MS,
            yield_on_timeout=True,
            stop_event=stop_event,
        ):
            # Dedup by path string; last event type for a given path wins.
            path_to_change: dict[str, Change] = {}
            for change_type, path_str in changes:
                path_to_change[path_str] = change_type
            changed: list[tuple[Change, Path]] = [
                (ct, Path(ps)) for ps, ct in path_to_change.items()
            ]

            if changed:
                count, reindexed = await _run_sync(
                    _reindex_files, changed, repo_path, storage, gitignore
                )
                if reindexed:
                    dirty_files |= reindexed
                    last_change_time = time.monotonic()
                    if first_dirty_time == 0.0:
                        first_dirty_time = last_change_time
                    logger.info(
                        'Reindexed %d file(s), %d paths dirty',
                        count,
                        len(reindexed),
                    )

            now = time.monotonic()
            quiet_elapsed = (
                last_change_time > 0
                and (now - last_change_time) >= QUIET_PERIOD
            )
            starvation = (
                first_dirty_time > 0
                and (now - first_dirty_time) >= MAX_DIRTY_AGE
            )
            # global_lock.locked() is safe here: single event loop, no await between
            # check and acquire.
            if (
                dirty_files
                and not global_lock.locked()
                and (quiet_elapsed or starvation)
            ):
                snapshot = dirty_files.copy()
                dirty_files.clear()
                first_dirty_time = 0.0

                current_commit = await asyncio.to_thread(
                    _get_head_sha, repo_path
                )
                run_coupling = current_commit != last_known_commit
                if run_coupling:
                    last_known_commit = current_commit

                try:
                    async with global_lock:
                        logger.info('Running incremental global phases...')
                        await _run_sync(
                            _run_incremental_global_phases,
                            storage,
                            repo_path,
                            snapshot,
                            run_coupling,
                        )
                        if run_coupling:
                            await asyncio.to_thread(
                                lambda: update_meta(
                                    repo_path,
                                    **compute_drift_inputs(
                                        repo_path,
                                        list(storage.get_file_index().keys()),
                                    ),
                                )
                            )
                except Exception:
                    logger.exception(
                        'Global phases failed; re-queueing dirty files'
                    )
                    dirty_files |= snapshot
                    last_change_time = time.monotonic()
    finally:
        if refresh_task is not None:
            refresh_task.cancel()
            try:
                await refresh_task
            except asyncio.CancelledError:
                pass

    logger.info('Watch stopped.')
