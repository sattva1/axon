"""Public reindex helpers shared between the flush coordinator and the watcher.

Extracted from watcher.py so the flush coordinator (watcher_flush.py) can
import them without crossing private-name boundaries.
"""

from __future__ import annotations

import logging
from pathlib import Path

from watchfiles import Change

from axon.config.ignore import should_ignore
from axon.config.languages import is_supported
from axon.core.drift import _get_head_sha
from axon.core.embeddings.embedder import _DEFAULT_MODEL, embed_graph, embed_nodes
from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import NodeLabel, RelType
from axon.core.ingestion.community import process_communities
from axon.core.ingestion.coupling import process_coupling
from axon.core.ingestion.dead_code import process_dead_code
from axon.core.ingestion.pipeline import reindex_files as _pipeline_reindex_files
from axon.core.ingestion.processes import process_processes
from axon.core.ingestion.walker import FileEntry, read_file
from axon.core.meta import load_meta, now_iso, update_meta
from axon.core.storage.base import EMBEDDING_DIMENSIONS, StorageBackend

logger = logging.getLogger(__name__)

_SMALL_CHANGE_THRESHOLD = 3

# Maximum time dirty files can accumulate before forcing a global phase
# (starvation / fairness bound). Moved to watcher_flush.py as the SSOT;
# kept here for backward compatibility with any callers that imported it
# from this module.
MAX_DIRTY_AGE: float = 60.0


def get_head_sha(repo_path: Path) -> str | None:
    """Return the current git HEAD sha, or None when not in a git repo."""
    return _get_head_sha(repo_path)


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


def reindex_files(
    changes: list[tuple[Change, Path]],
    repo_path: Path,
    storage: StorageBackend,
    gitignore_patterns: list[str] | None = None,
) -> tuple[int, set[str]]:
    """Re-index changed files through file-local phases.

    NOTE: this function does NOT write last_incremental_at. The flush
    coordinator owns the single authoritative write per flush (Major #1).

    Args:
        changes: List of (change_type, abs_path) pairs from the watcher.
        repo_path: Root of the repository.
        storage: Initialised writable storage backend.
        gitignore_patterns: Patterns to skip.

    Returns:
        (count_reindexed, set_of_relative_file_paths_reindexed).
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
        _pipeline_reindex_files(entries, repo_path, storage, rebuild_fts=False)

    return len(reindexed_paths), reindexed_paths


def compute_dirty_node_ids(
    graph: KnowledgeGraph, dirty_files: set[str]
) -> set[str]:
    """Find all node IDs in dirty files plus their immediate CALLS neighbors."""
    if not dirty_files:
        return set()

    dirty_node_ids = {
        n.id for n in graph.iter_nodes() if n.file_path in dirty_files
    }

    neighbor_ids: set[str] = set()
    for node_id in dirty_node_ids:
        for rel in graph.get_outgoing(node_id, RelType.CALLS):
            neighbor_ids.add(rel.target)
        for rel in graph.get_incoming(node_id, RelType.CALLS):
            neighbor_ids.add(rel.source)

    return dirty_node_ids | neighbor_ids


def run_full_global_phases(
    repo_path: Path, storage: StorageBackend
) -> None:
    """Run the full set of global phases (communities, processes, dead code).

    Callers are responsible for calling storage.rebuild_fts_indexes() afterwards
    so that any post-call work (e.g., coupling) is also covered by the rebuild.
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

    new_nodes = list(
        graph.get_nodes_by_label(NodeLabel.COMMUNITY)
    ) + list(graph.get_nodes_by_label(NodeLabel.PROCESS))
    new_rels = list(
        graph.get_relationships_by_type(RelType.MEMBER_OF)
    ) + list(graph.get_relationships_by_type(RelType.STEP_IN_PROCESS))
    if new_nodes:
        storage.add_nodes(new_nodes)
    if new_rels:
        storage.add_relationships(new_rels)

    dead_ids = {n.id for n in graph.iter_nodes() if n.is_dead}
    alive_ids = {n.id for n in graph.iter_nodes() if not n.is_dead}
    storage.update_dead_flags(dead_ids, alive_ids)


def run_incremental_global_phases(
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
        run_full_global_phases(repo_path, storage)
    else:
        logger.info(
            'Small change (%d files) -- skipping communities/processes',
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
        coupled_rels = list(
            graph.get_relationships_by_type(RelType.COUPLED_WITH)
        )
        if coupled_rels:
            storage.add_relationships(coupled_rels)
        logger.info('Coupling: %d pairs', num_coupled)

    if not ensure_current_embeddings(storage, repo_path):
        graph = storage.load_graph()
        dirty_node_ids = compute_dirty_node_ids(graph, dirty_files)
        if dirty_node_ids:
            logger.info('Re-embedding %d nodes...', len(dirty_node_ids))
            try:
                embeddings = embed_nodes(graph, dirty_node_ids)
                if embeddings:
                    storage.upsert_embeddings(embeddings)
            except Exception:
                logger.warning('Incremental embedding failed', exc_info=True)

    storage.rebuild_fts_indexes()

    logger.info('Incremental global phases complete.')
