"""Phase 4: Import resolution for Axon.

Takes the FileParseData produced by the parsing phase and resolves import
statements to actual File nodes in the knowledge graph, creating IMPORTS
relationships between the importing file and the target file.
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.ingestion.resolved import ResolvedEdge
from axon.core.parsers.base import ImportInfo

logger = logging.getLogger(__name__)

_JS_TS_EXTENSIONS = (".ts", ".js", ".tsx", ".jsx")

def build_file_index(graph: KnowledgeGraph) -> dict[str, str]:
    """Build an index mapping file paths to their graph node IDs.

    Iterates over all :pyclass:`NodeLabel.FILE` nodes in the graph and
    returns a dict keyed by ``file_path`` with node ``id`` as value.

    Args:
        graph: The knowledge graph containing File nodes.

    Returns:
        A dict like ``{"src/auth/validate.py": "file:src/auth/validate.py:"}``.
    """
    file_nodes = graph.get_nodes_by_label(NodeLabel.FILE)
    return {node.file_path: node.id for node in file_nodes}

def _detect_source_roots(file_index: dict[str, str]) -> set[str]:
    """Detect Python source root directories from the file index.

    A source root is a directory that contains a top-level Python package
    (a directory with ``__init__.py`` whose parent does NOT have
    ``__init__.py``).  For a ``src/`` layout like ``src/mypackage/__init__.py``
    where ``src/__init__.py`` does not exist, ``src`` is the source root.

    Limitation: nested namespace packages (PEP 420) without ``__init__.py``
    at intermediate levels are not detected. This heuristic assumes each
    top-level package has an ``__init__.py``.

    Returns:
        A set of source root prefixes (e.g. ``{"src"}``).
    """
    init_dirs: set[str] = set()
    for path in file_index:
        if path.endswith("/__init__.py"):
            init_dirs.add(str(PurePosixPath(path).parent))

    roots: set[str] = set()
    for d in init_dirs:
        parent = str(PurePosixPath(d).parent)
        if parent != "." and parent not in init_dirs:
            roots.add(parent)
    return roots


def resolve_import_path(
    importing_file: str,
    import_info: ImportInfo,
    file_index: dict[str, str],
    source_roots: set[str] | None = None,
) -> str | None:
    """Resolve an import statement to the target file's node ID.

    Uses the importing file's path, the parsed :class:`ImportInfo`, and the
    index of all known project files to determine which file is being
    imported.  Returns ``None`` for external/unresolvable imports.

    Args:
        importing_file: Relative path of the file containing the import
            (e.g. ``"src/auth/validate.py"``).
        import_info: The parsed import information.
        file_index: Mapping of relative file paths to their graph node IDs.
        source_roots: Optional set of source root prefixes discovered by
            :func:`_detect_source_roots`.  When provided, absolute Python
            imports are also tried with each prefix prepended.

    Returns:
        The node ID of the resolved target file, or ``None`` if the import
        cannot be resolved to a file in the project.
    """
    language = _detect_language(importing_file)

    if language == "python":
        return _resolve_python(importing_file, import_info, file_index, source_roots)
    if language in ("typescript", "javascript"):
        return _resolve_js_ts(importing_file, import_info, file_index)

    return None

def resolve_file_imports(
    fpd: FileParseData,
    file_index: dict[str, str],
    source_roots: set[str],
) -> list[ResolvedEdge]:
    """Resolve imports for a single file — pure read, no graph mutation.

    Returns one :class:`ResolvedEdge` per unique ``(source, target)`` pair
    with per-file merged symbols.  Cross-file symbol merging happens in the
    caller.
    """
    source_file_id = generate_id(NodeLabel.FILE, fpd.file_path)
    # Per-file dedup: merge symbols for same (source, target) pair.
    pair_symbols: dict[str, set[str]] = {}

    for imp in fpd.parse_result.imports:
        target_id = resolve_import_path(fpd.file_path, imp, file_index, source_roots)
        if target_id is None:
            continue
        if target_id not in pair_symbols:
            pair_symbols[target_id] = set()
        pair_symbols[target_id].update(imp.names)

    edges: list[ResolvedEdge] = []
    for target_id, symbols in pair_symbols.items():
        rel_id = f"imports:{source_file_id}->{target_id}"
        edges.append(ResolvedEdge(
            rel_id=rel_id,
            rel_type=RelType.IMPORTS,
            source=source_file_id,
            target=target_id,
            properties={"symbols": symbols},
        ))
    return edges


def _write_import_edges(
    all_edges: list[list[ResolvedEdge]],
    graph: KnowledgeGraph,
) -> None:
    """Merge cross-file symbol sets and write IMPORTS edges to the graph."""
    merged: dict[str, tuple[str, str, set[str]]] = {}
    for file_edges in all_edges:
        for edge in file_edges:
            if edge.rel_id in merged:
                merged[edge.rel_id][2].update(edge.properties["symbols"])
            else:
                merged[edge.rel_id] = (edge.source, edge.target, set(edge.properties["symbols"]))

    for rel_id, (source, target, symbols) in merged.items():
        graph.add_relationship(
            GraphRelationship(
                id=rel_id,
                type=RelType.IMPORTS,
                source=source,
                target=target,
                properties={"symbols": ",".join(sorted(symbols))},
            )
        )


def process_imports(
    parse_data: list[FileParseData],
    graph: KnowledgeGraph,
    *,
    parallel: bool = False,
    collect: bool = False,
) -> list[ResolvedEdge] | None:
    """Resolve imports and create IMPORTS relationships in the graph.

    For each file's parsed imports, resolves the target file and creates
    an ``IMPORTS`` relationship from the importing file node to the target
    file node.  Duplicate edges (same source -> same target) are skipped.

    Args:
        parse_data: Parse results from the parsing phase.
        graph: The knowledge graph to populate with IMPORTS relationships.
        parallel: When ``True``, resolve files in parallel using threads.
        collect: When ``True``, return flat list of edges instead of writing.
    """
    file_index = build_file_index(graph)
    source_roots = _detect_source_roots(file_index)

    if parallel:
        import os
        from concurrent.futures import ThreadPoolExecutor

        workers = min(os.cpu_count() or 4, 8, len(parse_data))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            all_edges = list(pool.map(
                lambda fpd: resolve_file_imports(fpd, file_index, source_roots),
                parse_data,
            ))
    else:
        all_edges = [resolve_file_imports(fpd, file_index, source_roots) for fpd in parse_data]

    if collect:
        return [edge for file_edges in all_edges for edge in file_edges]

    _write_import_edges(all_edges, graph)
    return None

def _detect_language(file_path: str) -> str:
    """Infer language from a file's extension."""
    suffix = PurePosixPath(file_path).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in (".ts", ".tsx"):
        return "typescript"
    if suffix in (".js", ".jsx"):
        return "javascript"
    return ""

def _resolve_python(
    importing_file: str,
    import_info: ImportInfo,
    file_index: dict[str, str],
    source_roots: set[str] | None = None,
) -> str | None:
    """Resolve a Python import to a file node ID.

    Handles:
    - Relative imports (``is_relative=True``): dot-prefixed module paths
      resolved relative to the importing file's directory.
    - Absolute imports: treated as dotted paths from the project root,
      with source root prefix discovery for ``src/`` layouts.

    Returns ``None`` for external (not in file_index) imports.
    """
    if import_info.is_relative:
        return _resolve_python_relative(importing_file, import_info, file_index)
    return _resolve_python_absolute(import_info, file_index, source_roots)

def _resolve_python_relative(
    importing_file: str,
    import_info: ImportInfo,
    file_index: dict[str, str],
) -> str | None:
    """Resolve a relative Python import (``from .foo import bar``).

    The number of leading dots determines how many directory levels to
    traverse upward from the importing file's parent directory.

    ``from .utils import helper``  -> one dot  -> same directory
    ``from ..models import User``  -> two dots -> parent directory
    """
    module = import_info.module
    # Contract: ImportInfo.module includes leading dots for relative imports.
    # e.g. ".utils" -> 1 dot (same dir), "..models" -> 2 dots (parent dir).
    assert module.startswith("."), f"Expected relative import, got {module!r}"

    dot_count = 0
    for ch in module:
        if ch == ".":
            dot_count += 1
        else:
            break

    remainder = module[dot_count:]

    base = PurePosixPath(importing_file).parent
    for _ in range(dot_count - 1):
        base = base.parent

    if remainder:
        segments = remainder.split(".")
        target_dir = base / PurePosixPath(*segments)
    else:
        target_dir = base

    return _try_python_paths(str(target_dir), file_index)

def _resolve_python_absolute(
    import_info: ImportInfo,
    file_index: dict[str, str],
    source_roots: set[str] | None = None,
) -> str | None:
    """Resolve an absolute Python import (``from mypackage.auth import validate``).

    Converts the dotted module path to a filesystem path and looks it up
    in the file index.  For ``src/`` layout projects where file paths have
    a prefix not present in the module name (e.g. ``src/mypackage/...``),
    also tries each discovered source root prefix.

    Returns ``None`` for external packages not present in the project.
    """
    module = import_info.module
    segments = module.split(".")
    target_path = str(PurePosixPath(*segments))

    # Try direct match first (flat layout).
    result = _try_python_paths(target_path, file_index)
    if result:
        return result

    # Try with each source root prefix (src/ layout).
    if source_roots:
        for root in source_roots:
            result = _try_python_paths(f"{root}/{target_path}", file_index)
            if result:
                return result

    return None

def _try_python_paths(base_path: str, file_index: dict[str, str]) -> str | None:
    """Try common Python file resolution patterns for *base_path*.

    Checks in order:
    1. ``base_path.py`` (direct module file)
    2. ``base_path/__init__.py`` (package directory)
    """
    candidates = [
        f"{base_path}.py",
        f"{base_path}/__init__.py",
    ]
    for candidate in candidates:
        if candidate in file_index:
            return file_index[candidate]
    return None

def _resolve_js_ts(
    importing_file: str,
    import_info: ImportInfo,
    file_index: dict[str, str],
) -> str | None:
    """Resolve a JavaScript/TypeScript import to a file node ID.

    Relative imports (starting with ``./`` or ``../``) are resolved against
    the importing file's directory.  Bare specifiers (e.g. ``'express'``)
    are treated as external and return ``None``.
    """
    module = import_info.module

    if not module.startswith("."):
        return None

    base = PurePosixPath(importing_file).parent
    resolved = base / module

    resolved_str = str(PurePosixPath(*resolved.parts))

    return _try_js_ts_paths(resolved_str, file_index)

def _try_js_ts_paths(base_path: str, file_index: dict[str, str]) -> str | None:
    """Try common JS/TS file resolution patterns for *base_path*.

    Checks in order:
    1. ``base_path`` as-is (already has extension)
    2. ``base_path`` + each known extension (.ts, .js, .tsx, .jsx)
    3. ``base_path/index`` + each known extension
    """
    # 1. Exact match (import already includes extension).
    if base_path in file_index:
        return file_index[base_path]

    # 2. Try appending extensions.
    for ext in _JS_TS_EXTENSIONS:
        candidate = f"{base_path}{ext}"
        if candidate in file_index:
            return file_index[candidate]

    # 3. Try as directory with index file.
    for ext in _JS_TS_EXTENSIONS:
        candidate = f"{base_path}/index{ext}"
        if candidate in file_index:
            return file_index[candidate]

    return None
