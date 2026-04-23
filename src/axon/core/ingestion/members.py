"""Resolve member accesses to ACCESSES edges.

Covers ENUM_MEMBER, CLASS_ATTRIBUTE, and MODULE_CONSTANT nodes.
"""

from __future__ import annotations

import logging

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphRelationship, NodeLabel, RelType
from axon.core.ingestion.imports import (
    _detect_source_roots,
    resolve_import_path,
)
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.ingestion.symbol_lookup import (
    FileSymbolIndex,
    build_file_symbol_index,
    find_containing_symbol,
)

logger = logging.getLogger(__name__)

# (parent_class_name, member_name) -> [node_id].
# Covers ENUM_MEMBER and CLASS_ATTRIBUTE only.
ParentQualifiedMemberIndex = dict[str, dict[str, list[str]]]

# (file_path, name) -> node_id. MODULE_CONSTANT only.
ModuleConstantIndex = dict[str, dict[str, str]]

# Back-compat alias so existing callers of ``EnumIndex`` still type-check.
EnumIndex = ParentQualifiedMemberIndex


def build_parent_qualified_member_index(
    graph: KnowledgeGraph,
) -> ParentQualifiedMemberIndex:
    """Index ENUM_MEMBER and CLASS_ATTRIBUTE nodes by parent class and name.

    Does NOT include MODULE_CONSTANT nodes - those are keyed on
    (file_path, name) and live in build_module_constant_index.

    Returns:
        Nested dict mapping class_name to member_name to list of node IDs.
    """
    idx: ParentQualifiedMemberIndex = {}
    for label in (NodeLabel.ENUM_MEMBER, NodeLabel.CLASS_ATTRIBUTE):
        for node in graph.get_nodes_by_label(label):
            (
                idx.setdefault(node.class_name, {})
                .setdefault(node.name, [])
                .append(node.id)
            )
    return idx


def build_enum_index(graph: KnowledgeGraph) -> ParentQualifiedMemberIndex:
    """Build an index of ENUM_MEMBER nodes keyed by parent class and name.

    Deprecated: use build_parent_qualified_member_index instead.

    Returns:
        Nested dict mapping class_name to member_name to list of node IDs.
    """
    return build_parent_qualified_member_index(graph)


def build_module_constant_index(graph: KnowledgeGraph) -> ModuleConstantIndex:
    """Index MODULE_CONSTANT nodes by file path and name.

    Returns:
        Nested dict mapping file_path to constant_name to node ID.
    """
    idx: ModuleConstantIndex = {}
    for node in graph.get_nodes_by_label(NodeLabel.MODULE_CONSTANT):
        idx.setdefault(node.file_path, {})[node.name] = node.id
    return idx


def build_imported_names(
    parse_data: list[FileParseData],
    file_index: dict[str, str],
    source_roots: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Build per-source-file map of local name to target file path.

    Uses the authoritative resolve_import_path helper from imports.py.
    Inverts file_index to map node_id to file_path.

    Returns {file_path -> {local_name -> target_file_path}}.
    Python-only - TypeScript/JavaScript imports are skipped.

    Args:
        parse_data: Per-file parse results.
        file_index: Mapping of file_path to node_id.
        source_roots: Optional pre-detected source root directories.

    Returns:
        Per-file dict mapping local imported names to their source file paths.
    """
    if source_roots is None:
        source_roots = _detect_source_roots(file_index)
    node_id_to_path: dict[str, str] = {v: k for k, v in file_index.items()}
    imported: dict[str, dict[str, str]] = {}
    for fpd in parse_data:
        if fpd.language != 'python':
            continue
        local_to_file: dict[str, str] = {}
        for imp in fpd.parse_result.imports:
            target_node_id = resolve_import_path(
                fpd.file_path, imp, file_index, source_roots
            )
            if target_node_id is None:
                continue
            target_file = node_id_to_path.get(target_node_id)
            if target_file is None:
                continue
            # Aliased names: ImportInfo.aliases maps local_alias -> original.
            for alias_local in imp.aliases.keys():
                local_to_file[alias_local] = target_file
            # Non-aliased names: original is the local binding.
            aliased_originals = set(imp.aliases.values())
            for original in imp.names:
                if original not in aliased_originals:
                    local_to_file[original] = target_file
        if local_to_file:
            imported[fpd.file_path] = local_to_file
    return imported


def process_member_accesses(
    parse_data: list[FileParseData],
    graph: KnowledgeGraph,
    parent_member_index: ParentQualifiedMemberIndex,
    module_const_index: ModuleConstantIndex | None = None,
    imported_names: dict[str, dict[str, str]] | None = None,
) -> int:
    """Emit ACCESSES edges for member accesses that resolve to known members.

    Handles three resolution paths:
    - access.parent non-empty: look up in parent_member_index (enum members
      and class attributes, including self.field resolved by the parser).
    - access.parent empty, same-file constant: look up in module_const_index
      by the access's file path.
    - access.parent empty, cross-file: look up target file via imported_names,
      then look up the constant in module_const_index for that file.

    Accesses at module scope (find_containing_symbol returns None) are
    silently dropped - documented known limit.

    Args:
        parse_data: Per-file parse results from the parsing phase.
        graph: The knowledge graph to write edges into.
        parent_member_index: Pre-built index from
            build_parent_qualified_member_index.
        module_const_index: Pre-built index from build_module_constant_index.
            When None, bare-identifier resolution is skipped.
        imported_names: Pre-built map from build_imported_names.
            When None, cross-file bare-identifier resolution is skipped.

    Returns:
        Number of ACCESSES edges emitted.
    """
    if module_const_index is None:
        module_const_index = {}
    if imported_names is None:
        imported_names = {}

    file_sym_index: FileSymbolIndex = build_file_symbol_index(
        graph, (NodeLabel.FUNCTION, NodeLabel.METHOD, NodeLabel.CLASS)
    )
    emitted = 0
    considered = 0
    for fpd in parse_data:
        src_consts = module_const_index.get(fpd.file_path, {})
        imports = imported_names.get(fpd.file_path, {})
        for access in fpd.parse_result.member_accesses:
            considered += 1
            targets: list[str] = []
            if access.parent:
                targets = parent_member_index.get(access.parent, {}).get(
                    access.name, []
                )
            else:
                same_id = src_consts.get(access.name)
                if same_id is not None:
                    targets = [same_id]
                else:
                    target_file = imports.get(access.name)
                    if target_file is not None:
                        other_id = module_const_index.get(target_file, {}).get(
                            access.name
                        )
                        if other_id is not None:
                            targets = [other_id]
            if not targets:
                continue
            caller_id = find_containing_symbol(
                access.line, fpd.file_path, file_sym_index
            )
            if caller_id is None:
                # Module-scope access - silent drop (documented known limit).
                continue
            for target_id in targets:
                target_node = graph.get_node(target_id)
                same_file = (
                    target_node is not None
                    and target_node.file_path == fpd.file_path
                )
                confidence = 1.0 if same_file else 0.8
                rel_id = (
                    f'{caller_id}->accesses->{target_id}'
                    f'@{access.line}:{access.mode}'
                )
                graph.add_relationship(
                    GraphRelationship(
                        id=rel_id,
                        type=RelType.ACCESSES,
                        source=caller_id,
                        target=target_id,
                        properties={
                            'access_mode': access.mode,
                            'confidence': confidence,
                        },
                    )
                )
                emitted += 1
    logger.info(
        'Member accesses considered=%d emitted=%d (drop=%.1f%%)',
        considered,
        emitted,
        100.0 * (considered - emitted) / max(considered, 1),
    )
    return emitted
