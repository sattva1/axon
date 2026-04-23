"""Resolve enum-member accesses to ACCESSES edges."""

from __future__ import annotations

import logging

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphRelationship, NodeLabel, RelType
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.ingestion.symbol_lookup import (
    FileSymbolIndex,
    build_file_symbol_index,
    find_containing_symbol,
)

logger = logging.getLogger(__name__)

# Parent-class-name -> {member_name -> [enum_member_node_id]}.
EnumIndex = dict[str, dict[str, list[str]]]


def build_enum_index(graph: KnowledgeGraph) -> EnumIndex:
    """Build an index of ENUM_MEMBER nodes keyed by parent class and name.

    Returns:
        Nested dict mapping class_name to member_name to list of node IDs.
    """
    idx: EnumIndex = {}
    for node in graph.get_nodes_by_label(NodeLabel.ENUM_MEMBER):
        idx.setdefault(node.class_name, {}).setdefault(node.name, []).append(
            node.id
        )
    return idx


def process_member_accesses(
    parse_data: list[FileParseData],
    graph: KnowledgeGraph,
    enum_index: EnumIndex,
) -> int:
    """Emit ACCESSES edges for each MemberAccess that resolves to a known
    ENUM_MEMBER.

    Uses ``find_containing_symbol`` to associate each access with a
    caller node. Accesses at class-body or module scope (where
    ``find_containing_symbol`` returns None) are silently dropped - this
    is a known limitation documented in Phase 5.

    Args:
        parse_data: Per-file parse results from the parsing phase.
        graph: The knowledge graph to write edges into.
        enum_index: Pre-built index from ``build_enum_index``.

    Returns:
        Number of ACCESSES edges emitted.
    """
    file_sym_index: FileSymbolIndex = build_file_symbol_index(
        graph, (NodeLabel.FUNCTION, NodeLabel.METHOD, NodeLabel.CLASS)
    )
    emitted = 0
    considered = 0
    for fpd in parse_data:
        for access in fpd.parse_result.member_accesses:
            considered += 1
            targets = enum_index.get(access.parent, {}).get(access.name, [])
            if not targets:
                continue
            caller_id = find_containing_symbol(
                access.line, fpd.file_path, file_sym_index
            )
            if caller_id is None:
                # Class-body scope or top-level - silent drop.
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
    drop_count = considered - emitted
    drop_pct = 100.0 * drop_count / max(considered, 1)
    logger.info(
        'Member accesses considered=%d emitted=%d (drop=%.1f%%)',
        considered,
        emitted,
        drop_pct,
    )
    return emitted
