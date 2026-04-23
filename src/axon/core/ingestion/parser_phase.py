"""Phase 3: Code parsing for Axon.

Takes file entries from the walker, parses each one with the appropriate
tree-sitter parser, and adds symbol nodes (Function, Class, Method, Interface,
TypeAlias, Enum) to the knowledge graph with DEFINES relationships from File
to Symbol.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.ingestion.walker import FileEntry
from axon.core.parsers.base import LanguageParser, ParseResult
from axon.core.parsers.python_lang import PythonParser
from axon.core.parsers.typescript import TypeScriptParser

logger = logging.getLogger(__name__)

_KIND_TO_LABEL: dict[str, NodeLabel] = {
    "function": NodeLabel.FUNCTION,
    "class": NodeLabel.CLASS,
    "method": NodeLabel.METHOD,
    "interface": NodeLabel.INTERFACE,
    "type_alias": NodeLabel.TYPE_ALIAS,
    "enum": NodeLabel.ENUM,
}


_PARSER_FACTORIES: dict[str, Callable[[], LanguageParser]] = {
    "python": PythonParser,
    "typescript": lambda: TypeScriptParser(dialect="typescript"),
    "tsx": lambda: TypeScriptParser(dialect="tsx"),
    "javascript": lambda: TypeScriptParser(dialect="javascript"),
}

@dataclass
class FileParseData:
    """Parse results for a single file, kept for later phases."""

    file_path: str
    language: str
    parse_result: ParseResult

_PARSER_CACHE: dict[str, LanguageParser] = {}
_PARSER_CACHE_LOCK = threading.Lock()

def get_parser(language: str) -> LanguageParser:
    """Return the appropriate tree-sitter parser for *language*.

    Parser instances are cached per language to avoid repeated instantiation
    of tree-sitter ``Parser`` objects.

    Args:
        language: One of ``"python"``, ``"typescript"``, or ``"javascript"``.

    Returns:
        A :class:`LanguageParser` instance ready to parse source code.

    Raises:
        ValueError: If *language* is not supported.
    """
    cached = _PARSER_CACHE.get(language)
    if cached is not None:
        return cached
    with _PARSER_CACHE_LOCK:
        cached = _PARSER_CACHE.get(language)
        if cached is not None:
            return cached

        factory = _PARSER_FACTORIES.get(language)
        if factory is None:
            raise ValueError(
                f"Unsupported language {language!r}. "
                f"Expected one of: python, typescript, tsx, javascript"
            )

        parser = factory()
        _PARSER_CACHE[language] = parser
        return parser

def parse_file(file_path: str, content: str, language: str) -> FileParseData:
    """Parse a single file and return structured parse data.

    If parsing fails for any reason the returned :class:`FileParseData` will
    contain an empty :class:`ParseResult` so that downstream phases can
    safely skip it.

    Args:
        file_path: Relative path to the file (used for identification).
        content: Raw source code of the file.
        language: Language identifier (``"python"``, ``"typescript"``, etc.).

    Returns:
        A :class:`FileParseData` carrying the parse result.
    """
    try:
        parser = get_parser(language)
        result = parser.parse(content, file_path)
    except Exception:
        logger.warning("Failed to parse %s (%s), skipping", file_path, language, exc_info=True)
        result = ParseResult()

    return FileParseData(file_path=file_path, language=language, parse_result=result)

def process_parsing(
    files: list[FileEntry],
    graph: KnowledgeGraph,
    max_workers: int = 8,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[FileParseData]:
    """Parse every file and populate the knowledge graph with symbol nodes.

    Parsing is done in parallel using a thread pool (tree-sitter releases
    the GIL during C parsing). Graph mutation remains sequential since
    :class:`KnowledgeGraph` is not thread-safe.

    For each symbol discovered during parsing a graph node is created with
    the appropriate label (Function, Class, Method, etc.) and a DEFINES
    relationship is added from the owning File node to the new symbol node.

    Args:
        files: File entries produced by the walker phase.
        graph: The knowledge graph to populate. File nodes are expected to
            already exist (created by the structure phase).
        max_workers: Maximum number of threads for parallel parsing.
        progress_callback: Optional callback receiving (done, total) file
            counts as graph population progresses.

    Returns:
        A list of :class:`FileParseData` objects that carry the full parse
        results (imports, calls, heritage, type_refs) for use by later phases.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        all_parse_data = list(
            executor.map(
                lambda f: parse_file(f.path, f.content, f.language),
                files,
            )
        )

    total_files = len(files)
    for i, (file_entry, parse_data) in enumerate(zip(files, all_parse_data)):
        file_id = generate_id(NodeLabel.FILE, file_entry.path)
        exported_names: set[str] = set(parse_data.parse_result.exports)

        class_bases: dict[str, list[str]] = {}
        for cls_name, kind, parent_name in parse_data.parse_result.heritage:
            if kind == "extends":
                class_bases.setdefault(cls_name, []).append(parent_name)

        for symbol in parse_data.parse_result.symbols:
            label = _KIND_TO_LABEL.get(symbol.kind)
            if label is None:
                logger.warning(
                    "Unknown symbol kind %r for %s in %s, skipping",
                    symbol.kind,
                    symbol.name,
                    file_entry.path,
                )
                continue

            symbol_name = (
                f"{symbol.class_name}.{symbol.name}"
                if symbol.kind == "method" and symbol.class_name
                else symbol.name
            )

            symbol_id = generate_id(label, file_entry.path, symbol_name)

            props: dict[str, Any] = {}
            if symbol.decorators:
                props["decorators"] = symbol.decorators
            if symbol.kind == "class" and symbol.name in class_bases:
                props["bases"] = class_bases[symbol.name]

            is_exported = symbol.name in exported_names

            graph.add_node(
                GraphNode(
                    id=symbol_id,
                    label=label,
                    name=symbol.name,
                    file_path=file_entry.path,
                    start_line=symbol.start_line,
                    end_line=symbol.end_line,
                    content=symbol.content,
                    signature=symbol.signature,
                    class_name=symbol.class_name,
                    language=file_entry.language,
                    is_exported=is_exported,
                    properties=props,
                )
            )

            rel_id = f"defines:{file_id}->{symbol_id}"
            graph.add_relationship(
                GraphRelationship(
                    id=rel_id,
                    type=RelType.DEFINES,
                    source=file_id,
                    target=symbol_id,
                )
            )

        for member in parse_data.parse_result.members:
            member_symbol = f'{member.parent}.{member.name}'
            member_id = generate_id(
                NodeLabel.ENUM_MEMBER, file_entry.path, member_symbol
            )
            member_node = GraphNode(
                id=member_id,
                label=NodeLabel.ENUM_MEMBER,
                name=member.name,
                file_path=file_entry.path,
                start_line=member.line,
                end_line=member.line,
                class_name=member.parent,
                language=file_entry.language,
            )
            graph.add_node(member_node)
            parent_id = generate_id(
                NodeLabel.ENUM, file_entry.path, member.parent
            )
            graph.add_relationship(
                GraphRelationship(
                    id=f'{parent_id}->defines->{member_id}',
                    type=RelType.DEFINES,
                    source=parent_id,
                    target=member_id,
                )
            )

        if progress_callback:
            progress_callback(i + 1, total_files)

    return all_parse_data
