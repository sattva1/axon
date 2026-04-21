"""Base parser interface and shared data structures.

Defines the intermediate representation produced by language-specific parsers
before the data is mapped into the knowledge graph.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class SymbolInfo:
    """A parsed symbol (function, class, method, etc.)."""

    name: str
    kind: str  # "function", "class", "method", "interface", "type_alias", "enum"
    start_line: int
    end_line: int
    content: str
    signature: str = ""
    class_name: str = ""  # for methods: the owning class
    decorators: list[str] = field(default_factory=list)  # e.g. ["staticmethod", "app.route"]


@dataclass
class ImportInfo:
    """A parsed import statement.

    Contract:
    - ``module``: the source module path (e.g. ``"os.path"``, ``"./utils"``).
    - ``names``: the symbols being imported from *module* (e.g. ``["join", "exists"]``).
      For ``import numpy as np``, ``names=["numpy"]`` (the last segment of the module),
      NOT the alias.  For ``from os.path import join``, ``names=["join"]``.
    - ``alias``: the local binding name when the import is aliased
      (e.g. ``"np"`` for ``import numpy as np``, ``""`` otherwise).
      Import resolution uses ``module`` to locate the target file; ``alias`` is
      only relevant for local-name lookups by callers.
    """

    module: str  # the module path (e.g., "os.path", "./utils")
    names: list[str] = field(default_factory=list)  # imported names (e.g., ["join", "exists"])
    is_relative: bool = False
    alias: str = ""  # local binding name when aliased (e.g. "np" for "import numpy as np")


@dataclass
class CallInfo:
    """A parsed function call."""

    name: str  # the called function/method name
    line: int
    receiver: str = ''  # for method calls: the object (e.g., "self", "user")
    arguments: list[str] = field(
        default_factory=list
    )  # bare identifier arguments (callbacks)
    # --- Phase 4a additions ---
    dispatch_kind: str = 'direct'
    in_try: bool = False
    in_except: bool = False
    in_finally: bool = False
    in_loop: bool = False
    awaited: bool = False
    context_managers: tuple[str, ...] = ()
    return_consumption: str = 'stored'

    def extra_props(self) -> dict[str, Any]:
        """Return non-default Phase-4a fields as serializable dict.

        Sparse encoding keeps the metadata_json column lean: edges with
        plain direct/stored synchronous semantics produce an empty dict.
        Defaults are read from the dataclass fields() so this stays in
        sync with the class declaration.
        """
        # Build a dict of current -> default field values and emit only
        # the deltas. Skip the original 4 fields (name/line/receiver/
        # arguments); they are separately represented on the edge.
        original = {'name', 'line', 'receiver', 'arguments'}
        extra: dict[str, Any] = {}
        for f in fields(self):
            if f.name in original:
                continue
            current = getattr(self, f.name)
            default = f.default
            # context_managers default is () - a tuple; serialise as list.
            if f.name == 'context_managers':
                if current:
                    extra[f.name] = list(current)
                continue
            if current != default:
                extra[f.name] = current

        return extra


@dataclass
class TypeRef:
    """A parsed type annotation reference."""

    name: str  # the type name (e.g., "User", "list", "str")
    kind: str  # "param", "return", "variable"
    line: int
    param_name: str = ""  # for param types: the parameter name


@dataclass
class ParseResult:
    """Complete parse result for a single file."""

    symbols: list[SymbolInfo] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    calls: list[CallInfo] = field(default_factory=list)
    type_refs: list[TypeRef] = field(default_factory=list)
    heritage: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # (class_name, kind, parent_name) where kind is "extends" or "implements"
    exports: list[str] = field(default_factory=list)  # names from __all__ or export statements


class LanguageParser(ABC):
    """Base interface for language-specific parsers."""

    @abstractmethod
    def parse(self, content: str, file_path: str) -> ParseResult: ...
