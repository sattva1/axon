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
    - ``names``: the *original* symbol names imported from *module*
      (e.g. ``["join", "exists"]``). For ``import numpy as np``,
      ``names=["numpy"]`` (the last segment of the module), NOT the alias.
      For ``from os.path import join as j``, ``names=["join"]`` (original),
      NOT ``"j"``. Downstream consumers that need the local name must
      consult ``aliases``.
    - ``alias``: the local binding name when the whole import is aliased
      (e.g. ``"np"`` for ``import numpy as np``, ``""`` otherwise).
      Import resolution uses ``module`` to locate the target file; ``alias``
      is only relevant for local-name lookups by callers.
    - ``aliases``: per-name alias map for ``from X import Y as Z`` forms.
      Maps local alias to original name: ``{"Z": "Y"}``. Empty when no
      per-name aliases are present.
    """

    module: str  # the module path (e.g., "os.path", "./utils")
    names: list[str] = field(default_factory=list)  # original imported names
    is_relative: bool = False
    alias: str = ''  # local binding name when aliased (e.g. "np" for "import numpy as np")
    aliases: dict[str, str] = field(
        default_factory=dict
    )  # {local_alias: original_name}


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
    exports: list[str] = field(
        default_factory=list
    )  # names from __all__ or export statements

    def build_import_type_map(self) -> dict[str, str]:
        """Map each locally-bound imported name to its canonical class name.

        For ``from X import Y`` -> ``{"Y": "Y"}``.
        For ``from X import Y as Z`` -> ``{"Z": "Y"}``.
        For ``import X as Y`` -> ``{"Y": X_last_segment}``.

        Module-aliased imports (``import a.b.c as cc``) map the alias
        to the last dotted segment only. Attribute-call constructors
        (``cc.Foo()``) are not captured by the binding tracker and fall
        through to ``dispatch_kind="direct"`` as expected.

        Called between pass 1 (_walk) and pass 2 (_extract_calls_recursive)
        by PythonParser.parse(). Do not call from ingestion layer.
        """
        mapping: dict[str, str] = {}
        for imp in self.imports:
            if imp.alias:
                # "import X[.Y] as Z" -- imp.names contains the last segment.
                mapping[imp.alias] = imp.names[0] if imp.names else imp.alias
            # aliased_originals: set of original names that have a local alias,
            # so the original name is NOT a valid local binding.
            aliased_originals = set(imp.aliases.values())
            for original_name in imp.names:
                if original_name not in aliased_originals:
                    # Directly imported without aliasing -- locally bound as-is.
                    mapping[original_name] = original_name
            # Per-name aliases: local alias -> original class name.
            for local_alias, original_name in imp.aliases.items():
                mapping[local_alias] = original_name
        return mapping


class LanguageParser(ABC):
    """Base interface for language-specific parsers."""

    @abstractmethod
    def parse(self, content: str, file_path: str) -> ParseResult: ...
