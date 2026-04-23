"""Python language parser using tree-sitter.

Extracts functions, classes, methods, imports, calls, type annotations,
and inheritance relationships from Python source code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from axon.core.parsers.base import (
    CallInfo,
    ImportInfo,
    LanguageParser,
    MemberAccess,
    MemberInfo,
    ParseResult,
    SymbolInfo,
    TypeRef,
)
from axon.core.python_lang_constants import (
    DATACLASS_DECORATORS,
    ENUM_BASES,
    PYDANTIC_BASES,
)

PY_LANGUAGE = Language(tspython.language())

_BUILTIN_TYPES: frozenset[str] = frozenset(
    {
        'str',
        'int',
        'float',
        'bool',
        'None',
        'list',
        'dict',
        'set',
        'tuple',
        'Any',
        'Optional',
        'bytes',
        'complex',
        'object',
        'type',
    }

)


@dataclass(frozen=True, slots=True)
class _Binding:
    """A locally-tracked variable binding with a resolved type name."""

    type_name: str  # canonical class name, e.g. "ThreadPoolExecutor"


_DISPATCH_BY_TYPE_METHOD: Final[dict[tuple[str, str], str]] = {
    ('ThreadPoolExecutor', 'submit'): 'thread_executor',
    ('ThreadPoolExecutor', 'map'): 'thread_executor',
    ('ProcessPoolExecutor', 'submit'): 'process_executor',
    ('ProcessPoolExecutor', 'map'): 'process_executor',
    ('TaskGroup', 'create_task'): 'detached_task',
}

# Pre-computed known-class set for static-style call fast-path
# (e.g., ThreadPoolExecutor.submit(fn) with no binding context).
_DISPATCH_KNOWN_CLASSES: Final[frozenset[str]] = frozenset(
    type_name for type_name, _method in _DISPATCH_BY_TYPE_METHOD
)


class _ParseContext:
    """All per-parse mutable state, threaded through _extract_calls_recursive.

    A fresh instance is constructed per PythonParser.parse() call, so
    state never crosses thread boundaries (parser instances are shared via
    _PARSER_CACHE for the parallel ingestion path).

    Depth counters are used instead of booleans so nested scopes compose
    correctly. context_managers is a stack: outermost with-item first.

    NOT for export. Private to python_lang.py. Do not surface on
    ParseResult; the ingestion layer consumes only CallInfo.
    """

    def __init__(self, import_local_to_type: dict[str, str]) -> None:
        # Scope depth counters (formerly _ScopeStack).
        self.try_depth = 0
        self.except_depth = 0
        self.finally_depth = 0
        self.loop_depth = 0
        self.awaited_depth = 0
        self.context_managers: list[str] = []
        # Binding frames (new in Phase 4a-follow-up).
        self._local_frames: list[dict[str, _Binding]] = [{}]  # module frame
        self._self_frames: list[dict[str, _Binding]] = []
        # Import resolution map, built between pass 1 and pass 2.
        self.import_local_to_type: dict[str, str] = import_local_to_type
        # Class-attribution stack: (class_name, self_param_name_or_None).
        self._class_attr_stack: list[tuple[str, str | None]] = []

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time snapshot of the current scope state."""
        return {
            'in_try': self.try_depth > 0,
            'in_except': self.except_depth > 0,
            'in_finally': self.finally_depth > 0,
            'in_loop': self.loop_depth > 0,
            'awaited': self.awaited_depth > 0,
            'context_managers': tuple(self.context_managers),
        }

    def push_local(self) -> None:
        """Push a new local variable frame (function body or with-scope)."""
        self._local_frames.append({})

    def pop_local(self) -> None:
        """Pop the innermost local variable frame."""
        self._local_frames.pop()

    def push_class(self, self_bindings: dict[str, _Binding]) -> None:
        """Push a class self-binding frame (set by prescan)."""
        self._self_frames.append(self_bindings)

    def pop_class(self) -> None:
        """Pop the innermost class self-binding frame."""
        self._self_frames.pop()

    def bind(self, name: str, binding: _Binding) -> None:
        """Record a local variable binding in the current frame."""
        self._local_frames[-1][name] = binding

    def lookup(self, name: str) -> _Binding | None:
        """Look up a local variable binding, searching from innermost frame."""
        for frame in reversed(self._local_frames):
            if name in frame:
                return frame[name]
        return None

    def lookup_self(self, attr: str) -> _Binding | None:
        """Look up a self.attr binding in the current class frame."""
        if not self._self_frames:
            return None
        return self._self_frames[-1].get(attr)

    def push_class_attribution(
        self, class_name: str, self_param: str | None
    ) -> None:
        """Push a class-attribution frame onto the stack."""
        self._class_attr_stack.append((class_name, self_param))

    def pop_class_attribution(self) -> None:
        """Pop the innermost class-attribution frame."""
        self._class_attr_stack.pop()

    def current_class_name(self) -> str | None:
        """Return the class name at the top of the attribution stack, or None."""
        if not self._class_attr_stack:
            return None
        return self._class_attr_stack[-1][0]

    def current_self_param(self) -> str | None:
        """Return the current method self-parameter name, or None.

        Returns None when not inside a method (e.g. inside a class body
        at top level, or inside a staticmethod).
        """
        if not self._class_attr_stack:
            return None
        return self._class_attr_stack[-1][1]


_DEFAULT_SCOPE_SNAPSHOT: dict[str, Any] = {
    'in_try': False,
    'in_except': False,
    'in_finally': False,
    'in_loop': False,
    'awaited': False,
    'context_managers': (),
}


def _classify_return_consumption(call_node: Node) -> str:
    """Classify how a call's return value is used by walking up the parent.

    Possible values:
    - "awaited"          -- direct parent is await.
    - "passed_through"   -- direct parent is return_statement.
    - "stored"           -- direct parent is assignment (LHS takes value)
                            or used as argument to a containing call
                            (argument-position).
    - "ignored"          -- the call stands alone as an expression statement.

    Note: argument-position is classified as "stored" in Phase 4a; a
    dedicated "argument" value may be added in a later phase.
    """
    parent = call_node.parent
    if parent is None:
        return 'stored'
    if parent.type == 'await':
        return 'awaited'
    if parent.type == 'return_statement':
        return 'passed_through'
    if parent.type == 'expression_statement':
        return 'ignored'
    if parent.type == 'assignment':
        return 'stored'
    # argument_list (func(call())) or any other compound expression.
    return 'stored'


# Closed set of dispatch kinds recognised in Phase 4a:
#   - "direct"            -- synchronous call (default).
#   - "detached_task"     -- fire-and-forget asyncio task or callback.
#   - "thread_executor"   -- submitted to a thread-pool executor.
#   - "process_executor"  -- submitted to a process-pool executor.
#   - "enqueued_job"      -- celery .apply_async / .delay on a task.
#   - "callback_registry" -- reserved; NOT detected in Phase 4a.
#
# New patterns should be added here. The Celery path reads result.symbols,
# which requires the two-pass invariant in PythonParser.parse(): _walk
# runs before _extract_calls_recursive.

_CELERY_DECORATORS: frozenset[str] = frozenset(
    {
        'shared_task',
        'task',
        'celery.task',
        'celery.shared_task',
        'app.task',
        'app.shared_task',
    }
)

# Regex for ALL_CAPS module-constant identifier reads.
# Requires a leading capital letter so lone `_`, `__`, and digit-only tokens
# are rejected by construction.
_ALL_CAPS_RE: re.Pattern[str] = re.compile(r'^[A-Z][A-Z0-9_]*$')

_LITERAL_NODE_TYPES: frozenset[str] = frozenset(
    {
        'integer',
        'float',
        'string',
        'true',
        'false',
        'none',
        'list',
        'tuple',
        'dictionary',
        'set',
        'unary_operator',  # handles -5, +3.14
    }
)


def _is_final_annotation(node: Node | None) -> bool:
    """Return True if node represents ``Final`` or ``Final[...]``."""
    if node is None:
        return False
    # tree-sitter wraps annotation in a "type" node
    inner = node
    if inner.type == 'type' and inner.children:
        inner = inner.children[0]
    if inner.type == 'identifier':
        return inner.text.decode('utf8') == 'Final'
    if inner.type == 'subscript':
        base = inner.child_by_field_name('value')
        if base is not None and base.type == 'identifier':
            return base.text.decode('utf8') == 'Final'
    if inner.type == 'generic_type':
        for ch in inner.children:
            if ch.type == 'identifier':
                return ch.text.decode('utf8') == 'Final'
    return False


def _is_literal_rhs(node: Node | None) -> bool:
    """Return True if node's type is a known literal type."""
    if node is None:
        return False
    return node.type in _LITERAL_NODE_TYPES


def _extract_self_param_name(func_node: Node) -> str | None:
    """Return the first parameter's identifier text, or None.

    Returns None when the first param is not a plain identifier
    (e.g. *args, **kwargs, or a staticmethod with no self/cls).
    Unconditional break after the first non-separator child - only the
    first param is the potential self receiver.
    """
    params = func_node.child_by_field_name('parameters')
    if params is None:
        return None
    for child in params.children:
        if child.type == 'identifier':
            return child.text.decode('utf8')
        if child.type in (
            'typed_parameter',
            'default_parameter',
            'typed_default_parameter',
        ):
            for sub in child.children:
                if sub.type == 'identifier':
                    return sub.text.decode('utf8')
            return None
        if child.type in ('(', ')', ','):
            continue  # punctuation, skip
        return None  # *args, **kwargs, or unrecognised - no self param
    return None


def _resolve_callee_to_type_name(
    call_node: Node, import_local_to_type: dict[str, str]
) -> str | None:
    """Return canonical class name for a call's callee, or None.

    Handles:
    - bare identifier: ``Foo()`` -> lookup in import_local_to_type,
      else return the identifier itself.
    - attribute ``pkg.Foo()``: return last segment (``Foo``) -- the
      dispatch table keys on class-name-as-written.
    """
    func = call_node.child_by_field_name('function')
    if func is None:
        return None
    if func.type == 'identifier':
        name = func.text.decode('utf8')
        return import_local_to_type.get(name, name)
    if func.type == 'attribute':
        last = None
        for ch in reversed(func.children):
            if ch.type == 'identifier':
                last = ch.text.decode('utf8')
                break
        return last
    return None


def _iter_with_items(with_node: Node):
    """Yield each with_item node from a with_statement or async_with_statement."""
    for child in with_node.children:
        if child.type != 'with_clause':
            continue
        for item in child.children:
            if item.type == 'with_item':
                yield item


def _try_extract_binding(assignment_node: Node, ctx: _ParseContext) -> None:
    """Record a local variable binding if the RHS is a constructor call.

    Handles ``x = SomeClass()`` where the LHS is a single identifier.
    Delegates type resolution to _resolve_callee_to_type_name.
    """
    left = assignment_node.child_by_field_name('left')
    right = assignment_node.child_by_field_name('right')
    if left is None or right is None:
        return
    if left.type != 'identifier':
        return
    if right.type != 'call':
        return
    type_name = _resolve_callee_to_type_name(right, ctx.import_local_to_type)
    if type_name is not None:
        ctx.bind(left.text.decode('utf8'), _Binding(type_name=type_name))


def _try_extract_with_as_bindings(with_node: Node, ctx: _ParseContext) -> None:
    """Record bindings for ``with Call() as name:`` forms.

    Called after push_local() for the with-scope, so bindings land in the
    correct frame. Uses _iter_with_items to avoid duplicating traversal.

    Tree-sitter-python represents ``with Call() as name:`` as:
      with_item -> value=as_pattern(call, as_pattern_target(identifier))
    so we unwrap the as_pattern to reach the call and the alias.
    """
    for item in _iter_with_items(with_node):
        value_node = item.child_by_field_name('value')
        if value_node is None:
            continue

        # Tree-sitter wraps "Call() as name" in an as_pattern node.
        if value_node.type == 'as_pattern':
            call_node = None
            alias_node = None
            for ch in value_node.children:
                if ch.type == 'call':
                    call_node = ch
                elif ch.type == 'as_pattern_target':
                    # The target is a wrapper; look for the identifier inside.
                    for sub in ch.children:
                        if sub.type == 'identifier':
                            alias_node = sub
                            break
            if call_node is None or alias_node is None:
                continue
        elif value_node.type == 'call':
            # Older grammar forms may have call directly on with_item.
            call_node = value_node
            alias_node = item.child_by_field_name('alias')
            if alias_node is None:
                continue
            if alias_node.type != 'identifier':
                continue
        else:
            continue

        type_name = _resolve_callee_to_type_name(
            call_node, ctx.import_local_to_type
        )
        if type_name is not None:
            ctx.bind(
                alias_node.text.decode('utf8'), _Binding(type_name=type_name)
            )


def _prescan_class_self_bindings(
    class_body: Node, import_local_to_type: dict[str, str]
) -> dict[str, _Binding]:
    """Scan a class body for self.ATTR = Call() assignments.

    Returns a dict mapping attr name to _Binding. Bounded at nested
    function_definition boundaries (inner helpers have a different self
    context and are not scanned).
    """
    bindings: dict[str, _Binding] = {}

    def _scan(node: Node) -> None:
        for child in node.children:
            if child.type == 'function_definition':
                _scan_method_body(child)
            elif child.type == 'decorated_definition':
                for sub in child.children:
                    if sub.type == 'function_definition':
                        _scan_method_body(sub)

    def _scan_method_body(func_node: Node) -> None:
        body = func_node.child_by_field_name('body')
        if body is None:
            return
        _scan_stmts(body)

    def _scan_stmts(node: Node) -> None:
        for child in node.children:
            if child.type == 'function_definition':
                # Stop: nested function has its own self context.
                continue
            if child.type == 'expression_statement':
                for sub in child.children:
                    if sub.type == 'assignment':
                        _try_scan_self_assignment(sub)
            _scan_stmts(child)

    def _try_scan_self_assignment(assignment_node: Node) -> None:
        left = assignment_node.child_by_field_name('left')
        right = assignment_node.child_by_field_name('right')
        if left is None or right is None:
            return
        if left.type != 'attribute':
            return
        if right.type != 'call':
            return
        obj = left.children[0] if left.children else None
        if obj is None or obj.type != 'identifier':
            return
        if obj.text.decode('utf8') != 'self':
            return
        attr_name = None
        for ch in reversed(left.children):
            if ch.type == 'identifier':
                attr_name = ch.text.decode('utf8')
                break
        if attr_name is None or attr_name == 'self':
            return
        type_name = _resolve_callee_to_type_name(right, import_local_to_type)
        if type_name is not None:
            bindings[attr_name] = _Binding(type_name=type_name)

    _scan(class_body)
    return bindings


def _classify_dispatch_kind(
    call_node: Node, result: ParseResult, ctx: _ParseContext
) -> str:
    """Classify a call's dispatch kind.

    Two classification regimes coexist:

    1. Type-resolved: receiver's type is known (via _ParseContext bindings
       or known-class-name fast path) -> consult _DISPATCH_BY_TYPE_METHOD.
    2. Structural (type-less): asyncio.create_task, Celery decorator
       inspection, run_in_executor first-arg shape, bare create_task --
       handled inline.

    Type-reference pseudo-CallInfos emitted by _make_type_reference_callinfo
    (raise/except paths) never reach this function; they are appended
    directly with dispatch_kind='direct'. Constructor calls inside raise
    statements (e.g. raise Err()) do reach this function but classify as
    'direct' because they have no method receiver.

    Unresolvable receiver types fall through to 'direct' rather than
    inferring from name substrings. This is intentional.

    TODO: future phases may unify the two regimes via a matcher-callable
    column on the table.
    """
    func = call_node.child_by_field_name('function')
    if func is None:
        return 'direct'

    if func.type == 'attribute':
        obj_node = func.children[0] if func.children else None
        method = ''
        for ch in reversed(func.children):
            if ch.type == 'identifier':
                method = ch.text.decode('utf8')
                break
        receiver_root = ''
        if obj_node is not None:
            if obj_node.type == 'identifier':
                receiver_root = obj_node.text.decode('utf8')
            elif obj_node.type == 'attribute':
                cur = obj_node
                while cur is not None and cur.type == 'attribute':
                    cur = cur.children[0] if cur.children else None
                if cur is not None and cur.type == 'identifier':
                    receiver_root = cur.text.decode('utf8')

        # === Type-resolved regime ===

        # 1. self.attr.method() -- look up attr in class self-frame.
        if obj_node is not None and obj_node.type == 'attribute':
            # Check if the obj_node root is "self".
            obj_root = obj_node.children[0] if obj_node.children else None
            if (
                obj_root is not None
                and obj_root.type == 'identifier'
                and obj_root.text.decode('utf8') == 'self'
            ):
                attr_name = ''
                for ch in reversed(obj_node.children):
                    if ch.type == 'identifier':
                        attr_name = ch.text.decode('utf8')
                        break
                if attr_name and attr_name != 'self':
                    binding = ctx.lookup_self(attr_name)
                    if binding is not None:
                        hit = _DISPATCH_BY_TYPE_METHOD.get(
                            (binding.type_name, method)
                        )
                        if hit is not None:
                            return hit

        # 2. Local/module-scope binding on receiver.
        if receiver_root:
            binding = ctx.lookup(receiver_root)
            if binding is not None:
                hit = _DISPATCH_BY_TYPE_METHOD.get((binding.type_name, method))
                if hit is not None:
                    return hit

            # 3. Static-style: receiver IS a known class name
            # (treat as synthetic binding with type_name=receiver_root).
            if receiver_root in _DISPATCH_KNOWN_CLASSES:
                hit = _DISPATCH_BY_TYPE_METHOD.get((receiver_root, method))
                if hit is not None:
                    return hit

        # === Structural (type-less) regime ===

        if receiver_root == 'asyncio' and method in {
            'create_task',
            'ensure_future',
        }:
            return 'detached_task'

        if method == 'create_task':
            return 'detached_task'

        if method in {'call_soon', 'call_later', 'call_soon_threadsafe'}:
            return 'detached_task'

        if method == 'run_in_executor':
            args_node = call_node.child_by_field_name('arguments')
            first_arg_is_none = False
            if args_node is not None:
                for ch in args_node.children:
                    if ch.is_named and ch.type not in {'keyword_argument'}:
                        first_arg_is_none = ch.type == 'none' or (
                            ch.type == 'identifier'
                            and ch.text.decode('utf8') == 'None'
                        )
                        break
            return 'detached_task' if first_arg_is_none else 'thread_executor'

        if method in {'apply_async', 'delay'} and receiver_root:
            for sym in result.symbols:
                if sym.name != receiver_root:
                    continue
                if any(dec in _CELERY_DECORATORS for dec in sym.decorators):
                    return 'enqueued_job'
                if any(
                    dec.split('.')[-1] in {'task', 'shared_task'}
                    and dec.split('.')[0] in {'celery', 'app'}
                    for dec in sym.decorators
                ):
                    return 'enqueued_job'

    if func.type == 'identifier':
        name = func.text.decode('utf8')
        if name in {'create_task', 'ensure_future'}:
            # Bare names flagged unconditionally; cross-checking imports
            # requires more than a single pass.
            return 'detached_task'

    return 'direct'


class PythonParser(LanguageParser):
    """Parses Python source code using tree-sitter."""

    def __init__(self) -> None:
        self._parser = Parser(PY_LANGUAGE)

    def parse(self, content: str, file_path: str) -> ParseResult:
        """Parse Python source and return structured information."""
        tree = self._parser.parse(bytes(content, "utf8"))
        result = ParseResult()
        root = tree.root_node
        # Pass 1: populate symbols, imports, annotations. Bindings NOT touched.
        self._walk(root, content, result, class_name='')
        # Between passes: extract module constants (structural, no scope state
        # needed) and build the local-name -> canonical-class-name map.
        self._extract_module_constants(root, result)
        import_local_to_type = result.build_import_type_map()
        # Pass 2: extract calls with full binding/scope context.
        ctx = _ParseContext(import_local_to_type=import_local_to_type)
        self._extract_calls_recursive(root, result, ctx)
        return result

    def _walk(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Recursively walk the AST to extract definitions and annotations.

        Call extraction is handled separately via ``_extract_calls_recursive``
        at each scope boundary (module, class, function) to avoid
        double-counting.
        """
        for child in node.children:
            match child.type:
                case "function_definition":
                    self._extract_function(child, content, result, class_name)
                case "class_definition":
                    self._extract_class(child, content, result)
                case "import_statement":
                    self._extract_import(child, result)
                case "import_from_statement":
                    self._extract_import_from(child, result)
                case "decorated_definition":
                    self._extract_decorated(child, content, result, class_name)
                case "expression_statement":
                    # Only extract variable annotations here; calls are
                    # handled by the scope-level _extract_calls_recursive.
                    self._extract_annotations_from_expression(child, result)
                case _:
                    self._walk(child, content, result, class_name)

    def _extract_function(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Extract a function or method definition."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode("utf8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        node_content = content[node.start_byte : node.end_byte]

        kind = "method" if class_name else "function"
        signature = self._build_signature(node, content)

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind=kind,
                start_line=start_line,
                end_line=end_line,
                content=node_content,
                signature=signature,
                class_name=class_name,
            )
        )

        self._extract_param_types(node, result)

        return_type = node.child_by_field_name("return_type")
        if return_type is not None:
            type_name = self._extract_type_name(return_type)
            if type_name and type_name not in _BUILTIN_TYPES:
                result.type_refs.append(
                    TypeRef(
                        name=type_name,
                        kind="return",
                        line=return_type.start_point[0] + 1,
                    )
                )

        # Call extraction is handled once at module level by parse().
        body = node.child_by_field_name("body")
        if body is not None:
            # Nested functions/classes inside a function are not methods,
            # so we pass class_name="" to keep them as standalone symbols.
            self._walk(body, content, result, class_name="")

    def _build_signature(self, func_node: Node, content: str) -> str:
        """Build a human-readable signature string for a function."""
        name_node = func_node.child_by_field_name("name")
        params_node = func_node.child_by_field_name("parameters")
        return_type = func_node.child_by_field_name("return_type")

        if name_node is None or params_node is None:
            return ""

        name = name_node.text.decode("utf8")
        params = params_node.text.decode("utf8")
        sig = f"def {name}{params}"

        if return_type is not None:
            sig += f" -> {return_type.text.decode('utf8')}"

        return sig

    def _extract_decorated(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Extract a decorated function or class, capturing decorator names.

        Tree-sitter wraps decorated definitions in a ``decorated_definition``
        node whose children are one or more ``decorator`` nodes followed by
        the actual ``function_definition`` or ``class_definition``.
        """
        decorators: list[str] = []
        definition_node: Node | None = None

        for child in node.children:
            if child.type == 'decorator':
                dec_name = self._extract_decorator_name(child)
                if dec_name:
                    decorators.append(dec_name)
            elif child.type in ('function_definition', 'class_definition'):
                definition_node = child

        if definition_node is None:
            return

        if definition_node.type == 'function_definition':
            count_before = len(result.symbols)
            self._extract_function(
                definition_node, content, result, class_name
            )
            if count_before < len(result.symbols):
                result.symbols[count_before].decorators = decorators
        else:
            # Pass decorators directly so _extract_class can compute
            # is_dataclass and write decorators onto the SymbolInfo.
            self._extract_class(
                definition_node, content, result, decorators=decorators
            )

    def _extract_decorator_name(self, decorator_node: Node) -> str:
        """Extract the dotted name from a decorator node.

        Handles three forms::

            @staticmethod          -> "staticmethod"
            @app.route             -> "app.route"
            @server.list_tools()   -> "server.list_tools"
        """
        for child in decorator_node.children:
            if child.type == "identifier":
                return child.text.decode("utf8")
            if child.type == "attribute":
                return child.text.decode("utf8")
            if child.type == "call":
                func = child.child_by_field_name("function")
                if func is not None:
                    return func.text.decode("utf8")
        return ""

    def _extract_param_types(self, func_node: Node, result: ParseResult) -> None:
        """Extract type annotations from function parameters."""
        params_node = func_node.child_by_field_name("parameters")
        if params_node is None:
            return

        for param in params_node.children:
            if param.type == "typed_parameter":
                self._extract_typed_param(param, result)
            elif param.type == "typed_default_parameter":
                self._extract_typed_param(param, result)

    def _extract_typed_param(self, param_node: Node, result: ParseResult) -> None:
        """Extract a single typed parameter's type reference."""
        param_name = ""
        for child in param_node.children:
            if child.type == "identifier":
                param_name = child.text.decode("utf8")
                break

        type_node = param_node.child_by_field_name("type")
        if type_node is None:
            return

        type_name = self._extract_type_name(type_node)
        if type_name and type_name not in _BUILTIN_TYPES:
            result.type_refs.append(
                TypeRef(
                    name=type_name,
                    kind='param',
                    line=type_node.start_point[0] + 1,
                    param_name=param_name,
                )
            )

    def _extract_class(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        decorators: list[str] | None = None,
    ) -> None:
        """Extract a class definition and its contents."""
        if decorators is None:
            decorators = []
        name_node = node.child_by_field_name('name')
        if name_node is None:
            return

        class_name = name_node.text.decode("utf8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        node_content = content[node.start_byte : node.end_byte]

        bases: list[str] = []
        superclasses = node.child_by_field_name('superclasses')
        if superclasses is not None:
            for child in superclasses.children:
                if not child.is_named:
                    continue
                if child.type == 'identifier':
                    parent_name = child.text.decode('utf8')
                    bases.append(parent_name)
                    result.heritage.append(
                        (class_name, 'extends', parent_name)
                    )
                elif child.type == 'attribute':
                    # e.g. class Foo(module.Base): - capture "module.Base"
                    parent_name = child.text.decode('utf8')
                    result.heritage.append(
                        (class_name, 'extends', parent_name)
                    )
                elif child.type == 'subscript':
                    # e.g. class Foo(Generic[T]): - capture "Generic"
                    base = child.child_by_field_name('value')
                    if base is not None:
                        parent_name = base.text.decode('utf8')
                        result.heritage.append(
                            (class_name, 'extends', parent_name)
                        )

        is_enum = bool(set(bases) & ENUM_BASES)
        is_pydantic = bool(set(bases) & PYDANTIC_BASES)
        # Direct set-intersection: matches full dotted decorator string.
        is_dataclass = bool(set(decorators) & DATACLASS_DECORATORS)

        symbol = SymbolInfo(
            name=class_name,
            kind='enum' if is_enum else 'class',
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            decorators=list(decorators),
        )
        result.symbols.append(symbol)

        body = node.child_by_field_name('body')
        if body is not None:
            if is_enum:
                # _walk handles nested defs; _extract_enum_members only
                # iterates assignment nodes at the enum-body level - no overlap.
                self._extract_enum_members(body, class_name, result)
            else:
                self._extract_class_attributes(
                    body, class_name, is_pydantic, is_dataclass, result
                )
            self._walk(body, content, result, class_name=class_name)

    def _extract_enum_members(
        self, class_body: Node, class_name: str, result: ParseResult
    ) -> None:
        """Extract enum member assignments from a class body.

        Iterates direct children looking for ``expression_statement`` nodes
        containing ``assignment`` or ``annotated_assignment``. Emits one
        ``MemberInfo`` per valid LHS identifier.

        Skips:
        - Non-identifier LHS (tuple unpacking, subscripts, attribute targets).
        - Dunder-style names (starts AND ends with ``_``).
        """
        for child in class_body.children:
            if child.type != 'expression_statement':
                continue
            for stmt in child.children:
                if stmt.type not in ('assignment', 'annotated_assignment'):
                    continue
                left = stmt.child_by_field_name('left')
                if left is None:
                    left = stmt.child_by_field_name('name')
                if left is None:
                    continue
                if left.type != 'identifier':
                    continue
                lhs = left.text.decode('utf8')
                if lhs.startswith('_') and lhs.endswith('_'):
                    continue
                result.members.append(
                    MemberInfo(
                        name=lhs,
                        parent=class_name,
                        kind='enum_member',
                        line=stmt.start_point[0] + 1,
                    )
                )

    @staticmethod
    def _class_attr_lhs(stmt: Node) -> tuple[str | None, int, bool]:
        """Return (name, line, is_annotated) for an assignment LHS.

        Returns (None, 0, False) when:
        - Node is not an assignment.
        - LHS is not a plain identifier.
        - Name is a dunder (starts and ends with double underscores, length > 4).

        In tree-sitter-python, all of ``x: int``, ``x: int = 5``, and
        ``x = 5`` are represented as ``assignment`` nodes. The presence of
        a ``type`` child field distinguishes annotated from plain assignments.
        """
        if stmt.type != 'assignment':
            return None, 0, False
        left = stmt.child_by_field_name('left')
        if left is None:
            return None, 0, False
        if left.type != 'identifier':
            return None, 0, False
        name = left.text.decode('utf8')
        if name.startswith('__') and name.endswith('__') and len(name) > 4:
            return None, 0, False
        has_annotation = stmt.child_by_field_name('type') is not None
        return name, stmt.start_point[0] + 1, has_annotation

    @staticmethod
    def _is_field_call_rhs(stmt: Node) -> bool:
        """Return True when the RHS is a call to ``Field`` or ``field``."""
        right = stmt.child_by_field_name('right')
        if right is None or right.type != 'call':
            return False
        func = right.child_by_field_name('function')
        if func is None:
            return False
        if func.type == 'identifier':
            return func.text.decode('utf8') in ('Field', 'field')
        if func.type == 'attribute':
            for ch in reversed(func.children):
                if ch.type == 'identifier':
                    return ch.text.decode('utf8') in ('Field', 'field')
        return False

    def _extract_class_attributes(
        self,
        class_body: Node,
        class_name: str,
        is_pydantic: bool,
        is_dataclass: bool,
        result: ParseResult,
    ) -> None:
        """Extract class attributes from a non-enum class body.

        Emits MemberInfo for annotated assignments unconditionally, and for
        plain (unannotated) assignments when the class is a Pydantic model or
        dataclass and the RHS is a Field/field call.

        In tree-sitter-python both ``x: int = 5`` and ``x = 5`` are
        assignment nodes; the presence of a ``type`` child field distinguishes
        them. See _class_attr_lhs for details.
        """
        for child in class_body.children:
            if child.type != 'expression_statement':
                continue
            for stmt in child.children:
                lhs_name, line, is_annotated = self._class_attr_lhs(stmt)
                if lhs_name is None:
                    continue
                if is_annotated:
                    # Always emit annotated class attributes.
                    result.members.append(
                        MemberInfo(
                            name=lhs_name,
                            parent=class_name,
                            kind='class_attribute',
                            line=line,
                        )
                    )
                elif (is_pydantic or is_dataclass) and self._is_field_call_rhs(
                    stmt
                ):
                    # Emit plain assignments only for framework-managed fields
                    # with a Field/field call RHS.
                    result.members.append(
                        MemberInfo(
                            name=lhs_name,
                            parent=class_name,
                            kind='class_attribute',
                            line=line,
                        )
                    )

    def _extract_module_constants(
        self, root_node: Node, result: ParseResult
    ) -> None:
        """Extract top-level module constants into result.members.

        Scans only the root-level expression_statement children; does NOT
        recurse into functions or classes. A constant is emitted when:
        - LHS is a single identifier (not a dunder).
        - RHS is a literal node type, OR the annotation is Final[...].
        """
        for child in root_node.children:
            if child.type != 'expression_statement':
                continue
            for stmt in child.children:
                if stmt.type != 'assignment':
                    continue
                left = stmt.child_by_field_name('left')
                if left is None:
                    continue
                if left.type != 'identifier':
                    continue
                name = left.text.decode('utf8')
                # Dunder exclusion: double-underscore both ends, length > 4.
                if (
                    name.startswith('__')
                    and name.endswith('__')
                    and len(name) > 4
                ):
                    continue
                right = stmt.child_by_field_name('right')
                annotation = stmt.child_by_field_name('type')
                if _is_final_annotation(annotation) or _is_literal_rhs(right):
                    result.members.append(
                        MemberInfo(
                            name=name,
                            parent='',
                            kind='module_constant',
                            line=stmt.start_point[0] + 1,
                        )
                    )

    @staticmethod
    def _is_capital_attr(attr_node: Node) -> tuple[str, str] | None:
        """Return (lhs_ident, attr_name) for a qualifying ``Capital.attr`` node.

        Returns None when the node does not match the pattern:
        - LHS must be a single identifier starting with an uppercase letter.
        - Rejects ``self.X``, ``cls.X``, chains ``pkg.Foo.X``, and numeric LHS.
        """
        if not attr_node.children:
            return None
        lhs_node = attr_node.children[0]
        if lhs_node.type != 'identifier':
            return None
        lhs = lhs_node.text.decode('utf8')
        if not lhs or not lhs[0].isupper():
            return None
        # Reject known non-enum receivers.
        if lhs in ('self', 'cls'):
            return None
        attr = None
        for ch in reversed(attr_node.children):
            if ch.type == 'identifier':
                attr = ch.text.decode('utf8')
                break
        if attr is None or attr == lhs:
            return None
        return (lhs, attr)

    @staticmethod
    def _is_self_attr(
        attr_node: Node, self_param: str
    ) -> tuple[str, str] | None:
        """Return (self_param, attr_name) for ``<self_param>.attr``.

        Only handles the single-level form. Chained ``self.x.y`` is NOT
        matched: the outer node's first child is itself an attribute node,
        so the identifier guard rejects it. This is a documented known limit.
        """
        if not attr_node.children:
            return None
        lhs_node = attr_node.children[0]
        if lhs_node.type != 'identifier':
            return None
        lhs = lhs_node.text.decode('utf8')
        if lhs != self_param:
            return None
        attr = None
        for ch in reversed(attr_node.children):
            if ch.type == 'identifier':
                attr = ch.text.decode('utf8')
                break
        if attr is None or attr == lhs:
            return None
        return (lhs, attr)

    def _try_emit_member_access(
        self,
        attr_node: Node,
        mode: str,
        result: ParseResult,
        ctx: _ParseContext | None = None,
    ) -> None:
        """Emit a MemberAccess for Capital.attr or self.attr."""
        # Capital.attr path - existing behaviour.
        pair = self._is_capital_attr(attr_node)
        if pair is not None:
            lhs, attr = pair
            result.member_accesses.append(
                MemberAccess(
                    parent=lhs,
                    name=attr,
                    line=attr_node.start_point[0] + 1,
                    mode=mode,
                )
            )
            return
        # self.attr path - new in Phase 7.
        self_param = ctx.current_self_param() if ctx is not None else None
        if self_param is None:
            return
        pair = self._is_self_attr(attr_node, self_param)
        if pair is None:
            return
        class_name = ctx.current_class_name()
        if class_name is None:
            return
        result.member_accesses.append(
            MemberAccess(
                parent=class_name,
                name=pair[1],
                line=attr_node.start_point[0] + 1,
                mode=mode,
            )
        )

    def _try_extract_member_access_from_assignment(
        self,
        assignment_node: Node,
        result: ParseResult,
        ctx: _ParseContext | None = None,
    ) -> None:
        """Emit MemberAccess for ``Capital.attr = ...`` or ``self.attr = ...``."""
        left = assignment_node.child_by_field_name('left')
        if left is not None and left.type == 'attribute':
            self._try_emit_member_access(left, 'write', result, ctx)
        # Also scan RHS for read accesses.
        right = assignment_node.child_by_field_name('right')
        if right is not None:
            self._scan_node_for_read_accesses(right, result, ctx)

    def _try_extract_member_access_from_augmented(
        self,
        aug_node: Node,
        result: ParseResult,
        ctx: _ParseContext | None = None,
    ) -> None:
        """Emit MemberAccess for ``Capital.attr += ...`` (both mode).

        Also emits a bare-identifier ALL_CAPS MemberAccess when the LHS is
        an identifier matching the ALL_CAPS pattern (e.g. ``MY_CONST += 1``).
        """
        left = aug_node.child_by_field_name('left')
        if left is not None:
            if left.type == 'attribute':
                self._try_emit_member_access(left, 'both', result, ctx)
            elif left.type == 'identifier':
                name = left.text.decode('utf8')
                if _ALL_CAPS_RE.match(name):
                    result.member_accesses.append(
                        MemberAccess(
                            parent='',
                            name=name,
                            line=aug_node.start_point[0] + 1,
                            mode='both',
                        )
                    )
        right = aug_node.child_by_field_name('right')
        if right is not None:
            self._scan_node_for_read_accesses(right, result, ctx)

    def _scan_node_for_read_accesses(
        self, node: Node, result: ParseResult, ctx: _ParseContext | None = None
    ) -> None:
        """Recursively scan a subtree for member reads.

        Stops at attribute nodes (handled by _try_emit_member_access) and
        at call nodes (handled by _extract_calls_recursive).
        Emits bare-identifier ALL_CAPS reads for module-constant access.
        """
        if node.type == 'attribute':
            self._try_emit_member_access(node, 'read', result, ctx)
            return
        if node.type == 'call':
            # Call nodes: check if function is a Capital.attr (e.g. Foo.BAR())
            func = node.child_by_field_name('function')
            if func is not None and func.type == 'attribute':
                self._try_emit_member_access(func, 'read', result, ctx)
            return
        if node.type == 'identifier':
            name = node.text.decode('utf8')
            # ALL_CAPS heuristic for module-constant reads.
            if _ALL_CAPS_RE.match(name):
                result.member_accesses.append(
                    MemberAccess(
                        parent='',
                        name=name,
                        line=node.start_point[0] + 1,
                        mode='read',
                    )
                )
            return
        for child in node.children:
            self._scan_node_for_read_accesses(child, result, ctx)

    def _extract_import(self, node: Node, result: ParseResult) -> None:
        """Extract a plain ``import X`` statement."""
        # ``import_statement`` children: "import", dotted_name [, ",", dotted_name ...]
        for child in node.children:
            if child.type == "dotted_name":
                module = child.text.decode("utf8")
                # For ``import os.path`` the imported name available locally is "path"
                # (the last segment), but the module is the full dotted path.
                parts = module.split(".")
                result.imports.append(
                    ImportInfo(
                        module=module,
                        names=[parts[-1]],
                    )
                )
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                alias_node = child.child_by_field_name("alias")
                if name_node is not None:
                    module = name_node.text.decode("utf8")
                    parts = module.split(".")
                    alias = alias_node.text.decode("utf8") if alias_node else ""
                    result.imports.append(
                        ImportInfo(
                            module=module,
                            names=[parts[-1]],
                            alias=alias,
                        )
                    )

    def _extract_import_from(self, node: Node, result: ParseResult) -> None:
        """Extract a ``from X import Y`` statement."""
        module_name_node = node.child_by_field_name("module_name")
        if module_name_node is None:
            return

        is_relative = module_name_node.type == "relative_import"
        module = module_name_node.text.decode("utf8")

        names: list[str] = []
        aliases: dict[str, str] = {}
        past_import = False
        for child in node.children:
            if child.type == "import":
                past_import = True
                continue
            if not past_import:
                continue
            # Handle both bare names and parenthesized import lists.
            if child.type in ('dotted_name', 'identifier'):
                names.append(child.text.decode('utf8'))
            elif child.type == 'aliased_import':
                # Single aliased form without parens: from X import Y as Z
                name_node = child.child_by_field_name('name')
                alias_node = child.child_by_field_name('alias')
                if name_node is not None:
                    original = name_node.text.decode('utf8')
                    names.append(original)
                    if alias_node is not None:
                        alias_text = alias_node.text.decode('utf8')
                        aliases[alias_text] = original
            elif child.type == 'import_as_names':
                for sub in child.children:
                    if sub.type in ('dotted_name', 'identifier'):
                        names.append(sub.text.decode('utf8'))
                    elif sub.type == 'aliased_import':
                        name_node = sub.child_by_field_name('name')
                        alias_node = sub.child_by_field_name('alias')
                        if name_node is not None:
                            original = name_node.text.decode('utf8')
                            names.append(original)  # unchanged (back-compat)
                            if alias_node is not None:
                                alias_text = alias_node.text.decode('utf8')
                                aliases[alias_text] = original
            elif child.type == 'wildcard_import':
                names.append('*')

        result.imports.append(
            ImportInfo(
                module=module,
                names=names,
                is_relative=is_relative,
                aliases=aliases,
            )
        )

    def _extract_annotations_from_expression(
        self,
        node: Node,
        result: ParseResult,
    ) -> None:
        """Extract variable annotations and __all__ from an expression_statement."""
        for child in node.children:
            if child.type == "assignment":
                self._try_extract_variable_annotation(child, result)
                self._try_extract_all_exports(child, result)

    def _try_extract_variable_annotation(self, assignment_node: Node, result: ParseResult) -> None:
        """Extract a type reference from a variable annotation if present."""
        type_node = assignment_node.child_by_field_name("type")
        if type_node is None:
            return

        type_name = self._extract_type_name(type_node)
        if type_name and type_name not in _BUILTIN_TYPES:
            result.type_refs.append(
                TypeRef(
                    name=type_name,
                    kind="variable",
                    line=type_node.start_point[0] + 1,
                )
            )

    @staticmethod
    def _try_extract_all_exports(assignment_node: Node, result: ParseResult) -> None:
        """Extract names from ``__all__ = [...]`` or ``__all__ = (...)`` assignments."""
        left = assignment_node.child_by_field_name("left")
        right = assignment_node.child_by_field_name("right")
        if left is None or right is None:
            return
        if left.type != "identifier" or left.text.decode("utf8") != "__all__":
            return
        if right.type not in ("list", "tuple"):
            return

        for child in right.children:
            if child.type == "string":
                text = child.text.decode("utf8")
                # Strip surrounding quotes (single, double, or triple).
                for quote in ('"""', "'''", '"', "'"):
                    if text.startswith(quote) and text.endswith(quote):
                        text = text[len(quote):-len(quote)]
                        break
                if text:
                    result.exports.append(text)

    def _has_staticmethod_decorator(self, func_node: Node) -> bool:
        """Return True when func_node is wrapped in a @staticmethod decorator.

        Checks whether the immediate parent of func_node is a
        decorated_definition whose decorator list contains 'staticmethod'.
        """
        parent = func_node.parent
        if parent is None or parent.type != 'decorated_definition':
            return False
        for child in parent.children:
            if child.type == 'decorator':
                name = self._extract_decorator_name(child)
                if name == 'staticmethod' or name.endswith('.staticmethod'):
                    return True
        return False

    def _extract_calls_recursive(
        self, node: Node, result: ParseResult, ctx: _ParseContext
    ) -> None:
        """Recursively find and extract all call nodes and exception refs."""
        t = node.type
        if t == 'call':
            self._extract_call(node, result, ctx)
            for child in node.children:
                self._extract_calls_recursive(child, result, ctx)
            return

        if t == 'try_statement':
            self._walk_try_statement(node, result, ctx)
            return

        if t == 'function_definition':
            self_param = _extract_self_param_name(node)
            inherited_class = ctx.current_class_name()
            pushed_attribution = False
            is_static = self._has_staticmethod_decorator(node)
            if (
                inherited_class is not None
                and self_param is not None
                and not is_static
            ):
                ctx.push_class_attribution(inherited_class, self_param)
                pushed_attribution = True
            ctx.push_local()
            try:
                body = node.child_by_field_name('body')
                if body is not None:
                    self._extract_calls_recursive(body, result, ctx)
            finally:
                ctx.pop_local()
                if pushed_attribution:
                    ctx.pop_class_attribution()
            return

        if t == 'class_definition':
            body = node.child_by_field_name('body')
            if body is not None:
                self_bindings = _prescan_class_self_bindings(
                    body, ctx.import_local_to_type
                )
                name_node = node.child_by_field_name('name')
                class_name_str = (
                    name_node.text.decode('utf8')
                    if name_node is not None
                    else ''
                )
                ctx.push_class(self_bindings)
                # Push with self_param=None at class-body level; method-level
                # frames (with the actual self param) are pushed when we enter
                # each function_definition inside the class body.
                ctx.push_class_attribution(class_name_str, None)
                try:
                    self._extract_calls_recursive(body, result, ctx)
                finally:
                    ctx.pop_class_attribution()
                    ctx.pop_class()
            return

        if t in {'with_statement', 'async_with_statement'}:
            cm_strings = self._extract_context_manager_strings(node)
            ctx.context_managers.extend(cm_strings)
            ctx.push_local()
            try:
                _try_extract_with_as_bindings(node, ctx)
                for child in node.children:
                    self._extract_calls_recursive(child, result, ctx)
            finally:
                ctx.pop_local()
                for _ in cm_strings:
                    ctx.context_managers.pop()
            return

        if t in {'for_statement', 'while_statement'}:
            ctx.loop_depth += 1
            try:
                for child in node.children:
                    self._extract_calls_recursive(child, result, ctx)
            finally:
                ctx.loop_depth -= 1
            return

        if t == 'await':
            ctx.awaited_depth += 1
            try:
                for child in node.children:
                    self._extract_calls_recursive(child, result, ctx)
            finally:
                ctx.awaited_depth -= 1
            return

        # return self.x / return MY_CONST — scan for member reads.
        if t == 'return_statement':
            for child in node.children:
                self._scan_node_for_read_accesses(child, result, ctx)
            # Also recurse to handle calls inside the return expression.
            for child in node.children:
                self._extract_calls_recursive(child, result, ctx)
            return

        # raise SomeError (without parens) - reference to the exception class.
        if t == 'raise_statement':
            for child in node.children:
                if child.type == 'identifier':
                    result.calls.append(
                        self._make_type_reference_callinfo(
                            name=child.text.decode('utf8'),
                            line=child.start_point[0] + 1,
                        )
                    )
            for child in node.children:
                self._extract_calls_recursive(child, result, ctx)
            return

        # Default: recurse. Capture assignment bindings before recursing.
        if t == 'expression_statement':
            for child in node.children:
                if child.type == 'assignment':
                    _try_extract_binding(child, ctx)
                    self._try_extract_member_access_from_assignment(
                        child, result, ctx
                    )
                elif child.type == 'augmented_assignment':
                    self._try_extract_member_access_from_augmented(
                        child, result, ctx
                    )
                elif child.type == 'attribute':
                    # Bare attribute read: ``Status.PENDING`` as a statement.
                    self._try_emit_member_access(child, 'read', result, ctx)

        for child in node.children:
            self._extract_calls_recursive(child, result, ctx)

    def _walk_try_statement(
        self, node: Node, result: ParseResult, ctx: _ParseContext
    ) -> None:
        """Walk a try_statement's children with correct scope bookkeeping.

        Children layout (tree-sitter-python):
          - "try" keyword, ":"
          - "block"          -- the try body
          - "except_clause"* -- zero or more
          - "else_clause"?   -- optional
          - "finally_clause"? -- optional

        Each arm is walked with the appropriate depth counter incremented.
        Nesting composes correctly because counters are incremented, not set.
        """
        for child in node.children:
            if child.type == 'block':
                ctx.try_depth += 1
                try:
                    self._extract_calls_recursive(child, result, ctx)
                finally:
                    ctx.try_depth -= 1
            elif child.type == 'except_clause':
                # Emit exception-type identifier pseudo-CallInfos with zeroed
                # scope fields (these are type references, not call-sites).
                self._extract_except_type_references(child, result)
                ctx.except_depth += 1
                try:
                    for sub in child.children:
                        if sub.type in {'identifier', 'tuple', 'as_pattern'}:
                            continue
                        self._extract_calls_recursive(sub, result, ctx)
                finally:
                    ctx.except_depth -= 1
            elif child.type == 'finally_clause':
                ctx.finally_depth += 1
                try:
                    self._extract_calls_recursive(child, result, ctx)
                finally:
                    ctx.finally_depth -= 1
            elif child.type == 'else_clause':
                # else runs only if no exception; treat as outside try scope.
                self._extract_calls_recursive(child, result, ctx)

    def _extract_except_type_references(
        self, except_clause_node: Node, result: ParseResult
    ) -> None:
        """Emit exception type identifier CallInfos with zeroed scope fields."""
        for child in except_clause_node.children:
            if child.type == 'identifier':
                result.calls.append(
                    self._make_type_reference_callinfo(
                        name=child.text.decode('utf8'),
                        line=child.start_point[0] + 1,
                    )
                )
            elif child.type == 'tuple':
                # except (ErrorA, ErrorB):
                for elem in child.children:
                    if elem.type == 'identifier':
                        result.calls.append(
                            self._make_type_reference_callinfo(
                                name=elem.text.decode('utf8'),
                                line=elem.start_point[0] + 1,
                            )
                        )
            elif child.type == 'as_pattern':
                # except ErrorA as e  OR  except (ErrorA, ErrorB) as e
                for sub in child.children:
                    if sub.type == 'identifier':
                        result.calls.append(
                            self._make_type_reference_callinfo(
                                name=sub.text.decode('utf8'),
                                line=sub.start_point[0] + 1,
                            )
                        )
                        break
                    if sub.type == 'tuple':
                        for elem in sub.children:
                            if elem.type == 'identifier':
                                result.calls.append(
                                    self._make_type_reference_callinfo(
                                        name=elem.text.decode('utf8'),
                                        line=elem.start_point[0] + 1,
                                    )
                                )
                        break

    @staticmethod
    def _make_type_reference_callinfo(name: str, line: int) -> CallInfo:
        """Return a CallInfo for a type reference with zeroed scope fields."""
        return CallInfo(
            name=name,
            line=line,
            dispatch_kind='direct',
            in_try=_DEFAULT_SCOPE_SNAPSHOT['in_try'],
            in_except=_DEFAULT_SCOPE_SNAPSHOT['in_except'],
            in_finally=_DEFAULT_SCOPE_SNAPSHOT['in_finally'],
            in_loop=_DEFAULT_SCOPE_SNAPSHOT['in_loop'],
            awaited=_DEFAULT_SCOPE_SNAPSHOT['awaited'],
            context_managers=_DEFAULT_SCOPE_SNAPSHOT['context_managers'],
            return_consumption='stored',
        )

    def _extract_context_manager_strings(self, with_node: Node) -> list[str]:
        """Return managed-expression text of each with-item, outer-to-inner.

        Truncates each expression at 80 chars. Skips the as alias target.
        """
        out: list[str] = []
        for item in _iter_with_items(with_node):
            value_node = item.child_by_field_name('value')
            if value_node is None:
                for ch in item.children:
                    if ch.is_named:
                        value_node = ch
                        break
            if value_node is not None:
                text = value_node.text.decode('utf8')
                if len(text) > 80:
                    text = text[:80]
                out.append(text)
        return out

    def _extract_call(
        self, call_node: Node, result: ParseResult, ctx: _ParseContext
    ) -> None:
        """Extract a single call node into a CallInfo."""
        func_node = call_node.child_by_field_name('function')
        if func_node is None:
            for child in call_node.children:
                if child.is_named:
                    func_node = child
                    break
        if func_node is None:
            return

        line = call_node.start_point[0] + 1
        arguments = self._extract_identifier_arguments(call_node)
        snap = ctx.snapshot()
        dispatch_kind = _classify_dispatch_kind(call_node, result, ctx)
        return_consumption = _classify_return_consumption(call_node)

        if func_node.type == 'identifier':
            result.calls.append(
                CallInfo(
                    name=func_node.text.decode('utf8'),
                    line=line,
                    arguments=arguments,
                    dispatch_kind=dispatch_kind,
                    in_try=snap['in_try'],
                    in_except=snap['in_except'],
                    in_finally=snap['in_finally'],
                    in_loop=snap['in_loop'],
                    awaited=snap['awaited'],
                    context_managers=snap['context_managers'],
                    return_consumption=return_consumption,
                )
            )
        elif func_node.type == 'attribute':
            name, receiver = self._extract_attribute_call(func_node)
            result.calls.append(
                CallInfo(
                    name=name,
                    line=line,
                    receiver=receiver,
                    arguments=arguments,
                    dispatch_kind=dispatch_kind,
                    in_try=snap['in_try'],
                    in_except=snap['in_except'],
                    in_finally=snap['in_finally'],
                    in_loop=snap['in_loop'],
                    awaited=snap['awaited'],
                    context_managers=snap['context_managers'],
                    return_consumption=return_consumption,
                )
            )

    def _extract_attribute_call(self, attr_node: Node) -> tuple[str, str]:
        """Extract (method_name, receiver) from an attribute node.

        For chained calls like ``obj.method1().method2()``, the outer call's
        function is ``attribute(call(...), "method2")``.  We extract
        ``method2`` as the name and the first identifier in the chain as
        the receiver.
        """
        method_name = ""
        for child in reversed(attr_node.children):
            if child.type == "identifier":
                method_name = child.text.decode("utf8")
                break

        receiver = ""
        obj_node = attr_node.children[0] if attr_node.children else None
        if obj_node is not None:
            if obj_node.type == "identifier":
                receiver = obj_node.text.decode("utf8")
            elif obj_node.type == "attribute":
                # Nested attribute access like ``self.logger.info()`` — use the root.
                receiver = self._root_identifier(obj_node)
            elif obj_node.type == "call":
                # Chained call like ``get_user().save()`` — try the innermost identifier.
                receiver = self._root_identifier(obj_node)

        return method_name, receiver

    @staticmethod
    def _extract_identifier_arguments(call_node: Node) -> list[str]:
        """Extract bare identifier arguments from a call node.

        Returns names of arguments that are plain identifiers (not literals,
        calls, or attribute accesses) — these are likely callback references
        like ``map(transform, items)`` or ``Depends(get_db)``.
        """
        args_node = call_node.child_by_field_name("arguments")
        if args_node is None:
            return []

        identifiers: list[str] = []
        for child in args_node.children:
            if child.type == "identifier":
                identifiers.append(child.text.decode("utf8"))
            elif child.type == "keyword_argument":
                value_node = child.child_by_field_name("value")
                if value_node is not None and value_node.type == "identifier":
                    identifiers.append(value_node.text.decode("utf8"))
        return identifiers

    def _root_identifier(self, node: Node) -> str:
        """Walk down into the leftmost identifier of an expression."""
        current = node
        while current is not None:
            if current.type == "identifier":
                return current.text.decode("utf8")
            if current.children:
                current = current.children[0]
            else:
                break
        return ""

    @staticmethod
    def _extract_type_name(type_node: Node) -> str:
        """Extract the primary type name from a type annotation node.

        For simple types like ``User``, returns ``"User"``.
        For generic types like ``list[User]``, returns ``"list"``.
        For complex types, returns the text of the first identifier found.
        """
        if type_node.type == "type" and type_node.children:
            inner = type_node.children[0]
            if inner.type == "identifier":
                return inner.text.decode("utf8")
            if inner.type == "generic_type":
                # e.g., ``Optional[User]`` — return "Optional"
                for child in inner.children:
                    if child.type == "identifier":
                        return child.text.decode("utf8")
            # Fallback: return text of first identifier found anywhere.
            return PythonParser._find_first_identifier(inner)
        if type_node.type == "identifier":
            return type_node.text.decode("utf8")
        return PythonParser._find_first_identifier(type_node)

    @staticmethod
    def _find_first_identifier(node: Node) -> str:
        """DFS for the first identifier node."""
        if node.type == "identifier":
            return node.text.decode("utf8")
        for child in node.children:
            found = PythonParser._find_first_identifier(child)
            if found:
                return found
        return ""
