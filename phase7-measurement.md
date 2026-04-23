# Phase 7 measurement — axon repo

Measurement artifact for the Phase 7 (class attributes + module constants
+ self-attr attribution + bare-identifier reads) rollout. Compared
against the Phase 5 baseline recorded in `phase5-measurement.md`.

## Methodology

Three consecutive timed runs of
`axon clean -f && axon analyze --no-embeddings .` on the axon repo
itself, using the `feature/phase7-class-attrs-module-consts` branch
after the full implementation landed (parser, ingestion, storage, MCP
tools + tests).

The first run warms the Kuzu FTS extension cache. Wall-clock is reported
from the steady-state median across three runs (5.77, 5.45, 5.51 s).

## Phase 5 baseline (from `phase5-measurement.md`)

| Metric | Value |
|---|---|
| Files indexed | 166 |
| `result.symbols` (excluding member kinds) | 2454 |
| Relationships | 10873 |
| `EnumMember` nodes | 23 |
| `ACCESSES` edges | 28 (all read) |
| Wall-clock | 4.57 s |
| Per-file index time | 27.5 ms/file |

## Phase 7

| Metric | Value |
|---|---|
| Files indexed | 170 |
| `result.symbols` (excluding member kinds) | 2615 |
| Relationships | 11804 |
| `EnumMember` nodes | 25 |
| `ClassAttribute` nodes | 143 (new label) |
| `ModuleConstant` nodes | 71 (new label) |
| `ACCESSES` edges | 46 (read + write modes) |
| Wall-clock (median of 3) | 5.51 s |
| Per-file index time | 32.4 ms/file |

### Per-label counts

```
File:             170
Folder:            36
Function:         637
Class:            432
Method:          1471
Interface:         64
TypeAlias:          9
Enum:               2   (NodeLabel, RelType)
EnumMember:        25   (+2 vs Phase 5 — added via new test fixtures)
ClassAttribute:   143   (NEW)
ModuleConstant:    71   (NEW)
Community:         43
Process:          209
```

### Sample distribution

- ClassAttribute growth (143 nodes) reflects axon's type-heavy Python
  (dataclasses, Pydantic-like models, annotated container classes).
- ModuleConstant growth (71 nodes) captures shared constants across
  parser, ingestion, storage, and MCP layers (`ENUM_BASES`,
  `PYDANTIC_BASES`, `DATACLASS_DECORATORS`, `_SCHEMA_VERSION`,
  `_MEMBER_LABELS`, `_LITERAL_NODE_TYPES`, `_CELERY_DECORATORS`, etc.).
- The +4 files vs Phase 5 baseline are Phase 7's own new source files
  (`axon` additions minimal) and new test files
  (`test_parser_python_members.py`, `test_reindex_members.py`,
  `test_storage_schema_v4.py`, `test_tools_member.py`).

## Gate formula results

Gate (per wishlist §Phase 7):

- ≤ 15% node-count growth AND ≤ +10% index-time delta → **proceed full scope**.
- Either metric 15–25% → scope-cut: Pydantic fields + module constants only.
- Either metric > 25% → re-scope.

**Node-count growth ratio** = new member nodes ÷ Phase 5 symbol count
`= (143 + 71) / 2454 = 8.72 %`  ≤ 15% → **proceed**

**Index-time delta (per-file)** = per-file delta vs. Phase 5
`= (32.4 − 27.5) / 27.5 = +17.8 %`  **15–25% band → scope-cut (per wishlist)**

**Index-time delta (aggregate)** = wall-clock delta
`= (5.51 − 4.57) / 4.57 = +20.6 %`  **15–25% band → scope-cut (per wishlist)**

### Notes on the time delta

The per-file time cost has several contributors, in order of likely
impact:

1. **New parser passes**: `_extract_class_attributes` (iterates every
   non-enum class body), `_extract_module_constants` (one pass over
   the file root), `_has_staticmethod_decorator` lookup per
   `function_definition` entry inside a class.
2. **ALL_CAPS identifier scan in all RHS subtrees** of
   `_scan_node_for_read_accesses` — many identifier nodes are now
   checked against a regex on every assignment and return statement.
3. **Two additional ingestion phases**:
   `build_module_constant_index`, `build_imported_names` — the latter
   calls `resolve_import_path` per `ImportInfo`, which re-parses
   path components for each import in each file.
4. The `build_imported_names` step adds O(files × imports-per-file)
   import resolution work that was not present in Phase 5.

## Verdict

The node-count growth gate **passes clearly** (8.72%, well under 15%).

The index-time gate **breaches the ≤ +10% threshold**, landing in the
15–25% band that the wishlist maps to "scope-cut: Pydantic fields +
module constants only; defer plain class attributes."

Two reasonable disposition paths:

**(A) Ship Phase 7 at full scope and accept the cost.**
- Rationale: the per-file delta is bounded (+17.8%), wall-clock is
  still under 6 s on a ~170-file repo, the node-count gate passed,
  and the functionality delivered (class attributes + module constants
  + self-attr + ALL_CAPS) is the core user value of FR-3.
- Risk: larger repos (e.g., SGT-972) would scale the absolute impact;
  a 10k-file repo would add ~10–15s on cold analyze.

**(B) Scope-cut per the wishlist guidance.**
- Keep Pydantic fields + module constants; revert plain class-attribute
  extraction (and the `DATACLASS_DECORATORS`/`is_dataclass` gating).
- Expected: ~half the parser-side cost goes away (class attributes are
  the dominant new node count, 143/(143+71)=66.8 %).

This measurement artifact records the state; the disposition is a
decision point for the user, since the gate breach is real.
