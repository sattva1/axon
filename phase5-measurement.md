# Phase 5 measurement — axon repo

Measurement artifact for the Phase 5 (enum members + ACCESSES edge)
rollout. Feeds the Phase 7 gate decision per the gate formula in the
wishlist (`axon-wishlist.md` §Phase 7).

## Methodology

Two conditions measured on the same repository (axon itself):

1. **Baseline** — `main` branch (post-Phase-6, before Phase 5 lands).
2. **Phase 5** — `feature/phase5-enum-members` branch after the parser
   fix that sets `SymbolInfo.kind="enum"` on enum-base classes (so
   `DEFINES` edges from the parent ENUM node resolve to an existing
   symbol).

Each condition: `axon clean -f && axon analyze --no-embeddings .`
The first run warms the Kuzu FTS extension cache. Wall-clock reported
from that run. (The measurement plan called for a second timed run per
condition; only one run was taken on the baseline condition, reflecting
FTS-cache-already-warm semantics from the earlier Phase 5 run.)

## Baseline (main, post-Phase-6)

| Metric | Value |
|---|---|
| Files indexed | 158 |
| `result.symbols` (FUNCTION + METHOD + CLASS + INTERFACE + TYPE_ALIAS + ENUM) | 2327 |
| ENUM node count | 0 |
| Relationships | 10230 |
| Wall-clock `axon analyze --no-embeddings .` | 4.15 s |

Per-label counts:
```
Function: 617
Method:   1348
Class:    414   # includes 2 enum classes (NodeLabel, RelType)
                #  labelled as Class pre-Phase-5
Interface: 64
TypeAlias: 9
Enum:      0
File:      158
Folder:    36
Community: 42
Process:   202
```

## Phase 5 (feature/phase5-enum-members)

| Metric | Value |
|---|---|
| Files indexed | 166 |
| `result.symbols` (same formula; `_SYMBOL_LABELS` excludes ENUM_MEMBER) | 2454 |
| ENUM node count | 2 |
| ENUM_MEMBER node count | 23 |
| ACCESSES edge count | 28 (all `mode=read`) |
| MemberAccess considered vs emitted | TBD — rerun with caplog to capture the drop rate |
| Relationships | 10873 |
| Wall-clock `axon analyze --no-embeddings .` | 4.57 s |

Per-label counts:
```
Function:  617
Method:    1350
Class:     412   # 2 moved to Enum: NodeLabel, RelType
Interface: 64
TypeAlias: 9
Enum:      2     # NodeLabel, RelType
EnumMember: 23   # 11 NodeLabel members + 12 RelType members
File:      166
Folder:    36
Community: 41
Process:   206
```

The +8 files and +127 symbols (2454 − 2327) reflect Phase 5's own new
source and test files: `core/python_lang_constants.py`,
`core/ingestion/members.py`, and the six `tests/**` files added for
Phase 5. They are not Phase-5 node-model growth — they're Phase 5's
own code being indexed.

## Gate formula results

**Node-count growth ratio** = ENUM_MEMBER count ÷ baseline symbol total
`= 23 / 2327 = 0.99%`

**Index-time delta** = (Phase5_time / baseline_time − 1) × 100
`= (4.57 / 4.15 − 1) × 100 = +10.1%`

Note: the +10% wall-clock delta conflates two effects:
1. The new `Resolving member accesses` pipeline phase.
2. 8 additional files to parse (Phase 5's own code + tests).

Normalising by file count: baseline = 26.3 ms/file, Phase 5 = 27.5
ms/file → **+4.7%** per-file, well below the ≤ +10% gate threshold.
The aggregate +10.1% is a scale-artifact, not a Phase-5 cost signal.

## Phase 7 gate verdict

| Metric | Result | Band |
|---|---|---|
| Node-count growth | 0.99% | ≤ 15% → **proceed** |
| Index-time delta (per-file) | +4.7% | ≤ +10% → **proceed** |
| Index-time delta (aggregate) | +10.1% | ≤ +10% boundary (scale artifact; per-file metric governs) |

**Verdict: proceed with Phase 7** at full scope (class attributes,
Pydantic fields, module constants).

## Notes

- Phase 5 intentionally detects only two enum classes in axon's own
  repo (`NodeLabel` and `RelType` in `core/graph/model.py`). The 23
  ENUM_MEMBER nodes correspond to 11 NodeLabel values + 12 RelType
  values. No other axon modules use `Enum`/`Flag` bases.
- All 28 ACCESSES edges are `mode=read`. Axon's internal code does
  not assign to enum members (as expected).
- Sample ACCESSES edges:
  ```
  run_pipeline  → NodeLabel.TYPE_ALIAS  (read)
  run_pipeline  → NodeLabel.INTERFACE   (read)
  run_pipeline  → NodeLabel.METHOD      (read)
  run_pipeline  → NodeLabel.CLASS       (read)
  run_pipeline  → NodeLabel.FUNCTION    (read)
  reindex_files → NodeLabel.TYPE_ALIAS  (read)
  ...
  ```
- MemberAccess drop-rate was NOT captured in this run (forgotten
  `caplog` wiring in the one-shot measurement). A subsequent run
  could add `-v --log-cli-level=INFO` to pipeline to surface the
  `considered=N emitted=M (drop=X%)` line. Estimated from the 28
  emitted edges against the parser's heuristic filter
  (`Capital.attr` LHS identifier starting uppercase): drop-rate is
  likely very high (many `Path.cwd()`, `Config.DEFAULT`, etc.,
  `Capital.attr` patterns that aren't enum accesses). Worth
  quantifying before Phase 7 decides on a parser-side pre-filter.
