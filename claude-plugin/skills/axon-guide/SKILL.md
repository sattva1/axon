---
name: axon-guide
description: Routing guide for the Axon code-graph MCP. Consult BEFORE grep, find, or multi-file Read sweeps when answering structural questions about a codebase — locating symbols, finding callers and callees, impact of a change, call paths, dead code, circular dependencies, module clusters, file coupling, concurrent dispatch, PR risk, or test selection. Maps common task phrasings to the right axon_* tool and lists the anti-patterns to avoid.
when_to_use: Pull this skill when the task involves any of — "who calls X", "uses of X", "references to", "what breaks if I change", "impact", "blast radius", "downstream", "trace call", "shortest path", "find symbol", "where is X defined", "overview of <file>", "before editing this file", "review PR", "risk", "what tests cover", "test selection", "dead code", "unused", "cycle", "circular import", "cluster", "community", "module boundary", "coupling", "co-change", "files that change together", "race", "concurrency", "thread safety", "auth handlers", "classes that validate". Also pull whenever an agent is about to grep/find for a function, class, or method name, or read multiple files in sequence looking for callers, definitions, or usages.
---

# Axon Routing Guide

Axon is a precomputed code knowledge graph. Every `axon_*` tool answers a structural question in **one call** that would otherwise take a 5–30 step grep/Read chain. This guide tells you which tool to reach for and — just as importantly — when to stop searching.

## The decision rule

Before reaching for `Grep`, `find`, or another `Read`, ask:

> **Is this a structural question** (callers, impact, dependencies, communities, dead code, cycles, file overview, PR risk, test selection)?

If yes → use an `axon_*` tool. If no (literal text search, error message, comment scan, log line) → use `Grep` / `Read`.

After a successful axon call you will have **file paths and line numbers**. The next step is a *targeted* `Read` of those specific ranges — not another grep.

---

## Tool routing table

| You want to… | Use | Beats grep because |
| --- | --- | --- |
| Find a symbol by **concept** ("auth handlers", "things that validate") | `axon_query` | hybrid keyword + vector, ranked, grouped by execution flow |
| Get the **full neighborhood** of a named symbol (callers + callees + types + importers + community) | `axon_context` | one call replaces ~5 greps |
| Know **what breaks** if I change `X` | `axon_impact` | full upstream graph with depth labels and confidence; grep only finds in-file callers |
| **Plain-English description** of what `X` does and where it fits | `axon_explain` | narrative summary, role flags, process flows |
| **Shortest call chain** from `A` to `B` | `axon_call_path` | BFS over CALLS edges with dispatch annotations |
| **Everything about a single file** before you edit it | `axon_file_context` | symbols + imports in/out + coupling + dead code + community in one call |
| **Files that change together** with `X` (hidden coupling) | `axon_coupling` | git co-change history; grep can't see this |
| List **dead / unreachable** symbols | `axon_dead_code` | reachability already computed at index time |
| Detect **circular dependencies** | `axon_cycles` | SCC analysis; grep sees edges, not paths |
| Show **module clusters** (Leiden communities) | `axon_communities` | computed over the full call graph |
| Find symbols that may **race / run concurrently** with `X` | `axon_concurrent_with` | follows non-direct dispatch edges (executors, tasks, callbacks) |
| Map a **diff** to changed symbols | `axon_detect_changes` | hunk-line → symbol, then feed into `axon_impact` |
| Score **PR risk** (LOW/MEDIUM/HIGH) | `axon_review_risk` | composite of blast radius, missing co-changes, community spans |
| Pick **tests affected** by a diff or set of symbols | `axon_test_impact` | traces upstream callers up to test files |
| Ad-hoc structural query no other tool answers | `axon_cypher` | last resort — read-only Cypher (read `axon://schema` first) |
| Discover indexed repos and their freshness | `axon_list_repos` | run this once if multi-repo or unsure |

---

## Anti-patterns (observed real failures)

These are mistakes the guide is here to prevent. If you catch yourself doing one of these, **stop and re-route to axon**.

1. **`find … -name "transaction.py"`** to locate a file → `axon_query "Transaction"` or `axon_context "Transaction"` returns the file path directly.
2. **`grep -n "set_request|get_request|merge_fields|..."` in a file** → `axon_context "set_request"` lists every caller, callee, and type reference in one call.
3. **`grep -rn "display_quantity"`** across `src/` → `axon_query "display_quantity"` ranks hits by relevance and groups by execution flow.
4. **Reading the same file in adjacent slices** (`Read lines 1-50`, then `100-159`, then `270-349`, then `780-839`) to find structure → `axon_file_context <path>` returns symbols + imports + coupling + dead code in one call. Do this **before** any line-range Read.
5. **Manually grepping for callers of a function** ("who calls `_process_order_modification`?") → `axon_impact "_process_order_modification"` returns the full caller graph with depth labels.
6. **Cross-repo grep across multiple checkouts** → `axon_list_repos` then pass `repo=<slug>`.
7. **Grepping a diff for impact** ("what does this PR break?") → `axon_review_risk` with the raw `git diff`.
8. **Searching for tests that cover changed code** → `axon_test_impact` with the diff or symbol list.
9. **Calling `axon_context` then immediately grepping the result** to verify — the result is authoritative; just `Read` the file:line it points to.

---

## Multi-repo posture

- Default = the **local repo** (the one Claude Code is invoked in).
- Need a foreign repo? Pass `repo=<slug>` (slug, absolute path, or relative path).
- Don't know slugs? Run `axon_list_repos` once and cache the answer in your head.
- Cross-repo querying tools that take a **diff** (`axon_detect_changes`, `axon_review_risk`, `axon_test_impact`) **refuse cross-repo diffs** — split the diff first.
- **Stale-major** repos refuse queries with a re-index hint (`axon analyze`). Stale-minor still answers.
- Single-symbol tools (`axon_context`, `axon_explain`, `axon_impact`, `axon_call_path`, `axon_concurrent_with`) append a **cross-repo footer** when the symbol also exists in foreign repos — follow the footer if the local hit was wrong.

---

## When axon is the wrong tool

axon answers **structural** questions. For these, reach for `Grep` / `Read` directly:

- Literal string search (error messages, log lines, magic strings, environment-variable names).
- Comment / docstring / TODO / FIXME scans.
- Reading source after axon has already located the file:line — just `Read` that range.
- Code that hasn't been indexed yet (just-edited buffers, new files added since the last `axon analyze` if `--watch` isn't running).
- Non-source files (configs, JSON, YAML, fixtures, vendored code).
- Repos where `axon_list_repos` shows no entry — index them first or grep as a fallback.

If `axon_list_repos` shows the watcher is dead or the repo is stale-major, **note it** and consider grep as a temporary fallback while you (or the user) re-runs `axon analyze`.

---

## Worked examples

### Example 1 — "What breaks if I change `Transaction.set_request`?"

Wrong:
```
Grep "set_request" -rn src/
Read core/models/transaction.py 1-50
Read core/models/transaction.py 270-409
Read core/models/transaction.py 780-839
Grep "merge_fields" -rn src/
… (15 more steps)
```

Right:
```
axon_context  symbol="set_request"          # locate it, see callers + callees
axon_impact   symbol="set_request" depth=3  # full upstream blast radius
Read          <file>:<line-range from results>   # only when you need the source
```

### Example 2 — "Give me an overview of `processor.py` before I edit it."

Wrong: read the file in 5 slices, grep for symbols, grep for imports.

Right:
```
axon_file_context  file_path="src/.../processor.py"
# Returns: every indexed symbol, imports in/out, top coupled files, dead code, community.
# Then Read only the specific functions you need to change.
```

### Example 3 — "Will this PR break anything? Pick relevant tests."

```
git diff main… > /tmp/pr.diff
axon_review_risk  diff=<contents of /tmp/pr.diff>
axon_test_impact  diff=<contents of /tmp/pr.diff>
```

You now have a risk score, the changed symbols' downstream dependents, and the affected test files — with no grep at all.

### Example 4 — "Find handlers that authenticate users."

Wrong: `grep -rn "auth"`, then disambiguate by hand.

Right:
```
axon_query  query="auth handler validate user credentials"  limit=10
```

Returns ranked results grouped by execution flow.

### Example 5 — "Are there race conditions around `set_request`?"

```
axon_concurrent_with  symbol="set_request" depth=3
```

Returns symbols reachable through executor / detached-task / callback edges — the actual race candidates, not just `executor.submit` call sites that grep would find.

---

## Quick reference — required parameters at a glance

- `axon_query` → `query` (and optional `limit`, `repo`)
- `axon_context` / `axon_explain` / `axon_impact` / `axon_concurrent_with` → `symbol` (+ `depth` where applicable, + `repo`)
- `axon_call_path` → `from_symbol`, `to_symbol` (+ `max_depth`, `repo`)
- `axon_file_context` / `axon_coupling` → `file_path` (+ `repo`)
- `axon_detect_changes` / `axon_review_risk` → `diff`
- `axon_test_impact` → `diff` **or** `symbols` (at least one)
- `axon_cypher` → `query` (read-only Cypher)
- `axon_communities` → optional `community` to drill in, otherwise lists all
- `axon_cycles` → optional `min_size` (default 2)
- `axon_dead_code` / `axon_list_repos` → no required arguments

---

## TL;DR

If the question is **structural**, axon answers it in one call. After axon gives you a file path and line, **stop searching** and `Read` that range. Use grep only for literals, comments, and unindexed content.
