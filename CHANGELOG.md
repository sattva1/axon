# Changelog

## Unreleased

### Multi-repo MCP support

A single `axon serve --watch` session can now query any other indexed repo on
the same machine via the new `repo=<slug>` parameter on all multi-repo tools.

**Drift detection (prerequisite)**

- Four-tier staleness probe: live-watcher recency check (tier 0), git HEAD
  comparison (tier 1), sentinel-file mtime scan (tier 2), and indexed-directory
  mtime scan (tier 3). Covers new files in source subdirectories and new
  top-level directories.
- Session-level `DriftCache` (30 s TTL) avoids re-probing the same repo on
  every tool call.
- `MetaFile` extended with `head_sha_at_index`, `repo_root_mtime`,
  `indexed_file_count`, `sentinel_files`, and `indexed_dirs`. Written at index
  and watcher commit-transition time.
- `update_meta` uses `fcntl.flock` to prevent watcher/analyzer interleave on
  writes.

**Multi-repo MCP (Phases 2-4)**

- New `repo=<slug>` parameter on `axon_context`, `axon_query`, `axon_impact`,
  `axon_explain`, `axon_call_path`, `axon_concurrent_with`, `axon_coupling`,
  `axon_file_context`, `axon_review_risk`, `axon_test_impact`,
  `axon_detect_changes`, `axon_cypher`, `axon_communities`, `axon_dead_code`,
  `axon_cycles`. Accepts slug, absolute path, or relative path.
- Path-keyed tools (`axon_coupling`, `axon_file_context`) and diff-keyed tools
  (`axon_review_risk`, `axon_test_impact`, `axon_detect_changes`) auto-route
  to the owning repo when `repo=` is omitted.
- Symbol-keyed tools append a cross-repo "Also exists in" footer from accessible
  foreign repos, and return a redirect response when the symbol is absent locally
  but present in a foreign repo.
- `axon_query` appends a per-repo hit-count footer from up to 5 foreign repos.
- `axon_list_repos` upgraded: uses the registry resolver (no glob scan), shows
  `Freshness:`, `Watcher:`, and `Reachable:` per entry, marks the local repo
  `(LOCAL)`, and appends a usage hint footer.
- Lazy `RepoPool` opens foreign Kuzu databases read-only on first access and
  caches connections for the session lifetime.
- Thread-safe lazy init for the resolver/pool/drift-cache via `asyncio.Lock`.
- Schema-version symmetry: foreign repos with a newer schema than the server
  are refused with an upgrade hint.

**Staleness handling (Phase 5)**

- Stale-minor foreign repos: a one-liner warning is prepended to every tool
  response noting the drift reason.
- Stale-major foreign repos: queries are refused before the handler runs, with
  a hint to re-run `axon analyze` in that repo.
- `DriftReport` carries an optional `slug` field; renderers receive a decorated
  copy so the warning can name the repo.

### Upgrade ordering

Phase 1 adds a `metadata_json` overflow column to the `CodeRelation` rel schema.
The column is created idempotently by any writer process (`axon index`,
`axon watch`). Because the MCP server (`axon serve`) opens the DB read-only
and cannot run ALTER TABLE, you must upgrade in this order:

1. Stop `axon serve` (MCP server).
2. Run `axon index` or let the watcher run once so the schema migrates.
3. Restart `axon serve`.

If you start the MCP server first after upgrading, it will refuse to start and
report the expected schema version. Re-run `axon index` and restart.
