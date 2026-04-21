# Changelog

## Unreleased

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
