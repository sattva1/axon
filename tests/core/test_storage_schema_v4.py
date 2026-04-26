"""Tests for KuzuBackend schema version 4 checks (Phase 7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from axon.core.storage.kuzu_backend import _SCHEMA_VERSION, KuzuBackend


class TestSchemaVersionFour:
    """Fresh DB writes v4; v3-tagged DB is rejected in write mode."""

    def test_fresh_db_writes_version_four(self, tmp_path: Path) -> None:
        """A new DB stores _SCHEMA_VERSION=4 in _Metadata."""
        assert _SCHEMA_VERSION == 4
        b = KuzuBackend()
        b.initialize(tmp_path / 'db')
        stored = b._read_stored_schema_version()
        b.close()
        assert stored == 4

    def test_v3_db_write_mode_raises_runtime_error(
        self, tmp_path: Path
    ) -> None:
        """A DB tagged with schema_version=3 raises RuntimeError in write mode."""
        db_path = tmp_path / 'db'
        b = KuzuBackend()
        b.initialize(db_path)
        b._conn.execute(
            "MERGE (_m:_Metadata {key: 'schema_version'}) SET _m.value = '3'"
        )
        b.close()

        b2 = KuzuBackend()
        with pytest.raises(RuntimeError, match='axon clean && axon analyze'):
            b2.initialize(db_path)
        b2.close()

    def test_v3_db_read_only_raises_runtime_error(
        self, tmp_path: Path
    ) -> None:
        """A DB tagged with schema_version=3 raises RuntimeError in read-only mode."""
        db_path = tmp_path / 'db'
        b = KuzuBackend()
        b.initialize(db_path)
        b._conn.execute(
            "MERGE (_m:_Metadata {key: 'schema_version'}) SET _m.value = '3'"
        )
        b.close()

        b2 = KuzuBackend()
        with pytest.raises(RuntimeError, match='axon clean && axon analyze'):
            b2.initialize(db_path, read_only=True)
        b2.close()

    def test_current_version_is_four(self) -> None:
        """_SCHEMA_VERSION module constant equals 4."""
        assert _SCHEMA_VERSION == 4

    def test_error_message_contains_rebuild_instruction(
        self, tmp_path: Path
    ) -> None:
        """RuntimeError raised for v3 DB includes 'axon clean' and 'axon analyze'."""
        db_path = tmp_path / 'db'
        b = KuzuBackend()
        b.initialize(db_path)
        b._conn.execute(
            "MERGE (_m:_Metadata {key: 'schema_version'}) SET _m.value = '3'"
        )
        b.close()

        b2 = KuzuBackend()
        with pytest.raises(RuntimeError) as exc_info:
            b2.initialize(db_path)
        b2.close()
        msg = str(exc_info.value)
        assert 'axon clean' in msg
        assert 'axon analyze' in msg

    def test_check_schema_version_rejects_newer(self, tmp_path: Path) -> None:
        """read-only open of a DB with stored version > _SCHEMA_VERSION raises RuntimeError.

        The error message must direct the user to upgrade the CLI, not to rebuild.
        """
        db_path = tmp_path / 'db'
        future_version = _SCHEMA_VERSION + 1
        b = KuzuBackend()
        b.initialize(db_path)
        b._conn.execute(
            f"MERGE (_m:_Metadata {{key: 'schema_version'}}) "
            f"SET _m.value = '{future_version}'"
        )
        b.close()

        b2 = KuzuBackend()
        with pytest.raises(RuntimeError, match='Upgrade the axon CLI'):
            b2.initialize(db_path, read_only=True)
        b2.close()
