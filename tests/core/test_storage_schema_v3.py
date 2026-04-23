"""Tests for KuzuBackend schema version 3 checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from axon.core.storage.kuzu_backend import _SCHEMA_VERSION, KuzuBackend


@pytest.fixture()
def fresh_db_path(tmp_path: Path) -> Path:
    """Return a path inside tmp_path suitable for a fresh KuzuDB."""
    return tmp_path / 'schema_test_db'


class TestSchemaVersion:
    """Fresh and existing databases handle schema version correctly."""

    def test_fresh_db_initializes_at_current_version(
        self, fresh_db_path: Path
    ) -> None:
        """A new DB writes _SCHEMA_VERSION into _Metadata on first open."""
        b = KuzuBackend()
        b.initialize(fresh_db_path)
        stored = b._read_stored_schema_version()
        b.close()
        assert stored == _SCHEMA_VERSION

    def test_fresh_db_write_mode_opens_without_error(
        self, fresh_db_path: Path
    ) -> None:
        """Opening a brand new directory in write mode does not raise."""
        b = KuzuBackend()
        b.initialize(fresh_db_path)
        b.close()

    def test_check_schema_version_write_mode_rejects_older_version(
        self, fresh_db_path: Path
    ) -> None:
        """_check_schema_version_write_mode raises for an outdated schema."""
        # Create the DB first so _Metadata table exists.
        b = KuzuBackend()
        b.initialize(fresh_db_path)
        # Overwrite the schema_version with a fake old value.
        b._conn.execute(
            "MERGE (_m:_Metadata {key: 'schema_version'}) SET _m.value = '2'"
        )
        b.close()

        b2 = KuzuBackend()
        with pytest.raises(RuntimeError, match='axon clean && axon analyze'):
            b2.initialize(fresh_db_path)
        b2.close()

    def test_check_schema_version_read_only_rejects_older_version(
        self, fresh_db_path: Path
    ) -> None:
        """_check_schema_version raises for a DB at schema version 2 (read-only)."""
        b = KuzuBackend()
        b.initialize(fresh_db_path)
        b._conn.execute(
            "MERGE (_m:_Metadata {key: 'schema_version'}) SET _m.value = '2'"
        )
        b.close()

        b2 = KuzuBackend()
        with pytest.raises(RuntimeError, match='axon clean && axon analyze'):
            b2.initialize(fresh_db_path, read_only=True)
        b2.close()

    def test_read_stored_schema_version_absent_returns_one(
        self, fresh_db_path: Path
    ) -> None:
        """_read_stored_schema_version returns 1 when _Metadata row is absent."""
        b = KuzuBackend()
        b.initialize(fresh_db_path)
        # Remove the row inserted by _create_schema.
        b._conn.execute(
            "MATCH (m:_Metadata) WHERE m.key = 'schema_version' DETACH DELETE m"
        )
        stored = b._read_stored_schema_version()
        b.close()
        assert stored == 1

    def test_read_stored_schema_version_returns_stored_value(
        self, fresh_db_path: Path
    ) -> None:
        """_read_stored_schema_version returns the value written by _create_schema."""
        b = KuzuBackend()
        b.initialize(fresh_db_path)
        stored = b._read_stored_schema_version()
        b.close()
        assert stored == _SCHEMA_VERSION

    def test_error_message_contains_rebuild_instruction(
        self, fresh_db_path: Path
    ) -> None:
        """RuntimeError message includes the rebuild command for both modes."""
        b = KuzuBackend()
        b.initialize(fresh_db_path)
        b._conn.execute(
            "MERGE (_m:_Metadata {key: 'schema_version'}) SET _m.value = '2'"
        )
        b.close()

        b2 = KuzuBackend()
        with pytest.raises(RuntimeError, match='axon clean') as exc_info:
            b2.initialize(fresh_db_path)
        b2.close()
        assert 'axon analyze' in str(exc_info.value)
