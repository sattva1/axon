"""Tests that CLI commands wrap RuntimeError from storage into typer.Exit(1)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from axon.cli.main import app

runner = CliRunner()

_FAKE_ERROR = (
    'Kuzu DB at /fake/db is on schema version 2 but this code expects '
    'version 3. Run `axon clean && axon analyze` to rebuild.'
)


@pytest.fixture(autouse=True)
def suppress_update_notice():
    """Silence update checker so it never interferes with exit codes."""
    with patch('axon.cli.main._maybe_notify_update'):
        yield


@pytest.fixture()
def repo_with_axon_dir(tmp_path: Path) -> Path:
    """Create a tmp dir that looks like a repo with an existing .axon dir."""
    axon_dir = tmp_path / '.axon'
    axon_dir.mkdir()
    db_dir = axon_dir / 'kuzu'
    db_dir.mkdir()
    # meta.json presence allows _has_existing_index to pass.
    (axon_dir / 'meta.json').write_text('{}', encoding='utf-8')
    return tmp_path


class TestAnalyzeSchemaErrorExit:
    """analyze command exits cleanly when storage raises RuntimeError."""

    def test_exit_code_is_one(self, repo_with_axon_dir: Path) -> None:
        """Exit code 1 when KuzuBackend.initialize raises RuntimeError."""
        with patch(
            'axon.cli.main.KuzuBackend.initialize',
            side_effect=RuntimeError(_FAKE_ERROR),
        ):
            result = runner.invoke(app, ['analyze', str(repo_with_axon_dir)])
        assert result.exit_code == 1

    def test_error_message_shown(self, repo_with_axon_dir: Path) -> None:
        """The error message is printed to output."""
        with patch(
            'axon.cli.main.KuzuBackend.initialize',
            side_effect=RuntimeError(_FAKE_ERROR),
        ):
            result = runner.invoke(app, ['analyze', str(repo_with_axon_dir)])
        assert 'Error' in result.output or 'schema' in result.output.lower()

    def test_no_traceback_in_output(self, repo_with_axon_dir: Path) -> None:
        """No Python traceback is printed to the runner output."""
        with patch(
            'axon.cli.main.KuzuBackend.initialize',
            side_effect=RuntimeError(_FAKE_ERROR),
        ):
            result = runner.invoke(app, ['analyze', str(repo_with_axon_dir)])
        assert 'Traceback' not in result.output


class TestInitializeWritableStorageSchemaError:
    """_initialize_writable_storage wraps RuntimeError into typer.Exit(1)."""

    def test_watch_exits_one_on_schema_error(
        self, repo_with_axon_dir: Path
    ) -> None:
        """watch command exits 1 when storage schema check fails."""
        with patch(
            'axon.cli.main.KuzuBackend.initialize',
            side_effect=RuntimeError(_FAKE_ERROR),
        ):
            result = runner.invoke(app, ['watch', str(repo_with_axon_dir)])
        assert result.exit_code == 1

    def test_watch_no_traceback(self, repo_with_axon_dir: Path) -> None:
        """watch command suppresses the traceback on schema error."""
        with patch(
            'axon.cli.main.KuzuBackend.initialize',
            side_effect=RuntimeError(_FAKE_ERROR),
        ):
            result = runner.invoke(app, ['watch', str(repo_with_axon_dir)])
        assert 'Traceback' not in result.output

    def test_error_output_contains_message(
        self, repo_with_axon_dir: Path
    ) -> None:
        """The RuntimeError message text appears in the output."""
        with patch(
            'axon.cli.main.KuzuBackend.initialize',
            side_effect=RuntimeError(_FAKE_ERROR),
        ):
            result = runner.invoke(app, ['watch', str(repo_with_axon_dir)])
        assert 'schema' in result.output.lower() or 'Error' in result.output
