"""Tests for _parse_duration_seconds and --global-refresh-interval CLI flag."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from axon.cli.main import _parse_duration_seconds, app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _suppress_update_notice() -> None:
    """Suppress version-update notice for all tests in this module."""
    with patch('axon.cli.main._maybe_notify_update'):
        yield


class TestParseDurationSeconds:
    def test_seconds_unit(self) -> None:
        """'30s' parses to 30."""
        assert _parse_duration_seconds('30s') == 30

    def test_minutes_unit(self) -> None:
        """'5m' parses to 300."""
        assert _parse_duration_seconds('5m') == 300

    def test_hours_unit(self) -> None:
        """'1h' parses to 3600."""
        assert _parse_duration_seconds('1h') == 3600

    def test_strips_whitespace(self) -> None:
        """Leading/trailing whitespace is stripped before parsing."""
        assert _parse_duration_seconds('  30s  ') == 30

    def test_large_number_seconds(self) -> None:
        """Large numeric value with 's' parses correctly."""
        assert _parse_duration_seconds('3600s') == 3600

    def test_zero_seconds(self) -> None:
        """'0s' parses to 0."""
        assert _parse_duration_seconds('0s') == 0

    def test_empty_string_raises_bad_parameter(self) -> None:
        """Empty string raises BadParameter."""
        with pytest.raises(typer.BadParameter):
            _parse_duration_seconds('')

    def test_missing_unit_raises_bad_parameter(self) -> None:
        """Bare number without unit raises BadParameter."""
        with pytest.raises(typer.BadParameter):
            _parse_duration_seconds('30')

    def test_unsupported_unit_raises_bad_parameter(self) -> None:
        """'30d' (days) raises BadParameter."""
        with pytest.raises(typer.BadParameter):
            _parse_duration_seconds('30d')

    def test_negative_raises_bad_parameter(self) -> None:
        """Negative duration raises BadParameter (regex requires \\d+)."""
        with pytest.raises(typer.BadParameter):
            _parse_duration_seconds('-5s')

    def test_decimal_raises_bad_parameter(self) -> None:
        """Decimal value raises BadParameter."""
        with pytest.raises(typer.BadParameter):
            _parse_duration_seconds('1.5s')

    def test_multiple_hours(self) -> None:
        """'2h' parses to 7200."""
        assert _parse_duration_seconds('2h') == 7200


class TestWatchGlobalRefreshIntervalFlag:
    def test_valid_duration_parses_without_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """axon watch --global-refresh-interval=30s parses without exit."""
        monkeypatch.chdir(tmp_path)

        mock_storage = MagicMock()
        axon_dir = tmp_path / '.axon'
        axon_dir.mkdir()
        (axon_dir / 'meta.json').write_text(
            '{"version":"1.0","embedding_model":""}', encoding='utf-8'
        )

        with patch('axon.cli.main._initialize_writable_storage') as mock_init:
            mock_init.return_value = (
                mock_storage,
                axon_dir,
                axon_dir / 'kuzu',
            )
            with patch('asyncio.run'):
                result = runner.invoke(
                    app,
                    ['watch', str(tmp_path), '--global-refresh-interval=30s'],
                )

        # Should not fail with exit code 2 (argument parse error).
        assert result.exit_code != 2, result.output

    def test_invalid_duration_fails_with_exit_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """axon watch --global-refresh-interval=invalid fails with exit code 2."""
        monkeypatch.chdir(tmp_path)

        axon_dir = tmp_path / '.axon'
        axon_dir.mkdir()
        (axon_dir / 'meta.json').write_text('{}', encoding='utf-8')

        with patch('axon.cli.main._initialize_writable_storage'):
            result = runner.invoke(
                app,
                ['watch', str(tmp_path), '--global-refresh-interval=invalid'],
            )

        assert result.exit_code == 2

    def test_interval_default_is_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without --global-refresh-interval, no interval is passed to watch_repo."""
        monkeypatch.chdir(tmp_path)

        mock_storage = MagicMock()
        axon_dir = tmp_path / '.axon'
        axon_dir.mkdir()
        (axon_dir / 'meta.json').write_text(
            '{"version":"1.0","embedding_model":""}', encoding='utf-8'
        )

        captured_kwargs: dict = {}

        async def fake_watch_repo(repo, db_path, **kwargs):
            captured_kwargs.update(kwargs)

        with patch('axon.cli.main._initialize_writable_storage') as mock_init:
            mock_init.return_value = (
                mock_storage,
                axon_dir,
                axon_dir / 'kuzu',
            )
            with patch('axon.cli.main.watch_repo', fake_watch_repo):
                with patch(
                    'axon.cli.main._configure_and_validate_accelerator'
                ):
                    runner.invoke(app, ['watch', str(tmp_path)])

        assert captured_kwargs.get('global_refresh_interval_seconds') is None
