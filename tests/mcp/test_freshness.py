"""Tests for axon.mcp.freshness - staleness detection helpers."""

from __future__ import annotations

from pathlib import Path

from axon.core.meta import MetaFile, update_meta
from axon.mcp.freshness import (
    _parse_iso,
    render_with_communities_warning,
    render_with_dead_code_warning,
    staleness_warning_communities,
    staleness_warning_dead_code,
)


class TestParseIso:
    def test_empty_string_returns_none(self) -> None:
        """Empty string returns None."""
        assert _parse_iso('') is None

    def test_non_date_string_returns_none(self) -> None:
        """Garbage string returns None."""
        assert _parse_iso('not-a-date') is None

    def test_utc_z_suffix_is_aware(self) -> None:
        """ISO string with Z suffix parses to tz-aware UTC datetime."""
        result = _parse_iso('2025-01-01T00:00:00Z')
        assert result is not None
        assert result.tzinfo is not None
        assert result.year == 2025

    def test_naive_datetime_coerced_to_utc(self) -> None:
        """Naive datetime string is coerced to UTC (critical regression test)."""
        result = _parse_iso('2025-01-01T00:00:00')
        assert result is not None
        assert result.tzinfo is not None
        # Coerced to UTC, so utcoffset should be zero.
        assert result.utcoffset().total_seconds() == 0

    def test_offset_aware_preserves_offset(self) -> None:
        """Datetime with +02:00 offset preserves that offset."""
        result = _parse_iso('2025-01-01T00:00:00+02:00')
        assert result is not None
        assert result.tzinfo is not None
        assert result.utcoffset().total_seconds() == 7200

    def test_subtraction_of_naive_and_aware_does_not_raise(self) -> None:
        """Subtracting two parsed values never raises TypeError, even for naive input."""
        a = _parse_iso('2025-01-01T00:10:00')
        b = _parse_iso('2025-01-01T00:00:00Z')
        assert a is not None and b is not None
        # Would raise TypeError if one is naive.
        delta = (a - b).total_seconds()
        assert delta == 600


class TestStalenessWarningDeadCode:
    def test_empty_timestamps_returns_empty(self) -> None:
        """MetaFile with empty timestamps produces no warning."""
        meta = MetaFile()
        assert staleness_warning_dead_code(meta) == ''

    def test_only_incremental_set_returns_empty(self) -> None:
        """Only last_incremental_at set (no refreshed timestamp) returns empty."""
        meta = MetaFile(last_incremental_at='2025-01-01T00:01:00+00:00')
        assert staleness_warning_dead_code(meta) == ''

    def test_only_refreshed_set_returns_empty(self) -> None:
        """Only dead_code_last_refreshed_at set returns empty."""
        meta = MetaFile(
            dead_code_last_refreshed_at='2025-01-01T00:01:00+00:00'
        )
        assert staleness_warning_dead_code(meta) == ''

    def test_timestamps_equal_returns_empty(self) -> None:
        """Equal timestamps produce no warning (delta = 0)."""
        ts = '2025-01-01T00:01:00+00:00'
        meta = MetaFile(last_incremental_at=ts, dead_code_last_refreshed_at=ts)
        assert staleness_warning_dead_code(meta) == ''

    def test_lag_exactly_60s_returns_empty(self) -> None:
        """Lag exactly at the 60s boundary produces no warning."""
        meta = MetaFile(
            last_incremental_at='2025-01-01T00:01:00+00:00',
            dead_code_last_refreshed_at='2025-01-01T00:00:00+00:00',
        )
        assert staleness_warning_dead_code(meta) == ''

    def test_lag_61s_returns_warning(self) -> None:
        """Lag of 61 seconds produces a warning containing expected content."""
        meta = MetaFile(
            last_incremental_at='2025-01-01T00:01:01+00:00',
            dead_code_last_refreshed_at='2025-01-01T00:00:00+00:00',
        )
        warning = staleness_warning_dead_code(meta)
        assert warning != ''
        assert 'axon analyze' in warning
        assert '61s' in warning
        assert warning.endswith('\n\n')

    def test_refreshed_in_future_returns_empty(self) -> None:
        """Negative delta (refreshed after incremental) produces no warning."""
        meta = MetaFile(
            last_incremental_at='2025-01-01T00:00:00+00:00',
            dead_code_last_refreshed_at='2025-01-01T00:01:00+00:00',
        )
        assert staleness_warning_dead_code(meta) == ''

    def test_warning_contains_refreshed_timestamp(self) -> None:
        """Warning includes the refreshed-at timestamp for orientation."""
        meta = MetaFile(
            last_incremental_at='2025-01-01T00:02:00+00:00',
            dead_code_last_refreshed_at='2025-01-01T00:00:00+00:00',
        )
        warning = staleness_warning_dead_code(meta)
        assert '2025-01-01' in warning


class TestStalenessWarningCommunities:
    def test_empty_timestamps_returns_empty(self) -> None:
        """MetaFile with empty timestamps produces no warning."""
        meta = MetaFile()
        assert staleness_warning_communities(meta) == ''

    def test_lag_61s_returns_warning(self) -> None:
        """Lag of 61 seconds on communities produces warning."""
        meta = MetaFile(
            last_incremental_at='2025-01-01T00:01:01+00:00',
            communities_last_refreshed_at='2025-01-01T00:00:00+00:00',
        )
        warning = staleness_warning_communities(meta)
        assert warning != ''
        assert '61s' in warning
        assert warning.endswith('\n\n')

    def test_lag_exactly_60s_returns_empty(self) -> None:
        """Boundary: exactly 60s lag produces no warning."""
        meta = MetaFile(
            last_incremental_at='2025-01-01T00:01:00+00:00',
            communities_last_refreshed_at='2025-01-01T00:00:00+00:00',
        )
        assert staleness_warning_communities(meta) == ''

    def test_refreshed_in_future_returns_empty(self) -> None:
        """Negative delta returns empty."""
        meta = MetaFile(
            last_incremental_at='2025-01-01T00:00:00+00:00',
            communities_last_refreshed_at='2025-01-01T00:01:00+00:00',
        )
        assert staleness_warning_communities(meta) == ''

    def test_uses_communities_field_not_dead_code(self) -> None:
        """communities warning reads communities_last_refreshed_at, not dead_code field."""
        meta = MetaFile(
            last_incremental_at='2025-01-01T00:02:00+00:00',
            # dead_code is fresh; communities is stale
            dead_code_last_refreshed_at='2025-01-01T00:02:00+00:00',
            communities_last_refreshed_at='2025-01-01T00:00:00+00:00',
        )
        assert staleness_warning_communities(meta) != ''
        assert staleness_warning_dead_code(meta) == ''


class TestRenderWithDeadCodeWarning:
    def test_none_repo_path_returns_body_unchanged(self) -> None:
        """render_with_dead_code_warning(None, body) leaves body intact."""
        body = 'some results'
        result = render_with_dead_code_warning(None, body)
        assert result == body

    def test_fresh_meta_returns_body_unchanged(self, tmp_path: Path) -> None:
        """Fresh meta (no lag) returns body without a prefix."""
        ts = '2025-01-01T00:00:00+00:00'
        update_meta(
            tmp_path, last_incremental_at=ts, dead_code_last_refreshed_at=ts
        )
        body = 'dead code results'
        result = render_with_dead_code_warning(tmp_path, body)
        assert result == body

    def test_stale_meta_prepends_warning(self, tmp_path: Path) -> None:
        """Stale meta causes warning to be prepended before body."""
        update_meta(
            tmp_path,
            last_incremental_at='2025-01-01T00:05:00+00:00',
            dead_code_last_refreshed_at='2025-01-01T00:00:00+00:00',
        )
        body = 'dead code results'
        result = render_with_dead_code_warning(tmp_path, body)
        assert result.endswith(body)
        assert 'axon analyze' in result
        assert len(result) > len(body)

    def test_missing_meta_file_returns_body_unchanged(
        self, tmp_path: Path
    ) -> None:
        """If no meta.json exists, load_meta returns defaults -> no warning."""
        body = 'some results'
        result = render_with_dead_code_warning(tmp_path, body)
        assert result == body


class TestRenderWithCommunitiesWarning:
    def test_none_repo_path_returns_body_unchanged(self) -> None:
        """render_with_communities_warning(None, body) leaves body intact."""
        body = 'community results'
        result = render_with_communities_warning(None, body)
        assert result == body

    def test_uses_communities_field_not_dead_code(
        self, tmp_path: Path
    ) -> None:
        """render_with_communities_warning reads communities_last_refreshed_at."""
        update_meta(
            tmp_path,
            last_incremental_at='2025-01-01T00:05:00+00:00',
            # dead_code is fresh; communities is stale
            dead_code_last_refreshed_at='2025-01-01T00:05:00+00:00',
            communities_last_refreshed_at='2025-01-01T00:00:00+00:00',
        )
        body = 'community results'
        communities_result = render_with_communities_warning(tmp_path, body)
        dead_code_result = render_with_dead_code_warning(tmp_path, body)

        assert 'axon analyze' in communities_result
        assert dead_code_result == body
