"""Tests for axon.mcp.freshness - staleness detection helpers."""

from __future__ import annotations

from pathlib import Path

from axon.core.drift import DriftLevel, DriftReport
from axon.core.meta import MetaFile, update_meta
from axon.mcp.freshness import (
    _parse_iso,
    render_with_communities_warning,
    render_with_dead_code_warning,
    render_with_drift_warning,
    staleness_warning_communities,
    staleness_warning_dead_code,
)


def _make_report(
    level: DriftLevel,
    reason: str = 'test reason',
    slug: str | None = None,
    watcher_alive: bool = False,
) -> DriftReport:
    """Build a minimal DriftReport for the given level."""
    return DriftReport(
        level=level,
        reason=reason,
        last_indexed_at='',
        head_sha=None,
        head_sha_at_index=None,
        files_changed_estimate=None,
        files_indexed_estimate=None,
        watcher_alive=watcher_alive,
        tier_used=None,
        slug=slug,
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


class TestRenderWithDriftWarning:
    """render_with_drift_warning behaviour for all DriftLevel values."""

    def test_noop_when_fresh(self) -> None:
        """FRESH report leaves body unchanged."""
        body = 'result body'
        report = _make_report(DriftLevel.FRESH)
        assert render_with_drift_warning(report, body) == body

    def test_noop_when_unknown(self) -> None:
        """UNKNOWN report leaves body unchanged."""
        body = 'result body'
        report = _make_report(DriftLevel.UNKNOWN)
        assert render_with_drift_warning(report, body) == body

    def test_prepends_when_stale_minor(self) -> None:
        """STALE_MINOR report prepends a warning header separated from body."""
        body = 'actual result'
        report = _make_report(DriftLevel.STALE_MINOR, reason='dirty tree')
        result = render_with_drift_warning(report, body)

        assert result.endswith(body)
        assert result != body
        # Warning and body must be separated by a blank line.
        assert '\n\n' in result
        warning_part = result[: result.index('\n\n')]
        assert (
            'drift' in warning_part.lower() or 'minor' in warning_part.lower()
        )

    def test_includes_slug_when_set(self) -> None:
        """Warning line quotes the slug when report.slug is set."""
        body = 'result'
        report = _make_report(
            DriftLevel.STALE_MINOR, reason='sentinel modified', slug='my-repo'
        )
        result = render_with_drift_warning(report, body)

        assert "'my-repo'" in result

    def test_omits_slug_when_none(self) -> None:
        """Warning still appears when slug is None, without a quoted slug token."""
        body = 'result'
        report = _make_report(
            DriftLevel.STALE_MINOR, reason='sentinel modified', slug=None
        )
        result = render_with_drift_warning(report, body)

        assert result != body
        # No quoted token should appear in the warning portion.
        warning_part = result[: result.index('\n\n')]
        assert "'" not in warning_part

    def test_reason_appears_in_warning(self) -> None:
        """The drift reason string is included in the warning line."""
        reason = 'HEAD advanced by 3 commits, 5 files changed'
        body = 'result'
        report = _make_report(DriftLevel.STALE_MINOR, reason=reason)
        result = render_with_drift_warning(report, body)

        assert reason in result

    def test_stale_major_watcher_alive_includes_catching_up(self) -> None:
        """STALE_MAJOR + watcher alive prepends a 'catching up' message."""
        body = 'result body'
        report = _make_report(
            DriftLevel.STALE_MAJOR, reason='diverged', watcher_alive=True
        )
        result = render_with_drift_warning(report, body)

        assert result.endswith(body)
        assert result != body
        warning_part = result[: result.index('\n\n')]
        assert 'catching up' in warning_part
        assert 'diverged' in warning_part

    def test_stale_major_watcher_dead_includes_axon_analyze(self) -> None:
        """STALE_MAJOR + watcher dead prepends a 'axon analyze' prompt."""
        body = 'result body'
        report = _make_report(
            DriftLevel.STALE_MAJOR, reason='diverged', watcher_alive=False
        )
        result = render_with_drift_warning(report, body)

        assert result.endswith(body)
        assert result != body
        warning_part = result[: result.index('\n\n')]
        assert '`axon analyze`' in warning_part

    def test_stale_major_includes_slug_when_set(self) -> None:
        """STALE_MAJOR warning quotes the slug when report.slug is set."""
        body = 'result'
        report = _make_report(
            DriftLevel.STALE_MAJOR, slug='my-repo', watcher_alive=False
        )
        result = render_with_drift_warning(report, body)

        assert "'my-repo'" in result

    def test_stale_major_uses_report_reason(self) -> None:
        """STALE_MAJOR warning includes the report.reason verbatim."""
        body = 'result'
        report = _make_report(
            DriftLevel.STALE_MAJOR,
            reason='custom reason X',
            watcher_alive=False,
        )
        result = render_with_drift_warning(report, body)

        assert 'custom reason X' in result
