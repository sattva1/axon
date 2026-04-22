"""Tests for axon.core.meta - MetaFile, load_meta, update_meta, helpers."""

from __future__ import annotations

import json
from dataclasses import fields as dc_fields
from datetime import datetime
from pathlib import Path

from axon.core.meta import (
    MetaFile,
    _sanitize_stats,
    load_meta,
    meta_path,
    now_iso,
    update_meta,
)


class TestMetaFileDefaults:
    def test_all_fields_present(self) -> None:
        """MetaFile() has every declared field with an appropriate default."""
        MetaFile()  # smoke: default construction does not raise
        field_names = {f.name for f in dc_fields(MetaFile)}
        expected = {
            'version',
            'name',
            'path',
            'embedding_model',
            'embedding_dimensions',
            'stats',
            'last_indexed_at',
            'last_incremental_at',
            'dead_code_last_refreshed_at',
            'communities_last_refreshed_at',
        }
        assert expected <= field_names

    def test_string_fields_default_to_empty(self) -> None:
        """String fields default to empty string."""
        meta = MetaFile()
        assert meta.version == ''
        assert meta.name == ''
        assert meta.path == ''
        assert meta.embedding_model == ''
        assert meta.last_indexed_at == ''
        assert meta.last_incremental_at == ''
        assert meta.dead_code_last_refreshed_at == ''
        assert meta.communities_last_refreshed_at == ''

    def test_numeric_field_defaults(self) -> None:
        """embedding_dimensions defaults to 0."""
        meta = MetaFile()
        assert meta.embedding_dimensions == 0

    def test_stats_defaults_to_empty_dict(self) -> None:
        """stats defaults to empty dict, not shared mutable default."""
        a = MetaFile()
        b = MetaFile()
        assert a.stats == {}
        assert a.stats is not b.stats


class TestMetaPath:
    def test_canonical_location(self, tmp_path: Path) -> None:
        """meta_path returns <repo>/.axon/meta.json."""
        result = meta_path(tmp_path)
        assert result == tmp_path / '.axon' / 'meta.json'


class TestNowIso:
    def test_parseable_back(self) -> None:
        """now_iso produces a string parseable by datetime.fromisoformat."""
        s = now_iso()
        parsed = datetime.fromisoformat(s)
        assert parsed.tzinfo is not None

    def test_tz_aware(self) -> None:
        """now_iso result is timezone-aware (UTC)."""
        s = now_iso()
        parsed = datetime.fromisoformat(s)
        assert parsed.utcoffset() is not None


class TestLoadMeta:
    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        """load_meta on non-existent file returns MetaFile() defaults."""
        result = load_meta(tmp_path)
        assert result == MetaFile()

    def test_torn_json_returns_defaults(self, tmp_path: Path) -> None:
        """load_meta on partial/corrupt JSON returns MetaFile() defaults."""
        axon_dir = tmp_path / '.axon'
        axon_dir.mkdir()
        (axon_dir / 'meta.json').write_text('{broken', encoding='utf-8')

        result = load_meta(tmp_path)
        assert result == MetaFile()

    def test_non_dict_json_list_returns_defaults(self, tmp_path: Path) -> None:
        """load_meta on JSON list returns MetaFile() defaults."""
        axon_dir = tmp_path / '.axon'
        axon_dir.mkdir()
        (axon_dir / 'meta.json').write_text('[]', encoding='utf-8')

        result = load_meta(tmp_path)
        assert result == MetaFile()

    def test_non_dict_json_string_returns_defaults(
        self, tmp_path: Path
    ) -> None:
        """load_meta on JSON string returns MetaFile() defaults."""
        axon_dir = tmp_path / '.axon'
        axon_dir.mkdir()
        (axon_dir / 'meta.json').write_text('"hello"', encoding='utf-8')

        result = load_meta(tmp_path)
        assert result == MetaFile()

    def test_non_dict_json_null_returns_defaults(self, tmp_path: Path) -> None:
        """load_meta on JSON null returns MetaFile() defaults."""
        axon_dir = tmp_path / '.axon'
        axon_dir.mkdir()
        (axon_dir / 'meta.json').write_text('null', encoding='utf-8')

        result = load_meta(tmp_path)
        assert result == MetaFile()

    def test_unknown_keys_dropped(self, tmp_path: Path) -> None:
        """load_meta silently drops keys not present in MetaFile."""
        axon_dir = tmp_path / '.axon'
        axon_dir.mkdir()
        data = {'version': '1.0', 'future_field': 'ignored', 'name': 'myrepo'}
        (axon_dir / 'meta.json').write_text(json.dumps(data), encoding='utf-8')

        result = load_meta(tmp_path)
        assert result.version == '1.0'
        assert result.name == 'myrepo'

    def test_known_keys_populated(self, tmp_path: Path) -> None:
        """load_meta populates all recognized fields from JSON."""
        axon_dir = tmp_path / '.axon'
        axon_dir.mkdir()
        data = {
            'version': '2.0',
            'name': 'repo',
            'path': '/some/path',
            'embedding_model': 'model-x',
            'embedding_dimensions': 384,
            'stats': {'files': 10},
            'last_indexed_at': '2025-01-01T00:00:00+00:00',
            'last_incremental_at': '2025-01-02T00:00:00+00:00',
            'dead_code_last_refreshed_at': '2025-01-02T00:00:00+00:00',
            'communities_last_refreshed_at': '2025-01-02T00:00:00+00:00',
        }
        (axon_dir / 'meta.json').write_text(json.dumps(data), encoding='utf-8')

        result = load_meta(tmp_path)
        assert result.version == '2.0'
        assert result.name == 'repo'
        assert result.embedding_dimensions == 384
        assert result.stats == {'files': 10}
        assert result.last_incremental_at == '2025-01-02T00:00:00+00:00'

    def test_malformed_stats_null_returns_empty(self, tmp_path: Path) -> None:
        """load_meta with stats=null sanitizes to empty dict."""
        axon_dir = tmp_path / '.axon'
        axon_dir.mkdir()
        data = {'version': '1.0', 'stats': None}
        (axon_dir / 'meta.json').write_text(json.dumps(data), encoding='utf-8')

        result = load_meta(tmp_path)
        assert result.stats == {}
        assert result.version == '1.0'

    def test_malformed_stats_list_returns_empty(self, tmp_path: Path) -> None:
        """load_meta with stats as a list sanitizes to empty dict."""
        axon_dir = tmp_path / '.axon'
        axon_dir.mkdir()
        data = {'version': '1.0', 'stats': [1, 2, 3]}
        (axon_dir / 'meta.json').write_text(json.dumps(data), encoding='utf-8')

        result = load_meta(tmp_path)
        assert result.stats == {}

    def test_malformed_stats_string_returns_empty(
        self, tmp_path: Path
    ) -> None:
        """load_meta with stats as a string sanitizes to empty dict."""
        axon_dir = tmp_path / '.axon'
        axon_dir.mkdir()
        data = {'version': '1.0', 'stats': 'bad'}
        (axon_dir / 'meta.json').write_text(json.dumps(data), encoding='utf-8')

        result = load_meta(tmp_path)
        assert result.stats == {}

    def test_mixed_stats_values_coerced(self, tmp_path: Path) -> None:
        """load_meta coerces float, bool, string, None stats values to 0."""
        axon_dir = tmp_path / '.axon'
        axon_dir.mkdir()
        data = {
            'stats': {
                'files': 10,
                'symbols': 3.14,
                'flag': True,
                'label': 'hello',
                'nothing': None,
            }
        }
        (axon_dir / 'meta.json').write_text(json.dumps(data), encoding='utf-8')

        result = load_meta(tmp_path)
        assert result.stats['files'] == 10
        assert result.stats['symbols'] == 0  # float -> 0
        assert 'flag' not in result.stats  # bool excluded entirely
        assert result.stats['label'] == 0  # string -> 0
        assert result.stats['nothing'] == 0  # None -> 0


class TestUpdateMeta:
    def test_creates_axon_dir_if_missing(self, tmp_path: Path) -> None:
        """update_meta creates .axon/ directory when it does not exist."""
        assert not (tmp_path / '.axon').exists()
        update_meta(tmp_path, version='1.0')
        assert (tmp_path / '.axon').exists()
        assert (tmp_path / '.axon' / 'meta.json').exists()

    def test_roundtrip_preserves_all_fields(self, tmp_path: Path) -> None:
        """Write-read roundtrip preserves all MetaFile fields."""
        update_meta(
            tmp_path,
            version='1.2',
            name='myrepo',
            path='/repo',
            embedding_model='model-a',
            embedding_dimensions=512,
            stats={'files': 5, 'symbols': 20},
            last_indexed_at='2025-01-01T00:00:00+00:00',
            last_incremental_at='2025-01-02T00:00:00+00:00',
            dead_code_last_refreshed_at='2025-01-02T00:00:00+00:00',
            communities_last_refreshed_at='2025-01-02T00:00:00+00:00',
        )

        result = load_meta(tmp_path)
        assert result.version == '1.2'
        assert result.name == 'myrepo'
        assert result.embedding_dimensions == 512
        assert result.stats == {'files': 5, 'symbols': 20}
        assert result.last_incremental_at == '2025-01-02T00:00:00+00:00'
        assert (
            result.dead_code_last_refreshed_at == '2025-01-02T00:00:00+00:00'
        )
        assert (
            result.communities_last_refreshed_at == '2025-01-02T00:00:00+00:00'
        )

    def test_partial_update_preserves_other_fields(
        self, tmp_path: Path
    ) -> None:
        """Partial update does not overwrite fields not passed."""
        update_meta(tmp_path, version='1.0', name='myrepo')
        update_meta(tmp_path, version='2.0')

        result = load_meta(tmp_path)
        assert result.version == '2.0'
        assert result.name == 'myrepo'  # untouched

    def test_stats_merge_preserves_siblings(self, tmp_path: Path) -> None:
        """Critical: stats update merges into existing dict, not replaces."""
        update_meta(tmp_path, stats={'files': 50, 'symbols': 200})
        update_meta(tmp_path, stats={'embeddings': 100})

        result = load_meta(tmp_path)
        assert result.stats['files'] == 50
        assert result.stats['symbols'] == 200
        assert result.stats['embeddings'] == 100

    def test_empty_stats_dict_is_noop_on_siblings(
        self, tmp_path: Path
    ) -> None:
        """update_meta with empty stats dict leaves existing stats intact."""
        update_meta(tmp_path, stats={'files': 42})
        update_meta(tmp_path, stats={})

        result = load_meta(tmp_path)
        assert result.stats['files'] == 42

    def test_atomic_write_no_tmp_file_leftover(self, tmp_path: Path) -> None:
        """After update_meta completes, no .tmp file remains on disk."""
        update_meta(tmp_path, version='1.0')

        axon_dir = tmp_path / '.axon'
        tmp_files = list(axon_dir.glob('*.tmp'))
        assert tmp_files == [], f'Unexpected .tmp files: {tmp_files}'

    def test_meta_json_is_valid_json(self, tmp_path: Path) -> None:
        """Written meta.json is valid JSON readable by stdlib json."""
        update_meta(tmp_path, version='1.0', stats={'files': 7})

        raw = (tmp_path / '.axon' / 'meta.json').read_text(encoding='utf-8')
        parsed = json.loads(raw)
        assert parsed['version'] == '1.0'
        assert parsed['stats']['files'] == 7


class TestSanitizeStats:
    def test_valid_int_values_kept(self) -> None:
        """Valid int values pass through unchanged."""
        result = _sanitize_stats({'a': 1, 'b': 100})
        assert result == {'a': 1, 'b': 100}

    def test_non_dict_input_returns_empty(self) -> None:
        """Non-dict input returns empty dict."""
        assert _sanitize_stats(None) == {}
        assert _sanitize_stats([]) == {}
        assert _sanitize_stats('bad') == {}
        assert _sanitize_stats(42) == {}

    def test_bool_excluded(self) -> None:
        """bool values are excluded even though bool is a subtype of int."""
        result = _sanitize_stats({'flag': True, 'other': False})
        assert 'flag' not in result
        assert 'other' not in result

    def test_float_coerced_to_zero(self) -> None:
        """float values are coerced to 0."""
        result = _sanitize_stats({'count': 3.14})
        assert result['count'] == 0

    def test_string_value_coerced_to_zero(self) -> None:
        """String values are coerced to 0."""
        result = _sanitize_stats({'label': 'hello'})
        assert result['label'] == 0

    def test_none_value_coerced_to_zero(self) -> None:
        """None values are coerced to 0."""
        result = _sanitize_stats({'x': None})
        assert result['x'] == 0

    def test_non_string_keys_skipped(self) -> None:
        """Keys that are not strings are skipped."""
        result = _sanitize_stats({1: 10, 'valid': 5})
        assert 1 not in result
        assert result.get('valid') == 5

    def test_empty_dict_returns_empty(self) -> None:
        """Empty dict input returns empty dict."""
        assert _sanitize_stats({}) == {}

    def test_mixed_preserves_valid_drops_invalid(self) -> None:
        """Mix of valid and invalid values: valid ones kept, others coerced."""
        result = _sanitize_stats(
            {'files': 10, 'symbols': 20, 'rate': 0.5, 'enabled': True}
        )
        assert result['files'] == 10
        assert result['symbols'] == 20
        assert result['rate'] == 0
        assert 'enabled' not in result

    def test_negative_int_kept(self) -> None:
        """Negative integers are valid ints and kept as-is."""
        result = _sanitize_stats({'delta': -5})
        assert result['delta'] == -5
