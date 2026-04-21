"""Tests for the shared test-file classifier and pytest config loader."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from axon.core.ingestion.test_classifier import (
    PytestConfig,
    _CONFIG_CACHE,
    is_test_file,
    load_pytest_config,
)


@pytest.fixture(autouse=True)
def clear_config_cache() -> None:
    """Clear the module-level config cache before each test for isolation."""
    _CONFIG_CACHE.clear()
    yield
    _CONFIG_CACHE.clear()


class TestDefaultHeuristic:
    """Default heuristic (no config) test-file classification."""

    def test_src_file_is_not_test(self) -> None:
        assert is_test_file('src/foo.py', None) is False

    def test_tests_dir_file_is_test(self) -> None:
        assert is_test_file('tests/test_foo.py', None) is True

    def test_test_dir_file_is_test(self) -> None:
        assert is_test_file('test/foo.py', None) is True

    def test_test_prefix_file_is_test(self) -> None:
        assert is_test_file('src/test_util.py', None) is True

    def test_conftest_is_test(self) -> None:
        assert is_test_file('src/conftest.py', None) is True

    def test_nested_src_file_is_not_test(self) -> None:
        assert is_test_file('src/foo/bar.py', None) is False

    @pytest.mark.parametrize(
        'path,expected',
        [
            ('tests/unit/test_auth.py', True),
            ('src/utils.py', False),
            ('test/integration/smoke.py', True),
            ('src/test_helpers.py', True),
            ('app/views/conftest.py', True),
        ],
    )
    def test_equivalence_with_old_heuristic(
        self, path: str, expected: bool
    ) -> None:
        """Confirm same results that the old dead_code._is_test_file would produce."""
        assert is_test_file(path, None) is expected


class TestConfigAwareNarrowing:
    """Config-aware narrowing rules applied by is_test_file."""

    def test_inside_testpaths_is_test(self) -> None:
        config = PytestConfig(testpaths=('tests/integration',))
        assert is_test_file('tests/integration/test_x.py', config) is True

    def test_outside_testpaths_is_not_test(self) -> None:
        config = PytestConfig(testpaths=('tests/integration',))
        assert is_test_file('tests/unit/test_y.py', config) is False

    def test_heuristic_match_outside_testpaths_excluded(self) -> None:
        """src/test_util.py matches heuristic but is outside testpaths."""
        config = PytestConfig(testpaths=('tests',))
        assert is_test_file('src/test_util.py', config) is False

    def test_norecursedirs_segment_excludes_file(self) -> None:
        config = PytestConfig(norecursedirs=('node_modules', '.venv'))
        assert (
            is_test_file('node_modules/pkg/tests/test_x.py', config) is False
        )

    def test_collect_ignore_exact_path_excludes(self) -> None:
        config = PytestConfig(collect_ignore=('tests/broken/test_x.py',))
        assert is_test_file('tests/broken/test_x.py', config) is False

    def test_collect_ignore_glob_excludes(self) -> None:
        config = PytestConfig(collect_ignore=('tests/broken/*.py',))
        assert is_test_file('tests/broken/test_y.py', config) is False

    def test_collect_ignore_glob_does_not_exclude_other_dirs(self) -> None:
        config = PytestConfig(collect_ignore=('tests/broken/*.py',))
        assert is_test_file('tests/good/test_z.py', config) is True

    def test_norecursedirs_tests_with_empty_testpaths_excludes(self) -> None:
        """norecursedirs=('tests',) and empty testpaths: path under tests/ is False."""
        config = PytestConfig(norecursedirs=('tests',), testpaths=())
        assert is_test_file('tests/foo/test_x.py', config) is False

    def test_non_test_file_under_testpaths_requires_heuristic_match(
        self,
    ) -> None:
        """Being inside testpaths is not sufficient; heuristic must also pass."""
        config = PytestConfig(testpaths=('tests/integration',))
        # utils.py is under testpaths but "utils.py" does not match any heuristic
        # pattern (no test_ prefix, no tests/test segment as own part, not conftest).
        # However, "tests" IS a path segment of "tests/integration/utils.py",
        # so the heuristic returns True and the file IS classified as a test file.
        assert is_test_file('tests/integration/utils.py', config) is True
        # A file with a path that doesn't satisfy the heuristic returns False.
        assert is_test_file('src/myapp/utils.py', config) is False


class TestLoadPytestConfig:
    """Pytest config loading from various file formats."""

    def test_priority_pyproject_beats_pytest_ini(self, tmp_path: Path) -> None:
        """pyproject.toml takes priority over pytest.ini when both exist."""
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.pytest.ini_options]\ntestpaths = ["tests/pyproject"]\n'
        )
        (tmp_path / 'pytest.ini').write_text(
            '[pytest]\ntestpaths = tests/ini\n'
        )
        config = load_pytest_config(tmp_path)
        assert config is not None
        assert config.testpaths == ('tests/pyproject',)

    def test_pyproject_toml_parsed_correctly(self, tmp_path: Path) -> None:
        """pyproject.toml testpaths and norecursedirs yield correct tuples."""
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.pytest.ini_options]\n'
            'testpaths = ["tests/integration"]\n'
            'norecursedirs = ["legacy"]\n'
        )
        config = load_pytest_config(tmp_path)
        assert config is not None
        assert config.testpaths == ('tests/integration',)
        assert config.norecursedirs == ('legacy',)

    def test_pytest_ini_parsed_correctly(self, tmp_path: Path) -> None:
        """pytest.ini whitespace-separated values produce correct tuples."""
        (tmp_path / 'pytest.ini').write_text(
            '[pytest]\ntestpaths = tests/integration\nnorecursedirs = legacy .venv\n'
        )
        config = load_pytest_config(tmp_path)
        assert config is not None
        assert config.testpaths == ('tests/integration',)
        assert config.norecursedirs == ('legacy', '.venv')

    def test_setup_cfg_parsed(self, tmp_path: Path) -> None:
        (tmp_path / 'setup.cfg').write_text(
            '[tool:pytest]\ntestpaths = tests\n'
        )
        config = load_pytest_config(tmp_path)
        assert config is not None
        assert config.testpaths == ('tests',)

    def test_tox_ini_parsed(self, tmp_path: Path) -> None:
        (tmp_path / 'tox.ini').write_text('[pytest]\ntestpaths = tests\n')
        config = load_pytest_config(tmp_path)
        assert config is not None
        assert config.testpaths == ('tests',)

    def test_missing_config_returns_none(self, tmp_path: Path) -> None:
        config = load_pytest_config(tmp_path)
        assert config is None

    def test_malformed_toml_returns_none(self, tmp_path: Path) -> None:
        """Malformed pyproject.toml causes load_pytest_config to return None.

        The implementation stores None in the cache and returns it immediately
        without falling through to the next candidate file.
        """
        (tmp_path / 'pyproject.toml').write_text(
            'this is not valid toml {{{\n'
        )
        (tmp_path / 'pytest.ini').write_text('[pytest]\ntestpaths = tests\n')
        config = load_pytest_config(tmp_path)
        # pyproject.toml is found first (exists), parse fails -> None returned.
        assert config is None

    def test_mtime_cache_returns_same_result(self, tmp_path: Path) -> None:
        """Second call with same mtime returns cached result."""
        (tmp_path / 'pytest.ini').write_text('[pytest]\ntestpaths = tests\n')
        first = load_pytest_config(tmp_path)
        second = load_pytest_config(tmp_path)
        assert first == second

    def test_mtime_change_triggers_reread(self, tmp_path: Path) -> None:
        """After touching the config file, load re-reads updated values."""
        cfg = tmp_path / 'pytest.ini'
        cfg.write_text('[pytest]\ntestpaths = tests\n')
        first = load_pytest_config(tmp_path)
        assert first is not None
        assert first.testpaths == ('tests',)

        # Bump mtime by writing new content with a later timestamp.
        time.sleep(0.01)
        cfg.write_text('[pytest]\ntestpaths = tests/integration\n')
        # Force mtime to differ by at least 1 second on filesystems that
        # round to integer seconds.
        new_mtime = cfg.stat().st_mtime + 1

        import os
        os.utime(cfg, (new_mtime, new_mtime))

        second = load_pytest_config(tmp_path)
        assert second is not None
        assert second.testpaths == ('tests/integration',)

    def test_force_refresh_bypasses_cache(self, tmp_path: Path) -> None:
        """force_refresh=True re-reads the file even when mtime is unchanged."""
        cfg = tmp_path / 'pytest.ini'
        cfg.write_text('[pytest]\ntestpaths = tests\n')
        first = load_pytest_config(tmp_path)
        assert first is not None

        # Mutate file content without changing mtime to simulate a cached stale
        # entry. We do this by writing, then restoring original mtime.
        original_stat = cfg.stat()
        cfg.write_text('[pytest]\ntestpaths = tests/force_refresh\n')

        import os
        os.utime(cfg, (original_stat.st_atime, original_stat.st_mtime))

        # Without force_refresh: same mtime -> cache hit -> old value.
        cached = load_pytest_config(tmp_path)
        assert cached is not None
        assert cached.testpaths == ('tests',)

        # With force_refresh: cache evicted -> new value.
        refreshed = load_pytest_config(tmp_path, force_refresh=True)
        assert refreshed is not None
        assert refreshed.testpaths == ('tests/force_refresh',)
