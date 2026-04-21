"""Shared test-file classifier and pytest config loader for Axon.

Public API
----------
- ``is_test_file(path, config=None)`` -- heuristic + config-aware check.
- ``load_pytest_config(repo_root, *, force_refresh=False)`` -- parse the
  first pytest config file found, with mtime-keyed caching.
- ``PytestConfig`` -- frozen dataclass holding parsed pytest settings.

Config file priority (matches pytest itself):
  1. pyproject.toml  [tool.pytest.ini_options]
  2. pytest.ini       [pytest]
  3. setup.cfg        [tool:pytest]
  4. tox.ini          [pytest]

The first source found wins; keys are not merged across files.

Caching contract
----------------
``load_pytest_config`` caches parsed results keyed on
``(str(config_file_path), mtime_float)``.  Before returning a cached result
the function stats the config file to verify the mtime matches.
``force_refresh=True`` bypasses the cache entirely and evicts any stale
entry for the same path before writing the fresh result.
"""

from __future__ import annotations

import configparser
import fnmatch
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover -- Python 3.10 shim
    try:
        import tomllib  # type: ignore[no-redef]
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True, slots=True)
class PytestConfig:
    """Parsed pytest configuration subset used by the test classifier.

    All fields default to empty tuples when the corresponding key is absent
    from the config file.
    """

    testpaths: tuple[str, ...] = ()
    norecursedirs: tuple[str, ...] = ()
    collect_ignore: tuple[str, ...] = ()


# mtime-keyed cache: (path_str, mtime) -> PytestConfig | None
_CONFIG_CACHE: dict[tuple[str, float], PytestConfig | None] = {}

# Config file names searched in priority order.
_CONFIG_CANDIDATES: tuple[tuple[str, str, str], ...] = (
    ('pyproject.toml', 'toml', 'tool.pytest.ini_options'),
    ('pytest.ini', 'ini', 'pytest'),
    ('setup.cfg', 'ini', 'tool:pytest'),
    ('tox.ini', 'ini', 'pytest'),
)


def _split_value(raw: Any) -> tuple[str, ...]:
    """Convert a TOML list or INI whitespace-separated string to a tuple.

    Args:
        raw: Either a list (from TOML) or a string (from INI).

    Returns:
        Tuple of non-empty stripped strings.
    """
    if isinstance(raw, list):
        return tuple(str(v).strip() for v in raw if str(v).strip())
    if isinstance(raw, str):
        return tuple(part for part in raw.split() if part)
    return ()


def _parse_toml_config(path: Path, section_path: str) -> PytestConfig | None:
    """Parse pytest settings from a TOML file.

    Args:
        path: Path to the TOML file.
        section_path: Dotted key path e.g. "tool.pytest.ini_options".

    Returns:
        Parsed config or None if the section is absent.
    """
    try:
        with path.open('rb') as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return None

    keys = section_path.split('.')
    section: Any = data
    for key in keys:
        if not isinstance(section, dict):
            return None
        section = section.get(key)
        if section is None:
            return None

    if not isinstance(section, dict):
        return None

    return PytestConfig(
        testpaths=_split_value(section.get('testpaths', ())),
        norecursedirs=_split_value(section.get('norecursedirs', ())),
        collect_ignore=_split_value(section.get('collect_ignore', ())),
    )


def _parse_ini_config(path: Path, section: str) -> PytestConfig | None:
    """Parse pytest settings from an INI-style file.

    Args:
        path: Path to the INI file.
        section: Section name e.g. "pytest" or "tool:pytest".

    Returns:
        Parsed config or None if the section is absent.
    """
    parser = configparser.ConfigParser()
    try:
        parser.read(path, encoding='utf-8')
    except configparser.Error:
        return None

    if not parser.has_section(section):
        return None

    def _get(key: str) -> tuple[str, ...]:
        if parser.has_option(section, key):
            return _split_value(parser.get(section, key))
        return ()

    return PytestConfig(
        testpaths=_get('testpaths'),
        norecursedirs=_get('norecursedirs'),
        collect_ignore=_get('collect_ignore'),
    )


def load_pytest_config(
    repo_root: Path, *, force_refresh: bool = False
) -> PytestConfig | None:
    """Load pytest configuration from the repo root.

    Searches for config files in priority order and returns the first
    successfully parsed result.  Results are cached keyed on
    ``(path, mtime)``; ``force_refresh=True`` bypasses the cache.

    Args:
        repo_root: Root directory of the repository to search.
        force_refresh: When True, ignore cached values and re-read.

    Returns:
        Parsed config, or None if no pytest config file is found.
    """
    for filename, fmt, section in _CONFIG_CANDIDATES:
        candidate = repo_root / filename
        try:
            stat = candidate.stat()
        except OSError:
            continue

        mtime = stat.st_mtime
        cache_key = (str(candidate), mtime)

        if not force_refresh and cache_key in _CONFIG_CACHE:
            return _CONFIG_CACHE[cache_key]

        if force_refresh:
            # Evict any stale entry for the same path.
            stale = [k for k in _CONFIG_CACHE if k[0] == str(candidate)]
            for k in stale:
                del _CONFIG_CACHE[k]

        if fmt == 'toml':
            config = _parse_toml_config(candidate, section)
        else:
            config = _parse_ini_config(candidate, section)

        _CONFIG_CACHE[cache_key] = config
        return config

    return None


def _default_heuristic(path: str) -> bool:
    """Return True when *path* looks like a test file by convention.

    A path is considered a test file when any path segment is "tests" or
    "test", any segment starts with "test_", or the path ends with
    "conftest.py".

    Args:
        path: File path to classify (forward slashes expected).

    Returns:
        True if the heuristic identifies the path as a test file.
    """
    parts = PurePosixPath(path).parts
    return (
        'tests' in parts
        or 'test' in parts
        or any(p.startswith('test_') for p in parts)
        or path.endswith('conftest.py')
    )


def is_test_file(path: str, config: PytestConfig | None = None) -> bool:
    """Classify whether *path* is a test file.

    When *config* is None the default heuristic is used: a path is a test
    file if any path segment is "tests" or "test", any segment starts with
    "test_", or the path ends with "conftest.py".

    When *config* is provided the following narrowing rules apply (in order):

    - If ``config.testpaths`` is non-empty and *path* is not under any of
      those roots, return False immediately.
    - If any path segment (or ancestor directory segment) matches a pattern
      in ``config.norecursedirs`` (via ``fnmatch.fnmatchcase``), return
      False.
    - If *path* matches any entry in ``config.collect_ignore``, return
      False.
    - If none of the exclusions fired and the default heuristic returns
      True, return True; otherwise False.

    Args:
        path: File path to classify (forward slashes expected).
        config: Optional parsed pytest configuration to apply.

    Returns:
        True if the path should be treated as a test file.
    """
    if config is None:
        return _default_heuristic(path)

    parts = PurePosixPath(path).parts

    if config.testpaths:
        under_testpath = any(
            path == tp or path.startswith(tp.rstrip('/') + '/')
            for tp in config.testpaths
        )
        if not under_testpath:
            return False

    if config.norecursedirs:
        for segment in parts:
            for pattern in config.norecursedirs:
                if fnmatch.fnmatchcase(segment, pattern):
                    return False

    if config.collect_ignore:
        for entry in config.collect_ignore:
            if path == entry or fnmatch.fnmatchcase(path, entry):
                return False

    return _default_heuristic(path)
