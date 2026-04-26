"""Canonical helpers for reading and probing the per-repo host.json file.

host.json is written by ``axon serve --watch`` / ``axon host`` to signal
that a live Axon process owns a given repo. It is read by the drift probe
(Tier 0), the CLI liveness checks, and the MCP dispatch layer.

All path construction goes through ``host_json_path`` so callers never
hard-code the ``.axon/host.json`` location.
"""

from __future__ import annotations

import json
from pathlib import Path


def host_json_path(repo_path: Path) -> Path:
    """Canonical path to the per-repo host.json sentinel file."""
    return repo_path / '.axon' / 'host.json'


def load_host_meta(repo_path: Path) -> dict | None:
    """Read and parse host.json for *repo_path*.

    Returns None when the file is absent or malformed.
    """
    path = host_json_path(repo_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return None


def is_host_alive_fast(repo_path: Path) -> bool:
    """Existence-based liveness check - no HTTP probe.

    Used by drift Tier 0 for sub-millisecond probing. Returns True when
    host.json exists, which implies a host process wrote (and has not yet
    removed) the file. No guarantee the process is still running; callers
    that need strong liveness must follow up with an HTTP probe.
    """
    return host_json_path(repo_path).exists()
