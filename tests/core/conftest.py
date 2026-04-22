from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """Minimal Python repository for core watcher/staleness tests."""
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'app.py').write_text(
        'def hello():\n    return "hello"\n', encoding='utf-8'
    )
    return tmp_path
