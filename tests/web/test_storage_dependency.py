"""Tests for the per-request KuzuBackend dependency from axon.web.dependencies.

Each read-only route opens a fresh KuzuBackend via Depends(storage_ro).
The one write route (/api/reindex) opens a KuzuBackend with read_only=False.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from axon.core.storage.kuzu_backend import KuzuBackend
from axon.web.dependencies import storage_ro

# ---------------------------------------------------------------------------
# Test app builders
# ---------------------------------------------------------------------------


def _make_probe_app(db_path: Path) -> FastAPI:
    """Build a minimal FastAPI app with one read-only route for probing.

    Sets up app.state.db_path so that storage_ro can open the right path.
    """
    app = FastAPI()
    app.state.db_path = db_path

    @app.get('/probe')
    def probe(storage: Annotated[KuzuBackend, Depends(storage_ro)]) -> dict:
        """Return the id of the injected backend for identity tracking."""
        return {'backend_id': id(storage)}

    return app


def _make_sentinel_app(db_path: Path) -> FastAPI:
    """Build an app where app.state.storage is a sentinel that raises on access."""

    class _RaisingStorage:
        def __getattr__(self, name: str) -> None:
            raise AssertionError(
                f'app.state.storage was accessed via attribute: {name}'
            )

    app = FastAPI()
    app.state.db_path = db_path
    # Deliberately plant a trip-wire; routes should never touch this.
    app.state.storage = _RaisingStorage()

    @app.get('/probe')
    def probe(storage: Annotated[KuzuBackend, Depends(storage_ro)]) -> dict:
        return {'ok': True}

    return app


def _make_full_app(db_path: Path, repo_path: Path) -> FastAPI:
    """Build a minimal app that includes the analysis router for /reindex."""
    from axon.web.routes.analysis import router as analysis_router

    app = FastAPI()
    app.state.db_path = db_path
    app.state.repo_path = repo_path
    app.state.watch = True
    app.state.event_listeners = []

    app.include_router(analysis_router, prefix='/api')
    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def real_db(tmp_path: Path) -> Path:
    """Create an initialised KuzuDB so RO opens succeed."""
    db_path = tmp_path / 'kuzu'
    backend = KuzuBackend()
    backend.initialize(db_path, read_only=False)
    backend.close()
    return db_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStorageDependency:
    """Per-request storage dependency behaviour tests."""

    def test_route_opens_ro_per_request_via_dependency(
        self, real_db: Path
    ) -> None:
        """Two hits to a storage_ro route produce two distinct backend instances."""
        init_calls: list[tuple] = []

        original_init = KuzuBackend.initialize

        def tracking_init(
            self: KuzuBackend,
            path: Path,
            *,
            read_only: bool = False,
            max_retries: int = 0,
            retry_delay: float = 0.3,
        ) -> None:
            init_calls.append((path, read_only))
            original_init(
                self,
                path,
                read_only=read_only,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )

        app = _make_probe_app(real_db)
        with patch.object(KuzuBackend, 'initialize', tracking_init):
            client = TestClient(app)
            r1 = client.get('/probe')
            r2 = client.get('/probe')

        assert r1.status_code == 200
        assert r2.status_code == 200
        # Two requests -> two distinct opens.
        assert len(init_calls) == 2
        # Both opens use read_only=True.
        assert all(ro is True for _, ro in init_calls)
        # Distinct backend instances per request.
        assert r1.json()['backend_id'] != r2.json()['backend_id']

    def test_route_does_not_consume_app_state_storage(
        self, real_db: Path
    ) -> None:
        """Route succeeds even when app.state.storage is a sentinel that raises."""
        app = _make_sentinel_app(real_db)
        client = TestClient(app)

        response = client.get('/probe')

        assert response.status_code == 200
        assert response.json()['ok'] is True

    def test_run_pipeline_route_opens_rw(
        self, tmp_path: Path, real_db: Path
    ) -> None:
        """POST /api/reindex opens a KuzuBackend with read_only=False."""
        app = _make_full_app(real_db, tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        rw_calls: list[bool] = []
        original_init = KuzuBackend.initialize

        def tracking_init(
            self: KuzuBackend,
            path: Path,
            *,
            read_only: bool = False,
            max_retries: int = 0,
            retry_delay: float = 0.3,
        ) -> None:
            rw_calls.append(read_only)
            original_init(
                self,
                path,
                read_only=read_only,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )

        with (
            patch('axon.web.routes.analysis.run_pipeline') as mock_pipeline,
            patch.object(KuzuBackend, 'initialize', tracking_init),
        ):
            mock_pipeline.return_value = None

            response = client.post('/api/reindex')

        assert response.status_code == 200

        # The reindex route must open at least one RW backend.
        assert any(ro is False for ro in rw_calls), (
            'Expected at least one read_only=False open for the reindex route'
        )
