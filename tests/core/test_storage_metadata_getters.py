"""Tests for get_callers_with_metadata and get_callees_with_metadata.

Covers Phase 4b: metadata_json propagation on CALLS edges, including
JSON parsing, empty metadata, and malformed JSON resilience.

Note on fixture design: the KuzuBackend serialises extra relationship
properties (those not in _DEDICATED_REL_PROPS) into the metadata_json
column automatically via _serialize_extra_props. So to store
dispatch_kind='thread_executor' on an edge, pass it directly in
``properties``, not as a pre-serialised string.

For the invalid-JSON test, we write directly to the DB via execute_raw
so we can bypass the serialisation layer and inject a raw bad string.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.storage.kuzu_backend import KuzuBackend


@pytest.fixture()
def backend(tmp_path: Path) -> KuzuBackend:
    """Return a KuzuBackend initialised in a temporary directory."""
    db_path = tmp_path / 'meta_test_db'
    b = KuzuBackend()
    b.initialize(db_path)
    yield b
    b.close()


def _fn(file_path: str, name: str) -> GraphNode:
    """Create a FUNCTION node with a deterministic ID."""
    return GraphNode(
        id=generate_id(NodeLabel.FUNCTION, file_path, name),
        label=NodeLabel.FUNCTION,
        name=name,
        file_path=file_path,
    )


def _calls(
    caller: GraphNode, callee: GraphNode, extra_props: dict | None = None
) -> GraphRelationship:
    """Build a CALLS relationship with optional extra props.

    Extra props (dispatch_kind, in_try, etc.) are stored directly in
    properties; the backend serialises non-dedicated props into metadata_json.
    """
    props: dict = {'confidence': 1.0}
    if extra_props:
        props.update(extra_props)
    return GraphRelationship(
        id=f'calls:{caller.id}->{callee.id}',
        type=RelType.CALLS,
        source=caller.id,
        target=callee.id,
        properties=props,
    )


class TestGetCallersWithMetadata:
    """get_callers_with_metadata: incoming CALLS edge metadata parsing."""

    def test_parses_json_metadata(self, backend: KuzuBackend) -> None:
        """dispatch_kind and in_try are correctly round-tripped via metadata_json."""
        caller = _fn('src/a.py', 'caller')
        callee = _fn('src/b.py', 'callee')
        backend.add_nodes([caller, callee])
        backend.add_relationships(
            [
                _calls(
                    caller,
                    callee,
                    {'dispatch_kind': 'thread_executor', 'in_try': True},
                )
            ]
        )

        results = backend.get_callers_with_metadata(callee.id)

        assert len(results) == 1
        node, conf, meta = results[0]
        assert node.id == caller.id
        assert conf == 1.0
        assert meta.get('dispatch_kind') == 'thread_executor'
        assert meta.get('in_try') is True

    def test_empty_metadata_returns_empty_dict(
        self, backend: KuzuBackend
    ) -> None:
        """Edge with no extra props returns an empty metadata dict."""
        caller = _fn('src/a.py', 'caller_plain')
        callee = _fn('src/b.py', 'callee_plain')
        backend.add_nodes([caller, callee])
        backend.add_relationships([_calls(caller, callee)])

        results = backend.get_callers_with_metadata(callee.id)

        assert len(results) == 1
        _node, _conf, meta = results[0]
        assert meta == {}

    def test_invalid_json_returns_empty_dict(
        self, backend: KuzuBackend
    ) -> None:
        """Malformed metadata_json stored via raw SQL does not raise.

        Bypasses the serialisation layer to inject a bad JSON string directly.
        """
        caller = _fn('src/a.py', 'caller_bad')
        callee = _fn('src/b.py', 'callee_bad')
        backend.add_nodes([caller, callee])
        # Insert without metadata, then patch via raw Cypher.
        backend.add_relationships([_calls(caller, callee)])
        backend.execute_raw(
            f'MATCH (a:Function)-[r:CodeRelation]->(b:Function) '
            f"WHERE a.id = '{caller.id}' AND b.id = '{callee.id}' "
            f"SET r.metadata_json = '{{not json'"
        )

        results = backend.get_callers_with_metadata(callee.id)

        assert len(results) == 1
        _node, _conf, meta = results[0]
        assert meta == {}

    def test_no_callers_returns_empty_list(self, backend: KuzuBackend) -> None:
        """Node with no incoming CALLS edges returns an empty list."""
        lone = _fn('src/lone.py', 'lonely')
        backend.add_nodes([lone])

        results = backend.get_callers_with_metadata(lone.id)

        assert results == []


class TestGetCalleesWithMetadata:
    """get_callees_with_metadata: outgoing CALLS edge metadata parsing."""

    def test_parses_json_metadata(self, backend: KuzuBackend) -> None:
        """dispatch_kind is correctly round-tripped on a callee query."""
        caller = _fn('src/c.py', 'runner')
        callee = _fn('src/d.py', 'task')
        backend.add_nodes([caller, callee])
        backend.add_relationships(
            [_calls(caller, callee, {'dispatch_kind': 'detached_task'})]
        )

        results = backend.get_callees_with_metadata(caller.id)

        assert len(results) == 1
        node, conf, meta = results[0]
        assert node.id == callee.id
        assert conf == 1.0
        assert meta.get('dispatch_kind') == 'detached_task'

    def test_empty_metadata_returns_empty_dict(
        self, backend: KuzuBackend
    ) -> None:
        """Edge with no extra props returns empty dict on callee query."""
        caller = _fn('src/c.py', 'runner_plain')
        callee = _fn('src/d.py', 'task_plain')
        backend.add_nodes([caller, callee])
        backend.add_relationships([_calls(caller, callee)])

        results = backend.get_callees_with_metadata(caller.id)

        assert len(results) == 1
        _node, _conf, meta = results[0]
        assert meta == {}

    def test_invalid_json_returns_empty_dict(
        self, backend: KuzuBackend
    ) -> None:
        """Malformed metadata_json does not raise on callee query."""
        caller = _fn('src/c.py', 'runner_bad')
        callee = _fn('src/d.py', 'task_bad')
        backend.add_nodes([caller, callee])
        backend.add_relationships([_calls(caller, callee)])
        backend.execute_raw(
            f'MATCH (a:Function)-[r:CodeRelation]->(b:Function) '
            f"WHERE a.id = '{caller.id}' AND b.id = '{callee.id}' "
            f"SET r.metadata_json = '{{not json'"
        )

        results = backend.get_callees_with_metadata(caller.id)

        assert len(results) == 1
        _node, _conf, meta = results[0]
        assert meta == {}

    def test_no_callees_returns_empty_list(self, backend: KuzuBackend) -> None:
        """Node with no outgoing CALLS edges returns an empty list."""
        lone = _fn('src/lone.py', 'lonesome')
        backend.add_nodes([lone])

        results = backend.get_callees_with_metadata(lone.id)

        assert results == []
