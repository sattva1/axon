"""Shared data types for resolution results.

These frozen dataclasses are the return types for all ``resolve_file_*()``
functions.  Being frozen + slots makes them safe to produce from multiple
threads and cheap to allocate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from axon.core.graph.model import RelType


@dataclass(slots=True, frozen=True)
class ResolvedEdge:
    """A resolved relationship ready to be written to the graph."""

    rel_id: str
    rel_type: RelType
    source: str
    target: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class NodePropertyPatch:
    """A deferred property mutation on an existing graph node."""

    node_id: str
    key: str
    value: Any
