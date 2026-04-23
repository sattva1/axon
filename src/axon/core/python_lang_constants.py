"""Python-language classification constants shared between parsers and ingestion.

Placed at ``core/`` rather than ``core/parsers/`` because the concept
(which base classes make a class an enum) is a classification rule used
at ingestion time (dead-code exemption, ENUM_MEMBER extraction), not
parser output. The parser merely gates extraction of
``_extract_enum_members`` on this set - a secondary use.
"""

from __future__ import annotations

ENUM_BASES: frozenset[str] = frozenset(
    {'Enum', 'IntEnum', 'StrEnum', 'Flag', 'IntFlag'}
)

PYDANTIC_BASES: frozenset[str] = frozenset({'BaseModel', 'RootModel'})
# Direct-inheritance detection only. Transitive inheritance is deferred.

DATACLASS_DECORATORS: frozenset[str] = frozenset(
    {'dataclass', 'dataclasses.dataclass', 'attr.s', 'attrs.define'}
)
# Matched against the exact decorator string captured by the parser.
