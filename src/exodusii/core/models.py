# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Typed data models used by the modern Exodus API."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from exodusii.core.entities import Entity
from exodusii.core.entities import entity
from exodusii.core.errors import ExodusInvalidEntityError


@dataclass(frozen=True, slots=True)
class InitParams:
    """Top-level Exodus database initialization parameters."""

    title: str = ""
    dimension: int = 0
    nodes: int = 0
    elements: int = 0
    element_blocks: int = 0
    node_sets: int = 0
    side_sets: int = 0
    edges: int = 0
    edge_blocks: int = 0
    edge_sets: int = 0
    faces: int = 0
    face_blocks: int = 0
    face_sets: int = 0
    element_sets: int = 0
    node_maps: int = 0
    element_maps: int = 0
    edge_maps: int = 0
    face_maps: int = 0

    def __post_init__(self) -> None:
        if self.dimension not in {0, 1, 2, 3}:
            raise ValueError("dimension must be 0, 1, 2, or 3")
        _require_nonnegative_int("dimension", self.dimension)

        for name in (
            "nodes",
            "elements",
            "element_blocks",
            "node_sets",
            "side_sets",
            "edges",
            "edge_blocks",
            "edge_sets",
            "faces",
            "face_blocks",
            "face_sets",
            "element_sets",
            "node_maps",
            "element_maps",
            "edge_maps",
            "face_maps",
        ):
            _require_nonnegative_int(name, getattr(self, name))

    @property
    def has_mesh(self) -> bool:
        """Return true if the database contains nodal or element mesh data."""

        return self.nodes > 0 or self.elements > 0

    @property
    def has_edges(self) -> bool:
        """Return true if the database contains edge objects."""

        return self.edges > 0 or self.edge_blocks > 0

    @property
    def has_faces(self) -> bool:
        """Return true if the database contains face objects."""

        return self.faces > 0 or self.face_blocks > 0

    def count(self, entity_value: Entity | str) -> int:
        """Return the top-level count for an entity."""

        ent = entity(entity_value)
        mapping = {
            Entity.NODE: self.nodes,
            Entity.ELEMENT: self.elements,
            Entity.EDGE: self.edges,
            Entity.FACE: self.faces,
            Entity.ELEMENT_BLOCK: self.element_blocks,
            Entity.EDGE_BLOCK: self.edge_blocks,
            Entity.FACE_BLOCK: self.face_blocks,
            Entity.NODE_SET: self.node_sets,
            Entity.SIDE_SET: self.side_sets,
            Entity.EDGE_SET: self.edge_sets,
            Entity.FACE_SET: self.face_sets,
            Entity.ELEMENT_SET: self.element_sets,
            Entity.NODE_MAP: self.node_maps,
            Entity.ELEMENT_MAP: self.element_maps,
            Entity.EDGE_MAP: self.edge_maps,
            Entity.FACE_MAP: self.face_maps,
        }
        try:
            return mapping[ent]
        except KeyError as exc:
            raise ExodusInvalidEntityError(
                f"{ent.value!r} does not have an InitParams count"
            ) from exc


@dataclass(frozen=True, slots=True)
class Block:
    """Metadata for an Exodus block."""

    id: int
    index: int
    entity: Entity | str
    element_type: str
    count: int
    nodes_per_entity: int
    edges_per_entity: int = 0
    faces_per_entity: int = 0
    attributes: int = 0
    name: str = ""

    def __post_init__(self) -> None:
        normalized = entity(self.entity)
        if not normalized.is_block:
            raise ExodusInvalidEntityError(f"{normalized.value!r} is not a block entity")

        object.__setattr__(self, "entity", normalized)
        object.__setattr__(self, "element_type", self.element_type.upper())

        _require_positive_int("id", self.id)
        _require_positive_int("index", self.index)
        _require_nonnegative_int("count", self.count)
        _require_nonnegative_int("nodes_per_entity", self.nodes_per_entity)
        _require_nonnegative_int("edges_per_entity", self.edges_per_entity)
        _require_nonnegative_int("faces_per_entity", self.faces_per_entity)
        _require_nonnegative_int("attributes", self.attributes)

    @property
    def is_element_block(self) -> bool:
        return self.entity is Entity.ELEMENT_BLOCK

    @property
    def is_edge_block(self) -> bool:
        return self.entity is Entity.EDGE_BLOCK

    @property
    def is_face_block(self) -> bool:
        return self.entity is Entity.FACE_BLOCK

    @property
    def legacy_num_block_elems(self) -> int:
        """Legacy-compatible element count attribute value."""

        return self.count

    @property
    def legacy_num_elem_nodes(self) -> int:
        """Legacy-compatible nodes-per-element attribute value."""

        return self.nodes_per_entity

    @property
    def legacy_num_elem_edges(self) -> int:
        """Legacy-compatible edges-per-element attribute value."""

        return self.edges_per_entity

    @property
    def legacy_num_elem_faces(self) -> int:
        """Legacy-compatible faces-per-element attribute value."""

        return self.faces_per_entity

    @property
    def legacy_num_elem_attrs(self) -> int:
        """Legacy-compatible attributes-per-element attribute value."""

        return self.attributes


@dataclass(frozen=True, slots=True)
class SetInfo:
    """Metadata and optional payload for an Exodus set."""

    id: int
    index: int
    entity: Entity | str
    count: int
    distribution_factors: int = 0
    name: str = ""
    entries: np.ndarray | None = None
    extra_entries: np.ndarray | None = None
    distribution_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        normalized = entity(self.entity)
        if not normalized.is_set:
            raise ExodusInvalidEntityError(f"{normalized.value!r} is not a set entity")

        object.__setattr__(self, "entity", normalized)

        _require_positive_int("id", self.id)
        _require_positive_int("index", self.index)
        _require_nonnegative_int("count", self.count)
        _require_nonnegative_int("distribution_factors", self.distribution_factors)

        if self.entries is not None:
            object.__setattr__(self, "entries", np.asarray(self.entries))

        if self.extra_entries is not None:
            object.__setattr__(self, "extra_entries", np.asarray(self.extra_entries))

        if self.distribution_values is not None:
            object.__setattr__(self, "distribution_values", np.asarray(self.distribution_values))

    @property
    def nodes(self) -> np.ndarray | None:
        """Node-set entries, if this is a node set."""

        return self.entries if self.entity is Entity.NODE_SET else None

    @property
    def elems(self) -> np.ndarray | None:
        """Side-set element entries, if this is a side set."""

        return self.entries if self.entity is Entity.SIDE_SET else None

    @property
    def sides(self) -> np.ndarray | None:
        """Side-set side entries, if this is a side set."""

        return self.extra_entries if self.entity is Entity.SIDE_SET else None

    @property
    def dist_facts(self) -> np.ndarray | None:
        """Distribution factor values, if present."""

        return self.distribution_values


@dataclass(frozen=True, slots=True)
class VariableInfo:
    """Metadata for an Exodus result variable."""

    name: str
    index: int
    entity: Entity | str

    def __post_init__(self) -> None:
        normalized = entity(self.entity)
        if not normalized.is_variable_location:
            raise ExodusInvalidEntityError(f"{normalized.value!r} is not a variable location")

        object.__setattr__(self, "entity", normalized)

        if not self.name:
            raise ValueError("variable name cannot be empty")
        _require_positive_int("index", self.index)

    @property
    def selector(self) -> str:
        """Return a legacy-style selector such as ``n/DISPLX``."""

        return f"{entity(self.entity).short_name}/{self.name}"


@dataclass(frozen=True, slots=True)
class QARecord:
    """A four-field Exodus QA record."""

    code_name: str
    code_qa: str
    date: str
    time: str

    def as_tuple(self) -> tuple[str, str, str, str]:
        """Return the record as a 4-tuple."""

        return (self.code_name, self.code_qa, self.date, self.time)


@dataclass(frozen=True, slots=True)
class InfoRecord:
    """A single Exodus information record."""

    text: str

    def __str__(self) -> str:
        return self.text


def _require_positive_int(name: str, value: Any) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative_int(name: str, value: Any) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


__all__ = ["Block", "InfoRecord", "InitParams", "QARecord", "SetInfo", "VariableInfo"]
