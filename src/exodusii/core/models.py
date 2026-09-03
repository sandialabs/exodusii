# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Typed data models used by the modern Exodus API.

This module defines frozen dataclasses that represent the core metadata
objects returned and accepted by the Exodus reader/writer: database
initialization parameters, block and set descriptors, variable metadata,
QA records, and information records.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from exodusii.core.entities import Entity
from exodusii.core.entities import entity
from exodusii.core.errors import ExodusInvalidEntityError


@dataclass(frozen=True, slots=True)
class InitParams:
    """Top-level Exodus database initialization parameters.

    Stores the counts of all entity types present in an Exodus database.
    All counts default to zero; ``dimension`` must be 0, 1, 2, or 3.

    Parameters
    ----------
    title : str, optional
        Database title string.  Defaults to ``""``.
    dimension : int, optional
        Spatial dimension: 0, 1, 2, or 3.  Defaults to ``0``.
    nodes : int, optional
        Number of nodes in the mesh.  Defaults to ``0``.
    elements : int, optional
        Number of elements in the mesh.  Defaults to ``0``.
    element_blocks : int, optional
        Number of element blocks.  Defaults to ``0``.
    node_sets : int, optional
        Number of node sets.  Defaults to ``0``.
    side_sets : int, optional
        Number of side sets.  Defaults to ``0``.
    edges : int, optional
        Number of edges.  Defaults to ``0``.
    edge_blocks : int, optional
        Number of edge blocks.  Defaults to ``0``.
    edge_sets : int, optional
        Number of edge sets.  Defaults to ``0``.
    faces : int, optional
        Number of faces.  Defaults to ``0``.
    face_blocks : int, optional
        Number of face blocks.  Defaults to ``0``.
    face_sets : int, optional
        Number of face sets.  Defaults to ``0``.
    element_sets : int, optional
        Number of element sets.  Defaults to ``0``.
    node_maps : int, optional
        Number of node maps.  Defaults to ``0``.
    element_maps : int, optional
        Number of element maps.  Defaults to ``0``.
    edge_maps : int, optional
        Number of edge maps.  Defaults to ``0``.
    face_maps : int, optional
        Number of face maps.  Defaults to ``0``.

    Raises
    ------
    ValueError
        If ``dimension`` is not 0, 1, 2, or 3, or if any count field is
        negative.
    TypeError
        If any count field is not an ``int``.
    """

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
        """Return ``True`` if the database contains nodal or element mesh data.

        Returns
        -------
        bool
            ``True`` when :attr:`nodes` ``> 0`` or :attr:`elements` ``> 0``.
        """

        return self.nodes > 0 or self.elements > 0

    @property
    def has_edges(self) -> bool:
        """Return ``True`` if the database contains edge objects.

        Returns
        -------
        bool
            ``True`` when :attr:`edges` ``> 0`` or :attr:`edge_blocks` ``> 0``.
        """

        return self.edges > 0 or self.edge_blocks > 0

    @property
    def has_faces(self) -> bool:
        """Return ``True`` if the database contains face objects.

        Returns
        -------
        bool
            ``True`` when :attr:`faces` ``> 0`` or :attr:`face_blocks` ``> 0``.
        """

        return self.faces > 0 or self.face_blocks > 0

    def count(self, entity_value: Entity | str) -> int:
        """Return the top-level count for an entity.

        Parameters
        ----------
        entity_value : Entity or str
            The entity type whose count is requested.  Accepts any value
            understood by :func:`~exodusii.core.entities.entity`, such as
            ``Entity.NODE``, ``"node"``, ``"element_block"``, etc.

        Returns
        -------
        int
            The number of objects of that entity type stored in the database
            according to these initialization parameters.

        Raises
        ------
        ExodusInvalidEntityError
            If ``entity_value`` is not a recognized entity, or if it is an
            entity type (e.g. ``GLOBAL``) that does not have an
            ``InitParams`` count field.

        Examples
        --------
        >>> params = InitParams(dimension=3, nodes=100, elements=50,
        ...                     element_blocks=2)
        >>> params.count(Entity.NODE)
        100
        >>> params.count("element_block")
        2
        """

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
    """Metadata for an Exodus block.

    Stores the identity, topology, and sizing information for a single
    element, edge, or face block.  The ``entity`` field is normalized to an
    :class:`~exodusii.core.entities.Entity` member on construction and must be
    one of ``ELEMENT_BLOCK``, ``EDGE_BLOCK``, or ``FACE_BLOCK``.

    Parameters
    ----------
    id : int
        Exodus block ID (positive, user-assigned).
    index : int
        One-based position of this block among all blocks of the same
        entity type (positive).
    entity : Entity or str
        Block entity type; must resolve to a block entity.  Accepts any
        alias understood by :func:`~exodusii.core.entities.entity`.
    element_type : str
        Exodus element-type string (e.g. ``"HEX8"``, ``"TRI3"``).
        Stored in upper case.
    count : int
        Number of elements (or edges/faces) in this block.
    nodes_per_entity : int
        Number of nodes per element (or per edge/face).
    edges_per_entity : int, optional
        Number of edges per element.  Defaults to ``0``.
    faces_per_entity : int, optional
        Number of faces per element.  Defaults to ``0``.
    attributes : int, optional
        Number of per-element attribute fields.  Defaults to ``0``.
    name : str, optional
        Optional user-assigned block name.  Defaults to ``""``.

    Raises
    ------
    ExodusInvalidEntityError
        If ``entity`` does not resolve to a block entity.
    ValueError
        If ``id``, ``index``, or any count field fails its range check.
    TypeError
        If ``id``, ``index``, or any count field is not an ``int``.

    Attributes
    ----------
    id : int
    index : int
    entity : Entity
    element_type : str
    count : int
    nodes_per_entity : int
    edges_per_entity : int
    faces_per_entity : int
    attributes : int
    name : str
    """

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
        """Return ``True`` if this block stores elements.

        Returns
        -------
        bool
            ``True`` when :attr:`entity` is ``Entity.ELEMENT_BLOCK``.
        """

        return self.entity is Entity.ELEMENT_BLOCK

    @property
    def is_edge_block(self) -> bool:
        """Return ``True`` if this block stores edges.

        Returns
        -------
        bool
            ``True`` when :attr:`entity` is ``Entity.EDGE_BLOCK``.
        """

        return self.entity is Entity.EDGE_BLOCK

    @property
    def is_face_block(self) -> bool:
        """Return ``True`` if this block stores faces.

        Returns
        -------
        bool
            ``True`` when :attr:`entity` is ``Entity.FACE_BLOCK``.
        """

        return self.entity is Entity.FACE_BLOCK

    @property
    def legacy_num_block_elems(self) -> int:
        """Legacy-compatible element count attribute value.

        Returns
        -------
        int
            Same as :attr:`count`.  Provided so that code written against the
            legacy Exodus API attribute naming convention continues to work.
        """

        return self.count

    @property
    def legacy_num_elem_nodes(self) -> int:
        """Legacy-compatible nodes-per-element attribute value.

        Returns
        -------
        int
            Same as :attr:`nodes_per_entity`.
        """

        return self.nodes_per_entity

    @property
    def legacy_num_elem_edges(self) -> int:
        """Legacy-compatible edges-per-element attribute value.

        Returns
        -------
        int
            Same as :attr:`edges_per_entity`.
        """

        return self.edges_per_entity

    @property
    def legacy_num_elem_faces(self) -> int:
        """Legacy-compatible faces-per-element attribute value.

        Returns
        -------
        int
            Same as :attr:`faces_per_entity`.
        """

        return self.faces_per_entity

    @property
    def legacy_num_elem_attrs(self) -> int:
        """Legacy-compatible attributes-per-element attribute value.

        Returns
        -------
        int
            Same as :attr:`attributes`.
        """

        return self.attributes


@dataclass(frozen=True, slots=True)
class SetInfo:
    """Metadata and optional payload for an Exodus set.

    Stores the identity and sizing information for a node set, side set,
    edge set, face set, or element set.  The ``entity`` field is normalized
    to an :class:`~exodusii.core.entities.Entity` member on construction and
    must be one of the set entity types.  Entry arrays are converted to
    :class:`numpy.ndarray` if provided.

    Parameters
    ----------
    id : int
        Exodus set ID (positive, user-assigned).
    index : int
        One-based position of this set among all sets of the same entity
        type (positive).
    entity : Entity or str
        Set entity type; must resolve to a set entity.
    count : int
        Number of entries in the set.
    distribution_factors : int, optional
        Number of distribution factor values associated with the set.
        Defaults to ``0``.
    name : str, optional
        Optional user-assigned set name.  Defaults to ``""``.
    entries : array_like or None, optional
        Primary entry array (node IDs for node sets; element IDs for side
        sets).  Converted to :class:`numpy.ndarray` when not ``None``.
    extra_entries : array_like or None, optional
        Secondary entry array (side ordinals for side sets).  Converted
        to :class:`numpy.ndarray` when not ``None``.
    distribution_values : array_like or None, optional
        Distribution factor values.  Converted to :class:`numpy.ndarray`
        when not ``None``.

    Raises
    ------
    ExodusInvalidEntityError
        If ``entity`` does not resolve to a set entity.
    ValueError
        If ``id``, ``index``, or count fields fail range checks.
    TypeError
        If ``id``, ``index``, or count fields are not ``int``.

    Attributes
    ----------
    id : int
    index : int
    entity : Entity
    count : int
    distribution_factors : int
    name : str
    entries : ndarray or None
    extra_entries : ndarray or None
    distribution_values : ndarray or None
    """

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
        """Node-set entries, if this is a node set.

        Returns
        -------
        ndarray or None
            The :attr:`entries` array when :attr:`entity` is
            ``Entity.NODE_SET``, otherwise ``None``.
        """

        return self.entries if self.entity is Entity.NODE_SET else None

    @property
    def elems(self) -> np.ndarray | None:
        """Side-set element entries, if this is a side set.

        Returns
        -------
        ndarray or None
            The :attr:`entries` array (element IDs) when :attr:`entity` is
            ``Entity.SIDE_SET``, otherwise ``None``.
        """

        return self.entries if self.entity is Entity.SIDE_SET else None

    @property
    def sides(self) -> np.ndarray | None:
        """Side-set side entries, if this is a side set.

        Returns
        -------
        ndarray or None
            The :attr:`extra_entries` array (side ordinals) when
            :attr:`entity` is ``Entity.SIDE_SET``, otherwise ``None``.
        """

        return self.extra_entries if self.entity is Entity.SIDE_SET else None

    @property
    def dist_facts(self) -> np.ndarray | None:
        """Distribution factor values, if present.

        Returns
        -------
        ndarray or None
            The :attr:`distribution_values` array, or ``None`` if no
            distribution factors were loaded.
        """

        return self.distribution_values


@dataclass(frozen=True, slots=True)
class VariableInfo:
    """Metadata for an Exodus result variable.

    Describes a single named variable stored at a particular entity location
    in an Exodus database.

    Parameters
    ----------
    name : str
        Variable name (must be non-empty).
    index : int
        One-based index of this variable among all variables at the same
        entity location (positive).
    entity : Entity or str
        Entity location at which the variable is stored.  Must resolve to
        a variable-location entity (one of ``GLOBAL``, ``NODE``,
        ``ELEMENT``, ``EDGE``, ``FACE``, ``NODE_SET``, ``SIDE_SET``,
        ``EDGE_SET``, ``FACE_SET``, or ``ELEMENT_SET``).

    Raises
    ------
    ExodusInvalidEntityError
        If ``entity`` does not resolve to a valid variable location.
    ValueError
        If ``name`` is empty or ``index`` is not positive.
    TypeError
        If ``index`` is not an ``int``.

    Attributes
    ----------
    name : str
    index : int
    entity : Entity
    """

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
        """Return a legacy-style selector string for this variable.

        The selector combines the entity short name and the variable name
        with a ``/`` separator, matching the format used by the legacy
        Exodus Python API (e.g. ``"n/DISPLX"`` for a nodal variable named
        ``DISPLX``).

        Returns
        -------
        str
            A string of the form ``"<short_name>/<variable_name>"``, for
            example ``"n/DISPLX"``, ``"e/STRESS"``, or ``"g/KINETIC_ENERGY"``.

        Examples
        --------
        >>> v = VariableInfo(name="DISPLX", index=1, entity=Entity.NODE)
        >>> v.selector
        'n/DISPLX'
        >>> v = VariableInfo(name="STRESS", index=3, entity=Entity.ELEMENT)
        >>> v.selector
        'e/STRESS'
        """

        return f"{entity(self.entity).short_name}/{self.name}"


@dataclass(frozen=True, slots=True)
class QARecord:
    """A four-field Exodus QA record.

    QA records are written by simulation codes to identify the software
    version, run date, and run time associated with results stored in the
    database.

    Parameters
    ----------
    code_name : str
        Name of the analysis code that produced the data.
    code_qa : str
        Version or QA string for the analysis code.
    date : str
        Date string for the run (format is user-defined, typically
        ``"MM/DD/YYYY"``).
    time : str
        Time string for the run (format is user-defined, typically
        ``"HH:MM:SS"``).

    Attributes
    ----------
    code_name : str
    code_qa : str
    date : str
    time : str
    """

    code_name: str
    code_qa: str
    date: str
    time: str

    def as_tuple(self) -> tuple[str, str, str, str]:
        """Return the record as a 4-tuple.

        Returns
        -------
        tuple of str
            A ``(code_name, code_qa, date, time)`` tuple.

        Examples
        --------
        >>> rec = QARecord("MySolver", "1.0.0", "01/01/2024", "12:00:00")
        >>> rec.as_tuple()
        ('MySolver', '1.0.0', '01/01/2024', '12:00:00')
        """

        return (self.code_name, self.code_qa, self.date, self.time)


@dataclass(frozen=True, slots=True)
class InfoRecord:
    """A single Exodus information record.

    Information records are arbitrary text strings stored in the Exodus
    database header.

    Parameters
    ----------
    text : str
        The information record string.

    Attributes
    ----------
    text : str
        The raw text of the information record.
    """

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
