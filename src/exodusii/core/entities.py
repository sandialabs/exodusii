# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Exodus entity types and string normalization.

Defines the :class:`Entity` enumeration that identifies every object and
variable-storage location in an Exodus database, together with the
:func:`entity` normalization function that converts the many accepted string
aliases to their canonical :class:`Entity` member.
"""

from enum import StrEnum

from exodusii.core.errors import ExodusInvalidEntityError


class Entity(StrEnum):
    """Exodus entity locations.

    The public API accepts strings for ease of use.  Internally, strings
    should be normalized to this enum using :func:`entity`.  Because
    :class:`Entity` inherits from :class:`~enum.StrEnum`, members compare
    equal to their string values (e.g. ``Entity.NODE == "node"``).

    Notes
    -----
    Full list of members and their roles:

    GLOBAL
        Database-wide scalar variables; not associated with any mesh object.
    NODE
        Nodal locations; variable storage and object counting.
    ELEMENT
        Element locations.
    EDGE
        Edge locations.
    FACE
        Face locations.
    ELEMENT_BLOCK
        Element block descriptor; groups elements of the same topology.
    EDGE_BLOCK
        Edge block descriptor.
    FACE_BLOCK
        Face block descriptor.
    NODE_SET
        Named collection of node IDs.
    SIDE_SET
        Named collection of (element, side) pairs.
    EDGE_SET
        Named collection of edge IDs.
    FACE_SET
        Named collection of face IDs.
    ELEMENT_SET
        Named collection of element IDs.
    NODE_MAP
        Optional renumbering map for nodes.
    ELEMENT_MAP
        Optional renumbering map for elements.
    EDGE_MAP
        Optional renumbering map for edges.
    FACE_MAP
        Optional renumbering map for faces.

    Examples
    --------
    Direct member access:

    >>> Entity.NODE
    <Entity.NODE: 'node'>
    >>> Entity.NODE == "node"
    True

    Parse from a string alias via :func:`entity`:

    >>> from exodusii.core.entities import entity
    >>> entity("elem")
    <Entity.ELEMENT: 'element'>
    >>> entity("side_set")
    <Entity.SIDE_SET: 'side_set'>
    """

    GLOBAL = "global"
    NODE = "node"
    ELEMENT = "element"
    EDGE = "edge"
    FACE = "face"

    ELEMENT_BLOCK = "element_block"
    EDGE_BLOCK = "edge_block"
    FACE_BLOCK = "face_block"

    NODE_SET = "node_set"
    SIDE_SET = "side_set"
    EDGE_SET = "edge_set"
    FACE_SET = "face_set"
    ELEMENT_SET = "element_set"

    NODE_MAP = "node_map"
    ELEMENT_MAP = "element_map"
    EDGE_MAP = "edge_map"
    FACE_MAP = "face_map"

    @property
    def short_name(self) -> str:
        """Return the compact one-letter selector used by the legacy API.

        Returns
        -------
        str
            A short string identifier for this entity, such as ``"n"`` for
            nodes, ``"e"`` for elements, ``"ss"`` for side sets, etc.
            These match the prefixes used by legacy Exodus selector strings
            (e.g. ``"n/DISPLX"``).
        """

        return _ENTITY_SHORT_NAMES[self]

    @property
    def is_variable_location(self) -> bool:
        """Return ``True`` if variables can naturally be stored on this entity.

        Returns
        -------
        bool
            ``True`` for ``GLOBAL``, ``NODE``, ``ELEMENT``, ``EDGE``,
            ``FACE``, ``NODE_SET``, ``SIDE_SET``, ``EDGE_SET``, ``FACE_SET``,
            and ``ELEMENT_SET``; ``False`` for block and map entities.
        """

        return self in _VARIABLE_LOCATIONS

    @property
    def is_object(self) -> bool:
        """Return ``True`` if this entity is a mesh object location.

        Returns
        -------
        bool
            ``True`` for ``NODE``, ``ELEMENT``, ``EDGE``, and ``FACE``.
        """

        return self in _OBJECT_ENTITIES

    @property
    def is_block(self) -> bool:
        """Return ``True`` if this entity is a block entity.

        Returns
        -------
        bool
            ``True`` for ``ELEMENT_BLOCK``, ``EDGE_BLOCK``, and
            ``FACE_BLOCK``.
        """

        return self in _BLOCK_ENTITIES

    @property
    def is_set(self) -> bool:
        """Return ``True`` if this entity is a set entity.

        Returns
        -------
        bool
            ``True`` for ``NODE_SET``, ``SIDE_SET``, ``EDGE_SET``,
            ``FACE_SET``, and ``ELEMENT_SET``.
        """

        return self in _SET_ENTITIES

    @property
    def is_map(self) -> bool:
        """Return ``True`` if this entity is a map entity.

        Returns
        -------
        bool
            ``True`` for ``NODE_MAP``, ``ELEMENT_MAP``, ``EDGE_MAP``, and
            ``FACE_MAP``.
        """

        return self in _MAP_ENTITIES


_ENTITY_SHORT_NAMES: dict[Entity, str] = {
    Entity.GLOBAL: "g",
    Entity.NODE: "n",
    Entity.ELEMENT: "e",
    Entity.EDGE: "d",
    Entity.FACE: "f",
    Entity.ELEMENT_BLOCK: "eb",
    Entity.EDGE_BLOCK: "edb",
    Entity.FACE_BLOCK: "fb",
    Entity.NODE_SET: "ns",
    Entity.SIDE_SET: "ss",
    Entity.EDGE_SET: "es",
    Entity.FACE_SET: "fs",
    Entity.ELEMENT_SET: "els",
    Entity.NODE_MAP: "nm",
    Entity.ELEMENT_MAP: "em",
    Entity.EDGE_MAP: "edm",
    Entity.FACE_MAP: "fm",
}

_VARIABLE_LOCATIONS = frozenset(
    {
        Entity.GLOBAL,
        Entity.NODE,
        Entity.ELEMENT,
        Entity.EDGE,
        Entity.FACE,
        Entity.NODE_SET,
        Entity.SIDE_SET,
        Entity.EDGE_SET,
        Entity.FACE_SET,
        Entity.ELEMENT_SET,
    }
)

_OBJECT_ENTITIES = frozenset({Entity.NODE, Entity.ELEMENT, Entity.EDGE, Entity.FACE})

_BLOCK_ENTITIES = frozenset({Entity.ELEMENT_BLOCK, Entity.EDGE_BLOCK, Entity.FACE_BLOCK})

_SET_ENTITIES = frozenset(
    {Entity.NODE_SET, Entity.SIDE_SET, Entity.EDGE_SET, Entity.FACE_SET, Entity.ELEMENT_SET}
)

_MAP_ENTITIES = frozenset({Entity.NODE_MAP, Entity.ELEMENT_MAP, Entity.EDGE_MAP, Entity.FACE_MAP})


def _entity_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


_ENTITY_ALIASES: dict[str, Entity] = {
    # Global variables
    "g": Entity.GLOBAL,
    "global": Entity.GLOBAL,
    "globals": Entity.GLOBAL,
    "global_variable": Entity.GLOBAL,
    "global_variables": Entity.GLOBAL,
    # Nodes
    "n": Entity.NODE,
    "node": Entity.NODE,
    "nodes": Entity.NODE,
    "nodal": Entity.NODE,
    # Elements
    "e": Entity.ELEMENT,
    "el": Entity.ELEMENT,
    "elem": Entity.ELEMENT,
    "elems": Entity.ELEMENT,
    "element": Entity.ELEMENT,
    "elements": Entity.ELEMENT,
    # Edges.  The legacy compact selector for edge variables is "d".
    "d": Entity.EDGE,
    "edge": Entity.EDGE,
    "edges": Entity.EDGE,
    # Faces
    "f": Entity.FACE,
    "face": Entity.FACE,
    "faces": Entity.FACE,
    # Element blocks
    "eb": Entity.ELEMENT_BLOCK,
    "block": Entity.ELEMENT_BLOCK,
    "blocks": Entity.ELEMENT_BLOCK,
    "elem_block": Entity.ELEMENT_BLOCK,
    "elem_blocks": Entity.ELEMENT_BLOCK,
    "element_block": Entity.ELEMENT_BLOCK,
    "element_blocks": Entity.ELEMENT_BLOCK,
    # Edge blocks
    "edb": Entity.EDGE_BLOCK,
    "edge_block": Entity.EDGE_BLOCK,
    "edge_blocks": Entity.EDGE_BLOCK,
    # Face blocks
    "fb": Entity.FACE_BLOCK,
    "face_block": Entity.FACE_BLOCK,
    "face_blocks": Entity.FACE_BLOCK,
    # Node sets
    "ns": Entity.NODE_SET,
    "nset": Entity.NODE_SET,
    "nsets": Entity.NODE_SET,
    "nodeset": Entity.NODE_SET,
    "nodesets": Entity.NODE_SET,
    "node_set": Entity.NODE_SET,
    "node_sets": Entity.NODE_SET,
    # Side sets
    "ss": Entity.SIDE_SET,
    "sset": Entity.SIDE_SET,
    "ssets": Entity.SIDE_SET,
    "sideset": Entity.SIDE_SET,
    "sidesets": Entity.SIDE_SET,
    "side_set": Entity.SIDE_SET,
    "side_sets": Entity.SIDE_SET,
    # Edge sets
    "es": Entity.EDGE_SET,
    "eset": Entity.EDGE_SET,
    "esets": Entity.EDGE_SET,
    "edgeset": Entity.EDGE_SET,
    "edgesets": Entity.EDGE_SET,
    "edge_set": Entity.EDGE_SET,
    "edge_sets": Entity.EDGE_SET,
    # Face sets
    "fs": Entity.FACE_SET,
    "fset": Entity.FACE_SET,
    "fsets": Entity.FACE_SET,
    "faceset": Entity.FACE_SET,
    "facesets": Entity.FACE_SET,
    "face_set": Entity.FACE_SET,
    "face_sets": Entity.FACE_SET,
    # Element sets
    "els": Entity.ELEMENT_SET,
    "elset": Entity.ELEMENT_SET,
    "elsets": Entity.ELEMENT_SET,
    "elemset": Entity.ELEMENT_SET,
    "elemsets": Entity.ELEMENT_SET,
    "elementset": Entity.ELEMENT_SET,
    "elementsets": Entity.ELEMENT_SET,
    "elem_set": Entity.ELEMENT_SET,
    "elem_sets": Entity.ELEMENT_SET,
    "element_set": Entity.ELEMENT_SET,
    "element_sets": Entity.ELEMENT_SET,
    # Maps
    "nm": Entity.NODE_MAP,
    "node_map": Entity.NODE_MAP,
    "node_maps": Entity.NODE_MAP,
    "em": Entity.ELEMENT_MAP,
    "elem_map": Entity.ELEMENT_MAP,
    "elem_maps": Entity.ELEMENT_MAP,
    "element_map": Entity.ELEMENT_MAP,
    "element_maps": Entity.ELEMENT_MAP,
    "edm": Entity.EDGE_MAP,
    "edge_map": Entity.EDGE_MAP,
    "edge_maps": Entity.EDGE_MAP,
    "fm": Entity.FACE_MAP,
    "face_map": Entity.FACE_MAP,
    "face_maps": Entity.FACE_MAP,
}


def entity(value: Entity | str) -> Entity:
    """Normalize an entity-like value to :class:`Entity`.

    Parameters
    ----------
    value : Entity or str
        An :class:`Entity` or a string alias such as ``"node"``, ``"n"``,
        ``"element"``, ``"elem"``, ``"side set"``, or ``"side_set"``.
        String lookup is case-insensitive; hyphens and spaces are treated
        as underscores.

    Returns
    -------
    Entity
        The normalized entity enum member.

    Raises
    ------
    ExodusInvalidEntityError
        If the entity is not recognized, or if ``value`` is neither an
        :class:`Entity` nor a ``str``.

    Examples
    --------
    Pass-through for an existing :class:`Entity`:

    >>> entity(Entity.NODE)
    <Entity.NODE: 'node'>

    Normalize common string aliases:

    >>> entity("n")
    <Entity.NODE: 'node'>
    >>> entity("elem")
    <Entity.ELEMENT: 'element'>
    >>> entity("side set")
    <Entity.SIDE_SET: 'side_set'>
    >>> entity("NODE_SET")
    <Entity.NODE_SET: 'node_set'>
    >>> entity("edb")
    <Entity.EDGE_BLOCK: 'edge_block'>
    """

    if isinstance(value, Entity):
        return value

    if not isinstance(value, str):
        raise ExodusInvalidEntityError(
            f"Expected an Exodus entity string or Entity enum, got {type(value).__name__}"
        )

    key = _entity_key(value)
    try:
        return _ENTITY_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(_ENTITY_ALIASES))
        raise ExodusInvalidEntityError(
            f"Unknown Exodus entity {value!r}. Expected one of: {valid}"
        ) from exc


def entity_aliases() -> dict[str, Entity]:
    """Return a copy of the known entity aliases.

    Returns
    -------
    dict of {str: Entity}
        Mapping from every recognized alias string (lower-cased, with
        underscores) to its canonical :class:`Entity` member.  The returned
        dict is a shallow copy; modifying it does not affect the internal
        alias table.

    Examples
    --------
    >>> aliases = entity_aliases()
    >>> aliases["n"]
    <Entity.NODE: 'node'>
    >>> aliases["sset"]
    <Entity.SIDE_SET: 'side_set'>
    """

    return dict(_ENTITY_ALIASES)


__all__ = ["Entity", "entity", "entity_aliases"]
