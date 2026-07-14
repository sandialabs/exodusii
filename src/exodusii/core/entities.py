# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Exodus entity types and string normalization."""

from enum import StrEnum

from exodusii.core.errors import ExodusInvalidEntityError


class Entity(StrEnum):
    """Exodus entity locations.

    The public API accepts strings for ease of use. Internally, strings should be
    normalized to this enum.
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
        """Return the compact one-letter selector used by the legacy API."""

        return _ENTITY_SHORT_NAMES[self]

    @property
    def is_variable_location(self) -> bool:
        """Return whether variables can naturally be stored on this entity."""

        return self in _VARIABLE_LOCATIONS

    @property
    def is_object(self) -> bool:
        """Return whether this entity is a mesh object location."""

        return self in _OBJECT_ENTITIES

    @property
    def is_block(self) -> bool:
        """Return whether this entity is a block entity."""

        return self in _BLOCK_ENTITIES

    @property
    def is_set(self) -> bool:
        """Return whether this entity is a set entity."""

        return self in _SET_ENTITIES

    @property
    def is_map(self) -> bool:
        """Return whether this entity is a map entity."""

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
    value
        An :class:`Entity` or a string alias such as ``"node"``, ``"n"``,
        ``"element"``, ``"elem"``, ``"side set"``, or ``"side_set"``.

    Returns
    -------
    Entity
        The normalized entity enum.

    Raises
    ------
    ExodusInvalidEntityError
        If the entity is not recognized.
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
    """Return a copy of the known entity aliases."""

    return dict(_ENTITY_ALIASES)


__all__ = ["Entity", "entity", "entity_aliases"]
