# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import pytest

from exodusii.core.entities import Entity
from exodusii.core.entities import entity
from exodusii.core.entities import entity_aliases
from exodusii.core.errors import ExodusInvalidEntityError


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("g", Entity.GLOBAL),
        ("global", Entity.GLOBAL),
        ("globals", Entity.GLOBAL),
        ("n", Entity.NODE),
        ("node", Entity.NODE),
        ("nodes", Entity.NODE),
        ("nodal", Entity.NODE),
        ("e", Entity.ELEMENT),
        ("elem", Entity.ELEMENT),
        ("element", Entity.ELEMENT),
        ("elements", Entity.ELEMENT),
        ("d", Entity.EDGE),
        ("edge", Entity.EDGE),
        ("edges", Entity.EDGE),
        ("f", Entity.FACE),
        ("face", Entity.FACE),
        ("faces", Entity.FACE),
        ("eb", Entity.ELEMENT_BLOCK),
        ("block", Entity.ELEMENT_BLOCK),
        ("elem_block", Entity.ELEMENT_BLOCK),
        ("element block", Entity.ELEMENT_BLOCK),
        ("element-block", Entity.ELEMENT_BLOCK),
        ("ns", Entity.NODE_SET),
        ("node_set", Entity.NODE_SET),
        ("node set", Entity.NODE_SET),
        ("nodeset", Entity.NODE_SET),
        ("ss", Entity.SIDE_SET),
        ("side_set", Entity.SIDE_SET),
        ("side set", Entity.SIDE_SET),
        ("sideset", Entity.SIDE_SET),
        ("es", Entity.EDGE_SET),
        ("edge_set", Entity.EDGE_SET),
        ("fs", Entity.FACE_SET),
        ("face_set", Entity.FACE_SET),
        ("els", Entity.ELEMENT_SET),
        ("elem_set", Entity.ELEMENT_SET),
        ("element_set", Entity.ELEMENT_SET),
        ("nm", Entity.NODE_MAP),
        ("node_map", Entity.NODE_MAP),
        ("em", Entity.ELEMENT_MAP),
        ("element_map", Entity.ELEMENT_MAP),
        ("edm", Entity.EDGE_MAP),
        ("edge_map", Entity.EDGE_MAP),
        ("fm", Entity.FACE_MAP),
        ("face_map", Entity.FACE_MAP),
    ],
)
def test_entity_aliases(value: str, expected: Entity) -> None:
    assert entity(value) is expected


def test_entity_accepts_entity() -> None:
    assert entity(Entity.NODE) is Entity.NODE


def test_entity_is_case_insensitive_and_trims_whitespace() -> None:
    assert entity("  NoDe  ") is Entity.NODE
    assert entity("  SIDE SET  ") is Entity.SIDE_SET


def test_invalid_entity_string_raises_exodus_invalid_entity_error() -> None:
    with pytest.raises(ExodusInvalidEntityError, match="Unknown Exodus entity"):
        entity("not an entity")


def test_invalid_entity_type_raises_exodus_invalid_entity_error() -> None:
    with pytest.raises(ExodusInvalidEntityError, match="Expected an Exodus entity"):
        entity(42)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("value", "short_name"),
    [
        (Entity.GLOBAL, "g"),
        (Entity.NODE, "n"),
        (Entity.ELEMENT, "e"),
        (Entity.EDGE, "d"),
        (Entity.FACE, "f"),
        (Entity.NODE_SET, "ns"),
        (Entity.SIDE_SET, "ss"),
    ],
)
def test_short_names(value: Entity, short_name: str) -> None:
    assert value.short_name == short_name


def test_variable_locations() -> None:
    assert Entity.GLOBAL.is_variable_location
    assert Entity.NODE.is_variable_location
    assert Entity.ELEMENT.is_variable_location
    assert Entity.NODE_SET.is_variable_location
    assert not Entity.ELEMENT_BLOCK.is_variable_location
    assert not Entity.NODE_MAP.is_variable_location


def test_object_entities() -> None:
    assert Entity.NODE.is_object
    assert Entity.ELEMENT.is_object
    assert Entity.EDGE.is_object
    assert Entity.FACE.is_object
    assert not Entity.NODE_SET.is_object


def test_block_entities() -> None:
    assert Entity.ELEMENT_BLOCK.is_block
    assert Entity.EDGE_BLOCK.is_block
    assert Entity.FACE_BLOCK.is_block
    assert not Entity.ELEMENT.is_block


def test_set_entities() -> None:
    assert Entity.NODE_SET.is_set
    assert Entity.SIDE_SET.is_set
    assert Entity.EDGE_SET.is_set
    assert Entity.FACE_SET.is_set
    assert Entity.ELEMENT_SET.is_set
    assert not Entity.NODE.is_set


def test_map_entities() -> None:
    assert Entity.NODE_MAP.is_map
    assert Entity.ELEMENT_MAP.is_map
    assert Entity.EDGE_MAP.is_map
    assert Entity.FACE_MAP.is_map
    assert not Entity.NODE.is_map


def test_entity_aliases_returns_copy() -> None:
    aliases = entity_aliases()
    aliases["node"] = Entity.ELEMENT

    assert entity("node") is Entity.NODE
