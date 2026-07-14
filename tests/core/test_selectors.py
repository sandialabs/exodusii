# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import pytest

from exodusii.core.entities import Entity
from exodusii.core.errors import ExodusInvalidEntityError
from exodusii.core.selectors import VariableSelector
from exodusii.core.selectors import parse_variable_selector
from exodusii.core.selectors import parse_variable_selectors


def test_variable_selector_normalizes_entity() -> None:
    selector = VariableSelector("DISPLX", entity="node")

    assert selector.name == "DISPLX"
    assert selector.entity is Entity.NODE
    assert selector.original is None


def test_variable_selector_strips_name() -> None:
    selector = VariableSelector("  DISPLX  ", entity="node")

    assert selector.name == "DISPLX"


def test_variable_selector_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name cannot be empty"):
        VariableSelector("   ", entity="node")


def test_variable_selector_rejects_non_variable_location() -> None:
    with pytest.raises(ExodusInvalidEntityError, match="not a valid variable location"):
        VariableSelector("block_name", entity="element_block")


@pytest.mark.parametrize(
    ("selector", "legacy", "qualified"),
    [
        (VariableSelector("TIME_STEP", entity="global"), "g/TIME_STEP", "global/TIME_STEP"),
        (VariableSelector("DISPLX", entity="node"), "n/DISPLX", "node/DISPLX"),
        (VariableSelector("ENERGY", entity="element"), "e/ENERGY", "element/ENERGY"),
        (VariableSelector("EDGE_VAR", entity="edge"), "d/EDGE_VAR", "edge/EDGE_VAR"),
        (VariableSelector("FACE_VAR", entity="face"), "f/FACE_VAR", "face/FACE_VAR"),
        (VariableSelector("NS_VAR", entity="node_set"), "ns/NS_VAR", "node_set/NS_VAR"),
        (VariableSelector("SS_VAR", entity="side_set"), "ss/SS_VAR", "side_set/SS_VAR"),
    ],
)
def test_variable_selector_string_forms(
    selector: VariableSelector, legacy: str, qualified: str
) -> None:
    assert selector.legacy == legacy
    assert selector.qualified == qualified


def test_variable_selector_is_global() -> None:
    selector = VariableSelector("TM_STEP", entity="global")

    assert selector.is_global
    assert not selector.is_spatial


def test_variable_selector_is_spatial() -> None:
    selector = VariableSelector("DISPLX", entity="node")

    assert not selector.is_global
    assert selector.is_spatial


@pytest.mark.parametrize(
    ("text", "expected_entity", "expected_name", "expected_legacy"),
    [
        ("g/TM_STEP", Entity.GLOBAL, "TM_STEP", "g/TM_STEP"),
        ("global/TM_STEP", Entity.GLOBAL, "TM_STEP", "g/TM_STEP"),
        ("n/DISPLX", Entity.NODE, "DISPLX", "n/DISPLX"),
        ("node/DISPLX", Entity.NODE, "DISPLX", "n/DISPLX"),
        ("nodal/DISPLX", Entity.NODE, "DISPLX", "n/DISPLX"),
        ("e/ENERGY_1", Entity.ELEMENT, "ENERGY_1", "e/ENERGY_1"),
        ("element/ENERGY_1", Entity.ELEMENT, "ENERGY_1", "e/ENERGY_1"),
        ("d/EDGE_VAR", Entity.EDGE, "EDGE_VAR", "d/EDGE_VAR"),
        ("edge/EDGE_VAR", Entity.EDGE, "EDGE_VAR", "d/EDGE_VAR"),
        ("f/FACE_VAR", Entity.FACE, "FACE_VAR", "f/FACE_VAR"),
        ("face/FACE_VAR", Entity.FACE, "FACE_VAR", "f/FACE_VAR"),
        ("ns/NS_VAR", Entity.NODE_SET, "NS_VAR", "ns/NS_VAR"),
        ("node_set/NS_VAR", Entity.NODE_SET, "NS_VAR", "ns/NS_VAR"),
        ("ss/SS_VAR", Entity.SIDE_SET, "SS_VAR", "ss/SS_VAR"),
        ("side_set/SS_VAR", Entity.SIDE_SET, "SS_VAR", "ss/SS_VAR"),
    ],
)
def test_parse_qualified_variable_selector(
    text: str, expected_entity: Entity, expected_name: str, expected_legacy: str
) -> None:
    selector = parse_variable_selector(text)

    assert selector.entity is expected_entity
    assert selector.name == expected_name
    assert selector.legacy == expected_legacy
    assert selector.original == text


def test_parse_qualified_variable_selector_strips_whitespace() -> None:
    selector = parse_variable_selector("  node / DISPLX  ")

    assert selector.entity is Entity.NODE
    assert selector.name == "DISPLX"
    assert selector.legacy == "n/DISPLX"


def test_parse_unqualified_variable_selector_with_default_entity() -> None:
    selector = parse_variable_selector("DISPLX", default_entity="node")

    assert selector.entity is Entity.NODE
    assert selector.name == "DISPLX"
    assert selector.legacy == "n/DISPLX"


def test_parse_variable_selector_returns_existing_selector() -> None:
    existing = VariableSelector("DISPLX", entity="node")

    parsed = parse_variable_selector(existing)

    assert parsed is existing


def test_parse_variable_selector_rejects_non_string_non_selector() -> None:
    with pytest.raises(TypeError, match="must be a string or VariableSelector"):
        parse_variable_selector(42)  # type: ignore[arg-type]


def test_parse_variable_selector_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        parse_variable_selector("   ")


def test_parse_unqualified_variable_selector_requires_default_entity() -> None:
    with pytest.raises(ValueError, match="requires a default_entity"):
        parse_variable_selector("DISPLX")


@pytest.mark.parametrize("text", ["/DISPLX", "node/", "node/DISPLX/extra"])
def test_parse_variable_selector_rejects_invalid_format(text: str) -> None:
    with pytest.raises(ValueError, match="invalid variable selector"):
        parse_variable_selector(text)


def test_parse_variable_selector_rejects_invalid_entity() -> None:
    with pytest.raises(ExodusInvalidEntityError):
        parse_variable_selector("not_an_entity/DISPLX")


def test_parse_variable_selector_rejects_non_variable_entity() -> None:
    with pytest.raises(ExodusInvalidEntityError, match="not a valid variable location"):
        parse_variable_selector("element_block/name")


def test_parse_variable_selectors() -> None:
    selectors = parse_variable_selectors(["n/DISPLX", "node/DISPLY"])

    assert len(selectors) == 2
    assert selectors[0].legacy == "n/DISPLX"
    assert selectors[1].legacy == "n/DISPLY"


def test_parse_variable_selectors_with_default_entity() -> None:
    selectors = parse_variable_selectors(["DISPLX", "DISPLY"], default_entity="node")

    assert [selector.legacy for selector in selectors] == ["n/DISPLX", "n/DISPLY"]


def test_parse_variable_selectors_require_same_entity_passes() -> None:
    selectors = parse_variable_selectors(["n/DISPLX", "node/DISPLY"], require_same_entity=True)

    assert len(selectors) == 2


def test_parse_variable_selectors_require_same_entity_rejects_mixed_entities() -> None:
    with pytest.raises(ValueError, match="must have the same entity"):
        parse_variable_selectors(["n/DISPLX", "e/ENERGY"], require_same_entity=True)


def test_parse_variable_selectors_accepts_empty_iterable() -> None:
    assert parse_variable_selectors([]) == ()
