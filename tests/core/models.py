# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from exodusii.core.entities import Entity
from exodusii.core.errors import ExodusInvalidEntityError
from exodusii.core.models import Block
from exodusii.core.models import InfoRecord
from exodusii.core.models import InitParams
from exodusii.core.models import QARecord
from exodusii.core.models import SetInfo
from exodusii.core.models import VariableInfo


def test_init_params_defaults() -> None:
    params = InitParams()

    assert params.title == ""
    assert params.dimension == 0
    assert params.nodes == 0
    assert params.elements == 0
    assert not params.has_mesh
    assert not params.has_edges
    assert not params.has_faces


def test_init_params_basic_values() -> None:
    params = InitParams(
        title="mesh", dimension=2, nodes=10, elements=5, element_blocks=1, node_sets=2, side_sets=3
    )

    assert params.title == "mesh"
    assert params.dimension == 2
    assert params.nodes == 10
    assert params.elements == 5
    assert params.element_blocks == 1
    assert params.has_mesh


@pytest.mark.parametrize("dimension", [0, 1, 2, 3])
def test_init_params_valid_dimensions(dimension: int) -> None:
    params = InitParams(dimension=dimension)
    assert params.dimension == dimension


@pytest.mark.parametrize("dimension", [-1, 4])
def test_init_params_rejects_invalid_dimension(dimension: int) -> None:
    with pytest.raises(ValueError, match="dimension must be"):
        InitParams(dimension=dimension)


def test_init_params_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="nodes must be nonnegative"):
        InitParams(nodes=-1)


def test_init_params_rejects_noninteger_counts() -> None:
    with pytest.raises(TypeError, match="nodes must be an int"):
        InitParams(nodes=1.5)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("entity_value", "expected"),
    [
        ("node", 10),
        ("element", 5),
        ("element_block", 2),
        ("node_set", 3),
        ("side_set", 4),
        ("edge", 6),
        ("face", 7),
    ],
)
def test_init_params_count(entity_value: str, expected: int) -> None:
    params = InitParams(
        nodes=10, elements=5, element_blocks=2, node_sets=3, side_sets=4, edges=6, faces=7
    )

    assert params.count(entity_value) == expected


def test_init_params_count_rejects_global() -> None:
    params = InitParams()

    with pytest.raises(ExodusInvalidEntityError, match="does not have an InitParams count"):
        params.count("global")


def test_block_normalizes_entity_and_element_type() -> None:
    block = Block(
        id=10,
        index=1,
        entity="element_block",
        element_type="hex8",
        count=20,
        nodes_per_entity=8,
        edges_per_entity=12,
        faces_per_entity=6,
        attributes=2,
        name="block 10",
    )

    assert block.id == 10
    assert block.index == 1
    assert block.entity is Entity.ELEMENT_BLOCK
    assert block.element_type == "HEX8"
    assert block.count == 20
    assert block.nodes_per_entity == 8
    assert block.edges_per_entity == 12
    assert block.faces_per_entity == 6
    assert block.attributes == 2
    assert block.name == "block 10"
    assert block.is_element_block
    assert not block.is_edge_block
    assert not block.is_face_block


def test_block_legacy_properties() -> None:
    block = Block(
        id=10,
        index=1,
        entity=Entity.ELEMENT_BLOCK,
        element_type="quad",
        count=3,
        nodes_per_entity=4,
        edges_per_entity=4,
        faces_per_entity=0,
        attributes=1,
    )

    assert block.legacy_num_block_elems == 3
    assert block.legacy_num_elem_nodes == 4
    assert block.legacy_num_elem_edges == 4
    assert block.legacy_num_elem_faces == 0
    assert block.legacy_num_elem_attrs == 1


def test_block_rejects_non_block_entity() -> None:
    with pytest.raises(ExodusInvalidEntityError, match="is not a block entity"):
        Block(id=1, index=1, entity="node", element_type="node", count=1, nodes_per_entity=1)


def test_block_rejects_bad_id() -> None:
    with pytest.raises(ValueError, match="id must be positive"):
        Block(
            id=0, index=1, entity="element_block", element_type="quad", count=1, nodes_per_entity=4
        )


def test_block_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="count must be nonnegative"):
        Block(
            id=1, index=1, entity="element_block", element_type="quad", count=-1, nodes_per_entity=4
        )


def test_set_info_node_set_payload() -> None:
    info = SetInfo(
        id=100,
        index=1,
        entity="node_set",
        count=3,
        distribution_factors=3,
        name="nodes",
        entries=[1, 2, 3],
        distribution_values=[1.0, 2.0, 3.0],
    )

    assert info.entity is Entity.NODE_SET
    assert info.name == "nodes"
    assert info.nodes is not None
    assert np.allclose(info.nodes, [1, 2, 3])
    assert info.elems is None
    assert info.sides is None
    assert info.dist_facts is not None
    assert np.allclose(info.dist_facts, [1.0, 2.0, 3.0])


def test_set_info_side_set_payload() -> None:
    info = SetInfo(
        id=200, index=2, entity="side_set", count=2, entries=[10, 11], extra_entries=[1, 2]
    )

    assert info.entity is Entity.SIDE_SET
    assert info.elems is not None
    assert np.allclose(info.elems, [10, 11])
    assert info.sides is not None
    assert np.allclose(info.sides, [1, 2])
    assert info.nodes is None


def test_set_info_rejects_non_set_entity() -> None:
    with pytest.raises(ExodusInvalidEntityError, match="is not a set entity"):
        SetInfo(id=1, index=1, entity="node", count=1)


def test_variable_info_normalizes_entity() -> None:
    info = VariableInfo(name="DISPLX", index=1, entity="node")

    assert info.name == "DISPLX"
    assert info.index == 1
    assert info.entity is Entity.NODE
    assert info.selector == "n/DISPLX"


def test_variable_info_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="variable name cannot be empty"):
        VariableInfo(name="", index=1, entity="node")


def test_variable_info_rejects_non_variable_location() -> None:
    with pytest.raises(ExodusInvalidEntityError, match="is not a variable location"):
        VariableInfo(name="foo", index=1, entity="element_block")


def test_variable_info_rejects_nonpositive_index() -> None:
    with pytest.raises(ValueError, match="index must be positive"):
        VariableInfo(name="foo", index=0, entity="node")


def test_qa_record_as_tuple() -> None:
    record = QARecord("code", "1.0", "2026-07-14", "12:00:00")

    assert record.as_tuple() == ("code", "1.0", "2026-07-14", "12:00:00")


def test_info_record_string_conversion() -> None:
    record = InfoRecord("created by test")

    assert str(record) == "created by test"


def test_models_are_frozen() -> None:
    params = InitParams(title="mesh")

    with pytest.raises(FrozenInstanceError):
        params.title = "other"  # type: ignore[misc]  # ty: ignore[invalid-assignment]
