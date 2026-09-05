# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import pytest

from exodusii.core.entities import Entity
from exodusii.core.errors import ExodusInvalidEntityError
from exodusii.core.names import EX
from exodusii.core.names import AttributeName
from exodusii.core.names import DimensionName
from exodusii.core.names import ExodusNames
from exodusii.core.names import VariableName


def test_attribute_names_match_exodus() -> None:
    assert AttributeName.TITLE == "title"
    assert AttributeName.FLOATING_POINT_WORD_SIZE == "floating_point_word_size"
    assert AttributeName.ELEMENT_TYPE == "elem_type"


def test_dimension_names_match_exodus() -> None:
    assert DimensionName.TIME == "time_step"
    assert DimensionName.NUM_NODES == "num_nodes"
    assert DimensionName.NUM_ELEMENTS == "num_elem"
    assert DimensionName.NUM_ELEMENT_BLOCKS == "num_el_blk"


def test_variable_names_match_exodus() -> None:
    assert VariableName.TIME == "time_whole"
    assert VariableName.COORD_X == "coordx"
    assert VariableName.GLOBAL_VARIABLE_VALUES == "vals_glo_var"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("node", "num_nodes"),
        ("edge", "num_edge"),
        ("face", "num_face"),
        ("element", "num_elem"),
        ("element_block", "num_el_blk"),
        ("node_set", "num_node_sets"),
        ("side_set", "num_side_sets"),
    ],
)
def test_dimension_count(value: str, expected: str) -> None:
    assert ExodusNames.dimension_count(value) == expected


def test_dimension_count_rejects_global() -> None:
    with pytest.raises(ExodusInvalidEntityError, match="does not have a count dimension"):
        ExodusNames.dimension_count("global")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("node", "node_num_map"),
        ("element", "elem_num_map"),
        ("edge", "edge_num_map"),
        ("face", "face_num_map"),
        ("element_block", "eb_prop1"),
        ("node_set", "ns_prop1"),
        ("side_set", "ss_prop1"),
    ],
)
def test_ids(value: str, expected: str) -> None:
    assert ExodusNames.ids(value) == expected


def test_ids_rejects_global() -> None:
    with pytest.raises(ExodusInvalidEntityError, match="does not have an ID variable"):
        ExodusNames.ids("global")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("element_block", "eb_status"),
        ("edge_block", "ed_status"),
        ("face_block", "fa_status"),
        ("node_set", "ns_status"),
        ("side_set", "ss_status"),
    ],
)
def test_status(value: str, expected: str) -> None:
    assert ExodusNames.status(value) == expected


def test_status_rejects_node() -> None:
    with pytest.raises(ExodusInvalidEntityError, match="does not have a status variable"):
        ExodusNames.status("node")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("global", "name_glo_var"),
        ("node", "name_nod_var"),
        ("element", "name_elem_var"),
        ("edge", "name_edge_var"),
        ("face", "name_face_var"),
        ("element_block", "eb_names"),
        ("node_set", "ns_names"),
        ("side_set", "ss_names"),
    ],
)
def test_names(value: str, expected: str) -> None:
    assert ExodusNames.names(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("global", "num_glo_var"),
        ("node", "num_nod_var"),
        ("element", "num_elem_var"),
        ("edge", "num_edge_var"),
        ("face", "num_face_var"),
        ("node_set", "num_nset_var"),
        ("side_set", "num_sset_var"),
    ],
)
def test_variable_count(value: str, expected: str) -> None:
    assert ExodusNames.variable_count(value) == expected


def test_variable_count_rejects_block() -> None:
    with pytest.raises(ExodusInvalidEntityError, match="does not have a variable-count"):
        ExodusNames.variable_count("element_block")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("element", "elem_var_tab"),
        ("edge", "edge_var_tab"),
        ("face", "face_var_tab"),
        ("node_set", "nset_var_tab"),
        ("side_set", "sset_var_tab"),
    ],
)
def test_variable_truth_table(value: str, expected: str) -> None:
    assert ExodusNames.variable_truth_table(value) == expected


def test_variable_truth_table_rejects_node() -> None:
    with pytest.raises(ExodusInvalidEntityError, match="does not have a variable truth table"):
        ExodusNames.variable_truth_table("node")


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        (0, "coordx"),
        (1, "coordy"),
        (2, "coordz"),
        ("0", "coordx"),
        ("1", "coordy"),
        ("2", "coordz"),
        ("x", "coordx"),
        ("y", "coordy"),
        ("z", "coordz"),
        ("X", "coordx"),
        ("Y", "coordy"),
        ("Z", "coordz"),
    ],
)
def test_coordinate(axis: int | str, expected: str) -> None:
    assert ExodusNames.coordinate(axis) == expected


@pytest.mark.parametrize("axis", [-1, 3, "q", object()])
def test_coordinate_rejects_invalid_axis(axis: object) -> None:
    with pytest.raises(ValueError, match="Expected coordinate axis"):
        ExodusNames.coordinate(axis)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_element_block_generated_names() -> None:
    assert ExodusNames.block_count(3) == "num_el_in_blk3"
    assert ExodusNames.nodes_per_element(3) == "num_nod_per_el3"
    assert ExodusNames.edges_per_element(3) == "num_edg_per_el3"
    assert ExodusNames.faces_per_element(3) == "num_fac_per_el3"
    assert ExodusNames.attributes_per_element(3) == "num_att_in_blk3"
    assert ExodusNames.element_connectivity(3) == "connect3"
    assert ExodusNames.element_edge_connectivity(3) == "edgconn3"
    assert ExodusNames.element_face_connectivity(3) == "facconn3"


def test_edge_block_generated_names() -> None:
    assert ExodusNames.edge_block_count(2) == "num_ed_in_blk2"
    assert ExodusNames.nodes_per_edge(2) == "num_nod_per_ed2"
    assert ExodusNames.edge_connectivity(2) == "ebconn2"


def test_face_block_generated_names() -> None:
    assert ExodusNames.face_block_count(2) == "num_fa_in_blk2"
    assert ExodusNames.nodes_per_face(2) == "num_nod_per_fa2"
    assert ExodusNames.face_connectivity(2) == "fbconn2"


def test_node_set_generated_names() -> None:
    assert ExodusNames.node_set_count(4) == "num_nod_ns4"
    assert ExodusNames.node_set_distribution_factor_count(4) == "num_df_ns4"
    assert ExodusNames.node_set_nodes(4) == "node_ns4"
    assert ExodusNames.node_set_distribution_factors(4) == "dist_fact_ns4"


def test_side_set_generated_names() -> None:
    assert ExodusNames.side_set_count(5) == "num_side_ss5"
    assert ExodusNames.side_set_distribution_factor_count(5) == "num_df_ss5"
    assert ExodusNames.side_set_elements(5) == "elem_ss5"
    assert ExodusNames.side_set_sides(5) == "side_ss5"
    assert ExodusNames.side_set_distribution_factors(5) == "dist_fact_ss5"


def test_result_variable_generated_names() -> None:
    assert ExodusNames.node_variable(1) == "vals_nod_var1"
    assert ExodusNames.element_variable(2, 3) == "vals_elem_var2eb3"
    assert ExodusNames.edge_variable(2, 3) == "vals_edge_var2eb3"
    assert ExodusNames.face_variable(2, 3) == "vals_face_var2fb3"
    assert ExodusNames.node_set_variable(2, 3) == "vals_nset_var2ns3"
    assert ExodusNames.side_set_variable(2, 3) == "vals_sset_var2ss3"


@pytest.mark.parametrize(
    "call",
    [
        lambda: ExodusNames.block_count(0),
        lambda: ExodusNames.nodes_per_element(-1),
        lambda: ExodusNames.node_variable(0),
        lambda: ExodusNames.element_variable(0, 1),
        lambda: ExodusNames.element_variable(1, 0),
    ],
)
def test_generated_names_require_positive_one_based_indices(call: object) -> None:
    with pytest.raises(ValueError, match="1-based and positive"):
        call()  # ty: ignore[call-non-callable]


def test_generated_names_require_int_indices() -> None:
    with pytest.raises(TypeError, match="must be an int"):
        ExodusNames.block_count(1.5)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_ex_alias_points_to_exodus_names() -> None:
    assert EX is ExodusNames
    assert EX.VAR_TIME == "time_whole"


def test_methods_accept_entity_enum() -> None:
    assert ExodusNames.names(Entity.NODE) == "name_nod_var"
