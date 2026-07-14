# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import pytest

from exodusii.core.entities import Entity
from exodusii.core.schema import block_entities
from exodusii.core.schema import block_spec
from exodusii.core.schema import map_spec
from exodusii.core.schema import object_spec
from exodusii.core.schema import set_entities
from exodusii.core.schema import set_spec
from exodusii.core.schema import variable_entities
from exodusii.core.schema import variable_spec
from exodusii.core.schema import variable_value_name


def test_object_specs() -> None:
    node = object_spec("node")
    assert node.entity is Entity.NODE
    assert node.count_dimension == "num_nodes"
    assert node.id_map_variable == "node_num_map"

    elem = object_spec("element")
    assert elem.entity is Entity.ELEMENT
    assert elem.count_dimension == "num_elem"
    assert elem.id_map_variable == "elem_num_map"


def test_element_block_spec() -> None:
    spec = block_spec("element_block")

    assert spec.entity is Entity.ELEMENT_BLOCK
    assert spec.object_entity is Entity.ELEMENT
    assert spec.count_dimension == "num_el_blk"
    assert spec.ids_variable == "eb_prop1"
    assert spec.status_variable == "eb_status"
    assert spec.names_variable == "eb_names"
    assert spec.object_count_dimension(2) == "num_el_in_blk2"
    assert spec.nodes_per_object_dimension(2) == "num_nod_per_el2"
    assert spec.connectivity_variable(2) == "connect2"
    assert spec.edges_per_object_dimension is not None
    assert spec.edges_per_object_dimension(2) == "num_edg_per_el2"
    assert spec.faces_per_object_dimension is not None
    assert spec.faces_per_object_dimension(2) == "num_fac_per_el2"
    assert spec.edge_connectivity_variable is not None
    assert spec.edge_connectivity_variable(2) == "edgconn2"
    assert spec.face_connectivity_variable is not None
    assert spec.face_connectivity_variable(2) == "facconn2"


def test_edge_block_spec() -> None:
    spec = block_spec("edge_block")

    assert spec.entity is Entity.EDGE_BLOCK
    assert spec.object_entity is Entity.EDGE
    assert spec.count_dimension == "num_ed_blk"
    assert spec.ids_variable == "ed_prop1"
    assert spec.status_variable == "ed_status"
    assert spec.names_variable == "ed_names"
    assert spec.object_count_dimension(3) == "num_ed_in_blk3"
    assert spec.nodes_per_object_dimension(3) == "num_nod_per_ed3"
    assert spec.connectivity_variable(3) == "ebconn3"


def test_face_block_spec() -> None:
    spec = block_spec("face_block")

    assert spec.entity is Entity.FACE_BLOCK
    assert spec.object_entity is Entity.FACE
    assert spec.count_dimension == "num_fa_blk"
    assert spec.ids_variable == "fa_prop1"
    assert spec.status_variable == "fa_status"
    assert spec.names_variable == "fa_names"
    assert spec.object_count_dimension(3) == "num_fa_in_blk3"
    assert spec.nodes_per_object_dimension(3) == "num_nod_per_fa3"
    assert spec.connectivity_variable(3) == "fbconn3"


@pytest.mark.parametrize(
    ("name", "entity", "ids", "status", "names", "entries", "dist"),
    [
        (
            "node_set",
            Entity.NODE_SET,
            "ns_prop1",
            "ns_status",
            "ns_names",
            "node_ns2",
            "dist_fact_ns2",
        ),
        (
            "side_set",
            Entity.SIDE_SET,
            "ss_prop1",
            "ss_status",
            "ss_names",
            "elem_ss2",
            "dist_fact_ss2",
        ),
        (
            "edge_set",
            Entity.EDGE_SET,
            "es_prop1",
            "es_status",
            "es_names",
            "edge_es2",
            "dist_fact_es2",
        ),
        (
            "face_set",
            Entity.FACE_SET,
            "fs_prop1",
            "fs_status",
            "fs_names",
            "face_fs2",
            "dist_fact_fs2",
        ),
        (
            "element_set",
            Entity.ELEMENT_SET,
            "els_prop1",
            "els_status",
            "els_names",
            "elem_els2",
            "dist_fact_els2",
        ),
    ],
)
def test_set_specs(
    name: str, entity: Entity, ids: str, status: str, names: str, entries: str, dist: str
) -> None:
    spec = set_spec(name)

    assert spec.entity is entity
    assert spec.ids_variable == ids
    assert spec.status_variable == status
    assert spec.names_variable == names
    assert spec.entries_variable(2) == entries
    assert spec.dist_factors_variable(2) == dist


def test_side_set_extra_entries() -> None:
    spec = set_spec("side_set")

    assert spec.extra_entries_variable is not None
    assert spec.extra_entries_variable(2) == "side_ss2"
    assert spec.extra_entries_name == "sides"


def test_edge_set_extra_entries() -> None:
    spec = set_spec("edge_set")

    assert spec.extra_entries_variable is not None
    assert spec.extra_entries_variable(2) == "ornt_es2"
    assert spec.extra_entries_name == "orientations"


@pytest.mark.parametrize(
    ("name", "count", "names"),
    [
        ("global", "num_glo_var", "name_glo_var"),
        ("node", "num_nod_var", "name_nod_var"),
        ("element", "num_elem_var", "name_elem_var"),
        ("edge", "num_edge_var", "name_edge_var"),
        ("face", "num_face_var", "name_face_var"),
        ("node_set", "num_nset_var", "name_nset_var"),
        ("side_set", "num_sset_var", "name_sset_var"),
        ("edge_set", "num_eset_var", "name_eset_var"),
        ("face_set", "num_fset_var", "name_fset_var"),
        ("element_set", "num_elset_var", "name_elset_var"),
    ],
)
def test_variable_specs(name: str, count: str, names: str) -> None:
    spec = variable_spec(name)

    assert spec.count_dimension == count
    assert spec.names_variable == names


def test_variable_value_names() -> None:
    assert variable_value_name("global", 1) == "vals_glo_var"
    assert variable_value_name("node", 2) == "vals_nod_var2"
    assert variable_value_name("element", 2, 3) == "vals_elem_var2eb3"
    assert variable_value_name("edge", 2, 3) == "vals_edge_var2eb3"
    assert variable_value_name("face", 2, 3) == "vals_face_var2fb3"
    assert variable_value_name("node_set", 2, 3) == "vals_nset_var2ns3"
    assert variable_value_name("side_set", 2, 3) == "vals_sset_var2ss3"
    assert variable_value_name("edge_set", 2, 3) == "vals_eset_var2es3"
    assert variable_value_name("face_set", 2, 3) == "vals_fset_var2fs3"
    assert variable_value_name("element_set", 2, 3) == "vals_elset_var2es3"


def test_map_specs() -> None:
    node = map_spec("node_map")
    assert node.count_dimension == "num_node_maps"
    assert callable(node.map_variable)
    assert node.map_variable(2) == "node_map2"
    assert node.names_variable == "nmap_names"
    assert node.property_variable is not None
    assert node.property_variable(2) == "nm_prop2"


def test_entity_groups() -> None:
    assert set(block_entities()) == {Entity.ELEMENT_BLOCK, Entity.EDGE_BLOCK, Entity.FACE_BLOCK}
    assert set(set_entities()) == {
        Entity.NODE_SET,
        Entity.SIDE_SET,
        Entity.EDGE_SET,
        Entity.FACE_SET,
        Entity.ELEMENT_SET,
    }
    assert Entity.GLOBAL in variable_entities()
    assert Entity.ELEMENT_SET in variable_entities()
