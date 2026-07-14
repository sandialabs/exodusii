# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import exodusii.exodus_h as ex
from exodusii.exodus_h import maps
from exodusii.exodus_h import types


def test_legacy_types_enum() -> None:
    assert types.node.value == 0
    assert types.element.value == 1
    assert types.edge.value == 2
    assert types.face.value == 3


def test_legacy_maps_enum() -> None:
    assert maps.elem_local_to_global.value == 0
    assert maps.node_local_to_global.value == 1
    assert maps.elem_block_elem_local_to_global.value == 10


def test_legacy_basic_constants() -> None:
    assert ex.ATT_TITLE == "title"
    assert ex.ATT_NAME_ELEM_TYPE == "elem_type"

    assert ex.DIM_TIME == "time_step"
    assert ex.DIM_NUM_DIM == "num_dim"
    assert ex.DIM_NUM_NODES == "num_nodes"
    assert ex.DIM_NUM_ELEM == "num_elem"

    assert ex.VAR_WHOLE_TIME == "time_whole"
    assert ex.VAR_COORD_X == "coordx"
    assert ex.VAR_COORD_Y == "coordy"
    assert ex.VAR_COORD_Z == "coordz"


def test_legacy_generated_names() -> None:
    assert ex.DIM_NUM_ELEM_IN_ELEM_BLK(2) == "num_el_in_blk2"
    assert ex.DIM_NUM_NODE_PER_ELEM(2) == "num_nod_per_el2"
    assert ex.VAR_ELEM_BLK_CONN(2) == "connect2"
    assert ex.VAR_ELEM_VAR(3, 2) == "vals_elem_var3eb2"
    assert ex.VAR_NODE_VAR(4) == "vals_nod_var4"
    assert ex.VAR_NODE_NODE_SET(5) == "node_ns5"
    assert ex.VAR_SIDE_SIDE_SET(6) == "side_ss6"


def test_legacy_exodus_entity_constants() -> None:
    assert ex.EX_ELEM_BLOCK == 1
    assert ex.EX_NODE_SET == 2
    assert ex.EX_SIDE_SET == 3
    assert ex.EX_GLOBAL == 13
    assert ex.EX_NODE == 15
    assert ex.EX_ELEM == 18


def test_ex_catstr() -> None:
    assert ex.ex_catstr("vals_elem_var", 1, "eb", 2) == "vals_elem_var1eb2"
