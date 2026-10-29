# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Names from the exodusii < 2 exodus_h module are still available."""

from exodusii.compat import exodus_h as ex


def test_restored_inquiry_codes() -> None:
    assert ex.EX_INQ_FILE_TYPE == 1
    assert ex.EX_INQ_TIME == 16
    assert ex.EX_INQ_COORD_FRAMES == 47


def test_restored_variable_names() -> None:
    assert ex.VAR_EDGE_SET_VAR(1, 2) == "vals_eset_var1es2"
    assert ex.VAR_FACE_SET_VAR(1, 2) == "vals_fset_var1fs2"
    assert ex.VAR_ELEM_SET_VAR(1, 2) == "vals_elset_var1es2"
    assert ex.VAR_EBEPEC(3) == "ebepecnt3"
    assert ex.VAR_NODE_SET_ATTRIB(2) == "nsattrb2"
    assert ex.VAR_NAME_NODE_MAP == "nmap_names"
    assert ex.VAR_NODE_SET_DF_COUNT_GLOBAL == "ns_df_cnt_global"
    assert ex.DIM_NUM_SIDE == "num_side"


def test_elem_unknown_is_an_int() -> None:
    assert ex.EX_ELEM_UNK == -1
