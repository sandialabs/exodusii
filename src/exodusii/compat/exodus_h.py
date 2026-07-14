# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Legacy Exodus C-header-style constants.

This module preserves the historical ``exodusii.exodus_h`` import surface while
the modern implementation uses :mod:`exodusii.core.names` internally.
"""

from enum import Enum

from exodusii.core.names import AttributeName
from exodusii.core.names import DimensionName
from exodusii.core.names import ExodusNames
from exodusii.core.names import VariableName


class types(Enum):
    node = 0
    element = 1
    edge = 2
    face = 3


class maps(Enum):
    elem_local_to_global = 0
    node_local_to_global = 1
    edge_local_to_global = 2
    face_local_to_global = 3
    elem_block_elem_local_to_global = 10
    elem_block_elem_global_to_local = 11
    edge_block_edge_local_to_global = 20
    edge_block_edge_global_to_local = 21
    face_block_face_local_to_global = 30
    face_block_face_global_to_local = 31


def ex_catstr(*args: object) -> str:
    """Concatenate Exodus name components."""

    return "".join(str(arg) for arg in args)


# Attributes
ATT_TITLE = AttributeName.TITLE.value
ATT_API_VERSION = AttributeName.API_VERSION.value
ATT_VERSION = AttributeName.VERSION.value
ATT_FILESIZE = AttributeName.FILE_SIZE.value
ATT_FLT_WORDSIZE = AttributeName.FLOATING_POINT_WORD_SIZE.value
ATT_FLT_WORDSIZE_BLANK = AttributeName.FLOATING_POINT_WORD_SIZE_LEGACY.value
ATT_NAME_ELEM_TYPE = AttributeName.ELEMENT_TYPE.value
ATT_PROP_NAME = AttributeName.PROPERTY_NAME.value

# Dimensions
DIM_TIME = DimensionName.TIME.value
DIM_STR = DimensionName.STRING_LENGTH.value
DIM_NAME = DimensionName.NAME_LENGTH.value
DIM_LIN = DimensionName.LINE_LENGTH.value
DIM_N4 = DimensionName.FOUR.value

DIM_NUM_DIM = DimensionName.NUM_DIMENSIONS.value
DIM_NUM_NODES = DimensionName.NUM_NODES.value
DIM_NUM_EDGE = DimensionName.NUM_EDGES.value
DIM_NUM_FACE = DimensionName.NUM_FACES.value
DIM_NUM_ELEM = DimensionName.NUM_ELEMENTS.value

DIM_NUM_ELEM_BLK = DimensionName.NUM_ELEMENT_BLOCKS.value
DIM_NUM_EDGE_BLK = DimensionName.NUM_EDGE_BLOCKS.value
DIM_NUM_FACE_BLK = DimensionName.NUM_FACE_BLOCKS.value

DIM_NUM_NODE_SET = DimensionName.NUM_NODE_SETS.value
DIM_NUM_SIDE_SET = DimensionName.NUM_SIDE_SETS.value
DIM_NUM_EDGE_SET = DimensionName.NUM_EDGE_SETS.value
DIM_NUM_FACE_SET = DimensionName.NUM_FACE_SETS.value
DIM_NUM_ELEM_SET = DimensionName.NUM_ELEMENT_SETS.value

DIM_NUM_GLO_VAR = DimensionName.NUM_GLOBAL_VARIABLES.value
DIM_NUM_NODE_VAR = DimensionName.NUM_NODE_VARIABLES.value
DIM_NUM_ELEM_VAR = DimensionName.NUM_ELEMENT_VARIABLES.value
DIM_NUM_EDGE_VAR = DimensionName.NUM_EDGE_VARIABLES.value
DIM_NUM_FACE_VAR = DimensionName.NUM_FACE_VARIABLES.value

DIM_NUM_NODE_SET_VAR = DimensionName.NUM_NODE_SET_VARIABLES.value
DIM_NUM_SIDE_SET_VAR = DimensionName.NUM_SIDE_SET_VARIABLES.value
DIM_NUM_EDGE_SET_VAR = DimensionName.NUM_EDGE_SET_VARIABLES.value
DIM_NUM_FACE_SET_VAR = DimensionName.NUM_FACE_SET_VARIABLES.value
DIM_NUM_ELEM_SET_VAR = DimensionName.NUM_ELEMENT_SET_VARIABLES.value

DIM_NUM_INFO = DimensionName.NUM_INFO_RECORDS.value
DIM_NUM_QA = DimensionName.NUM_QA_RECORDS.value

DIM_NUM_NODE_MAP = DimensionName.NUM_NODE_MAPS.value
DIM_NUM_ELEM_MAP = DimensionName.NUM_ELEMENT_MAPS.value
DIM_NUM_EDGE_MAP = DimensionName.NUM_EDGE_MAPS.value
DIM_NUM_FACE_MAP = DimensionName.NUM_FACE_MAPS.value

DIM_NUM_NODES_GLOBAL = DimensionName.NUM_NODES_GLOBAL.value
DIM_NUM_NODE_GLOBAL = DimensionName.NUM_NODES_GLOBAL.value
DIM_NUM_ELEM_GLOBAL = DimensionName.NUM_ELEMENTS_GLOBAL.value
DIM_NUM_NODE_SET_GLOBAL = DimensionName.NUM_NODE_SETS_GLOBAL.value
DIM_NUM_SIDE_SET_GLOBAL = DimensionName.NUM_SIDE_SETS_GLOBAL.value
DIM_NUM_ELEM_BLK_GLOBAL = DimensionName.NUM_ELEMENT_BLOCKS_GLOBAL.value

DIM_NUM_ELEM_IN_ELEM_BLK = ExodusNames.block_count
DIM_NUM_NODE_PER_ELEM = ExodusNames.nodes_per_element
DIM_NUM_EDGE_PER_ELEM = ExodusNames.edges_per_element
DIM_NUM_FACE_PER_ELEM = ExodusNames.faces_per_element
DIM_NUM_ATT_IN_ELEM_BLK = ExodusNames.attributes_per_element

DIM_NUM_EDGE_IN_EDGE_BLK = ExodusNames.edge_block_count
DIM_NUM_NODE_PER_EDGE = ExodusNames.nodes_per_edge

DIM_NUM_FACE_IN_FACE_BLK = ExodusNames.face_block_count
DIM_NUM_NODE_PER_FACE = ExodusNames.nodes_per_face
DIM_NUM_ATT_IN_FACE_BLK = lambda num: ex_catstr("num_att_in_fblk", num)
DIM_NUM_ATT_IN_EDGE_BLK = lambda num: ex_catstr("num_att_in_eblk", num)

DIM_NUM_NODE_NODE_SET = ExodusNames.node_set_count
DIM_NUM_DF_NODE_SET = ExodusNames.node_set_distribution_factor_count

DIM_NUM_SIDE_SIDE_SET = ExodusNames.side_set_count
DIM_NUM_DF_SIDE_SET = ExodusNames.side_set_distribution_factor_count

DIM_NUM_EDGE_EDGE_SET = lambda num: ex_catstr("num_edge_es", num)
DIM_NUM_DF_EDGE_SET = lambda num: ex_catstr("num_df_es", num)

DIM_NUM_FACE_FACE_SET = lambda num: ex_catstr("num_face_fs", num)
DIM_NUM_DF_FACE_SET = lambda num: ex_catstr("num_df_fs", num)

DIM_NUM_ELEM_ELEM_SET = lambda num: ex_catstr("num_ele_els", num)
DIM_NUM_DF_ELEM_SET = lambda num: ex_catstr("num_df_els", num)

DIM_NUM_ATTR = lambda num: ex_catstr("num_attr", num)
DIM_NUM_ATT_IN_NODE_SET = lambda num: ex_catstr("num_att_in_ns", num)
DIM_NUM_ATT_IN_SIDE_SET = lambda num: ex_catstr("num_att_in_ss", num)
DIM_NUM_ATT_IN_EDGE_SET = lambda num: ex_catstr("num_att_in_es", num)
DIM_NUM_ATT_IN_FACE_SET = lambda num: ex_catstr("num_att_in_fs", num)
DIM_NUM_ATT_IN_ELEM_SET = lambda num: ex_catstr("num_att_in_els", num)
DIM_NUM_ATT_IN_NODE_BLK = "num_att_in_nblk"

# Variables
VAR_WHOLE_TIME = VariableName.TIME.value

VAR_COORD = VariableName.COORDINATES.value
VAR_COORD_X = VariableName.COORD_X.value
VAR_COORD_Y = VariableName.COORD_Y.value
VAR_COORD_Z = VariableName.COORD_Z.value
VAR_NAME_COORD = VariableName.COORDINATE_NAMES.value

VAR_INFO = VariableName.INFO_RECORDS.value
VAR_QA_TITLE = VariableName.QA_RECORDS.value

VAR_ID_ELEM_BLK = VariableName.ELEMENT_BLOCK_IDS.value
VAR_STAT_ELEM_BLK = VariableName.ELEMENT_BLOCK_STATUS.value
VAR_NAME_ELEM_BLK = VariableName.ELEMENT_BLOCK_NAMES.value

VAR_ID_EDGE_BLK = VariableName.EDGE_BLOCK_IDS.value
VAR_STAT_EDGE_BLK = VariableName.EDGE_BLOCK_STATUS.value
VAR_NAME_EDGE_BLK = VariableName.EDGE_BLOCK_NAMES.value

VAR_ID_FACE_BLK = VariableName.FACE_BLOCK_IDS.value
VAR_STAT_FACE_BLK = VariableName.FACE_BLOCK_STATUS.value
VAR_NAME_FACE_BLK = VariableName.FACE_BLOCK_NAMES.value

VAR_NODE_SET_IDS = VariableName.NODE_SET_IDS.value
VAR_NODE_SET_STAT = VariableName.NODE_SET_STATUS.value
VAR_NAME_NODE_SET = VariableName.NODE_SET_NAMES.value

VAR_SIDE_SET_IDS = VariableName.SIDE_SET_IDS.value
VAR_SIDE_SET_STAT = VariableName.SIDE_SET_STATUS.value
VAR_NAME_SIDE_SET = VariableName.SIDE_SET_NAMES.value

VAR_EDGE_SET_IDS = VariableName.EDGE_SET_IDS.value
VAR_EDGE_SET_STAT = VariableName.EDGE_SET_STATUS.value
VAR_NAME_EDGE_SET = VariableName.EDGE_SET_NAMES.value

VAR_FACE_SET_IDS = VariableName.FACE_SET_IDS.value
VAR_FACE_SET_STAT = VariableName.FACE_SET_STATUS.value
VAR_NAME_FACE_SET = VariableName.FACE_SET_NAMES.value

VAR_ELEM_SET_IDS = VariableName.ELEMENT_SET_IDS.value
VAR_ELEM_SET_STAT = VariableName.ELEMENT_SET_STATUS.value
VAR_NAME_ELEM_SET = VariableName.ELEMENT_SET_NAMES.value

VAR_NAME_GLO_VAR = VariableName.GLOBAL_VARIABLE_NAMES.value
VAR_GLO_VAR = VariableName.GLOBAL_VARIABLE_VALUES.value

VAR_NAME_NODE_VAR = VariableName.NODE_VARIABLE_NAMES.value
VAR_NAME_ELEM_VAR = VariableName.ELEMENT_VARIABLE_NAMES.value
VAR_NAME_EDGE_VAR = VariableName.EDGE_VARIABLE_NAMES.value
VAR_NAME_FACE_VAR = VariableName.FACE_VARIABLE_NAMES.value

VAR_NAME_NODE_SET_VAR = VariableName.NODE_SET_VARIABLE_NAMES.value
VAR_NAME_SIDE_SET_VAR = VariableName.SIDE_SET_VARIABLE_NAMES.value
VAR_NAME_EDGE_SET_VAR = VariableName.EDGE_SET_VARIABLE_NAMES.value
VAR_NAME_FACE_SET_VAR = VariableName.FACE_SET_VARIABLE_NAMES.value
VAR_NAME_ELEM_SET_VAR = VariableName.ELEMENT_SET_VARIABLE_NAMES.value

VAR_ELEM_TAB = VariableName.ELEMENT_VARIABLE_TRUTH_TABLE.value
VAR_EDGE_BLK_TAB = VariableName.EDGE_VARIABLE_TRUTH_TABLE.value
VAR_FACE_BLK_TAB = VariableName.FACE_VARIABLE_TRUTH_TABLE.value
VAR_NODE_SET_TAB = VariableName.NODE_SET_VARIABLE_TRUTH_TABLE.value
VAR_SIDE_SET_TAB = VariableName.SIDE_SET_VARIABLE_TRUTH_TABLE.value
VAR_EDGE_SET_TAB = VariableName.EDGE_SET_VARIABLE_TRUTH_TABLE.value
VAR_FACE_SET_TAB = VariableName.FACE_SET_VARIABLE_TRUTH_TABLE.value
VAR_ELEM_SET_TAB = VariableName.ELEMENT_SET_VARIABLE_TRUTH_TABLE.value

VAR_NODE_NUM_MAP = VariableName.NODE_ID_MAP.value
VAR_ELEM_NUM_MAP = VariableName.ELEMENT_ID_MAP.value
VAR_EDGE_NUM_MAP = VariableName.EDGE_ID_MAP.value
VAR_FACE_NUM_MAP = VariableName.FACE_ID_MAP.value

VAR_NODE_SET_IDS_GLOBAL = VariableName.NODE_SET_IDS_GLOBAL.value
VAR_SIDE_SET_IDS_GLOBAL = VariableName.SIDE_SET_IDS_GLOBAL.value
VAR_ELEM_BLK_IDS_GLOBAL = VariableName.ELEMENT_BLOCK_IDS_GLOBAL.value
VAR_NODE_SET_NODE_COUNT_GLOBAL = VariableName.NODE_SET_NODE_COUNT_GLOBAL.value
VAR_SIDE_SET_SIDE_COUNT_GLOBAL = VariableName.SIDE_SET_SIDE_COUNT_GLOBAL.value
VAR_ELEM_BLK_COUNT_GLOBAL = VariableName.ELEMENT_BLOCK_COUNT_GLOBAL.value

VAR_ELEM_BLK_CONN = ExodusNames.element_connectivity
VAR_EDGE_CONN = ExodusNames.element_edge_connectivity
VAR_FACE_CONN = ExodusNames.element_face_connectivity

VAR_EDGE_BLK_CONN = ExodusNames.edge_connectivity
VAR_FACE_BLK_CONN = ExodusNames.face_connectivity

VAR_NODE_NODE_SET = ExodusNames.node_set_nodes
VAR_DF_NODE_SET = ExodusNames.node_set_distribution_factors

VAR_ELEM_SIDE_SET = ExodusNames.side_set_elements
VAR_SIDE_SIDE_SET = ExodusNames.side_set_sides
VAR_DF_SIDE_SET = ExodusNames.side_set_distribution_factors

VAR_NODE_VAR = ExodusNames.node_variable
VAR_ELEM_VAR = ExodusNames.element_variable
VAR_EDGE_VAR = ExodusNames.edge_variable
VAR_FACE_VAR = ExodusNames.face_variable
VAR_NODE_SET_VAR = ExodusNames.node_set_variable
VAR_SIDE_SET_VAR = ExodusNames.side_set_variable

VAR_EDGE_EDGE_SET = lambda num: ex_catstr("edge_es", num)
VAR_ORNT_EDGE_SET = lambda num: ex_catstr("ornt_es", num)
VAR_DF_EDGE_SET = lambda num: ex_catstr("dist_fact_es", num)

VAR_FACE_FACE_SET = lambda num: ex_catstr("face_fs", num)
VAR_ORNT_FACE_SET = lambda num: ex_catstr("ornt_fs", num)
VAR_DF_FACE_SET = lambda num: ex_catstr("dist_fact_fs", num)

VAR_ELEM_ELEM_SET = lambda num: ex_catstr("elem_els", num)
VAR_DF_ELEM_SET = lambda num: ex_catstr("dist_fact_els", num)

VAR_ELEM_ATTRIB = lambda num: ex_catstr("attrib", num)
VAR_NAME_ELEM_BLK_ATTRIB = lambda num: ex_catstr("attrib_name", num)
VAR_EDGE_BLK_ATTRIB = lambda num: ex_catstr("eattrb", num)
VAR_NAME_EDGE_BLK_ATTRIB = lambda num: ex_catstr("eattrib_name", num)
VAR_FACE_ATTRIB = lambda num: ex_catstr("fattrb", num)
VAR_NAME_FACE_BLK_ATTRIB = lambda num: ex_catstr("fattrib_name", num)

VAR_EB_PROP = lambda num: ex_catstr("eb_prop", num)
VAR_EDGE_PROP = lambda num: ex_catstr("ed_prop", num)
VAR_FACE_PROP = lambda num: ex_catstr("fa_prop", num)
VAR_NODE_SET_PROP = lambda num: ex_catstr("ns_prop", num)
VAR_SIDE_SET_PROP = lambda num: ex_catstr("ss_prop", num)
VAR_EDGE_SET_PROP = lambda num: ex_catstr("es_prop", num)
VAR_FACE_SET_PROP = lambda num: ex_catstr("fs_prop", num)
VAR_ELEM_SET_PROP = lambda num: ex_catstr("els_prop", num)

VAR_NODE_MAP = lambda num: ex_catstr("node_map", num)
VAR_ELEM_MAP = lambda num: ex_catstr("elem_map", num)
VAR_EDGE_MAP = lambda num: ex_catstr("edge_map", num)
VAR_FACE_MAP = lambda num: ex_catstr("face_map", num)

VAR_NODE_MAP_PROP = lambda num: ex_catstr("nm_prop", num)
VAR_ELEM_MAP_PROP = lambda num: ex_catstr("em_prop", num)
VAR_EDGE_MAP_PROP = lambda num: ex_catstr("edm_prop", num)
VAR_FACE_MAP_PROP = lambda num: ex_catstr("fam_prop", num)

# Legacy Exodus constants
MAX_STR_LENGTH = 32
MAX_VAR_NAME_LENGTH = 20
MAX_LINE_LENGTH = 80
MAX_ERR_LENGTH = 256

EX_FATAL = -1
EX_NOERR = 0
EX_WARN = 1

EX_NOCLOBBER = 0
EX_CLOBBER = 1
EX_NORMAL_MODEL = 2
EX_LARGE_MODEL = 4
EX_NETCDF4 = 8
EX_NOSHARE = 16
EX_SHARE = 32

EX_READ = 0
EX_WRITE = 1

EX_ELEM_BLOCK = 1
EX_NODE_SET = 2
EX_SIDE_SET = 3
EX_ELEM_MAP = 4
EX_NODE_MAP = 5
EX_EDGE_BLOCK = 6
EX_EDGE_SET = 7
EX_FACE_BLOCK = 8
EX_FACE_SET = 9
EX_ELEM_SET = 10
EX_EDGE_MAP = 11
EX_FACE_MAP = 12
EX_GLOBAL = 13
EX_NODE = 15
EX_EDGE = 16
EX_FACE = 17
EX_ELEM = 18

EX_VERBOSE = 1
EX_DEBUG = 2
EX_ABORT = 4

EX_ELEM_UNK = (-1,)
EX_ELEM_NULL_ELEMENT = 0
EX_ELEM_TRIANGLE = 1
EX_ELEM_QUAD = 2
EX_ELEM_HEX = 3
EX_ELEM_WEDGE = 4
EX_ELEM_TETRA = 5
EX_ELEM_TRUSS = 6
EX_ELEM_BEAM = 7
EX_ELEM_SHELL = 8
EX_ELEM_SPHERE = 9
EX_ELEM_CIRCLE = 10
EX_ELEM_TRISHELL = 11
EX_ELEM_PYRAMID = 12
