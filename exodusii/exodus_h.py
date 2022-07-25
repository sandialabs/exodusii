from enum import Enum


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


def ex_catstr(*args):
    return "".join(str(_) for _ in args)


# -------------------------------------------------------- exodusII_inc.h --- #
MAX_VAR_NAME_LENGTH = 20  # Internal use only

# Default "filesize" for newly created files.
# Set to 0 for normal filesize setting.
# Set to 1 for EXODUS_LARGE_MODEL setting to be the default
EXODUS_DEFAULT_SIZE = 1

# Exodus error return codes - function return values:
EX_FATAL = -1  # fatal error flag def
EX_NOERR = 0  # no error flag def
EX_WARN = 1  # warning flag def

# This file contains defined constants that are used internally in the EXODUS II API.
#
# The first group of constants refer to netCDF variables, attributes, or dimensions in
# which the EXODUS II data are stored.  Using the defined constants will allow the
# names of the netCDF entities to be changed easily in the future if needed.  The first
# three letters of the constant identify the netCDF entity as a variable (VAR),
# dimension (DIM), or attribute (ATT).
#
# NOTE: The entity name should not have any blanks in it.  Blanks are
#       technically legal but some netcdf utilities (ncgen in particular)
#       fail when they encounter a blank in a name.

ATT_TITLE = "title"  # the database title
ATT_API_VERSION = "api_version"  # the EXODUS II api vers number
ATT_VERSION = "version"  # the EXODUS II file vers number
ATT_FILESIZE = "file_size"  # 1=large, 0=normal

# word size of floating point numbers in file
ATT_FLT_WORDSIZE = "floating_point_word_size"

# word size of floating point numbers in file used for db version 2.01 and earlier
ATT_FLT_WORDSIZE_BLANK = "floating point word size"

ATT_NAME_ELEM_TYPE = "elem_type"  # element type names for each element block

DIM_NUM_NODES = "num_nodes"  # number of nodes
DIM_NUM_DIM = "num_dim"  # number of dimensions; 2- or 3-d
DIM_NUM_EDGE = "num_edge"  # number of edges (over all blks)
DIM_NUM_FACE = "num_face"  # number of faces (over all blks)
DIM_NUM_ELEM = "num_elem"  # number of elements
DIM_NUM_ELEM_BLK = "num_el_blk"  # number of element blocks
DIM_NUM_EDGE_BLK = "num_ed_blk"  # number of edge blocks
DIM_NUM_FACE_BLK = "num_fa_blk"  # number of face blocks

DIM_NUM_ELEM_GLOBAL = "num_elems_global"
DIM_NUM_NODE_GLOBAL = "num_nodes_global"

DIM_NUM_NODE_SET_GLOBAL = "num_ns_global"
DIM_NUM_SIDE_SET_GLOBAL = "num_ss_global"
DIM_NUM_ELEM_BLK_GLOBAL = "num_el_blk_global"

VAR_ELEM_BLK_COUNT_GLOBAL = "el_blk_cnt_global"
VAR_SIDE_SET_SIDE_COUNT_GLOBAL = "ss_side_cnt_global"
VAR_NODE_SET_NODE_COUNT_GLOBAL = "ns_node_cnt_global"

VAR_NODE_SET_DF_COUNT_GLOBAL = "ns_df_cnt_global"
VAR_SIDE_SET_DF_COUNT_GLOBAL = "ss_df_cnt_global"
VAR_EDGE_SET_DF_COUNT_GLOBAL = "es_df_cnt_global"
VAR_ELEM_SET_DF_COUNT_GLOBAL = "els_df_cnt_global"

VAR_ELEM_BLK_IDS_GLOBAL = "el_blk_ids_global"
VAR_NODE_SET_IDS_GLOBAL = "ns_ids_global"
VAR_SIDE_SET_IDS_GLOBAL = "ss_ids_global"

VAR_COORD = "coord"  # nodal coordinates
VAR_COORD_X = "coordx"  # X-dimension coordinate
VAR_COORD_Y = "coordy"  # Y-dimension coordinate
VAR_COORD_Z = "coordz"  # Z-dimension coordinate
VAR_NAME_COORD = "coor_names"  # names of coordinates
VAR_NAME_ELEM_BLK = "eb_names"  # names of element blocks
VAR_NAME_NODE_SET = "ns_names"  # names of node sets
VAR_NAME_SIDE_SET = "ss_names"  # names of side sets
VAR_NAME_ELEM_MAP = "emap_names"  # names of element maps
VAR_NAME_EDGE_MAP = "edmap_names"  # names of edge maps
VAR_NAME_FACE_MAP = "famap_names"  # names of face maps
VAR_NAME_NODE_MAP = "nmap_names"  # names of node maps
VAR_NAME_EDGE_BLK = "ed_names"  # names of edge blocks
VAR_NAME_FACE_BLK = "fa_names"  # names of face blocks
VAR_NAME_EDGE_SET = "es_names"  # names of edge sets
VAR_NAME_FACE_SET = "fs_names"  # names of face sets
VAR_NAME_ELEM_SET = "els_names"  # names of element sets
VAR_STAT_ELEM_BLK = "eb_status"  # element block status
VAR_STAT_EDGE_CONN = "econn_status"  # element block edge status
VAR_STAT_FACE_CONN = "fconn_status"  # element block face status
VAR_STAT_EDGE_BLK = "ed_status"  # edge block status
VAR_STAT_FACE_BLK = "fa_status"  # face block status
VAR_ID_ELEM_BLK = "eb_prop1"  # element block ids props
VAR_ID_EDGE_BLK = "ed_prop1"  # edge block ids props
VAR_ID_FACE_BLK = "fa_prop1"  # face block ids props

DIM_NUM_ATTR = lambda num: ex_catstr("num_attr", num)
# number of elements in element block num
DIM_NUM_ELEM_IN_ELEM_BLK = lambda num: ex_catstr("num_el_in_blk", num)

# number of nodes per element in element block num
DIM_NUM_NODE_PER_ELEM = lambda num: ex_catstr("num_nod_per_el", num)

# number of attributes in element block num
DIM_NUM_ATT_IN_ELEM_BLK = lambda num: ex_catstr("num_att_in_blk", num)

# number of edges in edge block num
DIM_NUM_EDGE_IN_EDGE_BLK = lambda num: ex_catstr("num_ed_in_blk", num)

# number of nodes per edge in edge block num
DIM_NUM_NODE_PER_EDGE = lambda num: ex_catstr("num_nod_per_ed", num)

# number of edges per element in element block num
DIM_NUM_EDGE_PER_ELEM = lambda num: ex_catstr("num_edg_per_el", num)

# number of attributes in edge block num
DIM_NUM_ATT_IN_EDGE_BLK = lambda num: ex_catstr("num_att_in_eblk", num)

# number of faces in face block num
DIM_NUM_FACE_IN_FACE_BLK = lambda num: ex_catstr("num_fa_in_blk", num)

# number of nodes per face in face block num
DIM_NUM_NODE_PER_FACE = lambda num: ex_catstr("num_nod_per_fa", num)

# number of faces per element in element block num
DIM_NUM_FACE_PER_ELEM = lambda num: ex_catstr("num_fac_per_el", num)

# number of attributes in face block num
DIM_NUM_ATT_IN_FACE_BLK = lambda num: ex_catstr("num_att_in_fblk", num)
DIM_NUM_ATT_IN_NODE_BLK = "num_att_in_nblk"

# element connectivity for element block num
VAR_ELEM_BLK_CONN = lambda num: ex_catstr("connect", num)

# array containing number of entity per entity for n-sided face/element blocks
VAR_EBEPEC = lambda num: ex_catstr("ebepecnt", num)

# list of attributes for element block num
VAR_ELEM_ATTRIB = lambda num: ex_catstr("attrib", num)

# list of attribute names for element block num
VAR_NAME_ELEM_BLK_ATTRIB = lambda num: ex_catstr("attrib_name", num)

# list of the numth property for all element blocks
VAR_EB_PROP = lambda num: ex_catstr("eb_prop", num)

# edge connectivity for element block num
VAR_EDGE_CONN = lambda num: ex_catstr("edgconn", num)

# edge connectivity for edge block num
VAR_EDGE_BLK_CONN = lambda num: ex_catstr("ebconn", num)

# list of attributes for edge block num
VAR_EDGE_BLK_ATTRIB = lambda num: ex_catstr("eattrb", num)
# list of attribute names for edge block num
VAR_NAME_EDGE_BLK_ATTRIB = lambda num: ex_catstr("eattrib_name", num)

VAR_NATTRIB = "nattrb"
VAR_NAME_NATTRIB = "nattrib_name"
DIM_NUM_ATT_IN_NODE_BLK = "num_att_in_nblk"
VAR_NODE_SET_ATTRIB = lambda num: ex_catstr("nsattrb", num)
VAR_NAME_NODE_SET_ATTRIB = lambda num: ex_catstr("nsattrib_name", num)
DIM_NUM_ATT_IN_NODE_SET = lambda num: ex_catstr("num_att_in_ns", num)
VAR_SIDE_SET_ATTRIB = lambda num: ex_catstr("ssattrb", num)
VAR_NAME_SIDE_SET_ATTRIB = lambda num: ex_catstr("ssattrib_name", num)
DIM_NUM_ATT_IN_SIDE_SET = lambda num: ex_catstr("num_att_in_ss", num)
VAR_EDGE_SET_ATTRIB = lambda num: ex_catstr("esattrb", num)
VAR_NAME_EDGE_SET_ATTRIB = lambda num: ex_catstr("esattrib_name", num)
DIM_NUM_ATT_IN_EDGE_SET = lambda num: ex_catstr("num_att_in_es", num)
VAR_FACE_SET_ATTRIB = lambda num: ex_catstr("fsattrb", num)
VAR_NAME_FACE_SET_ATTRIB = lambda num: ex_catstr("fsattrib_name", num)
DIM_NUM_ATT_IN_FACE_SET = lambda num: ex_catstr("num_att_in_fs", num)
VAR_ELEM_SET_ATTRIB = lambda num: ex_catstr("elsattrb", num)
VAR_NAME_ELEM_SET_ATTRIB = lambda num: ex_catstr("elsattrib_name", num)
DIM_NUM_ATT_IN_ELEM_SET = lambda num: ex_catstr("num_att_in_els", num)
VAR_EDGE_PROP = lambda num: ex_catstr("ed_prop", num)

# face connectivity for element block num
VAR_FACE_CONN = lambda num: ex_catstr("facconn", num)

# face connectivity for element block num
VAR_FACE_BLK_CONN = lambda num: ex_catstr("fbconn", num)

# face connectivity for face block num
VAR_FBEPEC = lambda num: ex_catstr("fbepecnt", num)

# array containing number of entity per entity for n-sided face/element blocks
VAR_FACE_ATTRIB = lambda num: ex_catstr("fattrb", num)

# list of attributes for face block num
VAR_NAME_FACE_BLK_ATTRIB = lambda num: ex_catstr("fattrib_name", num)

# list of attribute names for face block num
VAR_FACE_PROP = lambda num: ex_catstr("fa_prop", num)

# list of the numth property for all face blocks
ATT_PROP_NAME = "name"  # name attached to element

# block, node set, side set, element map, or map properties
DIM_NUM_SIDE_SET = "num_side_sets"  # number of side sets
VAR_SIDE_SET_STAT = "ss_status"  # side set status
VAR_SIDE_SET_IDS = "ss_prop1"  # side set id properties

# number of sides in side set num
DIM_NUM_SIDE_SIDE_SET = lambda num: ex_catstr("num_side_ss", num)

# number of distribution factors in side set num
DIM_NUM_DF_SIDE_SET = lambda num: ex_catstr("num_df_ss", num)

# the distribution factors for each node in side set num
VAR_DF_SIDE_SET = lambda num: ex_catstr("dist_fact_ss", num)

# list of elements in side set num
VAR_ELEM_SIDE_SET = lambda num: ex_catstr("elem_ss", num)

# list of sides in side set
VAR_SIDE_SIDE_SET = lambda num: ex_catstr("side_ss", num)

# list of the numth property for all side sets
VAR_SIDE_SET_PROP = lambda num: ex_catstr("ss_prop", num)

DIM_NUM_EDGE_SET = "num_edge_sets"  # number of edge sets
VAR_EDGE_SET_STAT = "es_status"  # edge set status
VAR_EDGE_SET_IDS = "es_prop1"  # edge set id properties

# number of edges in edge set num
DIM_NUM_EDGE_EDGE_SET = lambda num: ex_catstr("num_edge_es", num)

DIM_NUM_DF_EDGE_SET = lambda num: ex_catstr("num_df_es", num)

# number of distribution factors in edge set num
VAR_DF_EDGE_SET = lambda num: ex_catstr("dist_fact_es", num)

# list of edges in edge set num
VAR_EDGE_EDGE_SET = lambda num: ex_catstr("edge_es", num)

# list of orientations in the edge set.
VAR_ORNT_EDGE_SET = lambda num: ex_catstr("ornt_es", num)

# list of the numth property for all edge sets
VAR_EDGE_SET_PROP = lambda num: ex_catstr("es_prop", num)

DIM_NUM_FACE_SET = "num_face_sets"  # number of face sets
VAR_FACE_SET_STAT = "fs_status"  # face set status
VAR_FACE_SET_IDS = "fs_prop1"  # face set id properties

# number of faces in side set num
DIM_NUM_FACE_FACE_SET = lambda num: ex_catstr("num_face_fs", num)

# number of distribution factors in face set num
DIM_NUM_DF_FACE_SET = lambda num: ex_catstr("num_df_fs", num)

# the distribution factors for each node in face set num
VAR_DF_FACE_SET = lambda num: ex_catstr("dist_fact_fs", num)

# list of elements in face set num
VAR_FACE_FACE_SET = lambda num: ex_catstr("face_fs", num)

# list of sides in side set
VAR_ORNT_FACE_SET = lambda num: ex_catstr("ornt_fs", num)

# list of the numth property for all face sets
VAR_FACE_SET_PROP = lambda num: ex_catstr("fs_prop", num)

DIM_NUM_ELEM_SET = "num_elem_sets"  # number of elem sets

# number of elements in elem set num
DIM_NUM_ELEM_ELEM_SET = lambda num: ex_catstr("num_ele_els", num)

# number of distribution factors in element set num
DIM_NUM_DF_ELEM_SET = lambda num: ex_catstr("num_df_els", num)

VAR_ELEM_SET_STAT = "els_status"  # elem set status
VAR_ELEM_SET_IDS = "els_prop1"  # elem set id properties

# list of elements in elem set num
VAR_ELEM_ELEM_SET = lambda num: ex_catstr("elem_els", num)

# list of distribution factors in elem set num
VAR_DF_ELEM_SET = lambda num: ex_catstr("dist_fact_els", num)

# list of the numth property for all elem sets
VAR_ELEM_SET_PROP = lambda num: ex_catstr("els_prop", num)

DIM_NUM_NODE_SET = "num_node_sets"  # number of node sets

# number of nodes in node set num
DIM_NUM_NODE_NODE_SET = lambda num: ex_catstr("num_nod_ns", num)

# number of distribution factors in node set num
DIM_NUM_DF_NODE_SET = lambda num: ex_catstr("num_df_ns", num)

VAR_NODE_SET_STAT = "ns_status"  # node set status
VAR_NODE_SET_IDS = "ns_prop1"  # node set id properties

# list of nodes in node set num
VAR_NODE_NODE_SET = lambda num: ex_catstr("node_ns", num)

# list of distribution factors in node set num
VAR_DF_NODE_SET = lambda num: ex_catstr("dist_fact_ns", num)

# list of the numth property for all node sets
VAR_NODE_SET_PROP = lambda num: ex_catstr("ns_prop", num)

DIM_NUM_QA = "num_qa_rec"  # number of QA records
VAR_QA_TITLE = "qa_records"  # QA records
DIM_NUM_INFO = "num_info"  # number of information records
VAR_INFO = "info_records"  # information records
VAR_WHOLE_TIME = "time_whole"  # simulation times for whole time steps
VAR_ELEM_TAB = "elem_var_tab"  # element variable truth table
VAR_EDGE_BLK_TAB = "edge_var_tab"  # edge variable truth table
VAR_FACE_BLK_TAB = "face_var_tab"  # face variable truth table
VAR_ELEM_SET_TAB = "elset_var_tab"  # elemset variable truth table
VAR_SIDE_SET_TAB = "sset_var_tab"  # sideset variable truth table
VAR_FACE_SET_TAB = "fset_var_tab"  # faceset variable truth table
VAR_EDGE_SET_TAB = "eset_var_tab"  # edgeset variable truth table
VAR_NODE_SET_TAB = "nset_var_tab"  # nodeset variable truth table
DIM_NUM_GLO_VAR = "num_glo_var"  # number of global variables
VAR_NAME_GLO_VAR = "name_glo_var"  # names of global variables
VAR_GLO_VAR = "vals_glo_var"  # values of global variables
DIM_NUM_NODE_VAR = "num_nod_var"  # number of nodal variables
VAR_NAME_NODE_VAR = "name_nod_var"  # names of nodal variables

VAR_NODE_VAR = lambda num: ex_catstr("vals_nod_var", num)  # values of nodal variables

DIM_NUM_ELEM_VAR = "num_elem_var"  # number of element variables
VAR_NAME_ELEM_VAR = "name_elem_var"  # names of element variables

# values of element variable num1 in element block num2
VAR_ELEM_VAR = lambda num1, num2: ex_catstr("vals_elem_var", num1, "eb", num2)

DIM_NUM_EDGE_VAR = "num_edge_var"  # number of edge variables
VAR_NAME_EDGE_VAR = "name_edge_var"  # names of edge variables

# values of edge variable num1 in edge block num2
VAR_EDGE_VAR = lambda num1, num2: ex_catstr("vals_edge_var", num1, "eb", num2)

DIM_NUM_FACE_VAR = "num_face_var"  # number of face variables
VAR_NAME_FACE_VAR = "name_face_var"  # names of face variables

# values of face variable num1 in face block num2
VAR_FACE_VAR = lambda num1, num2: ex_catstr("vals_face_var", num1, "fb", num2)

DIM_NUM_NODE_SET_VAR = "num_nset_var"  # number of nodeset variables
VAR_NAME_NODE_SET_VAR = "name_nset_var"  # names of nodeset variables

# values of nodeset variable num1 in nodeset num2
VAR_NODE_SET_VAR = lambda num1, num2: ex_catstr("vals_nset_var", num1, "ns", num2)

DIM_NUM_EDGE_SET_VAR = "num_eset_var"  # number of edgeset variables
VAR_NAME_EDGE_SET_VAR = "name_eset_var"  # names of edgeset variables

# values of edgeset variable num1 in edgeset num2
VAR_EDGE_SET_VAR = lambda num1, num2: ex_catstr("vals_eset_var", num1, "es", num2)

DIM_NUM_FACE_SET_VAR = "num_fset_var"  # number of faceset variables
VAR_NAME_FACE_SET_VAR = "name_fset_var"  # names of faceset variables

# values of faceset variable num1 in faceset num2
VAR_FACE_SET_VAR = lambda num1, num2: ex_catstr("vals_fset_var", num1, "fs", num2)

DIM_NUM_SIDE_SET_VAR = "num_sset_var"  # number of sideset variables
VAR_NAME_SIDE_SET_VAR = "name_sset_var"  # names of sideset variables

# values of sideset variable num1 in sideset num2
VAR_SIDE_SET_VAR = lambda num1, num2: ex_catstr("vals_sset_var", num1, "ss", num2)

DIM_NUM_ELEM_SET_VAR = "num_elset_var"  # number of element set variables
VAR_NAME_ELEM_SET_VAR = "name_elset_var"  # names of elemset variables

# values of elemset variable num1 in elemset num2
VAR_ELEM_SET_VAR = lambda num1, num2: ex_catstr("vals_elset_var", num1, "es", num2)

# general dimension of length MAX_STR_LENGTH used for name lengths
DIM_STR = "len_string"
DIM_NAME = "len_name"
DIM_NUM_SIDE = "num_side"

# general dimension of length MAX_LINE_LENGTH used for long strings
DIM_LIN = "len_line"
DIM_N4 = "four"  # general dimension of length 4


# unlimited (expandable) dimension for time steps
DIM_TIME = "time_step"

VAR_ELEM_NUM_MAP = "elem_num_map"
VAR_EDGE_NUM_MAP = "edge_num_map"
VAR_FACE_NUM_MAP = "face_num_map"
VAR_NODE_NUM_MAP = "node_num_map"

DIM_NUM_ELEM_MAP = "num_elem_maps"  # number of element maps
VAR_ELEM_MAP = lambda num: ex_catstr("elem_map", num)  # the numth element map

# list of the numth property for all element maps
VAR_ELEM_MAP_PROP = lambda num: ex_catstr("em_prop", num)

DIM_NUM_EDGE_MAP = "num_edge_maps"  # number of edge maps
VAR_EDGE_MAP = lambda num: ex_catstr("edge_map", num)  # the numth edge map

# list of the numth property for all edge maps
VAR_EDGE_MAP_PROP = lambda num: ex_catstr("edm_prop", num)

DIM_NUM_FACE_MAP = "num_face_maps"  # number of face maps
VAR_FACE_MAP = lambda num: ex_catstr("face_map", num)  # the numth face map

# list of the numth property for all face maps
VAR_FACE_MAP_PROP = lambda num: ex_catstr("fam_prop", num)

DIM_NUM_NODE_MAP = "num_node_maps"  # number of node maps
VAR_NODE_MAP = lambda num: ex_catstr("node_map", num)  # the numth node map

# list of the numth property for all node maps
VAR_NODE_MAP_PROP = lambda num: ex_catstr("nm_prop", num)

EX_ELEM_UNK = (-1,)  # unknown entity
EX_ELEM_NULL_ELEMENT = 0
EX_ELEM_TRIANGLE = 1  # Triangle entity
EX_ELEM_QUAD = 2  # Quad entity
EX_ELEM_HEX = 3  # Hex entity
EX_ELEM_WEDGE = 4  # Wedge entity
EX_ELEM_TETRA = 5  # Tetra entity
EX_ELEM_TRUSS = 6  # Truss entity
EX_ELEM_BEAM = 7  # Beam entity
EX_ELEM_SHELL = 8  # Shell entity
EX_ELEM_SPHERE = 9  # Sphere entity
EX_ELEM_CIRCLE = 10  # Circle entity
EX_ELEM_TRISHELL = 11  # Triangular Shell entity
EX_ELEM_PYRAMID = 12  # Pyramid entity

# ------------------------------------------------------------ exodusII.h --- #
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
EX_NODE = 15  # not defined in exodus
EX_EDGE = 16  # not defined in exodus
EX_FACE = 17  # not defined in exodus
EX_ELEM = 18  # not defined in exodus

MAX_STR_LENGTH = 32
MAX_VAR_NAME_LENGTH = 20
MAX_LINE_LENGTH = 80
MAX_ERR_LENGTH = 256

EX_VERBOSE = 1
EX_DEBUG = 2
EX_ABORT = 4

EX_INQ_FILE_TYPE = 1  # inquire EXODUS II file type
EX_INQ_API_VERS = 2  # inquire API version number
EX_INQ_DB_VERS = 3  # inquire database version number
EX_INQ_TITLE = 4  # inquire database title
EX_INQ_DIM = 5  # inquire number of dimensions
EX_INQ_NODES = 6  # inquire number of nodes
EX_INQ_ELEM = 7  # inquire number of elements
EX_INQ_ELEM_BLK = 8  # inquire number of element blocks
EX_INQ_NODE_SETS = 9  # inquire number of node sets
EX_INQ_NS_NODE_LEN = 10  # inquire length of node set node list
EX_INQ_SIDE_SETS = 11  # inquire number of side sets
EX_INQ_SS_NODE_LEN = 12  # inquire length of side set node list
EX_INQ_SS_ELEM_LEN = 13  # inquire length of side set element list
EX_INQ_QA = 14  # inquire number of QA records
EX_INQ_INFO = 15  # inquire number of info records
EX_INQ_TIME = 16  # inquire number of time steps in the database
EX_INQ_EB_PROP = 17  # inquire number of element block properties
EX_INQ_NS_PROP = 18  # inquire number of node set properties
EX_INQ_SS_PROP = 19  # inquire number of side set properties
EX_INQ_NS_DF_LEN = 20  # inquire length of node set distribution factor list
EX_INQ_SS_DF_LEN = 21  # inquire length of side set distribution factor list
EX_INQ_LIB_VERS = 22  # inquire API Lib vers number
EX_INQ_EM_PROP = 23  # inquire number of element map properties
EX_INQ_NM_PROP = 24  # inquire number of node map properties
EX_INQ_ELEM_MAP = 25  # inquire number of element maps
EX_INQ_NODE_MAP = 26  # inquire number of node maps
EX_INQ_EDGE = 27  # inquire number of edges
EX_INQ_EDGE_BLK = 28  # inquire number of edge blocks
EX_INQ_EDGE_SETS = 29  # inquire number of edge sets
EX_INQ_ES_LEN = 30  # inquire length of concat edge set edge list
EX_INQ_ES_DF_LEN = 31  # inquire length of concat edge set dist factor list
EX_INQ_EDGE_PROP = 32  # inquire number of properties stored per edge block
EX_INQ_ES_PROP = 33  # inquire number of properties stored per edge set
EX_INQ_FACE = 34  # inquire number of faces
EX_INQ_FACE_BLK = 35  # inquire number of face blocks
EX_INQ_FACE_SETS = 36  # inquire number of face sets
EX_INQ_FS_LEN = 37  # inquire length of concat face set face list
EX_INQ_FS_DF_LEN = 38  # inquire length of concat face set dist factor list
EX_INQ_FACE_PROP = 39  # inquire number of properties stored per face block
EX_INQ_FS_PROP = 40  # inquire number of properties stored per face set
EX_INQ_ES = 41  # inquire number of element sets
EX_INQ_ELS_LEN = 42  # inquire length of concat element set element list
EX_INQ_ELS_DF_LEN = 43  # inquire length of concat element set dist factor list
EX_INQ_ELS_PROP = 44  # inquire number of properties stored per elem set
EX_INQ_EDGE_MAP = 45  # inquire number of edge maps
EX_INQ_FACE_MAP = 46  # inquire number of face maps
EX_INQ_COORD_FRAMES = 47  # inquire number of coordinate frames
