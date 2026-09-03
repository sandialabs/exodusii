# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Canonical Exodus II NetCDF dimension, variable, and attribute names.

The original Exodus C API exposes a large collection of preprocessor constants.
For the modern Python implementation, this module keeps the same on-disk names
but groups them into a smaller, typed namespace.

The compatibility layer can still expose legacy names such as ``VAR_WHOLE_TIME``
and ``DIM_NUM_NODES``.
"""

from enum import StrEnum

from exodusii.core.entities import Entity
from exodusii.core.entities import entity
from exodusii.core.errors import ExodusInvalidEntityError


class AttributeName(StrEnum):
    """Common Exodus global and variable attribute names."""

    TITLE = "title"
    API_VERSION = "api_version"
    VERSION = "version"
    FILE_SIZE = "file_size"
    FLOATING_POINT_WORD_SIZE = "floating_point_word_size"
    FLOATING_POINT_WORD_SIZE_LEGACY = "floating point word size"
    ELEMENT_TYPE = "elem_type"
    PROPERTY_NAME = "name"


class DimensionName(StrEnum):
    """Common Exodus dimension names."""

    TIME = "time_step"

    STRING_LENGTH = "len_string"
    NAME_LENGTH = "len_name"
    LINE_LENGTH = "len_line"
    FOUR = "four"

    NUM_DIMENSIONS = "num_dim"
    NUM_NODES = "num_nodes"
    NUM_EDGES = "num_edge"
    NUM_FACES = "num_face"
    NUM_ELEMENTS = "num_elem"

    NUM_ELEMENT_BLOCKS = "num_el_blk"
    NUM_EDGE_BLOCKS = "num_ed_blk"
    NUM_FACE_BLOCKS = "num_fa_blk"

    NUM_NODE_SETS = "num_node_sets"
    NUM_SIDE_SETS = "num_side_sets"
    NUM_EDGE_SETS = "num_edge_sets"
    NUM_FACE_SETS = "num_face_sets"
    NUM_ELEMENT_SETS = "num_elem_sets"

    NUM_GLOBAL_VARIABLES = "num_glo_var"
    NUM_NODE_VARIABLES = "num_nod_var"
    NUM_ELEMENT_VARIABLES = "num_elem_var"
    NUM_EDGE_VARIABLES = "num_edge_var"
    NUM_FACE_VARIABLES = "num_face_var"

    NUM_NODE_SET_VARIABLES = "num_nset_var"
    NUM_SIDE_SET_VARIABLES = "num_sset_var"
    NUM_EDGE_SET_VARIABLES = "num_eset_var"
    NUM_FACE_SET_VARIABLES = "num_fset_var"
    NUM_ELEMENT_SET_VARIABLES = "num_elset_var"

    NUM_INFO_RECORDS = "num_info"
    NUM_QA_RECORDS = "num_qa_rec"

    NUM_NODE_MAPS = "num_node_maps"
    NUM_ELEMENT_MAPS = "num_elem_maps"
    NUM_EDGE_MAPS = "num_edge_maps"
    NUM_FACE_MAPS = "num_face_maps"

    NUM_NODES_GLOBAL = "num_nodes_global"
    NUM_ELEMENTS_GLOBAL = "num_elems_global"
    NUM_NODE_SETS_GLOBAL = "num_ns_global"
    NUM_SIDE_SETS_GLOBAL = "num_ss_global"
    NUM_ELEMENT_BLOCKS_GLOBAL = "num_el_blk_global"


class VariableName(StrEnum):
    """Common Exodus variable names."""

    TIME = "time_whole"

    COORDINATES = "coord"
    COORD_X = "coordx"
    COORD_Y = "coordy"
    COORD_Z = "coordz"
    COORDINATE_NAMES = "coor_names"

    INFO_RECORDS = "info_records"
    QA_RECORDS = "qa_records"

    ELEMENT_BLOCK_IDS = "eb_prop1"
    ELEMENT_BLOCK_STATUS = "eb_status"
    ELEMENT_BLOCK_NAMES = "eb_names"

    EDGE_BLOCK_IDS = "ed_prop1"
    EDGE_BLOCK_STATUS = "ed_status"
    EDGE_BLOCK_NAMES = "ed_names"

    FACE_BLOCK_IDS = "fa_prop1"
    FACE_BLOCK_STATUS = "fa_status"
    FACE_BLOCK_NAMES = "fa_names"

    NODE_SET_IDS = "ns_prop1"
    NODE_SET_STATUS = "ns_status"
    NODE_SET_NAMES = "ns_names"

    SIDE_SET_IDS = "ss_prop1"
    SIDE_SET_STATUS = "ss_status"
    SIDE_SET_NAMES = "ss_names"

    EDGE_SET_IDS = "es_prop1"
    EDGE_SET_STATUS = "es_status"
    EDGE_SET_NAMES = "es_names"

    FACE_SET_IDS = "fs_prop1"
    FACE_SET_STATUS = "fs_status"
    FACE_SET_NAMES = "fs_names"

    ELEMENT_SET_IDS = "els_prop1"
    ELEMENT_SET_STATUS = "els_status"
    ELEMENT_SET_NAMES = "els_names"

    GLOBAL_VARIABLE_NAMES = "name_glo_var"
    GLOBAL_VARIABLE_VALUES = "vals_glo_var"

    NODE_VARIABLE_NAMES = "name_nod_var"
    ELEMENT_VARIABLE_NAMES = "name_elem_var"
    EDGE_VARIABLE_NAMES = "name_edge_var"
    FACE_VARIABLE_NAMES = "name_face_var"

    NODE_SET_VARIABLE_NAMES = "name_nset_var"
    SIDE_SET_VARIABLE_NAMES = "name_sset_var"
    EDGE_SET_VARIABLE_NAMES = "name_eset_var"
    FACE_SET_VARIABLE_NAMES = "name_fset_var"
    ELEMENT_SET_VARIABLE_NAMES = "name_elset_var"

    ELEMENT_VARIABLE_TRUTH_TABLE = "elem_var_tab"
    EDGE_VARIABLE_TRUTH_TABLE = "edge_var_tab"
    FACE_VARIABLE_TRUTH_TABLE = "face_var_tab"
    NODE_SET_VARIABLE_TRUTH_TABLE = "nset_var_tab"
    SIDE_SET_VARIABLE_TRUTH_TABLE = "sset_var_tab"
    EDGE_SET_VARIABLE_TRUTH_TABLE = "eset_var_tab"
    FACE_SET_VARIABLE_TRUTH_TABLE = "fset_var_tab"
    ELEMENT_SET_VARIABLE_TRUTH_TABLE = "elset_var_tab"

    NODE_ID_MAP = "node_num_map"
    ELEMENT_ID_MAP = "elem_num_map"
    EDGE_ID_MAP = "edge_num_map"
    FACE_ID_MAP = "face_num_map"

    NODE_SET_IDS_GLOBAL = "ns_ids_global"
    SIDE_SET_IDS_GLOBAL = "ss_ids_global"
    ELEMENT_BLOCK_IDS_GLOBAL = "el_blk_ids_global"
    NODE_SET_NODE_COUNT_GLOBAL = "ns_node_cnt_global"
    SIDE_SET_SIDE_COUNT_GLOBAL = "ss_side_cnt_global"
    NODE_SET_DF_COUNT_GLOBAL = "ns_df_cnt_global"
    SIDE_SET_DF_COUNT_GLOBAL = "ss_df_cnt_global"
    ELEMENT_BLOCK_COUNT_GLOBAL = "el_blk_cnt_global"


class ExodusNames:
    """Factory for Exodus II NetCDF names.

    This class intentionally contains no state. It exists to provide a compact,
    discoverable namespace for name generation.
    """

    DEFAULT_STRING_LENGTH = 32
    DEFAULT_LINE_LENGTH = 80

    ATTR_TITLE = AttributeName.TITLE.value
    ATTR_API_VERSION = AttributeName.API_VERSION.value
    ATTR_VERSION = AttributeName.VERSION.value
    ATTR_FILE_SIZE = AttributeName.FILE_SIZE.value
    ATTR_FLOATING_POINT_WORD_SIZE = AttributeName.FLOATING_POINT_WORD_SIZE.value
    ATTR_ELEMENT_TYPE = AttributeName.ELEMENT_TYPE.value
    ATTR_PROPERTY_NAME = AttributeName.PROPERTY_NAME.value

    DIM_TIME = DimensionName.TIME.value
    DIM_STRING_LENGTH = DimensionName.STRING_LENGTH.value
    DIM_NAME_LENGTH = DimensionName.NAME_LENGTH.value
    DIM_LINE_LENGTH = DimensionName.LINE_LENGTH.value
    DIM_FOUR = DimensionName.FOUR.value
    DIM_NUM_DIMENSIONS = DimensionName.NUM_DIMENSIONS.value
    DIM_NUM_NODES = DimensionName.NUM_NODES.value
    DIM_NUM_ELEMENTS = DimensionName.NUM_ELEMENTS.value
    DIM_NUM_ELEMENT_BLOCKS = DimensionName.NUM_ELEMENT_BLOCKS.value
    DIM_NUM_NODE_SETS = DimensionName.NUM_NODE_SETS.value
    DIM_NUM_SIDE_SETS = DimensionName.NUM_SIDE_SETS.value

    VAR_TIME = VariableName.TIME.value
    VAR_COORD_X = VariableName.COORD_X.value
    VAR_COORD_Y = VariableName.COORD_Y.value
    VAR_COORD_Z = VariableName.COORD_Z.value
    VAR_COORDINATE_NAMES = VariableName.COORDINATE_NAMES.value
    VAR_GLOBAL_VARIABLE_NAMES = VariableName.GLOBAL_VARIABLE_NAMES.value
    VAR_GLOBAL_VARIABLE_VALUES = VariableName.GLOBAL_VARIABLE_VALUES.value
    VAR_NODE_VARIABLE_NAMES = VariableName.NODE_VARIABLE_NAMES.value
    VAR_ELEMENT_VARIABLE_NAMES = VariableName.ELEMENT_VARIABLE_NAMES.value

    @staticmethod
    def dimension_count(entity_value: Entity | str) -> str:
        """Return the top-level count dimension for an entity."""

        ent = entity(entity_value)
        mapping = {
            Entity.NODE: DimensionName.NUM_NODES.value,
            Entity.EDGE: DimensionName.NUM_EDGES.value,
            Entity.FACE: DimensionName.NUM_FACES.value,
            Entity.ELEMENT: DimensionName.NUM_ELEMENTS.value,
            Entity.ELEMENT_BLOCK: DimensionName.NUM_ELEMENT_BLOCKS.value,
            Entity.EDGE_BLOCK: DimensionName.NUM_EDGE_BLOCKS.value,
            Entity.FACE_BLOCK: DimensionName.NUM_FACE_BLOCKS.value,
            Entity.NODE_SET: DimensionName.NUM_NODE_SETS.value,
            Entity.SIDE_SET: DimensionName.NUM_SIDE_SETS.value,
            Entity.EDGE_SET: DimensionName.NUM_EDGE_SETS.value,
            Entity.FACE_SET: DimensionName.NUM_FACE_SETS.value,
            Entity.ELEMENT_SET: DimensionName.NUM_ELEMENT_SETS.value,
            Entity.NODE_MAP: DimensionName.NUM_NODE_MAPS.value,
            Entity.ELEMENT_MAP: DimensionName.NUM_ELEMENT_MAPS.value,
            Entity.EDGE_MAP: DimensionName.NUM_EDGE_MAPS.value,
            Entity.FACE_MAP: DimensionName.NUM_FACE_MAPS.value,
        }
        try:
            return mapping[ent]
        except KeyError as exc:
            raise ExodusInvalidEntityError(
                f"{ent.value!r} does not have a count dimension"
            ) from exc

    @staticmethod
    def ids(entity_value: Entity | str) -> str:
        """Return the ID-property variable for a block, set, or object map entity."""

        ent = entity(entity_value)
        mapping = {
            Entity.ELEMENT_BLOCK: VariableName.ELEMENT_BLOCK_IDS.value,
            Entity.EDGE_BLOCK: VariableName.EDGE_BLOCK_IDS.value,
            Entity.FACE_BLOCK: VariableName.FACE_BLOCK_IDS.value,
            Entity.NODE_SET: VariableName.NODE_SET_IDS.value,
            Entity.SIDE_SET: VariableName.SIDE_SET_IDS.value,
            Entity.EDGE_SET: VariableName.EDGE_SET_IDS.value,
            Entity.FACE_SET: VariableName.FACE_SET_IDS.value,
            Entity.ELEMENT_SET: VariableName.ELEMENT_SET_IDS.value,
            Entity.NODE: VariableName.NODE_ID_MAP.value,
            Entity.ELEMENT: VariableName.ELEMENT_ID_MAP.value,
            Entity.EDGE: VariableName.EDGE_ID_MAP.value,
            Entity.FACE: VariableName.FACE_ID_MAP.value,
        }
        try:
            return mapping[ent]
        except KeyError as exc:
            raise ExodusInvalidEntityError(f"{ent.value!r} does not have an ID variable") from exc

    @staticmethod
    def status(entity_value: Entity | str) -> str:
        """Return the status variable for a block or set entity."""

        ent = entity(entity_value)
        mapping = {
            Entity.ELEMENT_BLOCK: VariableName.ELEMENT_BLOCK_STATUS.value,
            Entity.EDGE_BLOCK: VariableName.EDGE_BLOCK_STATUS.value,
            Entity.FACE_BLOCK: VariableName.FACE_BLOCK_STATUS.value,
            Entity.NODE_SET: VariableName.NODE_SET_STATUS.value,
            Entity.SIDE_SET: VariableName.SIDE_SET_STATUS.value,
            Entity.EDGE_SET: VariableName.EDGE_SET_STATUS.value,
            Entity.FACE_SET: VariableName.FACE_SET_STATUS.value,
            Entity.ELEMENT_SET: VariableName.ELEMENT_SET_STATUS.value,
        }
        try:
            return mapping[ent]
        except KeyError as exc:
            raise ExodusInvalidEntityError(
                f"{ent.value!r} does not have a status variable"
            ) from exc

    @staticmethod
    def names(entity_value: Entity | str) -> str:
        """Return the names variable for a block, set, coordinate, or variable location."""

        ent = entity(entity_value)
        mapping = {
            Entity.ELEMENT_BLOCK: VariableName.ELEMENT_BLOCK_NAMES.value,
            Entity.EDGE_BLOCK: VariableName.EDGE_BLOCK_NAMES.value,
            Entity.FACE_BLOCK: VariableName.FACE_BLOCK_NAMES.value,
            Entity.NODE_SET: VariableName.NODE_SET_NAMES.value,
            Entity.SIDE_SET: VariableName.SIDE_SET_NAMES.value,
            Entity.EDGE_SET: VariableName.EDGE_SET_NAMES.value,
            Entity.FACE_SET: VariableName.FACE_SET_NAMES.value,
            Entity.ELEMENT_SET: VariableName.ELEMENT_SET_NAMES.value,
            Entity.GLOBAL: VariableName.GLOBAL_VARIABLE_NAMES.value,
            Entity.NODE: VariableName.NODE_VARIABLE_NAMES.value,
            Entity.ELEMENT: VariableName.ELEMENT_VARIABLE_NAMES.value,
            Entity.EDGE: VariableName.EDGE_VARIABLE_NAMES.value,
            Entity.FACE: VariableName.FACE_VARIABLE_NAMES.value,
        }
        try:
            return mapping[ent]
        except KeyError as exc:
            raise ExodusInvalidEntityError(f"{ent.value!r} does not have a names variable") from exc

    @staticmethod
    def variable_count(entity_value: Entity | str) -> str:
        """Return the variable-count dimension for a variable location."""

        ent = entity(entity_value)
        mapping = {
            Entity.GLOBAL: DimensionName.NUM_GLOBAL_VARIABLES.value,
            Entity.NODE: DimensionName.NUM_NODE_VARIABLES.value,
            Entity.ELEMENT: DimensionName.NUM_ELEMENT_VARIABLES.value,
            Entity.EDGE: DimensionName.NUM_EDGE_VARIABLES.value,
            Entity.FACE: DimensionName.NUM_FACE_VARIABLES.value,
            Entity.NODE_SET: DimensionName.NUM_NODE_SET_VARIABLES.value,
            Entity.SIDE_SET: DimensionName.NUM_SIDE_SET_VARIABLES.value,
            Entity.EDGE_SET: DimensionName.NUM_EDGE_SET_VARIABLES.value,
            Entity.FACE_SET: DimensionName.NUM_FACE_SET_VARIABLES.value,
            Entity.ELEMENT_SET: DimensionName.NUM_ELEMENT_SET_VARIABLES.value,
        }
        try:
            return mapping[ent]
        except KeyError as exc:
            raise ExodusInvalidEntityError(
                f"{ent.value!r} does not have a variable-count dimension"
            ) from exc

    @staticmethod
    def variable_truth_table(entity_value: Entity | str) -> str:
        """Return the variable truth-table name for block or set variable locations."""

        ent = entity(entity_value)
        mapping = {
            Entity.ELEMENT: VariableName.ELEMENT_VARIABLE_TRUTH_TABLE.value,
            Entity.EDGE: VariableName.EDGE_VARIABLE_TRUTH_TABLE.value,
            Entity.FACE: VariableName.FACE_VARIABLE_TRUTH_TABLE.value,
            Entity.NODE_SET: VariableName.NODE_SET_VARIABLE_TRUTH_TABLE.value,
            Entity.SIDE_SET: VariableName.SIDE_SET_VARIABLE_TRUTH_TABLE.value,
            Entity.EDGE_SET: VariableName.EDGE_SET_VARIABLE_TRUTH_TABLE.value,
            Entity.FACE_SET: VariableName.FACE_SET_VARIABLE_TRUTH_TABLE.value,
            Entity.ELEMENT_SET: VariableName.ELEMENT_SET_VARIABLE_TRUTH_TABLE.value,
        }
        try:
            return mapping[ent]
        except KeyError as exc:
            raise ExodusInvalidEntityError(
                f"{ent.value!r} does not have a variable truth table"
            ) from exc

    @staticmethod
    def coordinate(axis: int | str) -> str:
        """Return the coordinate variable name for axis 0/1/2 or x/y/z."""

        if isinstance(axis, str):
            key = axis.strip().lower()
            mapping = {
                "0": VariableName.COORD_X.value,
                "1": VariableName.COORD_Y.value,
                "2": VariableName.COORD_Z.value,
                "x": VariableName.COORD_X.value,
                "y": VariableName.COORD_Y.value,
                "z": VariableName.COORD_Z.value,
            }
        else:
            key = str(axis)
            mapping = {
                "0": VariableName.COORD_X.value,
                "1": VariableName.COORD_Y.value,
                "2": VariableName.COORD_Z.value,
            }

        try:
            return mapping[key]
        except KeyError as exc:
            raise ValueError(
                f"Expected coordinate axis 0, 1, 2, 'x', 'y', or 'z'; got {axis!r}"
            ) from exc

    @staticmethod
    def block_count(block_index: int) -> str:
        """Return the number-of-elements dimension for an element block."""

        _validate_positive_index(block_index)
        return f"num_el_in_blk{block_index}"

    @staticmethod
    def nodes_per_element(block_index: int) -> str:
        """Return the nodes-per-element dimension for an element block."""

        _validate_positive_index(block_index)
        return f"num_nod_per_el{block_index}"

    @staticmethod
    def edges_per_element(block_index: int) -> str:
        """Return the edges-per-element dimension for an element block."""

        _validate_positive_index(block_index)
        return f"num_edg_per_el{block_index}"

    @staticmethod
    def faces_per_element(block_index: int) -> str:
        """Return the faces-per-element dimension for an element block."""

        _validate_positive_index(block_index)
        return f"num_fac_per_el{block_index}"

    @staticmethod
    def attributes_per_element(block_index: int) -> str:
        """Return the attributes-per-element dimension for an element block."""

        _validate_positive_index(block_index)
        return f"num_att_in_blk{block_index}"

    @staticmethod
    def element_connectivity(block_index: int) -> str:
        """Return the nodal connectivity variable for an element block."""

        _validate_positive_index(block_index)
        return f"connect{block_index}"

    @staticmethod
    def element_edge_connectivity(block_index: int) -> str:
        """Return the edge connectivity variable for an element block."""

        _validate_positive_index(block_index)
        return f"edgconn{block_index}"

    @staticmethod
    def element_face_connectivity(block_index: int) -> str:
        """Return the face connectivity variable for an element block."""

        _validate_positive_index(block_index)
        return f"facconn{block_index}"

    @staticmethod
    def edge_block_count(block_index: int) -> str:
        """Return the number-of-edges dimension for an edge block."""

        _validate_positive_index(block_index)
        return f"num_ed_in_blk{block_index}"

    @staticmethod
    def nodes_per_edge(block_index: int) -> str:
        """Return the nodes-per-edge dimension for an edge block."""

        _validate_positive_index(block_index)
        return f"num_nod_per_ed{block_index}"

    @staticmethod
    def edge_connectivity(block_index: int) -> str:
        """Return the nodal connectivity variable for an edge block."""

        _validate_positive_index(block_index)
        return f"ebconn{block_index}"

    @staticmethod
    def face_block_count(block_index: int) -> str:
        """Return the number-of-faces dimension for a face block."""

        _validate_positive_index(block_index)
        return f"num_fa_in_blk{block_index}"

    @staticmethod
    def nodes_per_face(block_index: int) -> str:
        """Return the nodes-per-face dimension for a face block."""

        _validate_positive_index(block_index)
        return f"num_nod_per_fa{block_index}"

    @staticmethod
    def face_connectivity(block_index: int) -> str:
        """Return the nodal connectivity variable for a face block."""

        _validate_positive_index(block_index)
        return f"fbconn{block_index}"

    @staticmethod
    def node_set_count(set_index: int) -> str:
        """Return the node-count dimension for a node set."""

        _validate_positive_index(set_index)
        return f"num_nod_ns{set_index}"

    @staticmethod
    def node_set_distribution_factor_count(set_index: int) -> str:
        """Return the distribution-factor dimension for a node set."""

        _validate_positive_index(set_index)
        return f"num_df_ns{set_index}"

    @staticmethod
    def node_set_nodes(set_index: int) -> str:
        """Return the node-list variable for a node set."""

        _validate_positive_index(set_index)
        return f"node_ns{set_index}"

    @staticmethod
    def node_set_distribution_factors(set_index: int) -> str:
        """Return the distribution-factor variable for a node set."""

        _validate_positive_index(set_index)
        return f"dist_fact_ns{set_index}"

    @staticmethod
    def side_set_count(set_index: int) -> str:
        """Return the side-count dimension for a side set."""

        _validate_positive_index(set_index)
        return f"num_side_ss{set_index}"

    @staticmethod
    def side_set_distribution_factor_count(set_index: int) -> str:
        """Return the distribution-factor dimension for a side set."""

        _validate_positive_index(set_index)
        return f"num_df_ss{set_index}"

    @staticmethod
    def side_set_elements(set_index: int) -> str:
        """Return the element-list variable for a side set."""

        _validate_positive_index(set_index)
        return f"elem_ss{set_index}"

    @staticmethod
    def side_set_sides(set_index: int) -> str:
        """Return the side-list variable for a side set."""

        _validate_positive_index(set_index)
        return f"side_ss{set_index}"

    @staticmethod
    def side_set_distribution_factors(set_index: int) -> str:
        """Return the distribution-factor variable for a side set."""

        _validate_positive_index(set_index)
        return f"dist_fact_ss{set_index}"

    @staticmethod
    def node_variable(variable_index: int) -> str:
        """Return a nodal result variable name."""

        _validate_positive_index(variable_index)
        return f"vals_nod_var{variable_index}"

    @staticmethod
    def element_variable(variable_index: int, block_index: int) -> str:
        """Return an element result variable name."""

        _validate_positive_index(variable_index, name="variable_index")
        _validate_positive_index(block_index, name="block_index")
        return f"vals_elem_var{variable_index}eb{block_index}"

    @staticmethod
    def edge_variable(variable_index: int, block_index: int) -> str:
        """Return an edge result variable name."""

        _validate_positive_index(variable_index, name="variable_index")
        _validate_positive_index(block_index, name="block_index")
        return f"vals_edge_var{variable_index}eb{block_index}"

    @staticmethod
    def face_variable(variable_index: int, block_index: int) -> str:
        """Return a face result variable name."""

        _validate_positive_index(variable_index, name="variable_index")
        _validate_positive_index(block_index, name="block_index")
        return f"vals_face_var{variable_index}fb{block_index}"

    @staticmethod
    def node_set_variable(variable_index: int, set_index: int) -> str:
        """Return a node-set result variable name."""

        _validate_positive_index(variable_index, name="variable_index")
        _validate_positive_index(set_index, name="set_index")
        return f"vals_nset_var{variable_index}ns{set_index}"

    @staticmethod
    def side_set_variable(variable_index: int, set_index: int) -> str:
        """Return a side-set result variable name."""

        _validate_positive_index(variable_index, name="variable_index")
        _validate_positive_index(set_index, name="set_index")
        return f"vals_sset_var{variable_index}ss{set_index}"


def _validate_positive_index(index: int, *, name: str = "index") -> None:
    if not isinstance(index, int):
        raise TypeError(f"{name} must be an int")
    if index < 1:
        raise ValueError(f"{name} must be 1-based and positive")


EX = ExodusNames

__all__ = ["EX", "AttributeName", "DimensionName", "ExodusNames", "VariableName"]
