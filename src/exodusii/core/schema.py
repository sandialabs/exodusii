# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Central Exodus entity schema.

This module is the compatibility map between logical Exodus entities and their
NetCDF dimension/variable names.  It is intentionally data-oriented so serial,
parallel, writer, copy, and legacy adapter code can share the same knowledge
instead of re-encoding Exodus naming conventions in multiple places.
"""

from collections.abc import Callable
from dataclasses import dataclass

from exodusii.core.entities import Entity
from exodusii.core.entities import entity
from exodusii.core.names import DimensionName
from exodusii.core.names import ExodusNames
from exodusii.core.names import VariableName

NameFactory1 = Callable[[int], str]
NameFactory2 = Callable[[int, int], str]


@dataclass(frozen=True, slots=True)
class ObjectSpec:
    """Schema for a top-level Exodus object entity."""

    entity: Entity
    count_dimension: str
    id_map_variable: str | None = None


@dataclass(frozen=True, slots=True)
class BlockSpec:
    """Schema for an Exodus block entity."""

    entity: Entity
    object_entity: Entity
    count_dimension: str
    ids_variable: str
    status_variable: str
    names_variable: str
    object_count_dimension: NameFactory1
    nodes_per_object_dimension: NameFactory1
    connectivity_variable: NameFactory1
    property_variable: NameFactory1 | None = None
    attributes_dimension: NameFactory1 | None = None
    attributes_variable: NameFactory1 | None = None
    attribute_names_variable: NameFactory1 | None = None
    edges_per_object_dimension: NameFactory1 | None = None
    faces_per_object_dimension: NameFactory1 | None = None
    edge_connectivity_variable: NameFactory1 | None = None
    face_connectivity_variable: NameFactory1 | None = None


@dataclass(frozen=True, slots=True)
class SetSpec:
    """Schema for an Exodus set entity."""

    entity: Entity
    object_entity: Entity
    count_dimension: str
    ids_variable: str
    status_variable: str
    names_variable: str
    entry_count_dimension: NameFactory1
    dist_factor_count_dimension: NameFactory1
    entries_variable: NameFactory1
    dist_factors_variable: NameFactory1
    property_variable: NameFactory1 | None = None
    extra_entries_variable: NameFactory1 | None = None
    extra_entries_name: str | None = None


@dataclass(frozen=True, slots=True)
class VariableSpec:
    """Schema for Exodus result variables."""

    entity: Entity
    count_dimension: str
    names_variable: str
    values_variable: NameFactory2
    truth_table_variable: str | None = None
    location_entity: Entity | None = None


@dataclass(frozen=True, slots=True)
class MapSpec:
    """Schema for Exodus maps."""

    entity: Entity
    count_dimension: str
    map_variable: NameFactory1 | str
    names_variable: str | None = None
    property_variable: NameFactory1 | None = None


OBJECT_SPECS: dict[Entity, ObjectSpec] = {
    Entity.NODE: ObjectSpec(
        entity=Entity.NODE,
        count_dimension=DimensionName.NUM_NODES.value,
        id_map_variable=VariableName.NODE_ID_MAP.value,
    ),
    Entity.ELEMENT: ObjectSpec(
        entity=Entity.ELEMENT,
        count_dimension=DimensionName.NUM_ELEMENTS.value,
        id_map_variable=VariableName.ELEMENT_ID_MAP.value,
    ),
    Entity.EDGE: ObjectSpec(
        entity=Entity.EDGE,
        count_dimension=DimensionName.NUM_EDGES.value,
        id_map_variable=VariableName.EDGE_ID_MAP.value,
    ),
    Entity.FACE: ObjectSpec(
        entity=Entity.FACE,
        count_dimension=DimensionName.NUM_FACES.value,
        id_map_variable=VariableName.FACE_ID_MAP.value,
    ),
}


BLOCK_SPECS: dict[Entity, BlockSpec] = {
    Entity.ELEMENT_BLOCK: BlockSpec(
        entity=Entity.ELEMENT_BLOCK,
        object_entity=Entity.ELEMENT,
        count_dimension=DimensionName.NUM_ELEMENT_BLOCKS.value,
        ids_variable=VariableName.ELEMENT_BLOCK_IDS.value,
        status_variable=VariableName.ELEMENT_BLOCK_STATUS.value,
        names_variable=VariableName.ELEMENT_BLOCK_NAMES.value,
        object_count_dimension=ExodusNames.block_count,
        nodes_per_object_dimension=ExodusNames.nodes_per_element,
        connectivity_variable=ExodusNames.element_connectivity,
        property_variable=lambda index: f"eb_prop{index}",
        attributes_dimension=ExodusNames.attributes_per_element,
        attributes_variable=lambda index: f"attrib{index}",
        attribute_names_variable=lambda index: f"attrib_name{index}",
        edges_per_object_dimension=ExodusNames.edges_per_element,
        faces_per_object_dimension=ExodusNames.faces_per_element,
        edge_connectivity_variable=ExodusNames.element_edge_connectivity,
        face_connectivity_variable=ExodusNames.element_face_connectivity,
    ),
    Entity.EDGE_BLOCK: BlockSpec(
        entity=Entity.EDGE_BLOCK,
        object_entity=Entity.EDGE,
        count_dimension=DimensionName.NUM_EDGE_BLOCKS.value,
        ids_variable=VariableName.EDGE_BLOCK_IDS.value,
        status_variable=VariableName.EDGE_BLOCK_STATUS.value,
        names_variable=VariableName.EDGE_BLOCK_NAMES.value,
        object_count_dimension=ExodusNames.edge_block_count,
        nodes_per_object_dimension=ExodusNames.nodes_per_edge,
        connectivity_variable=ExodusNames.edge_connectivity,
        property_variable=lambda index: f"ed_prop{index}",
        attributes_dimension=lambda index: f"num_att_in_eblk{index}",
        attributes_variable=lambda index: f"eattrb{index}",
        attribute_names_variable=lambda index: f"eattrib_name{index}",
    ),
    Entity.FACE_BLOCK: BlockSpec(
        entity=Entity.FACE_BLOCK,
        object_entity=Entity.FACE,
        count_dimension=DimensionName.NUM_FACE_BLOCKS.value,
        ids_variable=VariableName.FACE_BLOCK_IDS.value,
        status_variable=VariableName.FACE_BLOCK_STATUS.value,
        names_variable=VariableName.FACE_BLOCK_NAMES.value,
        object_count_dimension=ExodusNames.face_block_count,
        nodes_per_object_dimension=ExodusNames.nodes_per_face,
        connectivity_variable=ExodusNames.face_connectivity,
        property_variable=lambda index: f"fa_prop{index}",
        attributes_dimension=lambda index: f"num_att_in_fblk{index}",
        attributes_variable=lambda index: f"fattrb{index}",
        attribute_names_variable=lambda index: f"fattrib_name{index}",
    ),
}


SET_SPECS: dict[Entity, SetSpec] = {
    Entity.NODE_SET: SetSpec(
        entity=Entity.NODE_SET,
        object_entity=Entity.NODE,
        count_dimension=DimensionName.NUM_NODE_SETS.value,
        ids_variable=VariableName.NODE_SET_IDS.value,
        status_variable=VariableName.NODE_SET_STATUS.value,
        names_variable=VariableName.NODE_SET_NAMES.value,
        entry_count_dimension=ExodusNames.node_set_count,
        property_variable=lambda index: f"ns_prop{index}",
        dist_factor_count_dimension=ExodusNames.node_set_distribution_factor_count,
        entries_variable=ExodusNames.node_set_nodes,
        dist_factors_variable=ExodusNames.node_set_distribution_factors,
    ),
    Entity.SIDE_SET: SetSpec(
        entity=Entity.SIDE_SET,
        object_entity=Entity.ELEMENT,
        count_dimension=DimensionName.NUM_SIDE_SETS.value,
        ids_variable=VariableName.SIDE_SET_IDS.value,
        status_variable=VariableName.SIDE_SET_STATUS.value,
        names_variable=VariableName.SIDE_SET_NAMES.value,
        entry_count_dimension=ExodusNames.side_set_count,
        dist_factor_count_dimension=ExodusNames.side_set_distribution_factor_count,
        entries_variable=ExodusNames.side_set_elements,
        property_variable=lambda index: f"ss_prop{index}",
        dist_factors_variable=ExodusNames.side_set_distribution_factors,
        extra_entries_variable=ExodusNames.side_set_sides,
        extra_entries_name="sides",
    ),
    Entity.EDGE_SET: SetSpec(
        entity=Entity.EDGE_SET,
        object_entity=Entity.EDGE,
        count_dimension=DimensionName.NUM_EDGE_SETS.value,
        ids_variable=VariableName.EDGE_SET_IDS.value,
        status_variable=VariableName.EDGE_SET_STATUS.value,
        names_variable=VariableName.EDGE_SET_NAMES.value,
        entry_count_dimension=lambda index: f"num_edge_es{index}",
        dist_factor_count_dimension=lambda index: f"num_df_es{index}",
        entries_variable=lambda index: f"edge_es{index}",
        property_variable=lambda index: f"es_prop{index}",
        dist_factors_variable=lambda index: f"dist_fact_es{index}",
        extra_entries_variable=lambda index: f"ornt_es{index}",
        extra_entries_name="orientations",
    ),
    Entity.FACE_SET: SetSpec(
        entity=Entity.FACE_SET,
        object_entity=Entity.FACE,
        count_dimension=DimensionName.NUM_FACE_SETS.value,
        ids_variable=VariableName.FACE_SET_IDS.value,
        status_variable=VariableName.FACE_SET_STATUS.value,
        names_variable=VariableName.FACE_SET_NAMES.value,
        entry_count_dimension=lambda index: f"num_face_fs{index}",
        dist_factor_count_dimension=lambda index: f"num_df_fs{index}",
        entries_variable=lambda index: f"face_fs{index}",
        property_variable=lambda index: f"fs_prop{index}",
        dist_factors_variable=lambda index: f"dist_fact_fs{index}",
        extra_entries_variable=lambda index: f"ornt_fs{index}",
        extra_entries_name="orientations",
    ),
    Entity.ELEMENT_SET: SetSpec(
        entity=Entity.ELEMENT_SET,
        object_entity=Entity.ELEMENT,
        count_dimension=DimensionName.NUM_ELEMENT_SETS.value,
        ids_variable=VariableName.ELEMENT_SET_IDS.value,
        status_variable=VariableName.ELEMENT_SET_STATUS.value,
        names_variable=VariableName.ELEMENT_SET_NAMES.value,
        entry_count_dimension=lambda index: f"num_ele_els{index}",
        property_variable=lambda index: f"els_prop{index}",
        dist_factor_count_dimension=lambda index: f"num_df_els{index}",
        entries_variable=lambda index: f"elem_els{index}",
        dist_factors_variable=lambda index: f"dist_fact_els{index}",
    ),
}


def _global_values_variable(_variable_index: int, _location_index: int) -> str:
    return VariableName.GLOBAL_VARIABLE_VALUES.value


def _node_values_variable(variable_index: int, _location_index: int) -> str:
    return ExodusNames.node_variable(variable_index)


VARIABLE_SPECS: dict[Entity, VariableSpec] = {
    Entity.GLOBAL: VariableSpec(
        entity=Entity.GLOBAL,
        count_dimension=DimensionName.NUM_GLOBAL_VARIABLES.value,
        names_variable=VariableName.GLOBAL_VARIABLE_NAMES.value,
        values_variable=_global_values_variable,
    ),
    Entity.NODE: VariableSpec(
        entity=Entity.NODE,
        count_dimension=DimensionName.NUM_NODE_VARIABLES.value,
        names_variable=VariableName.NODE_VARIABLE_NAMES.value,
        values_variable=_node_values_variable,
        location_entity=Entity.NODE,
    ),
    Entity.ELEMENT: VariableSpec(
        entity=Entity.ELEMENT,
        count_dimension=DimensionName.NUM_ELEMENT_VARIABLES.value,
        names_variable=VariableName.ELEMENT_VARIABLE_NAMES.value,
        values_variable=ExodusNames.element_variable,
        truth_table_variable=VariableName.ELEMENT_VARIABLE_TRUTH_TABLE.value,
        location_entity=Entity.ELEMENT_BLOCK,
    ),
    Entity.EDGE: VariableSpec(
        entity=Entity.EDGE,
        count_dimension=DimensionName.NUM_EDGE_VARIABLES.value,
        names_variable=VariableName.EDGE_VARIABLE_NAMES.value,
        values_variable=ExodusNames.edge_variable,
        truth_table_variable=VariableName.EDGE_VARIABLE_TRUTH_TABLE.value,
        location_entity=Entity.EDGE_BLOCK,
    ),
    Entity.FACE: VariableSpec(
        entity=Entity.FACE,
        count_dimension=DimensionName.NUM_FACE_VARIABLES.value,
        names_variable=VariableName.FACE_VARIABLE_NAMES.value,
        values_variable=ExodusNames.face_variable,
        truth_table_variable=VariableName.FACE_VARIABLE_TRUTH_TABLE.value,
        location_entity=Entity.FACE_BLOCK,
    ),
    Entity.NODE_SET: VariableSpec(
        entity=Entity.NODE_SET,
        count_dimension=DimensionName.NUM_NODE_SET_VARIABLES.value,
        names_variable=VariableName.NODE_SET_VARIABLE_NAMES.value,
        values_variable=ExodusNames.node_set_variable,
        truth_table_variable=VariableName.NODE_SET_VARIABLE_TRUTH_TABLE.value,
        location_entity=Entity.NODE_SET,
    ),
    Entity.SIDE_SET: VariableSpec(
        entity=Entity.SIDE_SET,
        count_dimension=DimensionName.NUM_SIDE_SET_VARIABLES.value,
        names_variable=VariableName.SIDE_SET_VARIABLE_NAMES.value,
        values_variable=ExodusNames.side_set_variable,
        truth_table_variable=VariableName.SIDE_SET_VARIABLE_TRUTH_TABLE.value,
        location_entity=Entity.SIDE_SET,
    ),
    Entity.EDGE_SET: VariableSpec(
        entity=Entity.EDGE_SET,
        count_dimension=DimensionName.NUM_EDGE_SET_VARIABLES.value,
        names_variable=VariableName.EDGE_SET_VARIABLE_NAMES.value,
        values_variable=lambda variable_index, set_index: (
            f"vals_eset_var{variable_index}es{set_index}"
        ),
        truth_table_variable=VariableName.EDGE_SET_VARIABLE_TRUTH_TABLE.value,
        location_entity=Entity.EDGE_SET,
    ),
    Entity.FACE_SET: VariableSpec(
        entity=Entity.FACE_SET,
        count_dimension=DimensionName.NUM_FACE_SET_VARIABLES.value,
        names_variable=VariableName.FACE_SET_VARIABLE_NAMES.value,
        values_variable=lambda variable_index, set_index: (
            f"vals_fset_var{variable_index}fs{set_index}"
        ),
        truth_table_variable=VariableName.FACE_SET_VARIABLE_TRUTH_TABLE.value,
        location_entity=Entity.FACE_SET,
    ),
    Entity.ELEMENT_SET: VariableSpec(
        entity=Entity.ELEMENT_SET,
        count_dimension=DimensionName.NUM_ELEMENT_SET_VARIABLES.value,
        names_variable=VariableName.ELEMENT_SET_VARIABLE_NAMES.value,
        values_variable=lambda variable_index, set_index: (
            f"vals_elset_var{variable_index}es{set_index}"
        ),
        truth_table_variable=VariableName.ELEMENT_SET_VARIABLE_TRUTH_TABLE.value,
        location_entity=Entity.ELEMENT_SET,
    ),
}


MAP_SPECS: dict[Entity, MapSpec] = {
    Entity.NODE_MAP: MapSpec(
        entity=Entity.NODE_MAP,
        count_dimension=DimensionName.NUM_NODE_MAPS.value,
        map_variable=lambda index: f"node_map{index}",
        names_variable="nmap_names",
        property_variable=lambda index: f"nm_prop{index}",
    ),
    Entity.ELEMENT_MAP: MapSpec(
        entity=Entity.ELEMENT_MAP,
        count_dimension=DimensionName.NUM_ELEMENT_MAPS.value,
        map_variable=lambda index: f"elem_map{index}",
        names_variable="emap_names",
        property_variable=lambda index: f"em_prop{index}",
    ),
    Entity.EDGE_MAP: MapSpec(
        entity=Entity.EDGE_MAP,
        count_dimension=DimensionName.NUM_EDGE_MAPS.value,
        map_variable=lambda index: f"edge_map{index}",
        names_variable="edmap_names",
        property_variable=lambda index: f"edm_prop{index}",
    ),
    Entity.FACE_MAP: MapSpec(
        entity=Entity.FACE_MAP,
        count_dimension=DimensionName.NUM_FACE_MAPS.value,
        map_variable=lambda index: f"face_map{index}",
        names_variable="famap_names",
        property_variable=lambda index: f"fam_prop{index}",
    ),
}


def object_spec(value: Entity | str) -> ObjectSpec:
    """Return object schema."""

    return OBJECT_SPECS[entity(value)]


def block_spec(value: Entity | str) -> BlockSpec:
    """Return block schema."""

    return BLOCK_SPECS[entity(value)]


def set_spec(value: Entity | str) -> SetSpec:
    """Return set schema."""

    return SET_SPECS[entity(value)]


def variable_spec(value: Entity | str) -> VariableSpec:
    """Return variable schema."""

    return VARIABLE_SPECS[entity(value)]


def map_spec(value: Entity | str) -> MapSpec:
    """Return map schema."""

    return MAP_SPECS[entity(value)]


def variable_value_name(
    value: Entity | str, variable_index: int, location_index: int | None = None
) -> str:
    """Return the NetCDF value-variable name for a result variable."""

    return variable_spec(value).values_variable(variable_index, location_index or 1)


def variable_entities() -> tuple[Entity, ...]:
    """Return all entities with result variables."""

    return tuple(VARIABLE_SPECS)


def block_entities() -> tuple[Entity, ...]:
    """Return all block entities."""

    return tuple(BLOCK_SPECS)


def set_entities() -> tuple[Entity, ...]:
    """Return all set entities."""

    return tuple(SET_SPECS)


__all__ = [
    "BLOCK_SPECS",
    "MAP_SPECS",
    "OBJECT_SPECS",
    "SET_SPECS",
    "VARIABLE_SPECS",
    "BlockSpec",
    "MapSpec",
    "ObjectSpec",
    "SetSpec",
    "VariableSpec",
    "block_entities",
    "block_spec",
    "map_spec",
    "object_spec",
    "set_entities",
    "set_spec",
    "variable_entities",
    "variable_spec",
    "variable_value_name",
]
