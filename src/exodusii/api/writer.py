# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Modern Exodus database writer."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from exodusii.core.entities import Entity
from exodusii.core.errors import ExodusLookupError
from exodusii.core.errors import ExodusWriteError
from exodusii.core.names import AttributeName
from exodusii.core.names import DimensionName
from exodusii.core.names import ExodusNames
from exodusii.core.names import VariableName
from exodusii.core.schema import block_spec
from exodusii.core.schema import set_spec
from exodusii.core.schema import variable_spec
from exodusii.core.schema import variable_value_name
from exodusii.core.strings import encode_fixed_width
from exodusii.io.netcdf4_backend import NetCDF4Backend


class ExodusWriter:
    """Modern Exodus database writer.

    This class provides a compact writer-oriented API. Legacy ``put_*`` methods
    will be implemented separately in the compatibility adapter.
    """

    def __init__(self, backend: NetCDF4Backend) -> None:
        self._backend = backend
        self._initialized = False
        self._time_step = 0

        self._element_block_counter = 0
        self._node_set_counter = 0
        self._side_set_counter = 0

        self._element_block_indices: dict[int, int] = {}
        self._node_set_indices: dict[int, int] = {}
        self._side_set_indices: dict[int, int] = {}

        self._edge_block_counter = 0
        self._face_block_counter = 0
        self._edge_set_counter = 0
        self._face_set_counter = 0
        self._element_set_counter = 0

        self._edge_block_indices: dict[int, int] = {}
        self._face_block_indices: dict[int, int] = {}
        self._edge_set_indices: dict[int, int] = {}
        self._face_set_indices: dict[int, int] = {}
        self._element_set_indices: dict[int, int] = {}

    @classmethod
    def create(cls, path: str | Path) -> "ExodusWriter":
        """Create a new Exodus database."""

        return cls(NetCDF4Backend(path, mode="w"))

    @property
    def path(self) -> Path:
        """Database path."""

        return self._backend.path

    @property
    def backend(self) -> NetCDF4Backend:
        """Underlying backend."""

        return self._backend

    def close(self) -> None:
        """Close the database."""

        self._backend.close()

    def sync(self) -> None:
        """Flush pending writes."""

        self._backend.sync()

    def __enter__(self) -> "ExodusWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def initialize(
        self,
        title: str,
        dimension: int,
        node_count: int,
        element_count: int,
        *,
        element_blocks: int = 0,
        node_sets: int = 0,
        side_sets: int = 0,
        edge_count: int = 0,
        edge_blocks: int = 0,
        face_count: int = 0,
        face_blocks: int = 0,
        edge_sets: int = 0,
        face_sets: int = 0,
        element_sets: int = 0,
        floating_point_word_size: int = 8,
    ) -> None:
        """Initialize the Exodus database."""

        if self._initialized:
            raise ExodusWriteError("database is already initialized")
        if dimension not in {1, 2, 3}:
            raise ValueError("dimension must be 1, 2, or 3")

        self._backend.set_attribute(AttributeName.TITLE.value, title)
        self._backend.set_attribute(AttributeName.API_VERSION.value, 5.03)
        self._backend.set_attribute(AttributeName.VERSION.value, 5.03)
        self._backend.set_attribute(
            AttributeName.FLOATING_POINT_WORD_SIZE.value, floating_point_word_size
        )
        self._backend.set_attribute(AttributeName.FILE_SIZE.value, 1)

        self._create_standard_dimensions()
        self._backend.create_dimension(DimensionName.NUM_DIMENSIONS.value, dimension)

        if node_count:
            self._backend.create_dimension(DimensionName.NUM_NODES.value, node_count)
        if element_count:
            self._backend.create_dimension(DimensionName.NUM_ELEMENTS.value, element_count)
        if edge_count:
            self._backend.create_dimension(DimensionName.NUM_EDGES.value, edge_count)
        if face_count:
            self._backend.create_dimension(DimensionName.NUM_FACES.value, face_count)

        self._create_time_variable()

        if node_count:
            self._create_coordinate_variables(dimension)

        self._allocate_blocks(
            count=element_blocks,
            dimension_name=DimensionName.NUM_ELEMENT_BLOCKS.value,
            ids_name=VariableName.ELEMENT_BLOCK_IDS.value,
            status_name=VariableName.ELEMENT_BLOCK_STATUS.value,
            names_name=VariableName.ELEMENT_BLOCK_NAMES.value,
        )
        self._allocate_blocks(
            count=edge_blocks,
            dimension_name=DimensionName.NUM_EDGE_BLOCKS.value,
            ids_name=VariableName.EDGE_BLOCK_IDS.value,
            status_name=VariableName.EDGE_BLOCK_STATUS.value,
            names_name=VariableName.EDGE_BLOCK_NAMES.value,
        )
        self._allocate_blocks(
            count=face_blocks,
            dimension_name=DimensionName.NUM_FACE_BLOCKS.value,
            ids_name=VariableName.FACE_BLOCK_IDS.value,
            status_name=VariableName.FACE_BLOCK_STATUS.value,
            names_name=VariableName.FACE_BLOCK_NAMES.value,
        )
        self._allocate_sets(
            count=node_sets,
            dimension_name=DimensionName.NUM_NODE_SETS.value,
            ids_name=VariableName.NODE_SET_IDS.value,
            status_name=VariableName.NODE_SET_STATUS.value,
            names_name=VariableName.NODE_SET_NAMES.value,
        )
        self._allocate_sets(
            count=side_sets,
            dimension_name=DimensionName.NUM_SIDE_SETS.value,
            ids_name=VariableName.SIDE_SET_IDS.value,
            status_name=VariableName.SIDE_SET_STATUS.value,
            names_name=VariableName.SIDE_SET_NAMES.value,
        )
        self._allocate_sets(
            count=edge_sets,
            dimension_name=DimensionName.NUM_EDGE_SETS.value,
            ids_name=VariableName.EDGE_SET_IDS.value,
            status_name=VariableName.EDGE_SET_STATUS.value,
            names_name=VariableName.EDGE_SET_NAMES.value,
        )
        self._allocate_sets(
            count=face_sets,
            dimension_name=DimensionName.NUM_FACE_SETS.value,
            ids_name=VariableName.FACE_SET_IDS.value,
            status_name=VariableName.FACE_SET_STATUS.value,
            names_name=VariableName.FACE_SET_NAMES.value,
        )
        self._allocate_sets(
            count=element_sets,
            dimension_name=DimensionName.NUM_ELEMENT_SETS.value,
            ids_name=VariableName.ELEMENT_SET_IDS.value,
            status_name=VariableName.ELEMENT_SET_STATUS.value,
            names_name=VariableName.ELEMENT_SET_NAMES.value,
        )

        self._initialized = True

    def write_info_records(self, records: Sequence[str]) -> None:
        """Write Exodus information records."""

        if not records:
            return

        if not self._backend.has_dimension(DimensionName.NUM_INFO_RECORDS.value):
            self._backend.create_dimension(DimensionName.NUM_INFO_RECORDS.value, len(records))

        if not self._backend.has_variable(VariableName.INFO_RECORDS.value):
            self._backend.create_variable(
                VariableName.INFO_RECORDS.value,
                str,
                (DimensionName.NUM_INFO_RECORDS.value, DimensionName.LINE_LENGTH.value),
            )

        self._backend.write_variable(
            VariableName.INFO_RECORDS.value, encode_fixed_width(records, width=80)
        )

    def write_qa_records(self, records: Sequence[Sequence[str]]) -> None:
        """Write Exodus QA records."""

        if not records:
            return

        normalized = []
        for record in records:
            fields = [str(field) for field in record]
            fields = fields[:4] + [""] * max(0, 4 - len(fields))
            normalized.append(fields[:4])

        if not self._backend.has_dimension(DimensionName.NUM_QA_RECORDS.value):
            self._backend.create_dimension(DimensionName.NUM_QA_RECORDS.value, len(normalized))

        if not self._backend.has_variable(VariableName.QA_RECORDS.value):
            self._backend.create_variable(
                VariableName.QA_RECORDS.value,
                str,
                (
                    DimensionName.NUM_QA_RECORDS.value,
                    DimensionName.FOUR.value,
                    DimensionName.STRING_LENGTH.value,
                ),
            )

        encoded = np.asarray([encode_fixed_width(record, width=32) for record in normalized])
        self._backend.write_variable(VariableName.QA_RECORDS.value, encoded)

    def define_property(self, on: Entity, name: str, values: npt.ArrayLike) -> None:
        """Define/write a property array for a block or set entity."""

        values_array = np.asarray(values, dtype=np.int64)

        property_factory = None
        dimension = ""

        if on.is_block:
            block_schema = block_spec(on)
            property_factory = block_schema.property_variable
            dimension = block_schema.count_dimension
        elif on.is_set:
            set_schema = set_spec(on)
            property_factory = set_schema.property_variable
            dimension = set_schema.count_dimension
        else:
            raise ValueError(f"{on.value} does not have properties")

        if property_factory is None:
            raise ValueError(f"{on.value} does not have properties")

        count = self._dimension_size(dimension)
        if len(values_array) != count:
            raise ValueError(f"property value count must be {count}")

        property_index = self._next_property_index(property_factory)
        variable_name = property_factory(property_index)

        self._backend.create_variable(variable_name, int, (dimension,))
        self._backend.set_variable_attribute(variable_name, AttributeName.PROPERTY_NAME.value, name)
        self._backend.write_variable(variable_name, values_array)

    def _next_property_index(self, property_factory) -> int:
        index = 1
        while self._backend.has_variable(property_factory(index)):
            index += 1
        return index

    def define_variables(
        self, on: Entity, names: Sequence[str], *, truth_table: npt.ArrayLike | None = None
    ) -> None:
        """Define result variables for any supported entity."""

        self._require_initialized()
        spec = variable_spec(on)
        if on not in {Entity.GLOBAL, Entity.NODE}:
            if spec.location_entity is None:
                raise ValueError(f"{on.value} variables do not have a location entity")

            location_ids = self._ids_for_variable_location(spec.location_entity)
            if len(location_ids) == 0:
                label = spec.location_entity.value.replace("_", " ")
                raise ExodusWriteError(f"define {label}s before defining {on.value} variables")

        self._backend.create_dimension(spec.count_dimension, len(names))
        self._backend.create_variable(
            spec.names_variable, str, (spec.count_dimension, DimensionName.STRING_LENGTH.value)
        )
        self._backend.write_variable(spec.names_variable, encode_fixed_width(names, width=32))

        if on is Entity.GLOBAL:
            self._backend.create_variable(
                variable_value_name(on, 1), float, (DimensionName.TIME.value, spec.count_dimension)
            )
            return

        if on is Entity.NODE:
            for variable_index in range(1, len(names) + 1):
                self._backend.create_variable(
                    variable_value_name(on, variable_index),
                    float,
                    (DimensionName.TIME.value, DimensionName.NUM_NODES.value),
                )
            return

        location = spec.location_entity
        if location is None:
            raise ValueError(f"{on.value} variables do not have a location entity")

        location_ids = self._ids_for_variable_location(location)
        location_dimension = self._location_count_dimension(location)

        if truth_table is not None:
            table = np.asarray(truth_table, dtype=np.int64)
            expected_shape = (len(location_ids), len(names))
            if table.shape != expected_shape:
                raise ValueError(f"truth_table shape must be {expected_shape}")
        else:
            table = np.ones((len(location_ids), len(names)), dtype=np.int64)

        if spec.truth_table_variable is not None:
            self._backend.create_variable(
                spec.truth_table_variable, int, (location_dimension, spec.count_dimension)
            )
            self._backend.write_variable(spec.truth_table_variable, table)

        for location_position, location_id in enumerate(location_ids):
            location_index = self._location_index(location, int(location_id))
            count_dim = self._value_count_dimension(location, location_index)

            for variable_index in range(1, len(names) + 1):
                if table[location_position, variable_index - 1]:
                    self._backend.create_variable(
                        variable_value_name(on, variable_index, location_index),
                        float,
                        (DimensionName.TIME.value, count_dim),
                    )

    def write_block_attributes(
        self,
        on: Entity,
        block_id: int,
        values: npt.ArrayLike,
        *,
        names: Sequence[str] | None = None,
    ) -> None:
        """Write attributes for an element, edge, or face block."""

        spec = block_spec(on)
        if spec.attributes_dimension is None or spec.attributes_variable is None:
            raise ValueError(f"{on.value} does not support attributes")

        block_index = self._location_index(on, block_id)
        array = np.asarray(values, dtype=np.float64)

        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("attribute values must be one- or two-dimensional")

        block_count = self._dimension_size(spec.object_count_dimension(block_index))
        if array.shape[0] != block_count:
            raise ValueError(f"attribute row count must be {block_count}")

        attr_count = array.shape[1]
        attr_dim = spec.attributes_dimension(block_index)
        attr_var = spec.attributes_variable(block_index)

        if not self._backend.has_dimension(attr_dim):
            self._backend.create_dimension(attr_dim, attr_count)
        if not self._backend.has_variable(attr_var):
            self._backend.create_variable(
                attr_var, float, (spec.object_count_dimension(block_index), attr_dim)
            )

        self._backend.write_variable(attr_var, array)

        if names is not None:
            if spec.attribute_names_variable is None:
                raise ValueError(f"{on.value} does not support attribute names")
            if len(names) != attr_count:
                raise ValueError("attribute name count must match attribute column count")

            name_var = spec.attribute_names_variable(block_index)
            if not self._backend.has_variable(name_var):
                self._backend.create_variable(
                    name_var, str, (attr_dim, DimensionName.STRING_LENGTH.value)
                )
            self._backend.write_variable(name_var, encode_fixed_width(names, width=32))

    def set_block_status(self, entity: Entity, block_id: int, active: bool) -> None:
        """Set block status."""

        spec = block_spec(entity)
        block_index = self._location_index(spec.entity, block_id)
        self._backend.write_variable(spec.status_variable, 1 if active else 0, block_index - 1)

    def set_set_status(self, entity: Entity, set_id: int, active: bool) -> None:
        """Set set status."""

        spec = set_spec(entity)
        set_index = self._location_index(spec.entity, set_id)
        self._backend.write_variable(spec.status_variable, 1 if active else 0, set_index - 1)

    def write_coordinates(
        self, coords: npt.ArrayLike, *, names: Sequence[str] | None = None
    ) -> None:
        """Write nodal coordinates."""

        self._require_initialized()

        array = np.asarray(coords, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("coords must be a two-dimensional array")

        dimension = array.shape[1]
        expected_dimension = self._backend.dimension(DimensionName.NUM_DIMENSIONS.value)
        if dimension != expected_dimension:
            raise ValueError(
                f"coords dimension {dimension} does not match "
                f"initialized dimension {expected_dimension}"
            )

        expected_nodes = self._backend.dimension(DimensionName.NUM_NODES.value, 0)
        if array.shape[0] != expected_nodes:
            raise ValueError(
                f"coords node count {array.shape[0]} does not match "
                f"initialized node count {expected_nodes}"
            )

        coordinate_names = list(names) if names is not None else list("XYZ"[:dimension])
        if len(coordinate_names) != dimension:
            raise ValueError("coordinate name count must match dimension")

        self._backend.write_variable(
            VariableName.COORDINATE_NAMES.value, encode_fixed_width(coordinate_names, width=32)
        )

        for axis in range(dimension):
            self._backend.write_variable(ExodusNames.coordinate(axis), array[:, axis])

    def write_id_map(self, on: Entity, values: npt.ArrayLike) -> None:
        """Write an object ID map."""

        array = np.asarray(values, dtype=np.int64)

        if on is Entity.NODE:
            variable = VariableName.NODE_ID_MAP.value
            dimension = DimensionName.NUM_NODES.value
        elif on is Entity.ELEMENT:
            variable = VariableName.ELEMENT_ID_MAP.value
            dimension = DimensionName.NUM_ELEMENTS.value
        elif on is Entity.EDGE:
            variable = VariableName.EDGE_ID_MAP.value
            dimension = DimensionName.NUM_EDGES.value
        elif on is Entity.FACE:
            variable = VariableName.FACE_ID_MAP.value
            dimension = DimensionName.NUM_FACES.value
        else:
            raise ValueError(f"{on.value} does not have an object ID map")

        expected = self._dimension_size(dimension)
        if len(array) != expected:
            raise ValueError(f"{on.value} ID map length must be {expected}")

        if not self._backend.has_variable(variable):
            self._backend.create_variable(variable, int, (dimension,))

        self._backend.write_variable(variable, array)

    def write_node_id_map(self, values: npt.ArrayLike) -> None:
        """Write node ID map."""

        self.write_id_map(Entity.NODE, values)

    def write_element_id_map(self, values: npt.ArrayLike) -> None:
        """Write element ID map."""

        self.write_id_map(Entity.ELEMENT, values)

    def write_edge_id_map(self, values: npt.ArrayLike) -> None:
        """Write edge ID map."""

        self.write_id_map(Entity.EDGE, values)

    def write_face_id_map(self, values: npt.ArrayLike) -> None:
        """Write face ID map."""

        self.write_id_map(Entity.FACE, values)

    def define_block(
        self,
        entity: Entity,
        block_id: int,
        element_type: str,
        connectivity: npt.ArrayLike,
        *,
        name: str = "",
        zero_based: bool = False,
        edge_connectivity: npt.ArrayLike | None = None,
        face_connectivity: npt.ArrayLike | None = None,
        active: bool = True,
    ) -> None:
        """Define an element, edge, or face block."""

        self._require_initialized()
        spec = block_spec(entity)

        conn = np.asarray(connectivity, dtype=np.int64)
        if conn.ndim != 2:
            raise ValueError("connectivity must be a two-dimensional array")
        if zero_based:
            conn = conn + 1

        block_index = self._next_block_index(spec.entity, block_id)

        count_dim = spec.object_count_dimension(block_index)
        nodes_dim = spec.nodes_per_object_dimension(block_index)
        conn_name = spec.connectivity_variable(block_index)

        self._backend.create_dimension(count_dim, conn.shape[0])
        self._backend.create_dimension(nodes_dim, conn.shape[1])
        self._backend.create_variable(conn_name, int, (count_dim, nodes_dim))
        self._backend.set_variable_attribute(
            conn_name, AttributeName.ELEMENT_TYPE.value, element_type.upper()
        )
        self._backend.write_variable(conn_name, conn)

        if edge_connectivity is not None:
            if spec.edges_per_object_dimension is None or spec.edge_connectivity_variable is None:
                raise ValueError(f"{spec.entity.value} does not support edge connectivity")
            edge_conn = np.asarray(edge_connectivity, dtype=np.int64)
            if zero_based:
                edge_conn = edge_conn + 1
            edge_dim = spec.edges_per_object_dimension(block_index)
            self._backend.create_dimension(edge_dim, edge_conn.shape[1])
            self._backend.create_variable(
                spec.edge_connectivity_variable(block_index), int, (count_dim, edge_dim)
            )
            self._backend.write_variable(spec.edge_connectivity_variable(block_index), edge_conn)

        if face_connectivity is not None:
            if spec.faces_per_object_dimension is None or spec.face_connectivity_variable is None:
                raise ValueError(f"{spec.entity.value} does not support face connectivity")
            face_conn = np.asarray(face_connectivity, dtype=np.int64)
            if zero_based:
                face_conn = face_conn + 1
            face_dim = spec.faces_per_object_dimension(block_index)
            self._backend.create_dimension(face_dim, face_conn.shape[1])
            self._backend.create_variable(
                spec.face_connectivity_variable(block_index), int, (count_dim, face_dim)
            )
            self._backend.write_variable(spec.face_connectivity_variable(block_index), face_conn)

        self._backend.write_variable(spec.ids_variable, int(block_id), block_index - 1)
        self._backend.write_variable(spec.status_variable, 1 if active else 0, block_index - 1)
        self._backend.write_variable(
            spec.names_variable, encode_fixed_width(name, width=32), block_index - 1
        )

    def define_element_block(
        self,
        block_id: int,
        element_type: str,
        connectivity: npt.ArrayLike,
        *,
        name: str = "",
        zero_based: bool = False,
        edge_connectivity: npt.ArrayLike | None = None,
        face_connectivity: npt.ArrayLike | None = None,
        active: bool = True,
    ) -> None:
        """Define an element block."""

        self.define_block(
            Entity.ELEMENT_BLOCK,
            block_id,
            element_type,
            connectivity,
            name=name,
            zero_based=zero_based,
            edge_connectivity=edge_connectivity,
            face_connectivity=face_connectivity,
            active=active,
        )

    def define_edge_block(
        self,
        block_id: int,
        element_type: str,
        connectivity: npt.ArrayLike,
        *,
        name: str = "",
        zero_based: bool = False,
        active: bool = True,
    ) -> None:
        """Define an edge block."""

        self.define_block(
            Entity.EDGE_BLOCK,
            block_id,
            element_type,
            connectivity,
            name=name,
            zero_based=zero_based,
            active=active,
        )

    def define_face_block(
        self,
        block_id: int,
        element_type: str,
        connectivity: npt.ArrayLike,
        *,
        name: str = "",
        zero_based: bool = False,
        active: bool = True,
    ) -> None:
        """Define a face block."""

        self.define_block(
            Entity.FACE_BLOCK,
            block_id,
            element_type,
            connectivity,
            name=name,
            zero_based=zero_based,
            active=active,
        )

    def define_set(
        self,
        entity: Entity,
        set_id: int,
        entries: npt.ArrayLike,
        *,
        extra_entries: npt.ArrayLike | None = None,
        distribution_factors: npt.ArrayLike | None = None,
        name: str = "",
        active: bool = True,
    ) -> None:
        """Define a generic set."""

        self._require_initialized()
        spec = set_spec(entity)

        entry_array = np.asarray(entries, dtype=np.int64)
        if entry_array.ndim != 1:
            raise ValueError("entries must be a one-dimensional array")

        set_index = self._next_set_index(spec.entity, set_id)
        count_dim = spec.entry_count_dimension(set_index)

        self._backend.create_dimension(count_dim, len(entry_array))
        self._backend.create_variable(spec.entries_variable(set_index), int, (count_dim,))
        self._backend.write_variable(spec.entries_variable(set_index), entry_array)

        if extra_entries is not None:
            if spec.extra_entries_variable is None:
                raise ValueError(f"{spec.entity.value} does not support extra entries")
            extra_array = np.asarray(extra_entries, dtype=np.int64)
            if extra_array.shape != entry_array.shape:
                raise ValueError("extra_entries must have the same length as entries")
            self._backend.create_variable(spec.extra_entries_variable(set_index), int, (count_dim,))
            self._backend.write_variable(spec.extra_entries_variable(set_index), extra_array)

        if distribution_factors is not None:
            factors = np.asarray(distribution_factors, dtype=np.float64)
            if len(factors):
                dist_dim = spec.dist_factor_count_dimension(set_index)
                self._backend.create_dimension(dist_dim, len(factors))
                self._backend.create_variable(
                    spec.dist_factors_variable(set_index), float, (dist_dim,)
                )
                self._backend.write_variable(spec.dist_factors_variable(set_index), factors)

        self._backend.write_variable(spec.ids_variable, int(set_id), set_index - 1)
        self._backend.write_variable(spec.status_variable, 1 if active else 0, set_index - 1)
        self._backend.write_variable(
            spec.names_variable, encode_fixed_width(name, width=32), set_index - 1
        )

    def define_node_set(
        self,
        set_id: int,
        nodes: npt.ArrayLike,
        *,
        distribution_factors: npt.ArrayLike | None = None,
        name: str = "",
        active: bool = True,
    ) -> None:
        """Define a node set."""

        self.define_set(
            Entity.NODE_SET,
            set_id,
            nodes,
            distribution_factors=distribution_factors,
            name=name,
            active=active,
        )

    def define_side_set(
        self,
        set_id: int,
        elements: npt.ArrayLike,
        sides: npt.ArrayLike,
        *,
        distribution_factors: npt.ArrayLike | None = None,
        name: str = "",
        active: bool = True,
    ) -> None:
        """Define a side set."""

        self.define_set(
            Entity.SIDE_SET,
            set_id,
            elements,
            extra_entries=sides,
            distribution_factors=distribution_factors,
            name=name,
            active=active,
        )

    def define_edge_set(
        self,
        set_id: int,
        edges: npt.ArrayLike,
        *,
        orientations: npt.ArrayLike | None = None,
        distribution_factors: npt.ArrayLike | None = None,
        name: str = "",
        active: bool = True,
    ) -> None:
        """Define an edge set."""

        self.define_set(
            Entity.EDGE_SET,
            set_id,
            edges,
            extra_entries=orientations,
            distribution_factors=distribution_factors,
            name=name,
            active=active,
        )

    def define_face_set(
        self,
        set_id: int,
        faces: npt.ArrayLike,
        *,
        orientations: npt.ArrayLike | None = None,
        distribution_factors: npt.ArrayLike | None = None,
        name: str = "",
        active: bool = True,
    ) -> None:
        """Define a face set."""

        self.define_set(
            Entity.FACE_SET,
            set_id,
            faces,
            extra_entries=orientations,
            distribution_factors=distribution_factors,
            name=name,
            active=active,
        )

    def define_element_set(
        self,
        set_id: int,
        elements: npt.ArrayLike,
        *,
        distribution_factors: npt.ArrayLike | None = None,
        name: str = "",
        active: bool = True,
    ) -> None:
        """Define an element set."""

        self.define_set(
            Entity.ELEMENT_SET,
            set_id,
            elements,
            distribution_factors=distribution_factors,
            name=name,
            active=active,
        )

    def define_global_variables(self, names: Sequence[str]) -> None:
        """Define global result variables."""

        self.define_variables(Entity.GLOBAL, names)

    def define_node_variables(self, names: Sequence[str]) -> None:
        """Define nodal result variables."""

        self.define_variables(Entity.NODE, names)

    def define_element_variables(
        self, names: Sequence[str], *, truth_table: npt.ArrayLike | None = None
    ) -> None:
        """Define element result variables."""

        self.define_variables(Entity.ELEMENT, names, truth_table=truth_table)

    def define_edge_variables(
        self, names: Sequence[str], *, truth_table: npt.ArrayLike | None = None
    ) -> None:
        """Define edge result variables."""

        self.define_variables(Entity.EDGE, names, truth_table=truth_table)

    def define_face_variables(
        self, names: Sequence[str], *, truth_table: npt.ArrayLike | None = None
    ) -> None:
        """Define face result variables."""

        self.define_variables(Entity.FACE, names, truth_table=truth_table)

    def define_node_set_variables(
        self, names: Sequence[str], *, truth_table: npt.ArrayLike | None = None
    ) -> None:
        """Define node-set result variables."""

        self.define_variables(Entity.NODE_SET, names, truth_table=truth_table)

    def define_side_set_variables(
        self, names: Sequence[str], *, truth_table: npt.ArrayLike | None = None
    ) -> None:
        """Define side-set result variables."""

        self.define_variables(Entity.SIDE_SET, names, truth_table=truth_table)

    def define_edge_set_variables(
        self, names: Sequence[str], *, truth_table: npt.ArrayLike | None = None
    ) -> None:
        """Define edge-set result variables."""

        self.define_variables(Entity.EDGE_SET, names, truth_table=truth_table)

    def define_face_set_variables(
        self, names: Sequence[str], *, truth_table: npt.ArrayLike | None = None
    ) -> None:
        """Define face-set result variables."""

        self.define_variables(Entity.FACE_SET, names, truth_table=truth_table)

    def define_element_set_variables(
        self, names: Sequence[str], *, truth_table: npt.ArrayLike | None = None
    ) -> None:
        """Define element-set result variables."""

        self.define_variables(Entity.ELEMENT_SET, names, truth_table=truth_table)

    def write_time(self, value: float, *, step: int | None = None) -> int:
        """Write a time value and return the one-based time step."""

        self._require_initialized()

        if step is None:
            self._time_step += 1
            step = self._time_step
        elif step < 1:
            raise ValueError("step must be one-based and positive")
        else:
            self._time_step = max(self._time_step, step)

        self._backend.write_variable(VariableName.TIME.value, float(value), step - 1)
        return step

    def write_values(
        self,
        name: str,
        values: npt.ArrayLike,
        *,
        on: Entity,
        step: int | None = None,
        block_id: int | None = None,
        set_id: int | None = None,
    ) -> None:
        """Write values for a result variable."""

        step = self._current_or_requested_step(step)
        spec = variable_spec(on)
        variable_index = self._name_index(spec.names_variable, name)

        if on is Entity.GLOBAL:
            self._backend.write_variable(
                variable_value_name(on, variable_index),
                np.asarray(values, dtype=np.float64),
                step - 1,
            )
            return

        if on is Entity.NODE:
            self._backend.write_variable(
                variable_value_name(on, variable_index),
                np.asarray(values, dtype=np.float64),
                step - 1,
            )
            return

        if spec.location_entity is None:
            raise ValueError(f"{on.value} variables do not have a location entity")

        if spec.location_entity.is_block:
            if block_id is None:
                raise ValueError("block_id is required")
            location_index = self._location_index(spec.location_entity, block_id)
        elif spec.location_entity.is_set:
            if set_id is None:
                raise ValueError("set_id is required")
            location_index = self._location_index(spec.location_entity, set_id)
        else:
            raise ValueError(f"unsupported variable location {spec.location_entity.value}")

        self._backend.write_variable(
            variable_value_name(on, variable_index, location_index),
            np.asarray(values, dtype=np.float64),
            step - 1,
        )

    def write_element_edge_connectivity(
        self, block_id: int, connectivity: npt.ArrayLike, *, zero_based: bool = False
    ) -> None:
        """Write element-to-edge connectivity for an existing element block."""

        block_index = self._location_index(Entity.ELEMENT_BLOCK, block_id)
        spec = block_spec(Entity.ELEMENT_BLOCK)

        if spec.edges_per_object_dimension is None or spec.edge_connectivity_variable is None:
            raise ValueError("element blocks do not support edge connectivity")

        conn = np.asarray(connectivity, dtype=np.int64)
        if conn.ndim != 2:
            raise ValueError("edge connectivity must be two-dimensional")
        if zero_based:
            conn = conn + 1

        count_dim = spec.object_count_dimension(block_index)
        edge_dim = spec.edges_per_object_dimension(block_index)
        variable = spec.edge_connectivity_variable(block_index)

        if not self._backend.has_dimension(edge_dim):
            self._backend.create_dimension(edge_dim, conn.shape[1])
        if not self._backend.has_variable(variable):
            self._backend.create_variable(variable, int, (count_dim, edge_dim))

        self._backend.write_variable(variable, conn)

    def write_element_face_connectivity(
        self, block_id: int, connectivity: npt.ArrayLike, *, zero_based: bool = False
    ) -> None:
        """Write element-to-face connectivity for an existing element block."""

        block_index = self._location_index(Entity.ELEMENT_BLOCK, block_id)
        spec = block_spec(Entity.ELEMENT_BLOCK)

        if spec.faces_per_object_dimension is None or spec.face_connectivity_variable is None:
            raise ValueError("element blocks do not support face connectivity")

        conn = np.asarray(connectivity, dtype=np.int64)
        if conn.ndim != 2:
            raise ValueError("face connectivity must be two-dimensional")
        if zero_based:
            conn = conn + 1

        count_dim = spec.object_count_dimension(block_index)
        face_dim = spec.faces_per_object_dimension(block_index)
        variable = spec.face_connectivity_variable(block_index)

        if not self._backend.has_dimension(face_dim):
            self._backend.create_dimension(face_dim, conn.shape[1])
        if not self._backend.has_variable(variable):
            self._backend.create_variable(variable, int, (count_dim, face_dim))

        self._backend.write_variable(variable, conn)

    def write_global_values(self, values: npt.ArrayLike, *, step: int | None = None) -> None:
        """Write all global values at a time step."""

        step = self._current_or_requested_step(step)
        self._backend.write_variable(
            VariableName.GLOBAL_VARIABLE_VALUES.value,
            np.asarray(values, dtype=np.float64),
            step - 1,
        )

    def write_node_values(
        self, name: str, values: npt.ArrayLike, *, step: int | None = None
    ) -> None:
        """Write one nodal variable at a time step."""

        self.write_values(name, values, on=Entity.NODE, step=step)

    def write_element_values(
        self, name: str, values: npt.ArrayLike, *, block_id: int, step: int | None = None
    ) -> None:
        """Write one element variable for one block at a time step."""

        self.write_values(name, values, on=Entity.ELEMENT, block_id=block_id, step=step)

    def write_edge_values(
        self, name: str, values: npt.ArrayLike, *, block_id: int, step: int | None = None
    ) -> None:
        self.write_values(name, values, on=Entity.EDGE, block_id=block_id, step=step)

    def write_face_values(
        self, name: str, values: npt.ArrayLike, *, block_id: int, step: int | None = None
    ) -> None:
        self.write_values(name, values, on=Entity.FACE, block_id=block_id, step=step)

    def write_node_set_values(
        self, name: str, values: npt.ArrayLike, *, set_id: int, step: int | None = None
    ) -> None:
        self.write_values(name, values, on=Entity.NODE_SET, set_id=set_id, step=step)

    def _ids_for_variable_location(self, location: Entity) -> npt.NDArray[np.int64]:
        if location is Entity.ELEMENT_BLOCK:
            return np.asarray(list(self._element_block_indices), dtype=np.int64)
        if location is Entity.EDGE_BLOCK:
            return np.asarray(list(self._edge_block_indices), dtype=np.int64)
        if location is Entity.FACE_BLOCK:
            return np.asarray(list(self._face_block_indices), dtype=np.int64)
        if location is Entity.NODE_SET:
            return np.asarray(list(self._node_set_indices), dtype=np.int64)
        if location is Entity.SIDE_SET:
            return np.asarray(list(self._side_set_indices), dtype=np.int64)
        if location is Entity.EDGE_SET:
            return np.asarray(list(self._edge_set_indices), dtype=np.int64)
        if location is Entity.FACE_SET:
            return np.asarray(list(self._face_set_indices), dtype=np.int64)
        if location is Entity.ELEMENT_SET:
            return np.asarray(list(self._element_set_indices), dtype=np.int64)
        raise ValueError(f"unsupported variable location {location.value}")

    def _location_index(self, location: Entity, id_value: int) -> int:
        try:
            if location is Entity.ELEMENT_BLOCK:
                return self._element_block_indices[id_value]
            if location is Entity.EDGE_BLOCK:
                return self._edge_block_indices[id_value]
            if location is Entity.FACE_BLOCK:
                return self._face_block_indices[id_value]
            if location is Entity.NODE_SET:
                return self._node_set_indices[id_value]
            if location is Entity.SIDE_SET:
                return self._side_set_indices[id_value]
            if location is Entity.EDGE_SET:
                return self._edge_set_indices[id_value]
            if location is Entity.FACE_SET:
                return self._face_set_indices[id_value]
            if location is Entity.ELEMENT_SET:
                return self._element_set_indices[id_value]
        except KeyError as exc:
            label = location.value.replace("_", " ")
            raise ExodusLookupError(f"{label} ID {id_value} not found") from exc

        raise ValueError(f"unsupported variable location {location.value}")

    def _location_count_dimension(self, location: Entity) -> str:
        if location is Entity.ELEMENT_BLOCK:
            return DimensionName.NUM_ELEMENT_BLOCKS.value
        if location is Entity.EDGE_BLOCK:
            return DimensionName.NUM_EDGE_BLOCKS.value
        if location is Entity.FACE_BLOCK:
            return DimensionName.NUM_FACE_BLOCKS.value
        if location is Entity.NODE_SET:
            return DimensionName.NUM_NODE_SETS.value
        if location is Entity.SIDE_SET:
            return DimensionName.NUM_SIDE_SETS.value
        if location is Entity.EDGE_SET:
            return DimensionName.NUM_EDGE_SETS.value
        if location is Entity.FACE_SET:
            return DimensionName.NUM_FACE_SETS.value
        if location is Entity.ELEMENT_SET:
            return DimensionName.NUM_ELEMENT_SETS.value
        raise ValueError(f"unsupported variable location {location.value}")

    def _value_count_dimension(self, location: Entity, location_index: int) -> str:
        if location is Entity.ELEMENT_BLOCK:
            return ExodusNames.block_count(location_index)
        if location is Entity.EDGE_BLOCK:
            return ExodusNames.edge_block_count(location_index)
        if location is Entity.FACE_BLOCK:
            return ExodusNames.face_block_count(location_index)
        if location is Entity.NODE_SET:
            return ExodusNames.node_set_count(location_index)
        if location is Entity.SIDE_SET:
            return ExodusNames.side_set_count(location_index)
        if location is Entity.EDGE_SET:
            return f"num_edge_es{location_index}"
        if location is Entity.FACE_SET:
            return f"num_face_fs{location_index}"
        if location is Entity.ELEMENT_SET:
            return f"num_ele_els{location_index}"
        raise ValueError(f"unsupported variable location {location.value}")

    def _create_standard_dimensions(self) -> None:
        self._backend.create_dimension(DimensionName.TIME.value, None)
        self._backend.create_dimension(DimensionName.STRING_LENGTH.value, 32)
        self._backend.create_dimension(DimensionName.NAME_LENGTH.value, 256)
        self._backend.create_dimension(DimensionName.LINE_LENGTH.value, 80)
        self._backend.create_dimension(DimensionName.FOUR.value, 4)

    def _create_time_variable(self) -> None:
        self._backend.create_variable(VariableName.TIME.value, float, (DimensionName.TIME.value,))

    def _create_coordinate_variables(self, dimension: int) -> None:
        self._backend.create_variable(
            VariableName.COORDINATE_NAMES.value,
            str,
            (DimensionName.NUM_DIMENSIONS.value, DimensionName.STRING_LENGTH.value),
        )
        for axis in range(dimension):
            self._backend.create_variable(
                ExodusNames.coordinate(axis), float, (DimensionName.NUM_NODES.value,)
            )

    def _allocate_blocks(
        self, *, count: int, dimension_name: str, ids_name: str, status_name: str, names_name: str
    ) -> None:
        if not count:
            return

        self._backend.create_dimension(dimension_name, count)
        self._backend.create_variable(ids_name, int, (dimension_name,))
        self._backend.create_variable(status_name, int, (dimension_name,))
        self._backend.create_variable(
            names_name, str, (dimension_name, DimensionName.STRING_LENGTH.value)
        )
        self._backend.write_variable(ids_name, np.zeros(count, dtype=np.int32))
        self._backend.write_variable(status_name, np.zeros(count, dtype=np.int32))
        self._backend.write_variable(names_name, encode_fixed_width([""] * count, width=32))
        self._backend.set_variable_attribute(ids_name, AttributeName.PROPERTY_NAME.value, "ID")

    def _allocate_sets(
        self, *, count: int, dimension_name: str, ids_name: str, status_name: str, names_name: str
    ) -> None:
        self._allocate_blocks(
            count=count,
            dimension_name=dimension_name,
            ids_name=ids_name,
            status_name=status_name,
            names_name=names_name,
        )

    def _next_block_index(self, entity: Entity, block_id: int) -> int:
        if entity is Entity.ELEMENT_BLOCK:
            self._element_block_counter += 1
            counter = self._element_block_counter
            allocated = self._dimension_size(DimensionName.NUM_ELEMENT_BLOCKS.value)
            indices = self._element_block_indices
            label = "element blocks"
        elif entity is Entity.EDGE_BLOCK:
            self._edge_block_counter += 1
            counter = self._edge_block_counter
            allocated = self._dimension_size(DimensionName.NUM_EDGE_BLOCKS.value)
            indices = self._edge_block_indices
            label = "edge blocks"
        elif entity is Entity.FACE_BLOCK:
            self._face_block_counter += 1
            counter = self._face_block_counter
            allocated = self._dimension_size(DimensionName.NUM_FACE_BLOCKS.value)
            indices = self._face_block_indices
            label = "face blocks"
        else:
            raise ValueError(f"{entity.value} is not a block entity")

        if counter > allocated:
            raise ExodusWriteError(f"allocated number of {label} exceeded")

        indices[block_id] = counter
        return counter

    def _next_set_index(self, entity: Entity, set_id: int) -> int:
        if entity is Entity.NODE_SET:
            self._node_set_counter += 1
            counter = self._node_set_counter
            allocated = self._dimension_size(DimensionName.NUM_NODE_SETS.value)
            indices = self._node_set_indices
            label = "node sets"
        elif entity is Entity.SIDE_SET:
            self._side_set_counter += 1
            counter = self._side_set_counter
            allocated = self._dimension_size(DimensionName.NUM_SIDE_SETS.value)
            indices = self._side_set_indices
            label = "side sets"
        elif entity is Entity.EDGE_SET:
            self._edge_set_counter += 1
            counter = self._edge_set_counter
            allocated = self._dimension_size(DimensionName.NUM_EDGE_SETS.value)
            indices = self._edge_set_indices
            label = "edge sets"
        elif entity is Entity.FACE_SET:
            self._face_set_counter += 1
            counter = self._face_set_counter
            allocated = self._dimension_size(DimensionName.NUM_FACE_SETS.value)
            indices = self._face_set_indices
            label = "face sets"
        elif entity is Entity.ELEMENT_SET:
            self._element_set_counter += 1
            counter = self._element_set_counter
            allocated = self._dimension_size(DimensionName.NUM_ELEMENT_SETS.value)
            indices = self._element_set_indices
            label = "element sets"
        else:
            raise ValueError(f"{entity.value} is not a set entity")

        if counter > allocated:
            raise ExodusWriteError(f"allocated number of {label} exceeded")

        indices[set_id] = counter
        return counter

    def _next_element_block_index(self, block_id: int) -> int:
        self._element_block_counter += 1
        allocated = self._dimension_size(DimensionName.NUM_ELEMENT_BLOCKS.value)
        if self._element_block_counter > allocated:
            raise ExodusWriteError("allocated number of element blocks exceeded")

        self._element_block_indices[block_id] = self._element_block_counter
        return self._element_block_counter

    def _next_node_set_index(self, set_id: int) -> int:
        self._node_set_counter += 1
        allocated = self._dimension_size(DimensionName.NUM_NODE_SETS.value)
        if self._node_set_counter > allocated:
            raise ExodusWriteError("allocated number of node sets exceeded")

        self._node_set_indices[set_id] = self._node_set_counter
        return self._node_set_counter

    def _next_side_set_index(self, set_id: int) -> int:
        self._side_set_counter += 1
        allocated = self._dimension_size(DimensionName.NUM_SIDE_SETS.value)
        if self._side_set_counter > allocated:
            raise ExodusWriteError("allocated number of side sets exceeded")

        self._side_set_indices[set_id] = self._side_set_counter
        return self._side_set_counter

    def _current_or_requested_step(self, step: int | None) -> int:
        if step is None:
            if self._time_step < 1:
                raise ExodusWriteError("write a time value before writing result values")
            return self._time_step

        if step < 1:
            raise ValueError("step must be one-based and positive")
        return step

    def _name_index(self, variable_name: str, requested: str) -> int:
        names = self._backend.variable(variable_name, default=[])
        decoded = [str(name) for name in np.asarray(names).reshape(-1)]
        for index, name in enumerate(decoded, start=1):
            if name == requested or name.lower() == requested.lower():
                return index

        raise ExodusLookupError(f"variable {requested!r} not found")

    def _dimension_size(self, name: str, default: int = 0) -> int:
        value = self._backend.dimension(name, default)
        if value is None:
            return default
        return int(value)

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise ExodusWriteError("database is not initialized")


__all__ = ["ExodusWriter"]
