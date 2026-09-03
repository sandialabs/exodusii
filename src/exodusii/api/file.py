# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Modern read API for Exodus databases."""

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import numpy.typing as npt

from exodusii.core.entities import Entity
from exodusii.core.entities import entity
from exodusii.core.errors import ExodusInvalidEntityError
from exodusii.core.errors import ExodusLookupError
from exodusii.core.models import Block
from exodusii.core.models import InitParams
from exodusii.core.models import SetInfo
from exodusii.core.names import AttributeName
from exodusii.core.names import DimensionName
from exodusii.core.names import ExodusNames
from exodusii.core.names import VariableName
from exodusii.core.schema import block_spec
from exodusii.core.schema import set_spec
from exodusii.core.schema import variable_spec
from exodusii.core.schema import VariableSpec
from exodusii.core.schema import variable_value_name
from exodusii.core.strings import decode_text
from exodusii.core.strings import string_array
from exodusii.core.time import TimeSelector
from exodusii.core.time import resolve_time
from exodusii.io.backend import FileMode
from exodusii.io.backend import NetCDFBackend
from exodusii.io.netcdf4_backend import NetCDF4Backend


class ExodusFile:
    """Modern Exodus database reader.

    Parameters
    ----------
    backend
        NetCDF backend implementing :class:`exodusii.io.backend.NetCDFBackend`.
    """

    def __init__(self, backend: NetCDFBackend) -> None:
        self._backend = backend
        # Cache for immutable metadata reads (times, variable-name tables,
        # set/block id arrays, and index lookups).  ExodusFile is a pure
        # reader (all writes go through ExodusWriter with its own backend),
        # so these are stable for the lifetime of the instance.  The cache
        # is cleared on close()/sync() as a safety measure.
        self._cache: dict[str, Any] = {}

    @classmethod
    def open(cls, path: str | Path, mode: str = "r") -> "ExodusFile":
        """Open an Exodus database."""

        return cls(NetCDF4Backend(path, mode=mode))

    @property
    def backend(self) -> NetCDFBackend:
        """Underlying NetCDF backend."""

        return self._backend

    @property
    def path(self) -> Path:
        """Database path."""

        return self._backend.path

    @property
    def mode(self) -> FileMode:
        """Open mode."""

        return self._backend.mode

    def close(self) -> None:
        """Close the database."""

        self._cache.clear()
        self._backend.close()

    def sync(self) -> None:
        """Flush pending writes."""

        self._cache.clear()
        self._backend.sync()

    def __enter__(self) -> "ExodusFile":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def title(self) -> str:
        """Database title."""

        return str(self._backend.attribute(AttributeName.TITLE.value, ""))

    @property
    def version(self) -> float | None:
        """Exodus database version, if present."""

        value = self._backend.attribute(AttributeName.VERSION.value, None)
        return None if value is None else float(value)

    @property
    def api_version(self) -> float | None:
        """Exodus API version, if present."""

        value = self._backend.attribute(AttributeName.API_VERSION.value, None)
        return None if value is None else float(value)

    @property
    def storage_type(self) -> str:
        """Floating-point storage type: ``"f"`` for 4-byte, ``"d"`` for 8-byte."""

        word_size = self._backend.attribute(AttributeName.FLOATING_POINT_WORD_SIZE.value, None)
        if word_size is None:
            word_size = self._backend.attribute(
                AttributeName.FLOATING_POINT_WORD_SIZE_LEGACY.value, 8
            )

        return "f" if int(word_size) == 4 else "d"

    @property
    def dimension(self) -> int:
        """Spatial dimension."""

        return self.dimension_size(DimensionName.NUM_DIMENSIONS.value, default=0)

    @property
    def node_count(self) -> int:
        """Number of nodes."""

        return self.dimension_size(DimensionName.NUM_NODES.value, default=0)

    @property
    def edge_count(self) -> int:
        """Number of edges."""

        return self.dimension_size(DimensionName.NUM_EDGES.value, default=0)

    @property
    def face_count(self) -> int:
        """Number of faces."""

        return self.dimension_size(DimensionName.NUM_FACES.value, default=0)

    @property
    def element_count(self) -> int:
        """Number of elements."""

        return self.dimension_size(DimensionName.NUM_ELEMENTS.value, default=0)

    @property
    def element_block_count(self) -> int:
        """Number of element blocks."""

        return self.dimension_size(DimensionName.NUM_ELEMENT_BLOCKS.value, default=0)

    @property
    def node_set_count(self) -> int:
        """Number of node sets."""

        return self.dimension_size(DimensionName.NUM_NODE_SETS.value, default=0)

    @property
    def side_set_count(self) -> int:
        """Number of side sets."""

        return self.dimension_size(DimensionName.NUM_SIDE_SETS.value, default=0)

    def info_records(self) -> tuple[str, ...]:
        """Return Exodus information records."""

        values = self._backend.variable(VariableName.INFO_RECORDS.value, default=None)
        if values is None:
            return ()

        decoded = string_array(values)
        return tuple(str(value).rstrip(" \x00") for value in decoded)

    def qa_records(self) -> tuple[tuple[str, str, str, str], ...]:
        """Return Exodus QA records."""

        values = self._backend.variable(VariableName.QA_RECORDS.value, default=None)
        if values is None:
            return ()

        array = np.asarray(values)
        if array.size == 0:
            return ()

        records: list[tuple[str, str, str, str]] = []

        # Backend/string decoding may have flattened 3-D char arrays to strings
        # of four fields. Preserve that form if already decoded.
        if array.ndim == 1:
            for item in array:
                fields = _four_fields(str(item).split())
                records.append(fields)
            return tuple(records)

        if array.ndim == 3:
            for record in array:
                fields = _four_fields(list(_decode_name_table(record, expected_count=4)))
                records.append(fields)
            return tuple(records)

        return ()

    def set_ids(self, on: Entity | str, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return set IDs for a set entity."""

        spec = set_spec(on)
        cache_key = f"set_ids:{spec.entity.value}"
        ids = self._cache.get(cache_key)
        if ids is None:
            values = self._backend.variable(spec.ids_variable, default=None)
            if values is None:
                ids = np.asarray([], dtype=np.int64)
            else:
                ids = np.asarray(values, dtype=np.int64)
            ids.setflags(write=False)
            self._cache[cache_key] = ids

        if active_only:
            status = self.set_status(spec.entity)
            ids = ids[status.astype(bool)]

        return ids

    def set_status(self, on: Entity | str) -> npt.NDArray[np.int64]:
        """Return set status flags."""

        spec = set_spec(on)
        count = self.dimension_size(spec.count_dimension, default=0)
        values = self._backend.variable(spec.status_variable, default=None)

        if values is None:
            return np.ones(count, dtype=np.int64)

        return np.asarray(values, dtype=np.int64)

    def set_is_active(self, on: Entity | str, set_id: int) -> bool:
        """Return true if a set is active."""

        index = self._set_index(on, set_id)
        status = self.set_status(on)
        if index - 1 >= len(status):
            return False
        return bool(status[index - 1])

    def set(self, on: Entity | str, set_id: int) -> SetInfo:
        """Return generic set metadata and entries."""

        spec = set_spec(on)
        set_index = self._set_index(spec.entity, set_id)

        entries = np.asarray(
            self._backend.variable(spec.entries_variable(set_index), default=[]), dtype=np.int64
        )

        extra_entries = None
        if spec.extra_entries_variable is not None:
            values = self._backend.variable(spec.extra_entries_variable(set_index), default=None)
            if values is not None:
                extra_entries = np.asarray(values, dtype=np.int64)

        dist_facts = self._backend.variable(spec.dist_factors_variable(set_index), default=None)
        dist_array = None if dist_facts is None else np.asarray(dist_facts, dtype=np.float64)

        return SetInfo(
            id=set_id,
            index=set_index,
            entity=spec.entity,
            count=len(entries),
            distribution_factors=0 if dist_array is None else len(dist_array),
            name=self._entity_name(spec.entity, set_index),
            entries=entries,
            extra_entries=extra_entries,
            distribution_values=dist_array,
        )

    def init_params(self) -> InitParams:
        """Return top-level initialization parameters."""

        return InitParams(
            title=self.title,
            dimension=self.dimension,
            nodes=self.node_count,
            elements=self.element_count,
            element_blocks=self.element_block_count,
            node_sets=self.node_set_count,
            side_sets=self.side_set_count,
            edges=self.edge_count,
            edge_blocks=self.dimension_size(DimensionName.NUM_EDGE_BLOCKS.value, default=0),
            edge_sets=self.dimension_size(DimensionName.NUM_EDGE_SETS.value, default=0),
            faces=self.face_count,
            face_blocks=self.dimension_size(DimensionName.NUM_FACE_BLOCKS.value, default=0),
            face_sets=self.dimension_size(DimensionName.NUM_FACE_SETS.value, default=0),
            element_sets=self.dimension_size(DimensionName.NUM_ELEMENT_SETS.value, default=0),
            node_maps=self.dimension_size(DimensionName.NUM_NODE_MAPS.value, default=0),
            element_maps=self.dimension_size(DimensionName.NUM_ELEMENT_MAPS.value, default=0),
            edge_maps=self.dimension_size(DimensionName.NUM_EDGE_MAPS.value, default=0),
            face_maps=self.dimension_size(DimensionName.NUM_FACE_MAPS.value, default=0),
        )

    def dimensions(self) -> tuple[str, ...]:
        """Return all NetCDF dimension names."""

        return self._backend.dimensions()

    def variables(self) -> tuple[str, ...]:
        """Return all NetCDF variable names."""

        return self._backend.variables()

    def dimension_size(self, name: str, *, default: int | None = None) -> int:
        """Return a dimension size."""

        value = self._backend.dimension(name, default)
        if value is None:
            raise ExodusLookupError(f"dimension {name!r} not found")
        return int(value)

    def variable(self, name: str, *, default: Any = None, raw: bool = False) -> Any:
        """Return a raw NetCDF variable value."""

        return self._backend.variable(name, default=default, raw=raw)

    def times(self) -> npt.NDArray[np.float64]:
        """Return all time values."""

        cached = self._cache.get("times")
        if cached is None:
            values = self._backend.variable(VariableName.TIME.value, default=[])
            cached = np.asarray(values, dtype=np.float64)
            cached.setflags(write=False)
            self._cache["times"] = cached
        return cached

    def coordinate_names(self) -> npt.NDArray[np.str_]:
        """Return coordinate names."""

        default = np.asarray(["X", "Y", "Z"][: self.dimension], dtype=object)
        values = self._backend.variable(VariableName.COORDINATE_NAMES.value, default=default)
        names = _decode_name_table(values, expected_count=self.dimension)
        return np.asarray(names[: self.dimension], dtype=str)

    def coordinates(
        self, *, time: TimeSelector = None, displaced: bool = False
    ) -> npt.NDArray[np.float64]:
        """Return nodal coordinates.

        Supports both large-model files (separate ``coordx``/``coordy``/``coordz``
        variables, the modern default) and normal-model files (a combined 2D
        ``coord`` variable with shape ``(num_dim, num_nodes)``), matching the
        ``ex_large_model`` branching in the SEACAS C library (``ex_get_coord.c``).

        If ``displaced`` is true, displacement variables are added at the selected
        time.
        """

        # Try large-model format first (coordx / coordy / coordz).
        coord_names = [ExodusNames.coordinate(i) for i in range(self.dimension)]
        components = [self._backend.variable(name, default=None) for name in coord_names]

        if all(c is not None for c in components):
            coords = np.column_stack(components).astype(np.float64)
        else:
            # Fall back to normal-model combined ``coord`` variable
            # (shape: num_dim × num_nodes, stored row-major).
            combined = self._backend.variable(VariableName.COORDINATES.value, default=None)
            if combined is None:
                raise ValueError(
                    "nodal coordinates not found: neither 'coordx'/'coordy'/'coordz' "
                    "nor the combined 'coord' variable is present in the file"
                )
            # combined shape is (num_dim, num_nodes); transpose to (num_nodes, num_dim)
            coords = np.asarray(combined, dtype=np.float64).T

        if displaced:
            coords = coords + self.displacements(time=time)

        return coords

    def displacement_variable_names(self) -> tuple[str, ...]:
        """Return recognized displacement variable names."""

        candidates = {
            f"{base}{axis}"
            for base in ("displ", "disp", "displ_")
            for axis in "xyz"[: self.dimension]
        }
        names = self.variable_names(Entity.NODE)
        found = tuple(name for name in names if name.lower() in candidates)

        if not found:
            return ()
        if len(found) != self.dimension:
            raise ValueError("incorrect number of displacement variable names found")

        return found

    def displacements(self, *, time: TimeSelector = None) -> npt.NDArray[np.float64]:
        """Return nodal displacements at a selected time."""

        names = self.displacement_variable_names()
        if not names:
            return np.zeros((self.node_count, self.dimension), dtype=np.float64)

        return np.column_stack([self.values(name, on=Entity.NODE, time=time) for name in names])

    def ids(self, on: Entity | str) -> npt.NDArray[np.int64]:
        """Return Exodus IDs for an object, block, or set entity."""

        ent = entity(on)
        name = ExodusNames.ids(ent)

        default: npt.NDArray[np.int64] | None
        if ent is Entity.NODE:
            default = np.arange(1, self.node_count + 1, dtype=np.int64)
        elif ent is Entity.ELEMENT:
            default = np.arange(1, self.element_count + 1, dtype=np.int64)
        elif ent is Entity.EDGE:
            default = np.arange(1, self.edge_count + 1, dtype=np.int64)
        elif ent is Entity.FACE:
            default = np.arange(1, self.face_count + 1, dtype=np.int64)
        else:
            default = None

        values = self._backend.variable(name, default=default)
        if values is None:
            return np.asarray([], dtype=np.int64)

        return np.asarray(values, dtype=np.int64)

    def element_block_ids(self, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return element block IDs."""

        return self.block_ids(Entity.ELEMENT_BLOCK, active_only=active_only)

    def block_ids(self, on: Entity | str, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return block IDs for an element, edge, or face block entity."""

        spec = block_spec(on)
        cache_key = f"block_ids:{spec.entity.value}"
        ids = self._cache.get(cache_key)
        if ids is None:
            values = self._backend.variable(spec.ids_variable, default=None)
            if values is None:
                ids = np.asarray([], dtype=np.int64)
            else:
                ids = np.asarray(values, dtype=np.int64)
            ids.setflags(write=False)
            self._cache[cache_key] = ids

        if active_only:
            status = self.block_status(spec.entity)
            ids = ids[status.astype(bool)]

        return ids

    def block_status(self, on: Entity | str) -> npt.NDArray[np.int64]:
        """Return block status flags."""

        spec = block_spec(on)
        count = self.dimension_size(spec.count_dimension, default=0)
        values = self._backend.variable(spec.status_variable, default=None)

        if values is None:
            return np.ones(count, dtype=np.int64)

        return np.asarray(values, dtype=np.int64)

    def block_is_active(self, on: Entity | str, block_id: int) -> bool:
        """Return true if a block is active."""

        index = self._block_index(on, block_id)
        status = self.block_status(on)
        if index - 1 >= len(status):
            return False
        return bool(status[index - 1])

    def block(self, on: Entity | str, block_id: int) -> Block:
        """Return generic block metadata."""

        spec = block_spec(on)
        block_index = self._block_index(spec.entity, block_id)
        conn_name = spec.connectivity_variable(block_index)

        element_type = ""
        if self._backend.has_variable(conn_name):
            element_type = str(
                self._backend.variable_attribute(
                    conn_name, AttributeName.ELEMENT_TYPE.value, default=""
                )
            )

        attributes = (
            self.dimension_size(spec.attributes_dimension(block_index), default=0)
            if spec.attributes_dimension is not None
            else 0
        )
        edges_per_object = (
            self.dimension_size(spec.edges_per_object_dimension(block_index), default=0)
            if spec.edges_per_object_dimension is not None
            else 0
        )
        faces_per_object = (
            self.dimension_size(spec.faces_per_object_dimension(block_index), default=0)
            if spec.faces_per_object_dimension is not None
            else 0
        )

        return Block(
            id=block_id,
            index=block_index,
            entity=spec.entity,
            element_type=element_type,
            count=self.dimension_size(spec.object_count_dimension(block_index), default=0),
            nodes_per_entity=self.dimension_size(
                spec.nodes_per_object_dimension(block_index), default=0
            ),
            edges_per_entity=edges_per_object,
            faces_per_entity=faces_per_object,
            attributes=attributes,
            name=self._entity_name(spec.entity, block_index),
        )

    def block_connectivity(
        self, on: Entity | str, block_id: int, *, zero_based: bool = False
    ) -> npt.NDArray[np.int64]:
        """Return nodal connectivity for a block."""

        spec = block_spec(on)
        block_index = self._block_index(spec.entity, block_id)
        values = self._backend.variable(spec.connectivity_variable(block_index), default=None)
        if values is None:
            return np.empty((0, 0), dtype=np.int64)
        array = np.asarray(values, dtype=np.int64)
        return array - 1 if zero_based else array

    def element_edge_connectivity(
        self, block_id: int, *, zero_based: bool = False
    ) -> npt.NDArray[np.int64] | None:
        """Return element-to-edge connectivity for an element block."""

        spec = block_spec(Entity.ELEMENT_BLOCK)
        block_index = self._block_index(Entity.ELEMENT_BLOCK, block_id)

        if spec.edge_connectivity_variable is None:
            return None

        name = spec.edge_connectivity_variable(block_index)
        values = self._backend.variable(name, default=None)
        if values is None:
            return None

        array = np.asarray(values, dtype=np.int64)
        return array - 1 if zero_based else array

    def element_face_connectivity(
        self, block_id: int, *, zero_based: bool = False
    ) -> npt.NDArray[np.int64] | None:
        """Return element-to-face connectivity for an element block."""

        spec = block_spec(Entity.ELEMENT_BLOCK)
        block_index = self._block_index(Entity.ELEMENT_BLOCK, block_id)

        if spec.face_connectivity_variable is None:
            return None

        name = spec.face_connectivity_variable(block_index)
        values = self._backend.variable(name, default=None)
        if values is None:
            return None

        array = np.asarray(values, dtype=np.int64)
        return array - 1 if zero_based else array

    def node_set_ids(self, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return node set IDs."""

        return self.set_ids(Entity.NODE_SET, active_only=active_only)

    def side_set_ids(self, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return side set IDs."""

        return self.set_ids(Entity.SIDE_SET, active_only=active_only)

    def edge_set_ids(self, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return edge set IDs."""

        return self.set_ids(Entity.EDGE_SET, active_only=active_only)

    def face_set_ids(self, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return face set IDs."""

        return self.set_ids(Entity.FACE_SET, active_only=active_only)

    def element_set_ids(self, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return element set IDs."""

        return self.set_ids(Entity.ELEMENT_SET, active_only=active_only)

    def element_block(self, block_id: int) -> Block:
        """Return element block metadata."""

        return self.block(Entity.ELEMENT_BLOCK, block_id)

    def edge_block(self, block_id: int) -> Block:
        """Return edge block metadata."""

        return self.block(Entity.EDGE_BLOCK, block_id)

    def face_block(self, block_id: int) -> Block:
        """Return face block metadata."""

        return self.block(Entity.FACE_BLOCK, block_id)

    def element_connectivity(
        self, block_id: int, *, zero_based: bool = False
    ) -> npt.NDArray[np.int64]:
        """Return element nodal connectivity for a block."""

        return self.block_connectivity(Entity.ELEMENT_BLOCK, block_id, zero_based=zero_based)

    def edge_connectivity(
        self, block_id: int, *, zero_based: bool = False
    ) -> npt.NDArray[np.int64]:
        """Return edge-block nodal connectivity."""

        return self.block_connectivity(Entity.EDGE_BLOCK, block_id, zero_based=zero_based)

    def face_connectivity(
        self, block_id: int, *, zero_based: bool = False
    ) -> npt.NDArray[np.int64]:
        """Return face-block nodal connectivity."""

        return self.block_connectivity(Entity.FACE_BLOCK, block_id, zero_based=zero_based)

    def node_set(self, set_id: int) -> SetInfo:
        """Return node-set metadata and entries."""

        return self.set(Entity.NODE_SET, set_id)

    def side_set(self, set_id: int) -> SetInfo:
        """Return side-set metadata and entries."""

        return self.set(Entity.SIDE_SET, set_id)

    def edge_set(self, set_id: int) -> SetInfo:
        """Return edge-set metadata and entries."""

        return self.set(Entity.EDGE_SET, set_id)

    def face_set(self, set_id: int) -> SetInfo:
        """Return face-set metadata and entries."""

        return self.set(Entity.FACE_SET, set_id)

    def element_set(self, set_id: int) -> SetInfo:
        """Return element-set metadata and entries."""

        return self.set(Entity.ELEMENT_SET, set_id)

    def edge_block_ids(self, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return edge block IDs."""

        return self.block_ids(Entity.EDGE_BLOCK, active_only=active_only)

    def face_block_ids(self, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return face block IDs."""

        return self.block_ids(Entity.FACE_BLOCK, active_only=active_only)

    def property_names(self, on: Entity | str) -> tuple[str, ...]:
        """Return property names for a block or set entity."""

        ent = entity(on)

        property_factory = None

        if ent.is_block:
            block_schema = block_spec(ent)
            property_factory = block_schema.property_variable
        elif ent.is_set:
            set_schema = set_spec(ent)
            property_factory = set_schema.property_variable
        else:
            raise ExodusInvalidEntityError(f"{ent.value!r} does not have properties")

        if property_factory is None:
            return ()

        names: list[str] = []
        index = 1
        while True:
            variable_name = property_factory(index)
            if not self._backend.has_variable(variable_name):
                break
            name = self._backend.variable_attribute(
                variable_name, AttributeName.PROPERTY_NAME.value, default=""
            )
            names.append(str(name))
            index += 1

        if names and not names[0]:
            names[0] = "ID"

        return tuple(names)

    def property_values(self, on: Entity | str, name: str) -> npt.NDArray[np.int64]:
        """Return all values for one property."""

        ent = entity(on)
        property_index = self._property_index(ent, name)

        if ent.is_block:
            block_schema = block_spec(ent)
            property_factory = block_schema.property_variable
        elif ent.is_set:
            set_schema = set_spec(ent)
            property_factory = set_schema.property_variable
        else:
            raise ExodusInvalidEntityError(f"{ent.value!r} does not have properties")

        if property_factory is None:
            raise ExodusInvalidEntityError(f"{ent.value!r} does not have properties")

        variable_name = property_factory(property_index)
        return np.asarray(self._backend.variable(variable_name, default=[]), dtype=np.int64)

    def property_value(self, on: Entity | str, id_value: int, name: str) -> int:
        """Return one property value for a block or set ID."""

        ent = entity(on)
        values = self.property_values(ent, name)

        if ent.is_block:
            index = self._block_index(ent, id_value)
        elif ent.is_set:
            index = self._set_index(ent, id_value)
        else:
            raise ExodusInvalidEntityError(f"{ent.value!r} does not have properties")

        return int(values[index - 1])

    def _property_index(self, ent: Entity, name: str) -> int:
        requested = name.lower()
        for index, property_name in enumerate(self.property_names(ent), start=1):
            if property_name.lower() == requested:
                return index

        raise ExodusLookupError(f"property {name!r} not found for {ent.value}")

    def variable_names(self, on: Entity | str) -> tuple[str, ...]:
        """Return result variable names for a location."""

        ent = entity(on)
        if not ent.is_variable_location:
            raise ExodusInvalidEntityError(f"{ent.value!r} is not a variable location")

        cache_key = f"variable_names:{ent.value}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        spec = variable_spec(ent)
        expected_count = self.dimension_size(spec.count_dimension, default=0)
        if expected_count == 0:
            self._cache[cache_key] = ()
            return ()

        values = self._backend.variable(spec.names_variable, default=[])
        names = _decode_name_table(values, expected_count=expected_count)
        result = tuple(name for name in names if name)
        self._cache[cache_key] = result
        return result

    def variable_truth_table(
        self, on: Entity | str, *, id: int | None = None
    ) -> npt.NDArray[np.int64] | None:
        """Return variable truth table for block/set variables.

        When no explicit truth table is stored in the file, the table is derived
        dynamically by probing whether each per-block/set result variable exists in
        the NetCDF file — matching the behaviour of ``ex_get_truth_table`` in the
        SEACAS C library (``ex_get_truth_table.c:162–178``).

        Returns ``None`` only when the entity type does not support a truth table
        (e.g. ``Entity.GLOBAL``).
        """

        ent = entity(on)
        spec = variable_spec(ent)
        if spec.truth_table_variable is None:
            return None

        table = self._backend.variable(spec.truth_table_variable, default=None)
        if table is not None:
            array = np.asarray(table, dtype=np.int64)
        else:
            # Derive dynamically: probe whether vals_*_varN*M exists for every
            # (variable_index, location_index) pair — mirrors ex_get_truth_table.c.
            array = self._derive_truth_table(spec)
            if array is None:
                return None

        if id is None:
            return array

        if spec.location_entity is None:
            return array

        if spec.location_entity.is_block:
            index = self._block_index(spec.location_entity, id)
        elif spec.location_entity.is_set:
            index = self._set_index(spec.location_entity, id)
        else:
            return array

        return array[index - 1]

    def _derive_truth_table(self, spec: VariableSpec) -> npt.NDArray[np.int64] | None:
        """Derive truth table by probing per-block/set variable existence.

        Reproduces the dynamic derivation in ``ex_get_truth_table.c`` when no
        explicit truth table variable is stored.  Returns a ``(num_loc, num_var)``
        int64 array, or ``None`` if there are no variables of this type.
        """
        num_var = self.dimension_size(spec.count_dimension, default=0)
        if num_var == 0:
            return None

        if spec.location_entity is None:
            return None

        if spec.location_entity.is_block:
            ids = self.block_ids(spec.location_entity)
        elif spec.location_entity.is_set:
            ids = self.set_ids(spec.location_entity)
        else:
            return None

        num_loc = len(ids)
        if num_loc == 0:
            return None

        table = np.zeros((num_loc, num_var), dtype=np.int64)
        for loc_idx in range(num_loc):
            for var_idx in range(num_var):
                # variable_index and location_index are both 1-based
                nc_var = spec.values_variable(var_idx + 1, loc_idx + 1)
                if self._backend.has_variable(nc_var):
                    table[loc_idx, var_idx] = 1

        return table

    def values(
        self,
        name: str,
        *,
        on: Entity | str,
        time: TimeSelector = None,
        block: int | None = None,
        block_id: int | None = None,
        set_id: int | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return result variable values."""

        ent = entity(on)

        if ent is Entity.GLOBAL:
            return self._global_values(name, time=time)
        if ent is Entity.NODE:
            return self._node_values(name, time=time)
        if ent in {Entity.ELEMENT, Entity.EDGE, Entity.FACE}:
            return self._object_block_values(
                name, on=ent, time=time, block_id=block_id if block_id is not None else block
            )
        if ent in {
            Entity.NODE_SET,
            Entity.SIDE_SET,
            Entity.EDGE_SET,
            Entity.FACE_SET,
            Entity.ELEMENT_SET,
        }:
            return self._set_values(name, on=ent, time=time, set_id=set_id)

        raise NotImplementedError(f"values for {ent.value!r} are not implemented yet")

    def attribute_names(self, on: Entity | str, block_id: int) -> tuple[str, ...]:
        """Return block attribute names."""

        ent = entity(on)
        if not ent.is_block:
            raise ExodusInvalidEntityError(f"{ent.value!r} is not a block entity")

        spec = block_spec(ent)
        if spec.attribute_names_variable is None:
            return ()

        block_index = self._block_index(ent, block_id)
        values = self._backend.variable(spec.attribute_names_variable(block_index), default=None)
        if values is None:
            return ()

        expected = self.dimension_size(
            spec.attributes_dimension(block_index) if spec.attributes_dimension is not None else "",
            default=0,
        )
        if expected == 0:
            return ()

        return tuple(name for name in _decode_name_table(values, expected_count=expected) if name)

    def attributes(self, on: Entity | str, block_id: int) -> npt.NDArray[np.float64] | None:
        """Return all block attributes as ``(entity_count, attribute_count)``."""

        ent = entity(on)
        if not ent.is_block:
            raise ExodusInvalidEntityError(f"{ent.value!r} is not a block entity")

        spec = block_spec(ent)
        if spec.attributes_variable is None:
            return None

        block_index = self._block_index(ent, block_id)
        values = self._backend.variable(spec.attributes_variable(block_index), default=None)
        if values is None:
            return None

        return np.asarray(values, dtype=np.float64)

    def attribute_values(
        self, on: Entity | str, block_id: int, name: str
    ) -> npt.NDArray[np.float64]:
        """Return one block attribute column by name."""

        names = self.attribute_names(on, block_id)
        requested = name.lower()

        for index, attr_name in enumerate(names):
            if attr_name.lower() == requested:
                attrs = self.attributes(on, block_id)
                if attrs is None:
                    return np.asarray([], dtype=np.float64)
                return np.asarray(attrs[:, index], dtype=np.float64)

        raise ExodusLookupError(f"attribute {name!r} not found")

    def _object_block_values(
        self, name: str, *, on: Entity, time: TimeSelector = None, block_id: int | None = None
    ) -> npt.NDArray[np.float64]:
        spec = variable_spec(on)
        variable_index = self._variable_index(on, name)

        if spec.location_entity is None:
            raise ExodusInvalidEntityError(f"{on.value!r} does not have block variable locations")

        if block_id is None:
            block_ids = self.block_ids(spec.location_entity)
            chunks = [
                self._object_block_values(name, on=on, time=time, block_id=int(current_id))
                for current_id in block_ids
            ]
            if not chunks:
                return np.asarray([], dtype=np.float64)
            return np.concatenate(chunks, axis=0 if time is not None else 1)

        block_index = self._block_index(spec.location_entity, block_id)
        variable_name = variable_value_name(on, variable_index, block_index)
        values = self._backend.variable(variable_name, default=None)

        if values is None:
            count = self.block(spec.location_entity, block_id).count
            if time is None:
                return np.zeros((len(self.times()), count), dtype=np.float64)
            return np.zeros(count, dtype=np.float64)

        array = np.asarray(values, dtype=np.float64)
        if time is None:
            return array

        selection = resolve_time(self.times(), time)
        return np.asarray(array[selection.index], dtype=np.float64)

    def _set_values(
        self, name: str, *, on: Entity, time: TimeSelector = None, set_id: int | None = None
    ) -> npt.NDArray[np.float64]:
        spec = variable_spec(on)
        variable_index = self._variable_index(on, name)

        if spec.location_entity is None:
            raise ExodusInvalidEntityError(f"{on.value!r} does not have set variable locations")

        if set_id is None:
            set_ids = self.set_ids(spec.location_entity)
            chunks = [
                self._set_values(name, on=on, time=time, set_id=int(current_id))
                for current_id in set_ids
            ]
            if not chunks:
                return np.asarray([], dtype=np.float64)
            return np.concatenate(chunks, axis=0 if time is not None else 1)

        index = self._set_index(spec.location_entity, set_id)
        variable_name = variable_value_name(on, variable_index, index)
        values = self._backend.variable(variable_name, default=None)

        set_info = self.set(spec.location_entity, set_id)
        if values is None:
            if time is None:
                return np.zeros((len(self.times()), set_info.count), dtype=np.float64)
            return np.zeros(set_info.count, dtype=np.float64)

        array = np.asarray(values, dtype=np.float64)
        if time is None:
            return array

        selection = resolve_time(self.times(), time)
        return np.asarray(array[selection.index], dtype=np.float64)

    def _block_index(self, on: Entity | str, block_id: int) -> int:
        spec = block_spec(on)
        ids = self.block_ids(spec.entity)
        matches = np.nonzero(ids == block_id)[0]
        if not len(matches):
            raise ExodusLookupError(f"{spec.entity.value} ID {block_id} not found")
        return int(matches[0]) + 1

    def _set_index(self, on: Entity | str, set_id: int) -> int:
        spec = set_spec(on)
        ids = self.set_ids(spec.entity)
        matches = np.nonzero(ids == set_id)[0]
        if not len(matches):
            raise ExodusLookupError(f"{spec.entity.value} ID {set_id} not found")
        return int(matches[0]) + 1

    def _global_values(self, name: str, *, time: TimeSelector = None) -> npt.NDArray[np.float64]:
        variable_index = self._variable_index(Entity.GLOBAL, name)
        values = np.asarray(
            self._backend.variable(VariableName.GLOBAL_VARIABLE_VALUES.value), dtype=np.float64
        )[:, variable_index - 1]

        if time is None:
            return values

        selection = resolve_time(self.times(), time)
        return np.asarray(values[selection.index], dtype=np.float64)

    def _node_values(self, name: str, *, time: TimeSelector = None) -> npt.NDArray[np.float64]:
        variable_index = self._variable_index(Entity.NODE, name)
        values = np.asarray(
            self._backend.variable(ExodusNames.node_variable(variable_index)), dtype=np.float64
        )

        if time is None:
            return values

        selection = resolve_time(self.times(), time)
        return np.asarray(values[selection.index], dtype=np.float64)

    def _element_values(
        self, name: str, *, time: TimeSelector = None, block: int | None = None
    ) -> npt.NDArray[np.float64]:
        return self._object_block_values(name, on=Entity.ELEMENT, time=time, block_id=block)

    def _variable_index(self, ent: Entity, name: str) -> int:
        cache_key = f"variable_index:{ent.value}:{name}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        names = self.variable_names(ent)
        index = _one_based_name_index(names, name)
        self._cache[cache_key] = index
        return index

    def _id_index(self, ent: Entity, id_value: int) -> int:
        ids = self.ids(ent)
        matches = np.nonzero(ids == id_value)[0]
        if not len(matches):
            raise ExodusLookupError(f"{ent.value} ID {id_value} not found")
        return int(matches[0]) + 1

    def _entity_name(self, ent: Entity, one_based_index: int) -> str:
        values = self._backend.variable(ExodusNames.names(ent), default=[])
        names = _decode_name_table(values)
        zero_based_index = one_based_index - 1

        if zero_based_index < 0 or zero_based_index >= len(names):
            return ""

        return names[zero_based_index]


def _one_based_name_index(names: tuple[str, ...], requested: str) -> int:
    for index, name in enumerate(names, start=1):
        if name == requested:
            return index

    requested_lower = requested.lower()
    for index, name in enumerate(names, start=1):
        if name.lower() == requested_lower:
            return index

    available = ", ".join(names)
    raise ExodusLookupError(f"variable {requested!r} not found; available variables: {available}")


def _four_fields(values: list[str]) -> tuple[str, str, str, str]:
    padded = values[:4] + [""] * max(0, 4 - len(values))
    return (padded[0], padded[1], padded[2], padded[3])


def _decode_name_table(values: Any, *, expected_count: int | None = None) -> tuple[str, ...]:
    """Decode an Exodus fixed-width name table.

    Handles raw NetCDF character arrays and already-decoded netCDF4 arrays.
    Legacy files may pad fixed-width strings with ASCII ``"0"`` bytes.
    """

    array = np.asarray(values)

    if (
        expected_count is not None
        and array.ndim == 1
        and len(array) == expected_count
        and array.dtype.kind in {"U", "S", "O"}
    ):
        return tuple(_strip_exodus_padding(decode_text(item)) for item in array)

    decoded = string_array(values)
    return tuple(_strip_exodus_padding(str(item)) for item in decoded)


def _strip_exodus_padding(value: str) -> str:
    text = value.rstrip(" \x00")

    # Some legacy files appear with fixed-width strings padded using ASCII "0"
    # characters.  Do not strip meaningful zeros from ordinary names like
    # "nodeset_100".  Only treat zeros as padding when the decoded field still
    # looks fixed-width.
    without_zeros = text.rstrip("0")
    zero_count = len(text) - len(without_zeros)

    if zero_count >= 2 and len(text) >= 16:
        return without_zeros

    return text


__all__ = ["ExodusFile"]
