# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Legacy ExodusIIFile-compatible facade."""

import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import cast

import numpy as np
import numpy.typing as npt

from exodusii.api.file import ExodusFile
from exodusii.api.query import print_query
from exodusii.api.query import query
from exodusii.api.writer import ExodusWriter
from exodusii.core.entities import Entity
from exodusii.core.errors import ExodusInvalidModeError
from exodusii.core.errors import ExodusWriteError
from exodusii.core.names import DimensionName
from exodusii.core.names import ExodusNames
from exodusii.core.names import VariableName
from exodusii.core.schema import variable_spec


class ExodusIIFile:
    """Compatibility adapter preserving much of the historical ``exodusii_file`` API."""

    def __init__(self, filename: str | Path, mode: str = "r") -> None:
        if mode not in {"r", "w"}:
            raise ExodusInvalidModeError(f"invalid Exodus file mode {mode!r}")

        self.filename: str | Path = filename
        self.mode = mode

        self._reader: ExodusFile | None = None
        self._writer: ExodusWriter | None = None

        if mode == "r":
            self._reader = ExodusFile.open(filename, mode="r")
        else:
            self._writer = ExodusWriter.create(filename)

    @property
    def reader(self) -> ExodusFile:
        if self._reader is None:
            raise ExodusWriteError(f"{self.filename}: not readable")
        return self._reader

    @property
    def writer(self) -> ExodusWriter:
        if self._writer is None:
            raise ExodusWriteError(f"{self.filename}: not writable")
        return self._writer

    @property
    def fh(self) -> Any:
        """Raw netCDF4 dataset for legacy callers."""

        if self._reader is not None:
            backend = cast(Any, self._reader.backend)
            return backend.dataset

        return self.writer.backend.dataset

    @property
    def files(self) -> list[Any]:
        """Legacy list of raw file handles."""

        return [self.fh]

    def __contains__(self, name: str) -> bool:
        if self._reader is not None:
            return name in self._reader.variables()
        return name in self.writer.backend.variables()

    def __enter__(self) -> "ExodusIIFile":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
        if self._writer is not None:
            self._writer.close()

    def title(self) -> str:
        return self.reader.title

    def storage_type(self) -> str:
        return self.reader.storage_type

    def dimensions(self) -> tuple[str, ...]:
        return self.reader.dimensions()

    def variables(self) -> tuple[str, ...]:
        return self.reader.variables()

    def dimension_names(self) -> list[str]:
        return list(self.reader.dimensions())

    def variable_names(self) -> list[str]:
        return list(self.reader.variables())

    def get_dimension(self, name: str, default: int | None = None) -> int | None:
        return self.reader.backend.dimension(name, default)

    def get_variable(self, name: str, default: Any = None, raw: bool = False) -> Any:
        return self.reader.variable(name, default=default, raw=raw)

    def num_dimensions(self) -> int:
        return self.reader.dimension

    def num_nodes(self) -> int:
        return self.reader.node_count

    def num_edges(self) -> int:
        return self.reader.edge_count

    def num_faces(self) -> int:
        return self.reader.face_count

    def num_elems(self) -> int:
        return self.reader.element_count

    def num_blks(self) -> int:
        return self.reader.element_block_count

    def num_element_blocks(self) -> int:
        return self.num_blks()

    def num_elem_blk(self) -> int:
        return self.num_blks()

    def num_node_sets(self) -> int:
        return self.reader.node_set_count

    def num_side_sets(self) -> int:
        return self.reader.side_set_count

    def num_edge_blk(self) -> int:
        return self.reader.dimension_size(DimensionName.NUM_EDGE_BLOCKS.value, default=0)

    def num_edge_sets(self) -> int:
        return self.reader.dimension_size(DimensionName.NUM_EDGE_SETS.value, default=0)

    def num_face_blk(self) -> int:
        return self.reader.dimension_size(DimensionName.NUM_FACE_BLOCKS.value, default=0)

    def num_face_sets(self) -> int:
        return self.reader.dimension_size(DimensionName.NUM_FACE_SETS.value, default=0)

    def num_elem_sets(self) -> int:
        return self.reader.dimension_size(DimensionName.NUM_ELEMENT_SETS.value, default=0)

    def num_elem_maps(self) -> int:
        return self.reader.dimension_size(DimensionName.NUM_ELEMENT_MAPS.value, default=0)

    def num_node_maps(self) -> int:
        return self.reader.dimension_size(DimensionName.NUM_NODE_MAPS.value, default=0)

    def num_edge_maps(self) -> int:
        return self.reader.dimension_size(DimensionName.NUM_EDGE_MAPS.value, default=0)

    def num_face_maps(self) -> int:
        return self.reader.dimension_size(DimensionName.NUM_FACE_MAPS.value, default=0)

    def num_times(self) -> int:
        return len(self.get_times())

    def get_times(self) -> npt.NDArray[np.float64]:
        return self.reader.times()

    def get_time(self, time_step: int) -> float:
        return float(self.get_times()[time_step - 1])

    def get_time_step(self, target: float, pcttol: float = 1.0e-5) -> int:
        """Legacy API: return one-based nearest time-step index."""

        times = np.asarray(self.get_times(), dtype=np.float64)
        if times.size == 0:
            raise ValueError("no time steps found")

        index = int(np.abs(times - target).argmin())

        if abs(target) > 0.0:
            relerr = abs(float(times[index]) - target) / abs(target)
            if relerr > pcttol:
                import logging

                logging.warning("Solution time differs significantly from desired solution time.")

        return index + 1

    def get_coord_names(self) -> npt.NDArray[np.str_]:
        return self.reader.coordinate_names()

    def get_coords(self, time_step: int | None = None) -> npt.NDArray[np.float64]:
        if time_step is None:
            return self.reader.coordinates()
        return self.reader.coordinates(time=time_step - 1, displaced=True)

    def get_displ_variable_names(self, default: Any = None) -> tuple[str, ...] | Any:
        names = self.reader.displacement_variable_names()
        return names if names else default

    def get_displ(self, time_step: int, default: Any = None) -> npt.NDArray[np.float64] | Any:
        names = self.reader.displacement_variable_names()
        if not names:
            return default
        return self.reader.displacements(time=time_step - 1)

    def get_node_id_map(self) -> npt.NDArray[np.int64]:
        return self.reader.ids("node")

    def get_element_id_map(self) -> npt.NDArray[np.int64]:
        return self.reader.ids("element")

    def get_element_block_ids(self) -> npt.NDArray[np.int64]:
        return self.reader.element_block_ids()

    def get_element_block_id(self, block_iid: int) -> int:
        return int(self.get_element_block_ids()[block_iid - 1])

    def get_element_block_iid(self, block_id: int) -> int | None:
        return _one_based_index(self.get_element_block_ids(), block_id)

    def get_element_block(self, block_id: int) -> SimpleNamespace:
        block = self.reader.element_block(block_id)
        return SimpleNamespace(
            id=block.id,
            iid=block.index,
            elem_type=block.element_type,
            name=block.name,
            num_block_elems=block.count,
            num_elem_nodes=block.nodes_per_entity,
            num_elem_edges=block.edges_per_entity,
            num_elem_faces=block.faces_per_entity,
            num_elem_attrs=block.attributes,
        )

    def get_element_block_name(self, block_id: int) -> str:
        return str(self.get_element_block(block_id).name)

    def get_element_block_names(self) -> npt.NDArray[np.str_]:
        return np.asarray(
            [
                self.get_element_block_name(int(block_id))
                for block_id in self.get_element_block_ids()
            ]
        )

    def get_element_conn(self, block_id: int, **_: Any) -> npt.NDArray[np.int64]:
        return self.reader.element_connectivity(block_id)

    def num_elems_in_blk(self, block_id: int) -> int:
        return int(self.get_element_block(block_id).num_block_elems)

    def num_nodes_per_elem(self, block_id: int) -> int:
        return int(self.get_element_block(block_id).num_elem_nodes)

    def get_node_set_ids(self) -> npt.NDArray[np.int64]:
        return self.reader.node_set_ids()

    def get_node_set_id(self, set_iid: int) -> int:
        return int(self.get_node_set_ids()[set_iid - 1])

    def get_node_set_iid(self, set_id: int) -> int | None:
        return _one_based_index(self.get_node_set_ids(), set_id)

    def get_node_set(self, set_id: int) -> SimpleNamespace:
        info = self.reader.node_set(set_id)
        return SimpleNamespace(
            id=info.id,
            iid=info.index,
            name=info.name,
            num_nodes=info.count,
            num_dist_facts=info.distribution_factors,
            nodes=info.nodes,
            dist_facts=info.dist_facts,
        )

    def get_node_set_name(self, set_id: int) -> str:
        return str(self.get_node_set(set_id).name)

    def get_node_set_names(self) -> npt.NDArray[np.str_]:
        return np.asarray(
            [self.get_node_set_name(int(set_id)) for set_id in self.get_node_set_ids()]
        )

    def get_node_set_params(self, set_id: int) -> SimpleNamespace:
        info = self.get_node_set(set_id)
        return SimpleNamespace(num_nodes=info.num_nodes, num_dist_facts=info.num_dist_facts)

    def get_node_set_nodes(self, set_id: int) -> npt.NDArray[np.int64] | None:
        return self.get_node_set(set_id).nodes

    def get_node_set_dist_facts(self, set_id: int) -> npt.NDArray[np.float64] | None:
        return self.get_node_set(set_id).dist_facts

    def num_nodes_in_node_set(self, set_id: int) -> int:
        return int(self.get_node_set_params(set_id).num_nodes)

    def get_side_set_ids(self) -> npt.NDArray[np.int64]:
        return self.reader.side_set_ids()

    def get_side_set_iid(self, set_id: int) -> int | None:
        return _one_based_index(self.get_side_set_ids(), set_id)

    def get_side_set(self, set_id: int) -> SimpleNamespace:
        info = self.reader.side_set(set_id)
        return SimpleNamespace(
            id=info.id,
            iid=info.index,
            name=info.name,
            num_sides=info.count,
            num_dist_facts=info.distribution_factors,
            elems=info.elems,
            sides=info.sides,
            dist_facts=info.dist_facts,
        )

    def get_side_set_name(self, set_id: int) -> str:
        return str(self.get_side_set(set_id).name)

    def get_side_set_names(self) -> npt.NDArray[np.str_]:
        return np.asarray(
            [self.get_side_set_name(int(set_id)) for set_id in self.get_side_set_ids()]
        )

    def get_side_set_params(self, set_id: int) -> SimpleNamespace:
        info = self.get_side_set(set_id)
        return SimpleNamespace(num_sides=info.num_sides, num_dist_facts=info.num_dist_facts)

    def get_side_set_elems(self, set_id: int) -> npt.NDArray[np.int64] | None:
        return self.get_side_set(set_id).elems

    def get_side_set_sides(self, set_id: int) -> npt.NDArray[np.int64] | None:
        return self.get_side_set(set_id).sides

    def get_side_set_dist_facts(self, set_id: int) -> npt.NDArray[np.float64] | None:
        return self.get_side_set(set_id).dist_facts

    def num_sides_in_side_set(self, set_id: int) -> int:
        return int(self.get_side_set_params(set_id).num_sides)

    def get_global_variable_names(self) -> npt.NDArray[np.str_]:
        return np.asarray(self.reader.variable_names("global"))

    def get_node_variable_names(self) -> npt.NDArray[np.str_]:
        return np.asarray(self.reader.variable_names("node"))

    def get_element_variable_names(self) -> npt.NDArray[np.str_]:
        return np.asarray(self.reader.variable_names("element"))

    def get_global_variable_number(self) -> int:
        return len(self.get_global_variable_names())

    def get_node_variable_number(self) -> int:
        return len(self.get_node_variable_names())

    def get_element_variable_number(self) -> int:
        return len(self.get_element_variable_names())

    def get_all_global_variable_values(
        self, time_step: int | None = None
    ) -> npt.NDArray[np.float64] | None:
        values = self.reader.variable(VariableName.GLOBAL_VARIABLE_VALUES.value, default=None)
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float64)
        return array if time_step is None else array[time_step - 1]

    def get_global_variable_values(self, var_name: str) -> npt.NDArray[np.float64]:
        return self.reader.values(var_name, on="global")

    def get_global_variable_value(self, var_name: str, time_step: int) -> float:
        return float(self.get_global_variable_values(var_name)[time_step - 1])

    def get_node_variable_values(
        self, var_name: str, time_step: int | None = None
    ) -> npt.NDArray[np.float64]:
        return self.reader.values(
            var_name, on="node", time=None if time_step is None else time_step - 1
        )

    def get_element_variable_values(
        self, block_id: int | None, var_name: str, time_step: int | None = None
    ) -> npt.NDArray[np.float64]:
        return self.reader.values(
            var_name,
            on="element",
            block=block_id,
            time=None if time_step is None else time_step - 1,
        )

    def get_node_variable_history(self, var_name: str, node_id: int) -> npt.NDArray[np.float64]:
        values = self.get_node_variable_values(var_name, time_step=None)
        node_ids = self.get_node_id_map()
        index = _zero_based_index(node_ids, node_id)
        return values[:, index]

    def get_element_variable_history(self, var_name: str, elem_id: int) -> npt.NDArray[np.float64]:
        element_ids = self.get_element_id_map()
        global_index = _zero_based_index(element_ids, elem_id)

        start = 0
        for block_id in self.get_element_block_ids():
            count = self.num_elems_in_blk(int(block_id))
            stop = start + count
            if start <= global_index < stop:
                values = self.get_element_variable_values(int(block_id), var_name, time_step=None)
                return values[:, global_index - start]
            start = stop

        raise ValueError(f"unable to determine element block for element {elem_id}")

    def get_edge_id_map(self) -> npt.NDArray[np.int64]:
        return self.reader.ids(Entity.EDGE)

    def get_face_id_map(self) -> npt.NDArray[np.int64]:
        return self.reader.ids(Entity.FACE)

    def get_edge_variable_names(self) -> npt.NDArray[np.str_]:
        return self._legacy_variable_names(Entity.EDGE)

    def get_face_variable_names(self) -> npt.NDArray[np.str_]:
        return self._legacy_variable_names(Entity.FACE)

    def get_node_set_variable_names(self) -> npt.NDArray[np.str_]:
        return self._legacy_variable_names(Entity.NODE_SET)

    def get_side_set_variable_names(self) -> npt.NDArray[np.str_]:
        return self._legacy_variable_names(Entity.SIDE_SET)

    def get_edge_set_variable_names(self) -> npt.NDArray[np.str_]:
        return self._legacy_variable_names(Entity.EDGE_SET)

    def get_face_set_variable_names(self) -> npt.NDArray[np.str_]:
        return self._legacy_variable_names(Entity.FACE_SET)

    def get_element_set_variable_names(self) -> npt.NDArray[np.str_]:
        return self._legacy_variable_names(Entity.ELEMENT_SET)

    def get_edge_variable_number(self) -> int:
        return self._legacy_variable_number(Entity.EDGE)

    def get_face_variable_number(self) -> int:
        return self._legacy_variable_number(Entity.FACE)

    def get_node_set_variable_number(self) -> int:
        return self._legacy_variable_number(Entity.NODE_SET)

    def get_side_set_variable_number(self) -> int:
        return self._legacy_variable_number(Entity.SIDE_SET)

    def get_edge_set_variable_number(self) -> int:
        return self._legacy_variable_number(Entity.EDGE_SET)

    def get_face_set_variable_number(self) -> int:
        return self._legacy_variable_number(Entity.FACE_SET)

    def get_element_set_variable_number(self) -> int:
        return self._legacy_variable_number(Entity.ELEMENT_SET)

    def get_edge_block_ids(self) -> npt.NDArray[np.int64]:
        return self.reader.edge_block_ids()

    def get_face_block_ids(self) -> npt.NDArray[np.int64]:
        return self.reader.face_block_ids()

    def get_edge_block_iid(self, block_id: int) -> int | None:
        return (
            self.reader._block_index(Entity.EDGE_BLOCK, block_id)
            if block_id in self.get_edge_block_ids()
            else None
        )

    def get_face_block_iid(self, block_id: int) -> int | None:
        return (
            self.reader._block_index(Entity.FACE_BLOCK, block_id)
            if block_id in self.get_face_block_ids()
            else None
        )

    def get_edge_block(self, block_id: int) -> SimpleNamespace:
        block = self.reader.edge_block(block_id)
        return SimpleNamespace(
            id=block.id,
            iid=block.index,
            elem_type=block.element_type,
            name=block.name,
            num_block_edges=block.count,
            num_edge_nodes=block.nodes_per_entity,
            num_edge_attrs=block.attributes,
        )

    def get_face_block(self, block_id: int) -> SimpleNamespace:
        block = self.reader.face_block(block_id)
        return SimpleNamespace(
            id=block.id,
            iid=block.index,
            elem_type=block.element_type,
            name=block.name,
            num_block_faces=block.count,
            num_face_nodes=block.nodes_per_entity,
            num_face_attrs=block.attributes,
        )

    def get_edge_block_conn(self, block_id: int) -> npt.NDArray[np.int64]:
        return self.reader.edge_connectivity(block_id)

    def get_face_block_conn(self, block_id: int) -> npt.NDArray[np.int64]:
        return self.reader.face_connectivity(block_id)

    def num_edges_in_blk(self, block_id: int) -> int:
        return int(self.get_edge_block(block_id).num_block_edges)

    def num_faces_in_blk(self, block_id: int) -> int:
        return int(self.get_face_block(block_id).num_block_faces)

    def num_nodes_per_edge(self, block_id: int) -> int:
        return int(self.get_edge_block(block_id).num_edge_nodes)

    def num_nodes_per_face(self, block_id: int) -> int:
        return int(self.get_face_block(block_id).num_face_nodes)

    def get_edge_set_ids(self) -> npt.NDArray[np.int64]:
        return self.reader.edge_set_ids()

    def get_face_set_ids(self) -> npt.NDArray[np.int64]:
        return self.reader.face_set_ids()

    def get_element_set_ids(self) -> npt.NDArray[np.int64]:
        return self.reader.element_set_ids()

    def get_edge_set(self, set_id: int) -> SimpleNamespace:
        info = self.reader.edge_set(set_id)
        return SimpleNamespace(
            id=info.id,
            iid=info.index,
            name=info.name,
            num_edges=info.count,
            num_dist_facts=info.distribution_factors,
            edges=info.entries,
            orientations=info.extra_entries,
            dist_facts=info.dist_facts,
        )

    def get_face_set(self, set_id: int) -> SimpleNamespace:
        info = self.reader.face_set(set_id)
        return SimpleNamespace(
            id=info.id,
            iid=info.index,
            name=info.name,
            num_faces=info.count,
            num_dist_facts=info.distribution_factors,
            faces=info.entries,
            orientations=info.extra_entries,
            dist_facts=info.dist_facts,
        )

    def get_element_set(self, set_id: int) -> SimpleNamespace:
        info = self.reader.element_set(set_id)
        return SimpleNamespace(
            id=info.id,
            iid=info.index,
            name=info.name,
            num_elems=info.count,
            num_dist_facts=info.distribution_factors,
            elems=info.entries,
            dist_facts=info.dist_facts,
        )

    def get_element_variable_truth_table(self, block_id: int | None = None):
        return self._legacy_variable_truth_table(Entity.ELEMENT, block_id)

    def get_edge_variable_truth_table(self, block_id: int | None = None):
        return self._legacy_variable_truth_table(Entity.EDGE, block_id)

    def get_face_variable_truth_table(self, block_id: int | None = None):
        return self._legacy_variable_truth_table(Entity.FACE, block_id)

    def get_node_set_variable_truth_table(self, set_id: int | None = None):
        return self._legacy_variable_truth_table(Entity.NODE_SET, set_id)

    def get_side_set_variable_truth_table(self, set_id: int | None = None):
        return self._legacy_variable_truth_table(Entity.SIDE_SET, set_id)

    def get_edge_set_variable_truth_table(self, set_id: int | None = None):
        return self._legacy_variable_truth_table(Entity.EDGE_SET, set_id)

    def get_face_set_variable_truth_table(self, set_id: int | None = None):
        return self._legacy_variable_truth_table(Entity.FACE_SET, set_id)

    def get_element_set_variable_truth_table(self, set_id: int | None = None):
        return self._legacy_variable_truth_table(Entity.ELEMENT_SET, set_id)

    def get_edge_variable_values(
        self, block_id: int | None, var_name: str, time_step: int | None = None
    ):
        return self._legacy_variable_values(Entity.EDGE, block_id, var_name, time_step)

    def get_face_variable_values(
        self, block_id: int | None, var_name: str, time_step: int | None = None
    ):
        return self._legacy_variable_values(Entity.FACE, block_id, var_name, time_step)

    def get_node_set_variable_values(
        self, set_id: int | None, var_name: str, time_step: int | None = None
    ):
        return self._legacy_variable_values(Entity.NODE_SET, set_id, var_name, time_step)

    def get_side_set_variable_values(
        self, set_id: int | None, var_name: str, time_step: int | None = None
    ):
        return self._legacy_variable_values(Entity.SIDE_SET, set_id, var_name, time_step)

    def get_edge_set_variable_values(
        self, set_id: int | None, var_name: str, time_step: int | None = None
    ):
        return self._legacy_variable_values(Entity.EDGE_SET, set_id, var_name, time_step)

    def get_face_set_variable_values(
        self, set_id: int | None, var_name: str, time_step: int | None = None
    ):
        return self._legacy_variable_values(Entity.FACE_SET, set_id, var_name, time_step)

    def get_element_set_variable_values(
        self, set_id: int | None, var_name: str, time_step: int | None = None
    ):
        return self._legacy_variable_values(Entity.ELEMENT_SET, set_id, var_name, time_step)

    def get_iid(self, container: npt.ArrayLike, item: Any) -> int | None:
        """Legacy API: return one-based index of item in container."""

        matches = np.nonzero(np.asarray(container) == item)[0]
        return None if not len(matches) else int(matches[0]) + 1

    def get_info_records(self) -> list[str] | None:
        """Legacy API: information records."""

        records = self.reader.info_records()
        return [record[:80] for record in records] if records else None

    def get_qa_records(self):
        """Legacy API: QA records."""

        records = self.reader.qa_records()
        return list(records) if records else None

    def put_info(self, num_info: int, info: npt.ArrayLike) -> None:
        """Legacy API: write information records."""

        del num_info
        self.writer.write_info_records(_string_list(info))

    def put_qa(self, num_qa_records: int, qa_records: npt.ArrayLike) -> None:
        """Legacy API: write QA records."""

        del num_qa_records
        records = np.asarray(qa_records, dtype=object)
        if records.ndim == 1:
            records = records.reshape(1, -1)
        self.writer.write_qa_records(records.tolist())

    def get_coord_variable_names(self) -> list[str]:
        """Legacy API: coordinate variable names."""

        return [ExodusNames.coordinate(axis) for axis in range(self.num_dimensions())]

    def is_global_variable(self, variable: str) -> bool:
        return variable in self.get_global_variable_names().tolist()

    def is_node_variable(self, variable: str) -> bool:
        return variable in self.get_node_variable_names().tolist()

    def is_element_variable(self, variable: str) -> bool:
        return variable in self.get_element_variable_names().tolist()

    def is_edge_variable(self, variable: str) -> bool:
        return False

    def is_face_variable(self, variable: str) -> bool:
        return False

    def get_element_property_names(self) -> list[str]:
        return list(self.reader.property_names(Entity.ELEMENT_BLOCK))

    def get_edge_property_names(self) -> list[str]:
        return list(self.reader.property_names(Entity.EDGE_BLOCK))

    def get_face_property_names(self) -> list[str]:
        return list(self.reader.property_names(Entity.FACE_BLOCK))

    def get_node_set_property_names(self) -> list[str]:
        return list(self.reader.property_names(Entity.NODE_SET))

    def get_side_set_property_names(self) -> list[str]:
        return list(self.reader.property_names(Entity.SIDE_SET))

    def get_edge_set_property_names(self) -> list[str]:
        return list(self.reader.property_names(Entity.EDGE_SET))

    def get_face_set_property_names(self) -> list[str]:
        return list(self.reader.property_names(Entity.FACE_SET))

    def get_element_set_property_names(self) -> list[str]:
        return list(self.reader.property_names(Entity.ELEMENT_SET))

    def get_element_property_value(self, block_id: int, name: str) -> int:
        return self.reader.property_value(Entity.ELEMENT_BLOCK, block_id, name)

    def get_edge_property_value(self, block_id: int, name: str) -> int:
        return self.reader.property_value(Entity.EDGE_BLOCK, block_id, name)

    def get_face_property_value(self, block_id: int, name: str) -> int:
        return self.reader.property_value(Entity.FACE_BLOCK, block_id, name)

    def get_node_set_property_value(self, set_id: int, name: str) -> int:
        return self.reader.property_value(Entity.NODE_SET, set_id, name)

    def get_side_set_property_value(self, set_id: int, name: str) -> int:
        return self.reader.property_value(Entity.SIDE_SET, set_id, name)

    def get_edge_set_property_value(self, set_id: int, name: str) -> int:
        return self.reader.property_value(Entity.EDGE_SET, set_id, name)

    def get_face_set_property_value(self, set_id: int, name: str) -> int:
        return self.reader.property_value(Entity.FACE_SET, set_id, name)

    def get_element_set_property_value(self, set_id: int, name: str) -> int:
        return self.reader.property_value(Entity.ELEMENT_SET, set_id, name)

    def put_element_property(self, name: str, values: npt.ArrayLike) -> None:
        self.writer.define_property(Entity.ELEMENT_BLOCK, name, values)

    def put_node_set_property(self, name: str, values: npt.ArrayLike) -> None:
        self.writer.define_property(Entity.NODE_SET, name, values)

    def put_side_set_property(self, name: str, values: npt.ArrayLike) -> None:
        self.writer.define_property(Entity.SIDE_SET, name, values)

    def put_edge_property(self, name: str, values: npt.ArrayLike) -> None:
        self.writer.define_property(Entity.EDGE_BLOCK, name, values)

    def put_face_property(self, name: str, values: npt.ArrayLike) -> None:
        self.writer.define_property(Entity.FACE_BLOCK, name, values)

    def put_edge_set_property(self, name: str, values: npt.ArrayLike) -> None:
        self.writer.define_property(Entity.EDGE_SET, name, values)

    def put_face_set_property(self, name: str, values: npt.ArrayLike) -> None:
        self.writer.define_property(Entity.FACE_SET, name, values)

    def put_element_set_property(self, name: str, values: npt.ArrayLike) -> None:
        self.writer.define_property(Entity.ELEMENT_SET, name, values)

    def get_variable_type(self, name: str) -> str | None:
        """Legacy API: infer variable type code."""

        if self.is_global_variable(name):
            return "g"
        if self.is_element_variable(name):
            return "e"
        if self.is_node_variable(name):
            return "n"
        if self.is_edge_variable(name):
            return "d"
        if self.is_face_variable(name):
            return "f"
        return None

    def get_variable_values(
        self, type: str, name: str, time_step: int | None = None
    ) -> npt.NDArray[np.float64] | None:
        """Legacy API: generic variable values."""

        if type == "g":
            values = self.get_global_variable_values(name)
            return values if time_step is None else np.asarray(values[time_step - 1])
        if type == "n":
            return self.get_node_variable_values(name, time_step=time_step)
        if type == "e":
            return self.get_element_variable_values(None, name, time_step=time_step)
        return None

    # ------------------------------------------------------------------
    # Legacy block attribute methods
    # ------------------------------------------------------------------

    def get_element_attr(self, block_id: int):
        return self.reader.attributes(Entity.ELEMENT_BLOCK, block_id)

    def get_edge_attr(self, block_id: int):
        return self.reader.attributes(Entity.EDGE_BLOCK, block_id)

    def get_face_attr(self, block_id: int):
        return self.reader.attributes(Entity.FACE_BLOCK, block_id)

    def get_element_attribute_names(self, block_id: int) -> list[str]:
        return list(self.reader.attribute_names(Entity.ELEMENT_BLOCK, block_id))

    def get_edge_attribute_names(self, block_id: int) -> list[str]:
        return list(self.reader.attribute_names(Entity.EDGE_BLOCK, block_id))

    def get_face_attribute_names(self, block_id: int) -> list[str]:
        return list(self.reader.attribute_names(Entity.FACE_BLOCK, block_id))

    def get_element_attr_values(self, block_id: int, elem_attr_name: str):
        return self.reader.attribute_values(Entity.ELEMENT_BLOCK, block_id, elem_attr_name)

    def get_edge_attr_values(self, block_id: int, edge_attr_name: str):
        return self.reader.attribute_values(Entity.EDGE_BLOCK, block_id, edge_attr_name)

    def get_face_attr_values(self, block_id: int, face_attr_name: str):
        return self.reader.attribute_values(Entity.FACE_BLOCK, block_id, face_attr_name)

    def put_element_attr(self, block_id: int, attr: npt.ArrayLike) -> None:
        self.writer.write_block_attributes(Entity.ELEMENT_BLOCK, block_id, attr)

    def put_edge_attr(self, block_id: int, attr: npt.ArrayLike) -> None:
        self.writer.write_block_attributes(Entity.EDGE_BLOCK, block_id, attr)

    def put_face_attr(self, block_id: int, attr: npt.ArrayLike) -> None:
        self.writer.write_block_attributes(Entity.FACE_BLOCK, block_id, attr)

    def put_element_attribute_names(self, block_id: int, names: Any) -> None:
        block_index = self.writer._element_block_indices[block_id]
        attr_var = f"attrib{block_index}"
        count = self.writer._dimension_size(ExodusNames.block_count(block_index))
        attr_names = _string_list(names)

        if self.writer.backend.has_variable(attr_var):
            attrs = np.asarray(self.writer.backend.variable(attr_var), dtype=np.float64)
        else:
            attrs = np.zeros((count, len(attr_names)), dtype=np.float64)

        self.writer.write_block_attributes(Entity.ELEMENT_BLOCK, block_id, attrs, names=attr_names)

    def put_edge_attribute_names(self, block_id: int, names: Any) -> None:
        block_index = self.writer._edge_block_indices[block_id]
        attr_var = f"eattrb{block_index}"
        count = self.writer._dimension_size(ExodusNames.edge_block_count(block_index))
        attr_names = _string_list(names)

        if self.writer.backend.has_variable(attr_var):
            attrs = np.asarray(self.writer.backend.variable(attr_var), dtype=np.float64)
        else:
            attrs = np.zeros((count, len(attr_names)), dtype=np.float64)

        self.writer.write_block_attributes(Entity.EDGE_BLOCK, block_id, attrs, names=attr_names)

    def put_face_attribute_names(self, block_id: int, names: Any) -> None:
        block_index = self.writer._face_block_indices[block_id]
        attr_var = f"fattrb{block_index}"
        count = self.writer._dimension_size(ExodusNames.face_block_count(block_index))
        attr_names = _string_list(names)

        if self.writer.backend.has_variable(attr_var):
            attrs = np.asarray(self.writer.backend.variable(attr_var), dtype=np.float64)
        else:
            attrs = np.zeros((count, len(attr_names)), dtype=np.float64)

        self.writer.write_block_attributes(Entity.FACE_BLOCK, block_id, attrs, names=attr_names)

    def get(
        self,
        *variables: str,
        time: float | None = None,
        index: int | None = None,
        cycle: int | None = None,
        lineout: Any = None,
    ):
        """Legacy API: return structured query table."""

        if cycle is not None:
            raise NotImplementedError("cycle selection is not implemented in the refreshed adapter")

        selector_time = index if index is not None else time
        return query(self.reader, *variables, time=selector_time, lineout=lineout).data

    def put_init(
        self,
        title: str,
        num_dim: int,
        num_nodes: int,
        num_elem: int,
        num_elem_blk: int,
        num_node_sets: int,
        num_side_sets: int,
        **kwargs: Any,
    ) -> None:
        self.writer.initialize(
            title,
            num_dim,
            num_nodes,
            num_elem,
            element_blocks=num_elem_blk,
            node_sets=num_node_sets,
            side_sets=num_side_sets,
            edge_count=int(kwargs.get("num_edge") or 0),
            edge_blocks=int(kwargs.get("num_edge_blk") or 0),
            edge_sets=int(kwargs.get("num_edge_sets") or kwargs.get("num_edge_set") or 0),
            face_count=int(kwargs.get("num_face") or 0),
            face_blocks=int(kwargs.get("num_face_blk") or 0),
            face_sets=int(kwargs.get("num_face_sets") or kwargs.get("num_face_set") or 0),
            element_sets=int(kwargs.get("num_elem_sets") or kwargs.get("num_element_sets") or 0),
        )

    def put_coord(self, *coords: npt.ArrayLike) -> None:
        self.writer.write_coordinates(np.column_stack(coords))

    def put_coords(self, coords: npt.ArrayLike) -> None:
        self.writer.write_coordinates(coords)

    def put_coord_names(self, coord_names: Any) -> None:
        coords = _current_coordinates(self.writer)
        self.writer.write_coordinates(coords, names=_string_list(coord_names))

    def put_element_block(
        self, block_id: int, elem_type: str, num_block_elems: int, num_nodes_per_elem: int, **_: Any
    ) -> None:
        zeros = np.zeros((num_block_elems, num_nodes_per_elem), dtype=np.int64)
        self.writer.define_element_block(block_id, elem_type, zeros)

    def put_element_block_name(self, block_id: int, name: str) -> None:
        block_index = self.writer._element_block_indices[block_id]
        self.writer.backend.write_variable(
            VariableName.ELEMENT_BLOCK_NAMES.value, _fixed_name(name), block_index - 1
        )

    def put_element_block_names(self, names: Any) -> None:
        self.writer.backend.write_variable(
            VariableName.ELEMENT_BLOCK_NAMES.value, _fixed_names(_string_list(names))
        )

    def put_element_conn(self, block_id: int, connect: npt.ArrayLike, **_: Any) -> None:
        block_index = self.writer._element_block_indices[block_id]
        self.writer.backend.write_variable(
            ExodusNames.element_connectivity(block_index), np.asarray(connect, dtype=np.int64)
        )

    def put_node_set_param(
        self, set_id: int, num_nodes_in_set: int, num_dist_fact_in_set: int = 0
    ) -> None:
        nodes = np.zeros(num_nodes_in_set, dtype=np.int64)
        factors = np.ones(num_dist_fact_in_set, dtype=np.float64) if num_dist_fact_in_set else None
        self.writer.define_node_set(set_id, nodes, distribution_factors=factors)

    def put_node_set_name(self, set_id: int, name: str) -> None:
        set_index = self.writer._node_set_indices[set_id]
        self.writer.backend.write_variable(
            VariableName.NODE_SET_NAMES.value, _fixed_name(name), set_index - 1
        )

    def put_node_set_nodes(self, set_id: int, node_set_nodes: npt.ArrayLike) -> None:
        set_index = self.writer._node_set_indices[set_id]
        self.writer.backend.write_variable(
            ExodusNames.node_set_nodes(set_index), np.asarray(node_set_nodes, dtype=np.int64)
        )

    def put_node_set_dist_fact(self, set_id: int, node_set_dist_fact: npt.ArrayLike) -> None:
        set_index = self.writer._node_set_indices[set_id]
        self.writer.backend.write_variable(
            ExodusNames.node_set_distribution_factors(set_index),
            np.asarray(node_set_dist_fact, dtype=np.float64),
        )

    def put_side_set_param(
        self, set_id: int, num_sides_in_set: int, num_dist_fact_in_set: int = 0
    ) -> None:
        elements = np.zeros(num_sides_in_set, dtype=np.int64)
        sides = np.zeros(num_sides_in_set, dtype=np.int64)
        factors = np.ones(num_dist_fact_in_set, dtype=np.float64) if num_dist_fact_in_set else None
        self.writer.define_side_set(set_id, elements, sides, distribution_factors=factors)

    def put_side_set_name(self, set_id: int, name: str) -> None:
        set_index = self.writer._side_set_indices[set_id]
        self.writer.backend.write_variable(
            VariableName.SIDE_SET_NAMES.value, _fixed_name(name), set_index - 1
        )

    def put_side_set_sides(
        self, set_id: int, side_set_elems: npt.ArrayLike, side_set_sides: npt.ArrayLike
    ) -> None:
        set_index = self.writer._side_set_indices[set_id]
        self.writer.backend.write_variable(
            ExodusNames.side_set_elements(set_index), np.asarray(side_set_elems, dtype=np.int64)
        )
        self.writer.backend.write_variable(
            ExodusNames.side_set_sides(set_index), np.asarray(side_set_sides, dtype=np.int64)
        )

    def put_side_set_dist_fact(self, set_id: int, side_set_dist_fact: npt.ArrayLike) -> None:
        set_index = self.writer._side_set_indices[set_id]
        self.writer.backend.write_variable(
            ExodusNames.side_set_distribution_factors(set_index),
            np.asarray(side_set_dist_fact, dtype=np.float64),
        )

    def put_time(self, time_step: int, time_value: float) -> None:
        self.writer.write_time(time_value, step=time_step)

    def put_global_variable_params(self, num_vars: int) -> None:
        self.writer.define_global_variables([""] * num_vars)

    def put_global_variable_names(self, names: Any) -> None:
        self.writer.backend.write_variable(
            VariableName.GLOBAL_VARIABLE_NAMES.value, _fixed_names(_string_list(names))
        )

    def put_global_variable_values(
        self, time_step: int | None, vals_glo_var: npt.ArrayLike
    ) -> None:
        if time_step is None:
            self.writer.backend.write_variable(
                VariableName.GLOBAL_VARIABLE_VALUES.value,
                np.asarray(vals_glo_var, dtype=np.float64),
            )
        else:
            self.writer.write_global_values(vals_glo_var, step=time_step)

    def put_node_variable_params(self, num_vars: int) -> None:
        self.writer.define_node_variables([""] * num_vars)

    def put_node_variable_names(self, names: Any) -> None:
        self.writer.backend.write_variable(
            VariableName.NODE_VARIABLE_NAMES.value, _fixed_names(_string_list(names))
        )

    def put_node_variable_values(
        self, time_step: int | None, name: str, values: npt.ArrayLike
    ) -> None:
        if time_step is None:
            index = self.writer._name_index(VariableName.NODE_VARIABLE_NAMES.value, name)
            self.writer.backend.write_variable(
                ExodusNames.node_variable(index), np.asarray(values, dtype=np.float64)
            )
        else:
            self.writer.write_node_values(name, values, step=time_step)

    def put_element_variable_params(self, num_vars: int) -> None:
        self.writer.define_element_variables([""] * num_vars)

    def put_element_variable_names(self, names: Any) -> None:
        self.writer.backend.write_variable(
            VariableName.ELEMENT_VARIABLE_NAMES.value, _fixed_names(_string_list(names))
        )

    def put_element_variable_values(
        self, time_step: int | None, block_id: int, name: str, values: npt.ArrayLike
    ) -> None:
        if time_step is None:
            index = self.writer._name_index(VariableName.ELEMENT_VARIABLE_NAMES.value, name)
            block_index = self.writer._element_block_indices[block_id]
            self.writer.backend.write_variable(
                ExodusNames.element_variable(index, block_index),
                np.asarray(values, dtype=np.float64),
            )
        else:
            self.writer.write_element_values(name, values, block_id=block_id, step=time_step)

    def put_edge_variable_params(self, num_vars: int) -> None:
        self._put_variable_params(Entity.EDGE, num_vars)

    def put_face_variable_params(self, num_vars: int) -> None:
        self._put_variable_params(Entity.FACE, num_vars)

    def put_node_set_variable_params(self, num_vars: int) -> None:
        self._put_variable_params(Entity.NODE_SET, num_vars)

    def put_side_set_variable_params(self, num_vars: int) -> None:
        self._put_variable_params(Entity.SIDE_SET, num_vars)

    def put_edge_set_variable_params(self, num_vars: int) -> None:
        self._put_variable_params(Entity.EDGE_SET, num_vars)

    def put_face_set_variable_params(self, num_vars: int) -> None:
        self._put_variable_params(Entity.FACE_SET, num_vars)

    def put_element_set_variable_params(self, num_vars: int) -> None:
        self._put_variable_params(Entity.ELEMENT_SET, num_vars)

    def put_edge_variable_names(self, names: Any) -> None:
        self._put_variable_names(Entity.EDGE, names)

    def put_face_variable_names(self, names: Any) -> None:
        self._put_variable_names(Entity.FACE, names)

    def put_node_set_variable_names(self, names: Any) -> None:
        self._put_variable_names(Entity.NODE_SET, names)

    def put_side_set_variable_names(self, names: Any) -> None:
        self._put_variable_names(Entity.SIDE_SET, names)

    def put_edge_set_variable_names(self, names: Any) -> None:
        self._put_variable_names(Entity.EDGE_SET, names)

    def put_face_set_variable_names(self, names: Any) -> None:
        self._put_variable_names(Entity.FACE_SET, names)

    def put_element_set_variable_names(self, names: Any) -> None:
        self._put_variable_names(Entity.ELEMENT_SET, names)

    def put_element_variable_truth_table(self, table: npt.ArrayLike) -> None:
        self._put_variable_truth_table(Entity.ELEMENT, table)

    def put_edge_variable_truth_table(self, table: npt.ArrayLike) -> None:
        self._put_variable_truth_table(Entity.EDGE, table)

    def put_face_variable_truth_table(self, table: npt.ArrayLike) -> None:
        self._put_variable_truth_table(Entity.FACE, table)

    def put_node_set_variable_truth_table(self, table: npt.ArrayLike) -> None:
        self._put_variable_truth_table(Entity.NODE_SET, table)

    def put_side_set_variable_truth_table(self, table: npt.ArrayLike) -> None:
        self._put_variable_truth_table(Entity.SIDE_SET, table)

    def put_edge_set_variable_truth_table(self, table: npt.ArrayLike) -> None:
        self._put_variable_truth_table(Entity.EDGE_SET, table)

    def put_face_set_variable_truth_table(self, table: npt.ArrayLike) -> None:
        self._put_variable_truth_table(Entity.FACE_SET, table)

    def put_element_set_variable_truth_table(self, table: npt.ArrayLike) -> None:
        self._put_variable_truth_table(Entity.ELEMENT_SET, table)

    def put_edge_variable_values(
        self, time_step: int, block_id: int, name: str, values: npt.ArrayLike
    ) -> None:
        self.writer.write_values(name, values, on=Entity.EDGE, block_id=block_id, step=time_step)

    def put_face_variable_values(
        self, time_step: int, block_id: int, name: str, values: npt.ArrayLike
    ) -> None:
        self.writer.write_values(name, values, on=Entity.FACE, block_id=block_id, step=time_step)

    def put_node_set_variable_values(
        self, time_step: int, set_id: int, name: str, values: npt.ArrayLike
    ) -> None:
        self.writer.write_values(name, values, on=Entity.NODE_SET, set_id=set_id, step=time_step)

    def put_side_set_variable_values(
        self, time_step: int, set_id: int, name: str, values: npt.ArrayLike
    ) -> None:
        self.writer.write_values(name, values, on=Entity.SIDE_SET, set_id=set_id, step=time_step)

    def put_edge_set_variable_values(
        self, time_step: int, set_id: int, name: str, values: npt.ArrayLike
    ) -> None:
        self.writer.write_values(name, values, on=Entity.EDGE_SET, set_id=set_id, step=time_step)

    def put_face_set_variable_values(
        self, time_step: int, set_id: int, name: str, values: npt.ArrayLike
    ) -> None:
        self.writer.write_values(name, values, on=Entity.FACE_SET, set_id=set_id, step=time_step)

    def put_element_set_variable_values(
        self, time_step: int, set_id: int, name: str, values: npt.ArrayLike
    ) -> None:
        self.writer.write_values(name, values, on=Entity.ELEMENT_SET, set_id=set_id, step=time_step)

    # ------------------------------------------------------------------
    # Legacy edge/face block write methods
    # ------------------------------------------------------------------

    def put_edge_block(
        self,
        block_id: int,
        elem_type: str,
        num_block_edges: int,
        num_nodes_per_edge: int,
        num_attr: int = 0,
    ) -> None:
        """Legacy API: define an edge block."""

        del num_attr
        conn = np.zeros((num_block_edges, num_nodes_per_edge), dtype=np.int64)
        self.writer.define_edge_block(block_id, elem_type, conn)

    def put_edge_conn(self, block_id: int, connect: npt.ArrayLike) -> None:
        """Legacy API: write edge-block connectivity."""

        block_index = self.writer._edge_block_indices[block_id]
        self.writer.backend.write_variable(
            ExodusNames.edge_connectivity(block_index), np.asarray(connect, dtype=np.int64)
        )

    def put_face_block(
        self,
        block_id: int,
        elem_type: str,
        num_block_faces: int,
        num_nodes_per_face: int,
        num_attr: int = 0,
    ) -> None:
        """Legacy API: define a face block."""

        del num_attr
        conn = np.zeros((num_block_faces, num_nodes_per_face), dtype=np.int64)
        self.writer.define_face_block(block_id, elem_type, conn)

    def put_face_conn(self, block_id: int, connect: npt.ArrayLike) -> None:
        """Legacy API: write face-block connectivity."""

        block_index = self.writer._face_block_indices[block_id]
        self.writer.backend.write_variable(
            ExodusNames.face_connectivity(block_index), np.asarray(connect, dtype=np.int64)
        )

    def put_node_id_map(self, node_num_map: npt.ArrayLike) -> None:
        """Legacy API: write node ID map."""

        self.writer.write_node_id_map(node_num_map)

    def put_element_id_map(self, elem_num_map: npt.ArrayLike) -> None:
        """Legacy API: write element ID map."""

        self.writer.write_element_id_map(elem_num_map)

    def put_edge_id_map(self, edge_num_map: npt.ArrayLike) -> None:
        """Legacy API: write edge ID map."""

        self.writer.write_edge_id_map(edge_num_map)

    def put_face_id_map(self, face_num_map: npt.ArrayLike) -> None:
        """Legacy API: write face ID map."""

        self.writer.write_face_id_map(face_num_map)

    # ------------------------------------------------------------------
    # Legacy edge/face/element set write methods
    # ------------------------------------------------------------------

    def put_edge_set_param(
        self, set_id: int, num_edges: int, num_nodes_per_edge: int = 0, num_dist_facts: int = 0
    ) -> None:
        """Legacy API: define an edge set."""

        del num_nodes_per_edge
        edges = np.zeros(num_edges, dtype=np.int64)
        orientations = np.ones(num_edges, dtype=np.int64)
        factors = np.ones(num_dist_facts, dtype=np.float64) if num_dist_facts else None
        self.writer.define_edge_set(
            set_id, edges, orientations=orientations, distribution_factors=factors
        )

    def put_edge_set_edges(
        self, set_id: int, edges: npt.ArrayLike, orientations: npt.ArrayLike | None = None
    ) -> None:
        """Legacy API: write edge-set entries."""

        set_index = self.writer._edge_set_indices[set_id]
        self.writer.backend.write_variable(f"edge_es{set_index}", np.asarray(edges, dtype=np.int64))
        if orientations is not None:
            self.writer.backend.write_variable(
                f"ornt_es{set_index}", np.asarray(orientations, dtype=np.int64)
            )

    def put_edge_set_dist_fact(self, set_id: int, dist_facts: npt.ArrayLike) -> None:
        """Legacy API: write edge-set distribution factors."""

        set_index = self.writer._edge_set_indices[set_id]
        self.writer.backend.write_variable(
            f"dist_fact_es{set_index}", np.asarray(dist_facts, dtype=np.float64)
        )

    def put_face_set_param(
        self, set_id: int, num_faces: int, num_nodes_per_face: int = 0, num_dist_facts: int = 0
    ) -> None:
        """Legacy API: define a face set."""

        del num_nodes_per_face
        faces = np.zeros(num_faces, dtype=np.int64)
        orientations = np.ones(num_faces, dtype=np.int64)
        factors = np.ones(num_dist_facts, dtype=np.float64) if num_dist_facts else None
        self.writer.define_face_set(
            set_id, faces, orientations=orientations, distribution_factors=factors
        )

    def put_face_set_faces(
        self, set_id: int, faces: npt.ArrayLike, orientations: npt.ArrayLike | None = None
    ) -> None:
        """Legacy API: write face-set entries."""

        set_index = self.writer._face_set_indices[set_id]
        self.writer.backend.write_variable(f"face_fs{set_index}", np.asarray(faces, dtype=np.int64))
        if orientations is not None:
            self.writer.backend.write_variable(
                f"ornt_fs{set_index}", np.asarray(orientations, dtype=np.int64)
            )

    def put_face_set_dist_fact(self, set_id: int, dist_facts: npt.ArrayLike) -> None:
        """Legacy API: write face-set distribution factors."""

        set_index = self.writer._face_set_indices[set_id]
        self.writer.backend.write_variable(
            f"dist_fact_fs{set_index}", np.asarray(dist_facts, dtype=np.float64)
        )

    def put_element_set_param(self, set_id: int, num_elems: int, num_dist_facts: int = 0) -> None:
        """Legacy API: define an element set."""

        elems = np.zeros(num_elems, dtype=np.int64)
        factors = np.ones(num_dist_facts, dtype=np.float64) if num_dist_facts else None
        self.writer.define_element_set(set_id, elems, distribution_factors=factors)

    def put_element_set_elems(self, set_id: int, elems: npt.ArrayLike) -> None:
        """Legacy API: write element-set entries."""

        set_index = self.writer._element_set_indices[set_id]
        self.writer.backend.write_variable(
            f"elem_els{set_index}", np.asarray(elems, dtype=np.int64)
        )

    def put_element_set_dist_fact(self, set_id: int, dist_facts: npt.ArrayLike) -> None:
        """Legacy API: write element-set distribution factors."""

        set_index = self.writer._element_set_indices[set_id]
        self.writer.backend.write_variable(
            f"dist_fact_els{set_index}", np.asarray(dist_facts, dtype=np.float64)
        )

    def put_edge_set_name(self, set_id: int, name: str) -> None:
        set_index = self.writer._edge_set_indices[set_id]
        self.writer.backend.write_variable(
            VariableName.EDGE_SET_NAMES.value, _fixed_name(name), set_index - 1
        )

    def put_face_set_name(self, set_id: int, name: str) -> None:
        set_index = self.writer._face_set_indices[set_id]
        self.writer.backend.write_variable(
            VariableName.FACE_SET_NAMES.value, _fixed_name(name), set_index - 1
        )

    def put_element_set_name(self, set_id: int, name: str) -> None:
        set_index = self.writer._element_set_indices[set_id]
        self.writer.backend.write_variable(
            VariableName.ELEMENT_SET_NAMES.value, _fixed_name(name), set_index - 1
        )

    def _put_variable_params(self, on: Entity, num_vars: int) -> None:
        self.writer.define_variables(on, [""] * num_vars)

    def _put_variable_names(self, on: Entity, names: Any) -> None:
        spec = variable_spec(on)
        self.writer.backend.write_variable(spec.names_variable, _fixed_names(_string_list(names)))

    def _put_variable_truth_table(self, on: Entity, table: npt.ArrayLike) -> None:
        spec = variable_spec(on)
        if spec.truth_table_variable is None:
            return
        self.writer.backend.write_variable(
            spec.truth_table_variable, np.asarray(table, dtype=np.int64)
        )

    def print(
        self,
        *variables: str,
        time: float | None = None,
        index: int | None = None,
        cycle: int | None = None,
        file: Any = None,
        labels: bool = True,
        lineout: Any = None,
    ) -> None:
        """Legacy API: print structured query table."""

        if cycle is not None:
            raise NotImplementedError("cycle selection is not implemented in the refreshed adapter")

        selector_time = index if index is not None else time
        stream = file or sys.stdout
        print_query(
            self.reader, *variables, time=selector_time, lineout=lineout, labels=labels, file=stream
        )

    def describe(self, file: Any = None) -> None:
        """Legacy API: write database description."""

        from exodusii.cli.exoread import describe

        stream = file or sys.stdout
        describe(self.reader, file=stream)

    def _legacy_variable_names(self, on: Entity) -> npt.NDArray[np.str_]:
        return np.asarray(self.reader.variable_names(on))

    def _legacy_variable_number(self, on: Entity) -> int:
        return len(self.reader.variable_names(on))

    def _legacy_variable_truth_table(
        self, on: Entity, id_value: int | None = None
    ) -> npt.NDArray[np.int64] | None:
        return self.reader.variable_truth_table(on, id=id_value)

    def _legacy_variable_values(
        self, on: Entity, id_value: int | None, name: str, time_step: int | None
    ) -> npt.NDArray[np.float64]:
        time = None if time_step is None else time_step - 1

        if on in {Entity.ELEMENT, Entity.EDGE, Entity.FACE}:
            return self.reader.values(name, on=on, block_id=id_value, time=time)

        if on in {
            Entity.NODE_SET,
            Entity.SIDE_SET,
            Entity.EDGE_SET,
            Entity.FACE_SET,
            Entity.ELEMENT_SET,
        }:
            return self.reader.values(name, on=on, set_id=id_value, time=time)

        return self.reader.values(name, on=on, time=time)


def File(filename: str | Path, *files: str | Path, mode: str = "r"):
    """Open an Exodus database using the legacy factory name."""

    if files:
        if mode != "r":
            raise ExodusInvalidModeError("parallel Exodus files can only be opened in read mode")
        from exodusii.compat.legacy_parallel import ParallelExodusIIFile

        return ParallelExodusIIFile(filename, *files)

    return ExodusIIFile(filename, mode=mode)


exodusii_file = ExodusIIFile


def write_globals(
    data: Mapping[str, npt.ArrayLike],
    times: npt.ArrayLike,
    title: str | None = None,
    filename: str | Path = "Globals.exo",
) -> str:
    """Write a globals-only Exodus file using the legacy helper."""

    time_values = np.asarray(times, dtype=np.float64)
    names = list(data)

    with ExodusIIFile(filename, mode="w") as exo:
        exo.put_init(title or "", 1, 0, 0, 0, 0, 0)
        exo.put_global_variable_params(len(names))
        exo.put_global_variable_names(names)

        for step, time in enumerate(time_values, start=1):
            values = np.asarray(
                [np.asarray(data[name], dtype=np.float64)[step - 1] for name in names]
            )
            exo.put_time(step, float(time))
            exo.put_global_variable_values(step, values)

    return str(filename)


def _one_based_index(values: npt.ArrayLike, value: int) -> int | None:
    array = np.asarray(values)
    matches = np.nonzero(array == value)[0]
    return None if len(matches) == 0 else int(matches[0]) + 1


def _zero_based_index(values: npt.ArrayLike, value: int) -> int:
    index = _one_based_index(values, value)
    if index is None:
        raise ValueError(f"{value} is not in array")
    return index - 1


def _fixed_name(name: str) -> npt.NDArray[np.bytes_]:
    from exodusii.core.strings import encode_fixed_width

    return encode_fixed_width(name, width=32)


def _string_list(values: Any) -> list[str]:
    array = np.asarray(values)

    if array.ndim == 0:
        return [str(array.item())]

    return [str(value) for value in array.reshape(-1).tolist()]


def _fixed_names(names: list[str]) -> npt.NDArray[np.bytes_]:
    from exodusii.core.strings import encode_fixed_width

    return encode_fixed_width(names, width=32)


def _current_coordinates(writer: ExodusWriter) -> npt.NDArray[np.float64]:
    dimension = writer.backend.dimension(DimensionName.NUM_DIMENSIONS.value, 0) or 0
    node_count = writer.backend.dimension(DimensionName.NUM_NODES.value, 0) or 0
    coords = np.zeros((node_count, dimension), dtype=np.float64)
    for axis in range(dimension):
        coords[:, axis] = writer.backend.variable(
            ExodusNames.coordinate(axis), default=np.zeros(node_count)
        )
    return coords


__all__ = ["ExodusIIFile", "File", "exodusii_file", "write_globals"]
