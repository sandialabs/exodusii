# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Legacy parallel ExodusIIFile-compatible facade."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt

from exodusii.api.file import ExodusFile
from exodusii.api.parallel import ParallelExodusFile
from exodusii.core.entities import Entity
from exodusii.core.errors import ExodusLookupError
from exodusii.core.names import VariableName


class ParallelExodusIIFile:
    """Backward-compatible parallel Exodus file facade.

    This class intentionally exposes the historical method-style API while
    delegating modern aggregation behavior to :class:`ParallelExodusFile`.
    """

    def __init__(self, *files: str | Path) -> None:
        self._parallel = ParallelExodusFile.open(*files)

    @property
    def files(self) -> tuple[ExodusFile, ...]:
        """Component modern Exodus files."""

        return self._parallel.files

    @property
    def filename(self) -> str:
        """Legacy filename string."""

        return self._parallel.filename

    @property
    def path(self) -> str:
        """Comma-separated component paths."""

        return self._parallel.path

    def __contains__(self, name: str) -> bool:
        return name in self._parallel.files[0].variables()

    def __enter__(self) -> "ParallelExodusIIFile":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        """Delegate modern methods/properties to the wrapped parallel object."""

        return getattr(self._parallel, name)

    def close(self) -> None:
        """Close owned component files."""

        self._parallel.close()

    def write(self, filename: str | Path) -> str:
        """Write aggregate parallel database to a serial Exodus file."""

        return self._parallel.write(filename)

    # ------------------------------------------------------------------
    # Legacy method-style wrappers for property-style modern API
    # ------------------------------------------------------------------

    def title(self) -> str:
        """Legacy API: database title."""

        return self._parallel.title

    def storage_type(self) -> str:
        """Legacy API: floating-point storage type."""

        return self._parallel.storage_type

    # ------------------------------------------------------------------
    # Records
    # ------------------------------------------------------------------

    def info_records(self) -> tuple[str, ...]:
        return self._parallel.info_records()

    def qa_records(self) -> tuple[tuple[str, str, str, str], ...]:
        return self._parallel.qa_records()

    def get_info_records(self) -> list[str] | None:
        records = self.info_records()
        return [record[:80] for record in records] if records else None

    def get_qa_records(self):
        records = self.qa_records()
        return list(records) if records else None

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    def num_dimensions(self) -> int:
        return self._parallel.dimension

    def num_nodes(self) -> int:
        return self._parallel.node_count

    def num_edges(self) -> int:
        return self._parallel.edge_count

    def num_faces(self) -> int:
        return self._parallel.face_count

    def num_elems(self) -> int:
        return self._parallel.element_count

    def num_blks(self) -> int:
        return self._parallel.element_block_count

    def num_element_blocks(self) -> int:
        return self._parallel.element_block_count

    def num_elem_blk(self) -> int:
        return self._parallel.element_block_count

    def num_node_sets(self) -> int:
        return self._parallel.node_set_count

    def num_side_sets(self) -> int:
        return self._parallel.side_set_count

    def num_edge_blk(self) -> int:
        return self._parallel.edge_block_count

    def num_face_blk(self) -> int:
        return self._parallel.face_block_count

    def num_edge_sets(self) -> int:
        return len(self._parallel.edge_set_ids())

    def num_face_sets(self) -> int:
        return len(self._parallel.face_set_ids())

    def num_elem_sets(self) -> int:
        return len(self._parallel.element_set_ids())

    def num_elem_maps(self) -> int:
        return 0

    def num_node_maps(self) -> int:
        return 0

    def num_edge_maps(self) -> int:
        return 0

    def num_face_maps(self) -> int:
        return 0

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    def get_times(self) -> npt.NDArray[np.float64]:
        return self._parallel.times()

    def get_time(self, time_step: int) -> float:
        return float(self._parallel.times()[time_step - 1])

    def get_time_step(self, target: float, pcttol: float = 1.0e-5) -> int:
        times = np.asarray(self._parallel.times(), dtype=np.float64)
        if times.size == 0:
            raise ValueError("no time steps found")

        index = int(np.abs(times - target).argmin())

        if abs(target) > 0.0:
            relerr = abs(float(times[index]) - target) / abs(target)
            if relerr > pcttol:
                import logging

                logging.warning("Solution time differs significantly from desired solution time.")

        return index + 1

    # ------------------------------------------------------------------
    # Coordinates/displacements
    # ------------------------------------------------------------------

    def get_coord_names(self) -> npt.NDArray[np.str_]:
        return self._parallel.coordinate_names()

    def get_coords(self, time_step: int | None = None) -> npt.NDArray[np.float64]:
        return self._parallel.coordinates(
            time=None if time_step is None else time_step - 1, displaced=time_step is not None
        )

    def get_displ_variable_names(self, default: Any = None):
        names = self._parallel.displacement_variable_names()
        return names if names else default

    def get_displ(self, time_step: int, default: Any = None):
        names = self._parallel.displacement_variable_names()
        if not names:
            return default
        return self._parallel.displacements(time=time_step - 1)

    # ------------------------------------------------------------------
    # ID maps
    # ------------------------------------------------------------------

    def get_node_id_map(self, file: Any = None) -> npt.NDArray[np.int64]:
        if file is not None:
            file_index = self._file_index(file)
            return self._parallel._maps[file_index].node_lid_to_gid
        return self._parallel.ids(Entity.NODE)

    def get_element_id_map(self, file: Any = None) -> npt.NDArray[np.int64]:
        if file is not None:
            file_index = self._file_index(file)
            return self._parallel._maps[file_index].element_lid_to_gid
        return self._parallel.ids(Entity.ELEMENT)

    def get_edge_id_map(self, file: Any = None) -> npt.NDArray[np.int64]:
        if file is not None:
            file_index = self._file_index(file)
            return self._parallel._maps[file_index].edge_lid_to_gid
        return self._parallel.ids(Entity.EDGE)

    def get_face_id_map(self, file: Any = None) -> npt.NDArray[np.int64]:
        if file is not None:
            file_index = self._file_index(file)
            return self._parallel._maps[file_index].face_lid_to_gid
        return self._parallel.ids(Entity.FACE)

    def _file_index(self, file: Any) -> int:
        path = getattr(file, "path", None)

        if path is None:
            filepath = getattr(file, "filepath", None)
            if callable(filepath):
                path = Path(filepath())
            else:
                filename = getattr(file, "filename", None)
                path = Path(filename) if filename is not None else None

        for index, candidate in enumerate(self._parallel.files):
            if candidate is file:
                return index
            if path is not None and candidate.path == path:
                return index

        raise ExodusLookupError(f"component file {file!r} not found")

    # ------------------------------------------------------------------
    # Blocks
    # ------------------------------------------------------------------

    def get_element_block_ids(self) -> npt.NDArray[np.int64]:
        return self._parallel.element_block_ids()

    def get_edge_block_ids(self) -> npt.NDArray[np.int64]:
        return self._parallel.edge_block_ids()

    def get_face_block_ids(self) -> npt.NDArray[np.int64]:
        return self._parallel.face_block_ids()

    def get_element_block_id(self, block_iid: int) -> int:
        return int(self.get_element_block_ids()[block_iid - 1])

    def get_element_block_iid(self, block_id: int) -> int | None:
        return _one_based_index(self._parallel.element_block_ids(), block_id)

    def get_edge_block_iid(self, block_id: int) -> int | None:
        return _one_based_index(self._parallel.edge_block_ids(), block_id)

    def get_face_block_iid(self, block_id: int) -> int | None:
        return _one_based_index(self._parallel.face_block_ids(), block_id)

    def get_element_block(self, block_id: int):
        block = self._parallel.element_block(block_id)
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

    def get_edge_block(self, block_id: int):
        block = self._parallel.edge_block(block_id)
        return SimpleNamespace(
            id=block.id,
            iid=block.index,
            elem_type=block.element_type,
            name=block.name,
            num_block_edges=block.count,
            num_edge_nodes=block.nodes_per_entity,
            num_edge_attrs=block.attributes,
        )

    def get_face_block(self, block_id: int):
        block = self._parallel.face_block(block_id)
        return SimpleNamespace(
            id=block.id,
            iid=block.index,
            elem_type=block.element_type,
            name=block.name,
            num_block_faces=block.count,
            num_face_nodes=block.nodes_per_entity,
            num_face_attrs=block.attributes,
        )

    def get_element_conn(self, block_id: int, **_: Any) -> npt.NDArray[np.int64]:
        return self._parallel.element_connectivity(block_id)

    def get_edge_block_conn(self, block_id: int) -> npt.NDArray[np.int64]:
        return self._parallel.edge_connectivity(block_id)

    def get_face_block_conn(self, block_id: int) -> npt.NDArray[np.int64]:
        return self._parallel.face_connectivity(block_id)

    def num_elems_in_all_blks(self) -> npt.NDArray[np.int64]:
        counts = self._parallel._global_variable(VariableName.ELEMENT_BLOCK_COUNT_GLOBAL.value)
        if counts is not None:
            return counts

        return np.asarray(
            [
                self._parallel.element_block(int(block_id)).count
                for block_id in self._parallel.element_block_ids()
            ],
            dtype=np.int64,
        )

    def num_elems_in_blk(self, block_id: int) -> int:
        counts = self.num_elems_in_all_blks()
        ids = self._parallel.element_block_ids()
        index = _one_based_index_required(ids, block_id) - 1
        return int(counts[index])

    def num_nodes_per_elem(self, block_id: int) -> int:
        return self._parallel.element_block(block_id).nodes_per_entity

    def num_edges_in_blk(self, block_id: int) -> int:
        return self._parallel.edge_block(block_id).count

    def num_faces_in_blk(self, block_id: int) -> int:
        return self._parallel.face_block(block_id).count

    def num_nodes_per_edge(self, block_id: int) -> int:
        return self._parallel.edge_block(block_id).nodes_per_entity

    def num_nodes_per_face(self, block_id: int) -> int:
        return self._parallel.face_block(block_id).nodes_per_entity

    # ------------------------------------------------------------------
    # Sets
    # ------------------------------------------------------------------

    def get_node_set_ids(self) -> npt.NDArray[np.int64]:
        return self._parallel.node_set_ids()

    def get_side_set_ids(self) -> npt.NDArray[np.int64]:
        return self._parallel.side_set_ids()

    def get_edge_set_ids(self) -> npt.NDArray[np.int64]:
        return self._parallel.edge_set_ids()

    def get_face_set_ids(self) -> npt.NDArray[np.int64]:
        return self._parallel.face_set_ids()

    def get_element_set_ids(self) -> npt.NDArray[np.int64]:
        return self._parallel.element_set_ids()

    def get_node_set_id(self, set_iid: int) -> int:
        return int(self.get_node_set_ids()[set_iid - 1])

    def get_node_set_iid(self, set_id: int) -> int | None:
        return _one_based_index(self._parallel.node_set_ids(), set_id)

    def get_side_set_iid(self, set_id: int) -> int | None:
        return _one_based_index(self._parallel.side_set_ids(), set_id)

    def get_node_set(self, set_id: int):
        node_set = self._parallel.node_set(set_id)
        return SimpleNamespace(
            id=node_set.id,
            iid=node_set.index,
            name=node_set.name,
            num_nodes=node_set.count,
            num_dist_facts=node_set.distribution_factors,
            nodes=node_set.nodes,
            dist_facts=node_set.dist_facts,
        )

    def get_side_set(self, set_id: int):
        side_set = self._parallel.side_set(set_id)
        return SimpleNamespace(
            id=side_set.id,
            iid=side_set.index,
            name=side_set.name,
            num_sides=side_set.count,
            num_dist_facts=side_set.distribution_factors,
            elems=side_set.elems,
            sides=side_set.sides,
            dist_facts=side_set.dist_facts,
        )

    def get_edge_set(self, set_id: int):
        set_info = self._parallel.edge_set(set_id)
        return SimpleNamespace(
            id=set_info.id,
            iid=set_info.index,
            name=set_info.name,
            num_edges=set_info.count,
            num_dist_facts=set_info.distribution_factors,
            edges=set_info.entries,
            orientations=set_info.extra_entries,
            dist_facts=set_info.dist_facts,
        )

    def get_face_set(self, set_id: int):
        set_info = self._parallel.face_set(set_id)
        return SimpleNamespace(
            id=set_info.id,
            iid=set_info.index,
            name=set_info.name,
            num_faces=set_info.count,
            num_dist_facts=set_info.distribution_factors,
            faces=set_info.entries,
            orientations=set_info.extra_entries,
            dist_facts=set_info.dist_facts,
        )

    def get_element_set(self, set_id: int):
        set_info = self._parallel.element_set(set_id)
        return SimpleNamespace(
            id=set_info.id,
            iid=set_info.index,
            name=set_info.name,
            num_elems=set_info.count,
            num_dist_facts=set_info.distribution_factors,
            elems=set_info.entries,
            dist_facts=set_info.dist_facts,
        )

    def get_node_set_nodes(self, set_id: int):
        return self._parallel.node_set(set_id).nodes

    def get_node_set_dist_facts(self, set_id: int):
        return self._parallel.node_set(set_id).dist_facts

    def get_side_set_elems(self, set_id: int):
        return self._parallel.side_set(set_id).elems

    def get_side_set_sides(self, set_id: int):
        return self._parallel.side_set(set_id).sides

    def get_side_set_dist_facts(self, set_id: int):
        return self._parallel.side_set(set_id).dist_facts

    def get_node_set_params(self, set_id: int):
        node_set = self._parallel.node_set(set_id)
        return SimpleNamespace(
            num_nodes=node_set.count, num_dist_facts=node_set.distribution_factors
        )

    def get_side_set_params(self, set_id: int):
        side_set = self._parallel.side_set(set_id)
        return SimpleNamespace(
            num_sides=side_set.count, num_dist_facts=side_set.distribution_factors
        )

    def num_nodes_in_node_set(self, set_id: int) -> int:
        counts = self._parallel._global_variable(VariableName.NODE_SET_NODE_COUNT_GLOBAL.value)
        if counts is not None:
            ids = self._parallel.node_set_ids()
            index = _one_based_index_required(ids, set_id) - 1
            return int(counts[index])
        return self._parallel.node_set(set_id).count

    def num_sides_in_side_set(self, set_id: int) -> int:
        counts = self._parallel._global_variable(VariableName.SIDE_SET_SIDE_COUNT_GLOBAL.value)
        if counts is not None:
            ids = self._parallel.side_set_ids()
            index = _one_based_index_required(ids, set_id) - 1
            return int(counts[index])
        return self._parallel.side_set(set_id).count

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------

    def get_global_variable_names(self) -> npt.NDArray[np.str_]:
        return np.asarray(self._parallel.variable_names(Entity.GLOBAL))

    def get_node_variable_names(self) -> npt.NDArray[np.str_]:
        return np.asarray(self._parallel.variable_names(Entity.NODE))

    def get_element_variable_names(self) -> npt.NDArray[np.str_]:
        return np.asarray(self._parallel.variable_names(Entity.ELEMENT))

    def get_edge_variable_names(self) -> npt.NDArray[np.str_]:
        return np.asarray(self._parallel.variable_names(Entity.EDGE))

    def get_face_variable_names(self) -> npt.NDArray[np.str_]:
        return np.asarray(self._parallel.variable_names(Entity.FACE))

    def get_node_set_variable_names(self) -> npt.NDArray[np.str_]:
        return np.asarray(self._parallel.variable_names(Entity.NODE_SET))

    def get_side_set_variable_names(self) -> npt.NDArray[np.str_]:
        return np.asarray(self._parallel.variable_names(Entity.SIDE_SET))

    def get_edge_set_variable_names(self) -> npt.NDArray[np.str_]:
        return np.asarray(self._parallel.variable_names(Entity.EDGE_SET))

    def get_face_set_variable_names(self) -> npt.NDArray[np.str_]:
        return np.asarray(self._parallel.variable_names(Entity.FACE_SET))

    def get_element_set_variable_names(self) -> npt.NDArray[np.str_]:
        return np.asarray(self._parallel.variable_names(Entity.ELEMENT_SET))

    def get_global_variable_values(self, var_name: str) -> npt.NDArray[np.float64]:
        return self._parallel.values(var_name, on=Entity.GLOBAL)

    def get_all_global_variable_values(
        self, time_step: int | None = None
    ) -> npt.NDArray[np.float64]:
        names = self._parallel.variable_names(Entity.GLOBAL)
        values = np.column_stack([self._parallel.values(name, on=Entity.GLOBAL) for name in names])
        return values if time_step is None else values[time_step - 1]

    def get_node_variable_values(
        self, var_name: str, time_step: int | None = None
    ) -> npt.NDArray[np.float64]:
        return self._parallel.values(
            var_name, on=Entity.NODE, time=None if time_step is None else time_step - 1
        )

    def get_element_variable_values(
        self, block_id: int | None, var_name: str, time_step: int | None = None
    ) -> npt.NDArray[np.float64]:
        return self._parallel.values(
            var_name,
            on=Entity.ELEMENT,
            block_id=block_id,
            time=None if time_step is None else time_step - 1,
        )

    def get_edge_variable_values(
        self, block_id: int | None, var_name: str, time_step: int | None = None
    ) -> npt.NDArray[np.float64]:
        return self._parallel.values(
            var_name,
            on=Entity.EDGE,
            block_id=block_id,
            time=None if time_step is None else time_step - 1,
        )

    def get_face_variable_values(
        self, block_id: int | None, var_name: str, time_step: int | None = None
    ) -> npt.NDArray[np.float64]:
        return self._parallel.values(
            var_name,
            on=Entity.FACE,
            block_id=block_id,
            time=None if time_step is None else time_step - 1,
        )

    def get_node_set_variable_values(
        self, set_id: int | None, var_name: str, time_step: int | None = None
    ) -> npt.NDArray[np.float64]:
        return self._parallel.values(
            var_name,
            on=Entity.NODE_SET,
            set_id=set_id,
            time=None if time_step is None else time_step - 1,
        )

    def get_side_set_variable_values(
        self, set_id: int | None, var_name: str, time_step: int | None = None
    ) -> npt.NDArray[np.float64]:
        return self._parallel.values(
            var_name,
            on=Entity.SIDE_SET,
            set_id=set_id,
            time=None if time_step is None else time_step - 1,
        )

    def get_edge_set_variable_values(
        self, set_id: int | None, var_name: str, time_step: int | None = None
    ) -> npt.NDArray[np.float64]:
        return self._parallel.values(
            var_name,
            on=Entity.EDGE_SET,
            set_id=set_id,
            time=None if time_step is None else time_step - 1,
        )

    def get_face_set_variable_values(
        self, set_id: int | None, var_name: str, time_step: int | None = None
    ) -> npt.NDArray[np.float64]:
        return self._parallel.values(
            var_name,
            on=Entity.FACE_SET,
            set_id=set_id,
            time=None if time_step is None else time_step - 1,
        )

    def get_element_set_variable_values(
        self, set_id: int | None, var_name: str, time_step: int | None = None
    ) -> npt.NDArray[np.float64]:
        return self._parallel.values(
            var_name,
            on=Entity.ELEMENT_SET,
            set_id=set_id,
            time=None if time_step is None else time_step - 1,
        )

    def get_node_variable_history(self, var_name: str, node_id: int) -> npt.NDArray[np.float64]:
        node_ids = self._parallel.ids(Entity.NODE)
        row = _one_based_index_required(node_ids, node_id) - 1
        return self._parallel.values(var_name, on=Entity.NODE)[:, row]

    def get_element_variable_history(self, var_name: str, elem_id: int) -> npt.NDArray[np.float64]:
        element_ids = self._parallel.ids(Entity.ELEMENT)
        global_row = _one_based_index_required(element_ids, elem_id) - 1

        start = 0
        for block_id in self._parallel.element_block_ids():
            count = self._parallel.element_block(int(block_id)).count
            stop = start + count
            if start <= global_row < stop:
                values = self._parallel.values(var_name, on=Entity.ELEMENT, block_id=int(block_id))
                return values[:, global_row - start]
            start = stop

        raise ExodusLookupError(f"element ID {elem_id} not found")

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    def get_mapping(
        self, name: Any, invert: bool = False, contiguous: bool = False
    ) -> dict[Any, Any]:
        map_name = getattr(name, "name", str(name))

        mapping: dict[Any, Any]
        if map_name == "node_local_to_global":
            mapping = {
                (file_index, local_index): int(gid)
                for file_index, maps in enumerate(self._parallel._maps)
                for local_index, gid in enumerate(maps.node_lid_to_gid, start=1)
            }
        elif map_name == "elem_local_to_global":
            mapping = {
                (file_index, local_index): int(gid)
                for file_index, maps in enumerate(self._parallel._maps)
                for local_index, gid in enumerate(maps.element_lid_to_gid, start=1)
            }
        elif map_name == "elem_block_elem_local_to_global":
            mapping = {}
            for file_index, file in enumerate(self._parallel.files):
                for block_id in file.element_block_ids():
                    gids = self._parallel._block_element_gids_for_file(file_index, int(block_id))
                    for local_index, gid in enumerate(gids, start=1):
                        mapping[(file_index, int(block_id), local_index)] = int(gid)
        else:
            raise ValueError(f"invalid map name {name!r}")

        if contiguous:
            contiguous_map = _make_contiguous(mapping)
            mapping = {key: contiguous_map[gid] for key, gid in mapping.items()}

        if invert:
            return {value: key for key, value in mapping.items()}

        return mapping


def _one_based_index(values: npt.ArrayLike, value: int) -> int | None:
    matches = np.nonzero(np.asarray(values) == value)[0]
    if not len(matches):
        return None
    return int(matches[0]) + 1


def _one_based_index_required(values: npt.ArrayLike, value: int) -> int:
    index = _one_based_index(values, value)
    if index is None:
        raise ExodusLookupError(f"ID {value} not found")
    return index


def _make_contiguous(mapping: dict[Any, int]) -> dict[int, int]:
    gids = sorted(set(mapping.values()))
    return {gid: index for index, gid in enumerate(gids, start=1)}


parallel_exodusii_file = ParallelExodusIIFile
MFExodusIIFile = ParallelExodusIIFile


__all__ = ["MFExodusIIFile", "ParallelExodusIIFile", "parallel_exodusii_file"]
