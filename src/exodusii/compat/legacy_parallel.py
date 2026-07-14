# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Legacy parallel ExodusIIFile-compatible facade.

This adapter mirrors the serial compatibility split:

- modern parallel implementation: :class:`exodusii.api.parallel.ParallelExodusFile`
- legacy parallel facade: :class:`ParallelExodusIIFile`

The adapter intentionally delegates most behavior to the modern object.  At the
time this facade was introduced, :class:`ParallelExodusFile` still carried a
number of legacy methods for compatibility.  Keeping delegation here makes the
factory boundary correct now and allows those methods to be moved from the
modern class into this adapter over time.
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from exodusii.api.parallel import ParallelExodusFile


class ParallelExodusIIFile:
    """Backward-compatible parallel Exodus file facade."""

    def __init__(self, *files: str | Path) -> None:
        self._parallel = ParallelExodusFile.open(*files)

    @property
    def files(self):
        """Component files."""

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
        return name in self._parallel._files[0].variables()

    def __enter__(self) -> "ParallelExodusIIFile":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        """Delegate modern and transitional legacy methods to implementation."""

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

        return str(self._parallel.storage_type)

    # ------------------------------------------------------------------
    # Explicit legacy count wrappers
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
    # Frequently used legacy wrappers
    # ------------------------------------------------------------------

    def get_times(self) -> npt.NDArray[np.float64]:
        return self._parallel.times()

    def get_time(self, time_step: int) -> float:
        return float(self._parallel.times()[time_step - 1])

    def get_time_step(self, target: float, pcttol: float = 1.0e-5) -> int:
        return self._parallel.get_time_step(target, pcttol=pcttol)

    def get_coord_names(self) -> npt.NDArray[np.str_]:
        return self._parallel.coordinate_names()

    def get_coords(self, time_step: int | None = None) -> npt.NDArray[np.float64]:
        return self._parallel.coordinates(
            time=None if time_step is None else time_step - 1, displaced=time_step is not None
        )

    def get_displ_variable_names(self, default: Any = None):
        return self._parallel.get_displ_variable_names(default=default)

    def get_displ(self, time_step: int, default: Any = None):
        return self._parallel.get_displ(time_step, default=default)

    # ------------------------------------------------------------------
    # ID maps
    # ------------------------------------------------------------------

    def get_node_id_map(self, file: Any = None) -> npt.NDArray[np.int64]:
        return self._parallel.get_node_id_map(file)

    def get_element_id_map(self, file: Any = None) -> npt.NDArray[np.int64]:
        return self._parallel.get_element_id_map(file)

    def get_edge_id_map(self, file: Any = None) -> npt.NDArray[np.int64]:
        return self._parallel.get_edge_id_map(file)

    def get_face_id_map(self, file: Any = None) -> npt.NDArray[np.int64]:
        return self._parallel.get_face_id_map(file)

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
        return self._parallel.get_element_block_iid(block_id)

    def get_edge_block_iid(self, block_id: int) -> int | None:
        return self._parallel.get_edge_block_iid(block_id)

    def get_face_block_iid(self, block_id: int) -> int | None:
        return self._parallel.get_face_block_iid(block_id)

    def get_element_block(self, block_id: int):
        return self._parallel.get_element_block(block_id)

    def get_edge_block(self, block_id: int):
        return self._parallel.get_edge_block(block_id)

    def get_face_block(self, block_id: int):
        return self._parallel.get_face_block(block_id)

    def get_element_conn(self, block_id: int, **kwargs: Any) -> npt.NDArray[np.int64]:
        return self._parallel.get_element_conn(block_id, **kwargs)

    def get_edge_block_conn(self, block_id: int) -> npt.NDArray[np.int64]:
        return self._parallel.get_edge_block_conn(block_id)

    def get_face_block_conn(self, block_id: int) -> npt.NDArray[np.int64]:
        return self._parallel.get_face_block_conn(block_id)

    def num_elems_in_all_blks(self) -> npt.NDArray[np.int64]:
        return self._parallel.num_elems_in_all_blks()

    def num_elems_in_blk(self, block_id: int) -> int:
        return self._parallel.num_elems_in_blk(block_id)

    def num_nodes_per_elem(self, block_id: int) -> int:
        return self._parallel.num_nodes_per_elem(block_id)

    def num_edges_in_blk(self, block_id: int) -> int:
        return self._parallel.num_edges_in_blk(block_id)

    def num_faces_in_blk(self, block_id: int) -> int:
        return self._parallel.num_faces_in_blk(block_id)

    def num_nodes_per_edge(self, block_id: int) -> int:
        return self._parallel.num_nodes_per_edge(block_id)

    def num_nodes_per_face(self, block_id: int) -> int:
        return self._parallel.num_nodes_per_face(block_id)

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
        return self._parallel.get_node_set_iid(set_id)

    def get_side_set_iid(self, set_id: int) -> int | None:
        return self._parallel.get_side_set_iid(set_id)

    def get_node_set(self, set_id: int):
        return self._parallel.get_node_set(set_id)

    def get_side_set(self, set_id: int):
        return self._parallel.get_side_set(set_id)

    def get_edge_set(self, set_id: int):
        return self._parallel.get_edge_set(set_id)

    def get_face_set(self, set_id: int):
        return self._parallel.get_face_set(set_id)

    def get_element_set(self, set_id: int):
        return self._parallel.get_element_set(set_id)

    def get_node_set_nodes(self, set_id: int):
        return self._parallel.get_node_set_nodes(set_id)

    def get_node_set_dist_facts(self, set_id: int):
        return self._parallel.get_node_set_dist_facts(set_id)

    def get_side_set_elems(self, set_id: int):
        return self._parallel.get_side_set_elems(set_id)

    def get_side_set_sides(self, set_id: int):
        return self._parallel.get_side_set_sides(set_id)

    def get_side_set_dist_facts(self, set_id: int):
        return self._parallel.get_side_set_dist_facts(set_id)

    def get_node_set_params(self, set_id: int):
        return self._parallel.get_node_set_params(set_id)

    def get_side_set_params(self, set_id: int):
        return self._parallel.get_side_set_params(set_id)

    def num_nodes_in_node_set(self, set_id: int) -> int:
        return self._parallel.num_nodes_in_node_set(set_id)

    def num_sides_in_side_set(self, set_id: int) -> int:
        return self._parallel.num_sides_in_side_set(set_id)

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------

    def get_global_variable_names(self) -> npt.NDArray[np.str_]:
        return self._parallel.get_global_variable_names()

    def get_node_variable_names(self) -> npt.NDArray[np.str_]:
        return self._parallel.get_node_variable_names()

    def get_element_variable_names(self) -> npt.NDArray[np.str_]:
        return self._parallel.get_element_variable_names()

    def get_edge_variable_names(self) -> npt.NDArray[np.str_]:
        return self._parallel.get_edge_variable_names()

    def get_face_variable_names(self) -> npt.NDArray[np.str_]:
        return self._parallel.get_face_variable_names()

    def get_node_set_variable_names(self) -> npt.NDArray[np.str_]:
        return self._parallel.get_node_set_variable_names()

    def get_side_set_variable_names(self) -> npt.NDArray[np.str_]:
        return self._parallel.get_side_set_variable_names()

    def get_edge_set_variable_names(self) -> npt.NDArray[np.str_]:
        return self._parallel.get_edge_set_variable_names()

    def get_face_set_variable_names(self) -> npt.NDArray[np.str_]:
        return self._parallel.get_face_set_variable_names()

    def get_element_set_variable_names(self) -> npt.NDArray[np.str_]:
        return self._parallel.get_element_set_variable_names()

    def get_global_variable_values(self, var_name: str) -> npt.NDArray[np.float64]:
        return self._parallel.get_global_variable_values(var_name)

    def get_all_global_variable_values(
        self, time_step: int | None = None
    ) -> npt.NDArray[np.float64]:
        return self._parallel.get_all_global_variable_values(time_step)

    def get_node_variable_values(
        self, var_name: str, time_step: int | None = None
    ) -> npt.NDArray[np.float64]:
        return self._parallel.get_node_variable_values(var_name, time_step=time_step)

    def get_element_variable_values(
        self, block_id: int | None, var_name: str, time_step: int | None = None
    ) -> npt.NDArray[np.float64]:
        return self._parallel.get_element_variable_values(block_id, var_name, time_step=time_step)

    def get_edge_variable_values(
        self, block_id: int | None, var_name: str, time_step: int | None = None
    ) -> npt.NDArray[np.float64]:
        return self._parallel.get_edge_variable_values(block_id, var_name, time_step=time_step)

    def get_face_variable_values(
        self, block_id: int | None, var_name: str, time_step: int | None = None
    ) -> npt.NDArray[np.float64]:
        return self._parallel.get_face_variable_values(block_id, var_name, time_step=time_step)

    def get_node_variable_history(self, var_name: str, node_id: int) -> npt.NDArray[np.float64]:
        return self._parallel.get_node_variable_history(var_name, node_id)

    def get_element_variable_history(self, var_name: str, elem_id: int) -> npt.NDArray[np.float64]:
        return self._parallel.get_element_variable_history(var_name, elem_id)

    # ------------------------------------------------------------------
    # Misc legacy
    # ------------------------------------------------------------------

    def get_mapping(
        self, name: Any, invert: bool = False, contiguous: bool = False
    ) -> dict[Any, Any]:
        return self._parallel.get_mapping(name, invert=invert, contiguous=contiguous)


parallel_exodusii_file = ParallelExodusIIFile
MFExodusIIFile = ParallelExodusIIFile


__all__ = ["MFExodusIIFile", "ParallelExodusIIFile", "parallel_exodusii_file"]
