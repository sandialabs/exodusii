# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Legacy extension-style mesh helpers."""

import numpy as np

from exodusii.mesh.geometry import connected_average
from exodusii.mesh.geometry import element_volumes
from exodusii.mesh.geometry import entity_centers
from exodusii.mesh.geometry import nodal_volumes


def compute_element_centers(file, block_id: int | None = None, time_step: int | None = None):
    """Compute element centers using the legacy extension API."""

    if block_id is None:
        centers = [
            compute_element_centers(file, int(block), time_step=time_step)
            for block in file.get_element_block_ids()
        ]
        return np.concatenate(centers, axis=0) if centers else np.empty((0, file.num_dimensions()))

    conn = file.get_element_conn(block_id) - 1
    coords = file.get_coords(time_step=time_step)
    return entity_centers(conn, coords)


def compute_element_volumes(file, block_id: int, time_step: int | None = None):
    """Compute element volumes using the legacy extension API."""

    block = file.get_element_block(block_id)
    conn = file.get_element_conn(block_id) - 1
    coords = file.get_coords(time_step=time_step)
    return element_volumes(block.elem_type, conn, coords)


def compute_node_volumes(file, time_step: int | None = None):
    """Compute nodal volumes using the legacy extension API."""

    result = np.zeros(file.num_nodes(), dtype=np.float64)

    for block_id in file.get_element_block_ids():
        block = file.get_element_block(int(block_id))
        conn = file.get_element_conn(int(block_id)) - 1
        coords = file.get_coords(time_step=time_step)
        result += nodal_volumes(block.elem_type, conn, coords, num_nodes=file.num_nodes())

    return result


def compute_element_length(file, time: float):
    """Compute average characteristic element length."""

    time_step = _legacy_time_step(file, time)
    lengths = []
    weights = []

    for block_id in file.get_element_block_ids():
        block = file.get_element_block(int(block_id))
        conn = file.get_element_conn(int(block_id)) - 1
        coords = file.get_coords(time_step=time_step)
        volumes = compute_element_volumes(file, int(block_id), time_step=time_step)

        dimension = file.num_dimensions()
        lengths.extend(np.power(np.abs(volumes), 1.0 / dimension).tolist())
        weights.extend([1.0] * block.num_block_elems)

    if not lengths:
        raise ValueError("cannot compute element length of an empty mesh")

    return float(np.average(lengths, weights=weights))


def compute_node_variable_values_at_element_center(
    file, block_id: int | None, var_name: str, time_step: int | None = None
):
    """Average nodal variable values to element centers."""

    if block_id is None:
        values = [
            compute_node_variable_values_at_element_center(
                file, int(block), var_name, time_step=time_step
            )
            for block in file.get_element_block_ids()
        ]
        return np.concatenate(values, axis=0) if values else np.asarray([])

    if var_name == "coordinates":
        nodal_values = file.get_coords()
    elif var_name == "displacements":
        nodal_values = file.get_displ(time_step)
    else:
        nodal_values = file.get_node_variable_values(var_name, time_step=time_step)

    conn = file.get_element_conn(block_id) - 1
    return connected_average(conn, nodal_values)


def compute_edge_centers(file, block_id: int | None = None, time_step: int | None = None):
    """Compute edge centers using the legacy extension API.

    Returns the geometric centroid (mean of node coordinates) for each edge
    in *block_id*, or concatenated across all edge blocks when *block_id* is
    ``None``.
    """

    if block_id is None:
        centers = [
            compute_edge_centers(file, int(block), time_step=time_step)
            for block in file.get_edge_block_ids()
        ]
        return np.concatenate(centers, axis=0) if centers else np.empty((0, file.num_dimensions()))

    conn = file.get_edge_block_conn(block_id) - 1
    coords = file.get_coords(time_step=time_step)
    return entity_centers(conn, coords)


def compute_face_centers(file, block_id: int | None = None, time_step: int | None = None):
    """Compute face centers using the legacy extension API.

    Returns the geometric centroid (mean of node coordinates) for each face
    in *block_id*, or concatenated across all face blocks when *block_id* is
    ``None``.
    """

    if block_id is None:
        centers = [
            compute_face_centers(file, int(block), time_step=time_step)
            for block in file.get_face_block_ids()
        ]
        return np.concatenate(centers, axis=0) if centers else np.empty((0, file.num_dimensions()))

    conn = file.get_face_block_conn(block_id) - 1
    coords = file.get_coords(time_step=time_step)
    return entity_centers(conn, coords)


def compute_volume_averaged_elem_variable(
    file,
    block_id: int,
    time_step: int,
    func,
    intervals: int = 5,
    zfill: float | None = None,
    processes: int | None = None,
):
    """Compute a volume-averaged analytic function over elements.

    This compatibility implementation is serial. ``processes`` is accepted for
    API compatibility but ignored.
    """

    del processes

    block = file.get_element_block(block_id)
    coords = file.get_coords(time_step=time_step)
    if file.num_dimensions() == 2 and zfill is not None:
        coords = np.column_stack((coords, np.full(coords.shape[0], zfill)))

    conn = file.get_element_conn(block_id) - 1
    averaged = np.zeros(len(conn), dtype=np.float64)
    exact_time = file.get_time(time_step)

    from exodusii.mesh.elements import element_factory

    for element_index, node_indices in enumerate(conn):
        element = element_factory(block.elem_type, coords[node_indices])
        centers = element.subdiv(intervals)
        volumes = element.subvols(intervals)
        exact = np.asarray([func(center, exact_time) for center in centers], dtype=np.float64)
        averaged[element_index] = float(np.sum(volumes * exact) / np.sum(volumes))

    return averaged


def _legacy_time_step(file, target: float) -> int:
    times = np.asarray(file.get_times(), dtype=np.float64)
    return int(np.abs(times - target).argmin()) + 1


__all__ = [
    "compute_edge_centers",
    "compute_element_centers",
    "compute_element_length",
    "compute_element_volumes",
    "compute_face_centers",
    "compute_node_variable_values_at_element_center",
    "compute_node_volumes",
    "compute_volume_averaged_elem_variable",
]
