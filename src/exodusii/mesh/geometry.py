# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Array-based mesh geometry utilities."""

import numpy as np
import numpy.typing as npt

from exodusii.mesh.elements import element_factory

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def connected_average(conn: npt.ArrayLike, values: npt.ArrayLike) -> FloatArray:
    """Average nodal values over connectivity.

    Parameters
    ----------
    conn
        Connectivity array using zero-based indices, shape ``(num_entities, nodes_per_entity)``.
    values
        Nodal values, shape ``(num_nodes,)`` or ``(num_nodes, num_components)``.

    Returns
    -------
    ndarray
        Averaged values. If input values are scalar per node, result has shape
        ``(num_entities,)``. Otherwise result has shape ``(num_entities, num_components)``.
    """

    connectivity = _connectivity_array(conn)
    nodal_values = np.asarray(values, dtype=np.float64)

    if nodal_values.ndim not in {1, 2}:
        raise ValueError("values must be a one- or two-dimensional array")

    if connectivity.size and connectivity.max() >= nodal_values.shape[0]:
        raise IndexError("connectivity references a value index outside values")
    if connectivity.size and connectivity.min() < 0:
        raise IndexError("connectivity contains negative indices")

    return np.asarray(nodal_values[connectivity].mean(axis=1), dtype=np.float64)


def entity_centers(conn: npt.ArrayLike, coords: npt.ArrayLike) -> FloatArray:
    """Compute geometric centers from connectivity and coordinates."""

    centers = connected_average(conn, coords)

    if centers.ndim != 2:
        raise ValueError("coordinates must be two-dimensional")

    return centers


def element_volumes(
    element_type: str | bytes, conn: npt.ArrayLike, coords: npt.ArrayLike
) -> FloatArray:
    """Compute element measures for all elements in a connectivity array.

    For 2-D elements this returns areas. For 3-D elements this returns volumes.
    Connectivity is zero-based.
    """

    connectivity = _connectivity_array(conn)
    coordinate_array = _coordinate_array(coords)

    volumes = np.empty(connectivity.shape[0], dtype=np.float64)
    for index, node_indices in enumerate(connectivity):
        volumes[index] = element_factory(element_type, coordinate_array[node_indices]).volume

    return volumes


def nodal_volumes(
    element_type: str | bytes,
    conn: npt.ArrayLike,
    coords: npt.ArrayLike,
    *,
    num_nodes: int | None = None,
) -> FloatArray:
    """Distribute element measure equally to connected nodes."""

    connectivity = _connectivity_array(conn)
    coordinate_array = _coordinate_array(coords)

    if num_nodes is None:
        num_nodes = coordinate_array.shape[0]
    if num_nodes < 0:
        raise ValueError("num_nodes must be nonnegative")

    if connectivity.size and connectivity.max() >= num_nodes:
        raise IndexError("connectivity references a node outside num_nodes")

    volumes = element_volumes(element_type, connectivity, coordinate_array)
    nodal = np.zeros(num_nodes, dtype=np.float64)

    for element_index, node_indices in enumerate(connectivity):
        nodal[node_indices] += volumes[element_index] / len(node_indices)

    return nodal


def characteristic_element_length(
    element_type: str | bytes,
    conn: npt.ArrayLike,
    coords: npt.ArrayLike,
    *,
    dimension: int | None = None,
) -> float:
    r"""Compute average characteristic element length.

    The characteristic length of an element is \(V^{1/d}\), where \(V\) is
    area/volume and \(d\) is the spatial dimension.
    """

    coordinate_array = _coordinate_array(coords)
    if dimension is None:
        dimension = coordinate_array.shape[1]
    if dimension < 1:
        raise ValueError("dimension must be positive")

    volumes = np.abs(element_volumes(element_type, conn, coordinate_array))
    if volumes.size == 0:
        raise ValueError("cannot compute characteristic length of an empty mesh")

    return float(np.mean(np.power(volumes, 1.0 / dimension)))


def bounding_box(coords: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Return coordinate-wise minimum and maximum points."""

    coordinate_array = _coordinate_array(coords)

    if coordinate_array.shape[0] == 0:
        raise ValueError("cannot compute bounding box of empty coordinates")

    return (
        np.asarray(coordinate_array.min(axis=0), dtype=np.float64),
        np.asarray(coordinate_array.max(axis=0), dtype=np.float64),
    )


def _connectivity_array(conn: npt.ArrayLike) -> IntArray:
    connectivity = np.asarray(conn, dtype=np.int64)

    if connectivity.ndim != 2:
        raise ValueError("connectivity must be a two-dimensional array")
    if connectivity.shape[1] == 0:
        raise ValueError("connectivity must contain at least one node per entity")

    return connectivity


def _coordinate_array(coords: npt.ArrayLike) -> FloatArray:
    coordinate_array = np.asarray(coords, dtype=np.float64)

    if coordinate_array.ndim != 2:
        raise ValueError("coordinates must be a two-dimensional array")
    if coordinate_array.shape[1] == 0:
        raise ValueError("coordinates must contain at least one spatial dimension")

    return coordinate_array


__all__ = [
    "FloatArray",
    "IntArray",
    "bounding_box",
    "characteristic_element_length",
    "connected_average",
    "element_volumes",
    "entity_centers",
    "nodal_volumes",
]
