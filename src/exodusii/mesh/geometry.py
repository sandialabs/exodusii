# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Array-based mesh geometry utilities.

This module provides functions for computing geometric quantities directly
from connectivity and coordinate arrays without requiring an open Exodus
file.  All functions operate on plain NumPy arrays and carry no I/O
dependencies.

Functions
---------
connected_average
    Average nodal values over element connectivity.
entity_centers
    Geometric centroids of mesh entities.
element_volumes
    Areas (2-D) or volumes (3-D) of individual elements.
nodal_volumes
    Element volumes distributed equally to connected nodes.
characteristic_element_length
    Mesh-averaged characteristic element size (``V^{1/d}``).
bounding_box
    Coordinate-wise minimum and maximum extents.
"""

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
    """Compute geometric centroids of mesh entities.

    Each entity center is the simple arithmetic mean of its node coordinates
    (i.e. the centroid of the node cloud, not the true volumetric centroid).

    Parameters
    ----------
    conn : array_like
        Zero-based connectivity array, shape ``(n_elems, n_nodes_per_elem)``.
        Each row lists the node indices for one entity.
    coords : array_like
        Nodal coordinate array, shape ``(n_nodes, dim)``.  ``dim`` is
        typically 1, 2, or 3.

    Returns
    -------
    ndarray
        Center coordinates, shape ``(n_elems, dim)``.  Each row is the
        centroid of the corresponding entity.

    Raises
    ------
    ValueError
        When *coords* is not two-dimensional or *conn* is not two-dimensional.
    IndexError
        When *conn* references a node index outside *coords*.

    Examples
    --------
    >>> import numpy as np
    >>> coords = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    >>> conn   = np.array([[0, 1, 2, 3]])
    >>> entity_centers(conn, coords)
    array([[0.5, 0.5]])
    """

    centers = connected_average(conn, coords)

    if centers.ndim != 2:
        raise ValueError("coordinates must be two-dimensional")

    return centers


def element_volumes(
    element_type: str | bytes, conn: npt.ArrayLike, coords: npt.ArrayLike
) -> FloatArray:
    """Compute element measures (areas or volumes) for all elements.

    For 2-D elements this returns areas; for 3-D elements this returns
    volumes.  Connectivity indices are zero-based.

    Parameters
    ----------
    element_type : str or bytes
        Exodus element-type string (e.g. ``"quad4"``, ``"hex8"``,
        ``"tri3"``).  Passed to the element factory to select the
        appropriate quadrature/volume formula.
    conn : array_like
        Zero-based connectivity array, shape ``(n_elems, n_nodes_per_elem)``.
    coords : array_like
        Nodal coordinate array, shape ``(n_nodes, dim)``.

    Returns
    -------
    ndarray
        Per-element measures, shape ``(n_elems,)``.  Values are signed
        (positive for counter-clockwise / outward-normal elements).

    Examples
    --------
    >>> import numpy as np
    >>> coords = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    >>> conn   = np.array([[0, 1, 2, 3]])
    >>> element_volumes("quad4", conn, coords)
    array([1.])
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
    """Distribute element measures equally to connected nodes.

    Each element's area or volume is divided equally among its nodes and
    accumulated.  The result can be used as a nodal weight (e.g. for
    lumped-mass assembly or nodal averaging).

    Parameters
    ----------
    element_type : str or bytes
        Exodus element-type string; forwarded to :func:`element_volumes`.
    conn : array_like
        Zero-based connectivity array, shape ``(n_elems, n_nodes_per_elem)``.
    coords : array_like
        Nodal coordinate array, shape ``(n_nodes, dim)``.
    num_nodes : int or None, optional
        Total number of nodes in the mesh.  When ``None`` (default),
        inferred from the first axis of *coords*.  Explicit values are
        needed when *conn* does not reference all nodes.

    Returns
    -------
    ndarray
        Per-node accumulated measure, shape ``(num_nodes,)``.  Nodes not
        referenced by any element have a value of ``0.0``.

    Raises
    ------
    ValueError
        When ``num_nodes`` is negative.
    IndexError
        When *conn* references a node index ``>= num_nodes``.

    Notes
    -----
    The distribution algorithm is:

    .. code-block:: text

        for each element e with measure V_e and n_e connected nodes:
            nodal_volume[node_i] += V_e / n_e   for each node_i in element e

    This is the standard equal-weight (consistent) nodal distribution used
    in lumped-mass matrix assembly.

    Examples
    --------
    >>> import numpy as np
    >>> coords = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    >>> conn   = np.array([[0, 1, 2, 3]])
    >>> nodal_volumes("quad4", conn, coords)
    array([0.25, 0.25, 0.25, 0.25])
    """

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
    r"""Compute the average characteristic element length.

    The characteristic length of a single element is :math:`V^{1/d}`, where
    :math:`V` is the element area (2-D) or volume (3-D) and :math:`d` is the
    spatial dimension.  This function returns the mean across all elements.

    Parameters
    ----------
    element_type : str or bytes
        Exodus element-type string; forwarded to :func:`element_volumes`.
    conn : array_like
        Zero-based connectivity array, shape ``(n_elems, n_nodes_per_elem)``.
    coords : array_like
        Nodal coordinate array, shape ``(n_nodes, dim)``.
    dimension : int or None, optional
        Spatial dimension *d* used in the exponent ``1/d``.  When ``None``
        (default), inferred from the second axis of *coords*.

    Returns
    -------
    float
        Mean characteristic element length over all elements.

    Raises
    ------
    ValueError
        When *dimension* is less than 1 or the mesh contains no elements.

    Examples
    --------
    Unit square with one quad element (area = 1, dim = 2):

    >>> import numpy as np
    >>> coords = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    >>> conn   = np.array([[0, 1, 2, 3]])
    >>> characteristic_element_length("quad4", conn, coords)
    1.0

    Two equal unit squares (both give length 1.0):

    >>> coords2 = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],
    ...                      [2.,0.],[2.,1.]])
    >>> conn2   = np.array([[0,1,2,3],[1,4,5,2]])
    >>> characteristic_element_length("quad4", conn2, coords2)
    1.0
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
    """Return coordinate-wise minimum and maximum extents.

    Parameters
    ----------
    coords : array_like
        Nodal coordinate array, shape ``(n_nodes, dim)``.  Must contain at
        least one node.

    Returns
    -------
    lo : ndarray
        Array of minimum coordinate values, shape ``(dim,)``.
        ``lo[i]`` is ``min(coords[:, i])``.
    hi : ndarray
        Array of maximum coordinate values, shape ``(dim,)``.
        ``hi[i]`` is ``max(coords[:, i])``.

    Raises
    ------
    ValueError
        When *coords* is not two-dimensional or contains no nodes.

    Examples
    --------
    >>> import numpy as np
    >>> coords = np.array([[0., 0.], [3., 1.], [1., 4.]])
    >>> lo, hi = bounding_box(coords)
    >>> lo
    array([0., 0.])
    >>> hi
    array([3., 4.])
    """

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
