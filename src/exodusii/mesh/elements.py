# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Finite-element geometry primitives."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from typing import runtime_checkable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@runtime_checkable
class Element(Protocol):
    """Protocol implemented by supported element geometry classes."""

    coord: FloatArray

    @property
    def dimension(self) -> int:
        """Spatial dimension inferred from coordinates."""

    @property
    def center(self) -> FloatArray:
        """Element centroid."""

    @property
    def volume(self) -> float:
        """Element measure: area for 2-D elements, volume for 3-D elements."""

    def subdiv(self, intervals: int) -> FloatArray:
        """Return subelement centers."""

    def subcoord(self, intervals: int) -> FloatArray:
        """Return subelement nodal coordinates."""

    def subconn(self, intervals: int) -> IntArray:
        """Return subelement connectivity into :meth:`subcoord`."""

    def subvols(self, intervals: int) -> FloatArray:
        """Return subelement measures."""


@dataclass(slots=True)
class Quad4:
    """Four-node quadrilateral using Exodus node ordering."""

    coord: FloatArray

    dim = 2
    name = "QUAD4"
    nnode = 4

    def __init__(self, coord: npt.ArrayLike) -> None:
        self.coord = _coordinate_array(coord, expected_nodes=4, min_dimension=2)

    @property
    def dimension(self) -> int:
        return int(self.coord.shape[1])

    @property
    def center(self) -> FloatArray:
        return np.average(self.coord, axis=0)

    @property
    def volume(self) -> float:
        x = self.coord[:, 0]
        y = self.coord[:, 1]
        return float(
            0.5
            * abs(
                (x[0] * y[1] - x[1] * y[0])
                + (x[1] * y[2] - x[2] * y[1])
                + (x[2] * y[3] - x[3] * y[2])
                + (x[3] * y[0] - x[0] * y[3])
            )
        )

    def subdiv(self, intervals: int) -> FloatArray:
        intervals = _positive_intervals(intervals)
        points: list[FloatArray] = []
        for jj in range(intervals):
            eta = (0.5 + jj) / intervals
            for ii in range(intervals):
                xi = (0.5 + ii) / intervals
                points.append(_quad_bilinear(self.coord, xi, eta))
        return np.asarray(points, dtype=np.float64)

    def subcoord(self, intervals: int) -> FloatArray:
        if intervals <= 0:
            return np.asarray(self.coord, dtype=np.float64)

        points: list[FloatArray] = []
        for jj in range(intervals + 1):
            eta = jj / intervals
            for ii in range(intervals + 1):
                xi = ii / intervals
                points.append(_quad_bilinear(self.coord, xi, eta))
        return np.asarray(points, dtype=np.float64)

    def subconn(self, intervals: int) -> IntArray:
        intervals = _positive_intervals(intervals)
        conn: list[list[int]] = []
        row = intervals + 1
        for j in range(intervals):
            j0 = j * row
            j1 = (j + 1) * row
            for i in range(intervals):
                conn.append([j0 + i, j0 + i + 1, j1 + i + 1, j1 + i])
        return np.asarray(conn, dtype=np.int64)

    def subvols(self, intervals: int) -> FloatArray:
        subcoord = self.subcoord(intervals)
        subconn = self.subconn(intervals)
        return np.asarray([Quad4(subcoord[ix]).volume for ix in subconn], dtype=np.float64)


@dataclass(slots=True)
class Hex8:
    """Eight-node hexahedron using Exodus node ordering."""

    coord: FloatArray

    dim = 3
    name = "HEX8"
    nnode = 8

    def __init__(self, coord: npt.ArrayLike) -> None:
        self.coord = _coordinate_array(coord, expected_nodes=8, min_dimension=3)

    @property
    def dimension(self) -> int:
        return int(self.coord.shape[1])

    @property
    def center(self) -> FloatArray:
        return np.average(self.coord, axis=0)

    @property
    def volume(self) -> float:
        # Decompose the hex into five tetrahedra. This is exact for affine hexes
        # and robust for the simple generated subelements used by this package.
        tets = ((0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (1, 4, 5, 6), (3, 4, 6, 7))
        return float(sum(_tetrahedron_volume(self.coord[list(tet)]) for tet in tets))

    def subdiv(self, intervals: int) -> FloatArray:
        intervals = _positive_intervals(intervals)
        points: list[FloatArray] = []
        for kk in range(intervals):
            zeta = (0.5 + kk) / intervals
            for jj in range(intervals):
                eta = (0.5 + jj) / intervals
                for ii in range(intervals):
                    xi = (0.5 + ii) / intervals
                    points.append(_hex_trilinear(self.coord, xi, eta, zeta))
        return np.asarray(points, dtype=np.float64)

    def subcoord(self, intervals: int) -> FloatArray:
        if intervals <= 0:
            return np.asarray(self.coord, dtype=np.float64)

        points: list[FloatArray] = []
        for kk in range(intervals + 1):
            zeta = kk / intervals
            for jj in range(intervals + 1):
                eta = jj / intervals
                for ii in range(intervals + 1):
                    xi = ii / intervals
                    points.append(_hex_trilinear(self.coord, xi, eta, zeta))
        return np.asarray(points, dtype=np.float64)

    def subconn(self, intervals: int) -> IntArray:
        intervals = _positive_intervals(intervals)
        conn: list[list[int]] = []
        n = intervals + 1
        nn = n * n

        for k in range(intervals):
            k0 = k * nn
            k1 = (k + 1) * nn
            for j in range(intervals):
                j0k0 = k0 + j * n
                j1k0 = k0 + (j + 1) * n
                j0k1 = k1 + j * n
                j1k1 = k1 + (j + 1) * n
                for i in range(intervals):
                    conn.append(
                        [
                            j0k0 + i,
                            j0k0 + i + 1,
                            j1k0 + i + 1,
                            j1k0 + i,
                            j0k1 + i,
                            j0k1 + i + 1,
                            j1k1 + i + 1,
                            j1k1 + i,
                        ]
                    )

        return np.asarray(conn, dtype=np.int64)

    def subvols(self, intervals: int) -> FloatArray:
        subcoord = self.subcoord(intervals)
        subconn = self.subconn(intervals)
        return np.asarray([Hex8(subcoord[ix]).volume for ix in subconn], dtype=np.float64)


@dataclass(slots=True)
class Tri3:
    """Three-node triangle."""

    coord: FloatArray

    dim = 2
    name = "TRI3"
    nnode = 3

    def __init__(self, coord: npt.ArrayLike) -> None:
        self.coord = _coordinate_array(coord, expected_nodes=3, min_dimension=2)

    @property
    def dimension(self) -> int:
        return int(self.coord.shape[1])

    @property
    def center(self) -> FloatArray:
        return np.average(self.coord, axis=0)

    @property
    def volume(self) -> float:
        a = self.coord[0, :2]
        b = self.coord[1, :2]
        c = self.coord[2, :2]
        u = b - a
        v = c - a
        return float(0.5 * abs(u[0] * v[1] - u[1] * v[0]))

    def subdiv(self, intervals: int) -> FloatArray:
        coords, conn = _longest_edge_subdivision(self.coord, [[0, 1, 2]], intervals)
        return np.asarray([Tri3(coords[ix]).center for ix in conn], dtype=np.float64)

    def subcoord(self, intervals: int) -> FloatArray:
        coords, _ = _longest_edge_subdivision(self.coord, [[0, 1, 2]], intervals)
        return coords

    def subconn(self, intervals: int) -> IntArray:
        _, conn = _longest_edge_subdivision(self.coord, [[0, 1, 2]], intervals)
        return np.asarray(conn, dtype=np.int64)

    def subvols(self, intervals: int) -> FloatArray:
        coords, conn = _longest_edge_subdivision(self.coord, [[0, 1, 2]], intervals)
        return np.asarray([Tri3(coords[ix]).volume for ix in conn], dtype=np.float64)


@dataclass(slots=True)
class Tet4:
    """Four-node tetrahedron."""

    coord: FloatArray

    dim = 3
    name = "TET4"
    nnode = 4

    def __init__(self, coord: npt.ArrayLike) -> None:
        self.coord = _coordinate_array(coord, expected_nodes=4, min_dimension=3)

    @property
    def dimension(self) -> int:
        return int(self.coord.shape[1])

    @property
    def center(self) -> FloatArray:
        return np.average(self.coord, axis=0)

    @property
    def volume(self) -> float:
        return _tetrahedron_volume(self.coord)

    def subdiv(self, intervals: int) -> FloatArray:
        coords, conn = _longest_edge_subdivision(self.coord, [[0, 1, 2, 3]], intervals)
        return np.asarray([Tet4(coords[ix]).center for ix in conn], dtype=np.float64)

    def subcoord(self, intervals: int) -> FloatArray:
        coords, _ = _longest_edge_subdivision(self.coord, [[0, 1, 2, 3]], intervals)
        return coords

    def subconn(self, intervals: int) -> IntArray:
        _, conn = _longest_edge_subdivision(self.coord, [[0, 1, 2, 3]], intervals)
        return np.asarray(conn, dtype=np.int64)

    def subvols(self, intervals: int) -> FloatArray:
        coords, conn = _longest_edge_subdivision(self.coord, [[0, 1, 2, 3]], intervals)
        return np.asarray([Tet4(coords[ix]).volume for ix in conn], dtype=np.float64)


@dataclass(slots=True)
class Wedge6:
    """Six-node wedge/prism."""

    coord: FloatArray

    dim = 3
    name = "WEDGE6"
    nnode = 6

    def __init__(self, coord: npt.ArrayLike) -> None:
        self.coord = _coordinate_array(coord, expected_nodes=6, min_dimension=3)

    @property
    def dimension(self) -> int:
        return int(self.coord.shape[1])

    @property
    def center(self) -> FloatArray:
        return np.average(self.coord, axis=0)

    @property
    def volume(self) -> float:
        # Decompose into three tetrahedra.
        tets = ((0, 2, 1, 4), (0, 3, 5, 4), (0, 5, 2, 4))
        return float(sum(_tetrahedron_volume(self.coord[list(tet)]) for tet in tets))

    def subdiv(self, intervals: int) -> FloatArray:
        coords, conn = _wedge_subdivision(self.coord, intervals)
        return np.asarray([Wedge6(coords[ix]).center for ix in conn], dtype=np.float64)

    def subcoord(self, intervals: int) -> FloatArray:
        coords, _ = _wedge_subdivision(self.coord, intervals)
        return coords

    def subconn(self, intervals: int) -> IntArray:
        _, conn = _wedge_subdivision(self.coord, intervals)
        return np.asarray(conn, dtype=np.int64)

    def subvols(self, intervals: int) -> FloatArray:
        coords, conn = _wedge_subdivision(self.coord, intervals)
        return np.asarray([Wedge6(coords[ix]).volume for ix in conn], dtype=np.float64)


def element_factory(element_type: str | bytes, coord: npt.ArrayLike) -> Element:
    """Create an element geometry object from an Exodus element type string."""

    if isinstance(element_type, bytes):
        element_type = element_type.decode("ascii")

    key = element_type.strip().lower()

    if key in {"quad", "quad4", "shell4"}:
        return Quad4(coord)
    if key in {"hex", "hex8"}:
        return Hex8(coord)
    if key in {"tri", "tri3", "triangle", "triangle3"}:
        return Tri3(coord)
    if key in {"tet", "tet4", "tetra", "tetra4"}:
        return Tet4(coord)
    if key in {"wedge", "wedge6"}:
        return Wedge6(coord)

    raise ValueError(f"unknown element type {element_type!r}")


def _coordinate_array(
    coord: npt.ArrayLike, *, expected_nodes: int, min_dimension: int
) -> FloatArray:
    array = np.asarray(coord, dtype=np.float64)

    if array.ndim != 2:
        raise ValueError("element coordinates must be a two-dimensional array")
    if array.shape[0] != expected_nodes:
        raise ValueError(f"expected {expected_nodes} element nodes, got {array.shape[0]}")
    if array.shape[1] < min_dimension:
        raise ValueError(f"expected at least {min_dimension} coordinate dimensions")

    return array


def _positive_intervals(intervals: int) -> int:
    if not isinstance(intervals, int):
        raise TypeError("intervals must be an int")
    if intervals < 1:
        raise ValueError("intervals must be positive")
    return intervals


def _quad_bilinear(coord: FloatArray, xi: float, eta: float) -> FloatArray:
    return (
        (1.0 - eta) * (1.0 - xi) * coord[0]
        + (1.0 - eta) * xi * coord[1]
        + eta * xi * coord[2]
        + eta * (1.0 - xi) * coord[3]
    )


def _hex_trilinear(coord: FloatArray, xi: float, eta: float, zeta: float) -> FloatArray:
    return (
        (1.0 - zeta) * (1.0 - eta) * (1.0 - xi) * coord[0]
        + (1.0 - zeta) * (1.0 - eta) * xi * coord[1]
        + (1.0 - zeta) * eta * xi * coord[2]
        + (1.0 - zeta) * eta * (1.0 - xi) * coord[3]
        + zeta * (1.0 - eta) * (1.0 - xi) * coord[4]
        + zeta * (1.0 - eta) * xi * coord[5]
        + zeta * eta * xi * coord[6]
        + zeta * eta * (1.0 - xi) * coord[7]
    )


def _tetrahedron_volume(coord: FloatArray) -> float:
    a = coord[0]
    b = coord[1]
    c = coord[2]
    d = coord[3]
    return float(abs(np.dot(a - d, np.cross(b - d, c - d))) / 6.0)


def _distance_squared(a: FloatArray, b: FloatArray) -> float:
    delta = a - b
    return float(np.dot(delta, delta))


def _midpoint(a: FloatArray, b: FloatArray) -> FloatArray:
    return np.asarray((a + b) / 2.0, dtype=np.float64)


def _longest_edge_subdivision(
    initial_coords: FloatArray, initial_conn: Sequence[Sequence[int]], intervals: int
) -> tuple[FloatArray, list[list[int]]]:
    if intervals <= 1:
        return np.asarray(initial_coords, dtype=np.float64), [list(item) for item in initial_conn]

    coords: list[FloatArray] = [np.asarray(row, dtype=np.float64) for row in initial_coords]
    conn = [list(item) for item in initial_conn]
    iterations = 2 * intervals - 2

    for _ in range(iterations):
        next_conn: list[list[int]] = []

        for element in conn:
            i, j = _longest_edge(coords, element)
            remaining = [node for node in element if node not in {i, j}]

            midpoint = _midpoint(coords[i], coords[j])
            midpoint_index = len(coords)
            coords.append(midpoint)

            next_conn.append([i, midpoint_index, *remaining])
            next_conn.append([j, midpoint_index, *remaining])

        conn = next_conn

    return np.asarray(coords, dtype=np.float64), conn


def _longest_edge(coords: Sequence[FloatArray], element: Sequence[int]) -> tuple[int, int]:
    longest = -1.0
    pair = (element[0], element[1])

    for i, node_i in enumerate(element[:-1]):
        for node_j in element[i + 1 :]:
            distance = _distance_squared(coords[node_i], coords[node_j])
            if distance > longest:
                longest = distance
                pair = (node_i, node_j)

    return pair


def _wedge_subdivision(
    initial_coords: FloatArray, intervals: int
) -> tuple[FloatArray, list[list[int]]]:
    if intervals <= 1:
        return np.asarray(initial_coords, dtype=np.float64), [[0, 1, 2, 3, 4, 5]]

    coords: list[FloatArray] = [np.asarray(row, dtype=np.float64) for row in initial_coords]
    conn = [[0, 1, 2, 3, 4, 5]]
    iterations = 2 * intervals - 2

    for _ in range(iterations):
        next_conn: list[list[int]] = []

        for wedge in conn:
            i, j = _longest_bottom_wedge_edge(coords, wedge)
            k = next(node for node in (0, 1, 2) if node not in {i, j})

            gi = wedge[i]
            gj = wedge[j]
            gk = wedge[k]
            giu = wedge[i + 3]
            gju = wedge[j + 3]
            gku = wedge[k + 3]

            mp_bottom = _midpoint(coords[gi], coords[gj])
            mp_top = _midpoint(coords[giu], coords[gju])
            mp_i = _midpoint(coords[gi], coords[giu])
            mp_j = _midpoint(coords[gj], coords[gju])
            mp_k = _midpoint(coords[gk], coords[gku])
            mp_mid = _midpoint(mp_bottom, mp_top)

            first_new = len(coords)
            coords.extend([mp_bottom, mp_top, mp_i, mp_j, mp_k, mp_mid])

            mb = first_new
            mt = first_new + 1
            vi = first_new + 2
            vj = first_new + 3
            vk = first_new + 4
            vm = first_new + 5

            next_conn.extend(
                [
                    [gi, mb, gk, vi, vm, vk],
                    [mb, gj, gk, vm, vj, vk],
                    [vi, vm, vk, giu, mt, gku],
                    [vm, vj, vk, mt, gju, gku],
                ]
            )

        conn = next_conn

    return np.asarray(coords, dtype=np.float64), conn


def _longest_bottom_wedge_edge(
    coords: Sequence[FloatArray], wedge: Sequence[int]
) -> tuple[int, int]:
    longest = -1.0
    pair = (0, 1)

    for i in range(2):
        for j in range(i + 1, 3):
            distance = _distance_squared(coords[wedge[i]], coords[wedge[j]])
            if distance > longest:
                longest = distance
                pair = (i, j)

    return pair


__all__ = [
    "Element",
    "FloatArray",
    "Hex8",
    "IntArray",
    "Quad4",
    "Tet4",
    "Tri3",
    "Wedge6",
    "element_factory",
]
