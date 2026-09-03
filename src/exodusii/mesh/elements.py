# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Finite-element geometry primitives.

Provides concrete element classes (:class:`Quad4`, :class:`Hex8`,
:class:`Tri3`, :class:`Tet4`, :class:`Wedge6`) and the
:func:`element_factory` constructor, all of which implement the
:class:`Element` protocol.  Each class supports subdivision into
sub-elements via a common interface of ``subdiv``, ``subcoord``,
``subconn``, and ``subvols`` methods.
"""

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
    """Protocol implemented by supported element geometry classes.

    All concrete element classes in this module satisfy this protocol.
    Code that operates generically on elements should accept
    ``Element`` and use only the attributes and methods defined here.

    Attributes
    ----------
    coord : ndarray of float64, shape (n_nodes, dim)
        Nodal coordinates for the element.

    Notes
    -----
    The protocol is decorated with :func:`~typing.runtime_checkable`, so
    ``isinstance(obj, Element)`` works at runtime for duck-type checking.
    """

    coord: FloatArray

    @property
    def dimension(self) -> int:
        """Spatial dimension inferred from coordinates.

        Returns
        -------
        int
            Number of coordinate components per node (typically 2 or 3).
        """

    @property
    def center(self) -> FloatArray:
        """Element centroid.

        Returns
        -------
        ndarray of float64, shape (dim,)
            Centroid coordinates of the element.
        """

    @property
    def volume(self) -> float:
        """Element measure: area for 2-D elements, volume for 3-D elements.

        Returns
        -------
        float
            Geometric measure of the element (always non-negative).
        """

    def subdiv(self, intervals: int) -> FloatArray:
        """Return sub-element centroid coordinates.

        Parameters
        ----------
        intervals : int
            Number of subdivisions along each logical axis.  Must be a
            positive integer.  ``intervals=1`` returns the single centroid
            of the original element.

        Returns
        -------
        ndarray of float64, shape (n_sub, dim)
            Centroid of each sub-element produced by the subdivision.
        """

    def subcoord(self, intervals: int) -> FloatArray:
        """Return nodal coordinates for the sub-element mesh.

        Parameters
        ----------
        intervals : int
            Number of subdivisions along each logical axis.

        Returns
        -------
        ndarray of float64, shape (n_nodes, dim)
            All nodal coordinates needed by the sub-element connectivity
            returned by :meth:`subconn`.
        """

    def subconn(self, intervals: int) -> IntArray:
        """Return sub-element connectivity into :meth:`subcoord`.

        Parameters
        ----------
        intervals : int
            Number of subdivisions along each logical axis.

        Returns
        -------
        ndarray of int64, shape (n_sub, nodes_per_sub)
            Zero-based indices into the :meth:`subcoord` array for each
            sub-element.
        """

    def subvols(self, intervals: int) -> FloatArray:
        """Return sub-element measures (areas or volumes).

        Parameters
        ----------
        intervals : int
            Number of subdivisions along each logical axis.

        Returns
        -------
        ndarray of float64, shape (n_sub,)
            Geometric measure (area or volume) of each sub-element.
        """


@dataclass(slots=True)
class Quad4:
    """Four-node quadrilateral using Exodus node ordering.

    The four nodes are ordered counter-clockwise:
    ``0 → 1 → 2 → 3``, where node 0 is the lower-left corner.
    The element lives in the XY-plane; a third coordinate column is
    accepted and carried through but ignored in area calculations.

    Parameters
    ----------
    coord : array_like, shape (4, dim)
        Nodal coordinates with ``dim >= 2``.  Converted to
        ``float64`` on construction.

    Raises
    ------
    ValueError
        If ``coord`` does not have exactly 4 rows or fewer than 2 columns,
        or if it is not two-dimensional.

    Notes
    -----
    Area is computed via the shoelace formula applied to the XY
    coordinates:

    .. math::

        A = \\frac{1}{2}\\left|\\sum_{i=0}^{3}(x_i y_{i+1} - x_{i+1} y_i)\\right|

    where indices are taken modulo 4.

    Subdivision via :meth:`subdiv` uses a bilinear mapping from the
    reference square ``[0, 1]^2`` to physical space.

    Examples
    --------
    >>> import numpy as np
    >>> coord = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    >>> q = Quad4(coord)
    >>> q.volume
    1.0
    >>> q.center
    array([0.5, 0.5])
    >>> q.subdiv(2).shape
    (4, 2)
    """

    coord: FloatArray

    dim = 2
    name = "QUAD4"
    nnode = 4

    def __init__(self, coord: npt.ArrayLike) -> None:
        self.coord = _coordinate_array(coord, expected_nodes=4, min_dimension=2)

    @property
    def dimension(self) -> int:
        """Spatial dimension inferred from coordinates.

        Returns
        -------
        int
            Number of coordinate components per node (2 or 3).
        """

        return int(self.coord.shape[1])

    @property
    def center(self) -> FloatArray:
        """Element centroid computed as the average of the four node positions.

        Returns
        -------
        ndarray of float64, shape (dim,)
            Centroid coordinates.
        """

        return np.average(self.coord, axis=0)

    @property
    def volume(self) -> float:
        """Element area computed with the shoelace formula.

        Returns
        -------
        float
            Area of the quadrilateral in the XY-plane.  Always
            non-negative.
        """

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
        """Return the centroid of each sub-quadrilateral.

        Divides the element into ``intervals**2`` sub-quadrilaterals by
        sampling the bilinear map on a regular ``intervals x intervals``
        grid of cell centres.

        Parameters
        ----------
        intervals : int
            Number of subdivisions along each parametric axis.  Must be a
            positive integer.

        Returns
        -------
        ndarray of float64, shape (intervals**2, dim)
            Physical centroid coordinates of each sub-element, ordered
            row-major (eta varies slowest).

        Examples
        --------
        >>> import numpy as np
        >>> coord = np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.]])
        >>> Quad4(coord).subdiv(2)
        array([[0.5, 0.5],
               [1.5, 0.5],
               [0.5, 1.5],
               [1.5, 1.5]])
        """

        intervals = _positive_intervals(intervals)
        points: list[FloatArray] = []
        for jj in range(intervals):
            eta = (0.5 + jj) / intervals
            for ii in range(intervals):
                xi = (0.5 + ii) / intervals
                points.append(_quad_bilinear(self.coord, xi, eta))
        return np.asarray(points, dtype=np.float64)

    def subcoord(self, intervals: int) -> FloatArray:
        """Return nodal coordinates for the sub-element mesh.

        Produces an ``(intervals+1) x (intervals+1)`` grid of physical
        nodes by evaluating the bilinear map at each parametric grid point.

        Parameters
        ----------
        intervals : int
            Number of subdivisions along each parametric axis.  Values
            ``<= 0`` return the original four nodal coordinates unchanged.

        Returns
        -------
        ndarray of float64, shape ((intervals+1)**2, dim)
            Physical coordinates of all sub-mesh nodes, ordered row-major
            (eta varies slowest).

        Examples
        --------
        >>> import numpy as np
        >>> coord = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        >>> Quad4(coord).subcoord(1).shape
        (4, 2)
        """

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
        """Return sub-element connectivity into :meth:`subcoord`.

        Each sub-element is a ``Quad4`` described by four zero-based
        indices into the :meth:`subcoord` array, ordered counter-clockwise.

        Parameters
        ----------
        intervals : int
            Number of subdivisions along each parametric axis.  Must be a
            positive integer.

        Returns
        -------
        ndarray of int64, shape (intervals**2, 4)
            Connectivity for each sub-quadrilateral.

        Examples
        --------
        >>> import numpy as np
        >>> coord = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        >>> Quad4(coord).subconn(2)
        array([[0, 1, 4, 3],
               [1, 2, 5, 4],
               [3, 4, 7, 6],
               [4, 5, 8, 7]])
        """

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
        """Return the area of each sub-quadrilateral.

        Parameters
        ----------
        intervals : int
            Number of subdivisions along each parametric axis.  Must be a
            positive integer.

        Returns
        -------
        ndarray of float64, shape (intervals**2,)
            Area of each sub-element.

        Examples
        --------
        >>> import numpy as np
        >>> coord = np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.]])
        >>> Quad4(coord).subvols(2)
        array([1., 1., 1., 1.])
        """

        subcoord = self.subcoord(intervals)
        subconn = self.subconn(intervals)
        return np.asarray([Quad4(subcoord[ix]).volume for ix in subconn], dtype=np.float64)


@dataclass(slots=True)
class Hex8:
    """Eight-node hexahedron using Exodus node ordering.

    Nodes 0-3 form the bottom face and nodes 4-7 form the top face, both
    ordered counter-clockwise when viewed from outside the element.

    Parameters
    ----------
    coord : array_like, shape (8, dim)
        Nodal coordinates with ``dim >= 3``.  Converted to ``float64``
        on construction.

    Raises
    ------
    ValueError
        If ``coord`` does not have exactly 8 rows or fewer than 3 columns,
        or if it is not two-dimensional.

    Notes
    -----
    Volume is computed by decomposing the hexahedron into five tetrahedra.
    This is exact for affine hexahedra and robust for the sub-elements
    produced by :meth:`subcoord`.

    Subdivision via :meth:`subdiv` uses a trilinear map from the reference
    cube ``[0, 1]^3`` to physical space.

    Examples
    --------
    >>> import numpy as np
    >>> coord = np.array([
    ...     [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
    ...     [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
    ... ])
    >>> h = Hex8(coord)
    >>> h.volume
    1.0
    >>> h.center
    array([0.5, 0.5, 0.5])
    """

    coord: FloatArray

    dim = 3
    name = "HEX8"
    nnode = 8

    def __init__(self, coord: npt.ArrayLike) -> None:
        self.coord = _coordinate_array(coord, expected_nodes=8, min_dimension=3)

    @property
    def dimension(self) -> int:
        """Spatial dimension inferred from coordinates.

        Returns
        -------
        int
            Number of coordinate components per node (typically 3).
        """

        return int(self.coord.shape[1])

    @property
    def center(self) -> FloatArray:
        """Element centroid computed as the average of the eight node positions.

        Returns
        -------
        ndarray of float64, shape (dim,)
            Centroid coordinates.
        """

        return np.average(self.coord, axis=0)

    @property
    def volume(self) -> float:
        """Element volume computed via five-tetrahedron decomposition.

        Returns
        -------
        float
            Volume of the hexahedron.  Always non-negative.
        """

        # Decompose the hex into five tetrahedra. This is exact for affine hexes
        # and robust for the simple generated subelements used by this package.
        tets = ((0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (1, 4, 5, 6), (3, 4, 6, 7))
        return float(sum(_tetrahedron_volume(self.coord[list(tet)]) for tet in tets))

    def subdiv(self, intervals: int) -> FloatArray:
        """Return the centroid of each sub-hexahedron.

        Divides the element into ``intervals**3`` sub-hexahedra by sampling
        the trilinear map on a regular ``intervals x intervals x intervals``
        grid of cell centres.

        Parameters
        ----------
        intervals : int
            Number of subdivisions along each parametric axis.  Must be a
            positive integer.

        Returns
        -------
        ndarray of float64, shape (intervals**3, dim)
            Physical centroid coordinates, ordered zeta → eta → xi
            (xi varies fastest).

        Examples
        --------
        >>> import numpy as np
        >>> coord = np.array([
        ...     [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
        ...     [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
        ... ])
        >>> Hex8(coord).subdiv(1)
        array([[0.5, 0.5, 0.5]])
        """

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
        """Return nodal coordinates for the sub-element mesh.

        Produces an ``(intervals+1)^3`` grid of physical nodes by evaluating
        the trilinear map at each parametric grid point.

        Parameters
        ----------
        intervals : int
            Number of subdivisions along each parametric axis.  Values
            ``<= 0`` return the original eight nodal coordinates unchanged.

        Returns
        -------
        ndarray of float64, shape ((intervals+1)**3, dim)
            Physical coordinates of all sub-mesh nodes, ordered
            zeta → eta → xi (xi varies fastest).
        """

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
        """Return sub-element connectivity into :meth:`subcoord`.

        Each sub-element is a ``Hex8`` described by eight zero-based indices
        into the :meth:`subcoord` array in Exodus node ordering.

        Parameters
        ----------
        intervals : int
            Number of subdivisions along each parametric axis.  Must be a
            positive integer.

        Returns
        -------
        ndarray of int64, shape (intervals**3, 8)
            Connectivity for each sub-hexahedron.
        """

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
        """Return the volume of each sub-hexahedron.

        Parameters
        ----------
        intervals : int
            Number of subdivisions along each parametric axis.  Must be a
            positive integer.

        Returns
        -------
        ndarray of float64, shape (intervals**3,)
            Volume of each sub-element.

        Examples
        --------
        >>> import numpy as np
        >>> coord = np.array([
        ...     [0., 0., 0.], [2., 0., 0.], [2., 2., 0.], [0., 2., 0.],
        ...     [0., 0., 2.], [2., 0., 2.], [2., 2., 2.], [0., 2., 2.],
        ... ])
        >>> Hex8(coord).subvols(2).sum()
        8.0
        """

        subcoord = self.subcoord(intervals)
        subconn = self.subconn(intervals)
        return np.asarray([Hex8(subcoord[ix]).volume for ix in subconn], dtype=np.float64)


@dataclass(slots=True)
class Tri3:
    """Three-node triangle using Exodus node ordering.

    Nodes are ordered counter-clockwise: ``0 → 1 → 2``.  The triangle
    lives in the XY-plane; an optional third coordinate column is accepted
    and carried through but ignored in area calculations.

    Parameters
    ----------
    coord : array_like, shape (3, dim)
        Nodal coordinates with ``dim >= 2``.  Converted to ``float64``
        on construction.

    Raises
    ------
    ValueError
        If ``coord`` does not have exactly 3 rows or fewer than 2 columns,
        or if it is not two-dimensional.

    Notes
    -----
    Area is computed via the cross-product formula:

    .. math::

        A = \\frac{1}{2}|\\mathbf{u} \\times \\mathbf{v}|

    where :math:`\\mathbf{u} = P_1 - P_0` and
    :math:`\\mathbf{v} = P_2 - P_0`.

    Subdivision uses the longest-edge bisection algorithm, which
    recursively bisects the longest edge of each triangle.

    Examples
    --------
    >>> import numpy as np
    >>> coord = np.array([[0., 0.], [1., 0.], [0., 1.]])
    >>> t = Tri3(coord)
    >>> t.volume
    0.5
    >>> t.center
    array([0.33333333, 0.33333333])
    """

    coord: FloatArray

    dim = 2
    name = "TRI3"
    nnode = 3

    def __init__(self, coord: npt.ArrayLike) -> None:
        self.coord = _coordinate_array(coord, expected_nodes=3, min_dimension=2)

    @property
    def dimension(self) -> int:
        """Spatial dimension inferred from coordinates.

        Returns
        -------
        int
            Number of coordinate components per node (2 or 3).
        """

        return int(self.coord.shape[1])

    @property
    def center(self) -> FloatArray:
        """Element centroid computed as the average of the three node positions.

        Returns
        -------
        ndarray of float64, shape (dim,)
            Centroid (barycentre) coordinates.
        """

        return np.average(self.coord, axis=0)

    @property
    def volume(self) -> float:
        """Element area computed via the cross-product formula.

        Returns
        -------
        float
            Area of the triangle in the XY-plane.  Always non-negative.
        """

        a = self.coord[0, :2]
        b = self.coord[1, :2]
        c = self.coord[2, :2]
        u = b - a
        v = c - a
        return float(0.5 * abs(u[0] * v[1] - u[1] * v[0]))

    def subdiv(self, intervals: int) -> FloatArray:
        """Return the centroid of each sub-triangle.

        Subdivides by repeated longest-edge bisection.  For
        ``intervals=n``, the result contains ``2**(2*(n-1))`` sub-triangles
        when ``n >= 1`` (``1`` for ``intervals=1``).

        Parameters
        ----------
        intervals : int
            Subdivision level.  ``intervals=1`` returns the single centroid
            of the original triangle.

        Returns
        -------
        ndarray of float64, shape (n_sub, dim)
            Physical centroid coordinates of each sub-triangle.

        Examples
        --------
        >>> import numpy as np
        >>> coord = np.array([[0., 0.], [1., 0.], [0., 1.]])
        >>> Tri3(coord).subdiv(1)
        array([[0.33333333, 0.33333333]])
        """

        coords, conn = _longest_edge_subdivision(self.coord, [[0, 1, 2]], intervals)
        return np.asarray([Tri3(coords[ix]).center for ix in conn], dtype=np.float64)

    def subcoord(self, intervals: int) -> FloatArray:
        """Return nodal coordinates for the sub-element mesh.

        Parameters
        ----------
        intervals : int
            Subdivision level.

        Returns
        -------
        ndarray of float64, shape (n_nodes, dim)
            All nodal coordinates needed by the sub-element connectivity
            returned by :meth:`subconn`.
        """

        coords, _ = _longest_edge_subdivision(self.coord, [[0, 1, 2]], intervals)
        return coords

    def subconn(self, intervals: int) -> IntArray:
        """Return sub-element connectivity into :meth:`subcoord`.

        Parameters
        ----------
        intervals : int
            Subdivision level.

        Returns
        -------
        ndarray of int64, shape (n_sub, 3)
            Zero-based node indices for each sub-triangle.
        """

        _, conn = _longest_edge_subdivision(self.coord, [[0, 1, 2]], intervals)
        return np.asarray(conn, dtype=np.int64)

    def subvols(self, intervals: int) -> FloatArray:
        """Return the area of each sub-triangle.

        Parameters
        ----------
        intervals : int
            Subdivision level.

        Returns
        -------
        ndarray of float64, shape (n_sub,)
            Area of each sub-triangle.

        Examples
        --------
        >>> import numpy as np
        >>> coord = np.array([[0., 0.], [1., 0.], [0., 1.]])
        >>> Tri3(coord).subvols(1)
        array([0.5])
        """

        coords, conn = _longest_edge_subdivision(self.coord, [[0, 1, 2]], intervals)
        return np.asarray([Tri3(coords[ix]).volume for ix in conn], dtype=np.float64)


@dataclass(slots=True)
class Tet4:
    """Four-node tetrahedron using Exodus node ordering.

    Nodes 0-2 form the base face (counter-clockwise when viewed from
    outside) and node 3 is the apex.

    Parameters
    ----------
    coord : array_like, shape (4, dim)
        Nodal coordinates with ``dim >= 3``.  Converted to ``float64``
        on construction.

    Raises
    ------
    ValueError
        If ``coord`` does not have exactly 4 rows or fewer than 3 columns,
        or if it is not two-dimensional.

    Notes
    -----
    Volume is computed as:

    .. math::

        V = \\frac{1}{6}\\left|(\\mathbf{a}-\\mathbf{d}) \\cdot
            [(\\mathbf{b}-\\mathbf{d}) \\times (\\mathbf{c}-\\mathbf{d})]\\right|

    Subdivision uses the longest-edge bisection algorithm adapted for
    tetrahedra.

    Examples
    --------
    >>> import numpy as np
    >>> coord = np.array([
    ...     [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
    ... ])
    >>> t = Tet4(coord)
    >>> round(t.volume, 10)
    0.1666666667
    >>> t.center
    array([0.25, 0.25, 0.25])
    """

    coord: FloatArray

    dim = 3
    name = "TET4"
    nnode = 4

    def __init__(self, coord: npt.ArrayLike) -> None:
        self.coord = _coordinate_array(coord, expected_nodes=4, min_dimension=3)

    @property
    def dimension(self) -> int:
        """Spatial dimension inferred from coordinates.

        Returns
        -------
        int
            Number of coordinate components per node (typically 3).
        """

        return int(self.coord.shape[1])

    @property
    def center(self) -> FloatArray:
        """Element centroid computed as the average of the four node positions.

        Returns
        -------
        ndarray of float64, shape (dim,)
            Centroid coordinates.
        """

        return np.average(self.coord, axis=0)

    @property
    def volume(self) -> float:
        """Element volume computed via the scalar triple product.

        Returns
        -------
        float
            Volume of the tetrahedron.  Always non-negative.
        """

        return _tetrahedron_volume(self.coord)

    def subdiv(self, intervals: int) -> FloatArray:
        """Return the centroid of each sub-tetrahedron.

        Subdivides by repeated longest-edge bisection adapted for
        tetrahedra.

        Parameters
        ----------
        intervals : int
            Subdivision level.  ``intervals=1`` returns the single centroid
            of the original tetrahedron.

        Returns
        -------
        ndarray of float64, shape (n_sub, dim)
            Physical centroid coordinates of each sub-tetrahedron.

        Examples
        --------
        >>> import numpy as np
        >>> coord = np.array([
        ...     [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        ... ])
        >>> Tet4(coord).subdiv(1)
        array([[0.25, 0.25, 0.25]])
        """

        coords, conn = _longest_edge_subdivision(self.coord, [[0, 1, 2, 3]], intervals)
        return np.asarray([Tet4(coords[ix]).center for ix in conn], dtype=np.float64)

    def subcoord(self, intervals: int) -> FloatArray:
        """Return nodal coordinates for the sub-element mesh.

        Parameters
        ----------
        intervals : int
            Subdivision level.

        Returns
        -------
        ndarray of float64, shape (n_nodes, dim)
            All nodal coordinates needed by the sub-element connectivity
            returned by :meth:`subconn`.
        """

        coords, _ = _longest_edge_subdivision(self.coord, [[0, 1, 2, 3]], intervals)
        return coords

    def subconn(self, intervals: int) -> IntArray:
        """Return sub-element connectivity into :meth:`subcoord`.

        Parameters
        ----------
        intervals : int
            Subdivision level.

        Returns
        -------
        ndarray of int64, shape (n_sub, 4)
            Zero-based node indices for each sub-tetrahedron.
        """

        _, conn = _longest_edge_subdivision(self.coord, [[0, 1, 2, 3]], intervals)
        return np.asarray(conn, dtype=np.int64)

    def subvols(self, intervals: int) -> FloatArray:
        """Return the volume of each sub-tetrahedron.

        Parameters
        ----------
        intervals : int
            Subdivision level.

        Returns
        -------
        ndarray of float64, shape (n_sub,)
            Volume of each sub-tetrahedron.

        Examples
        --------
        >>> import numpy as np
        >>> coord = np.array([
        ...     [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        ... ])
        >>> abs(Tet4(coord).subvols(1).sum() - 1/6) < 1e-12
        True
        """

        coords, conn = _longest_edge_subdivision(self.coord, [[0, 1, 2, 3]], intervals)
        return np.asarray([Tet4(coords[ix]).volume for ix in conn], dtype=np.float64)


@dataclass(slots=True)
class Wedge6:
    """Six-node wedge (triangular prism) using Exodus node ordering.

    Nodes 0-2 form the triangular bottom face (counter-clockwise when
    viewed from below) and nodes 3-5 form the corresponding top face.

    Parameters
    ----------
    coord : array_like, shape (6, dim)
        Nodal coordinates with ``dim >= 3``.  Converted to ``float64``
        on construction.

    Raises
    ------
    ValueError
        If ``coord`` does not have exactly 6 rows or fewer than 3 columns,
        or if it is not two-dimensional.

    Notes
    -----
    Volume is computed by decomposing the wedge into three tetrahedra:
    ``(0,2,1,4)``, ``(0,3,5,4)``, and ``(0,5,2,4)``.  This decomposition
    is exact for affine wedges.

    Subdivision uses a dedicated longest-edge algorithm that bisects the
    longest edge of the triangular cross-section while respecting the
    prismatic topology.

    Examples
    --------
    >>> import numpy as np
    >>> coord = np.array([
    ...     [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
    ...     [0., 0., 1.], [1., 0., 1.], [0., 1., 1.],
    ... ])
    >>> w = Wedge6(coord)
    >>> round(w.volume, 10)
    0.5
    >>> w.center
    array([0.33333333, 0.33333333, 0.5       ])
    """

    coord: FloatArray

    dim = 3
    name = "WEDGE6"
    nnode = 6

    def __init__(self, coord: npt.ArrayLike) -> None:
        self.coord = _coordinate_array(coord, expected_nodes=6, min_dimension=3)

    @property
    def dimension(self) -> int:
        """Spatial dimension inferred from coordinates.

        Returns
        -------
        int
            Number of coordinate components per node (typically 3).
        """

        return int(self.coord.shape[1])

    @property
    def center(self) -> FloatArray:
        """Element centroid computed as the average of the six node positions.

        Returns
        -------
        ndarray of float64, shape (dim,)
            Centroid coordinates.
        """

        return np.average(self.coord, axis=0)

    @property
    def volume(self) -> float:
        """Element volume computed via three-tetrahedron decomposition.

        Returns
        -------
        float
            Volume of the wedge.  Always non-negative.
        """

        # Decompose into three tetrahedra.
        tets = ((0, 2, 1, 4), (0, 3, 5, 4), (0, 5, 2, 4))
        return float(sum(_tetrahedron_volume(self.coord[list(tet)]) for tet in tets))

    def subdiv(self, intervals: int) -> FloatArray:
        """Return the centroid of each sub-wedge.

        Subdivides by the dedicated wedge longest-edge bisection algorithm.

        Parameters
        ----------
        intervals : int
            Subdivision level.  ``intervals=1`` returns the single centroid
            of the original wedge.

        Returns
        -------
        ndarray of float64, shape (n_sub, dim)
            Physical centroid coordinates of each sub-wedge.

        Examples
        --------
        >>> import numpy as np
        >>> coord = np.array([
        ...     [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
        ...     [0., 0., 1.], [1., 0., 1.], [0., 1., 1.],
        ... ])
        >>> Wedge6(coord).subdiv(1).shape
        (1, 3)
        """

        coords, conn = _wedge_subdivision(self.coord, intervals)
        return np.asarray([Wedge6(coords[ix]).center for ix in conn], dtype=np.float64)

    def subcoord(self, intervals: int) -> FloatArray:
        """Return nodal coordinates for the sub-element mesh.

        Parameters
        ----------
        intervals : int
            Subdivision level.

        Returns
        -------
        ndarray of float64, shape (n_nodes, dim)
            All nodal coordinates needed by the sub-element connectivity
            returned by :meth:`subconn`.
        """

        coords, _ = _wedge_subdivision(self.coord, intervals)
        return coords

    def subconn(self, intervals: int) -> IntArray:
        """Return sub-element connectivity into :meth:`subcoord`.

        Parameters
        ----------
        intervals : int
            Subdivision level.

        Returns
        -------
        ndarray of int64, shape (n_sub, 6)
            Zero-based node indices for each sub-wedge in Exodus ordering.
        """

        _, conn = _wedge_subdivision(self.coord, intervals)
        return np.asarray(conn, dtype=np.int64)

    def subvols(self, intervals: int) -> FloatArray:
        """Return the volume of each sub-wedge.

        Parameters
        ----------
        intervals : int
            Subdivision level.

        Returns
        -------
        ndarray of float64, shape (n_sub,)
            Volume of each sub-wedge.

        Examples
        --------
        >>> import numpy as np
        >>> coord = np.array([
        ...     [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
        ...     [0., 0., 1.], [1., 0., 1.], [0., 1., 1.],
        ... ])
        >>> abs(Wedge6(coord).subvols(1).sum() - 0.5) < 1e-12
        True
        """

        coords, conn = _wedge_subdivision(self.coord, intervals)
        return np.asarray([Wedge6(coords[ix]).volume for ix in conn], dtype=np.float64)


def element_factory(element_type: str | bytes, coord: npt.ArrayLike) -> Element:
    """Create an element geometry object from an Exodus element type string.

    Parameters
    ----------
    element_type : str or bytes
        Exodus element type identifier.  Case-insensitive; leading/trailing
        whitespace is ignored.  Accepted aliases:

        * ``"quad"``, ``"quad4"``, ``"shell4"`` → :class:`Quad4`
        * ``"hex"``, ``"hex8"`` → :class:`Hex8`
        * ``"tri"``, ``"tri3"``, ``"triangle"``, ``"triangle3"`` → :class:`Tri3`
        * ``"tet"``, ``"tet4"``, ``"tetra"``, ``"tetra4"`` → :class:`Tet4`
        * ``"wedge"``, ``"wedge6"`` → :class:`Wedge6`

        If ``bytes`` are passed they are decoded as ASCII before lookup.
    coord : array_like
        Nodal coordinates passed directly to the element constructor.  The
        required shape depends on the element type (see each class's
        docstring).

    Returns
    -------
    Element
        A concrete element instance satisfying the :class:`Element`
        protocol.

    Raises
    ------
    ValueError
        If ``element_type`` is not a recognised alias.

    Examples
    --------
    >>> import numpy as np
    >>> coord = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    >>> el = element_factory("QUAD4", coord)
    >>> type(el).__name__
    'Quad4'
    >>> el.volume
    1.0
    >>> element_factory(b"hex8", np.zeros((8, 3)))
    Hex8(coord=...)
    """

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
