# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Geometric and temporal region predicates."""

from dataclasses import dataclass
from typing import Protocol
from typing import cast
from typing import runtime_checkable

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


@runtime_checkable
class Region(Protocol):
    """Protocol for geometric regions."""

    @property
    def dimension(self) -> int:
        """Spatial dimension of the region."""

    def contains(self, points: npt.ArrayLike) -> BoolArray:
        """Return whether point or points are inside the region."""


@runtime_checkable
class TimeDomain(Protocol):
    """Protocol for time-domain predicates."""

    def contains(self, times: npt.ArrayLike) -> BoolArray:
        """Return whether time or times are inside the domain."""


class _RegionOps:
    """Mixin providing boolean composition operators for regions.

    Any region implementing ``contains(points) -> BoolArray`` gains
    intersection (``&``), union (``|``), and complement (``~``) so that
    multi-condition selections compose into a single region object.  The
    composed region's ``contains`` combines the child masks elementwise, so it
    works unchanged with :func:`region_stats` / :func:`region_mass` (which only
    ever call ``region.contains``).
    """

    __slots__ = ()

    def contains(self, points: npt.ArrayLike) -> BoolArray:  # pragma: no cover - overridden
        raise NotImplementedError

    def __and__(self, other: "Region") -> "Intersection":
        return Intersection((cast("Region", self), other))

    def __or__(self, other: "Region") -> "Union":
        return Union((cast("Region", self), other))

    def __invert__(self) -> "Complement":
        return Complement(cast("Region", self))


@dataclass(frozen=True, slots=True)
class Circle(_RegionOps):
    """Closed 2-D circle."""

    center: FloatArray
    radius: float

    def __init__(self, center: npt.ArrayLike, radius: float) -> None:
        object.__setattr__(self, "center", _point(center, dimension=2, name="center"))
        object.__setattr__(self, "radius", _nonnegative_float(radius, name="radius"))

    @property
    def dimension(self) -> int:
        return 2

    def contains(self, points: npt.ArrayLike) -> BoolArray:
        points_array, _scalar = _points(points, dimension=self.dimension)
        distances = np.linalg.norm(points_array - self.center, axis=1)
        result = distances <= self.radius
        return result


@dataclass(frozen=True, slots=True)
class Sphere(_RegionOps):
    """Closed 3-D sphere."""

    center: FloatArray
    radius: float

    def __init__(self, center: npt.ArrayLike, radius: float) -> None:
        object.__setattr__(self, "center", _point(center, dimension=3, name="center"))
        object.__setattr__(self, "radius", _nonnegative_float(radius, name="radius"))

    @property
    def dimension(self) -> int:
        return 3

    def contains(self, points: npt.ArrayLike) -> BoolArray:
        points_array, _scalar = _points(points, dimension=self.dimension)
        distances = np.linalg.norm(points_array - self.center, axis=1)
        result = distances <= self.radius
        return result


@dataclass(frozen=True, slots=True)
class Rectangle(_RegionOps):
    """Closed axis-aligned 2-D rectangle."""

    origin: FloatArray
    width: float
    height: float

    def __init__(self, origin: npt.ArrayLike, width: float, height: float) -> None:
        object.__setattr__(self, "origin", _point(origin, dimension=2, name="origin"))
        object.__setattr__(self, "width", _nonnegative_float(width, name="width"))
        object.__setattr__(self, "height", _nonnegative_float(height, name="height"))

    @property
    def dimension(self) -> int:
        return 2

    @property
    def lower(self) -> FloatArray:
        return self.origin

    @property
    def upper(self) -> FloatArray:
        return self.origin + np.asarray([self.width, self.height], dtype=np.float64)

    def contains(self, points: npt.ArrayLike) -> BoolArray:
        points_array, _scalar = _points(points, dimension=self.dimension)
        result = np.all((points_array >= self.lower) & (points_array <= self.upper), axis=1)
        return result


@dataclass(frozen=True, slots=True)
class Quad(_RegionOps):
    """Closed 2-D quadrilateral region."""

    vertices: FloatArray

    def __init__(
        self, p1: npt.ArrayLike, p2: npt.ArrayLike, p3: npt.ArrayLike, p4: npt.ArrayLike
    ) -> None:
        vertices = np.vstack(
            [
                _point(p1, dimension=2, name="p1"),
                _point(p2, dimension=2, name="p2"),
                _point(p3, dimension=2, name="p3"),
                _point(p4, dimension=2, name="p4"),
            ]
        )
        object.__setattr__(self, "vertices", vertices)

    @property
    def dimension(self) -> int:
        return 2

    def contains(self, points: npt.ArrayLike) -> BoolArray:
        points_array, _scalar = _points(points, dimension=self.dimension)
        result = _points_in_polygon(points_array, self.vertices)
        return result


@dataclass(frozen=True, slots=True)
class Cylinder(_RegionOps):
    """Closed finite cylinder.

    In 2-D this behaves as a capsule around a line segment. In 3-D this is a
    finite circular cylinder with spherical-cap style endpoint inclusion because
    containment is based on distance to the segment.
    """

    p1: FloatArray
    p2: FloatArray
    radius: float

    def __init__(self, p1: npt.ArrayLike, p2: npt.ArrayLike, radius: float) -> None:
        point1 = _point(p1, name="p1")
        point2 = _point(p2, name="p2")
        if point1.shape != point2.shape:
            raise ValueError("p1 and p2 must have the same dimension")
        if point1.size not in {2, 3}:
            raise ValueError("cylinder points must be two- or three-dimensional")

        object.__setattr__(self, "p1", point1)
        object.__setattr__(self, "p2", point2)
        object.__setattr__(self, "radius", _nonnegative_float(radius, name="radius"))

        if np.allclose(point1, point2):
            raise ValueError("cylinder endpoints must be distinct")

    @property
    def dimension(self) -> int:
        return int(self.p1.size)

    def contains(self, points: npt.ArrayLike) -> BoolArray:
        points_array, _scalar = _points(points, dimension=self.dimension)
        result = _points_in_flat_capped_cylinder(points_array, self.p1, self.p2, self.radius)
        return result


@dataclass(frozen=True, slots=True)
class Halfspace(_RegionOps):
    """Closed half-space ``{ x : (x - point) . normal >= 0 }``.

    Useful for an axial cut (e.g. "downstream of the plate back face") that a
    finite Circle/Sphere/Cylinder cannot express.  Compose with a Cylinder via
    ``Cylinder(...) & Halfspace(...)`` to bound both the radius and the axial
    extent in one region.
    """

    point: FloatArray
    normal: FloatArray

    def __init__(self, point: npt.ArrayLike, normal: npt.ArrayLike) -> None:
        pt = _point(point, name="point")
        nrm = _point(normal, name="normal")
        if pt.shape != nrm.shape:
            raise ValueError("point and normal must have the same dimension")
        if pt.size not in {2, 3}:
            raise ValueError("half-space points must be two- or three-dimensional")
        norm = float(np.linalg.norm(nrm))
        if norm == 0.0:
            raise ValueError("normal must be nonzero")
        object.__setattr__(self, "point", pt)
        object.__setattr__(self, "normal", nrm / norm)

    @property
    def dimension(self) -> int:
        return int(self.point.size)

    def contains(self, points: npt.ArrayLike) -> BoolArray:
        points_array, _scalar = _points(points, dimension=self.dimension)
        signed = (points_array - self.point) @ self.normal
        return np.asarray(signed >= 0.0, dtype=np.bool_)


@dataclass(frozen=True, slots=True)
class Slab(_RegionOps):
    """Axis-aligned closed slab bounding a single coordinate axis.

    ``axis`` is ``0/1/2`` or ``'x'/'y'/'z'``.  ``lo``/``hi`` are optional bounds
    (``None`` => unbounded on that side), matching the semi-infinite semantics of
    :class:`BoundedTimeDomain`.  ``dimension`` (default 3) is the point-space
    dimension the slab is tested against; it only affects input validation, not
    which axis is bounded.
    """

    axis: int
    lo: float | None
    hi: float | None
    dim: int

    def __init__(
        self,
        axis: int | str,
        *,
        lo: float | None = None,
        hi: float | None = None,
        dimension: int = 3,
    ) -> None:
        ax = {"x": 0, "y": 1, "z": 2}.get(axis, axis) if isinstance(axis, str) else axis
        if ax not in (0, 1, 2):
            raise ValueError("axis must be one of 0/1/2 or 'x'/'y'/'z'")
        if dimension not in (2, 3):
            raise ValueError("dimension must be 2 or 3")
        if ax >= dimension:
            raise ValueError(f"axis {ax} out of range for dimension {dimension}")
        if lo is None and hi is None:
            raise ValueError("slab requires at least one of lo, hi")
        if lo is not None and hi is not None and float(hi) < float(lo):
            raise ValueError("slab hi must be >= lo")
        object.__setattr__(self, "axis", int(ax))
        object.__setattr__(self, "lo", None if lo is None else float(lo))
        object.__setattr__(self, "hi", None if hi is None else float(hi))
        object.__setattr__(self, "dim", int(dimension))

    @property
    def dimension(self) -> int:
        return self.dim

    def contains(self, points: npt.ArrayLike) -> BoolArray:
        points_array, _scalar = _points(points, dimension=self.dimension)
        column = points_array[:, self.axis]
        result = np.ones(column.shape, dtype=np.bool_)
        if self.lo is not None:
            result &= column >= self.lo
        if self.hi is not None:
            result &= column <= self.hi
        return result


@dataclass(frozen=True, slots=True)
class Intersection(_RegionOps):
    """Region that contains a point iff ALL child regions contain it."""

    regions: tuple["Region", ...]

    def __init__(self, regions: "tuple[Region, ...] | list[Region]") -> None:
        regs = tuple(regions)
        if len(regs) == 0:
            raise ValueError("Intersection requires at least one region")
        object.__setattr__(self, "regions", regs)

    @property
    def dimension(self) -> int:
        return int(self.regions[0].dimension)

    def contains(self, points: npt.ArrayLike) -> BoolArray:
        result: BoolArray | None = None
        for region in self.regions:
            mask = np.asarray(region.contains(points), dtype=np.bool_)
            result = mask if result is None else (result & mask)
        assert result is not None
        return result


@dataclass(frozen=True, slots=True)
class Union(_RegionOps):
    """Region that contains a point iff ANY child region contains it."""

    regions: tuple["Region", ...]

    def __init__(self, regions: "tuple[Region, ...] | list[Region]") -> None:
        regs = tuple(regions)
        if len(regs) == 0:
            raise ValueError("Union requires at least one region")
        object.__setattr__(self, "regions", regs)

    @property
    def dimension(self) -> int:
        return int(self.regions[0].dimension)

    def contains(self, points: npt.ArrayLike) -> BoolArray:
        result: BoolArray | None = None
        for region in self.regions:
            mask = np.asarray(region.contains(points), dtype=np.bool_)
            result = mask if result is None else (result | mask)
        assert result is not None
        return result


@dataclass(frozen=True, slots=True)
class Complement(_RegionOps):
    """Region that contains a point iff the wrapped region does NOT."""

    region: "Region"

    @property
    def dimension(self) -> int:
        return int(self.region.dimension)

    def contains(self, points: npt.ArrayLike) -> BoolArray:
        return ~np.asarray(self.region.contains(points), dtype=np.bool_)


@dataclass(frozen=True, slots=True)
class UnboundedTimeDomain:
    """Time domain containing every time."""

    def contains(self, times: npt.ArrayLike) -> BoolArray:
        array, _scalar = _times(times)
        result = np.ones(array.shape, dtype=np.bool_)
        return result


@dataclass(frozen=True, slots=True)
class BoundedTimeDomain:
    """Closed bounded or semi-bounded time interval."""

    lower: float | None = None
    upper: float | None = None

    def contains(self, times: npt.ArrayLike) -> BoolArray:
        array, _scalar = _times(times)
        result = np.ones(array.shape, dtype=np.bool_)

        if self.lower is not None:
            result &= array >= self.lower
        if self.upper is not None:
            result &= array <= self.upper

        return result


def circle(center: npt.ArrayLike, radius: float) -> Circle:
    """Create a closed 2-D circle."""

    return Circle(center, radius)


@dataclass(frozen=True, slots=True)
class Ring(_RegionOps):
    """Closed 2-D annular (ring) region.

    A point is inside the ring when it is within *outer_radius* of *center*
    and strictly outside *inner_radius*.  Both boundaries are inclusive, so a
    point exactly on either circle is considered inside.

    Parameters
    ----------
    center:
        2-D centre point.
    inner_radius:
        Radius of the inner (hollow) circle.  Must be non-negative and
        strictly less than *outer_radius*.
    outer_radius:
        Radius of the outer circle.  Must be positive.

    Raises
    ------
    ValueError
        If *inner_radius* >= *outer_radius* or either value is negative.
    """

    center: FloatArray
    inner_radius: float
    outer_radius: float

    def __init__(self, center: npt.ArrayLike, *, inner_radius: float, outer_radius: float) -> None:
        object.__setattr__(self, "center", _point(center, dimension=2, name="center"))
        inner = _nonnegative_float(inner_radius, name="inner_radius")
        outer = _nonnegative_float(outer_radius, name="outer_radius")
        if inner >= outer:
            raise ValueError(
                f"inner_radius ({inner_radius}) must be less than outer_radius ({outer_radius})"
            )
        object.__setattr__(self, "inner_radius", inner)
        object.__setattr__(self, "outer_radius", outer)

    @property
    def dimension(self) -> int:
        return 2

    def contains(self, points: npt.ArrayLike) -> BoolArray:
        points_array, _scalar = _points(points, dimension=self.dimension)
        distances = np.linalg.norm(points_array - self.center, axis=1)
        return np.asarray(
            (distances >= self.inner_radius) & (distances <= self.outer_radius), dtype=np.bool_
        )


def ring(center: npt.ArrayLike, *, inner_radius: float, outer_radius: float) -> Ring:
    """Create a closed 2-D annular region."""

    return Ring(center, inner_radius=inner_radius, outer_radius=outer_radius)


def sphere(center: npt.ArrayLike, radius: float) -> Sphere:
    """Create a closed 3-D sphere."""

    return Sphere(center, radius)


def rectangle(origin: npt.ArrayLike, width: float, height: float) -> Rectangle:
    """Create a closed axis-aligned 2-D rectangle."""

    return Rectangle(origin, width, height)


def quad(p1: npt.ArrayLike, p2: npt.ArrayLike, p3: npt.ArrayLike, p4: npt.ArrayLike) -> Quad:
    """Create a closed 2-D quadrilateral."""

    return Quad(p1, p2, p3, p4)


def cylinder(p1: npt.ArrayLike, p2: npt.ArrayLike, radius: float) -> Cylinder:
    """Create a closed finite cylinder/capsule."""

    return Cylinder(p1, p2, radius)


def halfspace(point: npt.ArrayLike, normal: npt.ArrayLike) -> Halfspace:
    """Create a closed half-space ``{x : (x - point) . normal >= 0}``."""

    return Halfspace(point, normal)


def slab(
    axis: int | str, *, lo: float | None = None, hi: float | None = None, dimension: int = 3
) -> Slab:
    """Create an axis-aligned closed slab bounding one coordinate axis."""

    return Slab(axis, lo=lo, hi=hi, dimension=dimension)


def intersection(*regions: "Region") -> Intersection:
    """Create a region that is the intersection (AND) of the given regions."""

    return Intersection(regions)


def union(*regions: "Region") -> Union:
    """Create a region that is the union (OR) of the given regions."""

    return Union(regions)


def complement(region: "Region") -> Complement:
    """Create a region that is the complement (NOT) of the given region."""

    return Complement(region)


def unbounded_time_domain() -> UnboundedTimeDomain:
    """Create a time domain containing all times."""

    return UnboundedTimeDomain()


def bounded_time_domain(
    lower: float | None = None, upper: float | None = None
) -> BoundedTimeDomain:
    """Create a closed bounded or semi-bounded time domain."""

    return BoundedTimeDomain(lower=lower, upper=upper)


# Backward-compatible spelling used by the old code.
bound_time_domain = bounded_time_domain


def _point(value: npt.ArrayLike, *, dimension: int | None = None, name: str) -> FloatArray:
    point = np.asarray(value, dtype=np.float64)

    if point.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional point")
    if dimension is not None and point.size != dimension:
        raise ValueError(f"{name} must be {dimension}-dimensional")
    if point.size == 0:
        raise ValueError(f"{name} cannot be empty")

    return point


def _points(points: npt.ArrayLike, *, dimension: int) -> tuple[FloatArray, bool]:
    array = np.asarray(points, dtype=np.float64)

    if array.ndim == 1:
        if array.size != dimension:
            raise ValueError(f"point must be {dimension}-dimensional")
        return array.reshape(1, dimension), True

    if array.ndim != 2:
        raise ValueError("points must be a point or a two-dimensional point array")
    if array.shape[1] != dimension:
        raise ValueError(f"points must be {dimension}-dimensional")

    return array, False


def _times(times: npt.ArrayLike) -> tuple[FloatArray, bool]:
    array = np.asarray(times, dtype=np.float64)

    if array.ndim == 0:
        return array.reshape(1), True
    if array.ndim != 1:
        raise ValueError("times must be scalar or one-dimensional")

    return array, False


def _nonnegative_float(value: float, *, name: str) -> float:
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _points_in_polygon(points: FloatArray, vertices: FloatArray) -> BoolArray:
    result = np.zeros(points.shape[0], dtype=np.bool_)

    for index, point in enumerate(points):
        result[index] = _point_in_polygon(point, vertices)

    return result


def _point_in_polygon(point: FloatArray, vertices: FloatArray) -> bool:
    if _point_on_polygon_boundary(point, vertices):
        return True

    x = point[0]
    y = point[1]
    inside = False
    count = vertices.shape[0]

    for i in range(count):
        j = (i - 1) % count
        xi, yi = vertices[i]
        xj, yj = vertices[j]

        crosses = (yi > y) != (yj > y)
        if crosses:
            x_intersection = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersection:
                inside = not inside

    return inside


def _point_on_polygon_boundary(point: FloatArray, vertices: FloatArray) -> bool:
    count = vertices.shape[0]

    for i in range(count):
        a = vertices[i]
        b = vertices[(i + 1) % count]
        if _point_on_segment(point, a, b):
            return True

    return False


def _point_on_segment(point: FloatArray, a: FloatArray, b: FloatArray) -> bool:
    segment = b - a
    relative = point - a
    cross = segment[0] * relative[1] - segment[1] * relative[0]

    if not np.isclose(cross, 0.0):
        return False

    dot = float(np.dot(relative, segment))
    if dot < 0.0:
        return False

    length_squared = float(np.dot(segment, segment))
    return dot <= length_squared


def _points_in_flat_capped_cylinder(
    points: FloatArray, p1: FloatArray, p2: FloatArray, radius: float
) -> BoolArray:
    axis = p2 - p1
    length_squared = float(np.dot(axis, axis))
    relative = points - p1

    # Parametric projection onto the cylinder axis.  Flat caps mean points must
    # project between p1 and p2, not merely be close to the segment endpoint.
    t = (relative @ axis) / length_squared
    between_caps = (t >= 0.0) & (t <= 1.0)

    projection = p1 + t[:, None] * axis
    radial_distance = np.linalg.norm(points - projection, axis=1)

    return np.asarray(between_caps & (radial_distance <= radius), dtype=np.bool_)


__all__ = [
    "BoolArray",
    "BoundedTimeDomain",
    "Circle",
    "Cylinder",
    "FloatArray",
    "Quad",
    "Rectangle",
    "Region",
    "Ring",
    "Sphere",
    "TimeDomain",
    "UnboundedTimeDomain",
    "bound_time_domain",
    "bounded_time_domain",
    "circle",
    "cylinder",
    "quad",
    "rectangle",
    "ring",
    "sphere",
    "unbounded_time_domain",
]
