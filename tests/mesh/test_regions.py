# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from exodusii.mesh.regions import Complement
from exodusii.mesh.regions import Intersection
from exodusii.mesh.regions import Union
from exodusii.mesh.regions import bound_time_domain
from exodusii.mesh.regions import bounded_time_domain
from exodusii.mesh.regions import circle
from exodusii.mesh.regions import complement
from exodusii.mesh.regions import cylinder
from exodusii.mesh.regions import halfspace
from exodusii.mesh.regions import intersection
from exodusii.mesh.regions import quad
from exodusii.mesh.regions import rectangle
from exodusii.mesh.regions import slab
from exodusii.mesh.regions import sphere
from exodusii.mesh.regions import unbounded_time_domain
from exodusii.mesh.regions import union


def test_region_cylinder_2d_axis_aligned() -> None:
    region = cylinder([0.0, 0.0], [1.0, 0.0], 0.5)

    points = [[0.0, -0.5], [1.0, -0.5], [1.0, 0.5], [0.0, 0.5], [1.0, -0.5025], [1.0, 0.5025]]

    assert region.dimension == 2
    assert region.contains(points[0])
    assert region.contains(points[1])
    assert region.contains(points[2])
    assert region.contains(points[3])
    assert not region.contains(points[4])
    assert not region.contains(points[5])

    contains = region.contains(points)
    assert contains.tolist() == [True, True, True, True, False, False]


def test_region_cylinder_2d_diagonal() -> None:
    region = cylinder([0.0, 0.0], [1.0, 1.0], 0.5)
    x = 0.5 * np.sqrt(2.0) / 2.0

    points = [
        [0.99 * x, -0.99 * x],
        [1.0 + 0.99 * x, 1.0 - 0.99 * x],
        [1.0 - 0.99 * x, 1.0 + 0.99 * x],
        [-0.99 * x, 0.99 * x],
        [1.0 + 1.005 * x, 1.0 - 1.005 * x],
        [1.0 - 1.005 * x, 1.0 + 1.005 * x],
    ]

    contains = region.contains(points)

    assert contains.tolist() == [True, True, True, True, False, False]


def test_region_cylinder_3d() -> None:
    region = cylinder([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.5)

    points = [[0.0, 0.5, 0.0], [0.0, 1.5, 0.0]]

    assert region.dimension == 3
    assert region.contains(points[0])
    assert not region.contains(points[1])
    assert region.contains(points).tolist() == [True, False]


def test_region_cylinder_rejects_mismatched_dimensions() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        cylinder([0.0, 0.0], [1.0, 0.0, 0.0], 0.5)


def test_region_cylinder_rejects_invalid_dimension() -> None:
    with pytest.raises(ValueError, match="two- or three-dimensional"):
        cylinder([0.0], [1.0], 0.5)


def test_region_cylinder_rejects_coincident_endpoints() -> None:
    with pytest.raises(ValueError, match="distinct"):
        cylinder([0.0, 0.0], [0.0, 0.0], 0.5)


def test_region_rectangle() -> None:
    region = rectangle([0.0, -2.5], 5.0, 5.0)
    points = [[0.0, 2.5], [-3.0, 2.5]]

    assert region.dimension == 2
    assert region.contains(points[0])
    assert not region.contains(points[1])
    assert region.contains(points).tolist() == [True, False]


def test_region_quad() -> None:
    region = quad([1.0, 1.0], [5.0, 2.0], [6.0, 5.0], [0.0, 3.0])
    points = [[2.0, 2.5], [4.0, 0.0]]

    assert region.dimension == 2
    assert region.contains(points[0])
    assert not region.contains(points[1])
    assert region.contains(points).tolist() == [True, False]


def test_region_quad_includes_boundary() -> None:
    region = quad([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0])

    assert region.contains([0.0, 0.0])
    assert region.contains([0.5, 0.0])
    assert region.contains([1.0, 1.0])


def test_region_circle() -> None:
    region = circle([0.0, 0.0], 1.0)
    points = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 1.1]]

    assert region.dimension == 2
    for point in points[:-1]:
        assert region.contains(point)
    assert not region.contains(points[-1])

    contains = region.contains(points)
    assert contains.tolist() == [True, True, True, True, True, False]


def test_region_sphere() -> None:
    region = sphere([0.0, 0.0, 0.0], 1.0)
    points = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.1, 0.8],
    ]

    assert region.dimension == 3
    for point in points[:-1]:
        assert region.contains(point)
    assert not region.contains(points[-1])

    contains = region.contains(points)
    assert contains.tolist() == [True, True, True, True, True, False]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: circle([0.0, 0.0], -1.0),
        lambda: sphere([0.0, 0.0, 0.0], -1.0),
        lambda: rectangle([0.0, 0.0], -1.0, 1.0),
        lambda: cylinder([0.0, 0.0], [1.0, 0.0], -1.0),
    ],
)
def test_regions_reject_negative_sizes(factory: object) -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        factory()  # ty: ignore[call-non-callable]


def test_region_rejects_bad_point_dimension() -> None:
    region = circle([0.0, 0.0], 1.0)

    with pytest.raises(ValueError, match="point must be 2-dimensional"):
        region.contains([0.0, 0.0, 0.0])


def test_region_rejects_bad_points_rank() -> None:
    region = circle([0.0, 0.0], 1.0)

    with pytest.raises(ValueError, match="points must be"):
        region.contains(np.zeros((2, 2, 2)))


def test_unbounded_time_domain_contains_everything() -> None:
    domain = unbounded_time_domain()

    assert domain.contains(0.0)
    assert domain.contains([-1.0, 0.0, 1.0]).tolist() == [True, True, True]


def test_bounded_time_domain_lower_and_upper() -> None:
    domain = bounded_time_domain(0.0, 1.0)

    assert domain.contains(0.0)
    assert domain.contains(1.0)
    assert not domain.contains(-0.1)
    assert not domain.contains(1.1)
    assert domain.contains([-0.1, 0.0, 0.5, 1.0, 1.1]).tolist() == [False, True, True, True, False]


def test_bounded_time_domain_semi_bounded_lower() -> None:
    domain = bounded_time_domain(0.0, None)

    assert domain.contains([-1.0, 0.0, 1.0]).tolist() == [False, True, True]


def test_bounded_time_domain_semi_bounded_upper() -> None:
    domain = bounded_time_domain(None, 1.0)

    assert domain.contains([-1.0, 0.0, 1.0, 2.0]).tolist() == [True, True, True, False]


def test_bound_time_domain_alias() -> None:
    domain = bound_time_domain(0.0, 1.0)

    assert domain.contains([0.0, 2.0]).tolist() == [True, False]


def test_time_domain_rejects_bad_rank() -> None:
    domain = unbounded_time_domain()

    with pytest.raises(ValueError, match="times must be scalar or one-dimensional"):
        domain.contains(np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# Halfspace / Slab / composition (EXODUSII-IMPROVEMENTS #9)
# ---------------------------------------------------------------------------


def test_halfspace_basic() -> None:
    hs = halfspace([0.012, 0.0, 0.0], [1.0, 0.0, 0.0])  # x >= 12mm
    pts = [[0.02, 0, 0], [0.012, 0, 0], [0.005, 0, 0]]
    assert hs.contains(pts).tolist() == [True, True, False]
    assert hs.dimension == 3


def test_halfspace_normalizes_normal() -> None:
    hs = halfspace([0, 0, 0], [5.0, 0.0, 0.0])
    assert np.allclose(np.linalg.norm(hs.normal), 1.0)


def test_halfspace_rejects_zero_normal() -> None:
    with pytest.raises(ValueError, match="normal must be nonzero"):
        halfspace([0, 0, 0], [0, 0, 0])


def test_slab_bounds_single_axis() -> None:
    sl = slab("x", lo=0.012)
    pts = [[0.02, 9, 9], [0.005, 0, 0]]
    assert sl.contains(pts).tolist() == [True, False]


def test_slab_semi_and_full() -> None:
    assert slab(1, hi=1.0).contains([[0, 0.5, 0], [0, 2.0, 0]]).tolist() == [True, False]
    assert slab("z", lo=-1.0, hi=1.0).contains([[0, 0, 0], [0, 0, 5]]).tolist() == [True, False]


def test_slab_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="axis must be one of"):
        slab("w", lo=0.0)
    with pytest.raises(ValueError, match="requires at least one"):
        slab("x")
    with pytest.raises(ValueError, match="hi must be >= lo"):
        slab("x", lo=1.0, hi=0.0)


def test_region_intersection_operator_and_factory() -> None:
    cyl = cylinder([-0.05, 0, 0], [0.30, 0, 0], 0.011)
    hs = halfspace([0.012, 0, 0], [1, 0, 0])
    pts = [[0.02, 0, 0], [0.005, 0, 0], [0.02, 0.02, 0]]
    expected = [True, False, False]  # inside radius AND downstream
    assert (cyl & hs).contains(pts).tolist() == expected
    assert isinstance(cyl & hs, Intersection)
    assert intersection(cyl, hs).contains(pts).tolist() == expected


def test_region_union_operator() -> None:
    cyl = cylinder([-0.05, 0, 0], [0.30, 0, 0], 0.011)
    hs = halfspace([0.012, 0, 0], [1, 0, 0])
    pts = [[0.02, 0, 0], [0.005, 0, 0], [0.02, 0.02, 0], [-0.01, 0, 0]]
    assert (cyl | hs).contains(pts).tolist() == [True, True, True, True]
    assert isinstance(cyl | hs, Union)
    assert union(cyl, hs).contains(pts).tolist() == [True, True, True, True]


def test_region_complement_operator() -> None:
    hs = halfspace([0.012, 0, 0], [1, 0, 0])
    pts = [[0.02, 0, 0], [0.005, 0, 0]]
    assert (~hs).contains(pts).tolist() == [False, True]
    assert isinstance(~hs, Complement)
    assert complement(hs).contains(pts).tolist() == [False, True]


def test_region_composition_dimension() -> None:
    cyl = cylinder([-0.05, 0, 0], [0.30, 0, 0], 0.011)
    hs = halfspace([0.012, 0, 0], [1, 0, 0])
    assert (cyl & hs).dimension == 3


def test_hollow_shell_via_complement() -> None:
    outer = cylinder([-0.05, 0, 0], [0.30, 0, 0], 0.013)
    inner = cylinder([-0.05, 0, 0], [0.30, 0, 0], 0.007)
    shell = outer & ~inner
    pts = [[0.0, 0.010, 0], [0.0, 0.004, 0], [0.0, 0.020, 0]]  # rim, core, outside
    assert shell.contains(pts).tolist() == [True, False, False]
