# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from exodusii.mesh.regions import bound_time_domain
from exodusii.mesh.regions import bounded_time_domain
from exodusii.mesh.regions import circle
from exodusii.mesh.regions import cylinder
from exodusii.mesh.regions import quad
from exodusii.mesh.regions import rectangle
from exodusii.mesh.regions import sphere
from exodusii.mesh.regions import unbounded_time_domain


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
