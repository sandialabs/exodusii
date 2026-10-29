# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Element geometry for 3-D surface elements and additional element types."""

import numpy as np
import pytest

from exodusii.mesh.elements import Line2
from exodusii.mesh.elements import Pyramid5
from exodusii.mesh.elements import element_factory
from exodusii.mesh.geometry import element_volumes

UNIT_SQUARE_XZ = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=float)


@pytest.mark.parametrize("etype", ["SHELL4", "SHELL", "QUAD4"])
def test_quad_area_in_3d(etype: str) -> None:
    assert element_factory(etype, UNIT_SQUARE_XZ).volume == pytest.approx(1.0)
    tilted = UNIT_SQUARE_XZ @ np.array([[1, 0, 0], [0, 0.6, 0.8], [0, -0.8, 0.6]])
    assert element_factory(etype, tilted).volume == pytest.approx(1.0)


@pytest.mark.parametrize("etype", ["TRI3", "TRISHELL", "TRISHELL3", "SHELL3"])
def test_triangle_area_in_3d(etype: str) -> None:
    tri = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    assert element_factory(etype, tri).volume == pytest.approx(0.5)


def test_planar_xy_results_unchanged() -> None:
    quad = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=float)
    assert element_factory("quad4", quad).volume == 2.0
    quad_z = np.column_stack([quad, np.full(4, 5.0)])
    assert element_factory("quad4", quad_z).volume == 2.0


def test_higher_order_types_use_corner_nodes() -> None:
    tet = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    mids = np.array(
        [(tet[a] + tet[b]) / 2 for a, b in ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))]
    )
    assert element_factory("TETRA10", np.vstack([tet, mids])).volume == pytest.approx(1 / 6)

    hex8 = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=float,
    )
    hex20 = np.vstack([hex8, np.zeros((12, 3))])
    assert element_factory("HEX20", hex20).volume == pytest.approx(1.0)

    tri6 = np.vstack([[[0, 0], [1, 0], [0, 1]], np.zeros((3, 2))])
    assert element_factory("TRIANGLE", tri6).volume == pytest.approx(0.5)


def test_pyramid_volume() -> None:
    pyramid = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [0.5, 0.7, 3]], dtype=float)
    element = element_factory("PYRAMID5", pyramid)
    assert isinstance(element, Pyramid5)
    assert element.volume == pytest.approx(4.0)  # base 4, height 3


def test_line_length_and_subdivision() -> None:
    line = element_factory("BEAM2", [[0, 0, 0], [3, 4, 0]])
    assert isinstance(line, Line2)
    assert line.volume == pytest.approx(5.0)
    assert line.subvols(5).sum() == pytest.approx(5.0)
    assert np.allclose(line.subdiv(2), [[0.75, 1, 0], [2.25, 3, 0]])


def test_element_volumes_for_shell_block() -> None:
    coords = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [2, 0, 0], [2, 0, 1]], dtype=float
    )
    conn = np.array([[0, 1, 2, 3], [1, 4, 5, 2]])
    assert np.allclose(element_volumes("SHELL4", conn, coords), [1.0, 1.0])


def test_unknown_type_still_raises() -> None:
    with pytest.raises(ValueError, match="unknown element type"):
        element_factory("SPHERE", [[0, 0, 0]])


def _wedge_isoparametric_volume(coord: np.ndarray, order: int = 10) -> float:
    gauss, weights = np.polynomial.legendre.leggauss(order)
    points, weights = (gauss + 1) / 2, weights / 2
    volume = 0.0
    for a, wa in zip(points, weights, strict=True):
        for b, wb in zip(points, weights, strict=True):
            for z, wz in zip(points, weights, strict=True):
                s, t = a, b * (1 - a)
                ds = np.array([-(1 - z), 1 - z, 0, -z, z, 0])
                dt = np.array([-(1 - z), 0, 1 - z, -z, 0, z])
                dz = np.array([-(1 - s - t), -s, -t, 1 - s - t, s, t])
                volume += (
                    wa * wb * wz * (1 - a) * np.linalg.det([ds @ coord, dt @ coord, dz @ coord])
                )
    return float(volume)


def test_wedge_volume_exact_for_warped_faces() -> None:
    rng = np.random.default_rng(0)
    wedge = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=float
    )
    for _ in range(5):
        warped = wedge + rng.uniform(-0.2, 0.2, wedge.shape)
        expected = _wedge_isoparametric_volume(warped)
        assert element_factory("WEDGE6", warped).volume == pytest.approx(expected, rel=1e-12)
    assert element_factory("WEDGE6", wedge).volume == pytest.approx(0.5)


def test_point_on_quad_boundary_is_scale_free() -> None:
    from exodusii.mesh.regions import quad

    small = quad([0, 0], [1e-6, 0], [1e-6, 1e-6], [0, 1e-6])
    assert not small.contains(np.array([[3e-9, 5e-6]]))[0]
    assert small.contains(np.array([[5e-7, 0.0]]))[0]
    big = quad([0, 0], [1e6, 0], [1e6, 1e6], [0, 1e6])
    assert big.contains(np.array([[5e5, 0.0]]))[0]
    assert not big.contains(np.array([[5e5, -1.0]]))[0]
