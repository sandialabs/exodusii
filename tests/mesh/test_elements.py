# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from exodusii.mesh.elements import Hex8
from exodusii.mesh.elements import Quad4
from exodusii.mesh.elements import Tet4
from exodusii.mesh.elements import Tri3
from exodusii.mesh.elements import Wedge6
from exodusii.mesh.elements import element_factory


def _unit_square_quad() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)


def _unit_cube_hex() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )


def _unit_right_triangle() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)


def _unit_tet() -> np.ndarray:
    return np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float
    )


def _unit_wedge() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )


def test_element_factory_accepts_strings() -> None:
    assert isinstance(element_factory("quad", _unit_square_quad()), Quad4)
    assert isinstance(element_factory("QUAD4", _unit_square_quad()), Quad4)
    assert isinstance(element_factory("hex", _unit_cube_hex()), Hex8)
    assert isinstance(element_factory("HEX8", _unit_cube_hex()), Hex8)
    assert isinstance(element_factory("tri3", _unit_right_triangle()), Tri3)
    assert isinstance(element_factory("tet4", _unit_tet()), Tet4)
    assert isinstance(element_factory("wedge6", _unit_wedge()), Wedge6)


def test_element_factory_accepts_bytes() -> None:
    assert isinstance(element_factory(b"quad4", _unit_square_quad()), Quad4)


def test_element_factory_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown element type"):
        element_factory("pyramid", np.zeros((5, 3)))


def test_quad4_center_and_volume() -> None:
    element = Quad4(_unit_square_quad())

    assert element.dimension == 2
    assert np.allclose(element.center, [0.5, 0.5])
    assert element.volume == pytest.approx(1.0)


def test_quad4_volume_diagonal_formula_bitwise() -> None:
    # exodusii < 2 used half the diagonal cross product; keep its exact
    # floating-point evaluation (shoelace differs in the last bits).
    rng = np.random.default_rng(11)
    for _ in range(100):
        c = _unit_square_quad() + 0.2 * rng.standard_normal((4, 2)) + 1.0e3
        x, y = c[:, 0], c[:, 1]
        expected = abs(0.5 * ((x[0] - x[2]) * (y[1] - y[3]) + (x[1] - x[3]) * (y[2] - y[0])))
        assert Quad4(c).volume == expected


def test_quad4_volume_orientation_independent() -> None:
    coord = _unit_square_quad() * 2.0

    assert Quad4(coord).volume == Quad4(coord[::-1]).volume == pytest.approx(4.0)


def test_quad4_volume_is_translation_invariant() -> None:
    coord = _unit_square_quad() + np.asarray([10.0, -3.0])
    element = Quad4(coord)

    assert element.volume == pytest.approx(1.0)


def test_quad4_subdivision() -> None:
    element = Quad4(_unit_square_quad())

    centers = element.subdiv(2)
    subcoord = element.subcoord(2)
    subconn = element.subconn(2)
    subvols = element.subvols(2)

    assert centers.shape == (4, 2)
    assert subcoord.shape == (9, 2)
    assert subconn.shape == (4, 4)
    assert subvols.shape == (4,)
    assert np.allclose(np.sum(subvols), element.volume)
    assert np.allclose(subvols, [0.25, 0.25, 0.25, 0.25])


def test_hex8_center_and_volume() -> None:
    element = Hex8(_unit_cube_hex())

    assert element.dimension == 3
    assert np.allclose(element.center, [0.5, 0.5, 0.5])
    assert element.volume == pytest.approx(1.0)


def test_hex8_subdivision() -> None:
    element = Hex8(_unit_cube_hex())

    centers = element.subdiv(2)
    subcoord = element.subcoord(2)
    subconn = element.subconn(2)
    subvols = element.subvols(2)

    assert centers.shape == (8, 3)
    assert subcoord.shape == (27, 3)
    assert subconn.shape == (8, 8)
    assert subvols.shape == (8,)
    assert np.allclose(np.sum(subvols), element.volume)
    assert np.allclose(subvols, np.full(8, 0.125))


def _twisted_hex() -> np.ndarray:
    # Unit cube with the top face rotated about z: the side faces are
    # non-planar, so a tetrahedral split is not exact.
    coord = _unit_cube_hex()
    theta = 0.6
    c, s = np.cos(theta), np.sin(theta)
    top = coord[4:, :2] - 0.5
    coord[4:, :2] = top @ np.asarray([[c, s], [-s, c]]) + 0.5
    return coord


def _gauss_hex_volume(coord: np.ndarray) -> float:
    # 2x2x2 Gauss quadrature of det(J) integrates the trilinear map exactly.
    ref = np.asarray(
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
        dtype=float,
    )  # fmt: skip
    g = 1.0 / np.sqrt(3.0)
    total = 0.0
    for a in (-g, g):
        for b in (-g, g):
            for c in (-g, g):
                dn = (
                    np.stack(
                        [
                            ref[:, 0] * (1 + ref[:, 1] * b) * (1 + ref[:, 2] * c),
                            (1 + ref[:, 0] * a) * ref[:, 1] * (1 + ref[:, 2] * c),
                            (1 + ref[:, 0] * a) * (1 + ref[:, 1] * b) * ref[:, 2],
                        ],
                        axis=1,
                    )
                    / 8.0
                )
                total += float(np.linalg.det(coord.T @ dn))
    return total


def test_hex8_volume_exact_for_nonplanar_faces() -> None:
    coord = _twisted_hex()

    assert Hex8(coord).volume == pytest.approx(_gauss_hex_volume(coord), rel=1e-14)


def test_hex8_volume_random_distortions_match_quadrature() -> None:
    rng = np.random.default_rng(1234)
    for _ in range(50):
        coord = _unit_cube_hex() + 0.2 * rng.standard_normal((8, 3))
        assert Hex8(coord).volume == pytest.approx(abs(_gauss_hex_volume(coord)), rel=1e-12)


def test_hex8_volume_matches_legacy_closed_form_bitwise() -> None:
    # exodusii < 2 value for _twisted_hex(), computed with the original
    # closed-form expression; downstream verification baselines depend on it.
    from exodusii.mesh.elements import hex8_volumes

    coord = _twisted_hex()
    signed = hex8_volumes(coord[np.newaxis])[0]

    assert Hex8(coord).volume == abs(signed)
    assert Hex8(coord).subvols(3).sum() == pytest.approx(Hex8(coord).volume, rel=1e-14)


def test_hex8_vectorised_volumes_match_scalar() -> None:
    from exodusii.mesh.elements import hex8_volumes

    rng = np.random.default_rng(7)
    coords = _unit_cube_hex()[np.newaxis] + 0.2 * rng.standard_normal((20, 8, 3))
    scalar = np.asarray([Hex8(c).volume for c in coords])

    assert np.array_equal(np.abs(hex8_volumes(coords)), scalar)


def test_tri3_center_and_volume() -> None:
    element = Tri3(_unit_right_triangle())

    assert element.dimension == 2
    assert np.allclose(element.center, [1.0 / 3.0, 1.0 / 3.0])
    assert element.volume == pytest.approx(0.5)


def test_tri3_volume_is_translation_invariant() -> None:
    coord = _unit_right_triangle() + np.asarray([2.0, 3.0])
    element = Tri3(coord)

    assert element.volume == pytest.approx(0.5)


def test_tri3_subdivision() -> None:
    element = Tri3(_unit_right_triangle())

    centers = element.subdiv(2)
    subcoord = element.subcoord(2)
    subconn = element.subconn(2)
    subvols = element.subvols(2)

    assert centers.shape[1] == 2
    assert subcoord.shape[1] == 2
    assert subconn.shape[1] == 3
    assert len(subconn) == 4
    assert subvols.shape == (4,)
    assert np.allclose(np.sum(subvols), element.volume)


def test_tet4_center_and_volume() -> None:
    element = Tet4(_unit_tet())

    assert element.dimension == 3
    assert np.allclose(element.center, [0.25, 0.25, 0.25])
    assert element.volume == pytest.approx(1.0 / 6.0)


def test_tet4_subdivision() -> None:
    element = Tet4(_unit_tet())

    centers = element.subdiv(2)
    subcoord = element.subcoord(2)
    subconn = element.subconn(2)
    subvols = element.subvols(2)

    assert centers.shape[1] == 3
    assert subcoord.shape[1] == 3
    assert subconn.shape[1] == 4
    assert len(subconn) == 4
    assert subvols.shape == (4,)
    assert np.allclose(np.sum(subvols), element.volume)


def test_wedge6_center_and_volume() -> None:
    element = Wedge6(_unit_wedge())

    assert element.dimension == 3
    assert np.allclose(element.center, [1.0 / 3.0, 1.0 / 3.0, 0.5])
    assert element.volume == pytest.approx(0.5)


def test_wedge6_subdivision() -> None:
    element = Wedge6(_unit_wedge())

    centers = element.subdiv(2)
    subcoord = element.subcoord(2)
    subconn = element.subconn(2)
    subvols = element.subvols(2)

    assert centers.shape[1] == 3
    assert subcoord.shape[1] == 3
    assert subconn.shape[1] == 6
    assert len(subconn) == 16
    assert subvols.shape == (16,)
    assert np.allclose(np.sum(subvols), element.volume)


def test_invalid_coordinate_rank() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        Quad4([0.0, 1.0, 2.0, 3.0])


def test_invalid_coordinate_node_count() -> None:
    with pytest.raises(ValueError, match="expected 4 element nodes"):
        Quad4(np.zeros((3, 2)))


def test_invalid_coordinate_dimension() -> None:
    with pytest.raises(ValueError, match="expected at least 3 coordinate dimensions"):
        Hex8(np.zeros((8, 2)))


@pytest.mark.parametrize(
    ("element_type", "coord_factory"), [(Quad4, _unit_square_quad), (Hex8, _unit_cube_hex)]
)
def test_structured_subdivision_rejects_nonpositive_intervals(
    element_type: type[Quad4] | type[Hex8], coord_factory: object
) -> None:
    element = element_type(coord_factory())  # type: ignore[operator]  # ty: ignore[call-non-callable]

    with pytest.raises(ValueError, match="intervals must be positive"):
        element.subdiv(0)


@pytest.mark.parametrize(
    ("element_type", "coord_factory"), [(Quad4, _unit_square_quad), (Hex8, _unit_cube_hex)]
)
def test_structured_subdivision_rejects_noninteger_intervals(
    element_type: type[Quad4] | type[Hex8], coord_factory: object
) -> None:
    element = element_type(coord_factory())  # type: ignore[operator]  # ty: ignore[call-non-callable]

    with pytest.raises(TypeError, match="intervals must be an int"):
        element.subdiv(1.5)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
