# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from exodusii.mesh.geometry import bounding_box
from exodusii.mesh.geometry import characteristic_element_length
from exodusii.mesh.geometry import connected_average
from exodusii.mesh.geometry import element_volumes
from exodusii.mesh.geometry import entity_centers
from exodusii.mesh.geometry import nodal_volumes


def test_connected_average_scalar_values() -> None:
    conn = np.asarray([[0, 1], [1, 2]], dtype=int)
    values = np.asarray([1.0, 3.0, 5.0])

    result = connected_average(conn, values)

    assert result.shape == (2,)
    assert np.allclose(result, [2.0, 4.0])


def test_connected_average_vector_values() -> None:
    conn = np.asarray([[0, 1], [1, 2]], dtype=int)
    values = np.asarray([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]], dtype=float)

    result = connected_average(conn, values)

    assert result.shape == (2, 2)
    assert np.allclose(result, [[1.0, 2.0], [3.0, 6.0]])


def test_connected_average_rejects_bad_connectivity_rank() -> None:
    with pytest.raises(ValueError, match="connectivity must be a two-dimensional array"):
        connected_average([0, 1, 2], [1.0, 2.0, 3.0])


def test_connected_average_rejects_empty_entities() -> None:
    with pytest.raises(ValueError, match="at least one node"):
        connected_average(np.empty((2, 0), dtype=int), [1.0, 2.0, 3.0])


def test_connected_average_rejects_bad_values_rank() -> None:
    conn = np.asarray([[0, 1]], dtype=int)
    values = np.zeros((2, 2, 2), dtype=float)

    with pytest.raises(ValueError, match="values must be"):
        connected_average(conn, values)


def test_connected_average_rejects_negative_connectivity() -> None:
    conn = np.asarray([[0, -1]], dtype=int)

    with pytest.raises(IndexError, match="negative indices"):
        connected_average(conn, [1.0, 2.0])


def test_connected_average_rejects_out_of_range_connectivity() -> None:
    conn = np.asarray([[0, 2]], dtype=int)

    with pytest.raises(IndexError, match="outside values"):
        connected_average(conn, [1.0, 2.0])


def test_entity_centers() -> None:
    conn = np.asarray([[0, 1, 2, 3]], dtype=int)
    coords = _unit_square_quad()

    result = entity_centers(conn, coords)

    assert result.shape == (1, 2)
    assert np.allclose(result[0], [0.5, 0.5])


def test_element_volumes_quad() -> None:
    conn = np.asarray([[0, 1, 2, 3]], dtype=int)
    coords = _unit_square_quad()

    result = element_volumes("quad4", conn, coords)

    assert result.shape == (1,)
    assert result[0] == pytest.approx(1.0)


def test_element_volumes_two_quads() -> None:
    coords = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=float
    )
    conn = np.asarray([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=int)

    result = element_volumes("quad4", conn, coords)

    assert np.allclose(result, [1.0, 1.0])


def test_element_volumes_hex() -> None:
    conn = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
    coords = _unit_cube_hex()

    result = element_volumes("hex8", conn, coords)

    assert result.shape == (1,)
    assert result[0] == pytest.approx(1.0)


def test_nodal_volumes_quad() -> None:
    conn = np.asarray([[0, 1, 2, 3]], dtype=int)
    coords = _unit_square_quad()

    result = nodal_volumes("quad4", conn, coords)

    assert result.shape == (4,)
    assert np.allclose(result, [0.25, 0.25, 0.25, 0.25])


def test_nodal_volumes_two_quads_shared_nodes() -> None:
    coords = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=float
    )
    conn = np.asarray([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=int)

    result = nodal_volumes("quad4", conn, coords)

    assert np.allclose(result, [0.25, 0.5, 0.25, 0.25, 0.5, 0.25])


def test_nodal_volumes_accepts_explicit_num_nodes() -> None:
    conn = np.asarray([[0, 1, 2, 3]], dtype=int)
    coords = _unit_square_quad()

    result = nodal_volumes("quad4", conn, coords, num_nodes=5)

    assert result.shape == (5,)
    assert result[-1] == 0.0


def test_nodal_volumes_rejects_negative_num_nodes() -> None:
    conn = np.asarray([[0, 1, 2, 3]], dtype=int)
    coords = _unit_square_quad()

    with pytest.raises(ValueError, match="num_nodes must be nonnegative"):
        nodal_volumes("quad4", conn, coords, num_nodes=-1)


def test_nodal_volumes_rejects_out_of_range_num_nodes() -> None:
    conn = np.asarray([[0, 1, 2, 3]], dtype=int)
    coords = _unit_square_quad()

    with pytest.raises(IndexError, match="outside num_nodes"):
        nodal_volumes("quad4", conn, coords, num_nodes=3)


def test_characteristic_element_length_2d() -> None:
    conn = np.asarray([[0, 1, 2, 3]], dtype=int)
    coords = _unit_square_quad()

    result = characteristic_element_length("quad4", conn, coords)

    assert result == pytest.approx(1.0)


def test_characteristic_element_length_3d() -> None:
    conn = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
    coords = _unit_cube_hex()

    result = characteristic_element_length("hex8", conn, coords)

    assert result == pytest.approx(1.0)


def test_characteristic_element_length_rejects_empty_mesh() -> None:
    conn = np.empty((0, 4), dtype=int)
    coords = _unit_square_quad()

    with pytest.raises(ValueError, match="empty mesh"):
        characteristic_element_length("quad4", conn, coords)


def test_characteristic_element_length_rejects_bad_dimension() -> None:
    conn = np.asarray([[0, 1, 2, 3]], dtype=int)
    coords = _unit_square_quad()

    with pytest.raises(ValueError, match="dimension must be positive"):
        characteristic_element_length("quad4", conn, coords, dimension=0)


def test_bounding_box() -> None:
    coords = np.asarray([[1.0, 2.0, 3.0], [-1.0, 4.0, 0.0], [2.0, -2.0, 5.0]], dtype=float)

    lower, upper = bounding_box(coords)

    assert np.allclose(lower, [-1.0, -2.0, 0.0])
    assert np.allclose(upper, [2.0, 4.0, 5.0])


def test_bounding_box_rejects_empty_coordinates() -> None:
    with pytest.raises(ValueError, match="empty coordinates"):
        bounding_box(np.empty((0, 2), dtype=float))


def test_bounding_box_rejects_bad_coordinate_rank() -> None:
    with pytest.raises(ValueError, match="coordinates must be a two-dimensional array"):
        bounding_box([0.0, 1.0])


def test_bounding_box_rejects_empty_spatial_dimension() -> None:
    with pytest.raises(ValueError, match="at least one spatial dimension"):
        bounding_box(np.empty((2, 0), dtype=float))


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
