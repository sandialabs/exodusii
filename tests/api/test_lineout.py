# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from exodusii.api.lineout import Lineout
from exodusii.api.lineout import lineout


def test_lineout_factory() -> None:
    result = lineout(x="x", y=1.0)

    assert isinstance(result, Lineout)
    assert result.x == "x"
    assert result.y == 1.0


def test_from_cli_2d_x_line() -> None:
    result = Lineout.from_cli("x/1.0")

    assert result.x == "x"
    assert result.y == 1.0
    assert result.z is None
    assert result.tol is None
    assert not result.needs_displacements


def test_from_cli_3d_displaced_y_line_with_tolerance() -> None:
    result = Lineout.from_cli("1.0/Y/3.0/T0.1")

    assert result.x == 1.0
    assert result.y == "Y"
    assert result.z == 3.0
    assert result.tol == pytest.approx(0.1)
    assert result.needs_displacements


def test_from_cli_rejects_too_many_parts() -> None:
    with pytest.raises(ValueError, match="at most 3 spatial"):
        Lineout.from_cli("x/y/z/1.0")


def test_from_cli_rejects_bad_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance parameter"):
        Lineout.from_cli("x/1.0/Tbad")


def test_from_cli_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Lineout.from_cli("")


def test_read_spatial_spec() -> None:
    assert Lineout.read_spatial_spec(None, "x") is None
    assert Lineout.read_spatial_spec("1.5", "x") == 1.5
    assert Lineout.read_spatial_spec("x", "x") == "x"
    assert Lineout.read_spatial_spec("X", "x") == "X"


def test_read_spatial_spec_rejects_wrong_axis() -> None:
    with pytest.raises(ValueError, match="expected specifier"):
        Lineout.read_spatial_spec("y", "x")


def test_apply_material_coordinate_lineout() -> None:
    header = ["COORDX", "COORDY", "TEMP"]
    data = np.asarray(
        [[0.0, 0.0, 10.0], [1.0, 0.0, 11.0], [0.0, 1.0, 20.0], [1.0, 1.0, 21.0]], dtype=float
    )

    result_header, result_data = Lineout(x="x", y=0.0, tol=1.0e-12).apply(header, data)

    assert result_header == ["COORDX", "TEMP"]
    assert np.allclose(result_data, [[0.0, 10.0], [1.0, 11.0]])


def test_apply_sorts_by_free_coordinate() -> None:
    header = ["COORDX", "COORDY", "TEMP"]
    data = np.asarray([[1.0, 0.0, 11.0], [0.0, 0.0, 10.0]], dtype=float)

    _, result_data = Lineout(x="x", y=0.0, tol=1.0e-12).apply(header, data)

    assert np.allclose(result_data[:, 0], [0.0, 1.0])


def test_apply_with_index_column() -> None:
    header = ["index", "COORDX", "COORDY", "TEMP"]
    data = np.asarray([[2.0, 1.0, 0.0, 11.0], [1.0, 0.0, 0.0, 10.0]], dtype=float)

    result_header, result_data = Lineout(x="x", y=0.0, tol=1.0e-12).apply(header, data)

    assert result_header == ["index", "COORDX", "TEMP"]
    assert np.allclose(result_data[:, 1], [0.0, 1.0])


def test_apply_displaced_coordinate_lineout() -> None:
    header = ["DISPLX", "DISPLY", "COORDX", "COORDY", "TEMP"]
    data = np.asarray(
        [[0.1, 0.0, 0.0, 0.0, 10.0], [0.1, 0.0, 1.0, 0.0, 11.0], [0.1, 0.0, 0.0, 1.0, 20.0]],
        dtype=float,
    )

    result_header, result_data = Lineout(x="X", y=0.0, tol=1.0e-12).apply(header, data)

    assert result_header == ["LOCATIONX", "TEMP"]
    assert np.allclose(result_data, [[0.1, 10.0], [1.1, 11.0]])


def test_apply_displaced_coordinate_requires_displacement_columns() -> None:
    header = ["COORDX", "COORDY", "TEMP"]
    data = np.asarray([[0.0, 0.0, 10.0]], dtype=float)

    with pytest.raises(ValueError, match="requires displacement"):
        Lineout(x="X", y=0.0).apply(header, data)


def test_apply_structured_array() -> None:
    dtype = np.dtype([("COORDX", "f8"), ("COORDY", "f8"), ("TEMP", "f8")])
    data = np.asarray([(0.0, 0.0, 10.0), (1.0, 0.0, 11.0), (0.0, 1.0, 20.0)], dtype=dtype)

    result = Lineout(x="x", y=0.0, tol=1.0e-12).apply(data)

    assert result.dtype.names == ("COORDX", "TEMP")
    assert np.allclose(result["COORDX"], [0.0, 1.0])
    assert np.allclose(result["TEMP"], [10.0, 11.0])


def test_apply_single_argument_requires_structured_array() -> None:
    with pytest.raises(TypeError, match="structured array"):
        Lineout(x="x").apply(np.asarray([[0.0, 1.0]]))


def test_apply_rejects_bad_data_rank() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        Lineout(x="x").apply(["COORDX"], np.zeros((1, 1, 1)))


def test_apply_rejects_header_without_coordinates() -> None:
    with pytest.raises(ValueError, match="coordinate columns"):
        Lineout(x="x").apply(["TEMP"], np.zeros((1, 1)))


def test_compute_default_tolerance() -> None:
    data = np.asarray([[0.0, 0.0], [10.0, 2.0]], dtype=float)

    tol = Lineout(x=1.0, y="y").compute_tol_from_bounding_box(data)

    assert tol == pytest.approx(1.0e-3)


def test_compute_default_tolerance_no_restricted_coordinates() -> None:
    data = np.asarray([[0.0, 0.0], [10.0, 2.0]], dtype=float)

    tol = Lineout(x="x", y="y").compute_tol_from_bounding_box(data)

    assert tol == 0.0


def test_compute_default_tolerance_empty_data() -> None:
    tol = Lineout(x=1.0).compute_tol_from_bounding_box(np.empty((0, 1)))

    assert tol == 0.0
