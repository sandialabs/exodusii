# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from io import StringIO
from pathlib import Path

import numpy as np
import pytest

import exodusii
from exodusii.api.compare import ComparisonResult
from exodusii.api.compare import allclose
from exodusii.api.compare import similar
from exodusii.api.file import ExodusFile
from exodusii.api.writer import ExodusWriter


def test_allclose_identical_files(tmp_path: Path) -> None:
    path1 = tmp_path / "one.exo"
    path2 = tmp_path / "two.exo"
    _write_file(path1)
    _write_file(path2)

    assert allclose(path1, path2)
    assert exodusii.allclose(path1, path2)


def test_allclose_accepts_open_files(tmp_path: Path) -> None:
    path1 = tmp_path / "one.exo"
    path2 = tmp_path / "two.exo"
    _write_file(path1)
    _write_file(path2)

    with ExodusFile.open(path1) as file1, ExodusFile.open(path2) as file2:
        assert allclose(file1, file2)


def test_allclose_detects_variable_difference(tmp_path: Path) -> None:
    path1 = tmp_path / "one.exo"
    path2 = tmp_path / "two.exo"
    _write_file(path1)
    _write_file(path2, temp_offset=1.0)

    assert not allclose(path1, path2)


def test_allclose_respects_tolerance(tmp_path: Path) -> None:
    path1 = tmp_path / "one.exo"
    path2 = tmp_path / "two.exo"
    _write_file(path1)
    _write_file(path2, temp_offset=1.0e-13)

    assert allclose(path1, path2, atol=1.0e-12, rtol=1.0e-12)
    assert not allclose(path1, path2, atol=1.0e-15, rtol=0.0)


def test_allclose_can_skip_variables(tmp_path: Path) -> None:
    path1 = tmp_path / "one.exo"
    path2 = tmp_path / "two.exo"
    _write_file(path1)
    _write_file(path2, temp_offset=1.0)

    assert allclose(path1, path2, variables=None)


def test_allclose_can_select_variables(tmp_path: Path) -> None:
    path1 = tmp_path / "one.exo"
    path2 = tmp_path / "two.exo"
    _write_file(path1)
    _write_file(path2, temp_offset=1.0)

    assert allclose(path1, path2, variables=["coordx", "coordy"])
    assert not allclose(path1, path2, variables=["vals_nod_var1"])


def test_allclose_can_negate_variable_selection(tmp_path: Path) -> None:
    path1 = tmp_path / "one.exo"
    path2 = tmp_path / "two.exo"
    _write_file(path1)
    _write_file(path2, temp_offset=1.0)

    assert allclose(path1, path2, variables="~vals_nod_var1")


def test_allclose_returns_detailed_result(tmp_path: Path) -> None:
    path1 = tmp_path / "one.exo"
    path2 = tmp_path / "two.exo"
    _write_file(path1)
    _write_file(path2, temp_offset=1.0)

    result = allclose(path1, path2, result=True)

    assert isinstance(result, ComparisonResult)
    assert not result.equal
    assert not bool(result)
    assert any("vals_nod_var1" in error for error in result.errors)


def test_allclose_verbose_writes_errors(tmp_path: Path) -> None:
    path1 = tmp_path / "one.exo"
    path2 = tmp_path / "two.exo"
    stream = StringIO()
    _write_file(path1)
    _write_file(path2, temp_offset=1.0)

    assert not allclose(path1, path2, verbose=stream)
    assert "==> Error:" in stream.getvalue()


def test_allclose_detects_dimension_difference(tmp_path: Path) -> None:
    path1 = tmp_path / "one.exo"
    path2 = tmp_path / "two.exo"
    _write_file(path1)
    _write_file(path2, node_count=5)

    assert not allclose(path1, path2, variables=None)


def test_allclose_can_skip_dimensions(tmp_path: Path) -> None:
    path1 = tmp_path / "one.exo"
    path2 = tmp_path / "two.exo"
    _write_file(path1)
    _write_file(path2, node_count=5)

    assert allclose(path1, path2, dimensions="~num_nodes", variables=None)


def test_similar_identical_files(tmp_path: Path) -> None:
    path1 = tmp_path / "one.exo"
    path2 = tmp_path / "two.exo"
    _write_file(path1)
    _write_file(path2)

    assert similar(path1, path2)


def test_similar_ignores_result_value_difference(tmp_path: Path) -> None:
    path1 = tmp_path / "one.exo"
    path2 = tmp_path / "two.exo"
    _write_file(path1)
    _write_file(path2, temp_offset=1.0)

    assert similar(path1, path2)


def test_similar_detects_coordinate_difference(tmp_path: Path) -> None:
    path1 = tmp_path / "one.exo"
    path2 = tmp_path / "two.exo"
    _write_file(path1)
    _write_file(path2, coord_offset=1.0)

    with pytest.raises(ValueError, match="same node coordinates"):
        similar(path1, path2)


def test_similar_detects_connectivity_difference(tmp_path: Path) -> None:
    path1 = tmp_path / "one.exo"
    path2 = tmp_path / "two.exo"
    _write_file(path1)
    _write_file(path2, connectivity=[[1, 2, 4, 3]])

    with pytest.raises(ValueError, match=r"element.*connectivity"):
        similar(path1, path2)


def test_similar_checks_requested_times(tmp_path: Path) -> None:
    path1 = tmp_path / "one.exo"
    path2 = tmp_path / "two.exo"
    _write_file(path1)
    _write_file(path2)

    assert similar(path1, path2, times=[0.0, 1.0])

    with pytest.raises(ValueError, match="requested times"):
        similar(path1, path2, times=[99.0])


def _write_file(
    path: Path,
    *,
    temp_offset: float = 0.0,
    coord_offset: float = 0.0,
    connectivity: list[list[int]] | None = None,
    node_count: int = 4,
) -> None:
    coords = _coords(node_count) + coord_offset
    conn = connectivity if connectivity is not None else [[1, 2, 3, 4]]

    with ExodusWriter.create(path) as writer:
        writer.initialize("compare", 2, node_count, 1, element_blocks=1)
        writer.write_coordinates(coords)
        writer.define_element_block(10, "quad", conn)
        writer.define_node_variables(["TEMP"])
        writer.define_element_variables(["ENERGY"])

        writer.write_time(0.0)
        writer.write_node_values("TEMP", np.arange(node_count, dtype=float) + temp_offset)
        writer.write_element_values("ENERGY", [1.0], block_id=10)

        writer.write_time(1.0)
        writer.write_node_values("TEMP", np.arange(node_count, dtype=float) + 10.0 + temp_offset)
        writer.write_element_values("ENERGY", [2.0], block_id=10)


def _coords(node_count: int) -> np.ndarray:
    base = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)

    if node_count == 4:
        return base

    extra = np.zeros((node_count - 4, 2), dtype=float)
    return np.vstack([base, extra])
