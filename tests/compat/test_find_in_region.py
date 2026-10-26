# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Legacy find_element_data_in_region / find_node_data_in_region."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

import exodusii
from exodusii.api.writer import ExodusWriter
from exodusii.region import bounded_time_domain
from exodusii.region import rectangle


def _write_strip(path: Path) -> None:
    # Three unit quads along x; element centers at x = 0.5, 1.5, 2.5.
    coords = np.asarray(
        [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1]], dtype=float
    )
    with ExodusWriter.create(path) as writer:
        writer.initialize("strip", 2, 8, 3, element_blocks=2)
        writer.write_coordinates(coords)
        writer.define_element_block(1, "quad", [[2, 3, 7, 6], [1, 2, 6, 5]])
        writer.define_element_block(2, "quad", [[3, 4, 8, 7]])
        writer.define_node_variables(["T", "DISPLX", "DISPLY"])
        writer.define_element_variables(["E"])
        for step in range(3):
            writer.write_time(float(step))
            writer.write_node_values("T", np.arange(8, dtype=float) + 10 * step)
            writer.write_node_values("DISPLX", np.full(8, 1.0 * step))
            writer.write_node_values("DISPLY", np.zeros(8))
            writer.write_element_values("E", [20.0 + step, 10.0 + step], block_id=1)
            writer.write_element_values("E", [30.0 + step], block_id=2)


def test_element_data_legacy_signature_and_shape(tmp_path: Path) -> None:
    path = tmp_path / "strip.exo"
    _write_strip(path)
    region = rectangle((0.0, 0.0), 2.0, 1.0)  # contains centers x=0.5, 1.5

    data = exodusii.find_element_data_in_region(str(path), ["E"], region)

    assert isinstance(data, OrderedDict)
    assert list(data) == [0, 1, 2]
    cycle = data[2]
    assert list(cycle) == ["cycle", "time", "X", "Y", "E"]
    assert cycle["cycle"] == 2
    assert cycle["time"] == 2.0
    # Rows sorted by X across blocks.
    assert np.allclose(cycle["X"], [0.5, 1.5])
    assert np.allclose(cycle["Y"], [0.5, 0.5])
    assert np.allclose(cycle["E"], [12.0, 22.0])


def test_element_data_time_domain_and_block_ids(tmp_path: Path) -> None:
    path = tmp_path / "strip.exo"
    _write_strip(path)
    exo = exodusii.File(path)
    everything = rectangle((0.0, 0.0), 3.0, 1.0)

    data = exodusii.find_element_data_in_region(
        exo, ["E"], everything, time_domain=bounded_time_domain(0.5, 1.5), block_ids=[2]
    )

    assert list(data) == [1]
    assert np.allclose(data[1]["X"], [2.5])
    assert np.allclose(data[1]["E"], [31.0])


def test_element_data_displaced_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "strip.exo"
    _write_strip(path)
    region = rectangle((2.0, 0.0), 2.0, 1.0)  # x in [2, 4]

    data = exodusii.find_element_data_in_region([path], ["E"], region, use_displaced_coords=True)

    # Displaced by +step in x; region keeps displaced centers with x in [2, 4].
    assert np.allclose(data[0]["X"], [2.5])
    assert np.allclose(data[0]["E"], [30.0])
    assert np.allclose(data[1]["X"], [2.5, 3.5])
    assert np.allclose(data[1]["E"], [21.0, 31.0])
    assert np.allclose(data[2]["X"], [2.5, 3.5])
    assert np.allclose(data[2]["E"], [12.0, 22.0])


def test_element_data_dimension_mismatch(tmp_path: Path) -> None:
    from exodusii.region import sphere

    path = tmp_path / "strip.exo"
    _write_strip(path)

    with pytest.raises(ValueError, match="dimension"):
        exodusii.find_element_data_in_region(path, ["E"], sphere((0, 0, 0), 1.0))


def test_node_data_legacy(tmp_path: Path) -> None:
    path = tmp_path / "strip.exo"
    _write_strip(path)
    region = rectangle((0.0, 0.0), 1.0, 1.0)  # nodes 1, 2, 5, 6

    data = exodusii.find_node_data_in_region(path, ["T"], region)

    assert list(data) == [0, 1, 2]
    assert list(data[1]) == ["cycle", "time", "X", "Y", "T"]
    assert np.allclose(np.sort(data[1]["T"]), [10.0, 11.0, 14.0, 15.0])
    assert np.allclose(np.sort(data[1]["X"]), [0.0, 0.0, 1.0, 1.0])


def test_find_in_region_module_alias() -> None:
    from exodusii import find_in_region

    assert find_in_region.find_element_data_in_region is exodusii.find_element_data_in_region
    assert find_in_region.find_node_data_in_region is exodusii.find_node_data_in_region
