# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Legacy 1-based time_step semantics (exodusii < 2)."""

from pathlib import Path

import numpy as np
import pytest

import exodusii
from exodusii.api.writer import ExodusWriter
from exodusii.compat._time import legacy_time_index


@pytest.mark.parametrize(
    ("time_step", "expected"), [(None, None), (1, 0), (3, 2), (-1, -1), (-2, -1), (-99, -1)]
)
def test_legacy_time_index(time_step: int | None, expected: int | None) -> None:
    assert legacy_time_index(time_step) == expected


def _write_three_steps(path: Path) -> None:
    with ExodusWriter.create(path) as writer:
        writer.initialize("ts", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float))
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_global_variables(["G"])
        writer.define_node_variables(["T", "DISPLX", "DISPLY"])
        writer.define_element_variables(["E"])
        for step in range(3):
            writer.write_time(float(step))
            writer.write_global_values([10.0 * step])
            writer.write_node_values("T", np.full(4, float(step)))
            writer.write_node_values("DISPLX", np.full(4, 0.5 * step))
            writer.write_node_values("DISPLY", np.zeros(4))
            writer.write_element_values("E", [100.0 + step], block_id=10)


@pytest.mark.parametrize("negative", [-1, -5])
def test_negative_time_step_is_last_step(tmp_path: Path, negative: int) -> None:
    # Before the fix time_step=-1 silently returned the second-to-last step.
    path = tmp_path / "ts.exo"
    _write_three_steps(path)

    exo = exodusii.File(path)

    assert np.array_equal(exo.get_node_variable_values("T", negative), np.full(4, 2.0))
    assert np.array_equal(exo.get_element_variable_values(10, "E", negative), [102.0])
    assert exo.get_global_variable_value("G", negative) == 20.0
    assert exo.get_all_global_variable_values(negative).tolist() == [20.0]
    assert np.array_equal(exo.get_variable_values("g", "G", negative), 20.0)
    assert np.allclose(exo.get_displ(negative)[:, 0], 1.0)
    assert np.allclose(exo.get_coords(time_step=negative)[:, 0], [1.0, 2.0, 2.0, 1.0])


def test_positive_time_step_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "ts.exo"
    _write_three_steps(path)

    exo = exodusii.File(path)

    assert np.array_equal(exo.get_node_variable_values("T", 1), np.zeros(4))
    assert np.array_equal(exo.get_node_variable_values("T", 2), np.ones(4))
    assert exo.get_time(3) == 2.0
    assert exo.get_node_variable_values("T").shape == (3, 4)


def test_legacy_get_missing_variable_is_value_error(tmp_path: Path) -> None:
    # vvtk grind_graphs skips absent variables via `except ValueError`.
    from exodusii.api.lineout import Lineout

    path = tmp_path / "ts.exo"
    _write_three_steps(path)

    with pytest.raises(ValueError, match="not found"):
        exodusii.File(path).get("e/MISSING", time=1.0, lineout=Lineout(x="X", y=0.5, tol=1.0))
