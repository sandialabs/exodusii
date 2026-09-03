# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Tests for the exodiff-style pure-Python comparison (:mod:`exodusii.api.diff`)."""

from pathlib import Path

import numpy as np

import exodusii
from exodusii.api.diff import TimeSelection
from exodusii.api.diff import DiffOptions
from exodusii.api.diff import DiffResult
from exodusii.api.diff import diff
from exodusii.api.file import ExodusFile
from exodusii.api.writer import ExodusWriter
from exodusii.core.entities import Entity
from exodusii.core.tolerance import Tolerance
from exodusii.core.tolerance import ToleranceMode

# ---------------------------------------------------------------------------
# Fixtures builders
# ---------------------------------------------------------------------------


def _coords(node_count: int = 4) -> np.ndarray:
    base = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    if node_count == 4:
        return base
    extra = np.zeros((node_count - 4, 2), dtype=float)
    return np.vstack([base, extra])


def _write_basic(
    path: Path,
    *,
    temp_offset: float = 0.0,
    energy_offset: float = 0.0,
    global_offset: float = 0.0,
    coord_offset: float = 0.0,
    node_count: int = 4,
    attr_b: float = 2.0,
    nan_temp: bool = False,
) -> None:
    """Write a small 1-block, 2-step file with global/node/element vars + attrs."""

    coords = _coords(node_count) + coord_offset
    with ExodusWriter.create(path) as writer:
        writer.initialize("diff", 2, node_count, 1, element_blocks=1)
        writer.write_coordinates(coords)
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.write_block_attributes(Entity.ELEMENT_BLOCK, 10, [[1.0, attr_b]], names=["A", "B"])
        writer.define_global_variables(["TM_STEP"])
        writer.define_node_variables(["TEMP"])
        writer.define_element_variables(["ENERGY"], truth_table=[[1]])

        temp0 = np.arange(node_count, dtype=float) + temp_offset
        if nan_temp:
            temp0 = temp0.copy()
            temp0[0] = np.nan

        writer.write_time(0.0)
        writer.write_global_values([0.0 + global_offset])
        writer.write_node_values("TEMP", temp0)
        writer.write_element_values("ENERGY", [1.0 + energy_offset], block_id=10)

        writer.write_time(1.0)
        writer.write_global_values([1.0 + global_offset])
        writer.write_node_values("TEMP", np.arange(node_count, dtype=float) + 10.0 + temp_offset)
        writer.write_element_values("ENERGY", [2.0 + energy_offset], block_id=10)


# ---------------------------------------------------------------------------
# Basic same / different
# ---------------------------------------------------------------------------


def test_identical_files_are_same(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a)
    _write_basic(b)

    result = diff(a, b)
    assert isinstance(result, DiffResult)
    assert result.same
    assert bool(result)
    assert not result.errors
    assert not any(vd.exceeded for vd in result.variable_diffs)


def test_diff_is_exported_from_package(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a)
    _write_basic(b)
    assert exodusii.diff(a, b).same


def test_accepts_open_files(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a)
    _write_basic(b)
    with ExodusFile.open(a) as f1, ExodusFile.open(b) as f2:
        assert diff(f1, f2).same


def test_detects_nodal_variable_difference(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a)
    _write_basic(b, temp_offset=1.0)

    result = diff(a, b)
    assert not result.same
    diffs = {vd.name: vd for vd in result.variable_diffs}
    assert "TEMP" in diffs
    assert diffs["TEMP"].exceeded
    assert diffs["TEMP"].entity == Entity.NODE.value


def test_detects_element_variable_difference(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a)
    _write_basic(b, energy_offset=1.0)

    result = diff(a, b)
    assert not result.same
    diffs = {vd.name: vd for vd in result.variable_diffs}
    assert "ENERGY" in diffs
    assert diffs["ENERGY"].block_id == 10


def test_detects_global_variable_difference(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a)
    _write_basic(b, global_offset=5.0)

    result = diff(a, b)
    assert not result.same
    assert any(vd.name == "TM_STEP" for vd in result.variable_diffs)


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------


def test_tolerance_absorbs_tiny_difference(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a)
    _write_basic(b, temp_offset=1.0e-13)

    # A pure relative tolerance with no floor treats 0.0 -> 1e-13 as an
    # infinite relative change (matching reference exodiff).  A small floor
    # absorbs near-zero noise, as does an absolute tolerance.
    floored = DiffOptions(default_tolerance=Tolerance(ToleranceMode.RELATIVE, 1.0e-6, 1.0e-12))
    assert diff(a, b, floored).same


def test_absolute_tolerance_mode(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a)
    _write_basic(b, temp_offset=1.0e-4)

    tight = DiffOptions(default_tolerance=Tolerance(ToleranceMode.ABSOLUTE, 1.0e-6, 0.0))
    loose = DiffOptions(default_tolerance=Tolerance(ToleranceMode.ABSOLUTE, 1.0e-2, 0.0))
    assert not diff(a, b, tight).same
    assert diff(a, b, loose).same


def test_per_variable_tolerance_override(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a)
    _write_basic(b, temp_offset=1.0)

    opts = DiffOptions(variable_tolerances={"TEMP": Tolerance(ToleranceMode.ABSOLUTE, 100.0, 0.0)})
    # TEMP diff is within the huge override; nothing else changed -> same.
    assert diff(a, b, opts).same


def test_per_category_tolerance_override(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a)
    _write_basic(b, temp_offset=1.0)

    opts = DiffOptions(nodal_tolerance=Tolerance(ToleranceMode.ABSOLUTE, 100.0, 0.0))
    assert diff(a, b, opts).same


# ---------------------------------------------------------------------------
# Exclusion and selection
# ---------------------------------------------------------------------------


def test_exclude_variable(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a)
    _write_basic(b, temp_offset=1.0)

    opts = DiffOptions(exclude=frozenset({"TEMP"}))
    assert diff(a, b, opts).same


def test_show_all_records_within_tolerance(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a)
    _write_basic(b)

    opts = DiffOptions(show_all=True)
    result = diff(a, b, opts)
    names = {vd.name for vd in result.variable_diffs}
    assert {"TEMP", "ENERGY", "TM_STEP"}.issubset(names)
    assert result.same  # nothing exceeded


# ---------------------------------------------------------------------------
# Structural differences
# ---------------------------------------------------------------------------


def test_detects_node_count_difference(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a)
    _write_basic(b, node_count=5)

    result = diff(a, b)
    assert not result.same
    assert any("node count" in e for e in result.errors)


def test_detects_missing_variable(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a)

    with ExodusWriter.create(b) as writer:
        writer.initialize("diff", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(_coords())
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.write_block_attributes(Entity.ELEMENT_BLOCK, 10, [[1.0, 2.0]], names=["A", "B"])
        writer.define_global_variables(["TM_STEP"])
        writer.define_element_variables(["ENERGY"], truth_table=[[1]])
        writer.write_time(0.0)
        writer.write_global_values([0.0])
        writer.write_element_values("ENERGY", [1.0], block_id=10)
        writer.write_time(1.0)
        writer.write_global_values([1.0])
        writer.write_element_values("ENERGY", [2.0], block_id=10)

    result = diff(a, b)
    assert not result.same
    assert any("TEMP" in e and "missing from file2" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------


def test_detects_coordinate_difference(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a)
    _write_basic(b, coord_offset=1.0)

    result = diff(a, b)
    assert not result.same
    assert any("coordinates differ" in e for e in result.errors)
    assert result.coordinate_max_delta is not None
    assert result.coordinate_max_delta > 0.5


def test_can_skip_coordinates(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a)
    _write_basic(b, coord_offset=1.0)

    opts = DiffOptions(compare_coordinates=False)
    result = diff(a, b, opts)
    assert result.same


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


def test_detects_attribute_difference(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a, attr_b=2.0)
    _write_basic(b, attr_b=9.0)

    result = diff(a, b)
    assert not result.same
    assert any(vd.name == "attr:B" for vd in result.variable_diffs)


def test_can_skip_attributes(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a, attr_b=2.0)
    _write_basic(b, attr_b=9.0)

    opts = DiffOptions(compare_attributes=False)
    assert diff(a, b, opts).same


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_nan_mismatch_is_a_difference(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a, nan_temp=False)
    _write_basic(b, nan_temp=True)

    result = diff(a, b)
    assert not result.same
    temp = next(vd for vd in result.variable_diffs if vd.name == "TEMP")
    assert temp.exceeded
    assert temp.max_delta == float("inf")


def test_matching_nan_is_not_a_difference(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a, nan_temp=True)
    _write_basic(b, nan_temp=True)

    result = diff(a, b)
    # NaN in the same place in both files -> not flagged as a NaN mismatch.
    temp = [vd for vd in result.variable_diffs if vd.name == "TEMP"]
    assert not temp or not temp[0].exceeded


# ---------------------------------------------------------------------------
# Time steps
# ---------------------------------------------------------------------------


def test_time_step_count_mismatch_warns(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_basic(a)

    with ExodusWriter.create(b) as writer:
        writer.initialize("diff", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(_coords())
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.write_block_attributes(Entity.ELEMENT_BLOCK, 10, [[1.0, 2.0]], names=["A", "B"])
        writer.define_global_variables(["TM_STEP"])
        writer.define_node_variables(["TEMP"])
        writer.define_element_variables(["ENERGY"], truth_table=[[1]])
        writer.write_time(0.0)
        writer.write_global_values([0.0])
        writer.write_node_values("TEMP", np.arange(4, dtype=float))
        writer.write_element_values("ENERGY", [1.0], block_id=10)

    result = diff(a, b)
    assert any("time step count differs" in w for w in result.warnings)
    # Only the overlapping step is compared, and it matches -> same.
    assert result.same


# ---------------------------------------------------------------------------
# Node-set variables (truth-table aware)
# ---------------------------------------------------------------------------


def _write_with_node_set(path: Path, *, nsvar_offset: float = 0.0) -> None:
    with ExodusWriter.create(path) as writer:
        writer.initialize("ns", 2, 4, 1, element_blocks=1, node_sets=1)
        writer.write_coordinates(_coords())
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_node_set(100, [1, 2])
        writer.define_node_set_variables(["NSVAR"], truth_table=[[1]])
        writer.write_time(0.0)
        writer.write_node_set_values("NSVAR", [1.0 + nsvar_offset, 2.0], set_id=100)


def test_node_set_variable_difference(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_with_node_set(a)
    _write_with_node_set(b, nsvar_offset=5.0)

    result = diff(a, b)
    assert not result.same
    assert any(vd.name == "NSVAR" and vd.set_id == 100 for vd in result.variable_diffs)


def test_node_set_variable_same(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_with_node_set(a)
    _write_with_node_set(b)
    assert diff(a, b).same


# ---------------------------------------------------------------------------
# Phase 2 — Time-step selection helpers
# ---------------------------------------------------------------------------


def _write_multistep(
    path: Path,
    *,
    n_steps: int = 5,
    temp_scale: float = 1.0,
    start_time: float = 0.0,
    dt: float = 1.0,
) -> None:
    """Write a 4-node, 1-element, n-step file with TEMP nodal variable."""
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    with ExodusWriter.create(path) as writer:
        writer.initialize("ms", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(coords)
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_node_variables(["TEMP"])
        for s in range(n_steps):
            t = start_time + s * dt
            writer.write_time(t)
            writer.write_node_values("TEMP", np.full(4, float(s) * temp_scale))


# ---------------------------------------------------------------------------
# Phase 2 — start / stop / increment
# ---------------------------------------------------------------------------


def test_start_selects_first_step(tmp_path: Path) -> None:
    """start=3 should skip steps 1-2; steps 3-5 are compared."""
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    # File-1 and file-2 identical except step 1 and 2 differ.
    # Without selection both would differ; with start=3 they should be same.
    _write_multistep(a, n_steps=5, temp_scale=1.0)

    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    with ExodusWriter.create(b) as writer:
        writer.initialize("ms", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(coords)
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_node_variables(["TEMP"])
        for s in range(5):
            writer.write_time(float(s))
            # Steps 0-1 (1-based 1-2) differ; steps 2-4 (1-based 3-5) match.
            val = float(s) if s >= 2 else float(s) + 100.0
            writer.write_node_values("TEMP", np.full(4, val))

    ts = TimeSelection(start=3)
    opts = DiffOptions(time_selection=ts)
    assert diff(a, b, opts).same


def test_stop_limits_last_step(tmp_path: Path) -> None:
    """stop=2 should compare only steps 1-2; step 3+ differ but are ignored."""
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_multistep(a, n_steps=4, temp_scale=1.0)

    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    with ExodusWriter.create(b) as writer:
        writer.initialize("ms", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(coords)
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_node_variables(["TEMP"])
        for s in range(4):
            writer.write_time(float(s))
            val = float(s) if s < 2 else float(s) + 100.0
            writer.write_node_values("TEMP", np.full(4, val))

    ts = TimeSelection(stop=2)
    opts = DiffOptions(time_selection=ts)
    assert diff(a, b, opts).same


def test_increment_skips_steps(tmp_path: Path) -> None:
    """increment=2 compares steps 1, 3, 5 of file-2; steps 2, 4 differ."""
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_multistep(a, n_steps=5, temp_scale=1.0)

    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    with ExodusWriter.create(b) as writer:
        writer.initialize("ms", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(coords)
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_node_variables(["TEMP"])
        for s in range(5):
            writer.write_time(float(s))
            # Odd 0-based indices (1-based 2, 4) differ.
            val = float(s) if s % 2 == 0 else float(s) + 100.0
            writer.write_node_values("TEMP", np.full(4, val))

    ts = TimeSelection(increment=2)
    opts = DiffOptions(time_selection=ts)
    assert diff(a, b, opts).same


# ---------------------------------------------------------------------------
# Phase 2 — LAST sentinel
# ---------------------------------------------------------------------------


def test_last_compares_only_final_step(tmp_path: Path) -> None:
    """start=-1 (LAST) should compare only the final step; earlier steps differ."""
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_multistep(a, n_steps=3, temp_scale=1.0)

    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    with ExodusWriter.create(b) as writer:
        writer.initialize("ms", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(coords)
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_node_variables(["TEMP"])
        for s in range(3):
            writer.write_time(float(s))
            # Only the last step (s=2) matches; others differ.
            val = float(s) if s == 2 else float(s) + 50.0
            writer.write_node_values("TEMP", np.full(4, val))

    ts = TimeSelection(start=-1)
    opts = DiffOptions(time_selection=ts)
    assert diff(a, b, opts).same


# ---------------------------------------------------------------------------
# Phase 2 — exclude_steps
# ---------------------------------------------------------------------------


def test_exclude_steps_skips_listed_steps(tmp_path: Path) -> None:
    """Steps in exclude_steps are not compared even when they differ."""
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_multistep(a, n_steps=4, temp_scale=1.0)

    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    with ExodusWriter.create(b) as writer:
        writer.initialize("ms", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(coords)
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_node_variables(["TEMP"])
        for s in range(4):
            writer.write_time(float(s))
            # Steps 2 and 4 (1-based) differ.
            val = float(s) if s in (0, 2) else float(s) + 200.0
            writer.write_node_values("TEMP", np.full(4, val))

    ts = TimeSelection(exclude_steps=frozenset({2, 4}))
    opts = DiffOptions(time_selection=ts)
    assert diff(a, b, opts).same


# ---------------------------------------------------------------------------
# Phase 2 — time_step_offset via TimeSelection
# ---------------------------------------------------------------------------


def test_time_step_offset_via_time_selection(tmp_path: Path) -> None:
    """time_step_offset=1 maps file-2 step n to file-1 step n+1.

    file-1 has 5 steps with TEMP = 0,1,2,3,4.
    file-2 has 4 steps with TEMP = 1,2,3,4 (same as file-1 steps 2..5).
    With offset=1, file-2 step 1 compares to file-1 step 2, etc. -> same.
    """
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_multistep(a, n_steps=5, temp_scale=1.0)

    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    with ExodusWriter.create(b) as writer:
        writer.initialize("ms", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(coords)
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_node_variables(["TEMP"])
        for s in range(4):
            writer.write_time(float(s + 1))
            # TEMP = s+1 to match file-1 steps 2..5 (0-based 1..4)
            writer.write_node_values("TEMP", np.full(4, float(s + 1)))

    ts = TimeSelection(time_step_offset=1)
    opts = DiffOptions(time_selection=ts)
    assert diff(a, b, opts).same


# ---------------------------------------------------------------------------
# Phase 2 — time_value_scale / time_value_offset
# ---------------------------------------------------------------------------


def test_time_value_scale_and_offset(tmp_path: Path) -> None:
    """time_value_scale/offset adjusts file-1 times before matching."""
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    # file-1 times: 0, 2, 4, 6, 8 (dt=2)
    _write_multistep(a, n_steps=5, dt=2.0)
    # file-2 times: 0, 1, 2, 3, 4 (dt=1) — same values, half the time axis
    _write_multistep(b, n_steps=5, dt=1.0)

    # With scale=0.5, file-1 adjusted times become 0, 1, 2, 3, 4 — matching file-2.
    ts = TimeSelection(time_value_scale=0.5, interpolating=True)
    opts = DiffOptions(
        time_selection=ts,
        time_tolerance=Tolerance(ToleranceMode.ABSOLUTE, 1.0e-10),
    )
    result = diff(a, b, opts)
    assert result.same, result.errors


# ---------------------------------------------------------------------------
# Phase 2 — interpolating
# ---------------------------------------------------------------------------


def test_interpolating_exact_match_is_same(tmp_path: Path) -> None:
    """When file-1 times exactly coincide with file-2 times, interpolating gives same result."""
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_multistep(a, n_steps=4)
    _write_multistep(b, n_steps=4)

    ts = TimeSelection(interpolating=True)
    opts = DiffOptions(time_selection=ts)
    assert diff(a, b, opts).same


def test_interpolating_midpoint_values(tmp_path: Path) -> None:
    """Interpolating at the midpoint between two file-2 steps gives correct values."""
    # file-1 times: 0.5, 1.5, 2.5  (midpoints between file-2 steps)
    # file-1 TEMP:  linearly interpolated values from file-2
    # file-2 times: 0, 1, 2, 3  with TEMP = s*10 at step s (0-based)
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)

    # file-1: times at midpoints; TEMP = interpolated value = (s + 0.5) * 10
    with ExodusWriter.create(a) as writer:
        writer.initialize("interp", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(coords)
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_node_variables(["TEMP"])
        for s in range(3):
            writer.write_time(float(s) + 0.5)
            writer.write_node_values("TEMP", np.full(4, (s + 0.5) * 10.0))

    # file-2: integer times, TEMP = s*10
    with ExodusWriter.create(b) as writer:
        writer.initialize("interp", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(coords)
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_node_variables(["TEMP"])
        for s in range(4):
            writer.write_time(float(s))
            writer.write_node_values("TEMP", np.full(4, float(s * 10)))

    ts = TimeSelection(interpolating=True)
    # Use a tight absolute tolerance; interpolated values should be exact.
    opts = DiffOptions(
        time_selection=ts,
        default_tolerance=Tolerance(ToleranceMode.ABSOLUTE, 1.0e-10),
        compare_coordinates=False,
    )
    result = diff(a, b, opts)
    assert result.same, result.errors


def test_interpolating_outside_range_skipped(tmp_path: Path) -> None:
    """File-1 times outside the file-2 time range are skipped when interpolating."""
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)

    # file-1: times 0, 1, 2, 5 — last step (5) is outside file-2 range [0,3].
    with ExodusWriter.create(a) as writer:
        writer.initialize("skip", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(coords)
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_node_variables(["TEMP"])
        for t in [0.0, 1.0, 2.0, 5.0]:
            writer.write_time(t)
            writer.write_node_values("TEMP", np.full(4, t))

    with ExodusWriter.create(b) as writer:
        writer.initialize("skip", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(coords)
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_node_variables(["TEMP"])
        for t in [0.0, 1.0, 2.0, 3.0]:
            writer.write_time(t)
            writer.write_node_values("TEMP", np.full(4, t))

    ts = TimeSelection(interpolating=True)
    opts = DiffOptions(
        time_selection=ts,
        default_tolerance=Tolerance(ToleranceMode.ABSOLUTE, 1.0e-10),
        compare_coordinates=False,
    )
    result = diff(a, b, opts)
    # Steps 0-2 match; step 3 (t=5.0) is out of range and skipped -> same.
    assert result.same, result.errors


# ---------------------------------------------------------------------------
# Phase 2 — backward compat: legacy time_step_offset on DiffOptions
# ---------------------------------------------------------------------------


def test_legacy_time_step_offset_still_works(tmp_path: Path) -> None:
    """time_step_offset on DiffOptions (no time_selection) still works as before.

    file-1: 5 steps, TEMP=0..4.
    file-2: 4 steps, TEMP=1..4 (matching file-1 steps 2..5).
    With legacy time_step_offset=1 -> file-2 step n maps to file-1 step n+1.
    """
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_multistep(a, n_steps=5)

    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    with ExodusWriter.create(b) as writer:
        writer.initialize("ms", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(coords)
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_node_variables(["TEMP"])
        for s in range(4):
            writer.write_time(float(s + 1))
            writer.write_node_values("TEMP", np.full(4, float(s + 1)))

    opts = DiffOptions(time_step_offset=1)
    assert diff(a, b, opts).same
