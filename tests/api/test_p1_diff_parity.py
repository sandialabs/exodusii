# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""SEACAS exodiff parity for diff/CLI (audit P1 batch).

* node/side sets are compared with sorted entries and their member lists must
  agree; side-set distribution factors are compared;
* interpolation covers every file-1 step, snaps to file-2 steps within the
  time tolerance and checks the final time; empty step selections are a
  difference; automatic step offsets;
* exit codes: variables in one file, count mismatches -> 2; truth-table
  mismatches are skipped;
* command-file parsing: bare per-variable values, FINAL TIME TOLERANCE,
  EXCLUDE TIMES ranges, NODESET/SIDESET MATCH, IGNORE SIDESET DISTRIBUTION
  FACTORS, SIDESET DISTRIBUTION, STEP OFFSET AUTOMATIC/MATCH.

exodiff exit status: 0 same, 1 error / incompatible mesh, 2 different.
"""

from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from exodusii.api.command_file import CommandFileError
from exodusii.api.command_file import parse_step_list
from exodusii.api.command_file import read_command_file
from exodusii.api.diff import STEP_OFFSET_AUTOMATIC
from exodusii.api.diff import STEP_OFFSET_MATCH
from exodusii.api.diff import DiffOptions
from exodusii.api.diff import TimeSelection
from exodusii.api.diff import diff
from exodusii.api.writer import ExodusWriter
from exodusii.cli.exodiff import main
from exodusii.core.entities import Entity
from exodusii.core.tolerance import Tolerance
from exodusii.core.tolerance import ToleranceMode

SAME, ERROR, DIFFERENT = 0, 1, 2

COORDS = np.asarray([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=float)
CONN = [[1, 2, 5, 4], [2, 3, 6, 5]]


def _cli(*args: object) -> tuple[int, str]:
    stream = StringIO()
    status = main([str(a) for a in args], file=stream)
    return status, stream.getvalue()


def _write(
    path: Path,
    *,
    times=(0.0, 1.0),
    temp=None,
    nodesets=None,
    nsvals=None,
    sidesets=None,
    ssvals=None,
    globals_=("G",),
    gvals=None,
    nodal=("T",),
    element=None,
    truth=None,
) -> Path:
    with ExodusWriter.create(path) as w:
        w.initialize(
            "t",
            2,
            node_count=6,
            element_count=2,
            element_blocks=2 if truth is not None else 1,
            node_sets=len(nodesets or {}),
            side_sets=len(sidesets or {}),
        )
        w.write_coordinates(COORDS)
        if truth is not None:
            w.define_element_block(10, "quad4", [CONN[0]])
            w.define_element_block(20, "quad4", [CONN[1]])
        else:
            w.define_element_block(10, "quad4", CONN)
        for set_id, spec in (nodesets or {}).items():
            w.define_node_set(set_id, spec)
        for set_id, spec in (sidesets or {}).items():
            w.define_side_set(set_id, spec[0], spec[1], distribution_factors=spec[2])
        if globals_:
            w.define_global_variables(list(globals_))
        if nodal:
            w.define_node_variables(list(nodal))
        if nsvals:
            w.define_node_set_variables(["NV"])
        if ssvals:
            w.define_side_set_variables(["SV"])
        if element:
            w.define_element_variables(list(element), truth_table=truth)
        for step, t in enumerate(times):
            w.write_time(t)
            if globals_:
                value = float(step) if gvals is None else float(gvals[step])
                w.write_global_values([value] * len(globals_))
            for name in nodal or ():
                w.write_node_values(name, temp if temp is not None else np.arange(6.0) + step)
            for set_id, values in (nsvals or {}).items():
                w.write_node_set_values("NV", values, set_id=set_id)
            for set_id, values in (ssvals or {}).items():
                w.write_values("SV", values, on=Entity.SIDE_SET, set_id=set_id)
            element_names = list(element or ())
            for name in element_names:
                for row, block_id in enumerate((10, 20) if truth is not None else (10,)):
                    if truth is not None and not truth[row][element_names.index(name)]:
                        continue
                    n = 1 if truth is not None else 2
                    w.write_element_values(name, np.full(n, 1.0 + step), block_id=block_id)
    return path


# ---------------------------------------------------------------------------
# Node sets / side sets
# ---------------------------------------------------------------------------


def test_node_set_order_does_not_matter(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.exo", nodesets={1: [1, 2, 3]}, nsvals={1: [10.0, 20.0, 30.0]})
    b = _write(tmp_path / "b.exo", nodesets={1: [3, 2, 1]}, nsvals={1: [30.0, 20.0, 10.0]})
    result = diff(a, b)
    assert result.same, (result.errors, result.variable_diffs)

    # A genuinely different value is still reported.
    c = _write(tmp_path / "c.exo", nodesets={1: [3, 2, 1]}, nsvals={1: [30.0, 21.0, 10.0]})
    assert not diff(a, c).same


def test_node_set_member_lists_must_agree(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.exo", nodesets={1: [1, 2, 3]}, nsvals={1: [10.0, 20.0, 30.0]})
    b = _write(tmp_path / "b.exo", nodesets={1: [4, 5, 6]}, nsvals={1: [10.0, 20.0, 30.0]})
    result = diff(a, b)
    assert any("node lists are not the same" in e for e in result.errors)
    assert _cli(a, b)[0] == ERROR


def test_side_set_order_and_distribution_factors(tmp_path: Path) -> None:
    a = _write(
        tmp_path / "a.exo",
        sidesets={1: ([1, 2], [1, 1], [1.0, 1.0, 2.0, 2.0])},
        ssvals={1: [1.0, 2.0]},
    )
    b = _write(
        tmp_path / "b.exo",
        sidesets={1: ([2, 1], [1, 1], [2.0, 2.0, 1.0, 1.0])},
        ssvals={1: [2.0, 1.0]},
    )
    result = diff(a, b)
    assert result.same, (result.errors, result.variable_diffs)

    c = _write(
        tmp_path / "c.exo",
        sidesets={1: ([1, 2], [1, 1], [1.0, 1.0, 2.0, 2.5])},
        ssvals={1: [1.0, 2.0]},
    )
    result = diff(a, c)
    assert not result.same
    (vd,) = [v for v in result.variable_diffs if v.name == "distribution factors"]
    assert vd.set_id == 1
    assert _cli(a, c)[0] == DIFFERENT

    # exodiff -ignore_sideset_df / IGNORE SIDESET DISTRIBUTION FACTORS
    assert diff(a, c, DiffOptions(compare_sideset_df=False)).same
    cmd = tmp_path / "cmd"
    cmd.write_text("IGNORE SIDESET DISTRIBUTION FACTORS\nSIDESET VARIABLES (all)\n")
    assert _cli("-f", cmd, a, c)[0] == SAME
    cmd.write_text("SIDESET DISTRIBUTION absolute 1.0\nSIDESET VARIABLES (all)\n")
    assert _cli("-f", cmd, a, c)[0] == SAME


# ---------------------------------------------------------------------------
# Time steps
# ---------------------------------------------------------------------------


def test_interpolation_compares_every_file1_step(tmp_path: Path) -> None:
    # G = t on both files, except file 1's last step (t = 2.0) has a bad value.
    # File 1 has 5 steps and file 2 only 3, so the old loop over the file-2
    # step range never reached file-1 steps 4 and 5.
    a = _write(
        tmp_path / "a.exo",
        times=(0.0, 0.5, 1.0, 1.5, 2.0),
        gvals=(0.0, 0.5, 1.0, 1.5, 9.0),
        nodal=(),
    )
    b = _write(tmp_path / "b.exo", times=(0.0, 1.0, 2.0), gvals=(0.0, 1.0, 2.0), nodal=())
    opts = DiffOptions(time_selection=TimeSelection(interpolating=True))
    result = diff(a, b, opts)
    (vd,) = [v for v in result.variable_diffs if v.name == "G"]
    assert vd.exceeded
    assert (vd.value1, vd.value2) == (9.0, 2.0)

    good = _write(
        tmp_path / "c.exo",
        times=(0.0, 0.5, 1.0, 1.5, 2.0),
        gvals=(0.0, 0.5, 1.0, 1.5, 2.0),
        nodal=(),
    )
    assert diff(good, b, opts).same


def test_interpolation_snaps_within_time_tolerance(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.exo", times=(0.0, 1.0 + 1e-9), nodal=())
    b = _write(tmp_path / "b.exo", times=(0.0, 1.0), nodal=())
    opts = DiffOptions(
        time_selection=TimeSelection(interpolating=True),
        final_time_tolerance=Tolerance(ToleranceMode.RELATIVE, 1e-6, 0.0),
    )
    # t = 1 + 1e-9 is within the time tolerance of file 2's t = 1: it is
    # matched to that step instead of falling outside file 2's range.
    result = diff(a, b, opts)
    assert result.same, (result.differences, result.variable_diffs)


def test_empty_step_selection_is_a_difference(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.exo")
    b = _write(tmp_path / "b.exo")
    result = diff(a, b, DiffOptions(time_selection=TimeSelection(start=5)))
    assert not result.same
    assert result.differences == [
        "time step selection (start 5, stop -1) resulted in no time steps being compared"
    ]
    assert _cli("--start", 5, a, b)[0] == DIFFERENT


@pytest.mark.parametrize(
    ("offset", "flag"), [(STEP_OFFSET_AUTOMATIC, "-TA"), (STEP_OFFSET_MATCH, "-TM")]
)
def test_automatic_step_offsets(tmp_path: Path, offset: int, flag: str) -> None:
    # File 1 has two extra leading steps; file 2 matches its last three.
    a = _write(tmp_path / "a.exo", times=(0.0, 1.0, 2.0, 3.0, 4.0), nodal=(), globals_=())
    b = _write(tmp_path / "b.exo", times=(2.0, 3.0, 4.0), nodal=(), globals_=())
    assert not diff(a, b).same
    assert diff(a, b, DiffOptions(time_selection=TimeSelection(time_step_offset=offset))).same
    assert _cli(flag, a, b)[0] == SAME


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------


def test_variable_in_one_file_is_a_difference(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.exo", nodal=("T", "U"))
    b = _write(tmp_path / "b.exo", nodal=("T",))
    result = diff(a, b)
    assert not result.errors
    assert result.differences == ["node variable 'U' is in the first file but not the second"]
    assert _cli(a, b)[0] == DIFFERENT


def test_count_mismatch_is_a_difference_and_stops(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.exo")
    b = tmp_path / "b.exo"
    with ExodusWriter.create(b) as w:
        w.initialize("t", 2, node_count=4, element_count=1, element_blocks=1)
        w.write_coordinates(COORDS[:4])
        w.define_element_block(10, "quad4", [[1, 2, 4, 3]])
        w.define_node_variables(["T"])
        w.write_time(0.0)
        w.write_node_values("T", np.zeros(4))
    result = diff(a, b)
    assert not result.errors
    assert not result.variable_diffs  # nothing else was compared
    assert "node count differs: 6 != 4" in result.differences
    status, text = _cli(a, b)
    assert status == DIFFERENT
    assert "inf" not in text


def test_truth_table_mismatch_is_skipped(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.exo", element=("E",), truth=[[1], [1]])
    b = _write(tmp_path / "b.exo", element=("E",), truth=[[1], [0]])
    result = diff(a, b)
    assert result.same, (result.errors, result.differences)
    assert any("truth tables differ" in w for w in result.warnings)
    assert _cli(a, b)[0] == SAME


def test_requested_variable_missing_from_both_files(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.exo")
    b = _write(tmp_path / "b.exo")
    cmd = tmp_path / "cmd"
    cmd.write_text("NODAL VARIABLES\n\tNOPE\n")
    status, text = _cli("-f", cmd, a, b)
    assert status == DIFFERENT
    assert "NOPE" in text


# ---------------------------------------------------------------------------
# Command-file parsing
# ---------------------------------------------------------------------------


def test_bare_per_variable_value(tmp_path: Path) -> None:
    cmd = tmp_path / "cmd"
    cmd.write_text("NODAL VARIABLES absolute 1e-9\n\tT 5\n\tU 0.5 floor 1e-3\n")
    opts = read_command_file(cmd).options
    nodal = opts.category_variable_tolerances[Entity.NODE]
    assert nodal["T"] == Tolerance(ToleranceMode.ABSOLUTE, 5.0, 0.0)
    assert nodal["U"] == Tolerance(ToleranceMode.ABSOLUTE, 0.5, 1e-3)

    a = _write(tmp_path / "a.exo", temp=np.zeros(6))
    b = _write(tmp_path / "b.exo", temp=np.ones(6))
    cmd.write_text("NODAL VARIABLES absolute 1e-9\n\tT 5\n")
    assert _cli("-f", cmd, a, b)[0] == SAME


def test_final_time_tolerance_is_separate(tmp_path: Path) -> None:
    cmd = tmp_path / "cmd"
    cmd.write_text("TIME STEPS absolute 1e-9\nFINAL TIME TOLERANCE 0.5\n")
    opts = read_command_file(cmd).options
    assert opts.time_tolerance == Tolerance(ToleranceMode.ABSOLUTE, 1e-9, 0.0)
    assert opts.final_time_tolerance == Tolerance(ToleranceMode.RELATIVE, 0.5, 0.0)


@pytest.mark.parametrize(
    ("spec", "steps"), [("2", [2]), ("1,4", [1, 4]), ("2-5", [2, 3, 4, 5]), ("1-3,7", [1, 2, 3, 7])]
)
def test_step_list_ranges(spec: str, steps: list[int]) -> None:
    assert parse_step_list(spec) == steps


def test_step_list_rejects_bad_ranges() -> None:
    for bad in ("0", "5-2", "x"):
        with pytest.raises(ValueError):
            parse_step_list(bad)


def test_exclude_times_ranges(tmp_path: Path) -> None:
    cmd = tmp_path / "cmd"
    cmd.write_text("EXCLUDE TIMES 1-3,5\n")
    opts = read_command_file(cmd).options
    assert opts.time_selection is not None
    assert opts.time_selection.exclude_steps == frozenset({1, 2, 3, 5})

    cmd.write_text("EXCLUDE TIMES 3-1\n")
    with pytest.raises(CommandFileError):
        read_command_file(cmd)

    a = _write(tmp_path / "a.exo", times=(0.0, 1.0, 2.0))
    b = _write(tmp_path / "b.exo", times=(0.0, 1.0, 2.0), temp=np.full(6, 99.0))
    assert _cli("--exclude-steps", "1-3", a, b)[0] == DIFFERENT  # all excluded
    assert _cli("--exclude-steps", "1-2", a, b)[0] == DIFFERENT


def test_step_offset_directives(tmp_path: Path) -> None:
    cmd = tmp_path / "cmd"
    for word, offset in (("AUTOMATIC", STEP_OFFSET_AUTOMATIC), ("MATCH", STEP_OFFSET_MATCH)):
        cmd.write_text(f"STEP OFFSET {word}\n")
        result = read_command_file(cmd)
        assert result.options.time_selection is not None
        assert result.options.time_selection.time_step_offset == offset
        assert not result.warnings


def test_nodeset_sideset_match_do_not_enable_coordinate_matching(tmp_path: Path) -> None:
    cmd = tmp_path / "cmd"
    for directive in ("NODESET MATCH", "SIDESET MATCH"):
        cmd.write_text(directive + "\n")
        assert read_command_file(cmd).options.coordinate_matching is False


def test_global_variables_use_scalar_floor(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.exo", times=(0.0,), gvals=(1.0,), nodal=())
    b = _write(tmp_path / "b.exo", times=(0.0,), gvals=(-1.0,), nodal=())
    floor = DiffOptions(default_tolerance=Tolerance(ToleranceMode.ABSOLUTE, 0.1, 1.0))
    assert diff(a, b, floor).same
