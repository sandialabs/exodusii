# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Tests for the exodiff command-file (control-file) reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import exodusii
from exodusii.api.command_file import CommandFileError
from exodusii.api.command_file import read_command_file
from exodusii.api.diff import diff
from exodusii.api.writer import ExodusWriter
from exodusii.core.entities import Entity
from exodusii.core.tolerance import ToleranceMode


def _write(path: Path, name: str, value: float, extra: float = 0.0) -> None:
    """Write a 1-block, 1-step file with two nodal vars DISPLX/DISPLY."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    with ExodusWriter.create(path) as writer:
        writer.initialize(name, 2, 4, 1, element_blocks=1)
        writer.write_coordinates(coords)
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_node_variables(["DISPLX", "DISPLY"])
        writer.write_time(0.0)
        writer.write_node_values("DISPLX", np.full(4, value))
        writer.write_node_values("DISPLY", np.full(4, value + extra))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_default_tolerance_directive(tmp_path: Path) -> None:
    cmd = tmp_path / "cmds"
    cmd.write_text("DEFAULT TOLERANCE absolute 1e-4 floor 1e-9\n")
    result = read_command_file(cmd)
    tol = result.options.default_tolerance
    assert tol.mode is ToleranceMode.ABSOLUTE
    assert tol.value == pytest.approx(1e-4)
    assert tol.floor == pytest.approx(1e-9)


def test_abbreviated_and_case_insensitive_keywords(tmp_path: Path) -> None:
    cmd = tmp_path / "cmds"
    cmd.write_text("def tol rel 1e-3\nCOORD abs 1e-7\n")
    opts = read_command_file(cmd).options
    assert opts.default_tolerance.mode is ToleranceMode.RELATIVE
    assert opts.default_tolerance.value == pytest.approx(1e-3)
    assert opts.coordinate_tolerance.mode is ToleranceMode.ABSOLUTE
    assert opts.coordinate_tolerance.value == pytest.approx(1e-7)


def test_comments_and_blank_lines_ignored(tmp_path: Path) -> None:
    cmd = tmp_path / "cmds"
    cmd.write_text("# a comment\n\n   \nDEFAULT TOLERANCE relative 1e-5\n# trailing\n")
    opts = read_command_file(cmd).options
    assert opts.default_tolerance.value == pytest.approx(1e-5)


def test_per_category_default_and_per_variable_override(tmp_path: Path) -> None:
    cmd = tmp_path / "cmds"
    cmd.write_text("NODAL VARIABLES absolute 1e-7\n\tDISPLX relative 1e-9\n\tDISPLY\n")
    opts = read_command_file(cmd).options
    assert opts.nodal_tolerance is not None
    assert opts.nodal_tolerance.mode is ToleranceMode.ABSOLUTE
    assert opts.nodal_tolerance.value == pytest.approx(1e-7)
    # DISPLX has an override; DISPLY inherits the category default.
    assert opts.variable_tolerances["DISPLX"].mode is ToleranceMode.RELATIVE
    assert opts.variable_tolerances["DISPLX"].value == pytest.approx(1e-9)
    assert opts.variable_tolerances["DISPLY"].mode is ToleranceMode.ABSOLUTE
    # Only-listed names become the include set for that category.
    assert opts.include_variables[Entity.NODE] == frozenset({"DISPLX", "DISPLY"})
    assert Entity.NODE not in opts.all_categories


def test_all_flag_makes_category_unrestricted(tmp_path: Path) -> None:
    cmd = tmp_path / "cmds"
    cmd.write_text("GLOBAL VARIABLES (all) relative 1e-4\n\tFOO absolute 1e-8\n")
    opts = read_command_file(cmd).options
    assert Entity.GLOBAL in opts.all_categories
    assert opts.variable_tolerances["FOO"].value == pytest.approx(1e-8)


def test_exclude_via_bang_name(tmp_path: Path) -> None:
    cmd = tmp_path / "cmds"
    cmd.write_text("NODAL VARIABLES\n\tDISPLX\n\t!VELZ\n")
    opts = read_command_file(cmd).options
    assert "VELZ" in opts.exclude
    assert opts.include_variables[Entity.NODE] == frozenset({"DISPLX"})


def test_ignore_case_and_case_sensitive(tmp_path: Path) -> None:
    cmd = tmp_path / "cmds"
    cmd.write_text("CASE SENSITIVE\n")
    assert read_command_file(cmd).options.ignore_case is False
    cmd.write_text("IGNORE CASE\n")
    assert read_command_file(cmd).options.ignore_case is True


def test_matching_switches_enable_coordinate_matching(tmp_path: Path) -> None:
    for directive in ("APPLY MATCHING", "NODESET MATCH", "SIDESET MATCH"):
        cmd = tmp_path / "cmds"
        cmd.write_text(directive + "\n")
        assert read_command_file(cmd).options.coordinate_matching is True


def test_interpolate_and_exclude_times(tmp_path: Path) -> None:
    cmd = tmp_path / "cmds"
    cmd.write_text("INTERPOLATE\nEXCLUDE TIMES 2 4 6\n")
    opts = read_command_file(cmd).options
    assert opts.time_selection is not None
    assert opts.time_selection.interpolating is True
    assert opts.time_selection.exclude_steps == frozenset({2, 4, 6})


def test_time_steps_tolerance(tmp_path: Path) -> None:
    cmd = tmp_path / "cmds"
    cmd.write_text("TIME STEPS absolute 1e-9\n")
    opts = read_command_file(cmd).options
    assert opts.time_tolerance.mode is ToleranceMode.ABSOLUTE
    assert opts.time_tolerance.value == pytest.approx(1e-9)


def test_ignore_mode_needs_no_value(tmp_path: Path) -> None:
    cmd = tmp_path / "cmds"
    cmd.write_text("DEFAULT TOLERANCE ignore\n")
    opts = read_command_file(cmd).options
    assert opts.default_tolerance.mode is ToleranceMode.IGNORE


def test_inert_directives_warn_but_parse(tmp_path: Path) -> None:
    cmd = tmp_path / "cmds"
    cmd.write_text("PEDANTIC\nCALCULATE NORMS\nIGNORE DUPS\n")
    result = read_command_file(cmd)
    assert len(result.warnings) == 3
    assert all("no effect" in w for w in result.warnings)


def test_unrecognized_directive_raises(tmp_path: Path) -> None:
    cmd = tmp_path / "cmds"
    cmd.write_text("FLUX CAPACITOR on\n")
    with pytest.raises(CommandFileError):
        read_command_file(cmd)


def test_tolerance_mode_without_value_raises(tmp_path: Path) -> None:
    cmd = tmp_path / "cmds"
    cmd.write_text("DEFAULT TOLERANCE relative floor 1e-9\n")
    with pytest.raises(CommandFileError):
        read_command_file(cmd)


def test_exported_from_package() -> None:
    assert exodusii.read_command_file is read_command_file


# ---------------------------------------------------------------------------
# End-to-end effect on diff()
# ---------------------------------------------------------------------------


def test_include_list_restricts_compared_variables(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write(a, "m", 1.0)
    _write(b, "m", 1.0, extra=5.0)  # DISPLY differs, DISPLX identical

    # Command file compares only DISPLX (nodal include list) -> files "same".
    cmd = tmp_path / "cmds"
    cmd.write_text("NODAL VARIABLES\n\tDISPLX\n")
    opts = read_command_file(cmd).options
    assert diff(a, b, opts).same

    # Without the restriction, DISPLY difference is caught.
    assert not diff(a, b).same


def test_exclude_hides_difference(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write(a, "m", 1.0)
    _write(b, "m", 1.0, extra=5.0)

    cmd = tmp_path / "cmds"
    cmd.write_text("NODAL VARIABLES (all)\n\t!DISPLY\n")
    opts = read_command_file(cmd).options
    assert diff(a, b, opts).same


def test_loose_default_tolerance_makes_same(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write(a, "m", 1.0)
    _write(b, "m", 1.0, extra=1e-9)  # tiny DISPLY difference

    cmd = tmp_path / "cmds"
    cmd.write_text("DEFAULT TOLERANCE absolute 1e-3\n")
    opts = read_command_file(cmd).options
    assert diff(a, b, opts).same
