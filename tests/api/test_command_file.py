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


# ---------------------------------------------------------------------------
# YAML format
# ---------------------------------------------------------------------------

from exodusii.api.command_file import ExodiffCommandFileReader  # noqa: E402
from exodusii.api.command_file import YamlCommandFileReader  # noqa: E402
from exodusii.api.command_file import command_file_reader  # noqa: E402
from exodusii.api.diff import DiffOptions  # noqa: E402
from exodusii.api.diff import TimeSelection  # noqa: E402
from exodusii.core.tolerance import Tolerance  # noqa: E402


def test_yaml_reader_basic(tmp_path: Path) -> None:
    cmd = tmp_path / "opts.yaml"
    cmd.write_text(
        "tolerances:\n"
        "  default: {mode: absolute, value: 1.0e-4, floor: 1.0e-9}\n"
        "  coordinate: {mode: absolute, value: 1.0e-8}\n"
        "  categories:\n"
        "    nodal: {mode: absolute, value: 1.0e-7}\n"
        "  variables:\n"
        "    DISPLX: {mode: relative, value: 1.0e-9}\n"
        "variables:\n"
        "  ignore_case: false\n"
        "  exclude: [VELZ]\n"
        "  include:\n"
        "    nodal: [DISPLX, DISPLY]\n"
        "    global: all\n"
        "mesh_matching: {enabled: true, tolerance: 1.0e-9}\n"
        "time: {start: 2, stop: 8, exclude_steps: [3], interpolate: true}\n"
        "report: {show_all: true}\n"
    )
    opts = read_command_file(cmd).options
    assert opts.default_tolerance.mode is ToleranceMode.ABSOLUTE
    assert opts.default_tolerance.value == pytest.approx(1e-4)
    assert opts.default_tolerance.floor == pytest.approx(1e-9)
    assert opts.coordinate_tolerance.value == pytest.approx(1e-8)
    assert opts.nodal_tolerance is not None
    assert opts.nodal_tolerance.mode is ToleranceMode.ABSOLUTE
    assert opts.variable_tolerances["DISPLX"].value == pytest.approx(1e-9)
    assert opts.ignore_case is False
    assert opts.exclude == frozenset({"VELZ"})
    assert opts.include_variables[Entity.NODE] == frozenset({"DISPLX", "DISPLY"})
    assert Entity.GLOBAL in opts.all_categories
    assert opts.coordinate_matching is True
    assert opts.matching_tolerance == pytest.approx(1e-9)
    assert opts.time_selection is not None
    assert opts.time_selection.start == 2
    assert opts.time_selection.exclude_steps == frozenset({3})
    assert opts.time_selection.interpolating is True
    assert opts.show_all is True


def test_yaml_tolerance_shorthand(tmp_path: Path) -> None:
    cmd = tmp_path / "opts.yaml"
    cmd.write_text("tolerances:\n  default: absolute 1e-3\n  time: 1e-9\n")
    opts = read_command_file(cmd).options
    assert opts.default_tolerance.mode is ToleranceMode.ABSOLUTE
    assert opts.default_tolerance.value == pytest.approx(1e-3)
    # bare number -> relative
    assert opts.time_tolerance.mode is ToleranceMode.RELATIVE
    assert opts.time_tolerance.value == pytest.approx(1e-9)


def test_yaml_bare_number_tolerance(tmp_path: Path) -> None:
    cmd = tmp_path / "opts.yaml"
    cmd.write_text("tolerances:\n  default: 1.0e-6\n")
    opts = read_command_file(cmd).options
    assert opts.default_tolerance.mode is ToleranceMode.RELATIVE
    assert opts.default_tolerance.value == pytest.approx(1e-6)


def test_yaml_empty_document(tmp_path: Path) -> None:
    cmd = tmp_path / "opts.yaml"
    cmd.write_text("# nothing here\n")
    opts = read_command_file(cmd).options
    assert opts == DiffOptions()  # all defaults


def test_yaml_bad_category_raises(tmp_path: Path) -> None:
    cmd = tmp_path / "opts.yaml"
    cmd.write_text("tolerances:\n  categories:\n    bogus: 1e-6\n")
    with pytest.raises(CommandFileError):
        read_command_file(cmd)


def test_yaml_top_level_not_mapping_raises(tmp_path: Path) -> None:
    cmd = tmp_path / "opts.yaml"
    cmd.write_text("- 1\n- 2\n")
    with pytest.raises(CommandFileError):
        read_command_file(cmd)


# ---------------------------------------------------------------------------
# Emit + round-trip
# ---------------------------------------------------------------------------


def _rich_options() -> DiffOptions:
    return DiffOptions(
        default_tolerance=Tolerance(ToleranceMode.RELATIVE, 1e-5, 1e-12),
        coordinate_tolerance=Tolerance(ToleranceMode.ABSOLUTE, 1e-8),
        time_tolerance=Tolerance(ToleranceMode.RELATIVE, 1e-6, 1e-15),
        nodal_tolerance=Tolerance(ToleranceMode.ABSOLUTE, 1e-7),
        attribute_tolerance=Tolerance(ToleranceMode.ABSOLUTE, 1e-10),
        variable_tolerances={"DISPLX": Tolerance(ToleranceMode.RELATIVE, 1e-9)},
        exclude=frozenset({"VELZ"}),
        include_variables={Entity.NODE: frozenset({"DISPLX", "DISPLY"})},
        all_categories=frozenset({Entity.GLOBAL}),
        ignore_case=False,
        compare_coordinates=False,
        compare_attributes=False,
        show_all=True,
        coordinate_matching=True,
        matching_tolerance=1e-9,
        require_unique_mapping=False,
        time_selection=TimeSelection(
            start=2,
            stop=10,
            increment=2,
            time_step_offset=1,
            exclude_steps=frozenset({3, 4}),
            time_value_scale=2.0,
            time_value_offset=0.5,
            interpolating=True,
        ),
    )


def test_to_yaml_roundtrip_preserves_everything(tmp_path: Path) -> None:
    o = _rich_options()
    path = tmp_path / "emitted.yaml"
    text = o.to_yaml(path)
    assert path.read_text() == text

    o2 = read_command_file(path).options
    assert o2.default_tolerance == o.default_tolerance
    assert o2.coordinate_tolerance == o.coordinate_tolerance
    assert o2.time_tolerance == o.time_tolerance
    assert o2.nodal_tolerance == o.nodal_tolerance
    assert o2.attribute_tolerance == o.attribute_tolerance
    assert o2.variable_tolerances == o.variable_tolerances
    assert o2.exclude == o.exclude
    assert o2.include_variables == o.include_variables
    assert o2.all_categories == o.all_categories
    assert o2.ignore_case == o.ignore_case
    assert o2.compare_coordinates == o.compare_coordinates
    assert o2.compare_attributes == o.compare_attributes
    assert o2.show_all == o.show_all
    assert o2.coordinate_matching == o.coordinate_matching
    assert o2.matching_tolerance == o.matching_tolerance
    assert o2.require_unique_mapping == o.require_unique_mapping
    assert o2.time_selection == o.time_selection


def test_default_options_roundtrip(tmp_path: Path) -> None:
    o = DiffOptions()
    path = tmp_path / "d.yaml"
    o.to_yaml(path)
    o2 = read_command_file(path).options
    assert o2 == o


def test_emit_is_valid_yaml_with_header(tmp_path: Path) -> None:
    text = DiffOptions().to_yaml()
    assert text.startswith("#")
    assert "version: 1" in text


# ---------------------------------------------------------------------------
# Factory sniffing
# ---------------------------------------------------------------------------


def test_factory_selects_yaml_by_suffix(tmp_path: Path) -> None:
    cmd = tmp_path / "opts.yaml"
    cmd.write_text("tolerances: {default: 1e-6}\n")
    assert isinstance(command_file_reader(cmd), YamlCommandFileReader)


def test_factory_selects_yaml_by_content(tmp_path: Path) -> None:
    cmd = tmp_path / "opts.cfg"  # non-yaml suffix
    cmd.write_text("tolerances:\n  default: {mode: relative, value: 1e-6}\n")
    assert isinstance(command_file_reader(cmd), YamlCommandFileReader)


def test_factory_falls_back_to_exodiff(tmp_path: Path) -> None:
    cmd = tmp_path / "cmds"
    cmd.write_text("DEFAULT TOLERANCE absolute 1e-4\n")
    reader = command_file_reader(cmd)
    assert isinstance(reader, ExodiffCommandFileReader)
    assert reader.read().options.default_tolerance.mode is ToleranceMode.ABSOLUTE


def test_exodiff_still_parses_via_factory(tmp_path: Path) -> None:
    cmd = tmp_path / "cmds"
    cmd.write_text("NODAL VARIABLES\n\tDISPLX\n")
    opts = read_command_file(cmd).options
    assert opts.include_variables[Entity.NODE] == frozenset({"DISPLX"})
