# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Tests for the ``exodiff`` command-line interface."""

import json
from io import StringIO
from pathlib import Path

import numpy as np

from exodusii.api.writer import ExodusWriter
from exodusii.cli.exodiff import build_parser
from exodusii.cli.exodiff import main

_SAME = 0
_ERROR = 1
_DIFFERENT = 2


def _coords() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)


def _write(path: Path, *, temp_offset: float = 0.0) -> None:
    with ExodusWriter.create(path) as writer:
        writer.initialize("cli", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(_coords())
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_global_variables(["TM_STEP"])
        writer.define_node_variables(["TEMP"])
        writer.write_time(0.0)
        writer.write_global_values([0.0])
        writer.write_node_values("TEMP", [10.0, 20.0, 30.0, 40.0])
        writer.write_time(1.0)
        writer.write_global_values([1.0])
        writer.write_node_values("TEMP", np.array([11.0, 21.0, 31.0, 41.0]) + temp_offset)


# ---------------------------------------------------------------------------
# Text mode
# ---------------------------------------------------------------------------


def test_identical_files_return_zero(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write(a)
    _write(b)
    stream = StringIO()

    status = main([str(a), str(b)], file=stream)

    assert status == _SAME
    assert "Files are the same" in stream.getvalue()


def test_different_files_return_two(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write(a)
    _write(b, temp_offset=5.0)
    stream = StringIO()

    status = main([str(a), str(b)], file=stream)

    assert status == _DIFFERENT
    text = stream.getvalue()
    assert "Files are different" in text
    assert "TEMP" in text


def test_missing_file_returns_error(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    _write(a)
    missing = tmp_path / "does_not_exist.exo"
    stream = StringIO()

    status = main([str(a), str(missing)], file=stream)

    assert status == _ERROR


def test_tolerance_flag_absorbs_difference(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write(a)
    _write(b, temp_offset=1.0e-4)
    stream = StringIO()

    # Loose absolute tolerance -> same.
    status = main(["--absolute", "-t", "1.0", str(a), str(b)], file=stream)
    assert status == _SAME


def test_exclude_flag(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write(a)
    _write(b, temp_offset=5.0)
    stream = StringIO()

    status = main(["-x", "TEMP", str(a), str(b)], file=stream)
    assert status == _SAME


def test_quiet_suppresses_variable_listing(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write(a)
    _write(b, temp_offset=5.0)
    stream = StringIO()

    status = main(["-q", str(a), str(b)], file=stream)
    assert status == _DIFFERENT
    text = stream.getvalue()
    # The per-variable line is suppressed, but the verdict remains.
    assert "TEMP:" not in text
    assert "Files are different" in text


# ---------------------------------------------------------------------------
# JSON mode
# ---------------------------------------------------------------------------


def test_json_output_same(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write(a)
    _write(b)
    stream = StringIO()

    status = main(["--format", "json", str(a), str(b)], file=stream)
    assert status == _SAME
    payload = json.loads(stream.getvalue())
    assert payload["ok"] is True
    assert payload["same"] is True
    assert payload["variable_diffs"] == []


def test_json_output_different(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write(a)
    _write(b, temp_offset=5.0)
    stream = StringIO()

    status = main(["--format", "json", str(a), str(b)], file=stream)
    assert status == _DIFFERENT
    payload = json.loads(stream.getvalue())
    assert payload["ok"] is True
    assert payload["same"] is False
    names = {vd["name"] for vd in payload["variable_diffs"]}
    assert "TEMP" in names


def test_json_output_error(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    _write(a)
    stream = StringIO()

    status = main(["--format", "json", str(a), str(tmp_path / "missing.exo")], file=stream)
    assert status == _ERROR
    payload = json.loads(stream.getvalue())
    assert payload["ok"] is False
    assert "error" in payload


def test_json_terse_is_single_line(tmp_path: Path) -> None:
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write(a)
    _write(b)
    stream = StringIO()

    main(["--format", "json", "--terse", str(a), str(b)], file=stream)
    output = stream.getvalue().strip()
    assert "\n" not in output


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parser_builds() -> None:
    parser = build_parser()
    args = parser.parse_args(["one.exo", "two.exo", "--absolute", "-t", "1e-3"])
    assert args.file1 == "one.exo"
    assert args.file2 == "two.exo"
    assert args.mode == "absolute"
    assert args.tolerance == 1e-3
