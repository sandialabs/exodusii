# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from exodusii.api.writer import ExodusWriter
from exodusii.cli.exoread import main


def test_exoread_describe(tmp_path: Path) -> None:
    path = tmp_path / "cli.exo"
    _write_cli_file(path)
    stream = StringIO()

    status = main([str(path)], file=stream)

    assert status == 0
    text = stream.getvalue()
    assert "Title: cli" in text
    assert "Dimension: 2" in text
    assert "Num nodes   : 4" in text
    assert "Node vars: 1" in text
    assert "TEMP" in text


def test_exoread_describe_legacy_layout(tmp_path: Path) -> None:
    # Byte-for-byte the exodusii < 2 describe() layout; scripts parse it.
    path = tmp_path / "cli.exo"
    _write_cli_file(path)
    stream = StringIO()

    main([str(path)], file=stream)

    lines = stream.getvalue().splitlines()
    expected_prefix = [
        "Title: cli",
        "Storage type: d",
        "Num info strings: 0",
        "Dimension: 2",
        "Num nodes   : 4",
        "Num edges   : 0",
        "Num faces   : 0",
        "Num elements: 1",
        "Elem blocks: 1 Ids = 10",
        "Face blocks: 0",
        "Edge blocks: 0",
        "Node sets: 0",
        "Side sets: 0",
        "Edge sets: 0",
        "Face sets: 0",
        "Elem sets: 0",
        "Global vars: 1",
        "  0 TM_STEP",
        "Elem vars: 0",
        "Face vars: 0",
        "Edge vars: 0",
        "Node vars: 1",
        "  0 TEMP",
        "Node Set vars: 0",
        "Side Set vars: 0",
        "Edge Set vars: 0",
        "Face Set vars: 0",
        "Elem Set vars: 0",
        "Time steps: 2",
    ]
    assert lines[: len(expected_prefix)] == expected_prefix
    assert lines[len(expected_prefix) :] == ["  1 0.0", "  2 1.0"]


def test_legacy_describe_to_filename(tmp_path: Path) -> None:
    import exodusii

    path = tmp_path / "cli.exo"
    _write_cli_file(path)
    out = tmp_path / "describe.txt"

    exodusii.File(path).describe(file=str(out))

    assert out.read_text().splitlines()[2] == "Num info strings: 0"


def test_exoread_global_variable(tmp_path: Path) -> None:
    path = tmp_path / "cli.exo"
    _write_cli_file(path)
    stream = StringIO()

    status = main(["-g", "TM_STEP", str(path)], file=stream)

    assert status == 0
    text = stream.getvalue()
    assert "TIME" in text
    assert "TM_STEP" in text
    assert "0.0000000000000000e+00" in text
    assert "1.0000000000000000e+00" in text


def test_exoread_global_variable_last_index_no_labels(tmp_path: Path) -> None:
    path = tmp_path / "cli.exo"
    _write_cli_file(path)
    stream = StringIO()

    status = main(["-g", "TM_STEP", "--index", "-1", "--nolabels", str(path)], file=stream)

    assert status == 0
    text = stream.getvalue()
    assert "TIME" not in text
    assert "TM_STEP" not in text
    assert "1.0000000000000000e+00" in text


def test_exoread_node_variable_with_object_index(tmp_path: Path) -> None:
    path = tmp_path / "cli.exo"
    _write_cli_file(path)
    stream = StringIO()

    status = main(["-n", "TEMP", "--index", "-1", "--object-index", str(path)], file=stream)

    assert status == 0
    text = stream.getvalue()
    assert "index" in text
    assert "TEMP" in text
    assert "4.1000000000000000e+01" in text


def test_exoread_lineout(tmp_path: Path) -> None:
    path = tmp_path / "cli.exo"
    _write_cli_file(path)
    stream = StringIO()

    status = main(
        ["-n", "coordinates", "--index", "-1", "--lineout", "x/0.0/T1e-12", str(path)], file=stream
    )

    assert status == 0
    text = stream.getvalue()
    assert "COORDX" in text
    assert "COORDY" not in text


def test_exoread_lineout_adds_coordinates(tmp_path: Path) -> None:
    # Legacy callers (nevada exoread wrapper) pass only the variable.
    path = tmp_path / "cli.exo"
    _write_cli_file(path)
    stream = StringIO()

    status = main(
        ["-n", "TEMP", "--index", "-1", "--lineout", "x/0.0/T1e-12", str(path)], file=stream
    )

    assert status == 0
    header = stream.getvalue().splitlines()[0].split()
    assert header == ["COORDX", "TEMP"]


def test_exoread_rejects_bad_args(tmp_path: Path) -> None:
    path = tmp_path / "cli.exo"
    _write_cli_file(path)

    with pytest.raises(SystemExit):
        main(["-g", "TM_STEP", "-n", "TEMP", str(path)])


def _write_cycle_file(path: Path) -> None:
    with ExodusWriter.create(path) as writer:
        writer.initialize("cyc", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(_unit_square_quad())
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_global_variables(["NSTEPS"])
        writer.define_node_variables(["TEMP"])
        for step, cycle in enumerate((0, 50, 100)):
            writer.write_time(0.5 * step)
            writer.write_global_values([float(cycle)])
            writer.write_node_values("TEMP", np.full(4, float(step)))


def test_exoread_cycle_option(tmp_path: Path) -> None:
    # nevada's exoread wrapper forwards cycle= as -c (exodusii < 2 option).
    path = tmp_path / "cyc.exo"
    _write_cycle_file(path)
    stream = StringIO()

    status = main(["-n", "TEMP", "-c", "50", "--nolabels", str(path)], file=stream)

    assert status == 0
    assert np.allclose(np.loadtxt(StringIO(stream.getvalue())), 1.0)


def test_exoread_cycle_not_found(tmp_path: Path) -> None:
    path = tmp_path / "cyc.exo"
    _write_cycle_file(path)

    with pytest.raises(ValueError, match="Cycle 7 not found"):
        main(["-g", "NSTEPS", "-c", "7", str(path)], file=StringIO())


def test_legacy_get_cycle(tmp_path: Path) -> None:
    import exodusii

    path = tmp_path / "cyc.exo"
    _write_cycle_file(path)
    exo = exodusii.File(path)

    data = exo.get("n/TEMP", cycle=100)

    assert np.allclose(data["n/TEMP"], 2.0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        exo.get("n/TEMP", cycle=100, index=1)


def _write_cli_file(path: Path) -> None:
    with ExodusWriter.create(path) as writer:
        writer.initialize("cli", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(_unit_square_quad())
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_global_variables(["TM_STEP"])
        writer.define_node_variables(["TEMP"])

        writer.write_time(0.0)
        writer.write_global_values([0.0])
        writer.write_node_values("TEMP", [10.0, 20.0, 30.0, 40.0])

        writer.write_time(1.0)
        writer.write_global_values([1.0])
        writer.write_node_values("TEMP", [11.0, 21.0, 31.0, 41.0])


def _unit_square_quad() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
