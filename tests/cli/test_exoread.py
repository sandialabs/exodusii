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


def test_exoread_rejects_bad_args(tmp_path: Path) -> None:
    path = tmp_path / "cli.exo"
    _write_cli_file(path)

    with pytest.raises(SystemExit):
        main(["-g", "TM_STEP", "-n", "TEMP", str(path)])


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
