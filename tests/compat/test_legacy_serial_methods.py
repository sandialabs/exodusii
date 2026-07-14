# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from io import StringIO
from pathlib import Path

import numpy as np

import exodusii


def test_legacy_serial_misc_methods(tmp_path: Path) -> None:
    path = tmp_path / "serial.exo"
    _write_serial(path)

    with exodusii.File(path) as exo:
        assert exo.get_iid(np.asarray([10, 20]), 20) == 2
        assert exo.get_iid(np.asarray([10, 20]), 30) is None

        assert exo.get_coord_variable_names() == ["coordx", "coordy"]

        assert exo.is_global_variable("TM_STEP")
        assert exo.is_node_variable("TEMP")
        assert exo.is_element_variable("ENERGY")
        assert not exo.is_edge_variable("EDGE")
        assert not exo.is_face_variable("FACE")

        assert exo.get_variable_type("TM_STEP") == "g"
        assert exo.get_variable_type("TEMP") == "n"
        assert exo.get_variable_type("ENERGY") == "e"
        assert exo.get_variable_type("missing") is None

        assert exo.get_edge_block_ids().size == 0
        assert exo.get_face_block_ids().size == 0
        assert exo.get_edge_set_ids().size == 0
        assert exo.get_face_set_ids().size == 0
        assert exo.get_element_set_ids().size == 0
        assert exo.get_edge_variable_names().size == 0
        assert exo.get_face_variable_names().size == 0


def test_legacy_serial_get_and_print(tmp_path: Path) -> None:
    path = tmp_path / "serial.exo"
    _write_serial(path)

    with exodusii.File(path) as exo:
        table = exo.get("g/TM_STEP")
        assert table.dtype.names == ("TIME", "TM_STEP")
        assert np.allclose(table["TM_STEP"], [0.0, 1.0])

        table = exo.get("n/TEMP", index=-1)
        assert table.dtype.names == ("TEMP",)
        assert np.allclose(table["TEMP"], [11.0, 21.0, 31.0, 41.0])

        stream = StringIO()
        exo.print("g/TM_STEP", index=-1, file=stream)
        assert "TM_STEP" in stream.getvalue()

        stream = StringIO()
        exo.describe(file=stream)
        assert "Title: serial" in stream.getvalue()


def test_legacy_serial_info_and_qa_defaults(tmp_path: Path) -> None:
    path = tmp_path / "serial.exo"
    _write_serial(path)

    with exodusii.File(path) as exo:
        assert exo.get_info_records() is None
        assert exo.get_qa_records() is None


def _write_serial(path: Path) -> None:
    with exodusii.File(path, mode="w") as exo:
        exo.put_init("serial", 2, 4, 1, 1, 0, 0)
        exo.put_coords(_coords())
        exo.put_element_block(10, "quad", 1, 4)
        exo.put_element_conn(10, [[1, 2, 3, 4]])

        exo.put_global_variable_params(1)
        exo.put_global_variable_names(["TM_STEP"])
        exo.put_node_variable_params(1)
        exo.put_node_variable_names(["TEMP"])
        exo.put_element_variable_params(1)
        exo.put_element_variable_names(["ENERGY"])

        exo.put_time(1, 0.0)
        exo.put_global_variable_values(1, [0.0])
        exo.put_node_variable_values(1, "TEMP", [10.0, 20.0, 30.0, 40.0])
        exo.put_element_variable_values(1, 10, "ENERGY", [0.5])

        exo.put_time(2, 1.0)
        exo.put_global_variable_values(2, [1.0])
        exo.put_node_variable_values(2, "TEMP", [11.0, 21.0, 31.0, 41.0])
        exo.put_element_variable_values(2, 10, "ENERGY", [1.5])


def _coords() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
