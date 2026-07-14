# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from io import StringIO
from pathlib import Path

import numpy as np
import pytest

import exodusii
from exodusii.api.file import ExodusFile
from exodusii.api.lineout import Lineout
from exodusii.api.query import QueryResult
from exodusii.api.query import print_query
from exodusii.api.query import query
from exodusii.api.writer import ExodusWriter
from exodusii.core.errors import ExodusInvalidEntityError


def test_query_global_all_times(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo:
        result = query(exo, "g/TM_STEP")

    assert isinstance(result, QueryResult)
    assert result.names == ("TIME", "TM_STEP")
    assert result.metadata["entity"] == "global"
    assert np.allclose(result.data["TIME"], [0.0, 1.0])
    assert np.allclose(result.data["TM_STEP"], [0.0, 1.0])


def test_query_global_single_time(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo:
        result = query(exo, "g/TM_STEP", time="last")

    assert result.names == ("TIME", "TM_STEP")
    assert len(result.data) == 1
    assert result.data["TIME"][0] == pytest.approx(1.0)
    assert result.data["TM_STEP"][0] == pytest.approx(1.0)


def test_query_node_single_time(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo:
        result = query(exo, "n/TEMP", time="last")

    assert result.names == ("TEMP",)
    assert result.metadata["entity"] == "node"
    assert result.metadata["time"] == 1.0
    assert np.allclose(result.data["TEMP"], [11.0, 21.0, 31.0, 41.0])


def test_query_node_object_index(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo:
        result = query(exo, "n/TEMP", time="last", object_index=True)

    assert result.names == ("index", "TEMP")
    assert np.allclose(result.data["index"], [1.0, 2.0, 3.0, 4.0])


def test_query_node_coordinates_and_displacements(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo:
        result = query(exo, "n/coordinates", "n/displacements", "n/TEMP", time="last")

    assert result.names == ("COORDX", "COORDY", "DISPLX", "DISPLY", "TEMP")
    assert np.allclose(result.data["COORDX"], [0.0, 1.0, 1.0, 0.0])
    assert np.allclose(result.data["DISPLX"], [0.1, 0.1, 0.1, 0.1])


def test_query_node_lineout(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo:
        result = query(
            exo, "n/coordinates", "n/TEMP", time="last", lineout=Lineout(x="x", y=0.0, tol=1.0e-12)
        )

    assert result.names == ("COORDX", "TEMP")
    assert np.allclose(result.data["COORDX"], [0.0, 1.0])
    assert np.allclose(result.data["TEMP"], [11.0, 21.0])


def test_query_element_single_time(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo:
        result = query(exo, "e/ENERGY", time="last")

    assert result.names == ("ENERGY",)
    assert result.metadata["entity"] == "element"
    assert np.allclose(result.data["ENERGY"], [1.5])


def test_query_rejects_empty_variables(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo, pytest.raises(ValueError, match="at least one variable"):
        query(exo)


def test_query_rejects_mixed_entities(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo, pytest.raises(ValueError, match="same entity"):
        query(exo, "n/TEMP", "e/ENERGY")


def test_query_rejects_unimplemented_entity(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with (
        ExodusFile.open(path) as exo,
        pytest.raises(ExodusInvalidEntityError, match="not implemented"),
    ):
        query(exo, "ss/FOO")


def test_print_query(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)
    stream = StringIO()

    with ExodusFile.open(path) as exo:
        print_query(exo, "g/TM_STEP", time="last", file=stream)

    text = stream.getvalue()
    assert "TIME" in text
    assert "TM_STEP" in text
    assert "1.0000000000000000e+00" in text


def test_top_level_query_exports() -> None:
    assert exodusii.query is query
    assert exodusii.print_query is print_query


def _write_query_file(path: Path) -> None:
    with ExodusWriter.create(path) as writer:
        writer.initialize("query", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(_unit_square_quad())
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])

        writer.define_global_variables(["TM_STEP"])
        writer.define_node_variables(["DISPLX", "DISPLY", "TEMP"])
        writer.define_element_variables(["ENERGY"])

        writer.write_time(0.0)
        writer.write_global_values([0.0])
        writer.write_node_values("DISPLX", [0.0, 0.0, 0.0, 0.0])
        writer.write_node_values("DISPLY", [0.0, 0.0, 0.0, 0.0])
        writer.write_node_values("TEMP", [10.0, 20.0, 30.0, 40.0])
        writer.write_element_values("ENERGY", [0.5], block_id=10)

        writer.write_time(1.0)
        writer.write_global_values([1.0])
        writer.write_node_values("DISPLX", [0.1, 0.1, 0.1, 0.1])
        writer.write_node_values("DISPLY", [0.0, 0.0, 0.0, 0.0])
        writer.write_node_values("TEMP", [11.0, 21.0, 31.0, 41.0])
        writer.write_element_values("ENERGY", [1.5], block_id=10)


def _unit_square_quad() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
