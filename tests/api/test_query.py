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
from exodusii.core.entities import Entity
from exodusii.core.errors import ExodusInvalidEntityError
from exodusii.core.selectors import VariableSelector


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


def test_query_node_lineout_adds_coordinates(tmp_path: Path) -> None:
    # exodusii < 2 added coordinate columns automatically for lineouts.
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo:
        result = query(exo, "n/TEMP", time="last", lineout=Lineout(x="x", y=0.0, tol=1.0e-12))

    assert result.names == ("COORDX", "TEMP")
    assert np.allclose(result.data["COORDX"], [0.0, 1.0])
    assert np.allclose(result.data["TEMP"], [11.0, 21.0])


def test_query_node_displaced_lineout_adds_displacements(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo:
        result = query(exo, "n/TEMP", time="last", lineout=Lineout(x="X", y=0.0, tol=1.0e-12))

    assert result.names == ("LOCATIONX", "TEMP")
    assert np.allclose(result.data["LOCATIONX"], [0.1, 1.1])
    assert np.allclose(result.data["TEMP"], [11.0, 21.0])


def test_query_lineout_explicit_coordinates_not_duplicated(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo:
        result = query(
            exo, "n/TEMP", "n/coordinates", time="last", lineout=Lineout(x="x", y=0.0, tol=1.0e-12)
        )

    assert result.names == ("COORDX", "TEMP")


def test_query_element_lineout_adds_centers(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo:
        result = query(exo, "e/ENERGY", time="last", lineout=Lineout(x="x", y=0.5, tol=1.0e-12))

    assert result.names == ("COORDX", "ENERGY")
    assert np.allclose(result.data["COORDX"], [0.5])
    assert np.allclose(result.data["ENERGY"], [1.5])


def test_query_global_lineout_does_not_add_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo, pytest.raises(ValueError, match="coordinate columns"):
        query(exo, "g/TM_STEP", lineout=Lineout(x="x", y=0.0))


def test_legacy_get_lineout_adds_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    data = exodusii.File(path).get("n/TEMP", index=-1, lineout=Lineout(x="x", y=0.0, tol=1e-12))

    assert data.dtype.names == ("COORDX", "TEMP")
    assert np.allclose(data["n/TEMP"], [11.0, 21.0])


@pytest.mark.parametrize(
    ("selector", "stored"),
    [("g/tm_step", "TM_STEP"), ("n/temp", "TEMP"), ("n/Temp", "TEMP"), ("e/energy", "ENERGY")],
)
def test_query_labels_use_stored_name(tmp_path: Path, selector: str, stored: str) -> None:
    # exodusii < 2 labelled columns with the name stored in the file, not the
    # requested spelling; downstream callers key dictionaries on the labels.
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo:
        result = query(exo, selector, time="last")

    assert result.names[-1] == stored
    assert np.isfinite(result.data[stored]).all()


def test_print_query_labels_use_stored_name(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)
    stream = StringIO()

    with ExodusFile.open(path) as exo:
        print_query(exo, "n/coordinates", "n/temp", time="last", file=stream)

    assert stream.getvalue().splitlines()[0].split() == ["COORDX", "COORDY", "TEMP"]


def test_legacy_get_lowercase_request_keeps_alias(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    data = exodusii.File(path).get("n/temp", index=-1)

    assert data.dtype.names == ("TEMP",)
    assert np.allclose(data["n/temp"], data["TEMP"])


def test_query_element_single_time(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo:
        result = query(exo, "e/ENERGY", time="last")

    assert result.names == ("ENERGY",)
    assert result.metadata["entity"] == "element"
    assert np.allclose(result.data["ENERGY"], [1.5])


def test_query_element_coordinates_and_displacements(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo:
        result = query(exo, "coordinates", "displacements", "e/ENERGY", time="last")

    assert result.names == ("COORDX", "COORDY", "DISPLX", "DISPLY", "ENERGY")
    assert result.metadata["entity"] == "element"
    # Single quad on the unit square: centroid at (0.5, 0.5).
    assert np.allclose(result.data["COORDX"], [0.5])
    assert np.allclose(result.data["COORDY"], [0.5])
    # Uniform nodal displacement averages to the same value at the center.
    assert np.allclose(result.data["DISPLX"], [0.1])
    assert np.allclose(result.data["DISPLY"], [0.0])
    assert np.allclose(result.data["ENERGY"], [1.5])


def test_query_element_index_with_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo:
        result = query(exo, "e/ENERGY", "coordinates", time="last", object_index=True)

    assert result.names == ("index", "ENERGY", "COORDX", "COORDY")
    assert np.allclose(result.data["index"], [1.0])
    assert np.allclose(result.data["COORDX"], [0.5])


def test_query_special_only_defaults_to_node(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo:
        result = query(exo, "coordinates", time="last")

    assert result.names == ("COORDX", "COORDY")
    assert result.metadata["entity"] == "node"
    assert np.allclose(result.data["COORDX"], [0.0, 1.0, 1.0, 0.0])


@pytest.mark.parametrize(
    "selectors",
    [
        ("e/coordinates",),
        ("element/coordinates",),
        ("e/displacements", "e/coordinates"),
        (VariableSelector("coordinates", Entity.ELEMENT),),
    ],
)
def test_query_qualified_special_only_uses_its_entity(
    tmp_path: Path, selectors: tuple[str | VariableSelector, ...]
) -> None:
    # exodusii < 2: "e/coordinates" alone gave element centers (xmhd_vrod keys
    # its exact-solution table on them); v2 used to fall back to the nodes.
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo:
        result = query(exo, *selectors, time="last")

    assert result.metadata["entity"] == "element"
    assert np.allclose(result.data["COORDX"], [0.5])
    assert np.allclose(result.data["COORDY"], [0.5])


def test_query_qualified_special_only_node(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo:
        result = query(exo, "coordinates", "n/displacements", time="last")

    assert result.metadata["entity"] == "node"
    assert np.allclose(result.data["COORDX"], [0.0, 1.0, 1.0, 0.0])


def test_query_element_coordinates_alone_match_with_variable(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    with ExodusFile.open(path) as exo:
        alone = query(exo, "e/coordinates", time="last").data
        mixed = query(exo, "e/coordinates", "e/ENERGY", time="last").data

    assert np.array_equal(alone["COORDX"], mixed["COORDX"])
    assert np.array_equal(alone["COORDY"], mixed["COORDY"])


def test_legacy_get_element_coordinates_alone(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)

    data = exodusii.File(path).get("e/coordinates", index=-1)

    assert data.dtype.names == ("COORDX", "COORDY")
    assert np.allclose(data["COORDX"], [0.5])


def test_exoread_element_coordinates_alone(tmp_path: Path) -> None:
    # nevada exoread(f, type="element", name=["coordinates"]) -> exoread -e coordinates
    path = tmp_path / "query.exo"
    _write_query_file(path)
    stream = StringIO()

    exodusii.exoread(["-e", "coordinates", "-i", "-1", "--nolabels", str(path)], file=stream)

    assert np.allclose(np.loadtxt(StringIO(stream.getvalue())), [0.5, 0.5])


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


def test_print_query_legacy_number_format(tmp_path: Path) -> None:
    # exodusii < 2 used f"{value: 20.16e}" (space for sign).
    path = tmp_path / "query.exo"
    _write_query_file(path)
    stream = StringIO()

    with ExodusFile.open(path) as exo:
        print_query(exo, "g/TM_STEP", time="last", labels=False, file=stream)

    assert stream.getvalue() == f" {1.0: 20.16e} {1.0: 20.16e}\n"
    assert stream.getvalue() == "  1.0000000000000000e+00  1.0000000000000000e+00\n"


def test_print_query_accepts_path(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)
    out = tmp_path / "table.txt"

    with ExodusFile.open(path) as exo:
        print_query(exo, "n/coordinates", "n/TEMP", time="last", labels=False, file=str(out))
        print_query(exo, "n/TEMP", time="last", file=out.with_suffix(".lab"))

    table = np.loadtxt(out)
    assert table.shape == (4, 3)
    assert np.allclose(table[:, 2], [11.0, 21.0, 31.0, 41.0])
    assert out.with_suffix(".lab").read_text().splitlines()[0].split() == ["TEMP"]


def test_legacy_print_and_describe_accept_filename(tmp_path: Path) -> None:
    # alegra training tests call exodusii_file(...).print(..., file="name.txt").
    path = tmp_path / "query.exo"
    _write_query_file(path)
    exo = exodusii.exodusii_file(path, "r")

    exo.print("e/coordinates", "e/ENERGY", labels=False, index=-1, file=str(tmp_path / "e.txt"))
    exo.describe(file=str(tmp_path / "d.txt"))

    assert np.allclose(np.loadtxt(tmp_path / "e.txt"), [0.5, 0.5, 1.5])
    assert "Title: query" in (tmp_path / "d.txt").read_text()


def test_legacy_print_leaves_stream_open(tmp_path: Path) -> None:
    path = tmp_path / "query.exo"
    _write_query_file(path)
    stream = StringIO()

    exodusii.exodusii_file(path, "r").print("g/TM_STEP", file=stream)
    stream.write("still open")

    assert stream.getvalue().endswith("still open")


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
