# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pytest

from exodusii.api.file import ExodusFile
from exodusii.api.writer import ExodusWriter
from exodusii.core.errors import ExodusLookupError
from exodusii.core.errors import ExodusWriteError


def _unit_square_quad() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)


def test_writer_creates_minimal_initialized_file(tmp_path: Path) -> None:
    path = tmp_path / "minimal.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize("minimal", 2, 4, 1, element_blocks=1, node_sets=0, side_sets=0)
        writer.write_coordinates(_unit_square_quad())
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]], name="block_10")
        writer.write_time(0.0)

    with ExodusFile.open(path) as exo:
        assert exo.title == "minimal"
        assert exo.dimension == 2
        assert exo.node_count == 4
        assert exo.element_count == 1
        assert exo.element_block_ids().tolist() == [10]
        assert exo.element_block(10).name == "block_10"
        assert exo.element_block(10).element_type == "QUAD"
        assert np.allclose(exo.coordinates(), _unit_square_quad())
        assert np.allclose(exo.element_connectivity(10), [[1, 2, 3, 4]])
        assert np.allclose(exo.times(), [0.0])


def test_writer_supports_zero_based_connectivity(tmp_path: Path) -> None:
    path = tmp_path / "zero_based.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize("zero based", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(_unit_square_quad())
        writer.define_element_block(10, "quad4", [[0, 1, 2, 3]], zero_based=True)

    with ExodusFile.open(path) as exo:
        assert np.allclose(exo.element_connectivity(10), [[1, 2, 3, 4]])
        assert np.allclose(exo.element_connectivity(10, zero_based=True), [[0, 1, 2, 3]])


def test_writer_node_and_side_sets(tmp_path: Path) -> None:
    path = tmp_path / "sets.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize("sets", 2, 4, 1, element_blocks=1, node_sets=1, side_sets=1)
        writer.write_coordinates(_unit_square_quad())
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_node_set(100, [1, 4], distribution_factors=[1.0, 2.0], name="nodeset_100")
        writer.define_side_set(200, [1], [2], distribution_factors=[3.0, 4.0], name="sideset_200")

    with ExodusFile.open(path) as exo:
        assert exo.node_set_ids().tolist() == [100]
        assert exo.side_set_ids().tolist() == [200]

        node_set = exo.node_set(100)
        assert node_set.name == "nodeset_100"
        assert np.allclose(node_set.nodes, [1, 4])  # ty: ignore[invalid-argument-type]
        assert np.allclose(node_set.dist_facts, [1.0, 2.0])  # ty: ignore[invalid-argument-type]

        side_set = exo.side_set(200)
        assert side_set.name == "sideset_200"
        assert np.allclose(side_set.elems, [1])  # ty: ignore[invalid-argument-type]
        assert np.allclose(side_set.sides, [2])  # ty: ignore[invalid-argument-type]
        assert np.allclose(side_set.dist_facts, [3.0, 4.0])  # ty: ignore[invalid-argument-type]


def test_writer_result_variables(tmp_path: Path) -> None:
    path = tmp_path / "results.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize("results", 2, 4, 1, element_blocks=1)
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

    with ExodusFile.open(path) as exo:
        assert exo.variable_names("global") == ("TM_STEP",)
        assert exo.variable_names("node") == ("DISPLX", "DISPLY", "TEMP")
        assert exo.variable_names("element") == ("ENERGY",)

        assert np.allclose(exo.times(), [0.0, 1.0])
        assert np.allclose(exo.values("TM_STEP", on="global"), [0.0, 1.0])
        assert np.allclose(exo.values("TEMP", on="node", time="last"), [11.0, 21.0, 31.0, 41.0])
        assert np.allclose(exo.values("ENERGY", on="element", block=10), [[0.5], [1.5]])
        assert np.allclose(exo.values("ENERGY", on="element", block=10, time="last"), [1.5])


def test_writer_displaced_coordinates_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "displaced.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize("displaced", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(_unit_square_quad())
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_node_variables(["DISPLX", "DISPLY"])

        writer.write_time(0.0)
        writer.write_node_values("DISPLX", [0.25, 0.25, 0.25, 0.25])
        writer.write_node_values("DISPLY", [0.0, 0.0, 0.0, 0.0])

    with ExodusFile.open(path) as exo:
        assert exo.displacement_variable_names() == ("DISPLX", "DISPLY")
        assert np.allclose(
            exo.coordinates(time="last", displaced=True),
            _unit_square_quad() + np.asarray([0.25, 0.0]),
        )


def test_writer_initialize_rejects_double_initialization(tmp_path: Path) -> None:
    path = tmp_path / "bad.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize("bad", 2, 0, 0)

        with pytest.raises(ExodusWriteError, match="already initialized"):
            writer.initialize("bad again", 2, 0, 0)


def test_writer_initialize_rejects_bad_dimension(tmp_path: Path) -> None:
    path = tmp_path / "bad.exo"

    with (
        ExodusWriter.create(path) as writer,
        pytest.raises(ValueError, match="dimension must be 1, 2, or 3"),
    ):
        writer.initialize("bad", 4, 0, 0)


def test_writer_requires_initialize_before_writes(tmp_path: Path) -> None:
    path = tmp_path / "bad.exo"

    with (
        ExodusWriter.create(path) as writer,
        pytest.raises(ExodusWriteError, match="database is not initialized"),
    ):
        writer.write_time(0.0)


def test_writer_rejects_bad_coordinate_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize("bad", 2, 4, 0)

        with pytest.raises(ValueError, match="coords must be a two-dimensional array"):
            writer.write_coordinates([0.0, 1.0, 2.0, 3.0])


def test_writer_rejects_coordinate_dimension_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize("bad", 2, 4, 0)

        with pytest.raises(ValueError, match="does not match initialized dimension"):
            writer.write_coordinates(np.zeros((4, 3)))


def test_writer_rejects_coordinate_node_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize("bad", 2, 4, 0)

        with pytest.raises(ValueError, match="does not match initialized node count"):
            writer.write_coordinates(np.zeros((3, 2)))


def test_writer_rejects_too_many_element_blocks(tmp_path: Path) -> None:
    path = tmp_path / "bad.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize("bad", 2, 4, 2, element_blocks=1)
        writer.write_coordinates(_unit_square_quad())
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])

        with pytest.raises(ExodusWriteError, match="allocated number of element blocks exceeded"):
            writer.define_element_block(20, "quad", [[1, 2, 3, 4]])


def test_writer_rejects_too_many_node_sets(tmp_path: Path) -> None:
    path = tmp_path / "bad.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize("bad", 2, 4, 0, node_sets=1)
        writer.write_coordinates(_unit_square_quad())
        writer.define_node_set(100, [1, 2])

        with pytest.raises(ExodusWriteError, match="allocated number of node sets exceeded"):
            writer.define_node_set(101, [3, 4])


def test_writer_rejects_too_many_side_sets(tmp_path: Path) -> None:
    path = tmp_path / "bad.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize("bad", 2, 4, 1, side_sets=1)
        writer.write_coordinates(_unit_square_quad())
        writer.define_side_set(200, [1], [1])

        with pytest.raises(ExodusWriteError, match="allocated number of side sets exceeded"):
            writer.define_side_set(201, [1], [2])


def test_writer_rejects_element_variables_before_blocks(tmp_path: Path) -> None:
    path = tmp_path / "bad.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize("bad", 2, 4, 1, element_blocks=1)

        with pytest.raises(ExodusWriteError, match="define element blocks before defining"):
            writer.define_element_variables(["ENERGY"])


def test_writer_rejects_result_values_before_time(tmp_path: Path) -> None:
    path = tmp_path / "bad.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize("bad", 2, 4, 0)
        writer.define_global_variables(["TM_STEP"])

        with pytest.raises(ExodusWriteError, match="write a time value"):
            writer.write_global_values([0.0])


def test_writer_rejects_missing_node_variable_name(tmp_path: Path) -> None:
    path = tmp_path / "bad.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize("bad", 2, 4, 0)
        writer.define_node_variables(["TEMP"])
        writer.write_time(0.0)

        with pytest.raises(ExodusLookupError, match="variable 'DISPLX' not found"):
            writer.write_node_values("DISPLX", [0.0, 0.0, 0.0, 0.0])


def test_writer_rejects_missing_element_block_for_values(tmp_path: Path) -> None:
    path = tmp_path / "bad.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize("bad", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(_unit_square_quad())
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_element_variables(["ENERGY"])
        writer.write_time(0.0)

        with pytest.raises(ExodusLookupError, match="element block ID 20 not found"):
            writer.write_element_values("ENERGY", [0.0], block_id=20)
