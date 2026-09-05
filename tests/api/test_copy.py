# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np

import exodusii
from exodusii.api.copy import copy
from exodusii.api.copy import copy_file
from exodusii.api.file import ExodusFile
from exodusii.api.writer import ExodusWriter


def test_copy_file_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source.exo"
    target = tmp_path / "target.exo"

    _write_source(source)

    written = copy_file(source, target)

    assert written == str(target)
    assert exodusii.allclose(source, target)


def test_copy_open_objects(tmp_path: Path) -> None:
    source = tmp_path / "source.exo"
    target = tmp_path / "target.exo"

    _write_source(source)

    with ExodusFile.open(source) as source_file, ExodusWriter.create(target) as target_writer:
        copy(source_file, target_writer)

    assert exodusii.allclose(source, target)


def test_copy_top_level_export(tmp_path: Path) -> None:
    source = tmp_path / "source.exo"
    target = tmp_path / "target.exo"

    _write_source(source)

    exodusii.copy_file(source, target)

    assert exodusii.allclose(source, target)


def test_copy_preserves_mesh(tmp_path: Path) -> None:
    source = tmp_path / "source.exo"
    target = tmp_path / "target.exo"

    _write_source(source)
    copy_file(source, target)

    with ExodusFile.open(target) as exo:
        assert exo.title == "copy source"
        assert exo.dimension == 2
        assert exo.node_count == 4
        assert exo.element_count == 1
        assert exo.element_block_ids().tolist() == [10]
        assert exo.element_block(10).name == "block_10"
        assert np.allclose(exo.element_connectivity(10), [[1, 2, 3, 4]])
        assert np.allclose(exo.coordinates(), _unit_square_quad())


def test_copy_preserves_sets(tmp_path: Path) -> None:
    source = tmp_path / "source.exo"
    target = tmp_path / "target.exo"

    _write_source(source)
    copy_file(source, target)

    with ExodusFile.open(target) as exo:
        node_set = exo.node_set(100)
        assert node_set.name == "nodeset_100"
        assert np.allclose(node_set.nodes, [1, 4])  # ty: ignore[invalid-argument-type]
        assert np.allclose(node_set.dist_facts, [1.0, 2.0])  # ty: ignore[invalid-argument-type]

        side_set = exo.side_set(200)
        assert side_set.name == "sideset_200"
        assert np.allclose(side_set.elems, [1])  # ty: ignore[invalid-argument-type]
        assert np.allclose(side_set.sides, [2])  # ty: ignore[invalid-argument-type]
        assert np.allclose(side_set.dist_facts, [3.0, 4.0])  # ty: ignore[invalid-argument-type]


def test_copy_preserves_results(tmp_path: Path) -> None:
    source = tmp_path / "source.exo"
    target = tmp_path / "target.exo"

    _write_source(source)
    copy_file(source, target)

    with ExodusFile.open(target) as exo:
        assert np.allclose(exo.times(), [0.0, 1.0])
        assert exo.variable_names("global") == ("TM_STEP",)
        assert exo.variable_names("node") == ("TEMP",)
        assert exo.variable_names("element") == ("ENERGY",)

        assert np.allclose(exo.values("TM_STEP", on="global"), [0.0, 1.0])
        assert np.allclose(exo.values("TEMP", on="node", time="last"), [11.0, 21.0, 31.0, 41.0])
        assert np.allclose(exo.values("ENERGY", on="element", block=10), [[0.5], [1.5]])


def _write_source(path: Path) -> None:
    with ExodusWriter.create(path) as writer:
        writer.initialize("copy source", 2, 4, 1, element_blocks=1, node_sets=1, side_sets=1)
        writer.write_coordinates(_unit_square_quad())
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]], name="block_10")
        writer.define_node_set(100, [1, 4], distribution_factors=[1.0, 2.0], name="nodeset_100")
        writer.define_side_set(200, [1], [2], distribution_factors=[3.0, 4.0], name="sideset_200")

        writer.define_global_variables(["TM_STEP"])
        writer.define_node_variables(["TEMP"])
        writer.define_element_variables(["ENERGY"])

        writer.write_time(0.0)
        writer.write_global_values([0.0])
        writer.write_node_values("TEMP", [10.0, 20.0, 30.0, 40.0])
        writer.write_element_values("ENERGY", [0.5], block_id=10)

        writer.write_time(1.0)
        writer.write_global_values([1.0])
        writer.write_node_values("TEMP", [11.0, 21.0, 31.0, 41.0])
        writer.write_element_values("ENERGY", [1.5], block_id=10)


def _unit_square_quad() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
