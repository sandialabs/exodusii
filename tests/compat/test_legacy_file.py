# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np

import exodusii
from exodusii.compat import ExodusIIFile


def test_legacy_file_write_and_read_mesh(tmp_path: Path) -> None:
    path = tmp_path / "legacy.exo"

    with exodusii.File(path, mode="w") as exo:
        exo.put_init("legacy", 2, 4, 1, 1, 1, 1)
        exo.put_coord([0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0])
        exo.put_element_block(10, "quad", 1, 4)
        exo.put_element_block_name(10, "block_10")
        exo.put_element_conn(10, np.asarray([[1, 2, 3, 4]]))
        exo.put_node_set_param(100, 2, 2)
        exo.put_node_set_name(100, "nodeset_100")
        exo.put_node_set_nodes(100, [1, 4])
        exo.put_node_set_dist_fact(100, [1.0, 2.0])
        exo.put_side_set_param(200, 1, 1)
        exo.put_side_set_name(200, "sideset_200")
        exo.put_side_set_sides(200, [1], [2])
        exo.put_side_set_dist_fact(200, [3.0])

    with exodusii.File(path) as exo:
        assert isinstance(exo, ExodusIIFile)
        assert exo.filename == path
        assert exo.title() == "legacy"
        assert exo.storage_type() == "d"
        assert exo.num_dimensions() == 2
        assert exo.num_nodes() == 4
        assert exo.num_elems() == 1
        assert exo.num_blks() == 1
        assert exo.num_node_sets() == 1
        assert exo.num_side_sets() == 1

        assert exo.get_coord_names().tolist() == ["X", "Y"]
        assert np.allclose(exo.get_coords(), [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

        assert exo.get_element_block_ids().tolist() == [10]
        assert exo.get_element_block_id(1) == 10
        assert exo.get_element_block_iid(10) == 1

        block = exo.get_element_block(10)
        assert block.elem_type == "QUAD"
        assert block.name == "block_10"
        assert block.num_block_elems == 1
        assert block.num_elem_nodes == 4

        assert np.allclose(exo.get_element_conn(10), [[1, 2, 3, 4]])

        node_set = exo.get_node_set(100)
        assert node_set.name == "nodeset_100"
        assert np.allclose(node_set.nodes, [1, 4])
        assert np.allclose(node_set.dist_facts, [1.0, 2.0])

        side_set = exo.get_side_set(200)
        assert side_set.name == "sideset_200"
        assert np.allclose(side_set.elems, [1])
        assert np.allclose(side_set.sides, [2])
        assert np.allclose(side_set.dist_facts, [3.0])


def test_legacy_file_write_and_read_results(tmp_path: Path) -> None:
    path = tmp_path / "legacy_results.exo"

    with exodusii.File(path, mode="w") as exo:
        exo.put_init("legacy results", 2, 4, 1, 1, 0, 0)
        exo.put_coords(np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
        exo.put_element_block(10, "quad", 1, 4)
        exo.put_element_conn(10, [[1, 2, 3, 4]])

        exo.put_global_variable_params(1)
        exo.put_global_variable_names(["TM_STEP"])
        exo.put_node_variable_params(3)
        exo.put_node_variable_names(["DISPLX", "DISPLY", "TEMP"])
        exo.put_element_variable_params(1)
        exo.put_element_variable_names(["ENERGY"])

        exo.put_time(1, 0.0)
        exo.put_global_variable_values(1, [0.0])
        exo.put_node_variable_values(1, "DISPLX", [0.0, 0.0, 0.0, 0.0])
        exo.put_node_variable_values(1, "DISPLY", [0.0, 0.0, 0.0, 0.0])
        exo.put_node_variable_values(1, "TEMP", [10.0, 20.0, 30.0, 40.0])
        exo.put_element_variable_values(1, 10, "ENERGY", [0.5])

        exo.put_time(2, 1.0)
        exo.put_global_variable_values(2, [1.0])
        exo.put_node_variable_values(2, "DISPLX", [0.1, 0.1, 0.1, 0.1])
        exo.put_node_variable_values(2, "DISPLY", [0.0, 0.0, 0.0, 0.0])
        exo.put_node_variable_values(2, "TEMP", [11.0, 21.0, 31.0, 41.0])
        exo.put_element_variable_values(2, 10, "ENERGY", [1.5])

    with exodusii.File(path) as exo:
        assert np.allclose(exo.get_times(), [0.0, 1.0])
        assert exo.get_global_variable_names().tolist() == ["TM_STEP"]
        assert exo.get_node_variable_names().tolist() == ["DISPLX", "DISPLY", "TEMP"]
        assert exo.get_element_variable_names().tolist() == ["ENERGY"]

        assert np.allclose(exo.get_global_variable_values("TM_STEP"), [0.0, 1.0])
        assert np.allclose(exo.get_all_global_variable_values(2), [1.0])
        assert np.allclose(
            exo.get_node_variable_values("TEMP", time_step=2), [11.0, 21.0, 31.0, 41.0]
        )
        assert np.allclose(exo.get_node_variable_history("TEMP", 2), [20.0, 21.0])
        assert np.allclose(exo.get_element_variable_values(10, "ENERGY", 2), [1.5])
        assert np.allclose(exo.get_element_variable_history("ENERGY", 1), [0.5, 1.5])
        assert exo.get_displ_variable_names() == ("DISPLX", "DISPLY")


def test_legacy_write_globals(tmp_path: Path) -> None:
    path = tmp_path / "globals.exo"
    times = np.asarray([0.0, 1.0, 2.0])
    data = {"foo": np.asarray([10.0, 11.0, 12.0]), "bar": np.asarray([20.0, 21.0, 22.0])}

    written = exodusii.write_globals(data, times, title="globals", filename=path)

    assert written == str(path)

    with exodusii.File(path) as exo:
        assert exo.title() == "globals"
        assert exo.num_dimensions() == 1
        assert exo.num_nodes() == 0
        assert exo.num_elems() == 0
        assert sorted(exo.get_global_variable_names().tolist()) == ["bar", "foo"]
        assert np.allclose(exo.get_global_variable_values("foo"), [10.0, 11.0, 12.0])
        assert np.allclose(exo.get_times(), times)


def test_legacy_aliases() -> None:
    assert exodusii.exo_file is exodusii.File
    assert exodusii.exodusii_file is ExodusIIFile
    assert exodusii.ExodusIIFile is ExodusIIFile
