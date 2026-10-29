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
        # exodusii < 2 legacy writers always produced single precision.
        assert exo.storage_type() == "f"
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


def test_legacy_put_element_block_allocates_attributes_and_subconnectivity(tmp_path) -> None:
    # exodusii < 2 honoured num_attr / num_edges_per_elem / num_faces_per_elem.
    import netCDF4

    path = tmp_path / "attr.exo"
    exo = ExodusIIFile(path, "w")
    exo.put_init("attr", 1, 2, 1, 1, 0, 0)
    exo.put_coord([0.0, 1.0])
    exo.put_element_block(1, "bar2", 1, 2, num_edges_per_elem=1, num_attr=2)
    exo.put_element_conn(1, [[1, 2]])
    exo.close()

    with netCDF4.Dataset(path) as ds:
        assert ds["attrib1"].shape == (1, 2)
        assert ds["edgconn1"].shape == (1, 1)

    reader = ExodusIIFile(path)
    try:
        assert reader.get_element_block(1).num_elem_attrs == 2
        assert reader.get_element_attr(1).tolist() == [[0.0, 0.0]]
    finally:
        reader.close()


def test_legacy_time_step_zero_is_last_step(datadir) -> None:
    # Kept from exodusii < 2, which indexed values[time_step - 1].
    exo = ExodusIIFile(datadir / "noh.exo")
    try:
        assert exo.get_time(0) == exo.get_times()[-1]
    finally:
        exo.close()


def _two_bar_blocks(path):
    exo = ExodusIIFile(path, "w")
    exo.put_init("tt", 1, 3, 2, 2, 0, 0)
    exo.put_coord([0.0, 1.0, 2.0])
    for block_id, conn in ((1, [[1, 2]]), (2, [[2, 3]])):
        exo.put_element_block(block_id, "bar2", 1, 2)
        exo.put_element_conn(block_id, conn)
    exo.put_element_variable_params(2)
    exo.put_element_variable_names(["A", "B"])
    return exo


def test_legacy_truth_table_controls_which_variables_exist(tmp_path) -> None:
    import netCDF4

    path = tmp_path / "tt.exo"
    exo = _two_bar_blocks(path)
    exo.put_element_variable_truth_table(np.array([[1, 0], [1, 1]]))
    exo.put_time(1, 0.0)
    exo.put_element_variable_values(1, 1, "A", [1.0])
    exo.put_element_variable_values(1, 2, "B", [3.0])
    exo.close()

    with netCDF4.Dataset(path) as ds:
        values = sorted(v for v in ds.variables if v.startswith("vals_elem_var"))
        assert values == ["vals_elem_var1eb1", "vals_elem_var1eb2", "vals_elem_var2eb2"]
        assert ds["elem_var_tab"][:].tolist() == [[1, 0], [1, 1]]


def test_legacy_without_truth_table_records_written_variables(tmp_path) -> None:
    path = tmp_path / "nott.exo"
    exo = _two_bar_blocks(path)
    exo.put_time(1, 0.0)
    exo.put_element_variable_values(1, 1, "A", [1.0])
    exo.put_element_variable_values(1, 2, "A", [2.0])
    exo.put_element_variable_values(1, 2, "B", [3.0])
    exo.close()

    reader = ExodusIIFile(path)
    try:
        assert reader.get_element_variable_truth_table().tolist() == [[1, 0], [1, 1]]
        assert reader.get_element_variable_values(None, "A", 1).tolist() == [1.0, 2.0]
        assert reader.get_element_variable_values(2, "B", 1).tolist() == [3.0]
    finally:
        reader.close()
