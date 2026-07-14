# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np

import exodusii


def test_legacy_edge_face_variable_methods(tmp_path: Path) -> None:
    path = tmp_path / "legacy_edge_face.exo"

    with exodusii.File(path, mode="w") as exo:
        exo.put_init(
            "legacy edge face",
            2,
            4,
            1,
            1,
            0,
            0,
            num_edge=4,
            num_edge_blk=1,
            num_face=1,
            num_face_blk=1,
        )
        exo.put_coords(_coords())
        exo.put_element_block(10, "quad", 1, 4)
        exo.put_element_conn(10, [[1, 2, 3, 4]])

        exo.writer.define_edge_block(20, "edge2", [[1, 2], [2, 3], [3, 4], [4, 1]])
        exo.writer.define_face_block(30, "quad", [[1, 2, 3, 4]])

        exo.put_edge_variable_params(1)
        exo.put_edge_variable_names(["EDGEVAR"])
        exo.put_edge_variable_truth_table([[1]])

        exo.put_face_variable_params(1)
        exo.put_face_variable_names(["FACEVAR"])
        exo.put_face_variable_truth_table([[1]])

        exo.put_time(1, 0.0)
        exo.put_edge_variable_values(1, 20, "EDGEVAR", [1.0, 2.0, 3.0, 4.0])
        exo.put_face_variable_values(1, 30, "FACEVAR", [5.0])

    with exodusii.File(path) as exo:
        assert exo.get_edge_variable_names().tolist() == ["EDGEVAR"]
        assert exo.get_face_variable_names().tolist() == ["FACEVAR"]
        assert exo.get_edge_variable_number() == 1
        assert exo.get_face_variable_number() == 1
        assert np.allclose(exo.get_edge_variable_truth_table(), [[1]])
        assert np.allclose(exo.get_face_variable_truth_table(), [[1]])
        assert np.allclose(exo.get_edge_variable_values(20, "EDGEVAR", 1), [1.0, 2.0, 3.0, 4.0])
        assert np.allclose(exo.get_face_variable_values(30, "FACEVAR", 1), [5.0])


def test_legacy_set_variable_methods(tmp_path: Path) -> None:
    path = tmp_path / "legacy_set_vars.exo"

    with exodusii.File(path, mode="w") as exo:
        exo.put_init("legacy set vars", 2, 4, 1, 0, 1, 1)
        exo.put_coords(_coords())
        exo.put_node_set_param(10, 2)
        exo.put_node_set_nodes(10, [1, 2])
        exo.put_side_set_param(20, 1)
        exo.put_side_set_sides(20, [1], [3])

        exo.put_node_set_variable_params(1)
        exo.put_node_set_variable_names(["NSVAR"])
        exo.put_node_set_variable_truth_table([[1]])

        exo.put_side_set_variable_params(1)
        exo.put_side_set_variable_names(["SSVAR"])
        exo.put_side_set_variable_truth_table([[1]])

        exo.put_time(1, 0.0)
        exo.put_node_set_variable_values(1, 10, "NSVAR", [1.0, 2.0])
        exo.put_side_set_variable_values(1, 20, "SSVAR", [3.0])

    with exodusii.File(path) as exo:
        assert exo.get_node_set_variable_names().tolist() == ["NSVAR"]
        assert exo.get_side_set_variable_names().tolist() == ["SSVAR"]
        assert np.allclose(exo.get_node_set_variable_truth_table(), [[1]])
        assert np.allclose(exo.get_side_set_variable_truth_table(), [[1]])
        assert np.allclose(exo.get_node_set_variable_values(10, "NSVAR", 1), [1.0, 2.0])
        assert np.allclose(exo.get_side_set_variable_values(20, "SSVAR", 1), [3.0])


def _coords() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
