# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np

from exodusii.api.file import ExodusFile
from exodusii.api.writer import ExodusWriter


def test_writer_edge_and_face_blocks(tmp_path: Path) -> None:
    path = tmp_path / "blocks.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize(
            "blocks",
            2,
            4,
            1,
            element_blocks=1,
            edge_count=4,
            edge_blocks=1,
            face_count=1,
            face_blocks=1,
        )
        writer.write_coordinates(_coords())
        writer.define_element_block(
            10, "quad", [[1, 2, 3, 4]], edge_connectivity=[[1, 2, 3, 4]], name="elem_block"
        )
        writer.define_edge_block(20, "edge2", [[1, 2], [2, 3], [3, 4], [4, 1]], name="edge_block")
        writer.define_face_block(30, "quad", [[1, 2, 3, 4]], name="face_block")

    with ExodusFile.open(path) as exo:
        assert exo.element_block_ids().tolist() == [10]
        assert exo.edge_block_ids().tolist() == [20]
        assert exo.face_block_ids().tolist() == [30]

        assert exo.element_block(10).name == "elem_block"
        assert exo.edge_block(20).name == "edge_block"
        assert exo.face_block(30).name == "face_block"

        assert np.allclose(exo.element_connectivity(10), [[1, 2, 3, 4]])
        assert np.allclose(exo.element_edge_connectivity(10), [[1, 2, 3, 4]])
        assert np.allclose(exo.edge_connectivity(20), [[1, 2], [2, 3], [3, 4], [4, 1]])
        assert np.allclose(exo.face_connectivity(30), [[1, 2, 3, 4]])


def test_writer_all_set_types(tmp_path: Path) -> None:
    path = tmp_path / "sets.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize(
            "sets", 2, 4, 1, node_sets=1, side_sets=1, edge_sets=1, face_sets=1, element_sets=1
        )
        writer.write_coordinates(_coords())

        writer.define_node_set(10, [1, 2], distribution_factors=[1.0, 2.0], name="node_set")
        writer.define_side_set(20, [1], [3], distribution_factors=[3.0], name="side_set")
        writer.define_edge_set(
            30, [1, 2], orientations=[1, -1], distribution_factors=[4.0, 5.0], name="edge_set"
        )
        writer.define_face_set(
            40, [1], orientations=[1], distribution_factors=[6.0], name="face_set"
        )
        writer.define_element_set(50, [1], distribution_factors=[7.0], name="element_set")

    with ExodusFile.open(path) as exo:
        assert exo.node_set_ids().tolist() == [10]
        assert exo.side_set_ids().tolist() == [20]
        assert exo.edge_set_ids().tolist() == [30]
        assert exo.face_set_ids().tolist() == [40]
        assert exo.element_set_ids().tolist() == [50]

        assert np.allclose(exo.node_set(10).nodes, [1, 2])
        assert np.allclose(exo.side_set(20).sides, [3])
        assert np.allclose(exo.edge_set(30).extra_entries, [1, -1])
        assert np.allclose(exo.face_set(40).extra_entries, [1])
        assert np.allclose(exo.element_set(50).entries, [1])


def _coords() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
