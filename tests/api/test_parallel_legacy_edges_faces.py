# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np

import exodusii
from exodusii.api.writer import ExodusWriter


def test_parallel_legacy_edge_face_methods(tmp_path: Path) -> None:
    part0 = tmp_path / "p0.exo"
    part1 = tmp_path / "p1.exo"

    _write_part(part0, node_offset=0, edge_values=[1.0, 2.0, 3.0, 4.0], face_value=10.0)
    _write_part(part1, node_offset=4, edge_values=[5.0, 6.0, 7.0, 8.0], face_value=20.0)

    with exodusii.File(part0, part1) as exo:
        assert exo.get_edge_block_ids().tolist() == [20]
        assert exo.get_face_block_ids().tolist() == [30]

        assert exo.get_edge_block_iid(20) == 1
        assert exo.get_face_block_iid(30) == 1

        assert exo.get_edge_block(20).num_block_edges == 8
        assert exo.get_face_block(30).num_block_faces == 2

        assert np.allclose(
            exo.get_edge_block_conn(20),
            [[1, 2], [2, 3], [3, 4], [4, 1], [5, 6], [6, 7], [7, 8], [8, 5]],
        )
        assert np.allclose(exo.get_face_block_conn(30), [[1, 2, 3, 4], [5, 6, 7, 8]])

        assert exo.get_edge_variable_names().tolist() == ["EDGEVAR"]
        assert exo.get_face_variable_names().tolist() == ["FACEVAR"]

        assert np.allclose(
            exo.get_edge_variable_values(20, "EDGEVAR", 1), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        )
        assert np.allclose(exo.get_face_variable_values(30, "FACEVAR", 1), [10.0, 20.0])


def _write_part(
    path: Path, *, node_offset: int, edge_values: list[float], face_value: float
) -> None:
    coords = np.asarray(
        [
            [0.0 + node_offset, 0.0],
            [1.0 + node_offset, 0.0],
            [1.0 + node_offset, 1.0],
            [0.0 + node_offset, 1.0],
        ],
        dtype=float,
    )

    with ExodusWriter.create(path) as writer:
        writer.initialize(
            "legacy edge face",
            2,
            4,
            1,
            element_blocks=1,
            edge_count=4,
            edge_blocks=1,
            face_count=1,
            face_blocks=1,
        )
        writer.write_coordinates(coords)
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_edge_block(20, "edge2", [[1, 2], [2, 3], [3, 4], [4, 1]])
        writer.define_face_block(30, "quad", [[1, 2, 3, 4]])

        writer.define_edge_variables(["EDGEVAR"])
        writer.define_face_variables(["FACEVAR"])

        writer.write_time(0.0)
        writer.write_edge_values("EDGEVAR", edge_values, block_id=20)
        writer.write_face_values("FACEVAR", [face_value], block_id=30)
