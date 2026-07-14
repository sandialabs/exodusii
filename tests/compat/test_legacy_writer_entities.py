# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np

import exodusii


def test_legacy_writer_edge_face_blocks(tmp_path: Path) -> None:
    path = tmp_path / "legacy_blocks.exo"

    with exodusii.File(path, mode="w") as exo:
        exo.put_init(
            "legacy blocks",
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

        exo.put_edge_block(20, "edge2", 4, 2)
        exo.put_edge_conn(20, [[1, 2], [2, 3], [3, 4], [4, 1]])
        exo.put_edge_id_map([101, 102, 103, 104])

        exo.put_face_block(30, "quad", 1, 4)
        exo.put_face_conn(30, [[1, 2, 3, 4]])
        exo.put_face_id_map([201])

    with exodusii.File(path) as exo:
        assert exo.get_edge_block_ids().tolist() == [20]
        assert exo.get_face_block_ids().tolist() == [30]
        assert np.allclose(exo.get_edge_block_conn(20), [[1, 2], [2, 3], [3, 4], [4, 1]])
        assert np.allclose(exo.get_face_block_conn(30), [[1, 2, 3, 4]])
        assert np.allclose(exo.reader.ids("edge"), [101, 102, 103, 104])
        assert np.allclose(exo.reader.ids("face"), [201])


def test_legacy_writer_edge_face_element_sets(tmp_path: Path) -> None:
    path = tmp_path / "legacy_sets.exo"

    with exodusii.File(path, mode="w") as exo:
        exo.put_init(
            "legacy sets",
            2,
            4,
            1,
            0,
            0,
            0,
            num_edge=4,
            num_edge_blk=0,
            num_edge_sets=1,
            num_face=1,
            num_face_blk=0,
            num_face_sets=1,
            num_elem_sets=1,
        )
        exo.put_coords(_coords())

        exo.put_edge_set_param(100, 2, num_dist_facts=2)
        exo.put_edge_set_name(100, "edge_set")
        exo.put_edge_set_edges(100, [1, 2], orientations=[1, -1])
        exo.put_edge_set_dist_fact(100, [1.0, 2.0])

        exo.put_face_set_param(200, 1, num_dist_facts=1)
        exo.put_face_set_name(200, "face_set")
        exo.put_face_set_faces(200, [1], orientations=[1])
        exo.put_face_set_dist_fact(200, [3.0])

        exo.put_element_set_param(300, 1, num_dist_facts=1)
        exo.put_element_set_name(300, "element_set")
        exo.put_element_set_elems(300, [1])
        exo.put_element_set_dist_fact(300, [4.0])

    with exodusii.File(path) as exo:
        assert exo.get_edge_set_ids().tolist() == [100]
        assert exo.get_face_set_ids().tolist() == [200]
        assert exo.get_element_set_ids().tolist() == [300]

        edge_set = exo.get_edge_set(100)
        assert edge_set.name == "edge_set"
        assert np.allclose(edge_set.edges, [1, 2])
        assert np.allclose(edge_set.orientations, [1, -1])
        assert np.allclose(edge_set.dist_facts, [1.0, 2.0])

        face_set = exo.get_face_set(200)
        assert face_set.name == "face_set"
        assert np.allclose(face_set.faces, [1])
        assert np.allclose(face_set.orientations, [1])
        assert np.allclose(face_set.dist_facts, [3.0])

        element_set = exo.get_element_set(300)
        assert element_set.name == "element_set"
        assert np.allclose(element_set.elems, [1])
        assert np.allclose(element_set.dist_facts, [4.0])


def test_legacy_put_init_allocates_extended_set_counts(tmp_path: Path) -> None:
    path = tmp_path / "extended_sets.exo"

    with exodusii.File(path, mode="w") as exo:
        exo.put_init(
            "extended sets",
            2,
            4,
            1,
            0,
            0,
            0,
            num_edge=4,
            num_edge_sets=1,
            num_face=1,
            num_face_sets=1,
            num_elem_sets=1,
        )

    with exodusii.File(path) as exo:
        assert exo.reader.dimension_size("num_edge_sets", default=0) == 1
        assert exo.reader.dimension_size("num_face_sets", default=0) == 1
        assert exo.reader.dimension_size("num_elem_sets", default=0) == 1


def _coords() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
