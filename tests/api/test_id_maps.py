# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pytest

from exodusii.api.compare import similar
from exodusii.api.copy import copy_file
from exodusii.api.file import ExodusFile
from exodusii.api.writer import ExodusWriter


def test_writer_and_reader_id_maps(tmp_path: Path) -> None:
    path = tmp_path / "maps.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize(
            "maps",
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
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_edge_block(20, "edge2", [[1, 2], [2, 3], [3, 4], [4, 1]])
        writer.define_face_block(30, "quad", [[1, 2, 3, 4]])

        writer.write_node_id_map([10, 20, 30, 40])
        writer.write_element_id_map([100])
        writer.write_edge_id_map([200, 201, 202, 203])
        writer.write_face_id_map([300])

    with ExodusFile.open(path) as exo:
        assert exo.ids("node").tolist() == [10, 20, 30, 40]
        assert exo.ids("element").tolist() == [100]
        assert exo.ids("edge").tolist() == [200, 201, 202, 203]
        assert exo.ids("face").tolist() == [300]


def test_copy_preserves_id_maps(tmp_path: Path) -> None:
    source = tmp_path / "source.exo"
    target = tmp_path / "target.exo"

    _write_mapped(source)
    copy_file(source, target)

    with ExodusFile.open(target) as exo:
        assert exo.ids("node").tolist() == [10, 20, 30, 40]
        assert exo.ids("element").tolist() == [100]
        assert exo.ids("edge").tolist() == [200, 201, 202, 203]
        assert exo.ids("face").tolist() == [300]


def test_similar_detects_id_map_difference(tmp_path: Path) -> None:
    one = tmp_path / "one.exo"
    two = tmp_path / "two.exo"

    _write_mapped(one)
    _write_mapped(two, node_ids=[10, 20, 30, 99])

    with pytest.raises(ValueError, match="node ID map"):
        similar(one, two)


def test_copy_does_not_materialize_default_id_maps(tmp_path: Path) -> None:
    source = tmp_path / "source.exo"
    target = tmp_path / "target.exo"

    with ExodusWriter.create(source) as writer:
        writer.initialize("no maps", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(_coords())
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])

    copy_file(source, target)

    with ExodusFile.open(target) as exo:
        assert "node_num_map" not in exo.variables()
        assert "elem_num_map" not in exo.variables()


def _write_mapped(path: Path, *, node_ids: list[int] | None = None) -> None:
    node_ids = node_ids or [10, 20, 30, 40]

    with ExodusWriter.create(path) as writer:
        writer.initialize(
            "mapped",
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
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_edge_block(20, "edge2", [[1, 2], [2, 3], [3, 4], [4, 1]])
        writer.define_face_block(30, "quad", [[1, 2, 3, 4]])
        writer.write_node_id_map(node_ids)
        writer.write_element_id_map([100])
        writer.write_edge_id_map([200, 201, 202, 203])
        writer.write_face_id_map([300])


def _coords() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
