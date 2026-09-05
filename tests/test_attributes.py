# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np

import exodusii
from exodusii.api.copy import copy_file
from exodusii.api.file import ExodusFile
from exodusii.api.writer import ExodusWriter
from exodusii.core.entities import Entity


def test_block_attributes_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "attrs.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize(
            "attrs",
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

        writer.write_block_attributes(Entity.ELEMENT_BLOCK, 10, [[1.0, 2.0]], names=["A", "B"])
        writer.write_block_attributes(
            Entity.EDGE_BLOCK, 20, [[1.0], [2.0], [3.0], [4.0]], names=["L"]
        )
        writer.write_block_attributes(Entity.FACE_BLOCK, 30, [[5.0]], names=["F"])

    with ExodusFile.open(path) as exo:
        assert exo.attribute_names("element_block", 10) == ("A", "B")
        assert np.allclose(exo.attributes("element_block", 10), [[1.0, 2.0]])  # ty: ignore[invalid-argument-type]
        assert np.allclose(exo.attribute_values("element_block", 10, "B"), [2.0])

        assert exo.attribute_names("edge_block", 20) == ("L",)
        assert np.allclose(exo.attribute_values("edge_block", 20, "L"), [1.0, 2.0, 3.0, 4.0])

        assert exo.attribute_names("face_block", 30) == ("F",)
        assert np.allclose(exo.attribute_values("face_block", 30, "F"), [5.0])


def test_legacy_block_attributes(tmp_path: Path) -> None:
    path = tmp_path / "legacy_attrs.exo"

    with exodusii.File(path, mode="w") as exo:
        exo.put_init("legacy attrs", 2, 4, 1, 1, 0, 0)
        exo.put_coords(_coords())
        exo.put_element_block(10, "quad", 1, 4)
        exo.put_element_conn(10, [[1, 2, 3, 4]])
        exo.put_element_attr(10, [[7.0, 8.0]])
        exo.put_element_attribute_names(10, ["A", "B"])

    with exodusii.File(path) as exo:
        assert exo.get_element_attribute_names(10) == ["A", "B"]
        assert np.allclose(exo.get_element_attr(10), [[7.0, 8.0]])
        assert np.allclose(exo.get_element_attr_values(10, "A"), [7.0])


def test_copy_preserves_block_attributes(tmp_path: Path) -> None:
    source = tmp_path / "source.exo"
    target = tmp_path / "target.exo"

    with ExodusWriter.create(source) as writer:
        writer.initialize("attrs", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(_coords())
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.write_block_attributes(Entity.ELEMENT_BLOCK, 10, [[1.0, 2.0]], names=["A", "B"])

    copy_file(source, target)

    with ExodusFile.open(target) as exo:
        assert exo.attribute_names("element_block", 10) == ("A", "B")
        assert np.allclose(exo.attributes("element_block", 10), [[1.0, 2.0]])  # ty: ignore[invalid-argument-type]


def _coords() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
