# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np

import exodusii
from exodusii.api.copy import copy_file
from exodusii.api.file import ExodusFile
from exodusii.api.writer import ExodusWriter


def test_copy_extended_entities_and_variables(tmp_path: Path) -> None:
    source = tmp_path / "source.exo"
    target = tmp_path / "target.exo"

    _write_extended(source)
    copy_file(source, target)

    assert exodusii.allclose(source, target)

    with ExodusFile.open(target) as exo:
        assert exo.edge_block_ids().tolist() == [20]
        assert exo.face_block_ids().tolist() == [30]
        assert exo.edge_set_ids().tolist() == [40]
        assert exo.face_set_ids().tolist() == [50]
        assert exo.element_set_ids().tolist() == [60]

        assert exo.variable_names("edge") == ("EDGEVAR",)
        assert exo.variable_names("face") == ("FACEVAR",)
        assert exo.variable_names("edge_set") == ("ESVAR",)
        assert exo.variable_names("face_set") == ("FSVAR",)
        assert exo.variable_names("element_set") == ("ELSVAR",)

        assert np.allclose(
            exo.values("EDGEVAR", on="edge", block_id=20, time=0), [1.0, 2.0, 3.0, 4.0]
        )
        assert np.allclose(exo.values("FACEVAR", on="face", block_id=30, time=0), [5.0])
        assert np.allclose(exo.values("ESVAR", on="edge_set", set_id=40, time=0), [6.0, 7.0])
        assert np.allclose(exo.values("FSVAR", on="face_set", set_id=50, time=0), [8.0])
        assert np.allclose(exo.values("ELSVAR", on="element_set", set_id=60, time=0), [9.0])


def test_copy_globals_written_once_and_correctly(tmp_path: Path) -> None:
    source = tmp_path / "source_globals.exo"
    target = tmp_path / "target_globals.exo"

    with ExodusWriter.create(source) as writer:
        writer.initialize("globals", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(_coords())
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_global_variables(["A", "B"])

        writer.write_time(0.0)
        writer.write_global_values([1.0, 2.0])

        writer.write_time(1.0)
        writer.write_global_values([3.0, 4.0])

    copy_file(source, target)

    with ExodusFile.open(target) as exo:
        assert exo.variable_names("global") == ("A", "B")
        assert np.allclose(exo.values("A", on="global"), [1.0, 3.0])
        assert np.allclose(exo.values("B", on="global"), [2.0, 4.0])


def _write_extended(path: Path) -> None:
    with ExodusWriter.create(path) as writer:
        writer.initialize(
            "extended",
            2,
            4,
            1,
            element_blocks=1,
            edge_count=4,
            edge_blocks=1,
            edge_sets=1,
            face_count=1,
            face_blocks=1,
            face_sets=1,
            element_sets=1,
        )
        writer.write_coordinates(_coords())
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_edge_block(20, "edge2", [[1, 2], [2, 3], [3, 4], [4, 1]])
        writer.define_face_block(30, "quad", [[1, 2, 3, 4]])

        writer.define_edge_set(40, [1, 2], orientations=[1, -1])
        writer.define_face_set(50, [1], orientations=[1])
        writer.define_element_set(60, [1])

        writer.define_edge_variables(["EDGEVAR"])
        writer.define_face_variables(["FACEVAR"])
        writer.define_edge_set_variables(["ESVAR"])
        writer.define_face_set_variables(["FSVAR"])
        writer.define_element_set_variables(["ELSVAR"])

        writer.write_time(0.0)
        writer.write_edge_values("EDGEVAR", [1.0, 2.0, 3.0, 4.0], block_id=20)
        writer.write_face_values("FACEVAR", [5.0], block_id=30)
        writer.write_values("ESVAR", [6.0, 7.0], on=writer_entity("edge_set"), set_id=40)
        writer.write_values("FSVAR", [8.0], on=writer_entity("face_set"), set_id=50)
        writer.write_values("ELSVAR", [9.0], on=writer_entity("element_set"), set_id=60)


def writer_entity(name: str):
    from exodusii.core.entities import entity

    return entity(name)


def _coords() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
