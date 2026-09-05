# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np

from exodusii.api.file import ExodusFile
from exodusii.api.writer import ExodusWriter


def test_element_edge_face_variables_and_truth_tables(tmp_path: Path) -> None:
    path = tmp_path / "vars.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize(
            "vars",
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

        writer.define_element_variables(["ENERGY"], truth_table=[[1]])
        writer.define_edge_variables(["EDGEVAR"], truth_table=[[1]])
        writer.define_face_variables(["FACEVAR"], truth_table=[[1]])

        writer.write_time(0.0)
        writer.write_element_values("ENERGY", [1.0], block_id=10)
        writer.write_edge_values("EDGEVAR", [2.0, 3.0, 4.0, 5.0], block_id=20)
        writer.write_face_values("FACEVAR", [6.0], block_id=30)

    with ExodusFile.open(path) as exo:
        assert exo.variable_names("element") == ("ENERGY",)
        assert exo.variable_names("edge") == ("EDGEVAR",)
        assert exo.variable_names("face") == ("FACEVAR",)

        assert np.allclose(exo.variable_truth_table("element"), [[1]])  # ty: ignore[invalid-argument-type]
        assert np.allclose(exo.variable_truth_table("edge"), [[1]])  # ty: ignore[invalid-argument-type]
        assert np.allclose(exo.variable_truth_table("face"), [[1]])  # ty: ignore[invalid-argument-type]

        assert np.allclose(exo.values("ENERGY", on="element", block_id=10, time=0), [1.0])
        assert np.allclose(
            exo.values("EDGEVAR", on="edge", block_id=20, time=0), [2.0, 3.0, 4.0, 5.0]
        )
        assert np.allclose(exo.values("FACEVAR", on="face", block_id=30, time=0), [6.0])


def test_set_variables_and_truth_tables(tmp_path: Path) -> None:
    path = tmp_path / "set_vars.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize(
            "set vars", 2, 4, 1, node_sets=1, side_sets=1, edge_sets=1, face_sets=1, element_sets=1
        )
        writer.write_coordinates(_coords())
        writer.define_node_set(10, [1, 2])
        writer.define_side_set(20, [1], [3])
        writer.define_edge_set(30, [1, 2], orientations=[1, -1])
        writer.define_face_set(40, [1], orientations=[1])
        writer.define_element_set(50, [1])

        writer.define_node_set_variables(["NSVAR"], truth_table=[[1]])
        writer.define_side_set_variables(["SSVAR"], truth_table=[[1]])
        writer.define_edge_set_variables(["ESVAR"], truth_table=[[1]])
        writer.define_face_set_variables(["FSVAR"], truth_table=[[1]])
        writer.define_element_set_variables(["ELSVAR"], truth_table=[[1]])

        writer.write_time(0.0)
        writer.write_node_set_values("NSVAR", [1.0, 2.0], set_id=10)
        writer.write_values("SSVAR", [3.0], on=writer_entity("side_set"), set_id=20)
        writer.write_values("ESVAR", [4.0, 5.0], on=writer_entity("edge_set"), set_id=30)
        writer.write_values("FSVAR", [6.0], on=writer_entity("face_set"), set_id=40)
        writer.write_values("ELSVAR", [7.0], on=writer_entity("element_set"), set_id=50)

    with ExodusFile.open(path) as exo:
        assert np.allclose(exo.variable_truth_table("node_set"), [[1]])  # ty: ignore[invalid-argument-type]
        assert np.allclose(exo.variable_truth_table("side_set"), [[1]])  # ty: ignore[invalid-argument-type]
        assert np.allclose(exo.variable_truth_table("edge_set"), [[1]])  # ty: ignore[invalid-argument-type]
        assert np.allclose(exo.variable_truth_table("face_set"), [[1]])  # ty: ignore[invalid-argument-type]
        assert np.allclose(exo.variable_truth_table("element_set"), [[1]])  # ty: ignore[invalid-argument-type]

        assert np.allclose(exo.values("NSVAR", on="node_set", set_id=10, time=0), [1.0, 2.0])
        assert np.allclose(exo.values("SSVAR", on="side_set", set_id=20, time=0), [3.0])
        assert np.allclose(exo.values("ESVAR", on="edge_set", set_id=30, time=0), [4.0, 5.0])
        assert np.allclose(exo.values("FSVAR", on="face_set", set_id=40, time=0), [6.0])
        assert np.allclose(exo.values("ELSVAR", on="element_set", set_id=50, time=0), [7.0])


def writer_entity(name: str):
    from exodusii.core.entities import entity

    return entity(name)


def _coords() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
