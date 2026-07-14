# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from exodusii.api.compare import similar
from exodusii.api.copy import copy_file
from exodusii.api.writer import ExodusWriter


def test_similar_extended_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.exo"
    target = tmp_path / "target.exo"

    _write_extended(source)
    copy_file(source, target)

    assert similar(source, target)


def test_similar_detects_edge_connectivity_difference(tmp_path: Path) -> None:
    file1 = tmp_path / "one.exo"
    file2 = tmp_path / "two.exo"

    _write_extended(file1)
    _write_extended(file2, edge_connectivity=[[1, 2], [2, 3], [3, 4], [1, 4]])

    with pytest.raises(ValueError, match="edge_block connectivity"):
        similar(file1, file2)


def test_similar_detects_set_difference(tmp_path: Path) -> None:
    file1 = tmp_path / "one.exo"
    file2 = tmp_path / "two.exo"

    _write_extended(file1)
    _write_extended(file2, edge_set_entries=[1, 3])

    with pytest.raises(ValueError, match=r"edge_set .* entries"):
        similar(file1, file2)


def test_similar_detects_truth_table_difference(tmp_path: Path) -> None:
    file1 = tmp_path / "one.exo"
    file2 = tmp_path / "two.exo"

    _write_extended(file1, edge_truth=[[1]])
    _write_extended(file2, edge_truth=[[0]])

    with pytest.raises(ValueError, match="edge variable truth table"):
        similar(file1, file2)


def _write_extended(
    path: Path,
    *,
    edge_connectivity: list[list[int]] | None = None,
    edge_set_entries: list[int] | None = None,
    edge_truth: list[list[int]] | None = None,
) -> None:
    edge_connectivity = edge_connectivity or [[1, 2], [2, 3], [3, 4], [4, 1]]
    edge_set_entries = edge_set_entries or [1, 2]
    edge_truth = edge_truth or [[1]]

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
        writer.write_coordinates([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_edge_block(20, "edge2", edge_connectivity)
        writer.define_face_block(30, "quad", [[1, 2, 3, 4]])

        writer.define_edge_set(40, edge_set_entries, orientations=[1] * len(edge_set_entries))
        writer.define_face_set(50, [1], orientations=[1])
        writer.define_element_set(60, [1])

        writer.define_edge_variables(["EDGEVAR"], truth_table=edge_truth)

        writer.write_time(0.0)
        if edge_truth[0][0]:
            writer.write_edge_values("EDGEVAR", [1.0, 2.0, 3.0, 4.0], block_id=20)
