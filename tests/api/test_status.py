# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from exodusii.api.compare import similar
from exodusii.api.copy import copy_file
from exodusii.api.file import ExodusFile
from exodusii.api.writer import ExodusWriter
from exodusii.core.entities import Entity


def test_writer_reads_block_and_set_status(tmp_path: Path) -> None:
    path = tmp_path / "status.exo"

    _write_status_file(path)

    with ExodusFile.open(path) as exo:
        assert exo.block_status("element_block").tolist() == [0]
        assert not exo.block_is_active("element_block", 10)
        assert exo.element_block_ids().tolist() == [10]
        assert exo.element_block_ids(active_only=True).tolist() == []

        assert exo.set_status("node_set").tolist() == [0]
        assert not exo.set_is_active("node_set", 100)
        assert exo.node_set_ids().tolist() == [100]
        assert exo.node_set_ids(active_only=True).tolist() == []


def test_copy_preserves_status(tmp_path: Path) -> None:
    source = tmp_path / "source.exo"
    target = tmp_path / "target.exo"

    _write_status_file(source)
    copy_file(source, target)

    with ExodusFile.open(target) as exo:
        assert exo.block_status("element_block").tolist() == [0]
        assert exo.set_status("node_set").tolist() == [0]


def test_similar_detects_status_difference(tmp_path: Path) -> None:
    active = tmp_path / "active.exo"
    inactive = tmp_path / "inactive.exo"

    _write_status_file(active, active=True)
    _write_status_file(inactive, active=False)

    with pytest.raises(ValueError, match="element_block status"):
        similar(active, inactive)


def _write_status_file(path: Path, *, active: bool = False) -> None:
    with ExodusWriter.create(path) as writer:
        writer.initialize("status", 2, 4, 1, element_blocks=1, node_sets=1)
        writer.write_coordinates([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]], active=active)
        writer.define_node_set(100, [1, 2], active=active)

        # Exercise explicit setters too.
        writer.set_block_status(Entity.ELEMENT_BLOCK, 10, active)
        writer.set_set_status(Entity.NODE_SET, 100, active)
