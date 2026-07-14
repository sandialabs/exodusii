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


def test_block_and_set_properties(tmp_path: Path) -> None:
    path = tmp_path / "props.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize("props", 2, 4, 1, element_blocks=1, node_sets=1)
        writer.write_coordinates(_coords())
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_node_set(100, [1, 2])
        writer.define_property(Entity.ELEMENT_BLOCK, "MAT", [7])
        writer.define_property(Entity.NODE_SET, "BC", [99])

    with ExodusFile.open(path) as exo:
        assert exo.property_names("element_block") == ("ID", "MAT")
        assert exo.property_names("node_set") == ("ID", "BC")
        assert np.allclose(exo.property_values("element_block", "ID"), [10])
        assert np.allclose(exo.property_values("element_block", "MAT"), [7])
        assert exo.property_value("element_block", 10, "MAT") == 7
        assert exo.property_value("node_set", 100, "BC") == 99


def test_legacy_property_methods(tmp_path: Path) -> None:
    path = tmp_path / "legacy_props.exo"

    with exodusii.File(path, mode="w") as exo:
        exo.put_init("legacy props", 2, 4, 1, 1, 1, 0)
        exo.put_coords(_coords())
        exo.put_element_block(10, "quad", 1, 4)
        exo.put_element_conn(10, [[1, 2, 3, 4]])
        exo.put_node_set_param(100, 2)
        exo.put_node_set_nodes(100, [1, 2])
        exo.put_element_property("MAT", [7])
        exo.put_node_set_property("BC", [99])

    with exodusii.File(path) as exo:
        assert exo.get_element_property_names() == ["ID", "MAT"]
        assert exo.get_node_set_property_names() == ["ID", "BC"]
        assert exo.get_element_property_value(10, "MAT") == 7
        assert exo.get_node_set_property_value(100, "BC") == 99


def test_copy_preserves_properties(tmp_path: Path) -> None:
    source = tmp_path / "source.exo"
    target = tmp_path / "target.exo"

    with ExodusWriter.create(source) as writer:
        writer.initialize("props", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(_coords())
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_property(Entity.ELEMENT_BLOCK, "MAT", [7])

    copy_file(source, target)

    with ExodusFile.open(target) as exo:
        assert exo.property_names("element_block") == ("ID", "MAT")
        assert exo.property_value("element_block", 10, "MAT") == 7


def _coords() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
