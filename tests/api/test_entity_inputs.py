# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Public entry points accept string entity names and modern reader objects."""

from pathlib import Path

import numpy as np
import pytest

import exodusii
from exodusii.api.file import ExodusFile
from exodusii.api.parallel import ParallelExodusFile
from exodusii.api.writer import ExodusWriter
from exodusii.core.errors import ExodusInvalidEntityError
from exodusii.mesh.regions import rectangle

COORDS = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=float)


def _write(path: Path) -> Path:
    with ExodusWriter.create(path) as w:
        w.initialize("t", 2, node_count=6, element_count=2, element_blocks=2, node_sets=1)
        w.write_coordinates(COORDS)
        w.define_block("element_block", 1, "quad4", [[1, 2, 5, 4]])
        w.define_block("element_block", 2, "quad4", [[2, 3, 6, 5]])
        w.define_set("node_set", 10, [1, 2])
        w.write_block_attributes("element_block", 1, [[3.0]], names=["T"])
        w.define_property("element_block", "MATL", [5, 6])
        w.set_block_status("element_block", 2, True)
        w.write_id_map("node", np.arange(1, 7))
        w.define_variables("element", ["DENSITY"])
        w.define_variables("global", ["G"])
        w.write_time(0.0)
        w.write_values("DENSITY", [2.0], on="element", block_id=1)
        w.write_values("DENSITY", [4.0], on="element", block_id=2)
        w.write_values("G", 1.0, on="global")
    return path


def test_writer_accepts_string_entities(tmp_path: Path) -> None:
    with ExodusFile.open(_write(tmp_path / "s.exo")) as exo:
        assert exo.attribute_values("element_block", 1, "T").tolist() == [3.0]
        assert exo.property_values("element_block", "MATL").tolist() == [5, 6]
        assert exo.values("G", on="global").tolist() == [1.0]


def test_extension_helpers_accept_modern_readers(tmp_path: Path) -> None:
    path = _write(tmp_path / "s.exo")
    with ExodusFile.open(path) as exo:
        assert np.allclose(exodusii.compute_element_volumes(exo, 1), [1.0])
        assert exodusii.compute_element_centers(exo).shape == (2, 2)
        assert exodusii.compute_node_volumes(exo).sum() == pytest.approx(2.0)
        assert exo.dimension == 2  # the caller's reader is still open

    parallel = ParallelExodusFile.open(path)
    try:
        assert np.allclose(exodusii.compute_element_volumes(parallel, 2), [1.0])
    finally:
        parallel.close()


def test_block_connectivity_accepts_object_entity(tmp_path: Path) -> None:
    with ExodusFile.open(_write(tmp_path / "s.exo")) as exo:
        assert exo.block_connectivity("element", 2).tolist() == [[2, 3, 6, 5]]
        with pytest.raises(ExodusInvalidEntityError, match="not a block entity"):
            exo.block_connectivity("node", 1)


def test_region_mass_over_all_blocks(tmp_path: Path) -> None:
    region = rectangle([-1, -1], 10, 10)
    with ExodusFile.open(_write(tmp_path / "s.exo")) as exo:
        assert exo.region_mass(block_id=1, region=region).mass == pytest.approx(2.0)
        total = exo.region_mass(region=region)
        assert total.mass == pytest.approx(6.0)
        assert total.blocks_used == (1, 2)
        assert total.count_total == 2
