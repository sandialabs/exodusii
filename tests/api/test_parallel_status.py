# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np

from exodusii.api.parallel import ParallelExodusFile
from exodusii.api.writer import ExodusWriter
from exodusii.core.entities import Entity


def test_parallel_ignores_inactive_local_block(tmp_path: Path) -> None:
    active = tmp_path / "active.exo"
    inactive = tmp_path / "inactive.exo"

    _write_part(active, active=True, temp=[1.0, 2.0, 3.0, 4.0], energy=[10.0])
    _write_part(inactive, active=False, temp=[5.0, 6.0, 7.0, 8.0], energy=[20.0])

    with ParallelExodusFile.open(active, inactive) as exo:
        assert exo.element_block_ids().tolist() == [10]
        assert exo.element_block(10).count == 1
        assert np.allclose(exo.element_connectivity(10), [[1, 2, 3, 4]])
        assert np.allclose(exo.values("ENERGY", on="element", block=10, time=0), [10.0])


def test_parallel_ignores_inactive_local_set(tmp_path: Path) -> None:
    active = tmp_path / "active.exo"
    inactive = tmp_path / "inactive.exo"

    _write_part(active, active=True, temp=[1.0, 2.0, 3.0, 4.0], energy=[10.0])
    _write_part(inactive, active=False, temp=[5.0, 6.0, 7.0, 8.0], energy=[20.0])

    with ParallelExodusFile.open(active, inactive) as exo:
        node_set = exo.node_set(100)
        assert np.allclose(node_set.nodes, [1, 4])


def _write_part(path: Path, *, active: bool, temp: list[float], energy: list[float]) -> None:
    with ExodusWriter.create(path) as writer:
        writer.initialize("parallel status", 2, 4, 1, element_blocks=1, node_sets=1)
        writer.write_coordinates([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]], active=active)
        writer.define_node_set(100, [1, 4], active=active)

        writer.define_node_variables(["TEMP"])
        writer.define_element_variables(["ENERGY"])

        writer.write_time(0.0)
        writer.write_node_values("TEMP", temp)
        if active:
            writer.write_element_values("ENERGY", energy, block_id=10)

        writer.set_block_status(Entity.ELEMENT_BLOCK, 10, active)
        writer.set_set_status(Entity.NODE_SET, 100, active)
