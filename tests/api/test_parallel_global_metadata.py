# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pytest

from exodusii.api.parallel import ParallelExodusFile
from exodusii.api.writer import ExodusWriter
from exodusii.core.errors import ExodusConsistencyError
from exodusii.core.names import DimensionName
from exodusii.core.names import VariableName


def test_parallel_uses_global_metadata(tmp_path: Path) -> None:
    part0 = tmp_path / "part0.exo"
    part1 = tmp_path / "part1.exo"

    _write_part(part0, node_offset=0)
    _write_part(part1, node_offset=4)

    _add_global_metadata(part0)

    with ParallelExodusFile.open(part0, part1) as exo:
        assert exo.node_count == 8
        assert exo.element_count == 2
        assert exo.element_block_count == 1
        assert exo.node_set_count == 1
        assert exo.side_set_count == 1
        assert exo.element_block_ids().tolist() == [10]
        assert exo.node_set_ids().tolist() == [100]
        assert exo.side_set_ids().tolist() == [200]
        assert exo.element_block(10).count == 2
        assert exo.node_set(100).count == 4
        assert exo.side_set(200).count == 2


def test_parallel_rejects_bad_global_node_count(tmp_path: Path) -> None:
    part0 = tmp_path / "part0.exo"
    part1 = tmp_path / "part1.exo"

    _write_part(part0, node_offset=0)
    _write_part(part1, node_offset=4)

    _add_global_metadata(part0, num_nodes_global=99)

    with pytest.raises(ExodusConsistencyError, match="global node count"):
        ParallelExodusFile.open(part0, part1)


def _write_part(path: Path, *, node_offset: int) -> None:
    coords = np.asarray(
        [
            [0.0 + node_offset, 0.0],
            [1.0 + node_offset, 0.0],
            [1.0 + node_offset, 1.0],
            [0.0 + node_offset, 1.0],
        ],
        dtype=float,
    )

    with ExodusWriter.create(path) as writer:
        writer.initialize("global meta", 2, 4, 1, element_blocks=1, node_sets=1, side_sets=1)
        writer.write_coordinates(coords)
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_node_set(100, [1, 4])
        writer.define_side_set(200, [1], [2])
        writer.write_time(0.0)


def _add_global_metadata(path: Path, *, num_nodes_global: int = 8) -> None:
    from exodusii.io.netcdf4_backend import NetCDF4Backend

    with NetCDF4Backend(path, mode="a") as backend:
        backend.create_dimension(DimensionName.NUM_NODES_GLOBAL.value, num_nodes_global)
        backend.create_dimension(DimensionName.NUM_ELEMENTS_GLOBAL.value, 2)
        backend.create_dimension(DimensionName.NUM_ELEMENT_BLOCKS_GLOBAL.value, 1)
        backend.create_dimension(DimensionName.NUM_NODE_SETS_GLOBAL.value, 1)
        backend.create_dimension(DimensionName.NUM_SIDE_SETS_GLOBAL.value, 1)

        backend.create_variable(
            VariableName.ELEMENT_BLOCK_IDS_GLOBAL.value,
            int,
            (DimensionName.NUM_ELEMENT_BLOCKS_GLOBAL.value,),
        )
        backend.create_variable(
            VariableName.NODE_SET_IDS_GLOBAL.value, int, (DimensionName.NUM_NODE_SETS_GLOBAL.value,)
        )
        backend.create_variable(
            VariableName.SIDE_SET_IDS_GLOBAL.value, int, (DimensionName.NUM_SIDE_SETS_GLOBAL.value,)
        )

        backend.create_variable(
            VariableName.ELEMENT_BLOCK_COUNT_GLOBAL.value,
            int,
            (DimensionName.NUM_ELEMENT_BLOCKS_GLOBAL.value,),
        )
        backend.create_variable(
            VariableName.NODE_SET_NODE_COUNT_GLOBAL.value,
            int,
            (DimensionName.NUM_NODE_SETS_GLOBAL.value,),
        )
        backend.create_variable(
            VariableName.SIDE_SET_SIDE_COUNT_GLOBAL.value,
            int,
            (DimensionName.NUM_SIDE_SETS_GLOBAL.value,),
        )

        backend.write_variable(VariableName.ELEMENT_BLOCK_IDS_GLOBAL.value, [10])
        backend.write_variable(VariableName.NODE_SET_IDS_GLOBAL.value, [100])
        backend.write_variable(VariableName.SIDE_SET_IDS_GLOBAL.value, [200])

        backend.write_variable(VariableName.ELEMENT_BLOCK_COUNT_GLOBAL.value, [2])
        backend.write_variable(VariableName.NODE_SET_NODE_COUNT_GLOBAL.value, [4])
        backend.write_variable(VariableName.SIDE_SET_SIDE_COUNT_GLOBAL.value, [2])
