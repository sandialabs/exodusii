# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Variables absent per the truth table, and block/set IDs <= 0."""

from pathlib import Path

import numpy as np
import pytest

from exodusii.api.file import ExodusFile
from exodusii.api.parallel import ParallelExodusFile
from exodusii.api.writer import ExodusWriter
from exodusii.compat.legacy_file import ExodusIIFile
from exodusii.compat.legacy_parallel import ParallelExodusIIFile

COORDS = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=float)


def _write(path: Path, block_ids=(1, 2), node_set_id=1) -> Path:
    with ExodusWriter.create(path) as w:
        w.initialize("t", 2, node_count=6, element_count=2, element_blocks=2, node_sets=1)
        w.write_coordinates(COORDS)
        w.define_element_block(block_ids[0], "quad4", [[1, 2, 5, 4]])
        w.define_element_block(block_ids[1], "quad4", [[2, 3, 6, 5]])
        w.define_node_set(node_set_id, [1, 2])
        w.define_element_variables(["E"], truth_table=[[1], [0]])
        w.define_node_set_variables(["NV"])
        w.write_time(0.0)
        w.write_element_values("E", [7.0], block_id=block_ids[0])
        w.write_node_set_values("NV", [1.0, 2.0], set_id=node_set_id)
    return path


def test_absent_block_variable_is_nan(tmp_path: Path) -> None:
    with ExodusFile.open(_write(tmp_path / "a.exo")) as exo:
        assert np.isnan(exo.values("E", on="element", block_id=2, time=0)).all()
        both = exo.values("E", on="element", time=0)
        assert both[0] == 7.0
        assert np.isnan(both[1])
        assert exo.values("E", on="element", time=None).shape == (1, 2)
        assert exo.values("E", on="element", block_id=2, time=0, fill=0.0).tolist() == [0.0]


def test_legacy_api_keeps_zero_fill(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.exo")
    exo = ExodusIIFile(path)
    try:
        assert exo.get_element_variable_values(2, "E", 1).tolist() == [0.0]
        assert exo.get_element_variable_values(None, "E", 1).tolist() == [7.0, 0.0]
    finally:
        exo.close()


def test_parallel_absent_block_variable(tmp_path: Path) -> None:
    paths = []
    for rank in range(2):
        path = tmp_path / f"p.exo.2.{rank}"
        _write(path)
        paths.append(path)
    with ParallelExodusFile.open(*paths) as pef:
        assert np.isnan(pef.values("E", on="element", block_id=2, time=0)).all()
    legacy = ParallelExodusIIFile(*[str(p) for p in paths])
    try:
        assert not np.isnan(legacy.get_element_variable_values(2, "E", 1)).any()
    finally:
        legacy.close()


@pytest.mark.parametrize(("block_ids", "set_id"), [((0, -3), -5), ((7, 0), 0)])
def test_zero_and_negative_ids_are_readable(tmp_path: Path, block_ids, set_id) -> None:
    with ExodusFile.open(_write(tmp_path / "ids.exo", block_ids, set_id)) as exo:
        assert exo.element_block_ids().tolist() == list(block_ids)
        assert exo.element_block(block_ids[0]).count == 1
        assert np.asarray(exo.node_set(set_id).nodes).tolist() == [1, 2]
        assert exo.values("E", on="element", block_id=block_ids[0], time=0).tolist() == [7.0]


def test_nfaced_block_reports_element_type(tmp_path: Path) -> None:
    import netCDF4

    path = tmp_path / "nfaced.exo"
    with ExodusWriter.create(path) as w:
        w.initialize("t", 3, node_count=8, element_count=1, element_blocks=1)
        w.write_coordinates(np.zeros((8, 3)))
        w.define_element_block(3, "hex8", [list(range(1, 9))])
    # Turn the block into an NFACED block: no connect1, type on facconn1.
    with netCDF4.Dataset(path, "a") as ds:
        ds.renameVariable("connect1", "unused_connect1")
        ds.createDimension("num_fac_per_el1", 6)
        conn = ds.createVariable("facconn1", "i4", ("num_el_in_blk1", "num_fac_per_el1"))
        conn.elem_type = "NFACED"
        conn[:] = [[1, 2, 3, 4, 5, 6]]

    with ExodusFile.open(path) as exo:
        assert exo.element_block(3).element_type == "NFACED"
