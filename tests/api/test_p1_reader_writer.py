# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Reader, writer and copy regressions (audit P1 batch).

* ``copy()`` crashing on a truth table with gaps, losing float32 precision
  and dropping element-to-edge/face connectivity;
* QA records split on whitespace;
* ``displacements(time=None)`` shape and ``coordinates(displaced=True)``;
* old-format (``file_size = 0``) combined nodal variables;
* element variables defined before all blocks;
* explicit-step ``write_time`` and the default step;
* writer global attributes and NUL-terminated strings;
* ``region_stats(where=...)`` with several blocks.
"""

from pathlib import Path

import netCDF4
import numpy as np
import pytest

from exodusii.api.copy import copy_file
from exodusii.api.file import ExodusFile
from exodusii.api.parallel import ParallelExodusFile
from exodusii.api.writer import ExodusWriter
from exodusii.core.errors import ExodusWriteError
from exodusii.mesh.regions import rectangle

QUADS = np.asarray([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=float)


def _chars(strings: list[str], width: int) -> np.ndarray:
    out = np.zeros((len(strings), width), dtype="S1")
    for row, text in enumerate(strings):
        raw = text.encode()
        out[row, : len(raw)] = np.frombuffer(raw, dtype="S1")
    return out


# ---------------------------------------------------------------------------
# copy()
# ---------------------------------------------------------------------------


def test_copy_preserves_sparse_truth_table(tmp_path: Path) -> None:
    src = tmp_path / "tt.exo"
    with ExodusWriter.create(src) as w:
        w.initialize("t", 2, node_count=6, element_count=2, element_blocks=2)
        w.write_coordinates(QUADS)
        w.define_element_block(1, "quad4", [[1, 2, 5, 4]])
        w.define_element_block(2, "quad4", [[2, 3, 6, 5]])
        w.define_element_variables(["A", "B"], truth_table=[[1, 0], [1, 1]])
        w.write_time(0.0)
        w.write_element_values("A", [1.0], block_id=1)
        w.write_element_values("A", [2.0], block_id=2)
        w.write_element_values("B", [3.0], block_id=2)

    dst = tmp_path / "tt_copy.exo"
    copy_file(src, dst)
    with ExodusFile.open(dst) as exo:
        table = exo.variable_truth_table("element")
        assert table is not None
        assert table.tolist() == [[1, 0], [1, 1]]
        assert np.allclose(exo.values("A", on="element", block_id=1, time=0), [1.0])
        assert np.allclose(exo.values("B", on="element", block_id=2, time=0), [3.0])


def test_copy_preserves_word_size_and_edge_face_connectivity(datadir: Path, tmp_path: Path) -> None:
    src = datadir / "edges.base.exo"
    dst = tmp_path / "edges_copy.exo"
    copy_file(src, dst)

    with netCDF4.Dataset(src) as a, netCDF4.Dataset(dst) as b:
        assert int(b.getncattr("floating_point_word_size")) == 4
        assert b["coordx"].dtype == np.float32
        assert set(a.variables) <= set(b.variables)
        for name in a.variables:
            if name.startswith(("edgconn", "facconn")):
                assert np.array_equal(a[name][:], b[name][:]), name


# ---------------------------------------------------------------------------
# QA records
# ---------------------------------------------------------------------------


def test_qa_records_keep_fields_with_spaces_and_empty_fields(tmp_path: Path) -> None:
    path = tmp_path / "qa.exo"
    records = [("CODE A", "1.0 beta", "2024/01/01", "12:00:00"), ("x", "", "d", "t")]
    with ExodusWriter.create(path) as w:
        w.initialize("t", 1, node_count=0, element_count=0)
        w.write_qa_records(records)

    with ExodusFile.open(path) as exo:
        assert exo.qa_records() == tuple(records)

    dst = tmp_path / "qa_copy.exo"
    copy_file(path, dst)
    with ExodusFile.open(dst) as exo:
        assert exo.qa_records() == tuple(records)


def test_qa_records_from_reference_file(datadir: Path) -> None:
    with ExodusFile.open(datadir / "noh.exo") as exo:
        assert exo.qa_records()[0][1] == "PArallel Mesh GENerator"


# ---------------------------------------------------------------------------
# Displacements
# ---------------------------------------------------------------------------


def _displaced(path: Path) -> Path:
    with ExodusWriter.create(path) as w:
        w.initialize("t", 2, node_count=3, element_count=0)
        w.write_coordinates(np.zeros((3, 2)))
        w.define_node_variables(["DISPLX", "DISPLY"])
        for step, t in enumerate((0.0, 1.0)):
            w.write_time(t)
            w.write_node_values("DISPLX", [step + 1.0] * 3)
            w.write_node_values("DISPLY", [-(step + 1.0)] * 3)
    return path


def test_displacements_shapes_and_displaced_coordinates(tmp_path: Path) -> None:
    with ExodusFile.open(_displaced(tmp_path / "d.exo")) as exo:
        history = exo.displacements()
        assert history.shape == (2, 3, 2)
        assert np.allclose(history[1], [[2.0, -2.0]] * 3)
        assert exo.displacements(time=0).shape == (3, 2)
        # time=None means the last step for displaced coordinates.
        assert np.allclose(exo.coordinates(displaced=True), [[2.0, -2.0]] * 3)
        assert np.allclose(exo.coordinates(displaced=True, time=0), [[1.0, -1.0]] * 3)


def test_parallel_displacements_shape(datadir: Path) -> None:
    files = sorted(datadir.glob("edges.exo.4.*"))
    with ParallelExodusFile.open(*files) as pef:
        n_steps = len(pef.times())
        assert pef.displacements().shape == (n_steps, pef.node_count, 3)
        assert pef.displacements(time=0).shape == (pef.node_count, 3)


# ---------------------------------------------------------------------------
# Old-format combined nodal variables
# ---------------------------------------------------------------------------


def test_reads_combined_nodal_variables(tmp_path: Path) -> None:
    path = tmp_path / "old.exo"
    with netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC") as d:
        d.setncattr("api_version", np.float32(4.0))
        d.setncattr("version", np.float32(4.0))
        d.setncattr("floating_point_word_size", np.int32(8))
        d.setncattr("file_size", np.int32(0))
        d.setncattr("title", "old")
        for name, size in (("len_string", 33), ("len_line", 81), ("four", 4), ("len_name", 33)):
            d.createDimension(name, size)
        d.createDimension("time_step", None)
        d.createDimension("num_dim", 1)
        d.createDimension("num_nodes", 3)
        d.createDimension("num_nod_var", 2)
        d.createVariable("coord", "f8", ("num_dim", "num_nodes"))[:] = [[0.0, 1.0, 2.0]]
        d.createVariable("name_nod_var", "S1", ("num_nod_var", "len_name"))[:] = _chars(
            ["T", "P"], 33
        )
        t = d.createVariable("time_whole", "f8", ("time_step",))
        v = d.createVariable("vals_nod_var", "f8", ("time_step", "num_nod_var", "num_nodes"))
        for step in range(2):
            t[step] = float(step)
            v[step] = [[step, step, step], [-1.0 - step, -2.0 - step, -3.0 - step]]

    with ExodusFile.open(path) as exo:
        assert np.allclose(exo.values("P", on="node", time=1), [-2.0, -3.0, -4.0])
        assert exo.values("T", on="node").shape == (2, 3)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def test_element_variables_require_all_blocks(tmp_path: Path) -> None:
    with ExodusWriter.create(tmp_path / "early.exo") as w:
        w.initialize("t", 1, node_count=3, element_count=2, element_blocks=2)
        w.write_coordinates([[0.0], [1.0], [2.0]])
        w.define_element_block(1, "bar2", [[1, 2]])
        with pytest.raises(ExodusWriteError, match="define all 2 element blocks"):
            w.define_element_variables(["s"])


def test_write_time_explicit_step_becomes_current(tmp_path: Path) -> None:
    path = tmp_path / "steps.exo"
    with ExodusWriter.create(path) as w:
        w.initialize("t", 1, node_count=2, element_count=0)
        w.write_coordinates([[0.0], [1.0]])
        w.define_node_variables(["u"])
        for t in (0.0, 1.0, 2.0):
            w.write_time(t)
            w.write_node_values("u", [t, t])
        assert w.write_time(0.5, step=1) == 1
        w.write_node_values("u", [-1.0, -1.0])
        # The next implicit step continues after the highest step written.
        assert w.write_time(3.0) == 4
        w.write_node_values("u", [3.0, 3.0])

    with ExodusFile.open(path) as exo:
        assert exo.times().tolist() == [0.5, 1.0, 2.0, 3.0]
        assert exo.values("u", on="node")[:, 0].tolist() == [-1.0, 1.0, 2.0, 3.0]


def test_writer_global_attributes_match_seacas(tmp_path: Path) -> None:
    path = tmp_path / "attrs.exo"
    long_name = "a_very_long_variable_name_with_more_than_32_chars"
    with ExodusWriter.create(path) as w:
        w.initialize("t", 1, node_count=1, element_count=0)
        w.write_coordinates([[0.0]])
        w.define_node_variables([long_name])

    with netCDF4.Dataset(path) as d:
        assert d.getncattr("api_version").dtype == np.float32
        assert d.getncattr("version").dtype == np.float32
        assert d.getncattr("floating_point_word_size").dtype == np.int32
        assert int(d.getncattr("int64_status")) == 0
        assert int(d.getncattr("maximum_name_length")) == len(long_name)


def test_writer_strings_are_nul_terminated(tmp_path: Path) -> None:
    path = tmp_path / "strings.exo"
    with ExodusWriter.create(path) as w:
        w.initialize("t", 1, node_count=1, element_count=0)
        w.write_coordinates([[0.0]])
        w.define_node_variables(["x" * 300])
        w.write_qa_records([("A" * 40, "b", "c", "d")])
        w.write_info_records(["L" * 100])

    with netCDF4.Dataset(path) as d:
        for name in ("name_nod_var", "qa_records", "info_records"):
            var = d[name]
            var.set_auto_chartostring(False)
            chars = np.asarray(var[:])
            assert (chars.reshape(-1, chars.shape[-1])[:, -1] == b"").all(), name
        assert int(d.getncattr("maximum_name_length")) == 255

    with ExodusFile.open(path) as exo:
        assert exo.variable_names("node") == ("x" * 255,)
        assert exo.qa_records()[0][0] == "A" * 32
        assert exo.info_records() == ("L" * 80,)


# ---------------------------------------------------------------------------
# region_stats
# ---------------------------------------------------------------------------


def test_region_stats_where_with_several_blocks(tmp_path: Path) -> None:
    path = tmp_path / "rr.exo"
    coords = np.asarray([[i, j] for i in range(5) for j in range(2)], dtype=float)

    def quad(k: int) -> list[int]:
        return [2 * k + 1, 2 * k + 3, 2 * k + 4, 2 * k + 2]

    with ExodusWriter.create(path) as w:
        w.initialize("rr", 2, node_count=10, element_count=4, element_blocks=2)
        w.write_coordinates(coords)
        w.define_element_block(1, "quad4", [quad(0), quad(1)])
        w.define_element_block(2, "quad4", [quad(2), quad(3)])
        # A only on block 2; B on both blocks.
        w.define_element_variables(["A", "B"], truth_table=[[0, 1], [1, 1]])
        w.write_time(0.0)
        w.write_element_values("A", [10.0, 20.0], block_id=2)
        w.write_element_values("B", [-1.0, -1.0], block_id=1)
        w.write_element_values("B", [1.0, -1.0], block_id=2)

    region = rectangle([-1, -1], 10, 10)
    with ExodusFile.open(path) as exo:
        single = exo.region_stats(
            "A", region=region, reduce=["mean", "count"], blocks="auto", where="B > 0"
        )
        assert single.stats == {"mean": 10.0, "count": 1.0}

        history = exo.region_stats(
            "A", region=region, reduce=["mean", "count"], blocks="auto", where="B > 0", time="all"
        )
        assert [step.stats for step in history.steps] == [{"mean": 10.0, "count": 1.0}]


def test_region_stats_history_without_eligible_blocks(tmp_path: Path) -> None:
    path = tmp_path / "none.exo"
    with ExodusWriter.create(path) as w:
        w.initialize("none", 2, node_count=6, element_count=2, element_blocks=2)
        w.write_coordinates(QUADS)
        w.define_element_block(1, "quad4", [[1, 2, 5, 4]])
        w.define_element_block(2, "quad4", [[2, 3, 6, 5]])
        w.define_element_variables(["A"], truth_table=[[0], [0]])
        w.write_time(0.0)
        w.write_time(1.0)

    region = rectangle([-1, -1], 10, 10)
    with ExodusFile.open(path) as exo:
        single = exo.region_stats("A", region=region, reduce=["sum", "mean"], blocks="auto")
        history = exo.region_stats(
            "A", region=region, reduce=["sum", "mean"], blocks="auto", time="all"
        )
    assert single.stats["sum"] == 0.0
    assert np.isnan(single.stats["mean"])
    assert len(history.steps) == 2
    for step in history.steps:
        assert step.stats["sum"] == 0.0
        assert np.isnan(step.stats["mean"])
