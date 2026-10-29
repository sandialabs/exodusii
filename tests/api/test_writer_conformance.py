# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Writer conformance regressions.

* ``write_values(..., on=GLOBAL)`` must write a single column, not the row.
* Integers that do not fit the on-disk type must raise, never wrap; the
  ``int64`` writer stores IDs, maps and connectivity as NC_INT64.
* Zero-entry blocks/sets are written as NULL entities (status 0, no
  dimensions), as SEACAS does.
* Non-side-set distribution factors reuse the entry-count dimension.
"""

from pathlib import Path

import netCDF4
import numpy as np
import pytest

from exodusii.api.copy import copy_file
from exodusii.api.file import ExodusFile
from exodusii.api.writer import ExodusWriter
from exodusii.core.entities import Entity
from exodusii.core.errors import ExodusWriteError
from exodusii.core.names import EX_ALL_INT64_DB


def test_write_values_global_writes_only_that_variable(tmp_path: Path) -> None:
    path = tmp_path / "globals.exo"
    with ExodusWriter.create(path) as w:
        w.initialize("t", 1, node_count=0, element_count=0)
        w.define_global_variables(["g1", "g2", "g3"])
        w.write_time(0.0)
        w.write_global_values([1.0, 2.0, 3.0])
        w.write_values("g2", 99.0, on=Entity.GLOBAL)
        w.write_values("g3", [7.0], on=Entity.GLOBAL)
        with pytest.raises(ValueError):
            w.write_values("g1", [1.0, 2.0], on=Entity.GLOBAL)

    with netCDF4.Dataset(path) as ds:
        assert ds["vals_glo_var"][:].tolist() == [[1.0, 99.0, 7.0]]


def test_int32_writer_rejects_out_of_range_ids(tmp_path: Path) -> None:
    with ExodusWriter.create(tmp_path / "i32.exo") as w:
        w.initialize("t", 1, node_count=2, element_count=1, element_blocks=1)
        w.write_coordinates([[0.0], [1.0]])
        with pytest.raises(ExodusWriteError, match="int64=True"):
            w.write_node_id_map([1, 5_000_000_001])
        with pytest.raises(ExodusWriteError, match="int64=True"):
            w.define_element_block(3_000_000_000, "bar2", [[1, 2]])

    with netCDF4.Dataset(tmp_path / "i32.exo") as ds:
        assert int(ds.getncattr("int64_status")) == 0


def test_int64_writer_round_trips_large_ids(tmp_path: Path) -> None:
    path = tmp_path / "i64.exo"
    big = 5_000_000_001
    with ExodusWriter.create(path, int64=True) as w:
        assert w.int64
        w.initialize("t", 1, node_count=2, element_count=1, element_blocks=1, node_sets=1)
        w.write_coordinates([[0.0], [1.0]])
        w.write_node_id_map([1, big])
        w.write_element_id_map([big])
        w.define_element_block(3_000_000_000, "bar2", [[1, 2]])
        w.define_node_set(4_000_000_000, [2])

    with netCDF4.Dataset(path) as ds:
        assert int(ds.getncattr("int64_status")) == EX_ALL_INT64_DB
        for name in (
            "node_num_map",
            "elem_num_map",
            "eb_prop1",
            "connect1",
            "ns_prop1",
            "node_ns1",
        ):
            assert ds[name].dtype == np.int64, name
        # Status arrays and truth tables remain NC_INT, as in SEACAS.
        assert ds["eb_status"].dtype == np.int32

    with ExodusFile.open(path) as exo:
        assert exo.int64
        assert exo.ids(Entity.NODE).tolist() == [1, big]
        assert exo.ids(Entity.ELEMENT).tolist() == [big]
        assert exo.element_block_ids().tolist() == [3_000_000_000]
        assert exo.node_set_ids().tolist() == [4_000_000_000]

    copied = tmp_path / "i64_copy.exo"
    copy_file(path, copied)
    with ExodusFile.open(copied) as exo:
        assert exo.int64
        assert exo.ids(Entity.NODE).tolist() == [1, big]


def test_zero_entry_block_and_set_are_null_entities(tmp_path: Path) -> None:
    path = tmp_path / "null.exo"
    with ExodusWriter.create(path) as w:
        w.initialize("t", 2, node_count=4, element_count=1, element_blocks=2, node_sets=2)
        w.write_coordinates(np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float))
        w.define_element_block(1, "quad4", [[1, 2, 3, 4]])
        w.define_element_block(2, "quad4", np.zeros((0, 4), dtype=int), name="empty")
        w.define_node_set(10, [])
        w.define_node_set(20, [1, 2])
        w.define_element_variables(["E"])
        w.write_time(0.0)
        w.write_element_values("E", [5.0], block_id=1)

    with netCDF4.Dataset(path) as ds:
        assert ds["eb_status"][:].tolist() == [1, 0]
        assert ds["ns_status"][:].tolist() == [0, 1]
        assert "num_el_in_blk2" not in ds.dimensions
        assert "num_nod_ns1" not in ds.dimensions
        assert ds["elem_var_tab"][:].tolist() == [[1], [0]]
        assert "vals_elem_var1eb2" not in ds.variables

    with ExodusFile.open(path) as exo:
        assert exo.element_block_ids().tolist() == [1, 2]
        assert exo.element_block(2).count == 0
        assert exo.element_block(2).name == "empty"
        assert exo.node_set(10).count == 0
        assert np.allclose(exo.values("E", on="element", block_id=1, time=0), [5.0])


def test_node_set_distribution_factors_reuse_entry_dimension(tmp_path: Path) -> None:
    path = tmp_path / "df.exo"
    with ExodusWriter.create(path) as w:
        w.initialize("t", 1, node_count=3, element_count=0, node_sets=1)
        w.write_coordinates([[0.0], [1.0], [2.0]])
        with pytest.raises(ValueError, match="distribution factor count"):
            w.define_node_set(1, [1, 2], distribution_factors=[1.0, 2.0, 3.0])

    path = tmp_path / "df_ok.exo"
    with ExodusWriter.create(path) as w:
        w.initialize("t", 1, node_count=3, element_count=0, node_sets=1)
        w.write_coordinates([[0.0], [1.0], [2.0]])
        w.define_node_set(1, [1, 2], distribution_factors=[0.5, 0.25])

    with netCDF4.Dataset(path) as ds:
        assert ds["dist_fact_ns1"].dimensions == ("num_nod_ns1",)
        assert "num_df_ns1" not in ds.dimensions
    with ExodusFile.open(path) as exo:
        assert np.allclose(exo.node_set(1).dist_facts, [0.5, 0.25])  # ty: ignore[invalid-argument-type]


def test_node_variables_without_nodes(tmp_path: Path) -> None:
    path = tmp_path / "empty_rank.exo"
    with ExodusWriter.create(path) as w:
        w.initialize("t", 2, node_count=0, element_count=0)
        w.define_node_variables(["U"])
        w.write_time(0.0)
    with ExodusFile.open(path) as exo:
        assert exo.variable_names("node") == ("U",)


def test_skipped_time_steps_read_as_nan(tmp_path: Path) -> None:
    path = tmp_path / "gap.exo"
    with ExodusWriter.create(path) as w:
        w.initialize("t", 1, node_count=2, element_count=0)
        w.write_coordinates([[0.0], [1.0]])
        w.define_node_variables(["u"])
        w.write_time(0.0)
        w.write_node_values("u", [1.0, 1.0])
        w.write_time(5.0, step=3)
        w.write_node_values("u", [3.0, 3.0])
    with ExodusFile.open(path) as exo:
        times = exo.times()
        assert times[0] == 0.0 and times[2] == 5.0 and np.isnan(times[1])
        assert np.isnan(exo.values("u", on="node", time=1)).all()


def test_rewriting_records_with_a_different_count_raises(tmp_path: Path) -> None:
    with ExodusWriter.create(tmp_path / "rec.exo") as w:
        w.initialize("t", 1, node_count=0, element_count=0)
        w.write_info_records(["a", "b"])
        w.write_info_records(["c", "d"])  # same count: overwrite is fine
        with pytest.raises(ExodusWriteError, match="info records"):
            w.write_info_records(["only one"])
        w.write_qa_records([("code", "1", "d", "t")])
        with pytest.raises(ExodusWriteError, match="QA records"):
            w.write_qa_records([("a", "1", "d", "t"), ("b", "1", "d", "t")])


def test_block_attributes_always_get_a_name_variable(tmp_path: Path) -> None:
    path = tmp_path / "attr.exo"
    with ExodusWriter.create(path) as w:
        w.initialize("t", 1, node_count=2, element_count=1, element_blocks=1)
        w.write_coordinates([[0.0], [1.0]])
        w.define_element_block(1, "bar2", [[1, 2]])
        w.write_block_attributes(Entity.ELEMENT_BLOCK, 1, [[1.0, 2.0]])
    with netCDF4.Dataset(path) as ds:
        assert ds["attrib_name1"].shape[0] == 2
    with ExodusFile.open(path) as exo:
        assert np.asarray(exo.attributes("element_block", 1)).tolist() == [[1.0, 2.0]]
