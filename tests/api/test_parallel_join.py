# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Fidelity of ``ParallelExodusFile.write`` (joining decomposed files).

Regression tests for the joined file:

* pointing side sets at the wrong elements when a block's element IDs are not
  one contiguous range,
* dropping node/element ID maps, element sets, attributes, QA/info, properties
  and node-set distribution factors,
* writing an all-ones truth table,
* failing on blocks / node sets that are empty on every processor,

and for the legacy ``get_element_variable_history`` returning another
element's data.
"""

import glob
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from exodusii.api.file import ExodusFile
from exodusii.api.parallel import ParallelExodusFile
from exodusii.api.writer import ExodusWriter
from exodusii.compat.legacy_parallel import ParallelExodusIIFile
from exodusii.core.entities import Entity

# A 1x4 strip of quads, global element IDs 10, 20, 30, 40 at x = 0..4.
# Block 1 = {10, 30}, block 2 = {20, 40}: neither block is a contiguous ID
# range.  Rank 0 owns 10 and 20, rank 1 owns 30 and 40.
ELEMENT_COLUMN = {10: 0, 20: 1, 30: 2, 40: 3}
ELEMENT_BLOCK = {10: 1, 20: 2, 30: 1, 40: 2}
RANKS = {0: [10, 20], 1: [30, 40]}


def _node_gid(i: int, j: int) -> int:
    return 100 + 2 * i + j


def _quad(column: int) -> list[int]:
    return [
        _node_gid(column, 0),
        _node_gid(column + 1, 0),
        _node_gid(column + 1, 1),
        _node_gid(column, 1),
    ]


def _write_strip(tmp_path: Path, *, empty_block: bool = False) -> list[Path]:
    """Write the two-rank strip; element variable E = element id * (step+1)."""

    paths = []
    for rank, elems in RANKS.items():
        nodes = sorted({n for e in elems for n in _quad(ELEMENT_COLUMN[e])})
        local = {gid: i + 1 for i, gid in enumerate(nodes)}
        xy = np.asarray([[(g - 100) // 2, (g - 100) % 2] for g in nodes], dtype=float)
        order = [e for b in (1, 2) for e in elems if ELEMENT_BLOCK[e] == b]
        side_elem = 20 if rank == 0 else 40

        path = tmp_path / f"strip.exo.2.{rank}"
        with ExodusWriter.create(path) as w:
            w.initialize(
                "strip",
                2,
                node_count=len(nodes),
                element_count=len(elems),
                element_blocks=3 if empty_block else 2,
                node_sets=2 if empty_block else 1,
                side_sets=1,
                element_sets=1,
            )
            w.write_coordinates(xy)
            w.write_node_id_map(nodes)
            w.write_element_id_map(order)
            for b in (1, 2):
                conn = [
                    [local[n] for n in _quad(ELEMENT_COLUMN[e])]
                    for e in order
                    if ELEMENT_BLOCK[e] == b
                ]
                w.define_element_block(b, "quad4", conn, name=f"block_{b}")
                w.write_block_attributes(
                    Entity.ELEMENT_BLOCK,
                    b,
                    [[float(e)] for e in order if ELEMENT_BLOCK[e] == b],
                    names=["EID"],
                )
            if empty_block:
                w.define_element_block(3, "quad4", np.zeros((0, 4), dtype=int))

            bottom = [local[n] for n in nodes if (n - 100) % 2 == 0]
            factors = None if rank == 1 else [0.5] * len(bottom)
            w.define_node_set(5, bottom, distribution_factors=factors, name="bottom")
            if empty_block:
                w.define_node_set(6, [])
            w.define_side_set(7, [order.index(side_elem) + 1], [1 if rank == 0 else 3])
            w.define_element_set(9, [order.index(side_elem) + 1], name="evens")
            w.define_property(Entity.ELEMENT_BLOCK, "MATL", [11, 22, 33][: 3 if empty_block else 2])
            w.write_qa_records([("CODE", "1.0", "2026/10/29", "12:00:00")])
            w.write_info_records(["decomposed strip"])

            w.define_element_variables(
                ["E", "F"], truth_table=[[1, 0], [1, 1]] + ([[0, 0]] if empty_block else [])
            )
            for step in range(2):
                w.write_time(float(step))
                for b in (1, 2):
                    ids = [e for e in order if ELEMENT_BLOCK[e] == b]
                    w.write_element_values("E", [e * (step + 1.0) for e in ids], block_id=b)
                w.write_element_values("F", [-1.0], block_id=2)
        paths.append(path)
    return paths


def _centroids(exo: ExodusFile) -> np.ndarray:
    conn = np.vstack(
        [
            exo.element_connectivity(int(b))
            for b in exo.element_block_ids()
            if exo.element_block(int(b)).count
        ]
    )
    return exo.coordinates()[conn - 1].mean(axis=1)


def test_join_side_set_and_element_set_reference_correct_elements(tmp_path: Path) -> None:
    joined = tmp_path / "joined.exo"
    with ParallelExodusFile.open(*_write_strip(tmp_path)) as pef:
        pef.write(joined)

    with ExodusFile.open(joined) as exo:
        centroids = _centroids(exo)
        elem_ids = exo.ids(Entity.ELEMENT)

        side_set = exo.side_set(7)
        assert elem_ids[np.asarray(side_set.elems) - 1].tolist() == [20, 40]
        assert np.allclose(centroids[np.asarray(side_set.elems) - 1], [[1.5, 0.5], [3.5, 0.5]])
        assert np.asarray(side_set.sides).tolist() == [1, 3]

        element_set = exo.element_set(9)
        assert elem_ids[np.asarray(element_set.entries) - 1].tolist() == [20, 40]


def test_join_writes_global_id_maps(tmp_path: Path) -> None:
    joined = tmp_path / "joined.exo"
    with ParallelExodusFile.open(*_write_strip(tmp_path)) as pef:
        pef.write(joined)

    with ExodusFile.open(joined) as exo:
        # Block-by-block order, sorted within each block (SEACAS epu).
        assert exo.ids(Entity.ELEMENT).tolist() == [10, 30, 20, 40]
        assert exo.ids(Entity.NODE).tolist() == list(range(100, 110))
        # The map is consistent with geometry: element 30 sits at x in [2, 3].
        assert np.allclose(_centroids(exo)[1], [2.5, 0.5])


def test_join_preserves_attributes_records_properties_and_truth_table(tmp_path: Path) -> None:
    joined = tmp_path / "joined.exo"
    with ParallelExodusFile.open(*_write_strip(tmp_path)) as pef:
        pef.write(joined)

    with ExodusFile.open(joined) as exo:
        assert exo.attribute_names("element_block", 1) == ("EID",)
        assert np.allclose(exo.attribute_values("element_block", 1, "EID"), [10.0, 30.0])
        assert np.allclose(exo.attribute_values("element_block", 2, "EID"), [20.0, 40.0])

        assert exo.qa_records() == (("CODE", "1.0", "2026/10/29", "12:00:00"),)
        assert exo.info_records() == ("decomposed strip",)
        assert exo.property_values("element_block", "MATL").tolist() == [11, 22]

        table = exo.variable_truth_table("element")
        assert table is not None
        assert table.tolist() == [[1, 0], [1, 1]]
        assert np.allclose(exo.values("E", on="element", block_id=1, time=1), [20.0, 60.0])
        assert np.allclose(exo.values("F", on="element", block_id=2, time=0), [-1.0, -1.0])

    with netCDF4.Dataset(joined) as ds:
        assert "vals_elem_var2eb1" not in ds.variables


def test_join_node_set_distribution_factors_default_to_one(tmp_path: Path) -> None:
    joined = tmp_path / "joined.exo"
    with ParallelExodusFile.open(*_write_strip(tmp_path)) as pef:
        # Rank 0 stores 0.5, rank 1 stores none, so epu fills 1.0.  The shared
        # node 104 takes the last processor's value, as epu does.
        node_set = pef.node_set(5)
        assert np.asarray(node_set.nodes).tolist() == [100, 102, 104, 106, 108]
        assert np.allclose(node_set.dist_facts, [0.5, 0.5, 1.0, 1.0, 1.0])  # ty: ignore[invalid-argument-type]
        pef.write(joined)

    with ExodusFile.open(joined) as exo:
        assert np.allclose(exo.node_set(5).dist_facts, [0.5, 0.5, 1.0, 1.0, 1.0])  # ty: ignore[invalid-argument-type]
    with netCDF4.Dataset(joined) as ds:
        assert ds["dist_fact_ns1"].dimensions == ("num_nod_ns1",)


def test_join_handles_blocks_and_sets_empty_on_every_processor(tmp_path: Path) -> None:
    joined = tmp_path / "joined.exo"
    with ParallelExodusFile.open(*_write_strip(tmp_path, empty_block=True)) as pef:
        assert pef.element_block_ids().tolist() == [1, 2, 3]
        assert pef.element_block(3).count == 0
        assert pef.node_set(6).count == 0
        pef.write(joined)

    with ExodusFile.open(joined) as exo:
        assert exo.element_block_ids().tolist() == [1, 2, 3]
        assert exo.element_block(3).count == 0
        assert not exo.block_is_active("element_block", 3)
        assert exo.node_set(6).count == 0
        table = exo.variable_truth_table("element")
        assert table is not None
        assert table.tolist() == [[1, 0], [1, 1], [0, 0]]


def test_legacy_element_variable_history_uses_global_id(tmp_path: Path) -> None:
    legacy = ParallelExodusIIFile(*[str(p) for p in _write_strip(tmp_path)])
    try:
        for elem_id in (10, 20, 30, 40):
            assert np.allclose(
                legacy.get_element_variable_history("E", elem_id), [elem_id, 2 * elem_id]
            )
    finally:
        legacy.close()


def test_join_matches_serial_reference(datadir: Path, tmp_path: Path) -> None:
    joined = tmp_path / "edges.exo"
    files = sorted(glob.glob(str(datadir / "edges.exo.4.*")))
    with ParallelExodusFile.open(*files) as pef:
        pef.write(joined)

    with netCDF4.Dataset(datadir / "edges.base.exo") as base, netCDF4.Dataset(joined) as out:
        for name in (
            "node_num_map",
            "elem_num_map",
            "edge_num_map",
            "face_num_map",
            "eb_prop1",
            "elem_var_tab",
            "edge_var_tab",
            "face_var_tab",
        ):
            assert np.array_equal(base[name][:], out[name][:]), name
        for index in range(1, len(base.dimensions["num_node_sets"]) + 1):
            name = f"dist_fact_ns{index}"
            assert np.allclose(base[name][:], out[name][:]), name


def test_join_keeps_large_ids(tmp_path: Path) -> None:
    big = 3_000_000_000
    paths = []
    for rank in range(2):
        path = tmp_path / f"big.exo.2.{rank}"
        with ExodusWriter.create(path, int64=True) as w:
            w.initialize("big", 1, node_count=2, element_count=1, element_blocks=1)
            w.write_coordinates([[float(rank)], [rank + 1.0]])
            w.write_node_id_map([big + rank, big + rank + 1])
            w.write_element_id_map([big + rank])
            w.define_element_block(1, "bar2", [[1, 2]])
        paths.append(path)

    joined = tmp_path / "big_joined.exo"
    with ParallelExodusFile.open(*paths) as pef:
        pef.write(joined)
    with ExodusFile.open(joined) as exo:
        assert exo.int64
        assert exo.ids(Entity.NODE).tolist() == [big, big + 1, big + 2]
        assert exo.ids(Entity.ELEMENT).tolist() == [big, big + 1]


def test_join_keeps_side_set_distribution_factors(datadir: Path, tmp_path: Path) -> None:
    # Nemesis ss_df_cnt_global is 0 in these files, but every component
    # stores dist_fact_ss2 and so does the serial reference (epu writes them).
    joined = tmp_path / "mkmesh.exo"
    files = sorted(glob.glob(str(datadir / "mkmesh.par.2.*")))
    with ParallelExodusFile.open(*files) as pef:
        pef.write(joined)

    with ExodusFile.open(datadir / "mkmesh.gen") as ref, ExodusFile.open(joined) as out:
        for set_id in ref.side_set_ids().tolist():
            expected = ref.side_set(set_id)
            actual = out.side_set(set_id)
            if expected.dist_facts is None:
                assert actual.dist_facts is None
                continue
            assert actual.dist_facts is not None
            # Compare per (element id, side), independent of storage order.
            ref_ids = ref.ids("element")[np.asarray(expected.elems) - 1]
            out_ids = out.ids("element")[np.asarray(actual.elems) - 1]
            per_side = len(expected.dist_facts) // expected.count
            ref_df = {
                (int(e), int(s)): tuple(expected.dist_facts[i * per_side : (i + 1) * per_side])
                for i, (e, s) in enumerate(zip(ref_ids, np.asarray(expected.sides), strict=True))
            }
            out_df = {
                (int(e), int(s)): tuple(actual.dist_facts[i * per_side : (i + 1) * per_side])
                for i, (e, s) in enumerate(zip(out_ids, np.asarray(actual.sides), strict=True))
            }
            assert out_df == ref_df


def test_component_files_sorted_by_numeric_rank(tmp_path: Path) -> None:
    from exodusii.api.parallel import _rank_sort_key

    names = [f"m.exo.12.{rank}" for rank in (10, 2, 0, 11, 1)]
    assert sorted(names, key=_rank_sort_key) == [f"m.exo.12.{r}" for r in (0, 1, 2, 10, 11)]
    padded = [f"m.exo.12.{rank:02d}" for rank in (10, 2, 0)]
    assert sorted(padded, key=_rank_sort_key) == sorted(padded)


def test_mismatched_step_counts_raise_consistency_error(tmp_path: Path) -> None:
    from exodusii.core.errors import ExodusConsistencyError

    paths = []
    for rank, steps in enumerate((2, 3)):
        path = tmp_path / f"s.exo.2.{rank}"
        with ExodusWriter.create(path) as w:
            w.initialize("t", 1, node_count=1, element_count=0)
            w.write_coordinates([[float(rank)]])
            for step in range(steps):
                w.write_time(float(step))
        paths.append(path)
    with pytest.raises(ExodusConsistencyError, match="number of time steps"):
        ParallelExodusFile.open(*paths)


def test_warns_when_file_count_differs_from_num_processors(datadir: Path) -> None:
    from exodusii.core.errors import ExodusConsistencyError

    files = sorted(glob.glob(str(datadir / "edges.exo.4.*")))[:3]
    with (
        pytest.warns(UserWarning, match="written for 4 processors"),
        pytest.raises(ExodusConsistencyError, match="num_nodes_global"),
    ):
        ParallelExodusFile.open(*files)
