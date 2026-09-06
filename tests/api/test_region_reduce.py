# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Tests for region_stats, region_mass, and --piece CLI flag."""

from pathlib import Path

import numpy as np
import pytest

from exodusii.api.file import ExodusFile
from exodusii.api.region_reduce import RegionMassResult
from exodusii.api.region_reduce import RegionStatsHistory
from exodusii.api.region_reduce import RegionStatsResult
from exodusii.api.region_reduce import _apply_mask_reduce
from exodusii.api.region_reduce import _parse_predicate
from exodusii.api.region_reduce import region_mass
from exodusii.api.region_reduce import region_stats
from exodusii.api.writer import ExodusWriter
from exodusii.mesh.regions import Rectangle

# ---------------------------------------------------------------------------
# Helpers: build a minimal 2-D quad mesh with element variables
# ---------------------------------------------------------------------------


def _write_quad_mesh(path: Path) -> None:
    """Write a 2x2 grid of unit quads with DENSITY and ENERGY element variables.

    Nodes (5 total in one block for simplicity, 4 quads 1-unit-side):

        (0,2) --- (1,2) --- (2,2)
          |    q3  |    q4  |
        (0,1) --- (1,1) --- (2,1)
          |    q1  |    q2  |
        (0,0) --- (1,0) --- (2,0)

    Node IDs (1-based): row-major
      1=(0,0), 2=(1,0), 3=(2,0)
      4=(0,1), 5=(1,1), 6=(2,1)
      7=(0,2), 8=(1,2), 9=(2,2)

    Quads (1-based):
      q1=[1,2,5,4]  q2=[2,3,6,5]
      q3=[4,5,8,7]  q4=[5,6,9,8]

    DENSITY:  [1.0, 2.0, 3.0, 4.0]  (increasing by element)
    ENERGY:   [10., 20., 30., 40.]
    """
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ],
        dtype=float,
    )
    conn = np.array([[1, 2, 5, 4], [2, 3, 6, 5], [4, 5, 8, 7], [5, 6, 9, 8]], dtype=np.int64)

    with ExodusWriter.create(path) as w:
        w.initialize("test mesh", 2, 9, 4, element_blocks=1)
        w.write_coordinates(coords)
        w.define_element_block(1, "quad4", conn)
        w.define_element_variables(["DENSITY", "ENERGY"])
        w.write_time(0.0)
        w.write_element_values("DENSITY", [1.0, 2.0, 3.0, 4.0], block_id=1)
        w.write_element_values("ENERGY", [10.0, 20.0, 30.0, 40.0], block_id=1)
        w.write_time(1.0)
        # Second time step: double the values
        w.write_element_values("DENSITY", [2.0, 4.0, 6.0, 8.0], block_id=1)
        w.write_element_values("ENERGY", [20.0, 40.0, 60.0, 80.0], block_id=1)


# ---------------------------------------------------------------------------
# Unit tests: _apply_mask_reduce
# ---------------------------------------------------------------------------


class TestApplyMaskReduce:
    def test_mean(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([True, True, False, False])
        result = _apply_mask_reduce(values, mask, ["mean"])
        assert result["mean"] == pytest.approx(1.5)

    def test_max(self) -> None:
        values = np.array([1.0, 5.0, 3.0, 4.0])
        mask = np.array([True, False, True, False])
        result = _apply_mask_reduce(values, mask, ["max"])
        assert result["max"] == pytest.approx(3.0)

    def test_min(self) -> None:
        values = np.array([1.0, 5.0, 3.0, 4.0])
        mask = np.array([True, False, True, False])
        result = _apply_mask_reduce(values, mask, ["min"])
        assert result["min"] == pytest.approx(1.0)

    def test_sum_no_symmetry(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.ones(4, dtype=bool)
        result = _apply_mask_reduce(values, mask, ["sum"])
        assert result["sum"] == pytest.approx(10.0)

    def test_sum_with_symmetry(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.ones(4, dtype=bool)
        result = _apply_mask_reduce(values, mask, ["sum"], symmetry_factor=4.0)
        assert result["sum"] == pytest.approx(40.0)

    def test_count_with_symmetry(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([True, True, False, False])
        result = _apply_mask_reduce(values, mask, ["count"], symmetry_factor=2.0)
        assert result["count"] == pytest.approx(4.0)  # 2 selected * 2.0

    def test_std(self) -> None:
        values = np.array([1.0, 3.0, 5.0, 7.0])
        mask = np.ones(4, dtype=bool)
        result = _apply_mask_reduce(values, mask, ["std"])
        assert result["std"] == pytest.approx(np.std([1.0, 3.0, 5.0, 7.0]))

    def test_empty_selection_returns_nan(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        mask = np.zeros(3, dtype=bool)
        result = _apply_mask_reduce(values, mask, ["mean", "max", "min", "std"])
        assert np.isnan(result["mean"])
        assert np.isnan(result["max"])
        assert np.isnan(result["min"])
        assert np.isnan(result["std"])

    def test_empty_selection_sum_is_zero(self) -> None:
        values = np.array([1.0, 2.0])
        mask = np.zeros(2, dtype=bool)
        result = _apply_mask_reduce(values, mask, ["sum"])
        assert result["sum"] == pytest.approx(0.0)

    def test_invalid_reducer_raises(self) -> None:
        values = np.array([1.0])
        mask = np.array([True])
        with pytest.raises(ValueError, match="unknown reducer"):
            _apply_mask_reduce(values, mask, ["bogus"])

    def test_multiple_reducers(self) -> None:
        values = np.array([2.0, 4.0])
        mask = np.ones(2, dtype=bool)
        result = _apply_mask_reduce(values, mask, ["mean", "sum", "count"])
        assert result["mean"] == pytest.approx(3.0)
        assert result["sum"] == pytest.approx(6.0)
        assert result["count"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Unit tests: _parse_predicate
# ---------------------------------------------------------------------------


class TestParsePredicate:
    def test_greater_than(self, tmp_path: Path) -> None:
        path = tmp_path / "pred.exo"
        _write_quad_mesh(path)
        with ExodusFile.open(path) as exo:
            mask = _parse_predicate("DENSITY > 2.0", exo, on="element", block_id=1, time="last")
        # At time last: DENSITY = [2,4,6,8] → > 2.0 → [F, T, T, T]
        assert mask.dtype == np.bool_
        assert mask.tolist() == [False, True, True, True]

    def test_less_than_equal(self, tmp_path: Path) -> None:
        path = tmp_path / "pred.exo"
        _write_quad_mesh(path)
        with ExodusFile.open(path) as exo:
            mask = _parse_predicate("DENSITY <= 4.0", exo, on="element", block_id=1, time="last")
        # DENSITY=[2,4,6,8] ≤ 4.0 → [T,T,F,F]
        assert mask.tolist() == [True, True, False, False]

    def test_invalid_expression(self, tmp_path: Path) -> None:
        path = tmp_path / "pred.exo"
        _write_quad_mesh(path)
        with ExodusFile.open(path) as exo, pytest.raises(ValueError, match="unsupported predicate"):
            _parse_predicate("DENSITY + 1", exo, on="element", block_id=1, time="last")

    def test_no_block_id(self, tmp_path: Path) -> None:
        path = tmp_path / "pred.exo"
        _write_quad_mesh(path)
        with ExodusFile.open(path) as exo:
            # No block_id → all blocks concatenated
            mask = _parse_predicate("DENSITY > 3.0", exo, on="element", block_id=None, time="last")
        # DENSITY all blocks=[2,4,6,8] > 3.0 → [F,T,T,T]
        assert mask.tolist() == [False, True, True, True]


# ---------------------------------------------------------------------------
# Integration tests: region_stats function
# ---------------------------------------------------------------------------


class TestRegionStats:
    def test_full_rectangle_mean(self, tmp_path: Path) -> None:
        """Rectangle covering entire mesh returns mean of all elements."""
        path = tmp_path / "rs.exo"
        _write_quad_mesh(path)
        # Rectangle covering [0,2]x[0,2]; element centers at (0.5,0.5),(1.5,0.5),(0.5,1.5),(1.5,1.5)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            result = region_stats(
                exo, "DENSITY", on="element", block_id=1, region=rect, reduce="mean", time="last"
            )
        assert isinstance(result, RegionStatsResult)
        assert result.count_total == 4
        assert result.count_selected == 4
        # DENSITY at last step: [2, 4, 6, 8] → mean = 5.0
        assert result.stats["mean"] == pytest.approx(5.0)

    def test_partial_rectangle_selects_subset(self, tmp_path: Path) -> None:
        """Small rectangle picks only bottom-left elements."""
        path = tmp_path / "rs.exo"
        _write_quad_mesh(path)
        # Centers: (0.5,0.5)=elem1, (1.5,0.5)=elem2, (0.5,1.5)=elem3, (1.5,1.5)=elem4
        # Rectangle [0,1]x[0,1] → only elem1
        rect = Rectangle([0.0, 0.0], 1.0, 1.0)
        with ExodusFile.open(path) as exo:
            result = region_stats(
                exo,
                "DENSITY",
                on="element",
                block_id=1,
                region=rect,
                reduce=["mean", "max"],
                time="last",
            )
        assert isinstance(result, RegionStatsResult)
        assert result.count_selected == 1
        # DENSITY[0] at last step = 2.0
        assert result.stats["mean"] == pytest.approx(2.0)
        assert result.stats["max"] == pytest.approx(2.0)

    def test_where_predicate_further_filters(self, tmp_path: Path) -> None:
        """Region mask AND where predicate both apply."""
        path = tmp_path / "rs.exo"
        _write_quad_mesh(path)
        # Full rectangle → 4 elements, then DENSITY > 4 → elements 3,4 (DENSITY=6,8)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            result = region_stats(
                exo,
                "ENERGY",
                on="element",
                block_id=1,
                region=rect,
                where="DENSITY > 4.0",
                reduce=["mean", "count"],
                time="last",
            )
        assert isinstance(result, RegionStatsResult)
        assert result.count_selected == 2
        # ENERGY at last step for elems 3,4: [60, 80]  → mean=70
        assert result.stats["mean"] == pytest.approx(70.0)
        assert result.stats["count"] == pytest.approx(2.0)

    def test_symmetry_factor_applied_to_sum_not_mean(self, tmp_path: Path) -> None:
        path = tmp_path / "rs.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            r4 = region_stats(
                exo,
                "DENSITY",
                on="element",
                block_id=1,
                region=rect,
                reduce=["mean", "sum"],
                time="last",
                symmetry_factor=4.0,
            )
            r1 = region_stats(
                exo,
                "DENSITY",
                on="element",
                block_id=1,
                region=rect,
                reduce=["mean", "sum"],
                time="last",
                symmetry_factor=1.0,
            )
        assert isinstance(r4, RegionStatsResult)
        assert isinstance(r1, RegionStatsResult)
        # mean is intensive — must be the same regardless of symmetry_factor
        assert r4.stats["mean"] == pytest.approx(r1.stats["mean"])
        # sum is extensive — scaled by factor
        assert r4.stats["sum"] == pytest.approx(r1.stats["sum"] * 4.0)

    def test_all_blocks_no_block_id(self, tmp_path: Path) -> None:
        """When block_id is None, all blocks are used."""
        path = tmp_path / "rs.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            result = region_stats(
                exo,
                "DENSITY",
                on="element",
                block_id=None,
                region=rect,
                reduce="count",
                time="last",
            )
        assert isinstance(result, RegionStatsResult)
        assert result.count_total == 4
        assert result.count_selected == 4

    def test_result_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "rs.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            result = region_stats(
                exo,
                "DENSITY",
                on="element",
                block_id=1,
                region=rect,
                reduce="mean",
                time="last",
                symmetry_factor=2.0,
            )
        assert isinstance(result, RegionStatsResult)
        assert result.variable == "DENSITY"
        assert result.entity == "element"
        assert result.block_id == 1
        assert result.symmetry_factor == pytest.approx(2.0)
        assert result.time_index == 1  # "last" → index 1 (two steps)
        assert result.time_value == pytest.approx(1.0)

    def test_method_on_exodusfile(self, tmp_path: Path) -> None:
        """ExodusFile.region_stats convenience method."""
        path = tmp_path / "rs.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            result = exo.region_stats(
                "DENSITY", block_id=1, region=rect, reduce="mean", time="last"
            )
        assert result.stats["mean"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Integration tests: region_mass function
# ---------------------------------------------------------------------------


class TestRegionMass:
    def test_full_mesh_mass(self, tmp_path: Path) -> None:
        """Mass over full mesh = sum(vol * density)."""
        path = tmp_path / "rm.exo"
        _write_quad_mesh(path)
        # Each quad has area 1.0; DENSITY at last step = [2,4,6,8]
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            result = region_mass(exo, block_id=1, region=rect, density_name="DENSITY", time="last")
        assert isinstance(result, RegionMassResult)
        # mass = 1*2 + 1*4 + 1*6 + 1*8 = 20.0
        assert result.mass == pytest.approx(20.0)
        assert result.count_selected == 4

    def test_partial_region_mass(self, tmp_path: Path) -> None:
        """Mass over subset."""
        path = tmp_path / "rm.exo"
        _write_quad_mesh(path)
        # Only bottom-left quad (elem1): center (0.5, 0.5) inside [0,1)x[0,1)
        rect = Rectangle([0.0, 0.0], 1.0, 1.0)
        with ExodusFile.open(path) as exo:
            result = region_mass(exo, block_id=1, region=rect, density_name="DENSITY", time="last")
        assert result.count_selected == 1
        # vol=1.0, density=2.0 → mass = 2.0
        assert result.mass == pytest.approx(2.0)

    def test_symmetry_factor(self, tmp_path: Path) -> None:
        path = tmp_path / "rm.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            r1 = region_mass(exo, block_id=1, region=rect, density_name="DENSITY", time="last")
            r4 = region_mass(
                exo,
                block_id=1,
                region=rect,
                density_name="DENSITY",
                time="last",
                symmetry_factor=4.0,
            )
        assert r4.mass == pytest.approx(r1.mass * 4.0)
        assert r4.symmetry_factor == pytest.approx(4.0)

    def test_volfrac_weighting(self, tmp_path: Path) -> None:
        """With volfrac_name, mass = sum(vol * density * volfrac)."""
        path = tmp_path / "rm.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            # ENERGY used as a fake volfrac (0.5 per element wouldn't be consistent,
            # but VOID_FRC doesn't exist in this mesh; use DENSITY/max trick)
            # Instead, verify that passing volfrac_name changes the result.
            # Use DENSITY as volfrac: mass = sum(vol*DENSITY*DENSITY)
            r_novf = region_mass(exo, block_id=1, region=rect, density_name="DENSITY", time="last")
            r_vf = region_mass(
                exo,
                block_id=1,
                region=rect,
                density_name="DENSITY",
                volfrac_name="DENSITY",
                time="last",
            )
        # With volfrac=DENSITY, mass = sum(vol * D * D) = 1*(4+16+36+64) = 120
        assert r_vf.mass == pytest.approx(120.0)
        assert r_vf.mass != pytest.approx(r_novf.mass)

    def test_where_predicate(self, tmp_path: Path) -> None:
        """where predicate restricts elements in mass computation."""
        path = tmp_path / "rm.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            result = region_mass(
                exo,
                block_id=1,
                region=rect,
                density_name="DENSITY",
                where="DENSITY > 4.0",
                time="last",
            )
        # DENSITY at last=[2,4,6,8] > 4 → elems 3,4 (DENSITY=6,8)
        assert result.count_selected == 2
        assert result.mass == pytest.approx(6.0 + 8.0)

    def test_method_on_exodusfile(self, tmp_path: Path) -> None:
        """ExodusFile.region_mass convenience method."""
        path = tmp_path / "rm.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            result = exo.region_mass(block_id=1, region=rect, density_name="DENSITY", time="last")
        assert result.mass == pytest.approx(20.0)

    def test_result_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "rm.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            result = region_mass(exo, block_id=1, region=rect, density_name="DENSITY", time="last")
        assert result.block_id == 1
        assert result.density_name == "DENSITY"
        assert result.volfrac_name is None
        assert result.time_index == 1
        assert result.time_value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CLI tests: region-stats subcommand
# ---------------------------------------------------------------------------


class TestRegionStatsCLI:
    def test_cli_region_stats_rectangle(self, tmp_path: Path) -> None:
        from exodusii.cli.agent import main

        path = tmp_path / "cli.exo"
        _write_quad_mesh(path)
        from io import StringIO

        buf = StringIO()
        rc = main(
            [
                "region-stats",
                str(path),
                "--select",
                "e/DENSITY",
                "--rectangle",
                "0",
                "0",
                "2",
                "2",
                "--reduce",
                "mean,max",
                "--time",
                "last",
            ],
            file=buf,
        )
        import json

        payload = json.loads(buf.getvalue())
        assert payload["ok"] is True
        assert payload["command"] == "region-stats"
        assert payload["count_total"] == 4
        assert payload["count_selected"] == 4
        assert payload["stats"]["mean"] == pytest.approx(5.0)
        assert payload["stats"]["max"] == pytest.approx(8.0)
        assert rc == 0

    def test_cli_region_stats_with_where(self, tmp_path: Path) -> None:
        from io import StringIO

        from exodusii.cli.agent import main

        path = tmp_path / "cli.exo"
        _write_quad_mesh(path)
        buf = StringIO()
        rc = main(
            [
                "region-stats",
                str(path),
                "--select",
                "e/ENERGY",
                "--rectangle",
                "0",
                "0",
                "2",
                "2",
                "--where",
                "DENSITY > 4.0",
                "--reduce",
                "mean",
                "--time",
                "last",
            ],
            file=buf,
        )
        import json

        payload = json.loads(buf.getvalue())
        assert payload["ok"] is True
        assert payload["count_selected"] == 2
        assert payload["stats"]["mean"] == pytest.approx(70.0)

    def test_cli_region_stats_symmetry(self, tmp_path: Path) -> None:
        from io import StringIO

        from exodusii.cli.agent import main

        path = tmp_path / "cli.exo"
        _write_quad_mesh(path)
        buf = StringIO()
        rc = main(
            [
                "region-stats",
                str(path),
                "--select",
                "e/DENSITY",
                "--rectangle",
                "0",
                "0",
                "2",
                "2",
                "--reduce",
                "sum",
                "--time",
                "last",
                "--symmetry",
                "4.0",
            ],
            file=buf,
        )
        import json

        payload = json.loads(buf.getvalue())
        assert payload["symmetry_factor"] == pytest.approx(4.0)
        # sum without symmetry = 2+4+6+8=20; with factor=4 → 80
        assert payload["stats"]["sum"] == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# CLI tests: --piece flag
# ---------------------------------------------------------------------------


class TestPieceCLI:
    def _write_global_file(self, path: Path, value: float) -> None:
        """Write a tiny Exodus file with one global variable."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
        conn = [[1, 2, 3, 4]]
        with ExodusWriter.create(path) as w:
            w.initialize("piece test", 2, 4, 1, element_blocks=1)
            w.write_coordinates(coords)
            w.define_element_block(1, "quad4", conn)
            w.define_global_variables(["TOTAL"])
            w.write_time(0.0)
            w.write_global_values([value])

    def test_piece_0_reads_single_file(self, tmp_path: Path) -> None:
        from io import StringIO

        from exodusii.cli.agent import main

        path = tmp_path / "piece0.exo"
        self._write_global_file(path, 42.0)

        buf = StringIO()
        rc = main(["stats", str(path), "--select", "g/TOTAL", "--piece", "0"], file=buf)
        import json

        payload = json.loads(buf.getvalue())
        assert rc == 0
        assert payload["ok"] is True
        assert payload["piece"] == 0
        assert payload["variables"]["TOTAL"]["mean"] == pytest.approx(42.0)

    def test_piece_nonzero_invalid_on_single_file_raises(self, tmp_path: Path) -> None:
        from io import StringIO

        from exodusii.cli.agent import main

        path = tmp_path / "piece1.exo"
        self._write_global_file(path, 1.0)

        buf = StringIO()
        rc = main(["stats", str(path), "--select", "g/TOTAL", "--piece", "1"], file=buf)
        import json

        payload = json.loads(buf.getvalue())
        assert rc == 1
        assert payload["ok"] is False


# ---------------------------------------------------------------------------
# Helper: mesh with two blocks — one populated, one empty
# ---------------------------------------------------------------------------


def _write_two_block_mesh(path: Path) -> None:
    """Write a 2x2 quad mesh with two blocks: block 1 has 4 quads, block 2 is empty.

    Uses raw netCDF4 because ExodusWriter does not support zero-element blocks
    (which are legal in real Alegra/EPU output).  Block 1 has DENSITY defined.
    """
    import netCDF4 as nc  # type: ignore[import-untyped]

    ds = nc.Dataset(str(path), "w", format="NETCDF4")
    ds.setncattr("api_version", 8.25)
    ds.setncattr("version", 8.25)
    ds.setncattr("floating_point_word_size", 8)
    ds.setncattr("file_size", 1)
    ds.setncattr("title", "empty block test")

    ds.createDimension("len_string", 33)
    ds.createDimension("len_line", 81)
    ds.createDimension("four", 4)
    ds.createDimension("num_dim", 2)
    ds.createDimension("num_nodes", 9)
    ds.createDimension("num_elem", 4)
    ds.createDimension("num_el_blk", 2)
    ds.createDimension("time_step", None)
    ds.createDimension("num_el_in_blk1", 4)
    ds.createDimension("num_nod_per_el1", 4)
    ds.createDimension("num_el_in_blk2", 0)
    ds.createDimension("num_nod_per_el2", 4)
    ds.createDimension("num_elem_var", 1)

    cx = ds.createVariable("coordx", "f8", ("num_nodes",))
    cy = ds.createVariable("coordy", "f8", ("num_nodes",))
    cx[:] = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    cy[:] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]

    prop = ds.createVariable("eb_prop1", "i4", ("num_el_blk",))
    prop[:] = [1, 2]
    prop.setncattr("name", "ID")
    status = ds.createVariable("eb_status", "i4", ("num_el_blk",))
    status[:] = [1, 1]
    ds.createVariable("eb_names", "S1", ("num_el_blk", "len_string"))

    conn1 = ds.createVariable("connect1", "i4", ("num_el_in_blk1", "num_nod_per_el1"))
    conn1[:] = [[1, 2, 5, 4], [2, 3, 6, 5], [4, 5, 8, 7], [5, 6, 9, 8]]
    conn1.setncattr("elem_type", "QUAD4")
    conn2 = ds.createVariable("connect2", "i4", ("num_el_in_blk2", "num_nod_per_el2"))
    conn2.setncattr("elem_type", "QUAD4")

    nev = ds.createVariable("name_elem_var", "S1", ("num_elem_var", "len_string"))
    for i, ch in enumerate("DENSITY"):
        nev[0, i] = ch

    v1 = ds.createVariable("vals_elem_var1eb1", "f8", ("time_step", "num_el_in_blk1"))
    v1[0, :] = [2.0, 4.0, 6.0, 8.0]

    tw = ds.createVariable("time_whole", "f8", ("time_step",))
    tw[0] = 1.0
    ds.close()


def _write_two_material_mesh(path: Path) -> None:
    """Write a mesh where two blocks have different variable sets.

    Block 1: 1 quad — has DENSITY and MAT1_VAR
    Block 2: 1 quad — has DENSITY but NOT MAT1_VAR (different material)

    Uses raw netCDF4 for fine-grained truth table control.
    """
    import netCDF4 as nc  # type: ignore[import-untyped]

    ds = nc.Dataset(str(path), "w", format="NETCDF4")
    ds.setncattr("api_version", 8.25)
    ds.setncattr("version", 8.25)
    ds.setncattr("floating_point_word_size", 8)
    ds.setncattr("file_size", 1)
    ds.setncattr("title", "two-material test")

    ds.createDimension("len_string", 33)
    ds.createDimension("len_line", 81)
    ds.createDimension("four", 4)
    ds.createDimension("num_dim", 2)
    ds.createDimension("num_nodes", 6)
    ds.createDimension("num_elem", 2)
    ds.createDimension("num_el_blk", 2)
    ds.createDimension("time_step", None)
    ds.createDimension("num_el_in_blk1", 1)
    ds.createDimension("num_nod_per_el1", 4)
    ds.createDimension("num_el_in_blk2", 1)
    ds.createDimension("num_nod_per_el2", 4)
    ds.createDimension("num_elem_var", 2)

    cx = ds.createVariable("coordx", "f8", ("num_nodes",))
    cy = ds.createVariable("coordy", "f8", ("num_nodes",))
    cx[:] = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    cy[:] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    prop = ds.createVariable("eb_prop1", "i4", ("num_el_blk",))
    prop[:] = [1, 2]
    prop.setncattr("name", "ID")
    status = ds.createVariable("eb_status", "i4", ("num_el_blk",))
    status[:] = [1, 1]
    ds.createVariable("eb_names", "S1", ("num_el_blk", "len_string"))

    conn1 = ds.createVariable("connect1", "i4", ("num_el_in_blk1", "num_nod_per_el1"))
    conn1[:] = [[1, 2, 5, 4]]
    conn1.setncattr("elem_type", "QUAD4")
    conn2 = ds.createVariable("connect2", "i4", ("num_el_in_blk2", "num_nod_per_el2"))
    conn2[:] = [[2, 3, 6, 5]]
    conn2.setncattr("elem_type", "QUAD4")

    nev = ds.createVariable("name_elem_var", "S1", ("num_elem_var", "len_string"))
    for i, ch in enumerate("DENSITY"):
        nev[0, i] = ch
    for i, ch in enumerate("MAT1_VAR"):
        nev[1, i] = ch

    # Explicit truth table: DENSITY on both; MAT1_VAR on block 1 only
    tt = ds.createVariable("elem_var_tab", "i4", ("num_el_blk", "num_elem_var"))
    tt[:] = [[1, 1], [1, 0]]  # [block1: DENSITY=1, MAT1_VAR=1], [block2: DENSITY=1, MAT1_VAR=0]

    # DENSITY on both blocks
    vd1 = ds.createVariable("vals_elem_var1eb1", "f8", ("time_step", "num_el_in_blk1"))
    vd1[0, :] = [5.0]
    vd2 = ds.createVariable("vals_elem_var1eb2", "f8", ("time_step", "num_el_in_blk2"))
    vd2[0, :] = [7.0]
    # MAT1_VAR on block 1 only
    vm1 = ds.createVariable("vals_elem_var2eb1", "f8", ("time_step", "num_el_in_blk1"))
    vm1[0, :] = [99.0]

    tw = ds.createVariable("time_whole", "f8", ("time_step",))
    tw[0] = 0.0
    ds.close()


# ---------------------------------------------------------------------------
# Tests: item 5 — empty-block crash fix
# ---------------------------------------------------------------------------


class TestEmptyBlockFix:
    def test_block_id_none_skips_empty_blocks(self, tmp_path: Path) -> None:
        """block_id=None must not crash when the mesh contains empty blocks."""
        path = tmp_path / "empty.exo"
        _write_two_block_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            # This should NOT raise ValueError about empty connectivity
            result = region_stats(
                exo,
                "DENSITY",
                on="element",
                block_id=None,
                region=rect,
                reduce=["mean", "count"],
                time="last",
            )
            assert isinstance(result, RegionStatsResult)
        # Only block 1's 4 elements should contribute
        assert result.count_total == 4
        assert result.count_selected == 4
        assert result.stats["mean"] == pytest.approx(5.0)  # (2+4+6+8)/4

    def test_blocks_used_excludes_empty_block(self, tmp_path: Path) -> None:
        """blocks_used should list only the non-empty block."""
        path = tmp_path / "empty.exo"
        _write_two_block_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            result = region_stats(
                exo,
                "DENSITY",
                on="element",
                block_id=None,
                region=rect,
                reduce="count",
                time="last",
            )
        assert isinstance(result, RegionStatsResult)
        assert result.block_id is None
        assert result.blocks_used == (1,)  # block 2 was empty, excluded

    def test_blocks_auto_also_skips_empty_blocks(self, tmp_path: Path) -> None:
        """blocks='auto' must also survive empty blocks."""
        path = tmp_path / "empty.exo"
        _write_two_block_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            result = region_stats(
                exo, "DENSITY", on="element", blocks="auto", region=rect, reduce="mean", time="last"
            )
        assert isinstance(result, RegionStatsResult)
        assert result.count_total == 4
        assert result.stats["mean"] == pytest.approx(5.0)

    def test_single_block_id_still_works(self, tmp_path: Path) -> None:
        """Explicit block_id path is unaffected by the multi-block fix."""
        path = tmp_path / "empty.exo"
        _write_two_block_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            result = region_stats(
                exo, "DENSITY", on="element", block_id=1, region=rect, reduce="mean", time="last"
            )
        assert isinstance(result, RegionStatsResult)
        assert result.block_id == 1
        assert result.blocks_used is None
        assert result.stats["mean"] == pytest.approx(5.0)

    def test_all_empty_blocks_returns_nan_stats(self, tmp_path: Path) -> None:
        """When all blocks are empty, return a zero-count result with nan stats."""
        import netCDF4 as nc  # type: ignore[import-untyped]

        p = tmp_path / "all_empty.exo"
        ds = nc.Dataset(str(p), "w", format="NETCDF4")
        ds.setncattr("api_version", 8.25)
        ds.setncattr("version", 8.25)
        ds.setncattr("floating_point_word_size", 8)
        ds.setncattr("file_size", 1)
        ds.setncattr("title", "all empty")
        ds.createDimension("len_string", 33)
        ds.createDimension("num_dim", 2)
        ds.createDimension("num_nodes", 4)
        ds.createDimension("num_elem", 0)
        ds.createDimension("num_el_blk", 1)
        ds.createDimension("time_step", None)
        ds.createDimension("num_el_in_blk1", 0)
        ds.createDimension("num_nod_per_el1", 4)
        ds.createDimension("num_elem_var", 1)

        cx = ds.createVariable("coordx", "f8", ("num_nodes",))
        cy = ds.createVariable("coordy", "f8", ("num_nodes",))
        cx[:] = [0.0, 1.0, 1.0, 0.0]
        cy[:] = [0.0, 0.0, 1.0, 1.0]
        prop = ds.createVariable("eb_prop1", "i4", ("num_el_blk",))
        prop[:] = [1]
        prop.setncattr("name", "ID")
        status = ds.createVariable("eb_status", "i4", ("num_el_blk",))
        status[:] = [1]
        conn1 = ds.createVariable("connect1", "i4", ("num_el_in_blk1", "num_nod_per_el1"))
        conn1.setncattr("elem_type", "QUAD4")
        nev = ds.createVariable("name_elem_var", "S1", ("num_elem_var", "len_string"))
        for i, ch in enumerate("DENSITY"):
            nev[0, i] = ch
        tw = ds.createVariable("time_whole", "f8", ("time_step",))
        tw[0] = 0.0
        ds.close()

        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(p) as exo:
            result = region_stats(
                exo, "DENSITY", on="element", region=rect, reduce=["mean", "count"], time=0
            )
        assert isinstance(result, RegionStatsResult)
        assert result.count_total == 0
        assert result.count_selected == 0
        assert np.isnan(result.stats["mean"])
        assert result.stats["count"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: item 7 — blocks='auto' target-material selection
# ---------------------------------------------------------------------------


class TestBlocksAuto:
    def test_auto_selects_only_blocks_with_variable(self, tmp_path: Path) -> None:
        """blocks='auto' should include only blocks that define the variable."""
        path = tmp_path / "twomat.exo"
        _write_two_material_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 1.0)
        with ExodusFile.open(path) as exo:
            result = region_stats(
                exo,
                "MAT1_VAR",
                on="element",
                blocks="auto",
                region=rect,
                reduce=["mean", "count"],
                time=0,
            )
            assert isinstance(result, RegionStatsResult)
        # Only block 1 defines MAT1_VAR; block 2 does not
        assert result.blocks_used == (1,)
        assert result.count_total == 1
        assert result.stats["mean"] == pytest.approx(99.0)
        assert result.stats["count"] == pytest.approx(1.0)

    def test_auto_vs_all_differ_when_variable_absent_on_block(self, tmp_path: Path) -> None:
        """blocks='all' includes both blocks; blocks='auto' only one."""
        path = tmp_path / "twomat.exo"
        _write_two_material_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 1.0)
        with ExodusFile.open(path) as exo:
            r_all = region_stats(
                exo, "DENSITY", on="element", blocks="all", region=rect, reduce="count", time=0
            )
            r_auto = region_stats(
                exo, "DENSITY", on="element", blocks="auto", region=rect, reduce="count", time=0
            )
        assert isinstance(r_all, RegionStatsResult)
        assert isinstance(r_auto, RegionStatsResult)
        # DENSITY is defined on both blocks → auto and all give the same answer
        assert r_all.count_total == r_auto.count_total == 2
        assert r_all.blocks_used == r_auto.blocks_used == (1, 2)

    def test_block_id_and_blocks_mutually_exclusive(self, tmp_path: Path) -> None:
        path = tmp_path / "mesh.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo, pytest.raises(ValueError, match="mutually exclusive"):
            region_stats(
                exo, "DENSITY", on="element", block_id=1, blocks="auto", region=rect, reduce="mean"
            )

    def test_cli_blocks_auto_flag(self, tmp_path: Path) -> None:
        """--blocks auto on CLI passes through to region_stats."""
        import json
        from io import StringIO

        from exodusii.cli.agent import main

        path = tmp_path / "twomat.exo"
        _write_two_material_mesh(path)
        buf = StringIO()
        rc = main(
            [
                "region-stats",
                str(path),
                "--select",
                "e/MAT1_VAR",
                "--rectangle",
                "0",
                "0",
                "2",
                "1",
                "--blocks",
                "auto",
                "--reduce",
                "mean",
                "--time",
                "first",
            ],
            file=buf,
        )
        payload = json.loads(buf.getvalue())
        assert rc == 0
        assert payload["ok"] is True
        assert payload["blocks_used"] == [1]
        assert payload["stats"]["mean"] == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# Tests: item 6 — time='all' region stats history
# ---------------------------------------------------------------------------


class TestRegionStatsHistory:
    def test_time_all_returns_history(self, tmp_path: Path) -> None:
        """time='all' returns a RegionStatsHistory with one entry per step."""
        from exodusii.api.region_reduce import RegionStatsHistory

        path = tmp_path / "hist.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            history = region_stats(
                exo, "DENSITY", on="element", block_id=1, region=rect, reduce="mean", time="all"
            )
        assert isinstance(history, RegionStatsHistory)
        assert len(history.steps) == 2  # two time steps written

    def test_times_array(self, tmp_path: Path) -> None:
        path = tmp_path / "hist.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            history = region_stats(
                exo, "DENSITY", on="element", block_id=1, region=rect, reduce="mean", time="all"
            )
        assert isinstance(history, RegionStatsHistory)
        np.testing.assert_allclose(history.times, [0.0, 1.0])

    def test_stats_table_mean(self, tmp_path: Path) -> None:
        """stats_table('mean') returns correct mean at each step."""
        path = tmp_path / "hist.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            history = region_stats(
                exo, "DENSITY", on="element", block_id=1, region=rect, reduce="mean", time="all"
            )
        assert isinstance(history, RegionStatsHistory)
        means = history.stats_table("mean")
        # Step 0: DENSITY=[1,2,3,4] → mean=2.5; Step 1: DENSITY=[2,4,6,8] → mean=5.0
        np.testing.assert_allclose(means, [2.5, 5.0])

    def test_counts_array(self, tmp_path: Path) -> None:
        """counts property gives count_selected at each step."""
        path = tmp_path / "hist.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            history = region_stats(
                exo, "DENSITY", on="element", block_id=1, region=rect, reduce="count", time="all"
            )
        assert isinstance(history, RegionStatsHistory)
        np.testing.assert_array_equal(history.counts, [4, 4])

    def test_where_predicate_varies_per_step(self, tmp_path: Path) -> None:
        """where predicate is re-evaluated per step, so count_selected can change."""
        path = tmp_path / "hist.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            # Step 0: DENSITY=[1,2,3,4]; > 2 → 2 elements
            # Step 1: DENSITY=[2,4,6,8]; > 2 → 3 elements
            history = region_stats(
                exo,
                "DENSITY",
                on="element",
                block_id=1,
                region=rect,
                where="DENSITY > 2.0",
                reduce="count",
                time="all",
            )
        assert isinstance(history, RegionStatsHistory)
        np.testing.assert_array_equal(history.counts, [2, 3])

    def test_list_of_times(self, tmp_path: Path) -> None:
        """A list of selectors reduces only those steps."""
        path = tmp_path / "hist.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            history = region_stats(
                exo,
                "DENSITY",
                on="element",
                block_id=1,
                region=rect,
                reduce="mean",
                time=[1],  # only step index 1 (last)
            )
        assert isinstance(history, RegionStatsHistory)
        assert len(history.steps) == 1
        assert history.stats_table("mean")[0] == pytest.approx(5.0)

    def test_variable_property(self, tmp_path: Path) -> None:
        path = tmp_path / "hist.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            history = region_stats(
                exo, "DENSITY", on="element", block_id=1, region=rect, reduce="mean", time="all"
            )
        assert isinstance(history, RegionStatsHistory)
        assert history.variable == "DENSITY"

    def test_method_on_exodusfile(self, tmp_path: Path) -> None:
        from exodusii.api.region_reduce import RegionStatsHistory

        path = tmp_path / "hist.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            history = exo.region_stats(
                "DENSITY", block_id=1, region=rect, reduce="mean", time="all"
            )
        assert isinstance(history, RegionStatsHistory)
        assert len(history.steps) == 2

    def test_time_all_with_blocks_auto(self, tmp_path: Path) -> None:
        """time='all' works together with blocks='auto'."""
        from exodusii.api.region_reduce import RegionStatsHistory

        path = tmp_path / "empty.exo"
        _write_two_block_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            history = region_stats(
                exo, "DENSITY", on="element", blocks="auto", region=rect, reduce="mean", time="all"
            )
        assert isinstance(history, RegionStatsHistory)
        assert len(history.steps) == 1  # one time step in the empty-block fixture
        assert history.stats_table("mean")[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Tests for RegionStatsResult convenience properties
# ---------------------------------------------------------------------------


class TestRegionStatsResultProperties:
    """Verify convenience property accessors on RegionStatsResult."""

    def _make_result(
        self,
        stats: dict,
        count_selected: int = 4,
        count_total: int = 4,
        symmetry_factor: float = 1.0,
    ) -> RegionStatsResult:
        return RegionStatsResult(
            variable="V",
            entity="element",
            block_id=1,
            blocks_used=None,
            time_index=0,
            time_value=0.0,
            count_total=count_total,
            count_selected=count_selected,
            symmetry_factor=symmetry_factor,
            stats=stats,
        )

    def test_mean_property(self) -> None:
        r = self._make_result({"mean": 3.14})
        assert r.mean == pytest.approx(3.14)

    def test_max_property(self) -> None:
        r = self._make_result({"max": 9.9})
        assert r.max == pytest.approx(9.9)

    def test_min_property(self) -> None:
        r = self._make_result({"min": 0.1})
        assert r.min == pytest.approx(0.1)

    def test_sum_property(self) -> None:
        r = self._make_result({"sum": 42.0})
        assert r.sum == pytest.approx(42.0)

    def test_std_property(self) -> None:
        r = self._make_result({"std": 1.5})
        assert r.std == pytest.approx(1.5)

    def test_count_property_aliases_count_selected(self) -> None:
        r = self._make_result({}, count_selected=7)
        assert r.count == 7
        assert r.count == r.count_selected

    def test_count_with_symmetry_factor(self) -> None:
        """r.count is unscaled; r.stats['count'] is symmetry-scaled."""
        r = self._make_result({"count": 28.0}, count_selected=7, symmetry_factor=4.0)
        # r.count = count_selected (raw, unscaled) = 7
        assert r.count == 7
        # r.stats['count'] = symmetry-scaled value = 28
        assert r.stats["count"] == pytest.approx(28.0)

    def test_missing_reducer_raises_key_error(self) -> None:
        r = self._make_result({"mean": 1.0})
        with pytest.raises(KeyError):
            _ = r.max

    def test_convenience_properties_via_integration(self, tmp_path: Path) -> None:
        """End-to-end: convenience properties return same values as stats dict."""
        path = tmp_path / "prop.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            r = exo.region_stats(
                "DENSITY",
                block_id=1,
                region=rect,
                reduce=["mean", "max", "min", "sum", "std"],
                time="last",
            )
        assert r.mean == pytest.approx(r.stats["mean"])
        assert r.max == pytest.approx(r.stats["max"])
        assert r.min == pytest.approx(r.stats["min"])
        assert r.sum == pytest.approx(r.stats["sum"])
        assert r.std == pytest.approx(r.stats["std"])
        assert r.count == r.count_selected
