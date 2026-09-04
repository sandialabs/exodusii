# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Tests for region_stats, region_mass, and --piece CLI flag."""

from pathlib import Path

import numpy as np
import pytest

from exodusii.api.file import ExodusFile
from exodusii.api.region_reduce import (
    RegionMassResult,
    RegionStatsResult,
    _apply_mask_reduce,
    _parse_predicate,
    region_mass,
    region_stats,
)
from exodusii.api.writer import ExodusWriter
from exodusii.mesh.regions import Circle, Rectangle


# ---------------------------------------------------------------------------
# Helpers: build a minimal 2-D quad mesh with element variables
# ---------------------------------------------------------------------------

def _write_quad_mesh(path: Path) -> None:
    """Write a 2×2 grid of unit quads with DENSITY and ENERGY element variables.

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
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
            [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
            [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
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
        with ExodusFile.open(path) as exo:
            with pytest.raises(ValueError, match="unsupported predicate"):
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
                exo, "DENSITY", on="element", block_id=1, region=rect, reduce=["mean", "max"], time="last"
            )
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
                exo, "ENERGY", on="element", block_id=1, region=rect,
                where="DENSITY > 4.0", reduce=["mean", "count"], time="last",
            )
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
                exo, "DENSITY", on="element", block_id=1, region=rect,
                reduce=["mean", "sum"], time="last", symmetry_factor=4.0,
            )
            r1 = region_stats(
                exo, "DENSITY", on="element", block_id=1, region=rect,
                reduce=["mean", "sum"], time="last", symmetry_factor=1.0,
            )
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
                exo, "DENSITY", on="element", block_id=None, region=rect, reduce="count", time="last"
            )
        assert result.count_total == 4
        assert result.count_selected == 4

    def test_result_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "rs.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            result = region_stats(
                exo, "DENSITY", on="element", block_id=1, region=rect,
                reduce="mean", time="last", symmetry_factor=2.0,
            )
        assert result.variable == "DENSITY"
        assert result.entity == "element"
        assert result.block_id == 1
        assert result.symmetry_factor == pytest.approx(2.0)
        assert result.time_index == 1   # "last" → index 1 (two steps)
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
                exo, block_id=1, region=rect, density_name="DENSITY", time="last",
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
            r_novf = region_mass(
                exo, block_id=1, region=rect, density_name="DENSITY", time="last"
            )
            r_vf = region_mass(
                exo, block_id=1, region=rect, density_name="DENSITY",
                volfrac_name="DENSITY", time="last",
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
                exo, block_id=1, region=rect, density_name="DENSITY",
                where="DENSITY > 4.0", time="last",
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
            result = exo.region_mass(
                block_id=1, region=rect, density_name="DENSITY", time="last"
            )
        assert result.mass == pytest.approx(20.0)

    def test_result_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "rm.exo"
        _write_quad_mesh(path)
        rect = Rectangle([0.0, 0.0], 2.0, 2.0)
        with ExodusFile.open(path) as exo:
            result = region_mass(
                exo, block_id=1, region=rect, density_name="DENSITY", time="last"
            )
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
                "region-stats", str(path),
                "--select", "e/DENSITY",
                "--rectangle", "0", "0", "2", "2",
                "--reduce", "mean,max",
                "--time", "last",
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
                "region-stats", str(path),
                "--select", "e/ENERGY",
                "--rectangle", "0", "0", "2", "2",
                "--where", "DENSITY > 4.0",
                "--reduce", "mean",
                "--time", "last",
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
                "region-stats", str(path),
                "--select", "e/DENSITY",
                "--rectangle", "0", "0", "2", "2",
                "--reduce", "sum",
                "--time", "last",
                "--symmetry", "4.0",
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
        rc = main(
            ["stats", str(path), "--select", "g/TOTAL", "--piece", "0"],
            file=buf,
        )
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
        rc = main(
            ["stats", str(path), "--select", "g/TOTAL", "--piece", "1"],
            file=buf,
        )
        import json
        payload = json.loads(buf.getvalue())
        assert rc == 1
        assert payload["ok"] is False
