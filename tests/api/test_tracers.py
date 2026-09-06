# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Tests for ExodusFile.tracer_ids() and ExodusFile.tracer() (item 4)."""

from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from exodusii.api.file import ExodusFile
from exodusii.api.writer import ExodusWriter
from exodusii.core.errors import ExodusLookupError

# ---------------------------------------------------------------------------
# Fixture: synthetic tracer file
# ---------------------------------------------------------------------------
#
# Tracers are stored as SPHERE elements (1 node per element), with ID as a
# nodal variable (float64, constant across time).  This matches Alegra output.


def _write_tracer_file(path: Path) -> None:
    """Write a minimal tracer Exodus file.

    5 tracers: IDs = [11, 12, 21, 22, 23]
    Nodal variables: ID (constant), VELX (time-varying), DROPPED (constant 0)
    2 time steps: t=0.0 and t=1.0

    VELX at t=0: [100.0, 200.0, 300.0, 400.0, 500.0]  (order matches ID order)
    VELX at t=1: [110.0, 220.0, 330.0, 440.0, 550.0]
    """
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=float,
    )
    # SPHERE elements: 1 node per element (connectivity is each node ↔ 1 element)
    conn = np.array([[1], [2], [3], [4], [5]], dtype=np.int64)

    with ExodusWriter.create(path) as w:
        w.initialize("tracer test", 3, 5, 5, element_blocks=1)
        w.write_coordinates(coords)
        w.define_element_block(1, "sphere", conn)
        w.define_node_variables(["ID", "VELX", "DROPPED"])
        w.write_time(0.0)
        w.write_node_values("ID", [11.0, 12.0, 21.0, 22.0, 23.0])
        w.write_node_values("VELX", [100.0, 200.0, 300.0, 400.0, 500.0])
        w.write_node_values("DROPPED", [0.0, 0.0, 0.0, 0.0, 0.0])
        w.write_time(1.0)
        w.write_node_values("ID", [11.0, 12.0, 21.0, 22.0, 23.0])
        w.write_node_values("VELX", [110.0, 220.0, 330.0, 440.0, 550.0])
        w.write_node_values("DROPPED", [0.0, 0.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Tests: tracer_ids()
# ---------------------------------------------------------------------------


class TestTracerIds:
    def test_returns_sorted_int64_array(self, tmp_path: Path) -> None:
        path = tmp_path / "tr.exo"
        _write_tracer_file(path)
        with ExodusFile.open(path) as exo:
            ids = exo.tracer_ids()
        assert ids.dtype == np.int64
        np.testing.assert_array_equal(ids, [11, 12, 21, 22, 23])

    def test_sorted_regardless_of_storage_order(self, tmp_path: Path) -> None:
        """IDs written in non-sorted order come back sorted."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
        conn = np.array([[1], [2]], dtype=np.int64)
        path = tmp_path / "tr_rev.exo"
        with ExodusWriter.create(path) as w:
            w.initialize("rev", 3, 2, 2, element_blocks=1)
            w.write_coordinates(coords)
            w.define_element_block(1, "sphere", conn)
            w.define_node_variables(["ID"])
            w.write_time(0.0)
            w.write_node_values("ID", [99.0, 7.0])  # deliberately reversed
        with ExodusFile.open(path) as exo:
            ids = exo.tracer_ids()
        np.testing.assert_array_equal(ids, [7, 99])

    def test_custom_id_variable(self, tmp_path: Path) -> None:
        """id_variable= lets the caller name the ID column differently."""
        coords = np.array([[0.0, 0.0, 0.0]], dtype=float)
        conn = np.array([[1]], dtype=np.int64)
        path = tmp_path / "tr_custom.exo"
        with ExodusWriter.create(path) as w:
            w.initialize("custom", 3, 1, 1, element_blocks=1)
            w.write_coordinates(coords)
            w.define_element_block(1, "sphere", conn)
            w.define_node_variables(["TRACER_ID"])
            w.write_time(0.0)
            w.write_node_values("TRACER_ID", [42.0])
        with ExodusFile.open(path) as exo:
            ids = exo.tracer_ids(id_variable="TRACER_ID")
        np.testing.assert_array_equal(ids, [42])

    def test_missing_id_variable_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "tr.exo"
        _write_tracer_file(path)
        with ExodusFile.open(path) as exo, pytest.raises(ExodusLookupError):
            exo.tracer_ids(id_variable="NO_SUCH_VAR")


# ---------------------------------------------------------------------------
# Tests: tracer()
# ---------------------------------------------------------------------------


class TestTracer:
    def test_all_tracers_at_last_step(self, tmp_path: Path) -> None:
        path = tmp_path / "tr.exo"
        _write_tracer_file(path)
        with ExodusFile.open(path) as exo:
            result = exo.tracer("VELX", time="last")
        assert set(result.keys()) == {11, 12, 21, 22, 23}
        assert result[11] == pytest.approx(110.0)
        assert result[21] == pytest.approx(330.0)
        assert result[23] == pytest.approx(550.0)

    def test_specific_ids_at_last_step(self, tmp_path: Path) -> None:
        path = tmp_path / "tr.exo"
        _write_tracer_file(path)
        with ExodusFile.open(path) as exo:
            result = exo.tracer("VELX", ids=[21, 22, 23], time="last")
        assert set(result.keys()) == {21, 22, 23}
        assert result[21] == pytest.approx(330.0)
        assert result[22] == pytest.approx(440.0)
        assert result[23] == pytest.approx(550.0)
        # IDs 11 and 12 must not appear
        assert 11 not in result
        assert 12 not in result

    def test_full_time_history(self, tmp_path: Path) -> None:
        """time=None returns ndarray of shape (n_steps,) per tracer."""
        path = tmp_path / "tr.exo"
        _write_tracer_file(path)
        with ExodusFile.open(path) as exo:
            result = exo.tracer("VELX", ids=[21], time=None)
        arr = result[21]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2,)
        np.testing.assert_allclose(arr, [300.0, 330.0])

    def test_mean_of_subset_matches_manual(self, tmp_path: Path) -> None:
        """Primary use-case: mean VELX of plug tracers 21/22/23."""
        path = tmp_path / "tr.exo"
        _write_tracer_file(path)
        with ExodusFile.open(path) as exo:
            vel = exo.tracer("VELX", ids=[21, 22, 23], time="last")
        mean_vel = float(np.mean(list(vel.values())))
        assert mean_vel == pytest.approx((330.0 + 440.0 + 550.0) / 3)

    def test_missing_id_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "tr.exo"
        _write_tracer_file(path)
        with (
            ExodusFile.open(path) as exo,
            pytest.raises(ExodusLookupError, match="tracer IDs not found"),
        ):
            exo.tracer("VELX", ids=[999], time="last")

    def test_partial_missing_raises(self, tmp_path: Path) -> None:
        """Raises if ANY requested ID is missing, not just all-missing."""
        path = tmp_path / "tr.exo"
        _write_tracer_file(path)
        with ExodusFile.open(path) as exo, pytest.raises(ExodusLookupError):
            exo.tracer("VELX", ids=[21, 999], time="last")

    def test_returns_float_not_array_at_single_step(self, tmp_path: Path) -> None:
        path = tmp_path / "tr.exo"
        _write_tracer_file(path)
        with ExodusFile.open(path) as exo:
            result = exo.tracer("VELX", ids=[11], time="last")
        assert isinstance(result[11], float)

    def test_first_step(self, tmp_path: Path) -> None:
        path = tmp_path / "tr.exo"
        _write_tracer_file(path)
        with ExodusFile.open(path) as exo:
            result = exo.tracer("VELX", ids=[11], time="first")
        assert result[11] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# CLI tests: tracers subcommand
# ---------------------------------------------------------------------------


class TestTracersCLI:
    def test_tracers_all_at_last(self, tmp_path: Path) -> None:
        from exodusii.cli.main import main

        path = tmp_path / "tr.exo"
        _write_tracer_file(path)
        buf = StringIO()
        rc = main(["tracers", str(path), "--select", "n/VELX", "--time", "last"], file=buf)
        import json

        payload = json.loads(buf.getvalue())
        assert rc == 0
        assert payload["ok"] is True
        assert payload["command"] == "tracers"
        assert set(payload["data"].keys()) == {"11", "12", "21", "22", "23"}
        assert payload["data"]["21"] == pytest.approx(330.0)

    def test_tracers_specific_ids(self, tmp_path: Path) -> None:
        from exodusii.cli.main import main

        path = tmp_path / "tr.exo"
        _write_tracer_file(path)
        buf = StringIO()
        rc = main(
            ["tracers", str(path), "--select", "n/VELX", "--ids", "21,22,23", "--time", "last"],
            file=buf,
        )
        import json

        payload = json.loads(buf.getvalue())
        assert rc == 0
        assert set(payload["data"].keys()) == {"21", "22", "23"}
        assert payload["requested_ids"] == [21, 22, 23]

    def test_tracers_missing_id_returns_error(self, tmp_path: Path) -> None:
        from exodusii.cli.main import main

        path = tmp_path / "tr.exo"
        _write_tracer_file(path)
        buf = StringIO()
        rc = main(
            ["tracers", str(path), "--select", "n/VELX", "--ids", "999", "--time", "last"], file=buf
        )
        import json

        payload = json.loads(buf.getvalue())
        assert rc == 1
        assert payload["ok"] is False
