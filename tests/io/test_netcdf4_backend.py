# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pytest

from exodusii.core.errors import ExodusDimensionError
from exodusii.core.errors import ExodusVariableError
from exodusii.core.errors import ExodusWriteError
from exodusii.core.strings import encode_fixed_width
from exodusii.io.netcdf4_backend import NetCDF4Backend
from exodusii.io.netcdf4_backend import open_netcdf4


def test_create_and_read_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "test.nc"

    with NetCDF4Backend(path, mode="w") as backend:
        backend.create_dimension("num_nodes", 4)
        backend.create_dimension("time_step", None)

        assert backend.has_dimension("num_nodes")
        assert backend.has_dimension("time_step")
        assert backend.dimension("num_nodes") == 4
        assert backend.dimension("time_step") == 0
        assert backend.dimension("missing", default=123) == 123
        assert backend.dimensions() == ("num_nodes", "time_step")

    with NetCDF4Backend(path, mode="r") as backend:
        assert backend.dimension("num_nodes") == 4
        assert backend.dimension("time_step") == 0


def test_create_and_read_numeric_variable(tmp_path: Path) -> None:
    path = tmp_path / "test.nc"

    with NetCDF4Backend(path, mode="w") as backend:
        backend.create_dimension("num_nodes", 4)
        backend.create_variable("node_num_map", int, ("num_nodes",))
        backend.write_variable("node_num_map", np.asarray([1, 2, 3, 4], dtype=np.int32))

        assert backend.has_variable("node_num_map")
        assert backend.variables() == ("node_num_map",)
        assert np.allclose(backend.variable("node_num_map"), [1, 2, 3, 4])
        assert backend.variable("missing", default="sentinel") == "sentinel"

    with NetCDF4Backend(path, mode="r") as backend:
        assert np.allclose(backend.variable("node_num_map"), [1, 2, 3, 4])


def test_create_and_read_unlimited_time_variable(tmp_path: Path) -> None:
    path = tmp_path / "test.nc"

    with NetCDF4Backend(path, mode="w") as backend:
        backend.create_dimension("time_step", None)
        backend.create_variable("time_whole", float, ("time_step",))

        backend.write_variable("time_whole", 0.0, 0)
        backend.write_variable("time_whole", 1.5, 1)
        backend.write_variable("time_whole", 2.5, 2)

        assert backend.dimension("time_step") == 3
        assert np.allclose(backend.variable("time_whole"), [0.0, 1.5, 2.5])

    with NetCDF4Backend(path, mode="r") as backend:
        assert backend.dimension("time_step") == 3
        assert np.allclose(backend.variable("time_whole"), [0.0, 1.5, 2.5])


def test_write_indexed_row(tmp_path: Path) -> None:
    path = tmp_path / "test.nc"

    with NetCDF4Backend(path, mode="w") as backend:
        backend.create_dimension("time_step", None)
        backend.create_dimension("num_glo_var", 3)
        backend.create_variable("vals_glo_var", float, ("time_step", "num_glo_var"))

        backend.write_variable("vals_glo_var", [1.0, 2.0, 3.0], 0)
        backend.write_variable("vals_glo_var", [4.0, 5.0, 6.0], 1)

        assert np.allclose(backend.variable("vals_glo_var"), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_string_variable_decodes_fixed_width_names(tmp_path: Path) -> None:
    path = tmp_path / "test.nc"

    with NetCDF4Backend(path, mode="w") as backend:
        backend.create_dimension("num_names", 2)
        backend.create_dimension("len_string", 8)
        backend.create_variable("names", str, ("num_names", "len_string"))
        backend.write_variable("names", encode_fixed_width(["alpha", "beta"], width=8))

    with NetCDF4Backend(path, mode="r") as backend:
        names = backend.variable("names")

        assert isinstance(names, np.ndarray)
        assert names.tolist() == ["alpha", "beta"]


def test_raw_variable_access(tmp_path: Path) -> None:
    path = tmp_path / "test.nc"

    with NetCDF4Backend(path, mode="w") as backend:
        backend.create_dimension("num_nodes", 1)
        backend.create_variable("node_num_map", int, ("num_nodes",))

        raw = backend.variable("node_num_map", raw=True)

        assert raw is backend.dataset.variables["node_num_map"]


def test_global_attributes(tmp_path: Path) -> None:
    path = tmp_path / "test.nc"

    with NetCDF4Backend(path, mode="w") as backend:
        backend.set_attribute("title", "example")
        backend.set_attribute("version", 5.03)

        assert backend.attribute("title") == "example"
        assert backend.attribute("version") == 5.03
        assert backend.attribute("missing", default="sentinel") == "sentinel"

    with NetCDF4Backend(path, mode="r") as backend:
        assert backend.attribute("title") == "example"


def test_variable_attributes(tmp_path: Path) -> None:
    path = tmp_path / "test.nc"

    with NetCDF4Backend(path, mode="w") as backend:
        backend.create_dimension("num_elem", 1)
        backend.create_dimension("num_nod_per_el", 4)
        backend.create_variable("connect1", int, ("num_elem", "num_nod_per_el"))
        backend.set_variable_attribute("connect1", "elem_type", "QUAD")

        assert backend.variable_attribute("connect1", "elem_type") == "QUAD"
        assert backend.variable_attribute("connect1", "missing", default="sentinel") == "sentinel"


def test_create_variable_rejects_missing_dimension(tmp_path: Path) -> None:
    path = tmp_path / "test.nc"

    with (
        NetCDF4Backend(path, mode="w") as backend,
        pytest.raises(ExodusDimensionError, match="missing dimensions"),
    ):
        backend.create_variable("x", float, ("missing",))


def test_create_variable_rejects_unsupported_dtype(tmp_path: Path) -> None:
    path = tmp_path / "test.nc"

    with NetCDF4Backend(path, mode="w") as backend:
        backend.create_dimension("n", 1)

        with pytest.raises(TypeError, match="unsupported NetCDF variable dtype"):
            backend.create_variable("x", object, ("n",))  # type: ignore[arg-type]


def test_write_variable_rejects_missing_variable(tmp_path: Path) -> None:
    path = tmp_path / "test.nc"

    with (
        NetCDF4Backend(path, mode="w") as backend,
        pytest.raises(ExodusVariableError, match="variable 'missing' not found"),
    ):
        backend.write_variable("missing", 1.0)


def test_variable_attribute_rejects_missing_variable(tmp_path: Path) -> None:
    path = tmp_path / "test.nc"

    with (
        NetCDF4Backend(path, mode="w") as backend,
        pytest.raises(ExodusVariableError, match="variable 'missing' not found"),
    ):
        backend.variable_attribute("missing", "attr")


def test_set_variable_attribute_rejects_missing_variable(tmp_path: Path) -> None:
    path = tmp_path / "test.nc"

    with (
        NetCDF4Backend(path, mode="w") as backend,
        pytest.raises(ExodusVariableError, match="variable 'missing' not found"),
    ):
        backend.set_variable_attribute("missing", "attr", "value")


def test_closed_backend_rejects_operations(tmp_path: Path) -> None:
    path = tmp_path / "test.nc"
    backend = NetCDF4Backend(path, mode="w")
    backend.close()

    with pytest.raises(ExodusWriteError, match="is closed"):
        backend.dimensions()


def test_open_netcdf4_factory(tmp_path: Path) -> None:
    path = tmp_path / "test.nc"

    with open_netcdf4(path, mode="w") as backend:
        assert isinstance(backend, NetCDF4Backend)
        assert backend.path == path
        assert backend.mode == "w"


def test_sync(tmp_path: Path) -> None:
    path = tmp_path / "test.nc"

    with NetCDF4Backend(path, mode="w") as backend:
        backend.create_dimension("n", 1)
        backend.sync()
