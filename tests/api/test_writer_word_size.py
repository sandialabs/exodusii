# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Floating-point word size handling in the writer and legacy put_init.

SEACAS defines floating-point variables with ``nc_flt_code`` (NC_FLOAT for
word size 4, NC_DOUBLE for 8).  exodusii < 2 legacy writers always produced
single-precision files; vvtk ``vdiff`` baselines depend on that rounding.
"""

from pathlib import Path

import netCDF4
import numpy as np
import pytest

import exodusii
from exodusii.api.writer import ExodusWriter

FLOAT_VARIABLES = ("time_whole", "coordx", "coordy", "vals_glo_var", "vals_nod_var1")


def _write_modern(path: Path, **init_kwargs: int) -> None:
    with ExodusWriter.create(path) as writer:
        writer.initialize("ws", 2, 4, 1, element_blocks=1, **init_kwargs)
        writer.write_coordinates(np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float))
        writer.define_element_block(1, "quad", [[1, 2, 3, 4]])
        writer.define_global_variables(["G"])
        writer.define_node_variables(["T"])
        writer.write_time(0.1)
        writer.write_global_values([1.0 / 3.0])
        writer.write_node_values("T", [0.1, 0.2, 0.3, 0.4])


def _float_dtypes(path: Path) -> dict[str, str]:
    with netCDF4.Dataset(path) as ds:
        return {name: str(ds[name].dtype) for name in FLOAT_VARIABLES if name in ds.variables}


def _word_size(path: Path) -> int:
    with netCDF4.Dataset(path) as ds:
        return int(ds.getncattr("floating_point_word_size"))


def test_writer_defaults_to_double(tmp_path: Path) -> None:
    path = tmp_path / "w8.exo"
    _write_modern(path)

    assert _word_size(path) == 8
    assert set(_float_dtypes(path).values()) == {"float64"}


def test_writer_word_size_4_stores_float32(tmp_path: Path) -> None:
    path = tmp_path / "w4.exo"
    _write_modern(path, floating_point_word_size=4)

    assert _word_size(path) == 4
    dtypes = _float_dtypes(path)
    assert set(dtypes) == set(FLOAT_VARIABLES)
    assert set(dtypes.values()) == {"float32"}

    with exodusii.ExodusFile.open(path) as exo:
        assert exo.values("G", on="global")[0] == pytest.approx(1.0 / 3.0, rel=1e-7)


def test_writer_rejects_bad_word_size(tmp_path: Path) -> None:
    with (
        ExodusWriter.create(tmp_path / "bad.exo") as writer,
        pytest.raises(ValueError, match="floating_point_word_size"),
    ):
        writer.initialize("bad", 2, 4, 1, floating_point_word_size=2)


def _write_legacy(path: Path, **kwargs: int) -> None:
    with exodusii.exo_file(path, mode="w") as exo:
        exo.put_init("legacy", 1, 0, 0, 0, 0, 0, **kwargs)
        exo.put_global_variable_params(2)
        exo.put_global_variable_names(["h", "L1_X"])
        exo.put_time(1, 1.0)
        exo.put_global_variable_values(1, [0.1, 1.196058566601690e-05])


def test_legacy_put_init_defaults_to_single_precision(tmp_path: Path) -> None:
    path = tmp_path / "legacy.exo"
    _write_legacy(path)

    assert _word_size(path) == 4
    with netCDF4.Dataset(path) as ds:
        assert ds["vals_glo_var"].dtype == np.float32
        assert ds["time_whole"].dtype == np.float32
        stored = float(ds["vals_glo_var"][0, 1])

    # Same rounding exodusii < 2 applied to vdiff norm files.
    assert stored == float(np.float32(1.196058566601690e-05))


def test_legacy_put_init_honours_explicit_word_size(tmp_path: Path) -> None:
    path = tmp_path / "legacy8.exo"
    _write_legacy(path, floating_point_word_size=8)

    assert _word_size(path) == 8
    with netCDF4.Dataset(path) as ds:
        assert ds["vals_glo_var"].dtype == np.float64


def test_legacy_put_init_keeps_seacas_name_lengths(tmp_path: Path) -> None:
    # Name dimensions follow SEACAS (len_name = NC_MAX_NAME); only float
    # precision reverts to the legacy default.
    path = tmp_path / "legacy.exo"
    _write_legacy(path)

    with netCDF4.Dataset(path) as ds:
        assert len(ds.dimensions["len_name"]) == 256
        assert len(ds.dimensions["len_string"]) == 33
        assert len(ds.dimensions["len_line"]) == 81
