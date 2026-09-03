# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Tests for ExodusFile metadata caching."""

from pathlib import Path

import numpy as np

from exodusii.api.file import ExodusFile
from exodusii.api.writer import ExodusWriter
from exodusii.core.entities import Entity


def _write_file(path: Path) -> None:
    with ExodusWriter.create(path) as writer:
        writer.initialize("cache", 2, 4, 1, element_blocks=1)
        writer.write_coordinates(
            np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
        )
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.define_global_variables(["TM_STEP"])
        writer.define_node_variables(["TEMP"])
        writer.define_element_variables(["ENERGY"])

        writer.write_time(0.0)
        writer.write_global_values([0.0])
        writer.write_node_values("TEMP", [10.0, 20.0, 30.0, 40.0])
        writer.write_element_values("ENERGY", [0.5], block_id=10)

        writer.write_time(1.0)
        writer.write_global_values([1.0])
        writer.write_node_values("TEMP", [11.0, 21.0, 31.0, 41.0])
        writer.write_element_values("ENERGY", [1.5], block_id=10)


def test_times_cached_and_correct(tmp_path: Path) -> None:
    path = tmp_path / "cache.exo"
    _write_file(path)

    with ExodusFile.open(path) as exo:
        first = exo.times()
        second = exo.times()

        assert np.allclose(first, [0.0, 1.0])
        # Cached read returns the same object.
        assert first is second
        # Cached array is read-only to protect the cache from callers.
        assert not first.flags.writeable


def test_variable_names_cached(tmp_path: Path) -> None:
    path = tmp_path / "cache.exo"
    _write_file(path)

    with ExodusFile.open(path) as exo:
        first = exo.variable_names(Entity.NODE)
        second = exo.variable_names(Entity.NODE)

        assert first == ("TEMP",)
        assert first is second


def test_block_ids_cached_and_correct(tmp_path: Path) -> None:
    path = tmp_path / "cache.exo"
    _write_file(path)

    with ExodusFile.open(path) as exo:
        first = exo.block_ids(Entity.ELEMENT_BLOCK)
        second = exo.block_ids(Entity.ELEMENT_BLOCK)

        assert np.array_equal(first, [10])
        assert first is second
        assert not first.flags.writeable

        # active_only filtering still works off the cached base array.
        active = exo.block_ids(Entity.ELEMENT_BLOCK, active_only=True)
        assert np.array_equal(active, [10])


def test_variable_index_cached(tmp_path: Path) -> None:
    path = tmp_path / "cache.exo"
    _write_file(path)

    with ExodusFile.open(path) as exo:
        idx1 = exo._variable_index(Entity.NODE, "TEMP")
        idx2 = exo._variable_index(Entity.NODE, "TEMP")
        assert idx1 == idx2 == 1


def test_cache_cleared_on_close(tmp_path: Path) -> None:
    path = tmp_path / "cache.exo"
    _write_file(path)

    exo = ExodusFile.open(path)
    exo.times()
    assert exo._cache
    exo.close()
    assert not exo._cache


def test_values_consistent_with_cache(tmp_path: Path) -> None:
    path = tmp_path / "cache.exo"
    _write_file(path)

    # Repeated values() calls (which use cached times/variable_index) must
    # return identical, correct results.
    with ExodusFile.open(path) as exo:
        a = exo.values("TEMP", on=Entity.NODE, time="last")
        b = exo.values("TEMP", on=Entity.NODE, time="last")
        assert np.allclose(a, [11.0, 21.0, 31.0, 41.0])
        assert np.allclose(a, b)
