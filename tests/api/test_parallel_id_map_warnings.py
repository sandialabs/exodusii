# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Diagnostics for parallel sets whose component files lack id maps.

* Missing ``node_num_map``/``elem_num_map`` can silently corrupt the joined
  mesh (shared border nodes), so one ``UserWarning`` per parallel set is
  emitted, attributed to the caller.
* Missing ``edge_num_map``/``face_num_map`` are routine (SEACAS
  ``ex_get_id_map`` returns ``1..n`` without complaint) and are only logged
  at ``DEBUG`` level.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pytest

import exodusii
from exodusii.api.parallel import ParallelExodusFile
from exodusii.api.writer import ExodusWriter

pytestmark = pytest.mark.id_map_warnings

LOGGER = "exodusii.api.parallel"


def test_edge_map_missing_is_not_a_warning(datadir: Path, caplog) -> None:
    # The bundled noh.exo.3.* set has node/element maps but no edge_num_map.
    files = sorted(datadir.glob("noh.exo.3.*"))
    assert files

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            exodusii.File(*map(str, files))

    debug = [r for r in caplog.records if r.name == LOGGER and "edge_num_map" in r.getMessage()]
    assert len(debug) == len(files)
    assert all(r.levelno == logging.DEBUG for r in debug)


def test_edge_map_missing_silent_at_default_log_level(datadir: Path, caplog) -> None:
    files = sorted(datadir.glob("noh.exo.3.*"))

    with caplog.at_level(logging.WARNING):
        exodusii.File(*map(str, files))

    assert not [r for r in caplog.records if r.name == LOGGER]


def test_missing_node_element_maps_warn_once_per_set(tmp_path: Path) -> None:
    paths = _write_parts(tmp_path, count=5)

    with pytest.warns(UserWarning) as record:
        ParallelExodusFile.open(*paths)

    messages = [str(w.message) for w in record if issubclass(w.category, UserWarning)]
    assert len(messages) == 1
    message = messages[0]
    assert "'node_num_map' missing in 5 of 5 file(s)" in message
    assert "'elem_num_map' missing in 5 of 5 file(s)" in message
    assert "+2 more" in message
    assert "sequential fallback" in message


def test_missing_map_warning_points_at_caller(tmp_path: Path) -> None:
    paths = _write_parts(tmp_path, count=2)

    with pytest.warns(UserWarning) as record:
        exodusii.File(*map(str, paths))

    assert len(record) == 1
    assert Path(record[0].filename).resolve() == Path(__file__).resolve()


def test_partial_missing_node_map_lists_only_affected_files(tmp_path: Path) -> None:
    with_maps = _write_parts(tmp_path, count=2, prefix="a", id_maps=True)
    without = _write_parts(tmp_path, count=1, prefix="b", node_offset=8)

    with pytest.warns(UserWarning) as record:
        ParallelExodusFile.open(*with_maps, *without)

    assert len(record) == 1
    message = str(record[0].message)
    assert "missing in 1 of 3 file(s)" in message
    assert str(without[0]) in message
    assert str(with_maps[0]) not in message


def test_complete_maps_emit_nothing(tmp_path: Path) -> None:
    paths = _write_parts(tmp_path, count=2, id_maps=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with ParallelExodusFile.open(*paths) as exo:
            assert exo.node_count == 8


def _write_parts(
    root: Path, *, count: int, prefix: str = "p", id_maps: bool = False, node_offset: int = 0
) -> list[Path]:
    paths = []
    for k in range(count):
        path = root / f"{prefix}.exo.{count}.{k}"
        x0 = float(k + node_offset)
        with ExodusWriter.create(path) as writer:
            writer.initialize("p", 2, 4, 1, element_blocks=1)
            writer.write_coordinates(np.array([[x0, 0.0], [x0 + 1, 0.0], [x0 + 1, 1.0], [x0, 1.0]]))
            writer.define_element_block(1, "quad", [[1, 2, 3, 4]])
            if id_maps:
                first = 4 * (k + node_offset) + 1
                writer.write_node_id_map(np.arange(first, first + 4))
                writer.write_element_id_map(np.array([k + node_offset + 1]))
            writer.write_time(0.0)
        paths.append(path)
    return paths
