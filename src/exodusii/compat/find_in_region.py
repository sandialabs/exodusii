# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Legacy region data extraction (``exodusii.find_in_region`` in exodusii < 2).

Both functions return an ``OrderedDict`` keyed by **0-based cycle** whose
values are dictionaries::

    data[cycle] = {
        "cycle": cycle, "time": t,
        "X": ndarray, "Y": ndarray, ["Z": ndarray],
        "VAR1": ndarray, ...
    }

Only cycles with at least one point inside the region are present, and rows
are sorted by the first coordinate.
"""

import logging
from collections import OrderedDict
from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from exodusii.mesh.regions import unbounded_time_domain

logger = logging.getLogger(__name__)

FileLike = Any


def find_node_data_in_region(
    files: FileLike | Sequence[FileLike],
    vars: Sequence[str],
    region: Any,
    time_domain: Any = None,
    use_displaced_coords: bool = False,
) -> "OrderedDict[int, dict[str, Any]]":
    """Find node data inside a geometric region for every selected cycle.

    Parameters
    ----------
    files : str, path, legacy file, or sequence of them
        Exodus database(s) to read.  Strings/paths are opened read-only.
    vars : sequence of str
        Node variables to extract.
    region : object
        Object with ``dimension`` and ``contains(points) -> bool array``.
    time_domain : object, optional
        Object with ``contains(times) -> bool array``; defaults to all times.
    use_displaced_coords : bool, optional
        Select points using displaced coordinates at each cycle.

    Returns
    -------
    OrderedDict
        ``data[cycle]`` dictionaries as described in the module docstring.
    """

    collected: OrderedDict[int, npt.NDArray[np.float64]] = OrderedDict()
    times, cycles = None, np.empty(0, dtype=np.int64)
    dimension = 0

    for i, file in enumerate(_open_all(files)):
        logger.info("Processing file %d", i + 1)
        if times is None:
            times, cycles = _times_and_cycles(file, time_domain)

        coords = None
        mask = None
        if not use_displaced_coords:
            coords = _coords(file.get_coords())
            dimension = _check_dimension(coords, region)
            mask = np.asarray(region.contains(coords), dtype=bool)
            if not np.any(mask):
                continue

        for cycle in cycles:
            if use_displaced_coords:
                coords = _coords(file.get_coords(time_step=int(cycle) + 1))
                dimension = _check_dimension(coords, region)
                mask = np.asarray(region.contains(coords), dtype=bool)
                if not np.any(mask):
                    continue
            assert coords is not None and mask is not None
            columns = [coords[mask]]
            for var in vars:
                values = file.get_node_variable_values(var, time_step=int(cycle) + 1)
                columns.append(np.asarray(values)[mask])
            _append(collected, int(cycle), np.column_stack(columns))

    return _finalize(collected, times, dimension, vars)


def find_element_data_in_region(
    files: FileLike | Sequence[FileLike],
    vars: Sequence[str],
    region: Any,
    time_domain: Any = None,
    block_ids: Iterable[int] | None = None,
    use_displaced_coords: bool = False,
) -> "OrderedDict[int, dict[str, Any]]":
    """Find element data (at element centers) inside a region for every cycle.

    Parameters
    ----------
    files : str, path, legacy file, or sequence of them
        Exodus database(s) to read.  Strings/paths are opened read-only.
    vars : sequence of str
        Element variables to extract.
    region : object
        Object with ``dimension`` and ``contains(points) -> bool array``.
    time_domain : object, optional
        Object with ``contains(times) -> bool array``; defaults to all times.
    block_ids : iterable of int, optional
        Restrict to these element blocks (default: all blocks).
    use_displaced_coords : bool, optional
        Select elements using displaced centers at each cycle.

    Returns
    -------
    OrderedDict
        ``data[cycle]`` dictionaries as described in the module docstring.
    """

    from exodusii.extension import compute_element_centers

    requested_blocks = None if block_ids is None else [int(b) for b in block_ids]
    collected: OrderedDict[int, npt.NDArray[np.float64]] = OrderedDict()
    times, cycles = None, np.empty(0, dtype=np.int64)
    dimension = 0

    for i, file in enumerate(_open_all(files)):
        logger.info("Processing file %d", i + 1)
        if times is None:
            times, cycles = _times_and_cycles(file, time_domain)

        blocks = (
            [int(b) for b in file.get_element_block_ids()]
            if requested_blocks is None
            else requested_blocks
        )
        for block_id in blocks:
            if not file.num_elems_in_blk(block_id):
                continue

            centers = None
            mask = None
            if not use_displaced_coords:
                centers = _coords(compute_element_centers(file, block_id))
                dimension = _check_dimension(centers, region)
                mask = np.asarray(region.contains(centers), dtype=bool)
                if not np.any(mask):
                    continue

            for cycle in cycles:
                if use_displaced_coords:
                    centers = _coords(
                        compute_element_centers(file, block_id, time_step=int(cycle) + 1)
                    )
                    dimension = _check_dimension(centers, region)
                    mask = np.asarray(region.contains(centers), dtype=bool)
                    if not np.any(mask):
                        continue
                assert centers is not None and mask is not None
                columns = [centers[mask]]
                for var in vars:
                    values = file.get_element_variable_values(
                        block_id, var, time_step=int(cycle) + 1
                    )
                    columns.append(np.asarray(values)[mask])
                _append(collected, int(cycle), np.column_stack(columns))

    return _finalize(collected, times, dimension, vars)


def _open_all(files: FileLike | Sequence[FileLike]) -> list[Any]:
    from exodusii.compat.legacy_file import ExodusIIFile

    items = [files] if _is_single(files) else list(files)
    return [ExodusIIFile(f, mode="r") if isinstance(f, str | Path) else f for f in items]


def _is_single(files: Any) -> bool:
    return isinstance(files, str | Path) or hasattr(files, "get_element_block_ids")


def _times_and_cycles(file: Any, time_domain: Any) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    domain = time_domain or unbounded_time_domain()
    times = np.asarray(file.get_times())
    cycles = np.asarray(domain.contains(times)).nonzero()[0]
    return times, cycles


def _coords(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.asarray(values)


def _check_dimension(coords: npt.NDArray[Any], region: Any) -> int:
    dimension = 1 if coords.ndim == 1 else int(coords.shape[1])
    if dimension != region.dimension:
        raise ValueError("Coordinate dimension does not match region")
    return dimension


def _append(
    collected: "OrderedDict[int, npt.NDArray[np.float64]]",
    cycle: int,
    rows: npt.NDArray[np.float64],
) -> None:
    if cycle in collected:
        collected[cycle] = np.vstack((collected[cycle], rows))
    else:
        collected[cycle] = rows


def _finalize(
    collected: "OrderedDict[int, npt.NDArray[np.float64]]",
    times: npt.NDArray[Any] | None,
    dimension: int,
    vars: Sequence[str],
) -> "OrderedDict[int, dict[str, Any]]":
    result: OrderedDict[int, dict[str, Any]] = OrderedDict()
    for cycle, rows in collected.items():
        order = np.argsort(rows[:, 0])
        assert times is not None
        cycle_data: dict[str, Any] = {"cycle": cycle, "time": times[cycle]}
        for axis, name in enumerate("XYZ"[:dimension]):
            cycle_data[name] = rows[order, axis]
        for column, var in enumerate(vars, start=dimension):
            cycle_data[var] = rows[order, column]
        result[cycle] = cycle_data
    return result


__all__ = ["find_element_data_in_region", "find_node_data_in_region"]
