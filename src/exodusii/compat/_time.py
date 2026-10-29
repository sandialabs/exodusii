# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Time-step conversion shared by the legacy compatibility adapters."""

from typing import Any

import numpy as np


def legacy_time_index(time_step: int | None) -> int | None:
    """Convert a legacy 1-based ``time_step`` to a 0-based time index.

    exodusii < 2 accepted 1-based time steps and treated *any* negative
    value as "the last step" (``time_step < 0 -> values[-1]``).  A plain
    ``time_step - 1`` would turn ``-1`` into ``-2`` (the second to last
    step), silently returning the wrong data.  ``0`` maps to ``-1`` (the last
    step) because exodusii < 2 indexed ``values[time_step - 1]``; this is kept
    for compatibility.

    Parameters
    ----------
    time_step : int or None
        1-based time step, a negative value for the last step, or ``None``.

    Returns
    -------
    int or None
        0-based index (``-1`` for the last step), or ``None`` when
        ``time_step`` is ``None``.
    """

    if time_step is None:
        return None
    step = int(time_step)
    if step < 0:
        return -1
    return step - 1


def cycle_time_index(exo: Any, cycle: int) -> int:
    """Return the 0-based time index whose cycle number equals ``cycle``.

    Mirrors exodusii < 2 ``parse_history_args``: the cycle number is read
    from the first global variable named ``cycle``, ``CYCLE``, ``nsteps`` or
    ``NSTEPS``; the requested cycle must be present, and the nearest index
    is returned.

    Parameters
    ----------
    exo : ExodusFile or ParallelExodusFile
        Modern reader exposing ``variable_names`` and ``values``.
    cycle : int
        Cycle number to locate.
    """

    names = exo.variable_names("global")
    for candidate in ("cycle", "CYCLE", "nsteps", "NSTEPS"):
        if candidate in names:
            break
    else:
        raise ValueError("Could not find cycle number variable")

    cycles = np.asarray(exo.values(candidate, on="global"), dtype=np.float64)
    if not np.any(cycles == cycle):
        raise ValueError(f"Cycle {cycle} not found")
    return int(np.abs(cycles - cycle).argmin())


__all__ = ["cycle_time_index", "legacy_time_index"]
