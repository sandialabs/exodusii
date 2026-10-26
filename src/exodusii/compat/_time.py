# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Time-step conversion shared by the legacy compatibility adapters."""


def legacy_time_index(time_step: int | None) -> int | None:
    """Convert a legacy 1-based ``time_step`` to a 0-based time index.

    exodusii < 2 accepted 1-based time steps and treated *any* negative
    value as "the last step" (``time_step < 0 -> values[-1]``).  A plain
    ``time_step - 1`` would turn ``-1`` into ``-2`` (the second to last
    step), silently returning the wrong data.

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


__all__ = ["legacy_time_index"]
