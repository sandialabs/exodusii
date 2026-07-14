# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Time selection utilities for Exodus databases."""

from dataclasses import dataclass
from typing import Any
from typing import Literal

import numpy as np
import numpy.typing as npt

from exodusii.core.errors import ExodusInvalidTimeError

TimeSelector = int | float | str | None
TimeDefault = Literal["first", "last"]


@dataclass(frozen=True, slots=True)
class TimeSelection:
    """Resolved Exodus time selection.

    Attributes
    ----------
    index
        Python zero-based index into the time array.
    step
        Exodus one-based time step.
    value
        Physical time value.
    requested
        Original selector requested by the caller.
    exact
        Whether the selector matched exactly. Integer index selections are exact.
        Floating-point selections are exact only if the selected time equals the
        requested value within the requested tolerance.
    """

    index: int
    step: int
    value: float
    requested: TimeSelector = None
    exact: bool = True

    @property
    def index0(self) -> int:
        """Alias for the Python zero-based index."""

        return self.index

    @property
    def step1(self) -> int:
        """Alias for the Exodus one-based time step."""

        return self.step


def resolve_time(
    times: npt.ArrayLike,
    selector: TimeSelector = None,
    *,
    default: TimeDefault = "last",
    nearest: bool = True,
    tolerance: float | None = None,
) -> TimeSelection:
    """Resolve a user time selector against an Exodus time array.

    Parameters
    ----------
    times
        Exodus time values.
    selector
        Time selector. Supported values are:

        - ``None``: use ``default``
        - ``"first"``: first time
        - ``"last"``: last time
        - ``int``: Python zero-based time index; negative indices are accepted
        - ``float``: physical time value

    default
        Selection used when ``selector`` is ``None``.
    nearest
        If true, floating-point physical times select the nearest available time.
        If false, floating-point physical times must match within ``tolerance``.
    tolerance
        Absolute tolerance used for floating-point exactness. If omitted, NumPy's
        default ``isclose`` tolerances are used for exactness checks.

    Returns
    -------
    TimeSelection
        Resolved zero-based index, one-based Exodus step, and physical time value.

    Raises
    ------
    ExodusInvalidTimeError
        If no times are available, the selector is unsupported, or the requested
        time/index is out of range.
    """

    array = _as_time_array(times)

    if selector is None:
        selector = default

    if isinstance(selector, str):
        return _resolve_time_string(array, selector)

    if _is_integer_selector(selector):
        return _resolve_time_index(array, int(selector), requested=selector)

    if _is_float_selector(selector):
        return _resolve_time_float(array, float(selector), nearest=nearest, tolerance=tolerance)

    raise ExodusInvalidTimeError(
        f"Unsupported time selector {selector!r}; expected None, 'first', 'last', int, or float"
    )


def resolve_time_index(times: npt.ArrayLike, index: int) -> TimeSelection:
    """Resolve a Python zero-based time index."""

    return _resolve_time_index(_as_time_array(times), index, requested=index)


def resolve_time_step(times: npt.ArrayLike, step: int) -> TimeSelection:
    """Resolve an Exodus one-based time step."""

    if not isinstance(step, int) or isinstance(step, bool):
        raise ExodusInvalidTimeError(f"Exodus time step must be an int, got {type(step).__name__}")
    if step < 1:
        raise ExodusInvalidTimeError("Exodus time step must be one-based and positive")

    return resolve_time_index(times, step - 1)


def nearest_time_index(times: npt.ArrayLike, value: float) -> int:
    """Return the Python zero-based index of the time nearest to ``value``."""

    array = _as_time_array(times)
    return int(np.abs(array - value).argmin())


def _as_time_array(times: npt.ArrayLike) -> npt.NDArray[np.float64]:
    array = np.asarray(times, dtype=np.float64)

    if array.ndim != 1:
        raise ExodusInvalidTimeError("times must be a one-dimensional array")

    if array.size == 0:
        raise ExodusInvalidTimeError("no time steps found")

    return array


def _resolve_time_string(times: npt.NDArray[np.float64], selector: str) -> TimeSelection:
    key = selector.strip().lower()

    if key == "first":
        index = 0
    elif key == "last":
        index = times.size - 1
    else:
        raise ExodusInvalidTimeError(
            f"Unsupported time selector {selector!r}; expected 'first' or 'last'"
        )

    return TimeSelection(
        index=index, step=index + 1, value=float(times[index]), requested=selector, exact=True
    )


def _resolve_time_index(
    times: npt.NDArray[np.float64], index: int, *, requested: Any
) -> TimeSelection:
    if index < 0:
        index = times.size + index

    if index < 0 or index >= times.size:
        raise ExodusInvalidTimeError(
            f"time index {requested!r} is out of range for {times.size} time step(s)"
        )

    return TimeSelection(
        index=index, step=index + 1, value=float(times[index]), requested=requested, exact=True
    )


def _resolve_time_float(
    times: npt.NDArray[np.float64], requested: float, *, nearest: bool, tolerance: float | None
) -> TimeSelection:
    index = nearest_time_index(times, requested)
    value = float(times[index])
    exact = _time_is_close(value, requested, tolerance=tolerance)

    if not nearest and not exact:
        raise ExodusInvalidTimeError(f"time {requested!r} was not found")

    return TimeSelection(index=index, step=index + 1, value=value, requested=requested, exact=exact)


def _time_is_close(value: float, requested: float, *, tolerance: float | None) -> bool:
    if tolerance is None:
        return bool(np.isclose(value, requested))

    if tolerance < 0:
        raise ExodusInvalidTimeError("time tolerance must be nonnegative")

    return abs(value - requested) <= tolerance


def _is_integer_selector(value: object) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool | np.bool_)


def _is_float_selector(value: object) -> bool:
    return isinstance(value, float | np.floating) and not isinstance(value, bool | np.bool_)


__all__ = [
    "TimeDefault",
    "TimeSelection",
    "TimeSelector",
    "nearest_time_index",
    "resolve_time",
    "resolve_time_index",
    "resolve_time_step",
]
