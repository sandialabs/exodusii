# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Tolerance model for Exodus database comparison.

This reproduces the tolerance semantics of the SEACAS ``exodiff`` tool
(``Tolerance.C``/``Tolerance.h``) so that Python-based comparisons agree with
the reference implementation.

Each :class:`Tolerance` has a :class:`ToleranceMode`, a ``value``, and a
``floor``.  :meth:`Tolerance.diff` returns ``True`` when two scalars differ by
more than the tolerance; :meth:`Tolerance.delta` returns the (mode-dependent)
magnitude of the difference used for reporting.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["Tolerance", "ToleranceMode"]


class ToleranceMode(enum.Enum):
    """Comparison tolerance modes.

    Mirrors the SEACAS ``exodiff`` ``ToleranceMode`` enumeration.  Each mode
    determines how :class:`Tolerance` interprets its ``value`` field when
    deciding whether two scalars (or array elements) are considered different.

    Notes
    -----
    Available modes:

    RELATIVE
        Scaled difference: ``|a - b| > value * max(|a|, |b|)``.
    ABSOLUTE
        Raw difference: ``|a - b| > value``.
    COMBINED
        Relative when ``max(|a|, |b|) > 1``, otherwise absolute.
    IGNORE
        Always report values as equal; never flags a difference.
    EIGEN_RELATIVE
        Like ``RELATIVE`` but compares magnitudes: ``| |a| - |b| |``.
    EIGEN_ABSOLUTE
        Like ``ABSOLUTE`` but compares magnitudes.
    EIGEN_COMBINED
        Like ``COMBINED`` but compares magnitudes.
    ULPS_FLOAT
        Unit-in-the-last-place distance at single (32-bit) precision.
    ULPS_DOUBLE
        Unit-in-the-last-place distance at double (64-bit) precision.
    """

    RELATIVE = "relative"
    ABSOLUTE = "absolute"
    COMBINED = "combined"
    IGNORE = "ignore"
    EIGEN_RELATIVE = "eigenrel"
    EIGEN_ABSOLUTE = "eigenabs"
    EIGEN_COMBINED = "eigencom"
    ULPS_FLOAT = "ulps_float"
    ULPS_DOUBLE = "ulps_double"

    @property
    def abbreviation(self) -> str:
        """Return the three-letter abbreviation used by exodiff reports.

        Returns
        -------
        str
            A three-letter string such as ``"rel"``, ``"abs"``, or ``"com"``
            that matches the abbreviation used in SEACAS ``exodiff`` output.
        """

        return _ABBREVIATIONS[self]

    @classmethod
    def parse(cls, value: ToleranceMode | str) -> ToleranceMode:
        """Parse a mode from a name, abbreviation, or :class:`ToleranceMode`.

        Accepts the canonical enum value string (e.g. ``"relative"``), the
        three-letter abbreviation (e.g. ``"rel"``), several spelled-out aliases
        (e.g. ``"eigen_relative"``), or an existing :class:`ToleranceMode`
        instance (returned unchanged).  Lookup is case-insensitive and
        strips surrounding whitespace.

        Parameters
        ----------
        value : ToleranceMode or str
            The mode to parse.  May be an existing :class:`ToleranceMode`
            member, a canonical value string such as ``"relative"``, a
            three-letter abbreviation such as ``"rel"``, or a recognised alias
            such as ``"eigen_relative"``.

        Returns
        -------
        ToleranceMode
            The matched :class:`ToleranceMode` member.

        Raises
        ------
        ValueError
            If ``value`` is a string that does not match any known mode name,
            abbreviation, or alias.

        Examples
        --------
        >>> ToleranceMode.parse("relative")
        <ToleranceMode.RELATIVE: 'relative'>
        >>> ToleranceMode.parse("rel")
        <ToleranceMode.RELATIVE: 'relative'>
        >>> ToleranceMode.parse(ToleranceMode.ABSOLUTE)
        <ToleranceMode.ABSOLUTE: 'absolute'>
        >>> ToleranceMode.parse("eigen_relative")
        <ToleranceMode.EIGEN_RELATIVE: 'eigenrel'>
        """

        if isinstance(value, ToleranceMode):
            return value
        key = value.strip().lower()
        if key in _BY_NAME:
            return _BY_NAME[key]
        raise ValueError(f"unknown tolerance mode {value!r}")


_ABBREVIATIONS: dict[ToleranceMode, str] = {
    ToleranceMode.RELATIVE: "rel",
    ToleranceMode.ABSOLUTE: "abs",
    ToleranceMode.COMBINED: "com",
    ToleranceMode.IGNORE: "ign",
    ToleranceMode.EIGEN_RELATIVE: "ere",
    ToleranceMode.EIGEN_ABSOLUTE: "eab",
    ToleranceMode.EIGEN_COMBINED: "eco",
    ToleranceMode.ULPS_FLOAT: "upf",
    ToleranceMode.ULPS_DOUBLE: "upd",
}

# Accept full names, abbreviations, and a few aliases.
_BY_NAME: dict[str, ToleranceMode] = {}
for _mode in ToleranceMode:
    _BY_NAME[_mode.value] = _mode
    _BY_NAME[_ABBREVIATIONS[_mode]] = _mode
_BY_NAME["eigen_relative"] = ToleranceMode.EIGEN_RELATIVE
_BY_NAME["eigen_absolute"] = ToleranceMode.EIGEN_ABSOLUTE
_BY_NAME["eigen_combined"] = ToleranceMode.EIGEN_COMBINED


def _ulps_distance(a: float, b: float, *, dtype: str) -> float:
    """Return the ULPs distance between ``a`` and ``b`` at the given precision.

    Reproduces exodiff's ``UlpsDiff*``: values of opposite sign are considered
    maximally different (``2 << 28``) unless exactly equal; otherwise the
    distance is the absolute difference of the integer bit patterns.
    """

    if dtype == "float":
        pack, ipack = "<f", "<i"
    else:
        pack, ipack = "<d", "<q"

    fa = np.asarray(a, dtype=np.float32 if dtype == "float" else np.float64)
    fb = np.asarray(b, dtype=np.float32 if dtype == "float" else np.float64)

    ia = struct.unpack(ipack, struct.pack(pack, float(fa)))[0]
    ib = struct.unpack(ipack, struct.pack(pack, float(fb)))[0]

    negative_a = ia < 0
    negative_b = ib < 0
    if negative_a != negative_b:
        if float(fa) == float(fb):  # +0 == -0
            return 0.0
        return float(2 << 28)

    return float(abs(ia - ib))


@dataclass(frozen=True, slots=True)
class Tolerance:
    """A comparison tolerance.

    Encapsulates the mode, threshold value, and floor used when determining
    whether two floating-point scalars (or arrays) are considered different.
    Scalar comparisons are provided by :meth:`diff` and :meth:`delta`;
    vectorized equivalents are :meth:`diff_array` and :meth:`delta_array`.

    Parameters
    ----------
    mode : ToleranceMode, optional
        Tolerance mode controlling how ``value`` is interpreted.
        Defaults to ``ToleranceMode.RELATIVE``.
    value : float, optional
        Tolerance threshold; interpretation depends on ``mode``.
        Defaults to ``0.0``.
    floor : float, optional
        Values whose magnitudes are both at or below this threshold are
        treated as equal regardless of ``mode``.  Defaults to ``0.0``.
    use_old_floor : bool, optional
        If ``True``, the legacy floor definition is used: values are below
        the floor when ``|a - b| < floor``.  The default (``False``) new
        definition treats values as equal when both ``|a| <= floor`` and
        ``|b| <= floor``.

    Examples
    --------
    Construct a relative tolerance and test two scalars:

    >>> tol = Tolerance(mode=ToleranceMode.RELATIVE, value=1e-6)
    >>> tol.diff(1.0, 1.0 + 1e-7)
    False
    >>> tol.diff(1.0, 1.0 + 1e-5)
    True

    Construct via the :meth:`make` factory using a string mode:

    >>> tol = Tolerance.make("abs", value=0.01)
    >>> tol.delta(3.0, 3.005)
    0.004999999999999893
    """

    mode: ToleranceMode = ToleranceMode.RELATIVE
    value: float = 0.0
    floor: float = 0.0
    use_old_floor: bool = False

    @classmethod
    def make(
        cls,
        mode: ToleranceMode | str = ToleranceMode.RELATIVE,
        value: float = 0.0,
        floor: float = 0.0,
        *,
        use_old_floor: bool = False,
    ) -> Tolerance:
        """Construct a tolerance, parsing ``mode`` if it is a string.

        This is the preferred factory for interactive use because it accepts
        mode strings and abbreviations in addition to :class:`ToleranceMode`
        members.

        Parameters
        ----------
        mode : ToleranceMode or str, optional
            Tolerance mode.  Accepts any value understood by
            :meth:`ToleranceMode.parse`, including full names (``"relative"``),
            abbreviations (``"rel"``), and aliases.  Defaults to
            ``ToleranceMode.RELATIVE``.
        value : float, optional
            Tolerance threshold.  Defaults to ``0.0``.
        floor : float, optional
            Floor below which values are treated as equal.  Defaults to
            ``0.0``.
        use_old_floor : bool, optional
            Use the legacy floor semantics (``|a - b| < floor``).  Defaults to
            ``False``.

        Returns
        -------
        Tolerance
            A new :class:`Tolerance` instance with the parsed mode and
            coerced float fields.

        Examples
        --------
        >>> Tolerance.make("rel", value=1e-4)  # doctest: +ELLIPSIS
        Tolerance(mode=<ToleranceMode.RELATIVE: 'relative'>, value=0.0001, ...)
        >>> Tolerance.make("abs", value=0.001, floor=1e-10)  # doctest: +ELLIPSIS
        Tolerance(mode=<ToleranceMode.ABSOLUTE: 'absolute'>, value=0.001, ...)
        """

        return cls(
            mode=ToleranceMode.parse(mode),
            value=float(value),
            floor=float(floor),
            use_old_floor=use_old_floor,
        )

    # -- scalar API ---------------------------------------------------------

    def _below_floor(self, v1: float, v2: float) -> bool:
        if self.use_old_floor:
            return abs(v1 - v2) < self.floor
        return abs(v1) <= self.floor and abs(v2) <= self.floor

    def diff(self, v1: float, v2: float) -> bool:
        """Return ``True`` if ``v1`` and ``v2`` differ by more than the tolerance.

        The comparison depends on :attr:`mode`:

        * ``IGNORE`` — always returns ``False``.
        * Both values below the floor — returns ``False``.
        * ``RELATIVE`` — ``|v1 - v2| > value * max(|v1|, |v2|)``; both-zero
          is treated as equal.
        * ``ABSOLUTE`` — ``|v1 - v2| > value``.
        * ``COMBINED`` — relative when ``max(|v1|, |v2|) > 1``, otherwise
          absolute.
        * ``ULPS_FLOAT`` / ``ULPS_DOUBLE`` — ULPs distance exceeds ``value``.
        * ``EIGEN_*`` — same as above but on magnitudes ``|v1|``, ``|v2|``.

        Parameters
        ----------
        v1 : float
            First scalar value.
        v2 : float
            Second scalar value.

        Returns
        -------
        bool
            ``True`` when the values differ by more than the tolerance,
            ``False`` otherwise.

        Examples
        --------
        Absolute mode — straightforward threshold test:

        >>> tol = Tolerance(mode=ToleranceMode.ABSOLUTE, value=0.01)
        >>> tol.diff(1.0, 1.005)
        False
        >>> tol.diff(1.0, 1.02)
        True

        Relative mode — both-zero treated as equal:

        >>> tol = Tolerance(mode=ToleranceMode.RELATIVE, value=1e-6)
        >>> tol.diff(0.0, 0.0)
        False

        IGNORE mode — always equal:

        >>> tol = Tolerance(mode=ToleranceMode.IGNORE)
        >>> tol.diff(0.0, 1e10)
        False
        """

        if self.mode is ToleranceMode.IGNORE:
            return False

        if self._below_floor(v1, v2):
            return False

        av1, av2 = abs(v1), abs(v2)
        mode = self.mode

        if mode is ToleranceMode.RELATIVE:
            if v1 == 0.0 and v2 == 0.0:
                return False
            return abs(v1 - v2) > self.value * max(av1, av2)
        if mode is ToleranceMode.ABSOLUTE:
            return abs(v1 - v2) > self.value
        if mode is ToleranceMode.COMBINED:
            tol = max(1.0, max(av1, av2))
            return abs(v1 - v2) >= tol * self.value
        if mode is ToleranceMode.ULPS_FLOAT:
            return _ulps_distance(v1, v2, dtype="float") > self.value
        if mode is ToleranceMode.ULPS_DOUBLE:
            return _ulps_distance(v1, v2, dtype="double") > self.value
        if mode is ToleranceMode.EIGEN_RELATIVE:
            if v1 == 0.0 and v2 == 0.0:
                return False
            return abs(av1 - av2) > self.value * max(av1, av2)
        if mode is ToleranceMode.EIGEN_ABSOLUTE:
            return abs(av1 - av2) > self.value
        if mode is ToleranceMode.EIGEN_COMBINED:
            tol = max(1.0, max(av1, av2))
            return abs(av1 - av2) >= tol * self.value
        return True

    def delta(self, v1: float, v2: float) -> float:
        """Return the mode-dependent magnitude of the difference.

        Returns ``0.0`` when the values are below the floor or the mode is
        ``IGNORE``.  The returned value is the same quantity compared against
        ``value`` inside :meth:`diff`, so ``diff`` is equivalent to
        ``delta(v1, v2) > value`` for most modes.

        Parameters
        ----------
        v1 : float
            First scalar value.
        v2 : float
            Second scalar value.

        Returns
        -------
        float
            Mode-dependent measure of the difference between ``v1`` and
            ``v2``.  Common cases:

            * ``RELATIVE`` — ``|v1 - v2| / max(|v1|, |v2|)``
            * ``ABSOLUTE`` — ``|v1 - v2|``
            * ``COMBINED`` — absolute or relative depending on magnitude
            * ``ULPS_FLOAT`` / ``ULPS_DOUBLE`` — ULPs distance (float)
            * ``EIGEN_*`` — same as above on magnitudes
            * ``IGNORE`` — always ``0.0``

        Examples
        --------
        >>> tol = Tolerance(mode=ToleranceMode.RELATIVE, value=1e-6)
        >>> tol.delta(2.0, 2.1)
        0.04999999999999982

        >>> tol = Tolerance(mode=ToleranceMode.ABSOLUTE, value=0.01)
        >>> tol.delta(5.0, 5.003)
        0.002999999999999558

        Both values at zero returns 0.0 for relative mode:

        >>> tol = Tolerance(mode=ToleranceMode.RELATIVE, value=1e-6)
        >>> tol.delta(0.0, 0.0)
        0.0
        """

        if self.mode is ToleranceMode.IGNORE:
            return 0.0
        if self._below_floor(v1, v2):
            return 0.0

        av1, av2 = abs(v1), abs(v2)
        mode = self.mode

        if mode is ToleranceMode.RELATIVE:
            if v1 == 0.0 and v2 == 0.0:
                return 0.0
            return abs(v1 - v2) / max(av1, av2)
        if mode is ToleranceMode.ABSOLUTE:
            return abs(v1 - v2)
        if mode is ToleranceMode.COMBINED:
            m = max(av1, av2)
            if m > 1.0:
                return abs(v1 - v2) / m
            return abs(v1 - v2)
        if mode is ToleranceMode.ULPS_FLOAT:
            return _ulps_distance(v1, v2, dtype="float")
        if mode is ToleranceMode.ULPS_DOUBLE:
            return _ulps_distance(v1, v2, dtype="double")
        if mode is ToleranceMode.EIGEN_RELATIVE:
            if v1 == 0.0 and v2 == 0.0:
                return 0.0
            return abs(av1 - av2) / max(av1, av2)
        if mode is ToleranceMode.EIGEN_ABSOLUTE:
            return abs(av1 - av2)
        if mode is ToleranceMode.EIGEN_COMBINED:
            m = max(av1, av2)
            if m > 1.0:
                return abs(av1 - av2) / m
            return abs(av1 - av2)
        return 0.0

    # -- vectorized API -----------------------------------------------------

    def diff_array(self, a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Return an element-wise boolean "differs" mask for two arrays.

        Vectorized equivalent of :meth:`diff`: ``True`` where the two values
        differ by more than the tolerance.  This shares its definition with
        the scalar :meth:`diff` so that array and scalar comparison paths never
        diverge.

        Parameters
        ----------
        a : array_like
            First array (or scalar).  Converted to ``float64``.
        b : array_like
            Second array (or scalar).  Converted to ``float64``.  Must be
            broadcastable against ``a``.

        Returns
        -------
        ndarray of bool
            Boolean array with the same shape as ``numpy.broadcast(a, b)``.
            Element ``[i]`` is ``True`` when ``a[i]`` and ``b[i]`` differ by
            more than the tolerance.

        Notes
        -----
        NaN handling is intentionally left to callers.  ``exodiff`` treats NaN
        mismatches separately; pairs where either value is NaN are reported as
        *not* differing here so that a dedicated NaN check can own that
        semantics.

        Examples
        --------
        >>> import numpy as np
        >>> tol = Tolerance(mode=ToleranceMode.ABSOLUTE, value=0.1)
        >>> tol.diff_array([1.0, 2.0, 3.0], [1.05, 2.2, 3.0])
        array([False,  True, False])

        Scalar broadcast:

        >>> tol.diff_array(np.zeros(4), 0.5)
        array([ True,  True,  True,  True])
        """

        x = np.asarray(a, dtype=np.float64)
        y = np.asarray(b, dtype=np.float64)
        shape = np.broadcast(x, y).shape

        if self.mode is ToleranceMode.IGNORE:
            return np.zeros(shape, dtype=bool)

        ax, ay = np.abs(x), np.abs(y)

        # Floor: entries below the floor are considered equal.
        if self.use_old_floor:
            active = np.abs(x - y) >= self.floor
        else:
            active = (ax >= self.floor) | (ay >= self.floor)

        active = np.broadcast_to(active, shape)
        if not np.any(active):
            return np.zeros(shape, dtype=bool)

        mode = self.mode
        maxab = np.maximum(ax, ay)
        both_zero = (x == 0.0) & (y == 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            if mode in (ToleranceMode.RELATIVE, ToleranceMode.EIGEN_RELATIVE):
                num = np.abs(ax - ay) if mode is ToleranceMode.EIGEN_RELATIVE else np.abs(x - y)
                exceeded = num > self.value * maxab
                mask = active & ~both_zero & exceeded
            elif mode is ToleranceMode.ABSOLUTE:
                mask = active & (np.abs(x - y) > self.value)
            elif mode is ToleranceMode.EIGEN_ABSOLUTE:
                mask = active & (np.abs(ax - ay) > self.value)
            elif mode in (ToleranceMode.COMBINED, ToleranceMode.EIGEN_COMBINED):
                num = np.abs(ax - ay) if mode is ToleranceMode.EIGEN_COMBINED else np.abs(x - y)
                denom = np.where(maxab > 1.0, maxab, 1.0)
                mask = active & (num >= denom * self.value)
            elif mode in (ToleranceMode.ULPS_FLOAT, ToleranceMode.ULPS_DOUBLE):
                mask = active & (self.delta_array(x, y) > self.value)
            else:  # pragma: no cover - all modes handled above
                mask = active

        return np.asarray(np.broadcast_to(mask, shape), dtype=bool)

    def delta_array(self, a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return an element-wise ``delta`` array for two equal-shaped arrays.

        Vectorized equivalent of :meth:`delta`, used for fast comparison of
        result-variable arrays.

        Parameters
        ----------
        a : array_like
            First array (or scalar).  Converted to ``float64``.
        b : array_like
            Second array (or scalar).  Converted to ``float64``.  Must be
            broadcastable against ``a``.

        Returns
        -------
        ndarray of float64
            Array of the same shape as ``numpy.broadcast(a, b)`` containing
            the mode-dependent difference magnitude for each element pair.
            Elements where either value is below the floor (or mode is
            ``IGNORE``) are set to ``0.0``.

        Examples
        --------
        >>> import numpy as np
        >>> tol = Tolerance(mode=ToleranceMode.ABSOLUTE, value=0.1)
        >>> tol.delta_array([1.0, 2.0], [1.05, 2.3])
        array([0.05, 0.3 ])

        Relative mode:

        >>> tol = Tolerance(mode=ToleranceMode.RELATIVE, value=1e-3)
        >>> tol.delta_array([100.0, 1.0], [101.0, 1.0])
        array([0.00990099, 0.        ])
        """

        x = np.asarray(a, dtype=np.float64)
        y = np.asarray(b, dtype=np.float64)
        out = np.zeros(np.broadcast(x, y).shape, dtype=np.float64)

        if self.mode is ToleranceMode.IGNORE:
            return out

        ax, ay = np.abs(x), np.abs(y)

        if self.use_old_floor:
            active = np.abs(x - y) >= self.floor
        else:
            active = (ax >= self.floor) | (ay >= self.floor)

        if not np.any(active):
            return out

        mode = self.mode
        maxab = np.maximum(ax, ay)

        if mode in (ToleranceMode.RELATIVE, ToleranceMode.EIGEN_RELATIVE):
            eigen = mode is ToleranceMode.EIGEN_RELATIVE
            num = np.abs(ax - ay) if eigen else np.abs(x - y)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(maxab > 0.0, num / maxab, 0.0)
            out = np.where(active, ratio, 0.0)
        elif mode is ToleranceMode.ABSOLUTE:
            out = np.where(active, np.abs(x - y), 0.0)
        elif mode is ToleranceMode.EIGEN_ABSOLUTE:
            out = np.where(active, np.abs(ax - ay), 0.0)
        elif mode in (ToleranceMode.COMBINED, ToleranceMode.EIGEN_COMBINED):
            eigen = mode is ToleranceMode.EIGEN_COMBINED
            num = np.abs(ax - ay) if eigen else np.abs(x - y)
            denom = np.where(maxab > 1.0, maxab, 1.0)
            out = np.where(active, num / denom, 0.0)
        elif mode in (ToleranceMode.ULPS_FLOAT, ToleranceMode.ULPS_DOUBLE):
            # ULPs is inherently scalar (bit patterns); fall back to a loop.
            dtype = "float" if mode is ToleranceMode.ULPS_FLOAT else "double"
            flat_x = x.ravel()
            flat_y = np.broadcast_to(y, x.shape).ravel()
            flat_active = np.broadcast_to(active, x.shape).ravel()
            flat_out = out.ravel().copy()
            for i in range(flat_out.size):
                if flat_active[i]:
                    flat_out[i] = _ulps_distance(float(flat_x[i]), float(flat_y[i]), dtype=dtype)
            out = flat_out.reshape(x.shape)

        return out
