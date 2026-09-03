# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Tests for the exodiff-compatible tolerance model."""

import numpy as np
import pytest

from exodusii.core.tolerance import Tolerance
from exodusii.core.tolerance import ToleranceMode


def test_mode_parse_and_abbreviation() -> None:
    assert ToleranceMode.parse("relative") is ToleranceMode.RELATIVE
    assert ToleranceMode.parse("rel") is ToleranceMode.RELATIVE
    assert ToleranceMode.parse("ABSOLUTE") is ToleranceMode.ABSOLUTE
    assert ToleranceMode.parse(ToleranceMode.COMBINED) is ToleranceMode.COMBINED
    assert ToleranceMode.RELATIVE.abbreviation == "rel"
    assert ToleranceMode.ULPS_DOUBLE.abbreviation == "upd"


def test_ignore_mode_never_differs() -> None:
    tol = Tolerance.make("ignore", 0.0)
    assert not tol.diff(0.0, 1.0e30)
    assert tol.delta(0.0, 1.0e30) == 0.0


def test_absolute() -> None:
    tol = Tolerance.make("absolute", 1.0e-6)
    assert not tol.diff(1.0, 1.0 + 1.0e-7)
    assert tol.diff(1.0, 1.0 + 1.0e-5)
    assert np.isclose(tol.delta(1.0, 1.0 + 1.0e-5), 1.0e-5)


def test_relative() -> None:
    tol = Tolerance.make("relative", 1.0e-6)
    # |v1-v2| > value * max(|v1|,|v2|)
    assert not tol.diff(1000.0, 1000.0 + 1.0e-4)  # rel diff 1e-7 < 1e-6
    assert tol.diff(1000.0, 1000.0 + 1.0)  # rel diff 1e-3 > 1e-6
    assert not tol.diff(0.0, 0.0)
    # relative delta = |v1-v2| / max(|v1|,|v2|) = 0.002 / 2.002
    assert np.isclose(tol.delta(2.0, 2.0 + 2.0e-3), 2.0e-3 / 2.002)


def test_combined_uses_abs_below_one_rel_above() -> None:
    tol = Tolerance.make("combined", 1.0e-3)
    # both < 1: absolute-like, tol = 1.0 * value
    assert not tol.diff(0.1, 0.1 + 1.0e-4)
    assert tol.diff(0.1, 0.1 + 2.0e-3)
    # large values: relative-like, tol = max * value
    assert not tol.diff(1000.0, 1000.0 + 0.5)  # 0.5 >= 1000*1e-3=1.0? no -> equal
    assert tol.diff(1000.0, 1000.0 + 2.0)  # 2.0 >= 1.0 -> different


def test_relative_floor_new_definition() -> None:
    # New floor: equal when both |v| <= floor.
    tol = Tolerance.make("relative", 1.0e-6, floor=1.0e-3)
    assert not tol.diff(1.0e-4, 5.0e-4)  # both below floor -> equal
    assert tol.diff(1.0e-4, 1.0)  # one above floor, large rel diff


def test_relative_floor_old_definition() -> None:
    # Old floor: equal when |v1 - v2| < floor.
    tol = Tolerance.make("relative", 1.0e-6, floor=1.0e-3, use_old_floor=True)
    assert not tol.diff(1.0, 1.0 + 5.0e-4)  # diff below floor -> equal
    assert tol.diff(1.0, 1.0 + 2.0e-3)  # diff above floor and rel > tol


def test_eigen_absolute_matches_negation() -> None:
    tol = Tolerance.make("eigenabs", 1.0e-6)
    # Eigen modes compare magnitudes: v and -v are equal.
    assert not tol.diff(3.0, -3.0)
    assert tol.diff(3.0, -3.5)


def test_ulps_double_sign_mismatch() -> None:
    tol = Tolerance.make("ulps_double", 4)
    # Opposite signs, not equal -> maximally different.
    assert tol.diff(1.0, -1.0)
    assert np.isclose(tol.delta(1.0, -1.0), float(2 << 28))
    # +0 and -0 are equal.
    assert not tol.diff(0.0, -0.0)


def test_ulps_double_close_values() -> None:
    tol = Tolerance.make("ulps_double", 4)
    x = 1.0
    # 2 ULPs away.
    y = np.nextafter(np.nextafter(x, np.inf), np.inf)
    assert not tol.diff(x, float(y))
    # Many ULPs away.
    z = x
    for _ in range(10):
        z = float(np.nextafter(z, np.inf))
    assert tol.diff(x, z)


def test_delta_array_matches_scalar_relative() -> None:
    tol = Tolerance.make("relative", 1.0e-6)
    a = np.array([1.0, 1000.0, 0.0, 2.0])
    b = np.array([1.0 + 1.0e-3, 1000.0 + 1.0, 0.0, 2.0 + 2.0e-3])
    vec = tol.delta_array(a, b)
    scalar = np.array([tol.delta(x, y) for x, y in zip(a, b)])
    assert np.allclose(vec, scalar)


def test_delta_array_matches_scalar_combined() -> None:
    tol = Tolerance.make("combined", 1.0e-3, floor=1.0e-9)
    a = np.array([0.1, 1000.0, 1.0e-12])
    b = np.array([0.1 + 2.0e-3, 1000.0 + 2.0, 2.0e-12])
    vec = tol.delta_array(a, b)
    scalar = np.array([tol.delta(x, y) for x, y in zip(a, b)])
    assert np.allclose(vec, scalar)


def test_delta_array_matches_scalar_ulps() -> None:
    tol = Tolerance.make("ulps_double", 4)
    a = np.array([1.0, 1.0, 5.0])
    b = np.array([-1.0, float(np.nextafter(1.0, np.inf)), 5.0])
    vec = tol.delta_array(a, b)
    scalar = np.array([tol.delta(x, y) for x, y in zip(a, b)])
    assert np.allclose(vec, scalar)


@pytest.mark.parametrize(
    ("mode", "value", "floor"),
    [
        ("relative", 1.0e-6, 0.0),
        ("absolute", 1.0e-6, 0.0),
        ("combined", 1.0e-3, 1.0e-9),
        ("eigenrel", 1.0e-6, 0.0),
        ("eigenabs", 1.0e-6, 0.0),
        ("eigencom", 1.0e-3, 0.0),
        ("ulps_double", 4, 0.0),
        ("ulps_float", 4, 0.0),
        ("ignore", 0.0, 0.0),
    ],
)
def test_diff_array_matches_scalar_diff(mode: str, value: float, floor: float) -> None:
    tol = Tolerance.make(mode, value, floor)
    a = np.array([1.0, 1000.0, 0.0, 2.0, -3.0, 0.1, 1.0e-12, 5.0])
    b = np.array(
        [
            1.0 + 1.0e-3,
            1000.0 + 1.0,
            0.0,
            2.0 + 2.0e-3,
            -3.5,
            0.1 + 2.0e-3,
            2.0e-12,
            float(np.nextafter(5.0, np.inf)),
        ]
    )
    vec = tol.diff_array(a, b)
    scalar = np.array([tol.diff(float(x), float(y)) for x, y in zip(a, b)])
    assert np.array_equal(vec, scalar)


def test_diff_array_respects_floor() -> None:
    tol = Tolerance.make("relative", 1.0e-6, floor=1.0e-3)
    a = np.array([1.0e-4, 1.0e-4])
    b = np.array([5.0e-4, 1.0])
    # First pair both below floor -> equal; second pair one above -> differs.
    assert np.array_equal(tol.diff_array(a, b), np.array([False, True]))


def test_diff_array_broadcasts_scalar() -> None:
    tol = Tolerance.make("absolute", 1.0e-6)
    a = np.array([1.0, 1.0, 1.0])
    mask = tol.diff_array(a, 1.0)
    assert mask.shape == (3,)
    assert not mask.any()
