# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from exodusii.core.errors import ExodusInvalidTimeError
from exodusii.core.time import TimeSelection
from exodusii.core.time import nearest_time_index
from exodusii.core.time import resolve_time
from exodusii.core.time import resolve_time_index
from exodusii.core.time import resolve_time_step


def test_time_selection_aliases() -> None:
    selection = TimeSelection(index=2, step=3, value=1.5)

    assert selection.index0 == 2
    assert selection.step1 == 3


def test_resolve_time_none_defaults_to_last() -> None:
    selection = resolve_time([0.0, 1.0, 2.0])

    assert selection.index == 2
    assert selection.step == 3
    assert selection.value == 2.0
    assert selection.requested == "last"
    assert selection.exact


def test_resolve_time_none_can_default_to_first() -> None:
    selection = resolve_time([0.0, 1.0, 2.0], default="first")

    assert selection.index == 0
    assert selection.step == 1
    assert selection.value == 0.0
    assert selection.requested == "first"


@pytest.mark.parametrize(
    ("selector", "index", "step", "value"),
    [
        ("first", 0, 1, 0.0),
        ("FIRST", 0, 1, 0.0),
        (" first ", 0, 1, 0.0),
        ("last", 2, 3, 2.0),
        ("LAST", 2, 3, 2.0),
        (" last ", 2, 3, 2.0),
    ],
)
def test_resolve_time_string_selectors(selector: str, index: int, step: int, value: float) -> None:
    selection = resolve_time([0.0, 1.0, 2.0], selector)

    assert selection.index == index
    assert selection.step == step
    assert selection.value == value
    assert selection.requested == selector
    assert selection.exact


def test_resolve_time_rejects_unknown_string() -> None:
    with pytest.raises(ExodusInvalidTimeError, match="Unsupported time selector"):
        resolve_time([0.0, 1.0, 2.0], "middle")


@pytest.mark.parametrize(
    ("selector", "index", "step", "value"),
    [
        (0, 0, 1, 0.0),
        (1, 1, 2, 1.0),
        (2, 2, 3, 2.0),
        (-1, 2, 3, 2.0),
        (-2, 1, 2, 1.0),
        (np.int64(1), 1, 2, 1.0),
    ],
)
def test_resolve_time_integer_indices(selector: int, index: int, step: int, value: float) -> None:
    selection = resolve_time([0.0, 1.0, 2.0], selector)

    assert selection.index == index
    assert selection.step == step
    assert selection.value == value
    assert selection.requested == selector
    assert selection.exact


@pytest.mark.parametrize("selector", [3, -4])
def test_resolve_time_rejects_out_of_range_integer_indices(selector: int) -> None:
    with pytest.raises(ExodusInvalidTimeError, match="out of range"):
        resolve_time([0.0, 1.0, 2.0], selector)


def test_resolve_time_rejects_bool_index() -> None:
    with pytest.raises(ExodusInvalidTimeError, match="Unsupported time selector"):
        resolve_time([0.0, 1.0, 2.0], True)  # type: ignore[arg-type]


def test_resolve_time_float_exact() -> None:
    selection = resolve_time([0.0, 1.0, 2.0], 1.0)

    assert selection.index == 1
    assert selection.step == 2
    assert selection.value == 1.0
    assert selection.requested == 1.0
    assert selection.exact


def test_resolve_time_float_nearest() -> None:
    selection = resolve_time([0.0, 1.0, 2.0], 1.2)

    assert selection.index == 1
    assert selection.step == 2
    assert selection.value == 1.0
    assert selection.requested == 1.2
    assert not selection.exact


def test_resolve_time_float_nearest_upper() -> None:
    selection = resolve_time([0.0, 1.0, 2.0], 1.8)

    assert selection.index == 2
    assert selection.step == 3
    assert selection.value == 2.0
    assert selection.requested == 1.8
    assert not selection.exact


def test_resolve_time_float_exact_with_tolerance() -> None:
    selection = resolve_time([0.0, 1.0, 2.0], 1.001, tolerance=0.01)

    assert selection.index == 1
    assert selection.exact


def test_resolve_time_float_not_exact_with_tolerance() -> None:
    selection = resolve_time([0.0, 1.0, 2.0], 1.2, tolerance=0.01)

    assert selection.index == 1
    assert not selection.exact


def test_resolve_time_float_requires_exact_when_nearest_false() -> None:
    with pytest.raises(ExodusInvalidTimeError, match="was not found"):
        resolve_time([0.0, 1.0, 2.0], 1.2, nearest=False)


def test_resolve_time_float_allows_exact_when_nearest_false() -> None:
    selection = resolve_time([0.0, 1.0, 2.0], 1.0, nearest=False)

    assert selection.index == 1
    assert selection.exact


def test_resolve_time_rejects_negative_tolerance() -> None:
    with pytest.raises(ExodusInvalidTimeError, match="tolerance must be nonnegative"):
        resolve_time([0.0, 1.0, 2.0], 1.0, tolerance=-1.0)


def test_resolve_time_rejects_empty_times() -> None:
    with pytest.raises(ExodusInvalidTimeError, match="no time steps found"):
        resolve_time([])


def test_resolve_time_rejects_non_1d_times() -> None:
    with pytest.raises(ExodusInvalidTimeError, match="one-dimensional"):
        resolve_time([[0.0, 1.0], [2.0, 3.0]])


def test_resolve_time_rejects_unsupported_selector_type() -> None:
    with pytest.raises(ExodusInvalidTimeError, match="Unsupported time selector"):
        resolve_time([0.0, 1.0, 2.0], object())  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_resolve_time_index() -> None:
    selection = resolve_time_index([0.0, 1.0, 2.0], 1)

    assert selection.index == 1
    assert selection.step == 2
    assert selection.value == 1.0
    assert selection.requested == 1


def test_resolve_time_step() -> None:
    selection = resolve_time_step([0.0, 1.0, 2.0], 2)

    assert selection.index == 1
    assert selection.step == 2
    assert selection.value == 1.0


@pytest.mark.parametrize("step", [0, -1])
def test_resolve_time_step_rejects_nonpositive_steps(step: int) -> None:
    with pytest.raises(ExodusInvalidTimeError, match="one-based and positive"):
        resolve_time_step([0.0, 1.0, 2.0], step)


@pytest.mark.parametrize("step", [1.5, True])
def test_resolve_time_step_rejects_noninteger_steps(step: object) -> None:
    with pytest.raises(ExodusInvalidTimeError, match="must be an int"):
        resolve_time_step([0.0, 1.0, 2.0], step)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_nearest_time_index() -> None:
    assert nearest_time_index([0.0, 1.0, 2.0], 1.8) == 2


def test_resolve_time_accepts_numpy_array() -> None:
    times = np.asarray([0.0, 1.0, 2.0])

    selection = resolve_time(times, -1)

    assert selection.index == 2
    assert selection.value == 2.0
