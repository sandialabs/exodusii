# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import pytest

from exodusii.core.errors import ExodusInvalidModeError
from exodusii.io.backend import mode_is_readable
from exodusii.io.backend import mode_is_writable
from exodusii.io.backend import normalize_file_mode


@pytest.mark.parametrize("mode", ["r", "w", "a", "r+"])
def test_normalize_file_mode_accepts_supported_modes(mode: str) -> None:
    assert normalize_file_mode(mode) == mode


@pytest.mark.parametrize(
    ("mode", "expected"), [(" r ", "r"), (" W ", "w"), (" A ", "a"), (" R+ ", "r+")]
)
def test_normalize_file_mode_strips_and_lowercases(mode: str, expected: str) -> None:
    assert normalize_file_mode(mode) == expected


@pytest.mark.parametrize("mode", ["", "rb", "rw", "x", "read", "write"])
def test_normalize_file_mode_rejects_unsupported_modes(mode: str) -> None:
    with pytest.raises(ExodusInvalidModeError, match="invalid Exodus file mode"):
        normalize_file_mode(mode)


@pytest.mark.parametrize(
    ("mode", "expected"), [("r", True), ("w", False), ("a", True), ("r+", True)]
)
def test_mode_is_readable(mode: str, expected: bool) -> None:
    assert mode_is_readable(normalize_file_mode(mode)) is expected


@pytest.mark.parametrize(
    ("mode", "expected"), [("r", False), ("w", True), ("a", True), ("r+", True)]
)
def test_mode_is_writable(mode: str, expected: bool) -> None:
    assert mode_is_writable(normalize_file_mode(mode)) is expected
