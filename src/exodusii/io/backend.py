# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Backend protocol for NetCDF access.

The modern Exodus API talks to this protocol instead of directly depending on
``netCDF4.Dataset`` throughout the codebase. The only concrete backend planned
for this refresh is the required ``netCDF4`` backend.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Protocol

from exodusii.core.errors import ExodusInvalidModeError

FileMode = Literal["r", "w", "a", "r+"]


class NetCDFBackend(Protocol):
    """Protocol implemented by NetCDF backends."""

    @property
    def path(self) -> Path:
        """Filesystem path for the open dataset."""

    @property
    def mode(self) -> FileMode:
        """Open mode for the dataset."""

    def close(self) -> None:
        """Close the dataset."""

    def sync(self) -> None:
        """Flush pending writes, if supported."""

    def dimensions(self) -> tuple[str, ...]:
        """Return all dimension names."""

    def variables(self) -> tuple[str, ...]:
        """Return all variable names."""

    def has_dimension(self, name: str) -> bool:
        """Return true if the dataset contains dimension ``name``."""

    def has_variable(self, name: str) -> bool:
        """Return true if the dataset contains variable ``name``."""

    def dimension(self, name: str, default: int | None = None) -> int | None:
        """Return a dimension size."""

    def variable(self, name: str, *, default: Any = None, raw: bool = False) -> Any:
        """Return a variable value or raw backend variable."""

    def create_dimension(self, name: str, size: int | None) -> None:
        """Create a dimension."""

    def create_variable(
        self, name: str, dtype: type[int] | type[float] | type[str] | str, dimensions: Sequence[str]
    ) -> None:
        """Create a variable."""

    def write_variable(self, name: str, value: Any, *indices: int) -> None:
        """Write a variable or indexed slice of a variable."""

    def attribute(self, name: str, default: Any = None) -> Any:
        """Return a global attribute."""

    def set_attribute(self, name: str, value: Any) -> None:
        """Set a global attribute."""

    def variable_attribute(self, variable: str, name: str, default: Any = None) -> Any:
        """Return a variable attribute."""

    def set_variable_attribute(self, variable: str, name: str, value: Any) -> None:
        """Set a variable attribute."""


def normalize_file_mode(mode: str) -> FileMode:
    """Normalize and validate a NetCDF file mode.

    Parameters
    ----------
    mode
        File mode. Supported values are ``"r"``, ``"w"``, ``"a"``, and ``"r+"``.

    Returns
    -------
    FileMode
        Normalized mode.

    Raises
    ------
    ExodusInvalidModeError
        If the mode is unsupported.
    """

    normalized = mode.strip().lower()

    if normalized in {"r", "w", "a", "r+"}:
        return normalized  # type: ignore[return-value]

    raise ExodusInvalidModeError(
        f"invalid Exodus file mode {mode!r}; expected 'r', 'w', 'a', or 'r+'"
    )


def mode_is_readable(mode: FileMode) -> bool:
    """Return true if ``mode`` permits reading."""

    return mode in {"r", "a", "r+"}


def mode_is_writable(mode: FileMode) -> bool:
    """Return true if ``mode`` permits writing."""

    return mode in {"w", "a", "r+"}


__all__ = [
    "FileMode",
    "NetCDFBackend",
    "mode_is_readable",
    "mode_is_writable",
    "normalize_file_mode",
]
