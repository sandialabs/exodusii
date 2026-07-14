# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Required :mod:`netCDF4` backend."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np
from netCDF4 import Dataset  # type: ignore[reportMissingTypeStubs]

from exodusii.core.errors import ExodusDimensionError
from exodusii.core.errors import ExodusVariableError
from exodusii.core.errors import ExodusWriteError
from exodusii.core.strings import decode_text
from exodusii.core.strings import stringify
from exodusii.io.backend import FileMode
from exodusii.io.backend import normalize_file_mode

NetCDFFormat = Literal[
    "NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_CLASSIC", "NETCDF3_64BIT_OFFSET", "NETCDF3_64BIT_DATA"
]


class NetCDF4Backend:
    """Thin wrapper around :class:`netCDF4.Dataset`.

    Parameters
    ----------
    path
        Dataset path.
    mode
        File mode. Supported values are ``"r"``, ``"w"``, ``"a"``, and ``"r+"``.
    format
        NetCDF file format used when creating files.
    """

    def __init__(
        self, path: str | Path, mode: str = "r", *, format: NetCDFFormat = "NETCDF4_CLASSIC"
    ) -> None:
        self._path = Path(path)
        self._mode = normalize_file_mode(mode)
        self._dataset: Any = Dataset(str(self._path), mode=self._mode, format=format)

        # Exodus stores fixed-width character arrays and ordinary numeric arrays.
        # Disable netCDF4 automatic conversions/masking so reads are raw ndarray data.
        self._dataset.set_auto_chartostring(False)
        self._dataset.set_auto_mask(False)
        self._dataset.set_auto_maskandscale(False)

    @property
    def path(self) -> Path:
        """Filesystem path for the open dataset."""

        return self._path

    @property
    def mode(self) -> FileMode:
        """Open mode for the dataset."""

        return self._mode

    @property
    def dataset(self) -> Any:
        """Raw :class:`netCDF4.Dataset` object."""

        return self._dataset

    def close(self) -> None:
        """Close the dataset."""

        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def sync(self) -> None:
        """Flush pending writes."""

        self._require_open()
        self._dataset.sync()

    def __enter__(self) -> "NetCDF4Backend":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def dimensions(self) -> tuple[str, ...]:
        """Return all dimension names."""

        self._require_open()
        return tuple(self._dataset.dimensions.keys())

    def variables(self) -> tuple[str, ...]:
        """Return all variable names."""

        self._require_open()
        return tuple(self._dataset.variables.keys())

    def has_dimension(self, name: str) -> bool:
        """Return true if the dataset contains dimension ``name``."""

        self._require_open()
        return name in self._dataset.dimensions

    def has_variable(self, name: str) -> bool:
        """Return true if the dataset contains variable ``name``."""

        self._require_open()
        return name in self._dataset.variables

    def dimension(self, name: str, default: int | None = None) -> int | None:
        """Return a dimension size."""

        self._require_open()

        if name not in self._dataset.dimensions:
            return default

        return len(self._dataset.dimensions[name])

    def variable(self, name: str, *, default: Any = None, raw: bool = False) -> Any:
        """Return a variable value or raw backend variable."""

        self._require_open()

        if name not in self._dataset.variables:
            return default

        variable = self._dataset.variables[name]
        if raw:
            return variable

        if hasattr(variable, "set_auto_chartostring"):
            variable.set_auto_chartostring(False)

        value = variable[...]
        if isinstance(value, np.ma.MaskedArray):
            if value.dtype.kind == "S":
                value = np.asarray(value.filled(b"\x00"))
            elif value.dtype.kind == "U":
                value = np.asarray(value.filled("\x00"))
            else:
                value = np.asarray(value.filled(0))

        if _is_string_like(value):
            # If netCDF4 has already converted a 2-D char array like
            # (num_names, len_string) into a 1-D array of strings, do not let
            # stringify() interpret that 1-D U1/S1 array as one character vector.
            if (
                isinstance(value, np.ndarray)
                and value.ndim == 1
                and len(getattr(variable, "dimensions", ())) > 1
            ):
                return np.asarray([decode_text(item) for item in value], dtype=object)

            return stringify(value)

        return value

    def create_dimension(self, name: str, size: int | None) -> None:
        """Create a dimension."""

        self._require_open()
        self._dataset.createDimension(name, size)

    def create_variable(
        self, name: str, dtype: type[int] | type[float] | type[str] | str, dimensions: Sequence[str]
    ) -> None:
        """Create a variable."""

        self._require_open()

        missing = [
            dimension for dimension in dimensions if dimension not in self._dataset.dimensions
        ]
        if missing:
            missing_text = ", ".join(missing)
            raise ExodusDimensionError(
                f"cannot create variable {name!r}; missing dimensions: {missing_text}"
            )

        self._dataset.createVariable(name, _dtype_code(dtype), tuple(dimensions))

    def write_variable(self, name: str, value: Any, *indices: int) -> None:
        """Write a variable or indexed slice of a variable.

        The indexed behavior mirrors the old helper semantics:

        - ``write_variable(name, value)`` writes the whole variable.
        - ``write_variable(name, value, i)`` writes ``var[i]`` for 1-D variables
          or ``var[i, :]`` for variables with rank greater than one.
        - ``write_variable(name, value, i, j)`` writes ``var[i, j]`` for 2-D
          variables or ``var[i, j, :]`` for variables with rank greater than two.
        """

        self._require_open()

        if name not in self._dataset.variables:
            raise ExodusVariableError(f"variable {name!r} not found")

        variable = self._dataset.variables[name]

        try:
            if not indices:
                if getattr(variable, "shape", ()) == ():
                    variable[...] = value
                else:
                    variable[:] = value
            else:
                ndim = len(getattr(variable, "dimensions", ()))
                if len(indices) > ndim:
                    raise IndexError("too many indices for variable")
                key: Any = (
                    tuple(indices) if len(indices) == ndim else (*tuple(indices), slice(None))
                )
                variable[key] = value
        except Exception as exc:
            raise ExodusWriteError(f"failed writing variable {name!r}") from exc

    def attribute(self, name: str, default: Any = None) -> Any:
        """Return a global attribute."""

        self._require_open()

        if name not in self._dataset.ncattrs():
            return default

        return self._dataset.getncattr(name)

    def set_attribute(self, name: str, value: Any) -> None:
        """Set a global attribute."""

        self._require_open()
        self._dataset.setncattr(name, value)

    def variable_attribute(self, variable: str, name: str, default: Any = None) -> Any:
        """Return a variable attribute."""

        self._require_open()

        if variable not in self._dataset.variables:
            raise ExodusVariableError(f"variable {variable!r} not found")

        nc_variable = self._dataset.variables[variable]
        if name not in nc_variable.ncattrs():
            return default

        return nc_variable.getncattr(name)

    def set_variable_attribute(self, variable: str, name: str, value: Any) -> None:
        """Set a variable attribute."""

        self._require_open()

        if variable not in self._dataset.variables:
            raise ExodusVariableError(f"variable {variable!r} not found")

        self._dataset.variables[variable].setncattr(name, value)

    def _require_open(self) -> None:
        if self._dataset is None:
            raise ExodusWriteError(f"NetCDF dataset {self._path} is closed")


def open_netcdf4(path: str | Path, mode: str = "r") -> NetCDF4Backend:
    """Open a NetCDF4 backend."""

    return NetCDF4Backend(path, mode=mode)


def _dtype_code(dtype: type[int] | type[float] | type[str] | str) -> str:
    if dtype is int:
        return "i4"
    if dtype is float:
        return "f8"
    if dtype is str:
        return "S1"
    if isinstance(dtype, str):
        return dtype

    raise TypeError(f"unsupported NetCDF variable dtype {dtype!r}")


def _is_string_like(value: Any) -> bool:
    if isinstance(value, str | bytes | np.str_ | np.bytes_):
        return True

    if isinstance(value, np.ndarray):
        return value.dtype.kind in {"S", "U"}

    return False


__all__ = ["NetCDF4Backend", "open_netcdf4"]
