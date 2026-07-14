# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Lineout filtering for tabular Exodus data."""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cmp_to_key

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class Lineout:
    """Restrict rows to points along a line parallel to an axis.

    Coordinate specifications may be:

    - ``"x"``, ``"y"``, ``"z"`` for free material coordinates
    - ``"X"``, ``"Y"``, ``"Z"`` for free displaced coordinates
    - floats for restricted coordinates
    """

    x: float | str | None = None
    y: float | str | None = None
    z: float | str | None = None
    tol: float | None = None

    @property
    def spec(self) -> tuple[float | str | None, float | str | None, float | str | None]:
        """Return ``(x, y, z)`` specification."""

        return (self.x, self.y, self.z)

    @property
    def needs_displacements(self) -> bool:
        """Return true if displaced coordinates are requested."""

        return self.x == "X" or self.y == "Y" or self.z == "Z"

    @classmethod
    def from_cli(cls, arg: str) -> "Lineout":
        """Parse a command-line lineout specification.

        Examples
        --------
        ``"x/1.0"`` selects a 2-D line along x with y fixed at 1.0.

        ``"1.0/Y/3.0/T0.1"`` selects a 3-D line along displaced y with x and z
        restricted using tolerance 0.1.
        """

        parts = [part.strip() for part in arg.split("/") if part.strip()]
        if not parts:
            raise ValueError("lineout: expected at least one spatial specifier")

        tol = None
        if parts[-1].startswith(("t", "T")):
            try:
                tol = float(parts[-1][1:])
            except ValueError as exc:
                raise ValueError("lineout: tolerance parameter must be a float") from exc
            parts = parts[:-1]

        if len(parts) > 3:
            raise ValueError("lineout: expected at most 3 spatial specifiers")

        x = cls.read_spatial_spec(parts[0], "x")
        y = cls.read_spatial_spec(parts[1], "y") if len(parts) > 1 else None
        z = cls.read_spatial_spec(parts[2], "z") if len(parts) > 2 else None

        return cls(x=x, y=y, z=z, tol=tol)

    @staticmethod
    def read_spatial_spec(spec: str | None, coord: str) -> float | str | None:
        """Parse one spatial specifier."""

        if spec is None:
            return None

        try:
            return float(spec)
        except ValueError:
            pass

        if spec.lower() != coord:
            raise ValueError(
                f"lineout: expected specifier {spec!r} to be {coord!r}, "
                f"{coord.upper()!r}, or a float"
            )

        return spec

    def apply(
        self,
        header_or_structured: Sequence[str] | npt.NDArray[np.void],
        data: npt.ArrayLike | None = None,
    ) -> tuple[list[str], npt.NDArray[np.float64]] | npt.NDArray[np.void]:
        """Apply the lineout to tabular data.

        Parameters
        ----------
        header_or_structured
            Either a sequence of column names or a structured NumPy array.
        data
            Dense data array when ``header_or_structured`` is a header.
        """

        if data is None:
            structured = np.asarray(header_or_structured)
            if structured.dtype.names is None:
                raise TypeError("single-argument apply expects a structured array")

            header = list(structured.dtype.names)
            dense = np.asarray(structured.tolist(), dtype=np.float64)
            output_header, output_dense = self._apply_dense(header, dense)

            dtype = np.dtype([(name, "f8") for name in output_header])
            return np.asarray(list(zip(*output_dense.T, strict=False)), dtype=dtype)

        header = list(header_or_structured)  # type: ignore[arg-type]
        dense = np.asarray(data, dtype=np.float64)
        return self._apply_dense(header, dense)

    def _apply_dense(
        self, header: list[str], data: npt.NDArray[np.float64]
    ) -> tuple[list[str], npt.NDArray[np.float64]]:
        if data.ndim != 2:
            raise ValueError("lineout data must be two-dimensional")

        output_header = list(header)
        output_data = np.asarray(data, dtype=np.float64).copy()

        start = 1 if output_header and output_header[0].lower() == "index" else 0
        dimension = _coordinate_dimension(output_header)
        coordinate_columns = list(range(start, start + dimension))

        if self.needs_displacements:
            output_header, output_data, coordinate_columns = _apply_displacements(
                output_header, output_data, start=start, dimension=dimension
            )

        tol = self.tol
        if tol is None and len(output_data):
            tol = self.compute_tol_from_bounding_box(output_data[:, coordinate_columns])
            self.tol = tol
        if tol is None:
            tol = 0.0

        sort_columns: list[int] = []
        remove_columns: list[int] = []

        for axis in range(dimension):
            spec = self.spec[axis]
            column = coordinate_columns[axis]

            if isinstance(spec, float):
                remove_columns.append(column)
                mask = np.abs(output_data[:, column] - spec) < tol
                output_data = output_data[mask]
            else:
                sort_columns.append(column)

        if sort_columns and len(output_data):
            output_data = np.asarray(
                sorted(output_data.tolist(), key=cmp_to_key(_line_compare(sort_columns))),
                dtype=np.float64,
            )

        if remove_columns:
            remove_columns = sorted(remove_columns, reverse=True)
            output_header = np.delete(
                np.asarray(output_header, dtype=object), remove_columns
            ).tolist()
            output_data = np.delete(output_data, remove_columns, axis=1)

        return output_header, output_data

    def compute_tol_from_bounding_box(self, data: npt.ArrayLike) -> float:
        """Compute default tolerance from restricted coordinate extents."""

        array = np.asarray(data, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("bounding-box data must be two-dimensional")
        if array.shape[0] == 0:
            return 0.0

        tolerances: list[float] = []
        for axis, spec in enumerate(self.spec[: array.shape[1]]):
            if isinstance(spec, float):
                lower = float(np.min(array[:, axis]))
                upper = float(np.max(array[:, axis]))
                tolerances.append(1.0e-4 * max(upper - lower, 1.0e-30))

        return min(tolerances) if tolerances else 0.0


def lineout(
    *,
    x: float | str | None = None,
    y: float | str | None = None,
    z: float | str | None = None,
    tol: float | None = None,
) -> Lineout:
    """Factory preserving the old lowercase constructor name."""

    return Lineout(x=x, y=y, z=z, tol=tol)


def _coordinate_dimension(header: Sequence[str]) -> int:
    upper = [name.upper() for name in header]
    if "COORDZ" in upper or "LOCATIONZ" in upper:
        return 3
    if "COORDY" in upper or "LOCATIONY" in upper:
        return 2
    if "COORDX" in upper or "LOCATIONX" in upper:
        return 1

    raise ValueError("lineout header does not contain coordinate columns")


def _apply_displacements(
    header: list[str], data: npt.NDArray[np.float64], *, start: int, dimension: int
) -> tuple[list[str], npt.NDArray[np.float64], list[int]]:
    upper = [name.upper() for name in header]

    if "DISPLX" not in upper:
        raise ValueError("lineout with displaced coordinates requires displacement columns")

    displacement_columns = list(range(start, start + dimension))
    coordinate_columns = list(range(start + dimension, start + 2 * dimension))

    output = data.copy()
    for displacement_column, coordinate_column in zip(
        displacement_columns, coordinate_columns, strict=True
    ):
        output[:, coordinate_column] += output[:, displacement_column]

    header_array = np.asarray(header, dtype=object)
    header_array = np.delete(header_array, displacement_columns)
    output = np.delete(output, displacement_columns, axis=1)

    for axis in range(dimension):
        header_array[start + axis] = f"LOCATION{'XYZ'[axis]}"

    return header_array.tolist(), output, list(range(start, start + dimension))


def _line_compare(columns: Sequence[int]):
    def compare(left: Sequence[float], right: Sequence[float]) -> int:
        for column in columns:
            result = _cmp(left[column], right[column])
            if result:
                return result
        return 0

    return compare


def _cmp(left: float, right: float) -> int:
    return int(left > right) - int(left < right)


__all__ = ["Lineout", "lineout"]
