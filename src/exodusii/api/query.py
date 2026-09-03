# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Tabular variable query helpers."""

from dataclasses import dataclass
from typing import Any
from typing import TextIO
from typing import cast

import numpy as np
import numpy.typing as npt

from exodusii.api.file import ExodusFile
from exodusii.api.lineout import Lineout
from exodusii.core.entities import Entity
from exodusii.core.entities import entity
from exodusii.core.errors import ExodusInvalidEntityError
from exodusii.core.selectors import VariableSelector
from exodusii.core.selectors import parse_variable_selectors
from exodusii.core.time import TimeSelector
from exodusii.core.time import resolve_time


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Structured query result."""

    data: npt.NDArray[np.void]

    @property
    def names(self) -> tuple[str, ...]:
        """Column names."""

        return tuple(self.data.dtype.names or ())

    @property
    def metadata(self) -> dict[str, Any]:
        """Structured array metadata."""

        return dict(self.data.dtype.metadata or {})


def query(
    exo: ExodusFile,
    *variables: str | VariableSelector,
    time: TimeSelector = None,
    lineout: Lineout | None = None,
    object_index: bool = False,
) -> QueryResult:
    """Query Exodus variables into a structured array.

    Qualified selector strings use the form ``ENTITY/NAME`` such as
    ``"g/TM_STEP"``, ``"n/TEMP"``, or ``"e/ENERGY"``.
    """

    selectors = parse_variable_selectors(variables, require_same_entity=True)
    if not selectors:
        raise ValueError("at least one variable selector is required")

    location = entity(selectors[0].entity)

    if location is Entity.GLOBAL:
        data = _query_global(exo, selectors, time=time)
    elif location is Entity.NODE:
        data = _query_node(exo, selectors, time=time, object_index=object_index)
    elif location is Entity.ELEMENT:
        data = _query_element(exo, selectors, time=time, object_index=object_index)
    else:
        raise ExodusInvalidEntityError(f"query for {location.value!r} is not implemented")

    if lineout is not None:
        data = cast(npt.NDArray[np.void], lineout.apply(data))

    return QueryResult(data=data)


def print_query(
    exo: ExodusFile,
    *variables: str | VariableSelector,
    time: TimeSelector = None,
    lineout: Lineout | None = None,
    object_index: bool = False,
    labels: bool = True,
    file: TextIO | None = None,
) -> None:
    """Print query results in a simple whitespace-delimited table."""

    import sys

    result = query(exo, *variables, time=time, lineout=lineout, object_index=object_index)
    stream = file or sys.stdout

    names = result.names
    if labels:
        stream.write(" " + " ".join(f"{name:>23s}" for name in names) + "\n")

    for row in result.data:
        stream.write(" " + " ".join(f"{float(row[name]):20.16e}" for name in names) + "\n")


def _query_global(
    exo: ExodusFile, selectors: tuple[VariableSelector, ...], *, time: TimeSelector
) -> npt.NDArray[np.void]:
    times = exo.times()
    names = ["TIME"]

    if time is None:
        # Full time history for every selected global variable.
        columns: list[npt.NDArray[np.float64]] = [times]
        for selector in selectors:
            columns.append(exo.values(selector.name, on=Entity.GLOBAL))
            names.append(selector.name)
        dense = np.column_stack(columns)
    else:
        # Single time step: read only that step from each variable instead
        # of pulling the entire history and slicing.
        selection = resolve_time(times, time)
        row: list[float] = [float(times[selection.index])]
        for selector in selectors:
            value = exo.values(selector.name, on=Entity.GLOBAL, time=time)
            row.append(float(np.asarray(value)))
            names.append(selector.name)
        dense = np.asarray([row], dtype=np.float64)

    return _structured(names, dense, metadata={"entity": Entity.GLOBAL.value})


def _query_node(
    exo: ExodusFile,
    selectors: tuple[VariableSelector, ...],
    *,
    time: TimeSelector,
    object_index: bool,
) -> npt.NDArray[np.void]:
    selection = resolve_time(exo.times(), time)
    columns: list[npt.NDArray[np.float64]] = []
    names: list[str] = []

    if object_index:
        columns.append(np.arange(1, exo.node_count + 1, dtype=np.float64))
        names.append("index")

    for selector in selectors:
        if selector.name.lower() == "coordinates":
            coords = exo.coordinates()
            _append_components(columns, names, coords, ("COORDX", "COORDY", "COORDZ"))
        elif selector.name.lower() == "displacements":
            displacements = exo.displacements(time=selection.index)
            _append_components(columns, names, displacements, ("DISPLX", "DISPLY", "DISPLZ"))
        else:
            columns.append(
                np.asarray(
                    exo.values(selector.name, on=Entity.NODE, time=selection.index),
                    dtype=np.float64,
                )
            )
            names.append(selector.name)

    dense = np.column_stack(columns) if columns else np.empty((exo.node_count, 0), dtype=np.float64)
    return _structured(
        names, dense, metadata={"entity": Entity.NODE.value, "time": selection.value}
    )


def _query_element(
    exo: ExodusFile,
    selectors: tuple[VariableSelector, ...],
    *,
    time: TimeSelector,
    object_index: bool,
) -> npt.NDArray[np.void]:
    selection = resolve_time(exo.times(), time)
    element_count = exo.element_count
    columns: list[npt.NDArray[np.float64]] = []
    names: list[str] = []

    if object_index:
        columns.append(np.arange(1, element_count + 1, dtype=np.float64))
        names.append("index")

    for selector in selectors:
        columns.append(
            np.asarray(
                exo.values(selector.name, on=Entity.ELEMENT, time=selection.index), dtype=np.float64
            )
        )
        names.append(selector.name)

    dense = np.column_stack(columns) if columns else np.empty((element_count, 0), dtype=np.float64)
    return _structured(
        names, dense, metadata={"entity": Entity.ELEMENT.value, "time": selection.value}
    )


def _append_components(
    columns: list[npt.NDArray[np.float64]],
    names: list[str],
    values: npt.NDArray[np.float64],
    component_names: tuple[str, str, str],
) -> None:
    for axis in range(values.shape[1]):
        columns.append(np.asarray(values[:, axis], dtype=np.float64))
        names.append(component_names[axis])


def _structured(
    names: list[str], dense: npt.ArrayLike, *, metadata: dict[str, Any] | None = None
) -> npt.NDArray[np.void]:
    array = np.asarray(dense, dtype=np.float64)
    dtype = np.dtype([(name, "f8") for name in names], metadata=metadata or {})

    if array.size == 0:
        return np.empty((array.shape[0],), dtype=dtype)

    return np.asarray(list(zip(*array.T, strict=False)), dtype=dtype)


__all__ = ["QueryResult", "print_query", "query"]
