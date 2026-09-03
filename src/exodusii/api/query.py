# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Tabular variable query helpers.

This module provides :func:`query` and :func:`print_query` for extracting
result variables from an Exodus database into structured NumPy arrays and
printing them as whitespace-delimited tables.

Variable selectors use the ``ENTITY/NAME`` format, where ``ENTITY`` is a
short entity code:

* ``g`` or ``global`` — global variables (one value per time step).
* ``n`` or ``node``   — nodal variables (one value per node per time step).
* ``e`` or ``element``— element variables (one value per element per time step).

When all selectors share the same entity type, :func:`query` returns a
:class:`QueryResult` whose ``data`` attribute is a structured NumPy array
with one column per selected variable (plus a ``TIME`` column for global
queries and, optionally, an ``index`` column for spatial queries).
"""

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
    """Structured result returned by :func:`query`.

    Wraps a structured NumPy array whose field names correspond to the
    queried variable names (and ``TIME`` for global queries, ``index`` when
    ``object_index=True``).

    Attributes
    ----------
    data : ndarray of void
        Structured NumPy array.  Each element is one row (one time step for
        global queries, or one mesh entity for spatial queries).  Field
        names match the requested variable names.

    Examples
    --------
    >>> result = query(exo, "g/ENERGY", time="last")
    >>> result.names
    ('TIME', 'ENERGY')
    >>> result.data["ENERGY"]
    array([1.23456789])
    """

    data: npt.NDArray[np.void]

    @property
    def names(self) -> tuple[str, ...]:
        """Return column names of the structured result array.

        Returns
        -------
        tuple of str
            Field names from ``data.dtype.names``, in column order.
            Returns an empty tuple when ``data`` has no named fields.
        """

        return tuple(self.data.dtype.names or ())

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata stored in the structured array dtype.

        The metadata dictionary is populated by :func:`query` and typically
        contains:

        * ``"entity"`` — the entity-type string (e.g. ``"global"``,
          ``"node"``, ``"element"``).
        * ``"time"`` — the resolved time value for spatial queries
          (absent for full-history global queries).

        Returns
        -------
        dict
            Copy of ``data.dtype.metadata``.  Empty dict when no metadata
            is present.
        """

        return dict(self.data.dtype.metadata or {})


def query(
    exo: ExodusFile,
    *variables: str | VariableSelector,
    time: TimeSelector = None,
    lineout: Lineout | None = None,
    object_index: bool = False,
) -> QueryResult:
    """Query Exodus variables into a structured array.

    Reads one or more result variables from *exo* and returns them as a
    structured NumPy array.  All selectors must refer to the same entity
    type.

    Parameters
    ----------
    exo : ExodusFile
        Open Exodus database to query.
    *variables : str or VariableSelector
        One or more variable selectors.  Each selector may be:

        * A qualified string of the form ``"ENTITY/NAME"``, e.g.
          ``"g/TM_STEP"``, ``"n/TEMP"``, ``"e/ENERGY"``.
        * A bare variable name (entity is inferred from context).
        * A :class:`~exodusii.core.selectors.VariableSelector` instance.

        The special names ``"coordinates"`` and ``"displacements"`` are
        supported for nodal queries and expand to per-axis columns.
    time : int, float, str, or None, optional
        Time-step selector.  Accepted values:

        * ``None`` (default) — return the full time history (global
          queries) or the last time step (spatial queries).
        * ``"first"`` — first time step.
        * ``"last"`` — last time step.
        * ``int`` — zero-based time-step index.
        * ``float`` — time value; the nearest step is selected.
    lineout : Lineout or None, optional
        When provided, the result array is filtered and sorted by
        :meth:`Lineout.apply` before being returned.  Useful for
        extracting a 1-D profile along a coordinate axis.
    object_index : bool, optional
        If ``True``, prepend a 1-based integer ``index`` column to the
        result array.  Applies to nodal and element queries only.
        Default ``False``.

    Returns
    -------
    QueryResult
        Structured result containing one row per time step (global) or one
        row per mesh entity (nodal/element).

    Raises
    ------
    ValueError
        When no variable selectors are provided or all selectors are empty.
    ExodusInvalidEntityError
        When the entity type inferred from the selectors is not supported
        (currently only global, nodal, and element queries are implemented).

    Examples
    --------
    Query global variables across all time steps:

    >>> result = query(exo, "g/TM_STEP", "g/ENERGY")
    >>> result.names
    ('TIME', 'TM_STEP', 'ENERGY')

    Query a nodal variable at a specific time:

    >>> result = query(exo, "n/TEMP", time="last")
    >>> result.data["TEMP"].shape
    (1024,)

    Query element variables with row indices:

    >>> result = query(exo, "e/STRESS", time=0, object_index=True)
    >>> result.names
    ('index', 'STRESS')
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
    """Print query results in a simple whitespace-delimited table.

    Calls :func:`query` with the given arguments and writes the result to a
    text stream as a fixed-width, whitespace-separated table with an
    optional header row.

    Parameters
    ----------
    exo : ExodusFile
        Open Exodus database to query.
    *variables : str or VariableSelector
        Variable selectors; forwarded to :func:`query`.
    time : int, float, str, or None, optional
        Time-step selector; forwarded to :func:`query`.
    lineout : Lineout or None, optional
        Lineout filter; forwarded to :func:`query`.
    object_index : bool, optional
        Prepend an ``index`` column; forwarded to :func:`query`.
        Default ``False``.
    labels : bool, optional
        If ``True`` (default), print a header row of right-aligned column
        names before the data rows.
    file : TextIO or None, optional
        Output stream.  Defaults to ``sys.stdout``.

    Notes
    -----
    Each numeric value is formatted as a 20-character field using the
    ``%20.16e`` format (16 significant digits, scientific notation).
    Column headers are right-aligned in 23-character fields to accommodate
    the wider data columns.

    Examples
    --------
    >>> print_query(exo, "g/ENERGY", "g/TM_STEP")
                    TIME                 ENERGY               TM_STEP
     0.0000000000000000e+00  1.0000000000000000e+00  ...
    """

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
