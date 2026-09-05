# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Comparison utilities for Exodus databases.

This module provides two levels of comparison:

* :func:`allclose` — numeric equality within configurable absolute and
  relative tolerances, operating directly on the underlying NetCDF
  dimensions and variables.
* :func:`similar` — structural equivalence check that verifies mesh
  topology, variable layout, block and set definitions, and connectivity
  without comparing result values.

Both functions accept open :class:`~exodusii.api.file.ExodusFile` objects or
file-system paths and handle file opening/closing automatically.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal
from typing import TextIO
from typing import overload

import numpy as np

from exodusii.api.file import ExodusFile
from exodusii.core.names import DimensionName

ExodusFileLike = ExodusFile | str | Path


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Detailed result of an :func:`allclose` comparison.

    Evaluates as a bool (``True`` when the comparison found no differences).

    Attributes
    ----------
    equal : bool
        ``True`` when no errors were detected.
    errors : tuple of str
        Descriptions of every dimension or variable that differed between
        the two files.  Empty when ``equal`` is ``True``.
    """

    equal: bool
    errors: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        """Return ``True`` when the comparison found no differences."""
        return self.equal


@overload
def allclose(
    file1: ExodusFileLike,
    file2: ExodusFileLike,
    *,
    atol: float = ...,
    rtol: float = ...,
    dimensions: bool | str | Iterable[str] | None = ...,
    variables: bool | str | Iterable[str] | None = ...,
    verbose: bool | TextIO = ...,
    result: Literal[True],
) -> ComparisonResult: ...


@overload
def allclose(
    file1: ExodusFileLike,
    file2: ExodusFileLike,
    *,
    atol: float = ...,
    rtol: float = ...,
    dimensions: bool | str | Iterable[str] | None = ...,
    variables: bool | str | Iterable[str] | None = ...,
    verbose: bool | TextIO = ...,
    result: Literal[False] = ...,
) -> bool: ...


def allclose(
    file1: ExodusFileLike,
    file2: ExodusFileLike,
    *,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-12,
    dimensions: bool | str | Iterable[str] | None = True,
    variables: bool | str | Iterable[str] | None = True,
    verbose: bool | TextIO = False,
    result: bool = False,
) -> bool | ComparisonResult:
    """Return true if two Exodus files are data-wise equal within tolerance.

    Parameters
    ----------
    file1, file2
        Open :class:`ExodusFile` objects or paths to Exodus files.
        If paths are provided the files are opened and closed automatically.
    atol : float, optional
        Absolute tolerance for numeric comparisons (default ``1e-12``).
    rtol : float, optional
        Relative tolerance for numeric comparisons (default ``1e-12``).
    dimensions : bool or str or iterable of str or None, optional
        Dimensions to compare.  ``True`` (default) compares all dimensions;
        ``False`` or ``None`` compares none.  A string of the form
        ``"~a|b"`` compares all dimensions *except* ``a`` and ``b``.
        A plain string ``"name"`` compares only that one dimension.
        An iterable of strings compares exactly those dimensions.
    variables : bool or str or iterable of str or None, optional
        Variables to compare.  Same selection rules as ``dimensions``.
    verbose : bool or TextIO, optional
        If ``True``, write error messages to ``sys.stderr``.  If a text
        stream, write errors there.  Default ``False`` (silent).
    result : bool, optional
        If ``True``, return a :class:`ComparisonResult` instead of a plain
        bool.  Default ``False``.

    Returns
    -------
    bool or ComparisonResult
        When ``result`` is ``False`` (default), returns a plain ``bool``
        that is ``True`` when the files are considered equal.  When
        ``result`` is ``True``, returns a :class:`ComparisonResult` with
        full error details.

    Examples
    --------
    Simple check with default tolerances:

    >>> allclose("before.exo", "after.exo")
    True

    Looser tolerances, only compare nodal variables:

    >>> allclose("a.exo", "b.exo", atol=1e-6, rtol=1e-6,
    ...          dimensions=False, variables="~elem_var1")
    True

    Collect detailed errors:

    >>> cr = allclose("a.exo", "b.exo", result=True)
    >>> if not cr:
    ...     for err in cr.errors:
    ...         print(err)
    """

    opened1 = _open_if_needed(file1)
    opened2 = _open_if_needed(file2)

    exo1, close1 = opened1
    exo2, close2 = opened2

    errors: list[str] = []

    try:
        dimension_names = _resolve_names(dimensions, exo1.dimensions(), exo2.dimensions())
        variable_names = _resolve_names(variables, exo1.variables(), exo2.variables())

        for name in dimension_names:
            if name == DimensionName.TIME.value:
                continue

            dim1 = exo1.backend.dimension(name, None)
            dim2 = exo2.backend.dimension(name, None)

            if dim1 is None:
                errors.append(f"dimension {name!r} not found in {exo1.path}")
            elif dim2 is None:
                errors.append(f"dimension {name!r} not found in {exo2.path}")
            elif dim1 != dim2:
                errors.append(f"dimension {name!r} differs: {dim1} != {dim2}")

        for name in variable_names:
            if not exo1.backend.has_variable(name):
                errors.append(f"variable {name!r} not found in {exo1.path}")
                continue
            if not exo2.backend.has_variable(name):
                errors.append(f"variable {name!r} not found in {exo2.path}")
                continue

            value1 = exo1.variable(name)
            value2 = exo2.variable(name)

            if not _values_close(value1, value2, atol=atol, rtol=rtol):
                errors.append(f"variable {name!r} differs")

    finally:
        if close1:
            exo1.close()
        if close2:
            exo2.close()

    _write_errors(errors, verbose)

    comparison = ComparisonResult(equal=not errors, errors=tuple(errors))
    return comparison if result else comparison.equal


def similar(
    file1: ExodusFileLike, file2: ExodusFileLike, times: Iterable[float] | None = None
) -> bool:
    """Return true if two Exodus files have the same mesh and variable layout.

    Checks that the two files are structurally compatible: identical spatial
    dimension, entity counts, variable name sets, block and set definitions,
    element connectivity, truth tables, and block/set status arrays.
    Result values (nodal and element variable data) are *not* compared.

    Parameters
    ----------
    file1, file2
        Open :class:`ExodusFile` objects or paths to Exodus files.
        If paths are provided the files are opened and closed automatically.
    times : iterable of float or None, optional
        When provided, each requested time value must be present (within
        absolute tolerance ``1e-12``) in *both* files.  If any value is
        absent from either file a :exc:`ValueError` is raised.

    Returns
    -------
    bool
        ``True`` when all structural checks pass.

    Raises
    ------
    ValueError
        When any structural check fails (differing counts, IDs, element
        types, connectivity, truth tables, or requested time values).

    Notes
    -----
    The checks performed are:

    * Spatial dimension and entity counts (nodes, edges, faces, elements,
      element blocks).
    * Node, element, edge, and face ID maps.
    * Variable name sets for every entity type (global, nodal, element,
      edge, face, all set types).
    * Nodal coordinates (via :func:`numpy.allclose` with default tolerances).
    * Element-block, edge-block, and face-block IDs, element types, entity
      counts, nodes-per-entity, and connectivity arrays.
    * Node-set, side-set, edge-set, face-set, and element-set IDs, entry
      counts, entries, and extra entries.
    * Variable truth tables for block and set variables.
    * Block and set active/inactive status arrays.
    * Element-block attribute names and values.

    Examples
    --------
    >>> similar("mesh_a.exo", "mesh_b.exo")
    True

    Require specific time steps to be present in both files:

    >>> similar("run1.exo", "run2.exo", times=[0.0, 0.5, 1.0])
    True
    """

    opened1 = _open_if_needed(file1)
    opened2 = _open_if_needed(file2)

    exo1, close1 = opened1
    exo2, close2 = opened2

    try:
        _compare_basic_counts(exo1, exo2)
        _compare_id_maps(exo1, exo2)
        _compare_variable_layout(exo1, exo2)
        _compare_coordinates(exo1, exo2)
        _compare_blocks(exo1, exo2)
        _compare_sets(exo1, exo2)
        _compare_truth_tables(exo1, exo2)
        _compare_status(exo1, exo2)
        _compare_attributes(exo1, exo2)

        if times is not None:
            _compare_times(exo1.times(), exo2.times(), times)

    finally:
        if close1:
            exo1.close()
        if close2:
            exo2.close()

    return True


def _compare_id_maps(exo1: ExodusFile, exo2: ExodusFile) -> None:
    for label in ("node", "element", "edge", "face"):
        ids1 = exo1.ids(label)
        ids2 = exo2.ids(label)
        if not np.array_equal(ids1, ids2):
            raise ValueError(f"files do not define the same {label} ID map")


def _compare_basic_counts(exo1: ExodusFile, exo2: ExodusFile) -> None:
    if exo1.dimension != exo2.dimension:
        raise ValueError("files do not have the same dimension")
    if exo1.node_count != exo2.node_count:
        raise ValueError("files do not have the same number of nodes")
    if exo1.edge_count != exo2.edge_count:
        raise ValueError("files do not have the same number of edges")
    if exo1.face_count != exo2.face_count:
        raise ValueError("files do not have the same number of faces")
    if exo1.element_count != exo2.element_count:
        raise ValueError("files do not have the same number of elements")
    if exo1.element_block_count != exo2.element_block_count:
        raise ValueError("files do not have the same number of element blocks")


def _compare_variable_layout(exo1: ExodusFile, exo2: ExodusFile) -> None:
    for label in (
        "global",
        "node",
        "element",
        "edge",
        "face",
        "node_set",
        "side_set",
        "edge_set",
        "face_set",
        "element_set",
    ):
        _compare_name_sets(exo1.variable_names(label), exo2.variable_names(label), label)


def _compare_coordinates(exo1: ExodusFile, exo2: ExodusFile) -> None:
    if exo1.node_count == 0 and exo2.node_count == 0:
        return

    if not np.allclose(exo1.coordinates(), exo2.coordinates()):
        raise ValueError("files do not have the same node coordinates")


def _compare_blocks(exo1: ExodusFile, exo2: ExodusFile) -> None:
    _compare_one_block_family(exo1, exo2, "element_block")
    _compare_one_block_family(exo1, exo2, "edge_block")
    _compare_one_block_family(exo1, exo2, "face_block")


def _compare_one_block_family(exo1: ExodusFile, exo2: ExodusFile, family: str) -> None:
    ids1 = exo1.block_ids(family)
    ids2 = exo2.block_ids(family)

    if not np.array_equal(ids1, ids2):
        raise ValueError(f"files do not define the same {family} IDs")

    for block_id in ids1:
        block1 = exo1.block(family, int(block_id))
        block2 = exo2.block(family, int(block_id))

        if block1.element_type != block2.element_type:
            raise ValueError(f"{family} {block_id} has different element type")
        if block1.count != block2.count:
            raise ValueError(f"{family} {block_id} has different entity count")
        if block1.nodes_per_entity != block2.nodes_per_entity:
            raise ValueError(f"{family} {block_id} has different nodes per entity")

        conn1 = exo1.block_connectivity(family, int(block_id))
        conn2 = exo2.block_connectivity(family, int(block_id))
        if not np.array_equal(conn1, conn2):
            raise ValueError(f"files do not have the same {family} connectivity")


def _compare_sets(exo1: ExodusFile, exo2: ExodusFile) -> None:
    for family in ("node_set", "side_set", "edge_set", "face_set", "element_set"):
        _compare_one_set_family(exo1, exo2, family)


def _compare_one_set_family(exo1: ExodusFile, exo2: ExodusFile, family: str) -> None:
    ids1 = exo1.set_ids(family)
    ids2 = exo2.set_ids(family)

    if not np.array_equal(ids1, ids2):
        raise ValueError(f"files do not define the same {family} IDs")

    for set_id in ids1:
        set1 = exo1.set(family, int(set_id))
        set2 = exo2.set(family, int(set_id))

        if set1.count != set2.count:
            raise ValueError(f"{family} {set_id} has different entry count")

        entries1 = np.asarray([] if set1.entries is None else set1.entries)
        entries2 = np.asarray([] if set2.entries is None else set2.entries)
        if not np.array_equal(entries1, entries2):
            raise ValueError(f"{family} {set_id} has different entries")

        extra1 = np.asarray([] if set1.extra_entries is None else set1.extra_entries)
        extra2 = np.asarray([] if set2.extra_entries is None else set2.extra_entries)
        if not np.array_equal(extra1, extra2):
            raise ValueError(f"{family} {set_id} has different extra entries")


def _compare_truth_tables(exo1: ExodusFile, exo2: ExodusFile) -> None:
    for family in (
        "element",
        "edge",
        "face",
        "node_set",
        "side_set",
        "edge_set",
        "face_set",
        "element_set",
    ):
        table1 = exo1.variable_truth_table(family)
        table2 = exo2.variable_truth_table(family)

        if table1 is None and table2 is None:
            continue
        if table1 is None or table2 is None:
            raise ValueError(f"files do not define the same {family} variable truth table")
        if not np.array_equal(table1, table2):
            raise ValueError(f"files do not define the same {family} variable truth table")


def _open_if_needed(file: Any) -> tuple[ExodusFile, bool]:
    if isinstance(file, ExodusFile):
        return file, False

    # Legacy ExodusIIFile adapter
    reader = getattr(file, "_reader", None)
    if isinstance(reader, ExodusFile):
        return reader, False

    # Parallel/modern objects are not raw ExodusFile but may already expose
    # the methods allclose needs poorly. For raw allclose, require an ExodusFile
    # or path.
    return ExodusFile.open(file), True


def _resolve_names(
    selection: bool | str | Iterable[str] | None, names1: Iterable[str], names2: Iterable[str]
) -> tuple[str, ...]:
    all_names = tuple(sorted(set(names1) | set(names2)))

    if selection is True:
        return all_names
    if selection is False or selection is None:
        return ()

    if isinstance(selection, str):
        if selection.startswith("~"):
            skipped = set(selection[1:].split("|"))
            return tuple(name for name in all_names if name not in skipped)

        return (selection,)

    return tuple(selection)


def _values_close(value1: Any, value2: Any, *, atol: float, rtol: float) -> bool:
    if value1 is None or value2 is None:
        return value1 is value2

    array1 = np.asarray(value1)
    array2 = np.asarray(value2)

    if array1.shape != array2.shape:
        return False

    if array1.dtype.kind in {"S", "U", "O"} or array2.dtype.kind in {"S", "U", "O"}:
        return np.array_equal(array1.astype(str), array2.astype(str))

    return bool(np.allclose(array1, array2, atol=atol, rtol=rtol))


def _write_errors(errors: Iterable[str], verbose: bool | TextIO) -> None:
    if not verbose:
        return

    if verbose is True:
        import sys

        stream = sys.stderr
    else:
        stream = verbose

    for error in errors:
        stream.write(f"==> Error: {error}\n")


def _compare_name_sets(names1: Iterable[str], names2: Iterable[str], label: str) -> None:
    set1 = {name.lower() for name in names1}
    set2 = {name.lower() for name in names2}

    if set1 != set2:
        raise ValueError(f"files do not define the same {label} variables")


def _compare_times(
    times1: np.ndarray, times2: np.ndarray, requested_times: Iterable[float]
) -> None:
    for requested in requested_times:
        found1 = bool(np.any(np.isclose(times1, requested, atol=1.0e-12, rtol=0.0)))
        found2 = bool(np.any(np.isclose(times2, requested, atol=1.0e-12, rtol=0.0)))

        if not found1 or not found2:
            raise ValueError("files do not contain the requested times")


def _compare_status(exo1: ExodusFile, exo2: ExodusFile) -> None:
    for family in ("element_block", "edge_block", "face_block"):
        status1 = exo1.block_status(family)
        status2 = exo2.block_status(family)
        if not np.array_equal(status1, status2):
            raise ValueError(f"files do not define the same {family} status")

    for family in ("node_set", "side_set", "edge_set", "face_set", "element_set"):
        status1 = exo1.set_status(family)
        status2 = exo2.set_status(family)
        if not np.array_equal(status1, status2):
            raise ValueError(f"files do not define the same {family} status")


def _compare_attributes(exo1: ExodusFile, exo2: ExodusFile) -> None:
    for family in ("element_block", "edge_block", "face_block"):
        _compare_one_attribute_family(exo1, exo2, family)


def _compare_one_attribute_family(exo1: ExodusFile, exo2: ExodusFile, family: str) -> None:
    ids = exo1.block_ids(family)

    for block_id in ids:
        names1 = exo1.attribute_names(family, int(block_id))
        names2 = exo2.attribute_names(family, int(block_id))

        if names1 != names2:
            raise ValueError(f"{family} {block_id} has different attribute names")

        attrs1 = exo1.attributes(family, int(block_id))
        attrs2 = exo2.attributes(family, int(block_id))

        if attrs1 is None and attrs2 is None:
            continue
        if attrs1 is None or attrs2 is None:
            raise ValueError(f"{family} {block_id} has different attributes")
        if not np.allclose(attrs1, attrs2):
            raise ValueError(f"{family} {block_id} has different attributes")


__all__ = ["ComparisonResult", "allclose", "similar"]
