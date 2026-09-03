# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Exodiff-style comparison of two Exodus databases.

This is a pure-Python, exodusii-based counterpart to the SEACAS ``exodiff``
tool.  It supports files whose mesh entities are in the same order (no
coordinate-based mesh matching) and compares mesh metadata, coordinates,
element attributes, and result variables (global, nodal, element, edge, face,
and set variables) using the :mod:`exodusii.core.tolerance` model.

The comparison is truth-table aware for block and set variables, reports
NaN mismatches as differences (matching exodiff's default ``ignore_nans``
off), and supports full time-step selection and linear interpolation
(Phase 2).

Time-step selection semantics mirror SEACAS exodiff:

* ``time_step_start`` / ``time_step_stop`` / ``time_step_increment`` select
  a range from file-2 (1-based; ``LAST`` sentinel ``-1`` for start selects
  the last step on both files).
* ``time_step_offset`` shifts file-2 step numbers by a constant to obtain the
  corresponding file-1 step (``file1_step = file2_step + offset``).
* ``time_value_scale`` / ``time_value_offset`` adjust file-1 time values
  before matching: ``t1_adj = t1 * scale + offset``.
* ``exclude_steps`` is a set of 1-based file-2 step numbers to skip.
* When ``interpolating`` is true, file-2 values at each file-1 time are
  obtained by linear interpolation between the two nearest file-2 steps.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np

from exodusii.api.file import ExodusFile
from exodusii.core.entities import Entity
from exodusii.core.tolerance import Tolerance
from exodusii.core.tolerance import ToleranceMode

__all__ = ["DiffOptions", "DiffResult", "TimeSelection", "VariableDiff", "diff"]

ExodusFileLike = ExodusFile | str | Path

# Result-variable entity locations exodiff compares.
_BLOCK_VAR_ENTITIES = (Entity.ELEMENT, Entity.EDGE, Entity.FACE)
_SET_VAR_ENTITIES = (
    Entity.NODE_SET,
    Entity.SIDE_SET,
    Entity.EDGE_SET,
    Entity.FACE_SET,
    Entity.ELEMENT_SET,
)

# Map a block-variable entity to the block entity holding its ids/truth table.
_BLOCK_LOCATION = {
    Entity.ELEMENT: Entity.ELEMENT_BLOCK,
    Entity.EDGE: Entity.EDGE_BLOCK,
    Entity.FACE: Entity.FACE_BLOCK,
}


@dataclass(frozen=True, slots=True)
class VariableDiff:
    """Worst difference found for a single variable at a single location.

    Each instance records the maximum scaled delta observed across all compared
    time steps and mesh entries for one variable, together with the location
    (time step, entry index, block or set id) at which that worst value
    occurred.

    Attributes
    ----------
    entity : str
        Mesh entity type string (e.g. ``"element"``, ``"node"``,
        ``"node_set"``).  Matches :attr:`exodusii.core.entities.Entity.value`.
    name : str
        Variable name as it appears in file-1.
    max_delta : float
        Largest scaled difference observed.  Set to ``inf`` when the arrays
        have a shape mismatch or when a NaN-vs-finite mismatch is detected.
    tolerance_mode : str
        Tolerance mode string used for this comparison (e.g. ``"relative"``,
        ``"absolute"``).  Matches
        :attr:`exodusii.core.tolerance.ToleranceMode.value`.
    exceeded : bool
        ``True`` when ``max_delta`` exceeds the configured tolerance threshold.
    time_index : int or None
        Zero-based time-step index of the worst entry.  ``None`` for
        global variables or when the location could not be determined.
    entry_index : int or None
        Zero-based index within the entity array (node index, element index,
        etc.) of the worst entry.  ``None`` when not applicable.
    block_id : int or None
        Element-, edge-, or face-block id of the worst entry.  ``None`` for
        non-block variables.
    set_id : int or None
        Node-set, side-set, edge-set, face-set, or element-set id of the worst
        entry.  ``None`` for non-set variables.
    value1 : float or None
        Raw value from file-1 at the worst location.  ``None`` when
        unavailable (e.g. shape mismatch).
    value2 : float or None
        Raw value from file-2 at the worst location.  ``None`` when
        unavailable.
    """

    entity: str
    name: str
    max_delta: float
    tolerance_mode: str
    exceeded: bool
    # Location of the worst value (best-effort; index within the entity).
    time_index: int | None = None
    entry_index: int | None = None
    block_id: int | None = None
    set_id: int | None = None
    value1: float | None = None
    value2: float | None = None


@dataclass(slots=True)
class DiffResult:
    """Outcome of comparing two Exodus databases.

    Returned by :func:`diff`.  Evaluates as a bool (``True`` when the
    databases are considered identical within tolerance).

    Attributes
    ----------
    same : bool
        ``True`` when no structural errors were found and no variable
        exceeded its tolerance.  Set by :func:`diff` after all comparisons
        are complete.
    file1 : str
        String path of the first database.
    file2 : str
        String path of the second database.
    errors : list of str
        Fatal structural mismatches: differing mesh dimensions, node or
        element counts, missing variables, truth-table presence mismatches,
        or coordinate shape differences.  Any non-empty errors list means
        ``same`` is ``False``.
    warnings : list of str
        Non-fatal notes: differing time-step counts or time-value mismatches.
        Warnings do not affect ``same``.
    variable_diffs : list of VariableDiff
        Per-variable worst-difference records.  By default only variables
        that exceeded tolerance are included; pass ``show_all=True`` to
        :class:`DiffOptions` to include all compared variables.
    coordinate_max_delta : float or None
        Maximum coordinate difference found during coordinate comparison,
        or ``None`` when coordinate comparison was skipped.
    """

    same: bool
    file1: str
    file2: str
    # Fatal structural errors (counts, missing variables, etc.).
    errors: list[str] = field(default_factory=list)
    # Non-fatal warnings.
    warnings: list[str] = field(default_factory=list)
    # Per-variable worst-difference records (only those that exceeded tol,
    # unless show_all is requested).
    variable_diffs: list[VariableDiff] = field(default_factory=list)
    # Coordinate worst difference, if compared.
    coordinate_max_delta: float | None = None

    def __bool__(self) -> bool:
        """Return ``True`` when the comparison found no differences."""
        return self.same


@dataclass(frozen=True, slots=True)
class TimeSelection:
    """Time-step selection and interpolation settings for :func:`diff`.

    All step numbers are **1-based** (matching SEACAS exodiff conventions).
    They refer to file-2 step indices; the corresponding file-1 step is
    ``file2_step + time_step_offset``.

    Parameters
    ----------
    start
        First file-2 step to compare (1-based, default 1).  The sentinel
        ``-1`` means "last step only" (``LAST``): the last step of both
        files is compared regardless of their indices.
    stop
        Last file-2 step to compare (inclusive, default ``-1`` = all).
    increment
        Step-index stride (default 1).
    time_step_offset
        Added to each selected file-2 step to obtain the file-1 step.
        ``file1_step = file2_step + time_step_offset``.
    exclude_steps
        Set of 1-based file-2 step numbers to skip entirely.
    time_value_scale
        Multiplier applied to file-1 time values before matching
        (``t1_adj = t1 * scale + offset``).
    time_value_offset
        Additive offset applied to file-1 time values before matching.
    interpolating
        If true, file-2 values at each selected file-1 time are obtained
        by linear interpolation between the surrounding file-2 steps.
        Steps whose file-1 time falls outside the file-2 time range are
        skipped.
    """

    start: int = 1
    stop: int = -1
    increment: int = 1
    time_step_offset: int = 0
    exclude_steps: frozenset[int] = field(default_factory=frozenset)
    time_value_scale: float = 1.0
    time_value_offset: float = 0.0
    interpolating: bool = False


@dataclass(frozen=True, slots=True)
class DiffOptions:
    """Options controlling a :func:`diff`.

    Parameters
    ----------
    default_tolerance
        Tolerance applied to result variables without a per-category or
        per-variable override.
    global_tolerance, nodal_tolerance, element_tolerance, edge_tolerance,
    face_tolerance, node_set_tolerance, side_set_tolerance,
    edge_set_tolerance, face_set_tolerance, element_set_tolerance,
    attribute_tolerance
        Per-category default tolerances.  If ``None`` (default),
        ``default_tolerance`` is used for that category.
    coordinate_tolerance
        Tolerance for nodal coordinates (exodiff default: absolute 1e-6).
    time_tolerance
        Tolerance for matching/comparing time values.
    variable_tolerances
        Per-variable tolerance overrides, keyed by variable name (case
        handling follows ``ignore_case``).  These take precedence over the
        per-category and default tolerances.
    exclude
        Variable names to exclude from comparison.
    ignore_case
        If true, variable-name matching is case-insensitive (exodiff default).
    time_step_offset
        Offset added to file-2 step indices to obtain file-1 step indices.
        Kept for backward compatibility; prefer ``time_selection.time_step_offset``.
        If both are set, ``time_selection.time_step_offset`` takes precedence.
    time_selection
        Full time-step selection and interpolation settings.  If ``None``
        (default), a :class:`TimeSelection` with ``time_step_offset`` applied
        is used.
    compare_coordinates
        If true, compare nodal coordinates.
    compare_attributes
        If true, compare element/edge/face block attributes.
    show_all
        If true, retain a record for every compared variable, not only those
        exceeding tolerance.
    """

    default_tolerance: Tolerance = field(
        default_factory=lambda: Tolerance(ToleranceMode.RELATIVE, 1.0e-6, 0.0)
    )
    global_tolerance: Tolerance | None = None
    nodal_tolerance: Tolerance | None = None
    element_tolerance: Tolerance | None = None
    edge_tolerance: Tolerance | None = None
    face_tolerance: Tolerance | None = None
    node_set_tolerance: Tolerance | None = None
    side_set_tolerance: Tolerance | None = None
    edge_set_tolerance: Tolerance | None = None
    face_set_tolerance: Tolerance | None = None
    element_set_tolerance: Tolerance | None = None
    attribute_tolerance: Tolerance | None = None
    coordinate_tolerance: Tolerance = field(
        default_factory=lambda: Tolerance(ToleranceMode.ABSOLUTE, 1.0e-6, 0.0)
    )
    time_tolerance: Tolerance = field(
        default_factory=lambda: Tolerance(ToleranceMode.RELATIVE, 1.0e-6, 1.0e-15)
    )
    variable_tolerances: Mapping[str, Tolerance] = field(default_factory=dict)
    exclude: frozenset[str] = field(default_factory=frozenset)
    ignore_case: bool = True
    time_step_offset: int = 0
    time_selection: TimeSelection | None = None
    compare_coordinates: bool = True
    compare_attributes: bool = True
    show_all: bool = False

    def _effective_time_selection(self) -> TimeSelection:
        """Return the active :class:`TimeSelection`, merging legacy offset."""
        if self.time_selection is not None:
            return self.time_selection
        return TimeSelection(time_step_offset=self.time_step_offset)

    def _category_default(self, ent: Entity) -> Tolerance:
        mapping = {
            Entity.GLOBAL: self.global_tolerance,
            Entity.NODE: self.nodal_tolerance,
            Entity.ELEMENT: self.element_tolerance,
            Entity.EDGE: self.edge_tolerance,
            Entity.FACE: self.face_tolerance,
            Entity.NODE_SET: self.node_set_tolerance,
            Entity.SIDE_SET: self.side_set_tolerance,
            Entity.EDGE_SET: self.edge_set_tolerance,
            Entity.FACE_SET: self.face_set_tolerance,
            Entity.ELEMENT_SET: self.element_set_tolerance,
        }
        category = mapping.get(ent)
        return category if category is not None else self.default_tolerance

    def tolerance_for(self, name: str, ent: Entity) -> Tolerance:
        """Return the tolerance to use for a variable name at a location.

        Per-variable overrides win over per-category defaults, which win over
        the global default.
        """

        if self.variable_tolerances:
            if self.ignore_case:
                lowered = {k.lower(): v for k, v in self.variable_tolerances.items()}
                found = lowered.get(name.lower())
            else:
                found = self.variable_tolerances.get(name)
            if found is not None:
                return found
        return self._category_default(ent)

    def is_excluded(self, name: str) -> bool:
        """Return ``True`` if *name* is excluded from comparison.

        Exclusion is case-insensitive when :attr:`ignore_case` is ``True``
        (the default).

        Parameters
        ----------
        name : str
            Variable name to check against the :attr:`exclude` set.

        Returns
        -------
        bool
            ``True`` when *name* matches an entry in :attr:`exclude`
            (respecting the :attr:`ignore_case` setting), ``False`` otherwise.

        Examples
        --------
        >>> opts = DiffOptions(exclude=frozenset({"TEMP", "PRESS"}))
        >>> opts.is_excluded("temp")
        True
        >>> opts.is_excluded("VELOCITY")
        False
        """

        if not self.exclude:
            return False
        if self.ignore_case:
            return name.lower() in {n.lower() for n in self.exclude}
        return name in self.exclude


def _open_if_needed(file: ExodusFileLike) -> tuple[ExodusFile, bool]:
    if isinstance(file, ExodusFile):
        return file, False
    return ExodusFile.open(file), True


def _match_names(
    names1: Iterable[str], names2: Iterable[str], *, ignore_case: bool
) -> tuple[list[str], list[str], list[str]]:
    """Return (common, only1, only2) preserving file-1 order for common names."""

    list1 = list(names1)
    list2 = list(names2)
    if ignore_case:
        lower2 = {n.lower() for n in list2}
        lower1 = {n.lower() for n in list1}
        common = [n for n in list1 if n.lower() in lower2]
        only1 = [n for n in list1 if n.lower() not in lower2]
        only2 = [n for n in list2 if n.lower() not in lower1]
    else:
        set2 = set(list2)
        set1 = set(list1)
        common = [n for n in list1 if n in set2]
        only1 = [n for n in list1 if n not in set2]
        only2 = [n for n in list2 if n not in set1]
    return common, only1, only2


def _worst(delta: np.ndarray) -> tuple[float, tuple[int, ...]]:
    """Return the maximum delta and its multi-index."""

    if delta.size == 0:
        return 0.0, ()
    flat_index = int(np.nanargmax(delta)) if np.any(~np.isnan(delta)) else 0
    index = np.unravel_index(flat_index, delta.shape)
    value = float(delta.flat[flat_index])
    return value, tuple(int(i) for i in index)


def _nan_mismatch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a boolean mask where NaN presence differs between a and b."""

    return np.isnan(a) != np.isnan(b)


def _compare_variable_series(
    values1: np.ndarray,
    values2: np.ndarray,
    tol: Tolerance,
    *,
    ent: Entity,
    name: str,
    block_id: int | None,
    set_id: int | None,
) -> VariableDiff | None:
    """Compare two arrays (possibly multi-step) for one variable."""

    a = np.asarray(values1, dtype=np.float64)
    b = np.asarray(values2, dtype=np.float64)

    if a.shape != b.shape:
        return VariableDiff(
            entity=ent.value,
            name=name,
            max_delta=float("inf"),
            tolerance_mode=tol.mode.value,
            exceeded=True,
            block_id=block_id,
            set_id=set_id,
        )

    delta = tol.delta_array(a, b)

    # NaN mismatches are always differences (ignore_nans is off by default).
    nan_mask = _nan_mismatch(a, b)
    # A finite delta comparison, plus NaN mismatches.
    diff_mask = tol.diff_array(a, b) | nan_mask
    exceeded = bool(np.any(diff_mask))

    # Report the worst delta location; prefer a NaN-mismatch entry if present
    # (its delta is meaningless, so mark it as infinite).
    if np.any(nan_mask):
        index = tuple(int(i) for i in np.argwhere(nan_mask)[0])
        max_delta = float("inf")
    else:
        max_delta, index = _worst(delta)

    time_index: int | None = None
    entry_index: int | None = None
    v1: float | None = None
    v2: float | None = None
    if index:
        v1 = float(a[index])
        v2 = float(b[index])
        if a.ndim >= 2:
            time_index = index[0]
            entry_index = index[1] if len(index) > 1 else None
        else:
            entry_index = index[0]

    return VariableDiff(
        entity=ent.value,
        name=name,
        max_delta=max_delta,
        tolerance_mode=tol.mode.value,
        exceeded=exceeded,
        time_index=time_index,
        entry_index=entry_index,
        block_id=block_id,
        set_id=set_id,
        value1=v1,
        value2=v2,
    )


def diff(
    file1: ExodusFileLike, file2: ExodusFileLike, options: DiffOptions | None = None
) -> DiffResult:
    """Compare two Exodus databases exodiff-style (matched mesh ordering).

    Performs a field-by-field comparison of two Exodus databases using the
    same default tolerances and time-step selection logic as the SEACAS
    ``exodiff`` tool.  Both files must share the same mesh topology (identical
    node/element ordering); coordinate-based mesh matching is not performed.

    Parameters
    ----------
    file1, file2
        Open :class:`ExodusFile` objects or paths to Exodus files.  If paths
        are provided, the files are opened and closed automatically.
    options
        Comparison options; defaults to :class:`DiffOptions` (relative
        tolerance 1e-6 for variables, absolute 1e-6 for coordinates).

    Returns
    -------
    DiffResult
        Structured comparison outcome.  ``result.same`` is ``True`` when no
        structural error was found and no variable exceeded tolerance.
        Evaluates to a bool directly via ``__bool__``.

    Examples
    --------
    Simple comparison with default tolerances:

    >>> result = diff("run_a.exo", "run_b.exo")
    >>> if not result:
    ...     for vd in result.variable_diffs:
    ...         print(vd.name, vd.max_delta)

    Tighter tolerance on nodal variables:

    >>> from exodusii.core.tolerance import Tolerance, ToleranceMode
    >>> opts = DiffOptions(
    ...     nodal_tolerance=Tolerance(ToleranceMode.ABSOLUTE, 1e-10, 0.0)
    ... )
    >>> result = diff("run_a.exo", "run_b.exo", options=opts)

    Compare only the last time step using :class:`TimeSelection`:

    >>> opts = DiffOptions(time_selection=TimeSelection(start=-1))
    >>> result = diff("run_a.exo", "run_b.exo", options=opts)
    """

    opts = options or DiffOptions()

    exo1, close1 = _open_if_needed(file1)
    exo2, close2 = _open_if_needed(file2)

    result = DiffResult(same=True, file1=str(exo1.path), file2=str(exo2.path))

    try:
        _compare_mesh_metadata(exo1, exo2, result)
        if opts.compare_coordinates:
            _compare_coordinates(exo1, exo2, opts, result)
        _compare_times(exo1, exo2, opts, result)
        if opts.compare_attributes:
            _compare_attributes(exo1, exo2, opts, result)
        _compare_all_variables(exo1, exo2, opts, result)
    finally:
        if close1:
            exo1.close()
        if close2:
            exo2.close()

    result.same = not result.errors and not any(vd.exceeded for vd in result.variable_diffs)
    return result


def _compare_mesh_metadata(exo1: ExodusFile, exo2: ExodusFile, result: DiffResult) -> None:
    if exo1.dimension != exo2.dimension:
        result.errors.append(f"dimension differs: {exo1.dimension} != {exo2.dimension}")
    if exo1.node_count != exo2.node_count:
        result.errors.append(f"node count differs: {exo1.node_count} != {exo2.node_count}")
    if exo1.element_count != exo2.element_count:
        result.errors.append(f"element count differs: {exo1.element_count} != {exo2.element_count}")
    ids1 = exo1.element_block_ids().tolist()
    ids2 = exo2.element_block_ids().tolist()
    if ids1 != ids2:
        result.errors.append(f"element block ids differ: {ids1} != {ids2}")


def _compare_coordinates(
    exo1: ExodusFile, exo2: ExodusFile, opts: DiffOptions, result: DiffResult
) -> None:
    if opts.coordinate_tolerance.mode is ToleranceMode.IGNORE:
        return
    if exo1.node_count != exo2.node_count:
        return
    coords1 = exo1.coordinates()
    coords2 = exo2.coordinates()
    if coords1.shape != coords2.shape:
        result.errors.append("coordinate shapes differ")
        return
    delta = opts.coordinate_tolerance.delta_array(coords1, coords2)
    max_delta, _ = _worst(delta)
    result.coordinate_max_delta = max_delta
    if np.any(opts.coordinate_tolerance.diff_array(coords1, coords2)):
        result.errors.append(f"coordinates differ (max delta {max_delta:.6e})")


def _compare_times(
    exo1: ExodusFile, exo2: ExodusFile, opts: DiffOptions, result: DiffResult
) -> None:
    """Warn when time step counts or matched time values differ."""
    times1 = exo1.times()
    times2 = exo2.times()
    n1 = len(times1)
    n2 = len(times2)
    if n1 != n2:
        result.warnings.append(f"time step count differs: {n1} != {n2}")

    ts = opts._effective_time_selection()
    steps = _steps_to_compare(exo1, exo2, opts, result)
    if ts.interpolating:
        # When interpolating, time-value warnings are suppressed: file-1 times
        # are matched to arbitrary file-2 times; mismatches are expected.
        return
    for i1, i2, _prop in steps:
        t1 = float(times1[i1]) * ts.time_value_scale + ts.time_value_offset
        t2 = float(times2[i2])
        if opts.time_tolerance.diff(t1, t2):
            result.warnings.append(
                f"time value differs at step {i1 + 1}: {times1[i1]:.6e} != {times2[i2]:.6e}"
            )


def _surrounding_steps(
    t: float, times2: np.ndarray
) -> tuple[int, int, float]:
    """Return ``(lo, hi, proportion)`` bracketing ``t`` in ``times2``.

    ``proportion`` satisfies ``times2[lo] + proportion * (times2[hi] - times2[lo]) == t``.
    Returns ``(-1, -1, 0.0)`` when ``t`` is outside the range.
    ``lo == hi`` when ``t`` exactly matches a step.
    """
    n = len(times2)
    if n == 0:
        return -1, -1, 0.0
    if t < float(times2[0]):
        return -1, -1, 0.0
    if t > float(times2[-1]):
        return -1, -1, 0.0

    # Binary search for bracketing interval.
    lo, hi = 0, n - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if float(times2[mid]) <= t:
            lo = mid
        else:
            hi = mid

    t_lo = float(times2[lo])
    t_hi = float(times2[hi])
    if lo == hi or t_hi == t_lo:
        return lo, lo, 0.0
    if t == t_lo:
        return lo, lo, 0.0
    if t == t_hi:
        return hi, hi, 0.0
    prop = (t - t_lo) / (t_hi - t_lo)
    return lo, hi, prop


# A step triple is (file1_0based_index, file2_0based_index_or_lo, proportion).
# proportion == 0.0  -> exact match (use file2 step at index directly).
# proportion  > 0.0  -> interpolate between file2[index] and file2[index+1].
StepTriple = tuple[int, int, float]


def _steps_to_compare(
    exo1: ExodusFile,
    exo2: ExodusFile,
    opts: DiffOptions,
    result: DiffResult,
) -> list[StepTriple]:
    """Return the list of (file1_idx, file2_lo_idx, proportion) triples.

    Implements full SEACAS exodiff time-step selection logic.
    """
    times1 = exo1.times()
    times2 = exo2.times()
    n1 = len(times1)
    n2 = len(times2)
    ts = opts._effective_time_selection()

    # ---- LAST sentinel: compare only the final step on each file.
    if ts.start == -1:
        if n1 == 0 or n2 == 0:
            result.warnings.append("no time steps to compare (LAST requested but files empty)")
            return []
        return [(n1 - 1, n2 - 1, 0.0)]

    # ---- Determine file-2 range [start2, stop2] (1-based, inclusive).
    start2 = max(1, ts.start)
    stop2 = ts.stop if ts.stop > 0 else n2
    stop2 = min(stop2, n2)
    offset = ts.time_step_offset

    if start2 > stop2:
        result.warnings.append(
            f"time step selection [{ts.start}, {ts.stop}] results in no steps to compare"
        )
        return []

    triples: list[StepTriple] = []
    step2 = start2
    while step2 <= stop2:
        if step2 not in ts.exclude_steps:
            step1 = step2 + offset  # 1-based file-1 step
            if 1 <= step1 <= n1:
                i1 = step1 - 1  # 0-based
                i2 = step2 - 1  # 0-based

                if ts.interpolating:
                    # Find surrounding file-2 steps for the (possibly scaled) file-1 time.
                    t1_adj = float(times1[i1]) * ts.time_value_scale + ts.time_value_offset
                    lo, hi, prop = _surrounding_steps(t1_adj, times2)
                    if lo == -1:
                        # Outside file-2 time range: skip.
                        step2 += ts.increment
                        continue
                    triples.append((i1, lo, prop if lo != hi else 0.0))
                else:
                    triples.append((i1, i2, 0.0))
        step2 += ts.increment

    return triples


def _compare_attributes(
    exo1: ExodusFile, exo2: ExodusFile, opts: DiffOptions, result: DiffResult
) -> None:
    tol = opts.attribute_tolerance or opts.default_tolerance
    if tol.mode is ToleranceMode.IGNORE:
        return
    for block_entity, value_entity in (
        (Entity.ELEMENT_BLOCK, Entity.ELEMENT),
        (Entity.EDGE_BLOCK, Entity.EDGE),
        (Entity.FACE_BLOCK, Entity.FACE),
    ):
        ids1 = exo1.block_ids(block_entity).tolist()
        for block_id in ids1:
            try:
                names1 = exo1.attribute_names(block_entity, block_id)
                names2 = exo2.attribute_names(block_entity, block_id)
            except Exception:
                continue
            common, _only1, _only2 = _match_names(names1, names2, ignore_case=opts.ignore_case)
            for attr_name in common:
                if opts.is_excluded(attr_name):
                    continue
                try:
                    col1 = exo1.attribute_values(block_entity, block_id, attr_name)
                    col2 = exo2.attribute_values(block_entity, block_id, attr_name)
                except Exception as exc:
                    result.errors.append(f"attribute {attr_name!r} block {block_id}: {exc}")
                    continue
                vd = _compare_variable_series(
                    col1,
                    col2,
                    tol,
                    ent=value_entity,
                    name=f"attr:{attr_name}",
                    block_id=block_id,
                    set_id=None,
                )
                _record(result, opts, vd)


def _compare_all_variables(
    exo1: ExodusFile, exo2: ExodusFile, opts: DiffOptions, result: DiffResult
) -> None:
    steps = _steps_to_compare(exo1, exo2, opts, result)

    _compare_global_variables(exo1, exo2, opts, result, steps)
    _compare_nodal_variables(exo1, exo2, opts, result, steps)
    for ent in _BLOCK_VAR_ENTITIES:
        _compare_block_variables(exo1, exo2, opts, result, steps, ent)
    for ent in _SET_VAR_ENTITIES:
        _compare_set_variables(exo1, exo2, opts, result, steps, ent)


def _record(result: DiffResult, opts: DiffOptions, vd: VariableDiff | None) -> None:
    if vd is None:
        return
    if vd.exceeded or opts.show_all:
        result.variable_diffs.append(vd)


def _interp_values(
    exo: ExodusFile,
    name: str,
    *,
    on: Entity,
    lo: int,
    hi: int,
    prop: float,
    block_id: int | None = None,
    set_id: int | None = None,
) -> np.ndarray:
    """Fetch values at a (possibly interpolated) step in ``exo``.

    When ``prop == 0``, returns the values at step ``lo`` directly.
    When ``prop > 0``, linearly interpolates between steps ``lo`` and ``hi``.
    """
    kw: dict = {"on": on, "time": lo}
    if block_id is not None:
        kw["block_id"] = block_id
    if set_id is not None:
        kw["set_id"] = set_id
    v_lo = np.asarray(exo.values(name, **kw), dtype=np.float64)
    if prop == 0.0 or lo == hi:
        return v_lo
    kw["time"] = hi
    v_hi = np.asarray(exo.values(name, **kw), dtype=np.float64)
    return v_lo + prop * (v_hi - v_lo)


def _names_to_compare(
    exo1: ExodusFile, exo2: ExodusFile, ent: Entity, opts: DiffOptions, result: DiffResult
) -> list[str]:
    names1 = exo1.variable_names(ent)
    names2 = exo2.variable_names(ent)
    common, only1, only2 = _match_names(names1, names2, ignore_case=opts.ignore_case)
    for name in only1:
        if not opts.is_excluded(name):
            result.errors.append(f"{ent.value} variable {name!r} missing from file2")
    for name in only2:
        if not opts.is_excluded(name):
            result.errors.append(f"{ent.value} variable {name!r} missing from file1")
    return [n for n in common if not opts.is_excluded(n)]


def _compare_global_variables(
    exo1: ExodusFile,
    exo2: ExodusFile,
    opts: DiffOptions,
    result: DiffResult,
    steps: list[StepTriple],
) -> None:
    for name in _names_to_compare(exo1, exo2, Entity.GLOBAL, opts, result):
        tol = opts.tolerance_for(name, Entity.GLOBAL)
        values1 = np.array(
            [float(exo1.values(name, on=Entity.GLOBAL, time=i1)) for i1, _lo, _p in steps]
        )
        values2 = np.array(
            [
                float(_interp_values(exo2, name, on=Entity.GLOBAL, lo=lo, hi=lo + (1 if p > 0 else 0), prop=p))
                for _i1, lo, p in steps
            ]
        )
        vd = _compare_variable_series(
            values1, values2, tol, ent=Entity.GLOBAL, name=name, block_id=None, set_id=None
        )
        _record(result, opts, vd)


def _compare_nodal_variables(
    exo1: ExodusFile,
    exo2: ExodusFile,
    opts: DiffOptions,
    result: DiffResult,
    steps: list[StepTriple],
) -> None:
    for name in _names_to_compare(exo1, exo2, Entity.NODE, opts, result):
        tol = opts.tolerance_for(name, Entity.NODE)
        rows1 = [exo1.values(name, on=Entity.NODE, time=i1) for i1, _lo, _p in steps]
        rows2 = [
            _interp_values(exo2, name, on=Entity.NODE, lo=lo, hi=lo + (1 if p > 0 else 0), prop=p)
            for _i1, lo, p in steps
        ]
        vd = _compare_variable_series(
            np.array(rows1),
            np.array(rows2),
            tol,
            ent=Entity.NODE,
            name=name,
            block_id=None,
            set_id=None,
        )
        _record(result, opts, vd)


def _variable_present(
    exo: ExodusFile, ent: Entity, name: str, object_id: int, name_index: int
) -> bool:
    """Return whether a block/set variable is present per the truth table.

    ``ent`` is the *variable* entity (e.g. ``Entity.ELEMENT`` or
    ``Entity.NODE_SET``); ``object_id`` is the block or set id.
    """

    try:
        row = exo.variable_truth_table(ent, id=object_id)
    except Exception:
        return False
    if row is None:
        return True
    if name_index < 0 or name_index >= len(row):
        return True
    return bool(row[name_index])


def _name_index(names: tuple[str, ...], name: str, *, ignore_case: bool) -> int:
    if ignore_case:
        lowered = name.lower()
        for i, candidate in enumerate(names):
            if candidate.lower() == lowered:
                return i
    else:
        for i, candidate in enumerate(names):
            if candidate == name:
                return i
    return -1


def _compare_block_variables(
    exo1: ExodusFile,
    exo2: ExodusFile,
    opts: DiffOptions,
    result: DiffResult,
    steps: list[StepTriple],
    ent: Entity,
) -> None:
    names = _names_to_compare(exo1, exo2, ent, opts, result)
    if not names:
        return
    block_entity = _BLOCK_LOCATION[ent]
    block_ids = exo1.block_ids(block_entity).tolist()
    names1 = exo1.variable_names(ent)
    names2 = exo2.variable_names(ent)
    for name in names:
        tol = opts.tolerance_for(name, ent)
        idx1 = _name_index(names1, name, ignore_case=opts.ignore_case)
        idx2 = _name_index(names2, name, ignore_case=opts.ignore_case)
        for block_id in block_ids:
            present1 = _variable_present(exo1, ent, name, block_id, idx1)
            present2 = _variable_present(exo2, ent, name, block_id, idx2)
            if not present1 and not present2:
                continue
            if present1 != present2:
                result.errors.append(
                    f"{ent.value} variable {name!r} block {block_id}: truth-table presence differs"
                )
                continue
            try:
                rows1 = [
                    exo1.values(name, on=ent, block_id=block_id, time=i1) for i1, _lo, _p in steps
                ]
                rows2 = [
                    _interp_values(
                        exo2, name, on=ent, lo=lo, hi=lo + (1 if p > 0 else 0), prop=p,
                        block_id=block_id,
                    )
                    for _i1, lo, p in steps
                ]
            except Exception as exc:
                result.errors.append(f"{ent.value} variable {name!r} block {block_id}: {exc}")
                continue
            vd = _compare_variable_series(
                np.array(rows1),
                np.array(rows2),
                tol,
                ent=ent,
                name=name,
                block_id=block_id,
                set_id=None,
            )
            _record(result, opts, vd)


def _compare_set_variables(
    exo1: ExodusFile,
    exo2: ExodusFile,
    opts: DiffOptions,
    result: DiffResult,
    steps: list[StepTriple],
    ent: Entity,
) -> None:
    names = _names_to_compare(exo1, exo2, ent, opts, result)
    if not names:
        return
    set_ids = exo1.set_ids(ent).tolist()
    names1 = exo1.variable_names(ent)
    names2 = exo2.variable_names(ent)
    for name in names:
        tol = opts.tolerance_for(name, ent)
        idx1 = _name_index(names1, name, ignore_case=opts.ignore_case)
        idx2 = _name_index(names2, name, ignore_case=opts.ignore_case)
        for set_id in set_ids:
            present1 = _variable_present(exo1, ent, name, set_id, idx1)
            present2 = _variable_present(exo2, ent, name, set_id, idx2)
            if not present1 and not present2:
                continue
            if present1 != present2:
                result.errors.append(
                    f"{ent.value} variable {name!r} set {set_id}: truth-table presence differs"
                )
                continue
            try:
                rows1 = [
                    exo1.values(name, on=ent, set_id=set_id, time=i1) for i1, _lo, _p in steps
                ]
                rows2 = [
                    _interp_values(
                        exo2, name, on=ent, lo=lo, hi=lo + (1 if p > 0 else 0), prop=p,
                        set_id=set_id,
                    )
                    for _i1, lo, p in steps
                ]
            except Exception as exc:
                result.errors.append(f"{ent.value} variable {name!r} set {set_id}: {exc}")
                continue
            vd = _compare_variable_series(
                np.array(rows1),
                np.array(rows2),
                tol,
                ent=ent,
                name=name,
                block_id=None,
                set_id=set_id,
            )
            _record(result, opts, vd)
