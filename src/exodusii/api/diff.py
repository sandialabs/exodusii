# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Exodiff-style comparison of two Exodus databases.

This is a pure-Python, exodusii-based counterpart to the SEACAS ``exodiff``
tool.  It supports both matched mesh ordering and coordinate-based mesh
matching (Phase 4), and compares mesh metadata, coordinates, element
attributes, and result variables (global, nodal, element, edge, face, and set
variables) using the :mod:`exodusii.core.tolerance` model.

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

Coordinate-based mesh matching (Phase 4):

* When ``DiffOptions.coordinate_matching`` is ``True``, a
  :class:`~exodusii.mesh.matching.MeshMap` is built before comparison by
  matching element centroids and nodes by spatial proximity.
* The matching algorithm mirrors SEACAS ``exodiff`` ``map.C``: centroid
  matching using a sorted-axis binary search, node map derivation from
  matched element local nodes, and a free-node fallback pass.
* Sideset face ordinals are checked after element remapping; mismatches
  are emitted as warnings (not fatal errors).
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path

import numpy as np

from exodusii.api.file import ExodusFile
from exodusii.core.entities import Entity
from exodusii.core.tolerance import Tolerance
from exodusii.core.tolerance import ToleranceMode
from exodusii.mesh.matching import MeshMap
from exodusii.mesh.matching import MeshMatchError
from exodusii.mesh.matching import build_mesh_map
from exodusii.mesh.matching import check_sideset_ordinals

__all__ = [
    "STEP_OFFSET_AUTOMATIC",
    "STEP_OFFSET_MATCH",
    "DiffOptions",
    "DiffResult",
    "TimeSelection",
    "VariableDiff",
    "diff",
]

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
        Zero-based file-1 time-step index of the worst entry.  ``None`` for
        non-transient data (attributes, distribution factors).
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
        Fatal structural mismatches (SEACAS exodiff exits 1 for these):
        differing mesh dimensions, node or element counts, element types,
        connectivity, ID maps that cannot be matched, missing variables,
        truth-table presence mismatches, or coordinate shape differences.
        Any non-empty errors list means ``same`` is ``False``.
    differences : list of str
        Non-variable findings that make the files *different* without being
        structural errors (exodiff exit status 2): time values that differ
        beyond the time tolerance, differing numbers of time steps, NaN
        values, and ID maps that differ in file-order mode.
    warnings : list of str
        Non-fatal notes.  Warnings do not affect ``same``.
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
    differences: list[str] = field(default_factory=list)
    # Non-fatal warnings.
    warnings: list[str] = field(default_factory=list)
    # Per-variable worst-difference records (only those that exceeded tol,
    # unless show_all is requested).
    variable_diffs: list[VariableDiff] = field(default_factory=list)
    # Coordinate worst difference, if compared.
    coordinate_max_delta: float | None = None
    # Mesh-map summary, when coordinate_matching was requested.
    mesh_map_built: bool = False
    unmatched_nodes: int = 0
    unmatched_elems: int = 0
    _steps: list[tuple[int, int, float]] | None = field(default=None, repr=False, compare=False)

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
        ``file1_step = file2_step + time_step_offset``.  The sentinels
        ``-1`` (:data:`STEP_OFFSET_AUTOMATIC`, align the last steps) and
        ``-2`` (:data:`STEP_OFFSET_MATCH`, align file-2's first time with the
        closest file-1 time) select exodiff's automatic offsets.
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
    sideset_df_tolerance, compare_sideset_df
        Side-set distribution factors are compared with this tolerance
        (SEACAS default relative 1e-6; command file ``SIDESET
        DISTRIBUTION``).  ``compare_sideset_df=False`` skips them (exodiff
        ``-ignore_sideset_df`` / ``IGNORE SIDESET DISTRIBUTION FACTORS``).
    final_time_tolerance
        When interpolating, the last compared file-1 time must match the last
        file-2 time within this tolerance (SEACAS ``-final_time_tolerance``,
        command file ``FINAL TIME TOLERANCE``; default relative ``0.0``).
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
    coordinate_matching
        If true, build a coordinate-based :class:`~exodusii.mesh.matching.MeshMap`
        before comparison.  This allows comparing files whose nodes and
        elements are in different orders but describe the same physical mesh.
        Default: ``False`` (matched-ordering mode, existing behavior).
    matching_tolerance
        Spatial tolerance used when building the mesh map.  Controls the
        maximum per-axis coordinate distance between two nodes for them to be
        considered the same physical node.  Independent of
        ``coordinate_tolerance`` (which governs reported coordinate
        *differences* after matching).  Only used when
        ``coordinate_matching=True``.  Default: absolute ``1e-6``.
    require_unique_mapping
        When ``True`` (default), raise
        :class:`~exodusii.mesh.matching.MeshMatchError` if any node or
        element cannot be uniquely matched within ``matching_tolerance``.
        When ``False``, issue a warning and continue with a partial map.
        Only used when ``coordinate_matching=True``.
    match_ids
        If true (default, SEACAS exodiff ``-match_ids``), nodes and elements
        are matched by their global IDs (``node_num_map`` /
        ``elem_num_map``), so files whose entities are stored in a different
        order compare equal, and ID sets that cannot be matched are errors.
        If false (exodiff ``-match_file_order``), entities are matched by file
        position and any difference between the ID maps is a difference.
        Ignored when ``coordinate_matching`` is true.
    ignore_nans
        If false (default, as in exodiff), any NaN value in either file is a
        difference -- including NaNs at the same location in both files.  If
        true (exodiff ``-ignore_nans``), NaN entries are skipped.
    categories_specified
        Variable categories named in an exodiff-format command file.  When not
        ``None``, categories absent from this set are not compared at all
        (exodiff: "If a variable type is not specified, no variables of that
        type will be considered").  ``None`` (default, no command file)
        compares every category.
    category_variable_tolerances, category_exclude
        Per-category per-variable tolerance overrides and ``!NAME``
        exclusions, as read from a command file.  These apply only to the
        category they were given for (``!T`` under ``NODAL VARIABLES`` does
        not exclude the global variable ``T``).  Element attributes use the
        key ``Entity.ELEMENT_BLOCK``.
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
    final_time_tolerance: Tolerance = field(
        default_factory=lambda: Tolerance(ToleranceMode.RELATIVE, 0.0, 0.0)
    )
    sideset_df_tolerance: Tolerance = field(
        default_factory=lambda: Tolerance(ToleranceMode.RELATIVE, 1.0e-6, 0.0)
    )
    compare_sideset_df: bool = True
    variable_tolerances: Mapping[str, Tolerance] = field(default_factory=dict)
    exclude: frozenset[str] = field(default_factory=frozenset)
    ignore_case: bool = True
    time_step_offset: int = 0
    time_selection: TimeSelection | None = None
    compare_coordinates: bool = True
    compare_attributes: bool = True
    show_all: bool = False
    # ── Phase 4: coordinate-based mesh matching ───────────────────────────
    coordinate_matching: bool = False
    matching_tolerance: float = 1.0e-6
    require_unique_mapping: bool = True
    # ── Command-file: per-category variable include lists ─────────────────
    #
    # ``include_variables`` maps a variable-location :class:`Entity` to the set
    # of variable names to compare for that category.  When a category has an
    # entry here, only those names are compared (still minus :attr:`exclude`).
    # Categories absent from the mapping fall back to "compare all common
    # names".  This mirrors the SEACAS ``exodiff`` command-file behavior where
    # a ``NODAL VARIABLES`` block listing specific names restricts the
    # comparison to those names unless ``(all)`` is given.
    #
    # ``all_categories`` records the categories for which ``(all)`` was
    # requested (or which had no explicit name list); such categories always
    # compare every common variable regardless of ``include_variables``.
    include_variables: Mapping[Entity, frozenset[str]] = field(default_factory=dict)
    all_categories: frozenset[Entity] = field(default_factory=frozenset)
    categories_specified: frozenset[Entity] | None = None
    category_variable_tolerances: Mapping[Entity, Mapping[str, Tolerance]] = field(
        default_factory=dict
    )
    category_exclude: Mapping[Entity, frozenset[str]] = field(default_factory=dict)
    match_ids: bool = True
    ignore_nans: bool = False

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

        for table in (self.category_variable_tolerances.get(ent), self.variable_tolerances):
            if not table:
                continue
            found = _lookup_name(table, name, ignore_case=self.ignore_case)
            if found is not None:
                return found
        return self._category_default(ent)

    def is_excluded(self, name: str, ent: Entity | None = None) -> bool:
        """Return ``True`` if *name* is excluded from comparison.

        Exclusion is case-insensitive when :attr:`ignore_case` is ``True``
        (the default).

        Parameters
        ----------
        name : str
            Variable name to check against the :attr:`exclude` set.
        ent : Entity, optional
            Variable category.  When given, the category's own exclusions
            (:attr:`category_exclude`) are also checked.

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

        for names in (self.exclude, self.category_exclude.get(ent) if ent is not None else None):
            if not names:
                continue
            if self.ignore_case:
                if name.lower() in {n.lower() for n in names}:
                    return True
            elif name in names:
                return True
        return False

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Serialize these options to a YAML command-file string.

        The emitted document is full/explicit (every field with its current
        value) and round-trips through
        :func:`exodusii.read_command_file`.

        Parameters
        ----------
        path
            If given, also write the YAML to this path.

        Returns
        -------
        str
            The YAML document.
        """

        from exodusii.api.command_file import diff_options_to_yaml

        text = diff_options_to_yaml(self)
        if path is not None:
            Path(path).write_text(text)
        return text

    def is_included(self, name: str, ent: Entity) -> bool:
        """Return ``True`` if *name* should be compared for category *ent*.

        A category is unrestricted (all common variables compared) when it is
        listed in :attr:`all_categories` or has no entry in
        :attr:`include_variables`.  Otherwise only the names in that category's
        include set are compared.  Case handling follows :attr:`ignore_case`.
        Exclusion (:meth:`is_excluded`) is applied separately by the caller.
        """

        if self.categories_specified is not None and ent not in self.categories_specified:
            return False
        if ent in self.all_categories:
            return True
        includes = self.include_variables.get(ent)
        if includes is None:
            # A command-file header without names or "(all)" selects nothing.
            return self.categories_specified is None
        if self.ignore_case:
            return name.lower() in {n.lower() for n in includes}
        return name in includes


def _lookup_name(
    table: Mapping[str, Tolerance], name: str, *, ignore_case: bool
) -> Tolerance | None:
    found = table.get(name)
    if found is not None or not ignore_case:
        return found
    lowered = name.lower()
    for key, value in table.items():
        if key.lower() == lowered:
            return value
    return None


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


def _compare_variable_series(
    values1: np.ndarray,
    values2: np.ndarray,
    tol: Tolerance,
    *,
    ent: Entity,
    name: str,
    block_id: int | None,
    set_id: int | None,
    ignore_nans: bool = False,
    scalar_test: bool = False,
    steps: list[StepTriple] | None = None,
) -> VariableDiff | None:
    """Compare two arrays (possibly multi-step) for one variable.

    *scalar_test* uses :meth:`Tolerance.diff` (exodiff ``Diff``, used for
    global variables) instead of :meth:`Tolerance.diff_array` (exodiff
    ``Delta > value``, used for field variables).

    Arrays are ``(n_steps, n_entries)``, or ``(n_steps,)`` for globals, with
    rows ordered as *steps*; the reported ``time_index`` is the file-1 step.

    NaN handling follows SEACAS exodiff: unless *ignore_nans*, a NaN in
    either file is a difference (even at the same location in both files).
    With *ignore_nans*, NaN entries are skipped.
    """

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

    any_nan = np.isnan(a) | np.isnan(b)
    with np.errstate(invalid="ignore"):
        delta = tol.delta_array(a, b)
        if scalar_test:
            exceeded_mask = (
                np.vectorize(tol.diff, otypes=[bool])(a, b) if a.size else a.astype(bool)
            )
        else:
            exceeded_mask = tol.diff_array(a, b)
        finite_diff = exceeded_mask & ~any_nan
    if ignore_nans:
        nan_mask = np.zeros_like(any_nan)
        delta = np.where(any_nan, np.nan, delta)
    else:
        nan_mask = any_nan
    diff_mask = finite_diff | nan_mask
    exceeded = bool(np.any(diff_mask))

    # A NaN location wins; its delta is meaningless, so report inf.
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
        if steps is not None:
            time_index = steps[index[0]][0]
            entry_index = index[1] if len(index) > 1 else None
        elif a.ndim >= 2:
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
    """Compare two Exodus databases exodiff-style.

    Performs a field-by-field comparison of two Exodus databases using the
    same default tolerances and time-step selection logic as the SEACAS
    ``exodiff`` tool.

    By default both files must share the same mesh topology (identical
    node/element ordering).  Pass ``DiffOptions(coordinate_matching=True)``
    to enable coordinate-based mesh matching, which builds a permutation map
    from spatial coordinates before comparison and allows comparing files
    whose nodes and elements are in different orders.

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

    Coordinate-based mesh matching (files with different node/element ordering):

    >>> opts = DiffOptions(coordinate_matching=True, matching_tolerance=1e-8)
    >>> result = diff("gold.exo", "permuted.exo", options=opts)
    """

    opts = options or DiffOptions()

    exo1, close1 = _open_if_needed(file1)
    exo2, close2 = _open_if_needed(file2)

    result = DiffResult(same=True, file1=str(exo1.path), file2=str(exo2.path))

    try:
        _compare_mesh_metadata(exo1, exo2, result, opts)
        if result.differences:
            # Nothing else is comparable (exodiff stops here too).
            result.same = False
            return result

        mesh_map: MeshMap | None = None
        if not opts.coordinate_matching and not result.errors:
            mesh_map = _id_mesh_map(exo1, exo2, opts, result)
        if opts.coordinate_matching and not result.errors:
            try:
                mesh_map = build_mesh_map(
                    exo1,
                    exo2,
                    matching_tolerance=opts.matching_tolerance,
                    require_unique_mapping=opts.require_unique_mapping,
                )
                result.mesh_map_built = True
                result.unmatched_nodes = mesh_map.unmatched_nodes
                result.unmatched_elems = mesh_map.unmatched_elems
            except (MeshMatchError, ValueError) as exc:
                result.errors.append(f"mesh matching failed: {exc}")

        if not result.errors:
            _compare_element_blocks(exo1, exo2, opts, result, mesh_map)
        if opts.compare_coordinates:
            _compare_coordinates(exo1, exo2, opts, result, mesh_map)
        _compare_times(exo1, exo2, opts, result)
        if opts.compare_attributes:
            _compare_attributes(exo1, exo2, opts, result, mesh_map)
        if not result.errors:
            _compare_sideset_distribution_factors(exo1, exo2, opts, result, mesh_map)
        _compare_all_variables(exo1, exo2, opts, result, mesh_map)
    finally:
        if close1:
            exo1.close()
        if close2:
            exo2.close()

    result.same = (
        not result.errors
        and not result.differences
        and not any(vd.exceeded for vd in result.variable_diffs)
    )
    return result


def _compare_mesh_metadata(
    exo1: ExodusFile, exo2: ExodusFile, result: DiffResult, opts: DiffOptions
) -> None:
    if exo1.dimension != exo2.dimension:
        result.errors.append(f"dimension differs: {exo1.dimension} != {exo2.dimension}")
    # Count mismatches are differences (exit 2), not errors, as in exodiff.
    if exo1.node_count != exo2.node_count:
        result.differences.append(f"node count differs: {exo1.node_count} != {exo2.node_count}")
    if exo1.element_count != exo2.element_count:
        result.differences.append(
            f"element count differs: {exo1.element_count} != {exo2.element_count}"
        )
    ids1 = exo1.element_block_ids().tolist()
    ids2 = exo2.element_block_ids().tolist()
    # Blocks are matched by ID, not position.
    if sorted(ids1) != sorted(ids2):
        # When coordinate matching is enabled, block IDs may differ; we
        # demote this from a fatal error to a warning and let the matching
        # algorithm reconcile the block assignment.
        if opts.coordinate_matching:
            result.warnings.append(
                f"element block ids differ: {ids1} != {ids2} "
                f"(coordinate matching will attempt to reconcile)"
            )
        else:
            result.errors.append(f"element block ids differ: {ids1} != {ids2}")


def _block_offsets(exo: ExodusFile) -> dict[int, int]:
    """Return ``{block_id: first global 0-based element index}`` in file order."""

    offsets: dict[int, int] = {}
    offset = 0
    for block_id in exo.element_block_ids().tolist():
        offsets[int(block_id)] = offset
        offset += exo.element_block(int(block_id)).count
    return offsets


def _match_ids(ids1: np.ndarray, ids2: np.ndarray) -> np.ndarray | None:
    """Return ``map2to1`` with ``ids1[map2to1[j]] == ids2[j]``, or ``None``.

    ``None`` means the two ID sets cannot be matched one-to-one.
    """

    if ids1.shape != ids2.shape:
        return None
    order1 = np.argsort(ids1, kind="stable")
    order2 = np.argsort(ids2, kind="stable")
    if not np.array_equal(ids1[order1], ids2[order2]):
        return None
    if len(ids1) and np.any(np.diff(ids1[order1]) == 0):
        return None  # duplicate IDs: no unique mapping
    map2to1 = np.empty(len(ids2), dtype=np.int64)
    map2to1[order2] = order1
    return map2to1


def _first_unmatched(ids1: np.ndarray, ids2: np.ndarray) -> int | None:
    missing = np.setdiff1d(ids1, ids2)
    return int(missing[0]) if len(missing) else None


def _id_mesh_map(
    exo1: ExodusFile, exo2: ExodusFile, opts: DiffOptions, result: DiffResult
) -> MeshMap | None:
    """Match nodes and elements by global ID.

    Returns ``None`` when the ID maps are identical, otherwise a
    :class:`MeshMap`; unmatched ID sets are recorded as errors.  With
    ``match_ids=False`` differing maps are recorded as a difference instead.
    """

    node_ids1 = np.asarray(exo1.ids(Entity.NODE), dtype=np.int64)
    node_ids2 = np.asarray(exo2.ids(Entity.NODE), dtype=np.int64)
    elem_ids1 = np.asarray(exo1.ids(Entity.ELEMENT), dtype=np.int64)
    elem_ids2 = np.asarray(exo2.ids(Entity.ELEMENT), dtype=np.int64)

    same_nodes = np.array_equal(node_ids1, node_ids2)
    same_elems = np.array_equal(elem_ids1, elem_ids2)
    if same_nodes and same_elems:
        return None

    if not opts.match_ids:
        if not same_nodes:
            result.differences.append("node ID maps differ (matching by file order)")
        if not same_elems:
            result.differences.append("element ID maps differ (matching by file order)")
        return None

    node_map = _match_ids(node_ids1, node_ids2)
    if node_map is None:
        missing = _first_unmatched(node_ids1, node_ids2)
        result.errors.append(
            "unable to match node IDs between the files"
            + (
                f" (node {missing} in the first file is not in the second)"
                if missing is not None
                else ""
            )
        )
        return None
    elem_map = _match_ids(elem_ids1, elem_ids2)
    if elem_map is None:
        missing = _first_unmatched(elem_ids1, elem_ids2)
        result.errors.append(
            "unable to match element IDs between the files"
            + (
                f" (element {missing} in the first file is not in the second)"
                if missing is not None
                else ""
            )
        )
        return None

    offsets1 = _block_offsets(exo1)
    offsets2 = _block_offsets(exo2)

    # Element matching must keep each element in the same block.
    def block_of(offsets: dict[int, int], count: int) -> np.ndarray:
        owner = np.empty(count, dtype=np.int64)
        ordered = sorted(offsets.items(), key=lambda kv: kv[1])
        for k, (block_id, start) in enumerate(ordered):
            stop = ordered[k + 1][1] if k + 1 < len(ordered) else count
            owner[start:stop] = block_id
        return owner

    if len(elem_map):
        owner1 = block_of(offsets1, len(elem_ids1))
        owner2 = block_of(offsets2, len(elem_ids2))
        moved = np.nonzero(owner1[elem_map] != owner2)[0]
        if len(moved):
            j = int(moved[0])
            result.errors.append(
                f"element {int(elem_ids2[j])} is in block {int(owner2[j])} in the second file "
                f"but block {int(owner1[elem_map[j]])} in the first"
            )
            return None

    return MeshMap(
        node_map=node_map,
        node_map_inv=np.argsort(node_map).astype(np.int64),
        elem_map=elem_map,
        elem_map_inv=np.argsort(elem_map).astype(np.int64),
        block_map={block_id: block_id for block_id in offsets2},
        block_elem_offsets1=offsets1,
        block_elem_offsets2=offsets2,
    )


def _element_types_compatible(type1: str, type2: str) -> bool:
    """Return whether two element types agree, as exodiff ``close_compare``.

    Case-insensitive; trailing node counts must match when both names have
    one (``QUAD`` == ``quad4``, ``tri`` == ``triangle6``, ``tri3`` != ``tri6``).
    """

    a = type1.strip().upper()
    b = type2.strip().upper()
    if a == b:
        return True
    if not a or not b:
        return False
    i, j = len(a), len(b)
    while i and j and a[i - 1].isdigit() and b[j - 1].isdigit():
        if a[i - 1] != b[j - 1]:
            return False
        i -= 1
        j -= 1
    while i and a[i - 1].isdigit():
        i -= 1
    while j and b[j - 1].isdigit():
        j -= 1
    length = min(i, j)
    return a[:length] == b[:length]


def _compare_element_blocks(
    exo1: ExodusFile,
    exo2: ExodusFile,
    opts: DiffOptions,
    result: DiffResult,
    mesh_map: MeshMap | None,
) -> None:
    """Compare element types, counts and connectivity; mismatches are errors."""

    inv_block_map: dict[int, int] = (
        {v: k for k, v in mesh_map.block_map.items()} if mesh_map is not None else {}
    )
    ids2 = set(exo2.element_block_ids().tolist())

    for block_id in exo1.element_block_ids().tolist():
        block_id2 = inv_block_map.get(block_id, block_id)
        if block_id2 not in ids2:
            continue  # reported by _compare_mesh_metadata / mesh matching
        block1 = exo1.element_block(int(block_id))
        block2 = exo2.element_block(int(block_id2))
        label = f"element block {block_id}"

        if block1.count != block2.count:
            result.errors.append(
                f"{label}: number of elements differs ({block1.count} != {block2.count})"
            )
            continue
        if block1.count == 0:
            continue
        if not _element_types_compatible(block1.element_type, block2.element_type):
            result.errors.append(
                f"{label}: element types differ ({block1.element_type} != {block2.element_type})"
            )
            continue
        if block1.nodes_per_entity != block2.nodes_per_entity:
            result.errors.append(
                f"{label}: nodes per element differ "
                f"({block1.nodes_per_entity} != {block2.nodes_per_entity})"
            )
            continue

        conn1 = np.asarray(exo1.element_connectivity(int(block_id)), dtype=np.int64)
        conn2 = np.asarray(exo2.element_connectivity(int(block_id2)), dtype=np.int64)
        if conn1.shape != conn2.shape:
            result.errors.append(f"{label}: connectivity shapes differ")
            continue

        if mesh_map is not None:
            expected = mesh_map.node_map_inv[conn1 - 1] + 1
            try:
                _perm, perm_inv = mesh_map.block_elem_perm(block_id2)
                conn2 = conn2[perm_inv]
            except (KeyError, ValueError, IndexError):
                pass
        else:
            expected = conn1

        mismatch = np.nonzero(np.any(expected != conn2, axis=1))[0]
        if len(mismatch):
            e = int(mismatch[0])
            n = int(np.nonzero(expected[e] != conn2[e])[0][0])
            result.errors.append(
                f"{label}: connectivity differs (first difference: node {n + 1} "
                f"of local element {e + 1})"
            )


def _compare_coordinates(
    exo1: ExodusFile,
    exo2: ExodusFile,
    opts: DiffOptions,
    result: DiffResult,
    mesh_map: MeshMap | None = None,
) -> None:
    if opts.coordinate_tolerance.mode is ToleranceMode.IGNORE:
        return
    if exo1.node_count != exo2.node_count:
        return
    if exo1.node_count == 0:
        # Globals-only databases (e.g. vvtk vdiff norm files) have no
        # coordinate variables; SEACAS exodiff compares them without error.
        return
    coords1 = exo1.coordinates()
    coords2 = exo2.coordinates()
    if coords1.shape != coords2.shape:
        result.errors.append("coordinate shapes differ")
        return
    # When a mesh map is available, reorder file-2 coordinates into file-1
    # node ordering before comparison.
    if mesh_map is not None:
        coords2 = coords2[mesh_map.node_map_inv]
    delta = opts.coordinate_tolerance.delta_array(coords1, coords2)
    max_delta, _ = _worst(delta)
    result.coordinate_max_delta = max_delta
    if np.any(opts.coordinate_tolerance.diff_array(coords1, coords2)):
        result.errors.append(f"coordinates differ (max delta {max_delta:.6e})")


def _compare_times(
    exo1: ExodusFile, exo2: ExodusFile, opts: DiffOptions, result: DiffResult
) -> None:
    """Record differing time values and unexplained step-count mismatches.

    A step-count mismatch is accepted when an offset (``n1 - offset == n2``),
    a stop step or interpolation accounts for it.
    """
    times1 = exo1.times()
    times2 = exo2.times()
    n1 = len(times1)
    n2 = len(times2)
    ts = opts._effective_time_selection()

    try:
        offset = _resolve_step_offset(times1, times2, ts.time_step_offset)
    except ValueError as exc:
        result.errors.append(str(exc))
        return
    if n1 != n2:
        explained = (n1 - offset == n2) or ts.stop > 0 or ts.interpolating or ts.start == -1
        if explained:
            result.warnings.append(
                f"time step count differs: {n1} != {n2} "
                "(the compared time steps account for the difference)"
            )
        else:
            result.differences.append(f"number of time steps differs: {n1} != {n2}")

    steps = _steps_to_compare(exo1, exo2, opts, result)
    result._steps = steps
    if ts.interpolating:
        # Times differ by design when interpolating; only the last selected
        # file-1 time must match file 2's last time (exodiff checks it even
        # when that step is outside file 2's range).
        last1 = min(n1, ts.stop) if ts.stop > 0 else n1
        if n1 and n2 and last1 >= max(1, ts.start):
            t1 = float(times1[last1 - 1]) * ts.time_value_scale + ts.time_value_offset
            t2 = float(times2[-1])
            if opts.final_time_tolerance.diff(t1, t2):
                result.differences.append(
                    f"final database times differ: {t1:.6e} != {t2:.6e} (not within "
                    f"{opts.final_time_tolerance.mode.value} tolerance "
                    f"{opts.final_time_tolerance.value:g})"
                )
        return
    for i1, i2, _prop in steps:
        t1 = float(times1[i1]) * ts.time_value_scale + ts.time_value_offset
        t2 = float(times2[i2])
        if opts.time_tolerance.diff(t1, t2):
            result.differences.append(
                f"time value differs at step {i1 + 1}: {times1[i1]:.6e} != {times2[i2]:.6e}"
            )


def _surrounding_steps(
    t: float, times2: np.ndarray, time_tolerance: Tolerance | None = None
) -> tuple[int, int, float]:
    """Return ``(lo, hi, proportion)`` bracketing ``t`` in ``times2``.

    ``proportion`` satisfies ``times2[lo] + proportion * (times2[hi] - times2[lo]) == t``.
    Returns ``(-1, -1, 0.0)`` when ``t`` is outside the range.  ``lo == hi``
    when ``t`` matches a step -- within *time_tolerance* when given, as SEACAS
    exodiff ``get_surrounding_times`` does.
    """
    n = len(times2)
    if n == 0:
        return -1, -1, 0.0

    def same(a: float, b: float) -> bool:
        if time_tolerance is None or time_tolerance.mode is ToleranceMode.IGNORE:
            return a == b
        return not time_tolerance.diff(a, b)

    # Snapping first also catches values just outside the range (round-off).
    for index, value in enumerate(times2):
        if same(t, float(value)):
            return index, index, 0.0
    if t < float(times2[0]) or t > float(times2[-1]):
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
    prop = (t - t_lo) / (t_hi - t_lo)
    return lo, hi, prop


# A step triple is (file1_0based_index, file2_0based_index_or_lo, proportion).
# proportion == 0.0  -> exact match (use file2 step at index directly).
# proportion  > 0.0  -> interpolate between file2[index] and file2[index+1].
StepTriple = tuple[int, int, float]

#: ``TimeSelection.time_step_offset`` sentinels (SEACAS exodiff ``-TA`` /
#: ``-TM``, command file ``STEP OFFSET AUTOMATIC`` / ``MATCH``).
STEP_OFFSET_AUTOMATIC = -1
STEP_OFFSET_MATCH = -2


def _resolve_step_offset(times1: np.ndarray, times2: np.ndarray, offset: int) -> int:
    """Resolve AUTOMATIC (align the last steps) and MATCH (align file 2's
    first time with the closest file-1 time) offsets."""

    n1, n2 = len(times1), len(times2)
    if offset == STEP_OFFSET_AUTOMATIC:
        if n1 < n2:
            raise ValueError("STEP OFFSET AUTOMATIC: the second file must have fewer time steps")
        return n1 - n2
    if offset == STEP_OFFSET_MATCH:
        if n1 < n2:
            raise ValueError("STEP OFFSET MATCH: the second file must have fewer time steps")
        if n2 == 0 or n1 == 0:
            return 0
        return int(np.argmin(np.abs(np.asarray(times1, dtype=np.float64) - float(times2[0]))))
    return offset


def _steps_to_compare(
    exo1: ExodusFile, exo2: ExodusFile, opts: DiffOptions, result: DiffResult | None
) -> list[StepTriple]:
    """Return ``(file1_idx, file2_lo_idx, proportion)`` triples, as exodiff loops.

    Steps run to ``min(n1, n2)`` (all file-1 steps when interpolating),
    capped by ``stop``.  An empty selection is recorded in *result* as a
    difference.
    """
    times1 = exo1.times()
    times2 = exo2.times()
    n1 = len(times1)
    n2 = len(times2)
    ts = opts._effective_time_selection()

    def no_steps(message: str) -> list[StepTriple]:
        if result is not None and (n1 or n2):
            result.differences.append(message)
        return []

    # ---- LAST sentinel: compare only the final step on each file.
    if ts.start == -1:
        if n1 == 0 or n2 == 0:
            return no_steps("no time steps to compare (LAST requested but a file has no steps)")
        return [(n1 - 1, n2 - 1, 0.0)]

    offset = _resolve_step_offset(times1, times2, ts.time_step_offset)

    last = n1 if ts.interpolating else min(n1, n2)
    if ts.stop > 0:
        last = min(last, ts.stop)
    start = max(1, ts.start)
    if start > last:
        if n1 == 0 and n2 == 0:
            return []
        return no_steps(
            f"time step selection (start {ts.start}, stop {ts.stop}) resulted in no time "
            "steps being compared"
        )

    triples: list[StepTriple] = []
    for step in range(start, last + 1, max(1, ts.increment)):
        if step in ts.exclude_steps:
            continue
        step1 = step + offset if not ts.interpolating else step
        if not 1 <= step1 <= n1:
            continue
        i1 = step1 - 1
        if ts.interpolating:
            t1_adj = float(times1[i1]) * ts.time_value_scale + ts.time_value_offset
            lo, hi, prop = _surrounding_steps(t1_adj, times2, opts.time_tolerance)
            if lo == -1:
                continue  # outside the file-2 time range
            triples.append((i1, lo, prop if lo != hi else 0.0))
        else:
            if not 1 <= step <= n2:
                continue
            triples.append((i1, step - 1, 0.0))

    if not triples and (n1 or n2):
        return no_steps("time step selection resulted in no time steps being compared")
    return triples


def _compare_attributes(
    exo1: ExodusFile,
    exo2: ExodusFile,
    opts: DiffOptions,
    result: DiffResult,
    mesh_map: MeshMap | None = None,
) -> None:
    base_tol = opts.attribute_tolerance or opts.default_tolerance
    if base_tol.mode is ToleranceMode.IGNORE:
        return
    attr_tols = opts.category_variable_tolerances.get(Entity.ELEMENT_BLOCK) or {}
    for block_entity, value_entity in (
        (Entity.ELEMENT_BLOCK, Entity.ELEMENT),
        (Entity.EDGE_BLOCK, Entity.EDGE),
        (Entity.FACE_BLOCK, Entity.FACE),
    ):
        ids1 = exo1.block_ids(block_entity).tolist()
        for block_id in ids1:
            # When mesh mapping is active, look up corresponding file-2 block id.
            block_id2 = block_id
            if mesh_map is not None and block_entity is Entity.ELEMENT_BLOCK:
                # block_map maps file-2 id → file-1 id; invert to get file-2 id for file-1 id.
                inv_block = {v: k for k, v in mesh_map.block_map.items()}
                block_id2 = inv_block.get(block_id, block_id)
            try:
                names1 = exo1.attribute_names(block_entity, block_id)
                names2 = exo2.attribute_names(block_entity, block_id2)
            except Exception:
                continue
            common, _only1, _only2 = _match_names(names1, names2, ignore_case=opts.ignore_case)
            for attr_name in common:
                if not opts.is_included(attr_name, Entity.ELEMENT_BLOCK) or opts.is_excluded(
                    attr_name, Entity.ELEMENT_BLOCK
                ):
                    continue
                try:
                    col1 = exo1.attribute_values(block_entity, block_id, attr_name)
                    col2 = exo2.attribute_values(block_entity, block_id2, attr_name)
                except Exception as exc:
                    result.errors.append(f"attribute {attr_name!r} block {block_id}: {exc}")
                    continue
                # Reorder file-2 attribute rows (one per element) into file-1 order.
                if mesh_map is not None and block_entity is Entity.ELEMENT_BLOCK:
                    try:
                        _perm, perm_inv = mesh_map.block_elem_perm(block_id2)
                        col2 = np.asarray(col2, dtype=np.float64)
                        col2 = col2[perm_inv] if col2.ndim == 1 else col2[perm_inv, :]
                    except Exception:
                        pass  # skip reorder if block offsets unavailable
                tol = _lookup_name(attr_tols, attr_name, ignore_case=opts.ignore_case) or base_tol
                vd = _compare_variable_series(
                    col1,
                    col2,
                    tol,
                    ent=value_entity,
                    name=f"attr:{attr_name}",
                    block_id=block_id,
                    set_id=None,
                    ignore_nans=opts.ignore_nans,
                )
                _record(result, opts, vd)


def _compare_all_variables(
    exo1: ExodusFile,
    exo2: ExodusFile,
    opts: DiffOptions,
    result: DiffResult,
    mesh_map: MeshMap | None = None,
) -> None:
    steps = (
        result._steps if result._steps is not None else _steps_to_compare(exo1, exo2, opts, None)
    )

    # When coordinate matching is active, check sideset face ordinals and
    # emit warnings for any that changed after element remapping.
    if mesh_map is not None:
        sideset_warns = check_sideset_ordinals(exo1, exo2, mesh_map)
        result.warnings.extend(sideset_warns)

    _compare_global_variables(exo1, exo2, opts, result, steps)
    _compare_nodal_variables(exo1, exo2, opts, result, steps, mesh_map)
    for ent in _BLOCK_VAR_ENTITIES:
        _compare_block_variables(exo1, exo2, opts, result, steps, ent, mesh_map)
    for ent in _SET_VAR_ENTITIES:
        _compare_set_variables(exo1, exo2, opts, result, steps, ent, mesh_map)


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

    def _wanted(name: str) -> bool:
        return opts.is_included(name, ent) and not opts.is_excluded(name, ent)

    # A variable in only one file is a difference (exit 2), not an error.
    for name in only1:
        if _wanted(name):
            result.differences.append(
                f"{ent.value} variable {name!r} is in the first file but not the second"
            )
    for name in only2:
        if _wanted(name):
            result.differences.append(
                f"{ent.value} variable {name!r} is in the second file but not the first"
            )
    includes = opts.include_variables.get(ent)
    if (
        includes
        and ent not in opts.all_categories
        and (opts.categories_specified is None or ent in opts.categories_specified)
    ):
        present = {n.lower() if opts.ignore_case else n for n in (*names1, *names2)}
        for name in sorted(includes):
            key = name.lower() if opts.ignore_case else name
            if key not in present and not opts.is_excluded(name, ent):
                result.differences.append(
                    f"{ent.value} variable {name!r} (requested) is not in the first file"
                )
    return [n for n in common if _wanted(n)]


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
                float(
                    _interp_values(
                        exo2, name, on=Entity.GLOBAL, lo=lo, hi=lo + (1 if p > 0 else 0), prop=p
                    )
                )
                for _i1, lo, p in steps
            ]
        )
        vd = _compare_variable_series(
            values1,
            values2,
            tol,
            ent=Entity.GLOBAL,
            name=name,
            block_id=None,
            set_id=None,
            ignore_nans=opts.ignore_nans,
            scalar_test=True,
            steps=steps,
        )
        _record(result, opts, vd)


def _compare_nodal_variables(
    exo1: ExodusFile,
    exo2: ExodusFile,
    opts: DiffOptions,
    result: DiffResult,
    steps: list[StepTriple],
    mesh_map: MeshMap | None = None,
) -> None:
    for name in _names_to_compare(exo1, exo2, Entity.NODE, opts, result):
        tol = opts.tolerance_for(name, Entity.NODE)
        rows1 = [exo1.values(name, on=Entity.NODE, time=i1) for i1, _lo, _p in steps]
        rows2 = [
            _interp_values(exo2, name, on=Entity.NODE, lo=lo, hi=lo + (1 if p > 0 else 0), prop=p)
            for _i1, lo, p in steps
        ]
        arr2 = np.array(rows2)
        # Reorder file-2 nodal array columns into file-1 node ordering.
        if mesh_map is not None:
            arr2 = arr2[:, mesh_map.node_map_inv]
        vd = _compare_variable_series(
            np.array(rows1),
            arr2,
            tol,
            ent=Entity.NODE,
            name=name,
            block_id=None,
            set_id=None,
            ignore_nans=opts.ignore_nans,
            steps=steps,
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
    mesh_map: MeshMap | None = None,
) -> None:
    names = _names_to_compare(exo1, exo2, ent, opts, result)
    if not names:
        return
    block_entity = _BLOCK_LOCATION[ent]
    block_ids = exo1.block_ids(block_entity).tolist()
    names1 = exo1.variable_names(ent)
    names2 = exo2.variable_names(ent)

    # Build inverse block map: file-1 block id → file-2 block id.
    inv_block_map: dict[int, int] = {}
    if mesh_map is not None and block_entity is Entity.ELEMENT_BLOCK:
        inv_block_map = {v: k for k, v in mesh_map.block_map.items()}

    for name in names:
        tol = opts.tolerance_for(name, ent)
        idx1 = _name_index(names1, name, ignore_case=opts.ignore_case)
        idx2 = _name_index(names2, name, ignore_case=opts.ignore_case)
        for block_id in block_ids:
            # Resolve the corresponding file-2 block id.
            block_id2 = inv_block_map.get(block_id, block_id) if inv_block_map else block_id
            present1 = _variable_present(exo1, ent, name, block_id, idx1)
            present2 = _variable_present(exo2, ent, name, block_id2, idx2)
            if not present1 and not present2:
                continue
            if present1 != present2:
                # exodiff skips such blocks without reporting a difference.
                result.warnings.append(
                    f"{ent.value} variable {name!r} block {block_id}: stored in only one file "
                    "(truth tables differ); not compared"
                )
                continue
            try:
                rows1 = [
                    exo1.values(name, on=ent, block_id=block_id, time=i1) for i1, _lo, _p in steps
                ]
                rows2 = [
                    _interp_values(
                        exo2,
                        name,
                        on=ent,
                        lo=lo,
                        hi=lo + (1 if p > 0 else 0),
                        prop=p,
                        block_id=block_id2,
                    )
                    for _i1, lo, p in steps
                ]
            except Exception as exc:
                result.errors.append(f"{ent.value} variable {name!r} block {block_id}: {exc}")
                continue
            arr2 = np.array(rows2)
            # Reorder file-2 element array columns into file-1 element ordering.
            if mesh_map is not None and block_entity is Entity.ELEMENT_BLOCK:
                try:
                    _perm, perm_inv = mesh_map.block_elem_perm(block_id2)
                    arr2 = arr2[:, perm_inv]
                except Exception:
                    pass  # skip reorder if block offsets unavailable
            vd = _compare_variable_series(
                np.array(rows1),
                arr2,
                tol,
                ent=ent,
                name=name,
                block_id=block_id,
                set_id=None,
                ignore_nans=opts.ignore_nans,
                steps=steps,
            )
            _record(result, opts, vd)


def _sorted_set_entries(
    exo: ExodusFile,
    ent: Entity,
    set_id: int,
    *,
    node_map: np.ndarray | None = None,
    elem_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return sorted ``(keys, order)`` for a node or side set, else ``None``.

    Keys are node numbers, or ``element * 8 + side`` for side sets, in
    file-2 numbering when ``node_map`` / ``elem_map`` (0-based file-1 ->
    file-2) are given.
    """

    try:
        info = exo.set(ent, set_id)
    except Exception:
        return None
    entries = info.entries
    if entries is None:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    entries = np.asarray(entries, dtype=np.int64)

    if ent is Entity.NODE_SET:
        keys = entries if node_map is None else node_map[entries - 1] + 1
    elif ent is Entity.SIDE_SET:
        elems = entries if elem_map is None else elem_map[entries - 1] + 1
        sides = np.asarray(
            info.extra_entries if info.extra_entries is not None else np.zeros_like(entries),
            dtype=np.int64,
        )
        keys = elems * 8 + sides
    else:
        return None
    order = np.argsort(keys, kind="stable").astype(np.int64)
    return keys[order], order


def _set_maps(mesh_map: MeshMap | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (file-1 -> file-2) node and element index maps, or ``None``."""

    if mesh_map is None:
        return None, None
    return mesh_map.node_map_inv, mesh_map.elem_map_inv


def _compare_set_variables(
    exo1: ExodusFile,
    exo2: ExodusFile,
    opts: DiffOptions,
    result: DiffResult,
    steps: list[StepTriple],
    ent: Entity,
    mesh_map: MeshMap | None = None,
) -> None:
    """Compare set variables of sets matched by ID.

    Node and side set entries are compared in sorted order, so storage
    order does not matter; their member lists must agree (a mesh error).
    """

    names = _names_to_compare(exo1, exo2, ent, opts, result)
    if not names:
        return
    set_ids = exo1.set_ids(ent).tolist()
    set_ids2 = set(exo2.set_ids(ent).tolist())
    names1 = exo1._variable_name_table(ent)
    names2 = exo2._variable_name_table(ent)
    node_map, elem_map = _set_maps(mesh_map)

    alignment: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}
    for set_id in set_ids:
        if set_id not in set_ids2:
            continue
        sorted1 = _sorted_set_entries(exo1, ent, set_id, node_map=node_map, elem_map=elem_map)
        sorted2 = _sorted_set_entries(exo2, ent, set_id)
        if sorted1 is None or sorted2 is None:
            alignment[set_id] = None  # other set types: compare in file order
            continue
        keys1, order1 = sorted1
        keys2, order2 = sorted2
        if keys1.shape != keys2.shape:
            result.errors.append(
                f"{ent.value} {set_id}: entry count differs ({len(keys1)} != {len(keys2)})"
            )
            continue
        mismatch = np.nonzero(keys1 != keys2)[0]
        if len(mismatch):
            kind = "node list" if ent is Entity.NODE_SET else "side list"
            result.errors.append(
                f"{ent.value} {set_id}: {kind}s are not the same in the two files "
                f"(first difference at sorted position {int(mismatch[0]) + 1})"
            )
            continue
        alignment[set_id] = (order1, order2)

    for name in names:
        tol = opts.tolerance_for(name, ent)
        idx1 = _name_index(names1, name, ignore_case=opts.ignore_case)
        idx2 = _name_index(names2, name, ignore_case=opts.ignore_case)
        for set_id in set_ids:
            if set_id not in alignment:
                continue
            present1 = _variable_present(exo1, ent, name, set_id, idx1)
            present2 = _variable_present(exo2, ent, name, set_id, idx2)
            if not present1 and not present2:
                continue
            if present1 != present2:
                result.warnings.append(
                    f"{ent.value} variable {name!r} set {set_id}: stored in only one file "
                    "(truth tables differ); not compared"
                )
                continue
            try:
                rows1 = [exo1.values(name, on=ent, set_id=set_id, time=i1) for i1, _lo, _p in steps]
                rows2 = [
                    _interp_values(
                        exo2,
                        name,
                        on=ent,
                        lo=lo,
                        hi=lo + (1 if p > 0 else 0),
                        prop=p,
                        set_id=set_id,
                    )
                    for _i1, lo, p in steps
                ]
            except Exception as exc:
                result.errors.append(f"{ent.value} variable {name!r} set {set_id}: {exc}")
                continue

            arr1 = np.array(rows1)
            arr2 = np.array(rows2)
            aligned = alignment[set_id]
            if aligned is not None and arr1.ndim == 2 and arr2.ndim == 2:
                order1, order2 = aligned
                arr1 = arr1[:, order1]
                arr2 = arr2[:, order2]

            vd = _compare_variable_series(
                arr1,
                arr2,
                tol,
                ent=ent,
                name=name,
                block_id=None,
                set_id=set_id,
                ignore_nans=opts.ignore_nans,
                steps=steps,
            )
            _record(result, opts, vd)


def _compare_sideset_distribution_factors(
    exo1: ExodusFile,
    exo2: ExodusFile,
    opts: DiffOptions,
    result: DiffResult,
    mesh_map: MeshMap | None,
) -> None:
    """Compare side-set distribution factors, sides in sorted order.

    Sets where either file stores no factors are skipped.
    """

    if not opts.compare_sideset_df or opts.sideset_df_tolerance.mode is ToleranceMode.IGNORE:
        return
    _node_map, elem_map = _set_maps(mesh_map)
    ids2 = set(exo2.side_set_ids().tolist())
    for set_id in exo1.side_set_ids().tolist():
        if set_id not in ids2:
            continue
        info1 = exo1.side_set(int(set_id))
        info2 = exo2.side_set(int(set_id))
        df1 = info1.dist_facts
        df2 = info2.dist_facts
        if df1 is None or df2 is None or not len(df1) or not len(df2):
            continue
        df1 = np.asarray(df1, dtype=np.float64)
        df2 = np.asarray(df2, dtype=np.float64)
        n1 = info1.count
        n2 = info2.count
        if n1 != n2 or len(df1) != len(df2) or n1 == 0 or len(df1) % n1:
            result.differences.append(
                f"side_set {set_id}: distribution factor counts differ ({len(df1)} != {len(df2)})"
            )
            continue
        per_side = len(df1) // n1
        sorted1 = _sorted_set_entries(exo1, Entity.SIDE_SET, int(set_id), elem_map=elem_map)
        sorted2 = _sorted_set_entries(exo2, Entity.SIDE_SET, int(set_id))
        rows1 = df1.reshape(n1, per_side)
        rows2 = df2.reshape(n2, per_side)
        if sorted1 is not None and sorted2 is not None and np.array_equal(sorted1[0], sorted2[0]):
            rows1 = rows1[sorted1[1]]
            rows2 = rows2[sorted2[1]]
        vd = _compare_variable_series(
            rows1.reshape(1, -1),
            rows2.reshape(1, -1),
            opts.sideset_df_tolerance,
            ent=Entity.SIDE_SET,
            name="distribution factors",
            block_id=None,
            set_id=int(set_id),
            ignore_nans=opts.ignore_nans,
        )
        if vd is not None:
            vd = replace(vd, time_index=None)
        _record(result, opts, vd)
