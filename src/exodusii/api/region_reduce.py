# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Region-masked reduction utilities for Exodus databases.

This module provides :class:`RegionStatsResult`, the helper functions
:func:`region_stats` and :func:`region_mass`, and supporting utilities for
computing statistics over a geometric region with an optional field-threshold
predicate.

These are exposed as methods on :class:`~exodusii.api.file.ExodusFile` and
:class:`~exodusii.api.parallel.ParallelExodusFile`; direct use of this module
is not normally required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from exodusii.mesh.geometry import element_volumes
from exodusii.mesh.geometry import entity_centers
from exodusii.mesh.regions import Region

if TYPE_CHECKING:
    from exodusii.api.file import ExodusFile
    from exodusii.core.time import TimeSelector

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

#: Reducers that are *extensive* (scale linearly with volume / count).
#: Symmetry factors are applied only to these.
EXTENSIVE_REDUCERS: frozenset[str] = frozenset({"sum", "count", "mass"})

#: All supported reducer names.
VALID_REDUCERS: frozenset[str] = frozenset({"mean", "max", "min", "sum", "count", "std"})


@dataclass(frozen=True, slots=True)
class RegionStatsResult:
    """Result of a region-masked reduction.

    Attributes
    ----------
    variable : str
        Name of the result variable that was reduced.
    entity : str
        Entity type string (e.g. ``"element"``).
    block_id : int or None
        Element block ID, or ``None`` when all blocks were used.
    time_index : int or None
        Zero-based time step index used for the reduction, or ``None`` when
        *time=None* was passed (which selects the last step).
    time_value : float or None
        Physical time value at *time_index*.
    count_total : int
        Total number of entities in the block (before masking).
    count_selected : int
        Number of entities that satisfied both the region and the predicate.
    symmetry_factor : float
        Factor applied to extensive reducers (``sum``, ``count``).  ``1.0``
        means no symmetry scaling.
    stats : dict[str, float]
        Mapping from reducer name to computed value.  Extensive reducers have
        already been multiplied by *symmetry_factor*.
    """

    variable: str
    entity: str
    block_id: int | None
    time_index: int | None
    time_value: float | None
    count_total: int
    count_selected: int
    symmetry_factor: float
    stats: dict[str, float]


@dataclass(frozen=True, slots=True)
class RegionMassResult:
    """Result of a :func:`region_mass` computation.

    Attributes
    ----------
    block_id : int
        Element block ID.
    time_index : int
        Zero-based time step index.
    time_value : float
        Physical time value.
    count_total : int
        Total elements in block.
    count_selected : int
        Elements inside the region (after optional predicate).
    density_name : str
        Name of the density variable used.
    volfrac_name : str or None
        Name of the volume-fraction variable, or ``None`` if not used.
    symmetry_factor : float
        Symmetry scaling applied to the returned mass.
    mass : float
        Total mass = ``symmetry_factor * sum(vol * density [* volfrac])``
        over selected elements.
    """

    block_id: int
    time_index: int
    time_value: float
    count_total: int
    count_selected: int
    density_name: str
    volfrac_name: str | None
    symmetry_factor: float
    mass: float


# ---------------------------------------------------------------------------
# Core reduction helpers
# ---------------------------------------------------------------------------


def _apply_mask_reduce(
    values: FloatArray, mask: BoolArray, reducers: list[str], *, symmetry_factor: float = 1.0
) -> dict[str, float]:
    """Apply *reducers* to *values[mask]*.

    Parameters
    ----------
    values : ndarray, shape (n,)
        The field values for all entities.
    mask : ndarray of bool, shape (n,)
        True where an entity should be included.
    reducers : list of str
        One or more of ``"mean"``, ``"max"``, ``"min"``, ``"sum"``,
        ``"count"``, ``"std"``.
    symmetry_factor : float
        Multiply extensive reducers (``"sum"``, ``"count"``) by this value.

    Returns
    -------
    dict mapping reducer name → float
    """
    selected = values[mask]
    result: dict[str, float] = {}

    for reducer in reducers:
        if reducer == "mean":
            result["mean"] = float(np.mean(selected)) if selected.size else float("nan")
        elif reducer == "max":
            result["max"] = float(np.max(selected)) if selected.size else float("nan")
        elif reducer == "min":
            result["min"] = float(np.min(selected)) if selected.size else float("nan")
        elif reducer == "sum":
            raw = float(np.sum(selected)) if selected.size else 0.0
            result["sum"] = raw * symmetry_factor
        elif reducer == "count":
            result["count"] = float(int(mask.sum())) * symmetry_factor
        elif reducer == "std":
            result["std"] = float(np.std(selected)) if selected.size else float("nan")
        else:
            raise ValueError(
                f"unknown reducer {reducer!r}; valid reducers: " + ", ".join(sorted(VALID_REDUCERS))
            )

    return result


# ---------------------------------------------------------------------------
# Safe predicate parser
# ---------------------------------------------------------------------------

# Accepted comparisons: VAR OP VALUE  (e.g. "EQPS_2 > 1.0")
_PREDICATE_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?P<op>>=|<=|!=|==|>|<)\s*"
    r"(?P<value>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)

_OPS: dict[str, object] = {
    ">": np.greater,
    "<": np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
}


def _parse_predicate(
    expr: str, exo: ExodusFile, *, on: str, block_id: int | None, time: TimeSelector
) -> BoolArray:
    """Parse and evaluate a simple field-predicate expression.

    The only supported syntax is ``VARNAME OP VALUE`` where *OP* is one of
    ``>``, ``<``, ``>=``, ``<=``, ``==``, ``!=`` and *VALUE* is a numeric
    literal.  This restriction is intentional; arbitrary Python expressions
    are not supported.

    Parameters
    ----------
    expr : str
        Predicate string, e.g. ``"EQPS_2 > 1.0"``.
    exo : ExodusFile
        Open database.
    on : str
        Entity location (e.g. ``"element"``).
    block_id : int or None
        Block to read the predicate variable from.
    time : TimeSelector
        Time step selector forwarded to :meth:`~ExodusFile.values`.

    Returns
    -------
    ndarray of bool
        Boolean mask of shape ``(n_entities,)`` where the predicate is True.

    Raises
    ------
    ValueError
        If the expression does not match the supported syntax.
    """
    match = _PREDICATE_RE.match(expr)
    if match is None:
        raise ValueError(
            f"unsupported predicate expression {expr!r}. "
            "Only 'VARNAME OP VALUE' form is accepted (e.g. 'EQPS_2 > 1.0')."
        )

    name = match.group("name")
    op_fn = _OPS[match.group("op")]
    threshold = float(match.group("value"))

    field_values: FloatArray
    if block_id is not None:
        field_values = np.asarray(
            exo.values(name, on=on, block_id=block_id, time=time), dtype=np.float64
        )
    else:
        field_values = np.asarray(exo.values(name, on=on, time=time), dtype=np.float64)

    return np.asarray(op_fn(field_values, threshold), dtype=np.bool_)


# ---------------------------------------------------------------------------
# High-level public functions (called from ExodusFile methods)
# ---------------------------------------------------------------------------


def region_stats(
    exo: ExodusFile,
    name: str,
    *,
    on: str = "element",
    block_id: int | None = None,
    region: Region,
    where: str | None = None,
    reduce: list[str] | str,
    time: TimeSelector = None,
    symmetry_factor: float = 1.0,
) -> RegionStatsResult:
    """Compute statistics of *name* inside a geometric *region*.

    Parameters
    ----------
    exo : ExodusFile
        Open Exodus database (or any object with the same interface, e.g.
        :class:`~exodusii.api.parallel.ParallelExodusFile`).
    name : str
        Result variable name (element, edge, or face variable).
    on : str, optional
        Entity location.  Currently only ``"element"`` is fully supported.
    block_id : int or None, optional
        Restrict to a single element block.  When ``None``, values from all
        blocks are concatenated (element centers from all blocks are used for
        the region test).
    region : Region
        A geometric region that implements ``region.contains(points)``
        returning a boolean mask.  The region is tested against element
        centroids.
    where : str or None, optional
        Optional field-threshold predicate in ``"VARNAME OP VALUE"`` form,
        e.g. ``"EQPS_2 > 1.0"``.  The predicate is AND-ed with the region
        mask.
    reduce : str or list of str
        One or more of ``"mean"``, ``"max"``, ``"min"``, ``"sum"``,
        ``"count"``, ``"std"``.
    time : TimeSelector, optional
        Time step selector.  ``None`` selects the last available step.
    symmetry_factor : float, optional
        Scale factor for extensive reducers (``"sum"``, ``"count"``).
        Default ``1.0`` (no scaling).

    Returns
    -------
    RegionStatsResult
    """
    from exodusii.core.time import resolve_time

    if isinstance(reduce, str):
        reduce_list = [r.strip() for r in reduce.split(",")]
    else:
        reduce_list = list(reduce)

    for r in reduce_list:
        if r not in VALID_REDUCERS:
            raise ValueError(
                f"unknown reducer {r!r}; valid reducers: " + ", ".join(sorted(VALID_REDUCERS))
            )

    # Resolve time to a concrete index so we use one consistent snapshot
    times = exo.times()
    effective_time: TimeSelector = time
    if effective_time is None:
        effective_time = "last"
    selection = resolve_time(times, effective_time)
    time_index = selection.index
    time_value = selection.value

    # Get coordinates and connectivity to compute element centers
    coords = exo.coordinates()

    if block_id is not None:
        conn = exo.element_connectivity(block_id, zero_based=True)
        centers = entity_centers(conn, coords)
        field = np.asarray(
            exo.values(name, on=on, block_id=block_id, time=time_index), dtype=np.float64
        )
    else:
        # All blocks: concatenate
        block_ids_array = exo.element_block_ids()
        centers_list: list[FloatArray] = []
        field_list: list[FloatArray] = []
        for bid in block_ids_array:
            bid_int = int(bid)
            conn = exo.element_connectivity(bid_int, zero_based=True)
            centers_list.append(entity_centers(conn, coords))
            field_list.append(
                np.asarray(
                    exo.values(name, on=on, block_id=bid_int, time=time_index), dtype=np.float64
                )
            )
        centers = np.vstack(centers_list)
        field = np.concatenate(field_list)

    count_total = int(centers.shape[0])

    # Region mask
    region_mask = np.asarray(region.contains(centers), dtype=np.bool_)

    # Field predicate mask
    if where is not None:
        pred_mask = _parse_predicate(where, exo, on=on, block_id=block_id, time=time_index)
        combined_mask = region_mask & pred_mask
    else:
        combined_mask = region_mask

    count_selected = int(combined_mask.sum())
    stats = _apply_mask_reduce(field, combined_mask, reduce_list, symmetry_factor=symmetry_factor)

    return RegionStatsResult(
        variable=name,
        entity=on,
        block_id=block_id,
        time_index=time_index,
        time_value=float(time_value),
        count_total=count_total,
        count_selected=count_selected,
        symmetry_factor=symmetry_factor,
        stats=stats,
    )


def region_mass(
    exo: ExodusFile,
    *,
    block_id: int,
    region: Region,
    density_name: str = "DENSITY",
    volfrac_name: str | None = None,
    where: str | None = None,
    time: TimeSelector = None,
    symmetry_factor: float = 1.0,
) -> RegionMassResult:
    """Compute the mass inside a geometric region.

    Mass is computed as::

        mass = symmetry_factor * sum(abs(vol_i) * density_i [* volfrac_i])

    for all elements whose centroid lies inside *region* (and that satisfy
    the optional *where* predicate).

    Parameters
    ----------
    exo : ExodusFile
        Open Exodus database.
    block_id : int
        Element block ID to use.
    region : Region
        Geometric region predicate.
    density_name : str, optional
        Name of the element density variable.  Default ``"DENSITY"``.
    volfrac_name : str or None, optional
        Name of an optional element volume-fraction variable (e.g.
        ``"VOLFRC_2"``).  When ``None``, no volume fraction is applied.
    where : str or None, optional
        Optional field-threshold predicate, e.g. ``"EQPS_2 > 1.0"``.
    time : TimeSelector, optional
        Time step selector.  ``None`` selects the last available step.
    symmetry_factor : float, optional
        Scale factor applied to the returned mass (e.g. ``4.0`` for a
        quarter-symmetry model).  Default ``1.0``.

    Returns
    -------
    RegionMassResult
    """
    from exodusii.core.time import resolve_time

    times = exo.times()
    effective_time: TimeSelector = time
    if effective_time is None:
        effective_time = "last"
    selection = resolve_time(times, effective_time)
    time_index = selection.index
    time_value = selection.value

    coords = exo.coordinates()
    conn = exo.element_connectivity(block_id, zero_based=True)
    centers = entity_centers(conn, coords)

    block = exo.element_block(block_id)
    vols = np.abs(np.asarray(element_volumes(block.element_type, conn, coords), dtype=np.float64))
    density = np.asarray(
        exo.values(density_name, on="element", block_id=block_id, time=time_index), dtype=np.float64
    )

    count_total = int(centers.shape[0])

    # Region mask
    region_mask = np.asarray(region.contains(centers), dtype=np.bool_)

    # Optional field predicate
    if where is not None:
        pred_mask = _parse_predicate(where, exo, on="element", block_id=block_id, time=time_index)
        combined_mask = region_mask & pred_mask
    else:
        combined_mask = region_mask

    count_selected = int(combined_mask.sum())

    # Volume-fraction weighting
    if volfrac_name is not None:
        volfrac = np.asarray(
            exo.values(volfrac_name, on="element", block_id=block_id, time=time_index),
            dtype=np.float64,
        )
        mass_array = vols * density * volfrac
    else:
        mass_array = vols * density

    raw_mass = float(mass_array[combined_mask].sum()) if count_selected else 0.0
    total_mass = raw_mass * symmetry_factor

    return RegionMassResult(
        block_id=block_id,
        time_index=time_index,
        time_value=float(time_value),
        count_total=count_total,
        count_selected=count_selected,
        density_name=density_name,
        volfrac_name=volfrac_name,
        symmetry_factor=symmetry_factor,
        mass=total_mass,
    )


__all__ = [
    "EXTENSIVE_REDUCERS",
    "VALID_REDUCERS",
    "RegionMassResult",
    "RegionStatsResult",
    "region_mass",
    "region_stats",
]
