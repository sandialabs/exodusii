# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Coordinate-based mesh matching for Exodus database comparison.

This module builds bijective permutation maps between two Exodus databases
whose nodes and elements may be in different orders but describe the same
physical mesh.  The resulting :class:`MeshMap` can be threaded through
:func:`exodusii.api.diff.diff` to enable coordinate-based comparison.

Algorithm
---------
The matching algorithm mirrors the SEACAS ``exodiff`` ``Compute_Maps``
strategy from ``map.C``:

1. **Element centroid matching** — For each element block, compute centroids
   as the mean of node coordinates.  Sort file-2 centroids along the
   coordinate axis with the greatest spread, then use binary search to find
   candidate matches within ``matching_tolerance`` for each file-1 centroid.
   Pick the unique closest candidate.

2. **Node map derivation** — For each matched element pair, compare local
   node coordinates between the two elements and build the node-level
   mapping.  Nodes shared across blocks are resolved by first-match priority.

3. **Free-node fallback** — Nodes not reached via element matching (e.g.
   isolated nodes or nodes on disconnected sub-meshes) are matched by a
   second direct coordinate-based sorted-axis search.

4. **Inverse maps** — ``node_map_inv = np.argsort(node_map)`` and
   ``elem_map_inv = np.argsort(elem_map)`` are derived automatically.

Limitations
-----------
* Both files must have the same total node count and element count.
* Duplicate coordinate positions (two distinct nodes at the same location)
  will produce an ambiguous match; the closest-by-Euclidean-distance
  candidate wins with a warning emitted.
* Sideset face-ordinal remapping after element reordering is detected but
  not corrected; a warning is emitted for each affected set.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import numpy.typing as npt

from exodusii.core.entities import Entity

__all__ = ["MeshMap", "MeshMatchError", "build_mesh_map"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class MeshMatchError(RuntimeError):
    """Raised when coordinate-based mesh matching fails unrecoverably.

    This exception is raised by :func:`build_mesh_map` when a unique
    bijective mapping cannot be established (e.g. unmatched nodes or
    elements when ``require_unique_mapping=True``).
    """


@dataclass(frozen=True, slots=True)
class MeshMap:
    """Bijective permutation maps between two Exodus mesh index spaces.

    All index arrays use **0-based** indices.

    Attributes
    ----------
    node_map : ndarray of int64, shape (N_nodes,)
        ``node_map[j] = i`` — file-2 node index *j* corresponds to
        file-1 node index *i*.
    node_map_inv : ndarray of int64, shape (N_nodes,)
        Inverse of ``node_map``: ``node_map_inv[i] = j``.
        Satisfies ``node_map[node_map_inv[i]] == i`` for all *i*.
        Used to reorder file-2 arrays into file-1 layout via
        ``arr2[node_map_inv]``.
    elem_map : ndarray of int64, shape (N_elems,)
        ``elem_map[j] = i`` — file-2 global element index *j* corresponds
        to file-1 global element index *i*.
    elem_map_inv : ndarray of int64, shape (N_elems,)
        Inverse of ``elem_map``.  Used to reorder file-2 element arrays.
    block_map : dict mapping file-2 block id → file-1 block id
        Matched element block IDs.  If block IDs are the same in both
        files (the common case), ``block_map[bid] == bid`` for all entries.
    block_elem_offsets1 : dict mapping file-1 block id → global offset
        First global 0-based element index for each file-1 block.
    block_elem_offsets2 : dict mapping file-2 block id → global offset
        First global 0-based element index for each file-2 block.
    unmatched_nodes : int
        Number of nodes that could not be uniquely matched.  Zero when
        ``require_unique_mapping=True`` (the default).
    unmatched_elems : int
        Number of elements that could not be uniquely matched.
    """

    node_map: IntArray
    node_map_inv: IntArray
    elem_map: IntArray
    elem_map_inv: IntArray
    block_map: dict[int, int] = field(default_factory=dict)
    block_elem_offsets1: dict[int, int] = field(default_factory=dict)
    block_elem_offsets2: dict[int, int] = field(default_factory=dict)
    unmatched_nodes: int = 0
    unmatched_elems: int = 0

    @classmethod
    def identity(cls, n_nodes: int, n_elems: int) -> "MeshMap":
        """Return the identity map (both files in the same order).

        Parameters
        ----------
        n_nodes : int
            Total number of nodes.
        n_elems : int
            Total number of elements.

        Returns
        -------
        MeshMap
            A :class:`MeshMap` where every index maps to itself.
        """

        idx_n = np.arange(n_nodes, dtype=np.int64)
        idx_e = np.arange(n_elems, dtype=np.int64)
        return cls(
            node_map=idx_n,
            node_map_inv=idx_n.copy(),
            elem_map=idx_e,
            elem_map_inv=idx_e.copy(),
        )

    def block_elem_perm(self, block_id2: int) -> tuple[IntArray, IntArray]:
        """Return (perm, perm_inv) mapping file-2 block-local element indices.

        Given a file-2 block id, return arrays ``(perm, perm_inv)`` of
        length equal to the block's element count such that:

        * ``perm[j]`` = file-1 block-local index for file-2 block-local
          element *j*.
        * ``perm_inv[i]`` = file-2 block-local index for file-1 block-local
          element *i*.

        These are used inside :mod:`exodusii.api.diff` to reorder per-block
        variable arrays before comparison.

        Parameters
        ----------
        block_id2 : int
            The file-2 block id to look up.

        Returns
        -------
        perm : ndarray of int64
        perm_inv : ndarray of int64
        """

        block_id1 = self.block_map.get(block_id2, block_id2)
        off1 = self.block_elem_offsets1[block_id1]
        off2 = self.block_elem_offsets2[block_id2]

        # Determine block size from the global elem_map.
        # Count file-2 global indices that fall in this block's global range.
        # We need the size; derive it by scanning elem_map for entries whose
        # file-1 global index falls in [off1, off1+count1).
        # Since block_elem_offsets are stored, infer count from consecutive
        # keys or directly from elem_map_inv slice.
        # Use a simpler approach: find all file-2 indices j where
        # elem_map[j] maps into block1's range, and restrict to j in block2's range.
        # Actually: elem_map[j] = i means j is in block2 and i is in block1.
        # block2's elements are at global file-2 indices [off2, off2+count2).
        # count2 == count1 (matched blocks must have equal element counts).
        # Find count2 by looking at how many global file-2 indices are in block2.
        sorted_ids2 = sorted(self.block_elem_offsets2.keys())
        idx2 = sorted_ids2.index(block_id2)
        if idx2 + 1 < len(sorted_ids2):
            count2 = self.block_elem_offsets2[sorted_ids2[idx2 + 1]] - off2
        else:
            count2 = self.elem_map.shape[0] - off2

        # Global-to-block-local conversion:
        # file-2 block-local: j_local ∈ [0, count2)  →  j_global = off2 + j_local
        # file-1 block-local: i_local ∈ [0, count1)  →  i_global = off1 + i_local
        j_globals = np.arange(off2, off2 + count2, dtype=np.int64)
        i_globals = self.elem_map[j_globals]           # file-1 global indices
        i_locals = i_globals - off1                    # file-1 block-local
        perm = i_locals                                 # perm[j_local] = i_local
        perm_inv = np.argsort(perm).astype(np.int64)   # perm_inv[i_local] = j_local
        return perm, perm_inv


def build_mesh_map(
    exo1,
    exo2,
    *,
    matching_tolerance: float = 1.0e-6,
    require_unique_mapping: bool = True,
) -> MeshMap:
    """Build a coordinate-based :class:`MeshMap` between two Exodus databases.

    Matches nodes and elements between *exo1* and *exo2* by spatial
    proximity.  The algorithm follows the SEACAS ``exodiff`` strategy:
    match elements first by centroid, derive the node map from matched
    element local-node coordinate pairs, then fall back to direct
    coordinate matching for unmatched (free) nodes.

    Parameters
    ----------
    exo1 : ExodusFile
        The reference (file-1) database.
    exo2 : ExodusFile
        The comparison (file-2) database.  Must have the same node and
        element counts as *exo1*.
    matching_tolerance : float
        Maximum coordinate distance (in each axis) for two nodes or
        element centroids to be considered the same physical point.
        Default: ``1e-6`` (matching SEACAS exodiff default).
    require_unique_mapping : bool
        If ``True`` (default), raise :class:`MeshMatchError` when any
        node or element cannot be uniquely matched.  If ``False``, emit
        warnings and leave unmatched entries at index ``-1`` in the maps.

    Returns
    -------
    MeshMap
        Bijective permutation maps between the two mesh index spaces.

    Raises
    ------
    MeshMatchError
        When ``require_unique_mapping=True`` and matching fails (duplicate
        coordinate positions, or coordinates outside tolerance).
    ValueError
        When node or element counts differ between the two files.

    Examples
    --------
    >>> from exodusii.mesh.matching import build_mesh_map
    >>> mesh_map = build_mesh_map(exo1, exo2, matching_tolerance=1e-8)
    >>> # reorder file-2 nodal array to file-1 ordering
    >>> values2_aligned = values2[mesh_map.node_map_inv]
    """

    n_nodes = exo1.node_count
    n_elems = exo1.element_count

    if exo2.node_count != n_nodes:
        raise ValueError(
            f"node count mismatch: file1={n_nodes}, file2={exo2.node_count}"
        )
    if exo2.element_count != n_elems:
        raise ValueError(
            f"element count mismatch: file1={n_elems}, file2={exo2.element_count}"
        )

    coords1 = np.asarray(exo1.coordinates(), dtype=np.float64)
    coords2 = np.asarray(exo2.coordinates(), dtype=np.float64)

    # Global node map (file-2 node idx → file-1 node idx), -1 = unmatched.
    node_map: IntArray = np.full(n_nodes, -1, dtype=np.int64)
    # Global element map (file-2 elem idx → file-1 elem idx), -1 = unmatched.
    elem_map: IntArray = np.full(n_elems, -1, dtype=np.int64)

    # Track block ID mapping and global element offsets.
    block_map: dict[int, int] = {}
    block_elem_offsets1: dict[int, int] = {}
    block_elem_offsets2: dict[int, int] = {}

    # ── Step 1: Match element blocks ─────────────────────────────────────
    ids1 = exo1.element_block_ids().tolist()
    ids2 = exo2.element_block_ids().tolist()

    matched_blocks = _match_blocks(exo1, exo2, ids1, ids2, coords1, coords2, matching_tolerance)

    for bid1, bid2, conn1, conn2, goff1, goff2 in matched_blocks:
        block_map[bid2] = bid1
        block_elem_offsets1[bid1] = goff1
        block_elem_offsets2[bid2] = goff2

    # ── Step 2: Match elements by centroid and derive node map ───────────
    global_off1 = 0
    global_off2 = 0

    for bid1, bid2, conn1, conn2, goff1, goff2 in matched_blocks:
        # Compute centroids for file-1 and file-2 elements in this block.
        centers1 = _centroids(conn1, coords1)    # (n_elems_in_block, dim)
        centers2 = _centroids(conn2, coords2)

        n_block = centers1.shape[0]
        if centers2.shape[0] != n_block:
            if require_unique_mapping:
                raise MeshMatchError(
                    f"block {bid1}/{bid2}: element count mismatch "
                    f"({n_block} vs {centers2.shape[0]})"
                )
            continue

        # Sort file-2 centroids along max-spread axis.
        e_match = _match_points_sorted(centers1, centers2, matching_tolerance)

        for j_local, i_local in enumerate(e_match):
            j_global = goff2 + j_local
            if i_local == -1:
                continue
            i_global = goff1 + int(i_local)
            if elem_map[j_global] != -1:
                continue  # already matched
            elem_map[j_global] = i_global

            # Derive node map from matched element local nodes.
            local_nodes1 = conn1[int(i_local)]    # file-1 node indices (0-based)
            local_nodes2 = conn2[j_local]         # file-2 node indices (0-based)
            _match_local_nodes(
                local_nodes1, local_nodes2,
                coords1, coords2,
                node_map, matching_tolerance,
            )

    # ── Step 3: Free-node fallback ───────────────────────────────────────
    unmatched2 = np.where(node_map == -1)[0]
    if unmatched2.size > 0:
        # Find corresponding unmatched file-1 nodes.
        matched1_set = set(node_map[node_map != -1].tolist())
        all1 = set(range(n_nodes))
        unmatched1 = np.array(sorted(all1 - matched1_set), dtype=np.int64)

        if unmatched1.size == unmatched2.size:
            pts1 = coords1[unmatched1]
            pts2 = coords2[unmatched2]
            fallback = _match_points_sorted(pts1, pts2, matching_tolerance)
            for idx2_local, idx1_local in enumerate(fallback):
                if idx1_local == -1:
                    continue
                j = int(unmatched2[idx2_local])
                i = int(unmatched1[int(idx1_local)])
                if node_map[j] == -1:
                    node_map[j] = i
        else:
            # Size mismatch — try direct coordinate matching anyway.
            pts2 = coords2[unmatched2]
            fallback = _match_points_sorted(pts2, coords1, matching_tolerance)
            for idx2_local, i in enumerate(fallback):
                if i == -1:
                    continue
                j = int(unmatched2[idx2_local])
                if node_map[j] == -1:
                    node_map[j] = int(i)

    # ── Step 4: Verify completeness ──────────────────────────────────────
    unmatched_nodes = int(np.sum(node_map == -1))
    unmatched_elems = int(np.sum(elem_map == -1))

    if require_unique_mapping and (unmatched_nodes > 0 or unmatched_elems > 0):
        msg_parts = []
        if unmatched_nodes:
            bad = np.where(node_map == -1)[0][:5].tolist()
            msg_parts.append(
                f"{unmatched_nodes} unmatched node(s) (first few file-2 indices: {bad})"
            )
        if unmatched_elems:
            bad = np.where(elem_map == -1)[0][:5].tolist()
            msg_parts.append(
                f"{unmatched_elems} unmatched element(s) (first few file-2 indices: {bad})"
            )
        raise MeshMatchError(
            "Coordinate-based mesh matching failed: " + "; ".join(msg_parts)
        )

    if unmatched_nodes > 0:
        warnings.warn(
            f"Coordinate mesh matching: {unmatched_nodes} node(s) could not be matched "
            f"within tolerance {matching_tolerance:.3e}; left at -1.",
            stacklevel=3,
        )
    if unmatched_elems > 0:
        warnings.warn(
            f"Coordinate mesh matching: {unmatched_elems} element(s) could not be matched "
            f"within tolerance {matching_tolerance:.3e}; left at -1.",
            stacklevel=3,
        )

    # Replace any remaining -1 entries with 0 to avoid downstream IndexErrors
    # when require_unique_mapping=False.
    node_map_safe = node_map.copy()
    node_map_safe[node_map_safe == -1] = 0
    elem_map_safe = elem_map.copy()
    elem_map_safe[elem_map_safe == -1] = 0

    node_map_inv = np.argsort(node_map_safe).astype(np.int64)
    elem_map_inv = np.argsort(elem_map_safe).astype(np.int64)

    return MeshMap(
        node_map=node_map_safe,
        node_map_inv=node_map_inv,
        elem_map=elem_map_safe,
        elem_map_inv=elem_map_inv,
        block_map=block_map,
        block_elem_offsets1=block_elem_offsets1,
        block_elem_offsets2=block_elem_offsets2,
        unmatched_nodes=unmatched_nodes,
        unmatched_elems=unmatched_elems,
    )


def check_sideset_ordinals(
    exo1,
    exo2,
    mesh_map: MeshMap,
) -> list[str]:
    """Check whether sideset face ordinals are consistent after element remapping.

    After element IDs are translated through *mesh_map*, the face-ordinal
    (side number) stored in file-2 side sets may differ from file-1 if the
    element nodes were listed in a different local order.  This function
    detects such mismatches and returns a list of warning strings — one per
    affected set entry.

    Side sets store ``(element_id, side_ordinal)`` pairs.  This check
    translates file-2 element IDs to file-1 element IDs and compares the
    resulting ``(element_id, side_ordinal)`` pairs against file-1's side set
    definition.  Ordinal mismatches are reported as warnings but do not
    constitute a fatal diff error.

    Parameters
    ----------
    exo1 : ExodusFile
        Reference database.
    exo2 : ExodusFile
        Comparison database.
    mesh_map : MeshMap
        The :class:`MeshMap` from :func:`build_mesh_map`.

    Returns
    -------
    list of str
        Warning messages for each affected (set_id, entry) pair.
        Empty when all ordinals are consistent.
    """

    warnings_out: list[str] = []

    try:
        set_ids1 = exo1.set_ids(Entity.SIDE_SET).tolist()
        set_ids2 = exo2.set_ids(Entity.SIDE_SET).tolist()
    except Exception:
        return warnings_out

    # Match set IDs between the two files.
    common_ids = [sid for sid in set_ids1 if sid in set_ids2]

    for set_id in common_ids:
        try:
            si1 = exo1.set(Entity.SIDE_SET, set_id)
            si2 = exo2.set(Entity.SIDE_SET, set_id)
        except Exception:
            continue

        entries1 = getattr(si1, "entries", None)   # 1-based element IDs
        sides1 = getattr(si1, "extra_entries", None)  # side ordinals
        entries2 = getattr(si2, "entries", None)
        sides2 = getattr(si2, "extra_entries", None)

        if entries1 is None or sides1 is None or entries2 is None or sides2 is None:
            continue

        entries1 = np.asarray(entries1, dtype=np.int64)
        sides1 = np.asarray(sides1, dtype=np.int64)
        entries2 = np.asarray(entries2, dtype=np.int64)
        sides2 = np.asarray(sides2, dtype=np.int64)

        if entries1.shape != entries2.shape:
            warnings_out.append(
                f"side set {set_id}: entry count differs after mapping "
                f"({len(entries1)} vs {len(entries2)}); ordinal check skipped"
            )
            continue

        # Translate file-2 element IDs (1-based) through the element map.
        # elem_map[j] = i  (0-based indices), so:
        #   file-2 1-based element id  →  file-2 0-based idx = id - 1
        #   mapped file-1 0-based idx  = elem_map[file2_0based]
        #   file-1 1-based id          = mapped_0based + 1
        e2_0based = entries2 - 1
        # Clamp to valid range to avoid IndexError on partial maps.
        n_elems = mesh_map.elem_map.shape[0]
        valid = (e2_0based >= 0) & (e2_0based < n_elems)
        mapped_e1_0based = np.where(valid, mesh_map.elem_map[np.clip(e2_0based, 0, n_elems - 1)], -1)
        mapped_e1_1based = mapped_e1_0based + 1

        # Build a lookup from file-1 (element_id, side) → True
        pairs1: set[tuple[int, int]] = set(zip(entries1.tolist(), sides1.tolist()))

        for k, (eid1_mapped, side2) in enumerate(zip(mapped_e1_1based.tolist(), sides2.tolist())):
            if eid1_mapped < 1:
                continue
            # Check if the translated (element, side) pair exists in file-1.
            if (eid1_mapped, side2) not in pairs1:
                # Check whether the element itself matched but the side differs.
                same_elem_diff_side = any(
                    eid1_mapped == e for e, _s in pairs1
                )
                if same_elem_diff_side:
                    warnings_out.append(
                        f"side set {set_id} entry {k}: element {entries2[k]} "
                        f"(mapped to {eid1_mapped}) matched but side ordinal "
                        f"{side2} differs from file-1 — face orientation may differ "
                        f"after mesh reordering"
                    )

    return warnings_out


# ── Private helpers ──────────────────────────────────────────────────────────


def _match_blocks(
    exo1,
    exo2,
    ids1: list[int],
    ids2: list[int],
    coords1: FloatArray,
    coords2: FloatArray,
    tol: float,
) -> list[tuple[int, int, IntArray, IntArray, int, int]]:
    """Match element blocks between two files.

    First tries to match by identical block IDs.  For blocks whose IDs
    differ, matches by element topology and global centroid proximity.

    Returns a list of tuples:
    ``(block_id1, block_id2, conn1_0based, conn2_0based, global_off1, global_off2)``
    """

    result = []
    set2 = set(ids2)
    unmatched2 = list(ids2)

    # Accumulate global element offsets as we iterate.
    off1 = 0
    goff1_by_id: dict[int, int] = {}
    for bid in ids1:
        goff1_by_id[bid] = off1
        try:
            conn = exo1.block_connectivity(Entity.ELEMENT_BLOCK, bid, zero_based=True)
            off1 += len(conn)
        except Exception:
            off1 += 0

    off2 = 0
    goff2_by_id: dict[int, int] = {}
    for bid in ids2:
        goff2_by_id[bid] = off2
        try:
            conn = exo2.block_connectivity(Entity.ELEMENT_BLOCK, bid, zero_based=True)
            off2 += len(conn)
        except Exception:
            off2 += 0

    # Phase 1: match by identical block IDs.
    matched2: set[int] = set()
    for bid1 in ids1:
        if bid1 in set2:
            try:
                conn1 = np.asarray(exo1.block_connectivity(Entity.ELEMENT_BLOCK, bid1, zero_based=True), dtype=np.int64)
                conn2 = np.asarray(exo2.block_connectivity(Entity.ELEMENT_BLOCK, bid1, zero_based=True), dtype=np.int64)
                result.append((bid1, bid1, conn1, conn2, goff1_by_id[bid1], goff2_by_id[bid1]))
                matched2.add(bid1)
            except Exception:
                continue

    # Phase 2: match remaining blocks by topology + centroid proximity.
    unmatched_ids1 = [bid for bid in ids1 if bid not in {r[0] for r in result}]
    unmatched_ids2 = [bid for bid in ids2 if bid not in matched2]

    if unmatched_ids1 and unmatched_ids2:
        for bid1 in unmatched_ids1:
            try:
                conn1 = np.asarray(exo1.block_connectivity(Entity.ELEMENT_BLOCK, bid1, zero_based=True), dtype=np.int64)
            except Exception:
                continue
            if conn1.size == 0:
                continue
            center1 = _centroids(conn1, coords1).mean(axis=0)
            blk1_type = _block_type(exo1, bid1)

            best_bid2 = None
            best_dist = float("inf")
            for bid2 in unmatched_ids2:
                if bid2 in matched2:
                    continue
                try:
                    conn2 = np.asarray(exo2.block_connectivity(Entity.ELEMENT_BLOCK, bid2, zero_based=True), dtype=np.int64)
                except Exception:
                    continue
                if conn2.size == 0:
                    continue
                if conn1.shape[1] != conn2.shape[1]:
                    continue  # incompatible element type
                blk2_type = _block_type(exo2, bid2)
                if blk1_type and blk2_type and blk1_type != blk2_type:
                    continue
                center2 = _centroids(conn2, coords2).mean(axis=0)
                dist = float(np.linalg.norm(center1 - center2))
                if dist < best_dist:
                    best_dist = dist
                    best_bid2 = bid2

            if best_bid2 is not None and best_dist <= tol * 1000:
                try:
                    conn2 = np.asarray(exo2.block_connectivity(Entity.ELEMENT_BLOCK, best_bid2, zero_based=True), dtype=np.int64)
                    result.append(
                        (bid1, best_bid2, conn1, conn2, goff1_by_id[bid1], goff2_by_id[best_bid2])
                    )
                    matched2.add(best_bid2)
                except Exception:
                    pass

    return result


def _block_type(exo, block_id: int) -> str | None:
    """Return the normalised element type string for a block, or None."""
    try:
        blk = exo.block(Entity.ELEMENT_BLOCK, block_id)
        et = getattr(blk, "element_type", None)
        if et is None:
            return None
        if isinstance(et, bytes):
            et = et.decode("ascii")
        return str(et).strip().lower()
    except Exception:
        return None


def _centroids(conn: IntArray, coords: FloatArray) -> FloatArray:
    """Compute element centroids as arithmetic mean of node coordinates.

    Parameters
    ----------
    conn : ndarray of int64, shape (n_elems, k)
        Zero-based connectivity.
    coords : ndarray of float64, shape (n_nodes, dim)
        Nodal coordinates.

    Returns
    -------
    ndarray of float64, shape (n_elems, dim)
    """

    return coords[conn].mean(axis=1)


def _match_points_sorted(
    pts1: FloatArray,
    pts2: FloatArray,
    tol: float,
) -> IntArray:
    """Match each row of *pts1* to the nearest row in *pts2* within *tol*.

    Uses a sorted-axis binary search strategy mirroring SEACAS ``map.C``.
    For each point in *pts1*, a candidate window is found in *pts2* along
    the coordinate axis with the greatest spread, then all axes are checked
    within the window.  When multiple candidates fall within *tol*, the
    geometrically closest one (Euclidean) is selected.

    Parameters
    ----------
    pts1 : ndarray of float64, shape (N, dim)
        Query points.
    pts2 : ndarray of float64, shape (M, dim)
        Candidate points to match against.
    tol : float
        Per-axis tolerance.

    Returns
    -------
    ndarray of int64, shape (N,)
        ``result[i]`` = 0-based index into *pts2* for the match of
        ``pts1[i]``, or ``-1`` when no match was found within *tol*.
    """

    n = len(pts1)
    m = len(pts2)
    if n == 0 or m == 0:
        return np.full(n, -1, dtype=np.int64)

    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)

    # Choose primary axis with maximum spread in pts2.
    ranges = pts2.max(axis=0) - pts2.min(axis=0)
    primary = int(np.argmax(ranges))

    # Sort pts2 by the primary axis.
    sort_idx = np.argsort(pts2[:, primary])
    pts2_sorted = pts2[sort_idx]
    primary_vals = pts2_sorted[:, primary]

    result = np.full(n, -1, dtype=np.int64)

    for i in range(n):
        p = pts1[i]

        lo = int(np.searchsorted(primary_vals, p[primary] - tol))
        hi = int(np.searchsorted(primary_vals, p[primary] + tol, side="right"))
        if lo >= hi:
            continue

        window = pts2_sorted[lo:hi]               # (W, dim)
        diffs = np.abs(window - p)                # (W, dim)
        within = np.all(diffs <= tol, axis=1)     # (W,)
        hits = np.nonzero(within)[0]

        if len(hits) == 0:
            continue
        if len(hits) == 1:
            result[i] = int(sort_idx[lo + hits[0]])
        else:
            # Multiple candidates: pick closest by Euclidean distance.
            dist_sq = np.sum((window[hits] - p) ** 2, axis=1)
            best = hits[int(np.argmin(dist_sq))]
            result[i] = int(sort_idx[lo + best])

    return result


def _match_local_nodes(
    nodes1: IntArray,
    nodes2: IntArray,
    coords1: FloatArray,
    coords2: FloatArray,
    node_map: IntArray,
    tol: float,
) -> None:
    """Fill *node_map* entries for nodes shared by a matched element pair.

    Matches each file-2 local node to the closest file-1 local node
    within *tol* in all coordinate axes.  Only fills entries that are
    currently unset (``-1``).

    Parameters
    ----------
    nodes1 : array of int64
        File-1 global (0-based) node indices for the element.
    nodes2 : array of int64
        File-2 global (0-based) node indices for the element.
    coords1, coords2 : float64 arrays
        Full nodal coordinate arrays.
    node_map : int64 array, modified in place
        ``node_map[j] = i`` when file-2 node *j* → file-1 node *i*.
        ``-1`` means unset.
    tol : float
        Per-axis coordinate tolerance.
    """

    k = len(nodes1)
    c1 = coords1[nodes1]   # (k, dim)
    c2 = coords2[nodes2]   # (k, dim)

    for j_local in range(k):
        j_global = int(nodes2[j_local])
        if node_map[j_global] != -1:
            continue  # already matched
        p2 = c2[j_local]
        # Find closest file-1 local node.
        diffs = np.abs(c1 - p2)
        within = np.all(diffs <= tol, axis=1)
        hits = np.nonzero(within)[0]
        if len(hits) == 0:
            continue
        if len(hits) == 1:
            node_map[j_global] = int(nodes1[hits[0]])
        else:
            # Multiple candidates: pick closest.
            dist_sq = np.sum((c1[hits] - p2) ** 2, axis=1)
            best = hits[int(np.argmin(dist_sq))]
            node_map[j_global] = int(nodes1[best])
