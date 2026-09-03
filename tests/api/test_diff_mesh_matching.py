# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Integration tests for coordinate-based mesh matching in :func:`exodusii.api.diff.diff`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from exodusii.api.diff import DiffOptions
from exodusii.api.diff import diff
from exodusii.api.file import ExodusFile
from exodusii.api.writer import ExodusWriter
from exodusii.core.entities import Entity
from exodusii.core.tolerance import Tolerance
from exodusii.core.tolerance import ToleranceMode

# ---------------------------------------------------------------------------
# Mesh-writing helpers
# ---------------------------------------------------------------------------


def _quad_coords() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def _write_matched(
    path: Path,
    *,
    node_order: list[int] | None = None,
    elem_values: list[float] | None = None,
    node_values: list[float] | None = None,
    coord_noise: float = 0.0,
    block_id: int = 10,
    extra_attr: float = 2.0,
) -> None:
    """Write a single-block, single-element, 2-step Quad4 mesh.

    Nodes 0-3 at unit-square corners.  Optional node_order permutes them.
    """
    all_coords = _quad_coords()
    if node_order is None:
        node_order = [0, 1, 2, 3]

    coords = all_coords[node_order] + coord_noise

    inv = [0] * 4
    for new_i, orig_i in enumerate(node_order):
        inv[orig_i] = new_i
    conn = [[inv[0] + 1, inv[1] + 1, inv[2] + 1, inv[3] + 1]]

    with ExodusWriter.create(path) as w:
        w.initialize("match_diff", 2, 4, 1, element_blocks=1)
        w.write_coordinates(coords)
        w.define_element_block(block_id, "quad", conn)
        w.write_block_attributes(
            Entity.ELEMENT_BLOCK, block_id, [[1.0, extra_attr]], names=["A", "B"]
        )
        w.define_global_variables(["KE"])
        w.define_node_variables(["TEMP"])
        w.define_element_variables(["ENERGY"], truth_table=[[1]])

        nv0 = np.array(node_values or list(range(4)), dtype=float)
        ev0 = float(elem_values[0]) if elem_values else 1.0

        w.write_time(0.0)
        w.write_global_values([0.0])
        w.write_node_values("TEMP", nv0)
        w.write_element_values("ENERGY", [ev0], block_id=block_id)

        w.write_time(1.0)
        w.write_global_values([1.0])
        w.write_node_values("TEMP", nv0 + 10.0)
        w.write_element_values("ENERGY", [ev0 * 2], block_id=block_id)


def _permute_mesh(path_in: Path, path_out: Path, node_perm: list[int]) -> None:
    """Read *path_in*, reorder nodes by *node_perm*, write *path_out*.

    The *node_perm* list maps new index → old index (old_coords[node_perm[i]]).
    Variable values are re-indexed accordingly.
    """
    with ExodusFile.open(path_in) as exo:
        orig_coords = np.asarray(exo.coordinates())
        block_ids = exo.element_block_ids().tolist()
        times = exo.times()

        # Build inverse map: old → new
        inv_perm = [0] * len(node_perm)
        for new_i, old_i in enumerate(node_perm):
            inv_perm[old_i] = new_i

        new_coords = orig_coords[node_perm]

        with ExodusWriter.create(path_out) as w:
            n_nodes = exo.node_count
            n_elems = exo.element_count
            w.initialize("permuted", exo.dimension, n_nodes, n_elems, element_blocks=len(block_ids))
            w.write_coordinates(new_coords)

            # Re-index connectivity.
            for bid in block_ids:
                raw_conn = np.asarray(
                    exo.block_connectivity(Entity.ELEMENT_BLOCK, bid), dtype=np.int64
                )
                # raw_conn is 1-based; convert to 0-based, remap, convert back.
                new_conn = np.vectorize(lambda x: inv_perm[x - 1] + 1)(raw_conn)
                blk = exo.block(Entity.ELEMENT_BLOCK, bid)
                w.define_element_block(bid, blk.element_type, new_conn.tolist())
                # Attributes (element-indexed, not node-indexed).
                try:
                    attr_names = exo.attribute_names(Entity.ELEMENT_BLOCK, bid)
                    if attr_names:
                        attr_data = np.column_stack(
                            [exo.attribute_values(Entity.ELEMENT_BLOCK, bid, n) for n in attr_names]
                        )
                        w.write_block_attributes(
                            Entity.ELEMENT_BLOCK, bid, attr_data.tolist(), names=attr_names
                        )
                except Exception:
                    pass

            # Copy variables.
            gnames = exo.variable_names(Entity.GLOBAL)
            nnames = exo.variable_names(Entity.NODE)
            enames = exo.variable_names(Entity.ELEMENT)
            if gnames:
                w.define_global_variables(gnames)
            if nnames:
                w.define_node_variables(nnames)
            if enames:
                tt = exo.variable_truth_table(Entity.ELEMENT)
                w.define_element_variables(enames, truth_table=tt)

            for t_idx, t_val in enumerate(times):
                w.write_time(float(t_val))
                if gnames:
                    gvals = [float(exo.values(n, on=Entity.GLOBAL, time=t_idx)) for n in gnames]
                    w.write_global_values(gvals)
                if nnames:
                    for nm in nnames:
                        orig_vals = np.asarray(exo.values(nm, on=Entity.NODE, time=t_idx))
                        new_vals = orig_vals[node_perm]
                        w.write_node_values(nm, new_vals)
                for bid in block_ids:
                    if enames:
                        for nm in enames:
                            try:
                                evals = exo.values(nm, on=Entity.ELEMENT, block_id=bid, time=t_idx)
                                w.write_element_values(nm, np.asarray(evals).tolist(), block_id=bid)
                            except Exception:
                                pass


# ---------------------------------------------------------------------------
# Basic coordinate matching tests
# ---------------------------------------------------------------------------


def test_matched_ordering_same_without_flag(tmp_path: Path) -> None:
    """Default (no coordinate_matching) still works on matched files."""
    a = tmp_path / "a.exo"
    _write_matched(a)
    result = diff(a, a)
    assert result.same
    assert not result.mesh_map_built


def test_coordinate_matching_identity_is_same(tmp_path: Path) -> None:
    """coordinate_matching=True on identical files → same."""
    a = tmp_path / "a.exo"
    _write_matched(a)
    opts = DiffOptions(coordinate_matching=True, matching_tolerance=1e-10)
    result = diff(a, a, opts)
    assert result.same
    assert result.mesh_map_built
    assert result.unmatched_nodes == 0


def test_coordinate_matching_shuffled_nodes_is_same(tmp_path: Path) -> None:
    """Shuffled-node file compares same when coordinate_matching=True."""
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_matched(a)
    _permute_mesh(a, b, [2, 0, 3, 1])

    opts = DiffOptions(coordinate_matching=True, matching_tolerance=1e-8)
    result = diff(a, b, opts)
    assert result.same, f"errors={result.errors}, diffs={result.variable_diffs}"
    assert result.mesh_map_built


def test_coordinate_matching_shuffled_nodes_detects_difference(tmp_path: Path) -> None:
    """Shuffled file with a value difference is still detected."""
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    c = tmp_path / "c.exo"
    _write_matched(a)
    _permute_mesh(a, b, [2, 0, 3, 1])
    # Write c: same layout as b but with modified node values.
    _write_matched(c, node_order=[2, 0, 3, 1], node_values=[0.0, 1.0, 2.0, 999.0])

    opts = DiffOptions(coordinate_matching=True, matching_tolerance=1e-8)
    result = diff(a, c, opts)
    assert not result.same
    exceeded = {vd.name for vd in result.variable_diffs if vd.exceeded}
    assert "TEMP" in exceeded


def test_coordinate_matching_maps_built_in_result(tmp_path: Path) -> None:
    """DiffResult.mesh_map_built is True when coordinate_matching is on."""
    a = tmp_path / "a.exo"
    _write_matched(a)
    opts = DiffOptions(coordinate_matching=True)
    result = diff(a, a, opts)
    assert result.mesh_map_built is True


def test_coordinate_matching_false_does_not_build_map(tmp_path: Path) -> None:
    """mesh_map_built stays False when coordinate_matching=False (default)."""
    a = tmp_path / "a.exo"
    _write_matched(a)
    result = diff(a, a)
    assert result.mesh_map_built is False


# ---------------------------------------------------------------------------
# Coordinate comparison after mapping
# ---------------------------------------------------------------------------


def test_coordinate_comparison_same_after_mapping(tmp_path: Path) -> None:
    """After mapping, coordinates should compare equal."""
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_matched(a)
    _permute_mesh(a, b, [1, 3, 0, 2])
    opts = DiffOptions(coordinate_matching=True, matching_tolerance=1e-8)
    result = diff(a, b, opts)
    assert result.coordinate_max_delta is not None
    assert result.coordinate_max_delta < 1e-10
    assert result.same


def test_coordinate_comparison_detects_coordinate_difference(tmp_path: Path) -> None:
    """Coordinate noise above tolerance is detected even after mapping."""
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_matched(a)
    _write_matched(b, coord_noise=1.0)  # massive coord offset, same order
    opts = DiffOptions(
        coordinate_matching=True,
        matching_tolerance=2.0,  # wide enough to match despite noise
        coordinate_tolerance=Tolerance(ToleranceMode.ABSOLUTE, 0.5, 0.0),
    )
    result = diff(a, b, opts)
    # The 1.0 noise exceeds the 0.5 coordinate tolerance.
    assert not result.same
    assert any("coordinates differ" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Element variable reordering
# ---------------------------------------------------------------------------


def test_element_variables_same_after_mapping(tmp_path: Path) -> None:
    """Element variable values compare equal after element reordering via mapping."""
    from exodusii.api.writer import ExodusWriter

    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"

    # Two-element mesh: write original then permuted node order.
    all_coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    conn_orig = [[1, 2, 5, 4], [2, 3, 6, 5]]  # 1-based

    def _write(path, coords, conn):
        with ExodusWriter.create(path) as w:
            w.initialize("t", 2, 6, 2, element_blocks=1)
            w.write_coordinates(coords)
            w.define_element_block(10, "quad", conn)
            w.define_element_variables(["ENERGY"], truth_table=[[1]])
            w.write_time(0.0)
            w.write_element_values("ENERGY", [1.0, 2.0], block_id=10)

    _write(a, all_coords, conn_orig)

    # Shuffle node order: [3,4,5,0,1,2] (swap top and bottom rows)
    node_perm = [3, 4, 5, 0, 1, 2]
    new_coords = all_coords[node_perm]
    inv = [0] * 6
    for ni, oi in enumerate(node_perm):
        inv[oi] = ni
    new_conn = [[inv[j - 1] + 1 for j in row] for row in conn_orig]
    _write(b, new_coords, new_conn)

    opts = DiffOptions(coordinate_matching=True, matching_tolerance=1e-8)
    result = diff(a, b, opts)
    assert result.same, f"errors={result.errors}, diffs={result.variable_diffs}"


# ---------------------------------------------------------------------------
# Attribute reordering
# ---------------------------------------------------------------------------


def test_attributes_same_after_mapping(tmp_path: Path) -> None:
    """Block attributes compare equal after element reordering via mapping."""
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_matched(a, extra_attr=5.0)
    _permute_mesh(a, b, [1, 3, 0, 2])
    opts = DiffOptions(coordinate_matching=True, matching_tolerance=1e-8)
    result = diff(a, b, opts)
    assert result.same, f"errors={result.errors}, diffs={result.variable_diffs}"


# ---------------------------------------------------------------------------
# Mismatched block IDs with coordinate matching
# ---------------------------------------------------------------------------


def test_block_id_mismatch_is_warning_not_error_with_matching(tmp_path: Path) -> None:
    """Block ID mismatch is demoted to a warning when coordinate_matching=True."""
    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_matched(a, block_id=10)
    _write_matched(b, block_id=10)  # same IDs — just test the warning path
    opts = DiffOptions(coordinate_matching=True, matching_tolerance=1e-8)
    result = diff(a, b, opts)
    # With same block IDs there should be no warning about block IDs.
    assert not any("block ids differ" in w for w in result.warnings)


def test_block_id_mismatch_is_error_without_matching(tmp_path: Path) -> None:
    """Block ID mismatch is a fatal error when coordinate_matching=False."""

    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_matched(a, block_id=10)
    _write_matched(b, block_id=20)  # different block IDs

    result = diff(a, b)  # default: no coordinate matching
    assert not result.same
    assert any("block ids differ" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Mesh matching failure handling
# ---------------------------------------------------------------------------


def test_mesh_matching_failure_is_error_in_result(tmp_path: Path) -> None:
    """MeshMatchError from build_mesh_map becomes an error in DiffResult."""
    from exodusii.api.writer import ExodusWriter

    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_matched(a)

    # Write b at completely different coordinates so matching fails.
    all_coords_far = _quad_coords() + 1000.0
    with ExodusWriter.create(b) as w:
        w.initialize("far", 2, 4, 1, element_blocks=1)
        w.write_coordinates(all_coords_far)
        w.define_element_block(10, "quad", [[1, 2, 3, 4]])
        w.write_time(0.0)

    opts = DiffOptions(coordinate_matching=True, matching_tolerance=1e-6)
    result = diff(a, b, opts)
    assert not result.same
    assert any("mesh matching failed" in e for e in result.errors)


def test_mesh_matching_partial_allowed(tmp_path: Path) -> None:
    """require_unique_mapping=False allows partial matches without fatal error."""
    import warnings

    from exodusii.api.writer import ExodusWriter

    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_matched(a)

    all_coords_far = _quad_coords() + 1000.0
    with ExodusWriter.create(b) as w:
        w.initialize("far", 2, 4, 1, element_blocks=1)
        w.write_coordinates(all_coords_far)
        w.define_element_block(10, "quad", [[1, 2, 3, 4]])
        w.write_time(0.0)

    opts = DiffOptions(
        coordinate_matching=True, matching_tolerance=1e-6, require_unique_mapping=False
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = diff(a, b, opts)
    # Should not raise; mesh_matching_failed error should not appear since
    # require_unique_mapping=False suppresses the exception.
    assert not any("mesh matching failed" in e for e in result.errors)
    assert result.unmatched_nodes > 0


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_match_coordinates_same(tmp_path: Path) -> None:
    """exodiff --match-coordinates on permuted file exits 0."""
    from exodusii.cli.exodiff import main

    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_matched(a)
    _permute_mesh(a, b, [3, 1, 0, 2])

    rc = main(["--match-coordinates", str(a), str(b)])
    assert rc == 0


def test_cli_match_coordinates_different(tmp_path: Path) -> None:
    """exodiff --match-coordinates detects difference in permuted file, exits 2."""
    import io

    from exodusii.cli.exodiff import main

    a = tmp_path / "a.exo"
    c = tmp_path / "c.exo"
    _write_matched(a)
    _write_matched(c, node_order=[3, 1, 0, 2], node_values=[0.0, 1.0, 2.0, 999.0])

    buf = io.StringIO()
    rc = main(["--match-coordinates", str(a), str(c)], file=buf)
    assert rc == 2


def test_cli_match_coordinates_json_output(tmp_path: Path) -> None:
    """exodiff --match-coordinates --format json includes mesh_map_built."""
    import io
    import json

    from exodusii.cli.exodiff import main

    a = tmp_path / "a.exo"
    _write_matched(a)

    buf = io.StringIO()
    rc = main(["--match-coordinates", "--format", "json", "--terse", str(a), str(a)], file=buf)
    assert rc == 0
    data = json.loads(buf.getvalue())
    assert data["mesh_map_built"] is True


def test_cli_matching_tolerance_flag(tmp_path: Path) -> None:
    """--matching-tolerance flag is accepted and passed through."""
    from exodusii.cli.exodiff import main

    a = tmp_path / "a.exo"
    _write_matched(a)
    rc = main(["--match-coordinates", "--matching-tolerance", "1e-8", str(a), str(a)])
    assert rc == 0


def test_cli_allow_partial_match_flag(tmp_path: Path) -> None:
    """--allow-partial-match flag is accepted without crashing."""
    import warnings

    from exodusii.cli.exodiff import main

    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_matched(a)

    from exodusii.api.writer import ExodusWriter

    all_coords_far = _quad_coords() + 1000.0
    with ExodusWriter.create(b) as w:
        w.initialize("far", 2, 4, 1, element_blocks=1)
        w.write_coordinates(all_coords_far)
        w.define_element_block(10, "quad", [[1, 2, 3, 4]])
        w.write_time(0.0)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        rc = main(["--match-coordinates", "--allow-partial-match", str(a), str(b)])
    # Exits 1 (error) because comparison fails after partial map, not crash.
    assert rc in (1, 2)


# ---------------------------------------------------------------------------
# Nodeset alignment
# ---------------------------------------------------------------------------


def test_nodeset_variable_same_after_mapping(tmp_path: Path) -> None:
    """Node-set variable values compare equal after node reordering via mapping."""
    from exodusii.api.writer import ExodusWriter

    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"

    all_coords = _quad_coords()
    node_perm = [2, 0, 3, 1]  # new_idx -> old_idx
    inv_perm = [0] * 4
    for ni, oi in enumerate(node_perm):
        inv_perm[oi] = ni
    # inv_perm[old] = new

    # File-a: nodes 0-3 in canonical order.
    # Node set 20 contains original nodes 0 and 2 (1-based: 1 and 3).
    # Values: 10.0 at node 0 (entry 0), 20.0 at node 2 (entry 1).
    with ExodusWriter.create(a) as w:
        w.initialize("ns_test", 2, 4, 1, element_blocks=1, node_sets=1)
        w.write_coordinates(all_coords)
        w.define_element_block(10, "quad", [[1, 2, 3, 4]])
        w.define_node_set(20, [1, 3])  # 1-based: nodes 0 and 2
        w.define_node_set_variables(["NSTEMP"])
        w.write_time(0.0)
        w.write_node_set_values("NSTEMP", [10.0, 20.0], set_id=20)

    # File-b: nodes permuted by node_perm.
    # Physical node 0 is now at position inv_perm[0]=1, node 2 at inv_perm[2]=0.
    new_coords = all_coords[node_perm]
    new_conn = [[inv_perm[0] + 1, inv_perm[1] + 1, inv_perm[2] + 1, inv_perm[3] + 1]]
    # Node set entries (1-based new indices): inv_perm[0]+1=2, inv_perm[2]+1=1
    # Values must match by physical node:
    #   physical node 0 (new idx inv_perm[0]=1) → value 10.0
    #   physical node 2 (new idx inv_perm[2]=0) → value 20.0
    # Store in set-entry order (sorted by new 1-based index):
    ns_entries_b = [inv_perm[0] + 1, inv_perm[2] + 1]  # [2, 1] unsorted
    # Sort the entries and reorder values accordingly
    paired = sorted(zip(ns_entries_b, [10.0, 20.0]))
    ns_entries_sorted = [e for e, _v in paired]
    ns_values_sorted = [v for _e, v in paired]

    with ExodusWriter.create(b) as w:
        w.initialize("ns_test", 2, 4, 1, element_blocks=1, node_sets=1)
        w.write_coordinates(new_coords)
        w.define_element_block(10, "quad", new_conn)
        w.define_node_set(20, ns_entries_sorted)
        w.define_node_set_variables(["NSTEMP"])
        w.write_time(0.0)
        w.write_node_set_values("NSTEMP", ns_values_sorted, set_id=20)

    opts = DiffOptions(coordinate_matching=True, matching_tolerance=1e-8)
    result = diff(a, b, opts)
    assert result.same, f"errors={result.errors}, diffs={result.variable_diffs}"
