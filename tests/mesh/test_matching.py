# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Unit tests for :mod:`exodusii.mesh.matching`."""

from pathlib import Path

import numpy as np
import pytest

from exodusii.mesh.matching import MeshMap
from exodusii.mesh.matching import MeshMatchError
from exodusii.mesh.matching import _match_points_sorted
from exodusii.mesh.matching import build_mesh_map

# ---------------------------------------------------------------------------
# _match_points_sorted unit tests
# ---------------------------------------------------------------------------


def test_match_points_sorted_identity():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    result = _match_points_sorted(pts, pts, tol=1e-10)
    np.testing.assert_array_equal(result, [0, 1, 2, 3])


def test_match_points_sorted_permuted():
    pts1 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    pts2 = pts1[[2, 0, 3, 1]]  # shuffle
    result = _match_points_sorted(pts1, pts2, tol=1e-10)
    # pts1[i] should match pts2[result[i]]
    for i, j in enumerate(result):
        np.testing.assert_allclose(pts1[i], pts2[j], atol=1e-12)


def test_match_points_sorted_no_match():
    pts1 = np.array([[0.0, 0.0]])
    pts2 = np.array([[5.0, 5.0]])
    result = _match_points_sorted(pts1, pts2, tol=1e-6)
    assert result[0] == -1


def test_match_points_sorted_multiple_candidates_picks_closest():
    # Two candidates within tol; should pick the closer one.
    pts1 = np.array([[0.0, 0.0]])
    pts2 = np.array([[0.0, 0.5e-7], [0.0, 0.9e-7]])  # both within 1e-6
    result = _match_points_sorted(pts1, pts2, tol=1e-6)
    assert result[0] == 0  # 0.5e-7 is closer than 0.9e-7


def test_match_points_sorted_3d():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 10, size=(50, 3))
    perm = rng.permutation(50)
    pts_shuffled = pts[perm]
    result = _match_points_sorted(pts, pts_shuffled, tol=1e-8)
    for i, j in enumerate(result):
        assert j != -1, f"point {i} unmatched"
        np.testing.assert_allclose(pts[i], pts_shuffled[j], atol=1e-10)


def test_match_points_sorted_empty_pts1():
    result = _match_points_sorted(np.zeros((0, 3)), np.ones((5, 3)), tol=1e-6)
    assert result.shape == (0,)


def test_match_points_sorted_empty_pts2():
    pts1 = np.ones((3, 2))
    result = _match_points_sorted(pts1, np.zeros((0, 2)), tol=1e-6)
    np.testing.assert_array_equal(result, [-1, -1, -1])


# ---------------------------------------------------------------------------
# MeshMap helpers
# ---------------------------------------------------------------------------


def test_meshmap_identity():
    mm = MeshMap.identity(6, 4)
    np.testing.assert_array_equal(mm.node_map, [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(mm.node_map_inv, [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(mm.elem_map, [0, 1, 2, 3])
    np.testing.assert_array_equal(mm.elem_map_inv, [0, 1, 2, 3])
    assert mm.unmatched_nodes == 0
    assert mm.unmatched_elems == 0


def test_meshmap_node_map_inv_is_true_inverse():
    # Build a nontrivial permutation and verify inverse.
    perm = np.array([3, 1, 0, 2], dtype=np.int64)
    inv = np.argsort(perm).astype(np.int64)
    mm = MeshMap(
        node_map=perm,
        node_map_inv=inv,
        elem_map=np.arange(2, dtype=np.int64),
        elem_map_inv=np.arange(2, dtype=np.int64),
    )
    for i in range(4):
        assert mm.node_map[mm.node_map_inv[i]] == i


# ---------------------------------------------------------------------------
# build_mesh_map integration tests (use ExodusWriter to create fixtures)
# ---------------------------------------------------------------------------


def _write_quad_mesh(path: Path, node_order: list[int], elem_order: list[int]) -> None:
    """Write a 4-node, 1-element quad mesh with nodes/elements in given order."""
    from exodusii.api.writer import ExodusWriter

    # Original coords: [0,0], [1,0], [1,1], [0,1]
    all_coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    coords = all_coords[node_order]

    # Original connectivity (1-based): [1,2,3,4]
    # After node reordering: node original index i is now at position node_order.index(i)
    inv_node = [0] * 4
    for new_idx, orig_idx in enumerate(node_order):
        inv_node[orig_idx] = new_idx
    # 1-based connectivity in new node numbering
    conn = [[inv_node[0] + 1, inv_node[1] + 1, inv_node[2] + 1, inv_node[3] + 1]]

    with ExodusWriter.create(path) as w:
        w.initialize("match_test", 2, 4, 1, element_blocks=1)
        w.write_coordinates(coords)
        w.define_element_block(10, "quad", conn)
        w.define_node_variables(["TEMP"])
        # Write node values in new ordering
        node_values = np.array([float(i) for i in node_order])
        w.write_time(0.0)
        w.write_node_values("TEMP", node_values)


def _write_two_quad_mesh(path: Path, node_order: list[int]) -> None:
    """Write a 6-node, 2-element quad mesh (two side-by-side quads)."""
    from exodusii.api.writer import ExodusWriter

    # Original layout:
    # Nodes 0-5: [0,0],[1,0],[2,0],[0,1],[1,1],[2,1]
    # Elem 0: [0,1,4,3]  Elem 1: [1,2,5,4]
    all_coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    coords = all_coords[node_order]
    inv_node = [0] * 6
    for new_idx, orig_idx in enumerate(node_order):
        inv_node[orig_idx] = new_idx

    conn_orig = [[0, 1, 4, 3], [1, 2, 5, 4]]
    conn = [[inv_node[j] + 1 for j in row] for row in conn_orig]

    with ExodusWriter.create(path) as w:
        w.initialize("two_quad", 2, 6, 2, element_blocks=1)
        w.write_coordinates(coords)
        w.define_element_block(10, "quad", conn)
        w.define_node_variables(["TEMP"])
        w.define_element_variables(["ENERGY"], truth_table=[[1]])
        node_values = np.array([float(i) for i in node_order])
        w.write_time(0.0)
        w.write_node_values("TEMP", node_values)
        w.write_element_values("ENERGY", [1.0, 2.0], block_id=10)


def test_build_mesh_map_identity(tmp_path: Path) -> None:
    """Identity permutation: map should be trivial."""
    from exodusii.api.file import ExodusFile

    a = tmp_path / "a.exo"
    _write_quad_mesh(a, [0, 1, 2, 3], [0])
    with ExodusFile.open(a) as exo1, ExodusFile.open(a) as exo2:
        mm = build_mesh_map(exo1, exo2, matching_tolerance=1e-10)
    np.testing.assert_array_equal(mm.node_map, mm.node_map[mm.node_map_inv])  # bijection
    assert mm.unmatched_nodes == 0
    assert mm.unmatched_elems == 0


def test_build_mesh_map_shuffled_nodes(tmp_path: Path) -> None:
    """Permuted nodes: build_mesh_map should recover the correct mapping."""
    from exodusii.api.file import ExodusFile

    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_quad_mesh(a, [0, 1, 2, 3], [0])
    _write_quad_mesh(b, [2, 0, 3, 1], [0])  # different node order

    with ExodusFile.open(a) as exo1, ExodusFile.open(b) as exo2:
        mm = build_mesh_map(exo1, exo2, matching_tolerance=1e-8)
        coords1 = np.asarray(exo1.coordinates())
        coords2 = np.asarray(exo2.coordinates())

    # After applying node_map_inv, coords2 should equal coords1.
    np.testing.assert_allclose(coords1, coords2[mm.node_map_inv], atol=1e-12)
    assert mm.unmatched_nodes == 0
    assert mm.unmatched_elems == 0


def test_build_mesh_map_count_mismatch_raises(tmp_path: Path) -> None:
    """Mismatched node count should raise ValueError."""
    from exodusii.api.file import ExodusFile
    from exodusii.api.writer import ExodusWriter

    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_quad_mesh(a, [0, 1, 2, 3], [0])

    # Write b with 5 nodes instead of 4.
    with ExodusWriter.create(b) as w:
        w.initialize("x", 2, 5, 1, element_blocks=1)
        w.write_coordinates(np.zeros((5, 2)))
        w.define_element_block(10, "quad", [[1, 2, 3, 4]])
        w.write_time(0.0)

    with (
        ExodusFile.open(a) as exo1,
        ExodusFile.open(b) as exo2,
        pytest.raises(ValueError, match="node count mismatch"),
    ):
        build_mesh_map(exo1, exo2)


def test_build_mesh_map_no_match_raises(tmp_path: Path) -> None:
    """Points far apart should fail with MeshMatchError when require_unique_mapping."""
    from exodusii.api.file import ExodusFile
    from exodusii.api.writer import ExodusWriter

    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_quad_mesh(a, [0, 1, 2, 3], [0])

    # Write b with completely different coordinates.
    with ExodusWriter.create(b) as w:
        w.initialize("x", 2, 4, 1, element_blocks=1)
        far_coords = np.array([[100.0, 100.0], [101.0, 100.0], [101.0, 101.0], [100.0, 101.0]])
        w.write_coordinates(far_coords)
        w.define_element_block(10, "quad", [[1, 2, 3, 4]])
        w.write_time(0.0)

    with ExodusFile.open(a) as exo1, ExodusFile.open(b) as exo2, pytest.raises(MeshMatchError):
        build_mesh_map(exo1, exo2, matching_tolerance=1e-6, require_unique_mapping=True)


def test_build_mesh_map_no_match_partial_allowed(tmp_path: Path) -> None:
    """require_unique_mapping=False should not raise, just set unmatched counts."""
    import warnings

    from exodusii.api.file import ExodusFile
    from exodusii.api.writer import ExodusWriter

    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_quad_mesh(a, [0, 1, 2, 3], [0])

    with ExodusWriter.create(b) as w:
        w.initialize("x", 2, 4, 1, element_blocks=1)
        far_coords = np.array([[100.0, 100.0], [101.0, 100.0], [101.0, 101.0], [100.0, 101.0]])
        w.write_coordinates(far_coords)
        w.define_element_block(10, "quad", [[1, 2, 3, 4]])
        w.write_time(0.0)

    with (
        ExodusFile.open(a) as exo1,
        ExodusFile.open(b) as exo2,
        warnings.catch_warnings(record=True),
    ):
        warnings.simplefilter("always")
        mm = build_mesh_map(exo1, exo2, matching_tolerance=1e-6, require_unique_mapping=False)
    assert mm.unmatched_nodes > 0


def test_build_mesh_map_two_element_block(tmp_path: Path) -> None:
    """Two-element mesh with shuffled nodes: mapping recovers correct alignment."""
    from exodusii.api.file import ExodusFile

    a = tmp_path / "a.exo"
    b = tmp_path / "b.exo"
    _write_two_quad_mesh(a, [0, 1, 2, 3, 4, 5])
    _write_two_quad_mesh(b, [5, 2, 4, 1, 3, 0])  # shuffled

    with ExodusFile.open(a) as exo1, ExodusFile.open(b) as exo2:
        mm = build_mesh_map(exo1, exo2, matching_tolerance=1e-8)
        coords1 = np.asarray(exo1.coordinates())
        coords2 = np.asarray(exo2.coordinates())

    np.testing.assert_allclose(coords1, coords2[mm.node_map_inv], atol=1e-12)
    assert mm.unmatched_nodes == 0
    assert mm.unmatched_elems == 0
