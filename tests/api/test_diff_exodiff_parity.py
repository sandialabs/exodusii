# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""SEACAS exodiff parity for :func:`exodusii.api.diff.diff` and the CLI.

Regression tests for cases where the Python diff reported "Files are the
same" while SEACAS exodiff reports a difference (or an incompatible mesh):

* differing time values / numbers of time steps,
* differing connectivity, element type, or node/element ID maps,
* NaN at the same location in both files,
* command files: unlisted categories and category-scoped settings,

plus inverted maps in ``--match-coordinates`` when elements or free nodes are
permuted.

exodiff exit status: 0 same, 1 error / incompatible mesh, 2 different.
"""

from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from exodusii.api.diff import DiffOptions
from exodusii.api.diff import TimeSelection
from exodusii.api.diff import diff
from exodusii.api.file import ExodusFile
from exodusii.api.writer import ExodusWriter
from exodusii.cli.exodiff import main
from exodusii.mesh.matching import build_mesh_map

SAME, ERROR, DIFFERENT = 0, 1, 2

COORDS = np.asarray([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=float)
CONN = ((1, 2, 5, 4), (2, 3, 6, 5))


def _mesh(
    path: Path,
    *,
    times=(0.0, 1.0),
    conn=CONN,
    etype="quad4",
    coords=COORDS,
    node_ids=None,
    elem_ids=None,
    temp=None,
    globals_=None,
) -> Path:
    with ExodusWriter.create(path) as w:
        w.initialize("t", 2, node_count=len(coords), element_count=len(conn), element_blocks=1)
        w.write_coordinates(coords)
        if node_ids is not None:
            w.write_node_id_map(node_ids)
        if elem_ids is not None:
            w.write_element_id_map(elem_ids)
        w.define_element_block(10, etype, [list(c) for c in conn])
        if globals_ is not None:
            w.define_global_variables(list(globals_))
        w.define_node_variables(["T"])
        for t in times:
            w.write_time(t)
            if globals_ is not None:
                w.write_global_values(list(globals_.values()))
            w.write_node_values(
                "T", temp if temp is not None else np.arange(len(coords), dtype=float)
            )
    return path


def _cli(*args: object) -> tuple[int, str]:
    stream = StringIO()
    status = main([str(a) for a in args], file=stream)
    return status, stream.getvalue()


# ---------------------------------------------------------------------------
# Time steps
# ---------------------------------------------------------------------------


def test_differing_time_values_are_a_difference(tmp_path: Path) -> None:
    a = _mesh(tmp_path / "a.exo", times=(0.0, 1.0))
    b = _mesh(tmp_path / "b.exo", times=(0.0, 1.5))

    result = diff(a, b)
    assert not result.same
    assert any("time value differs at step 2" in d for d in result.differences)

    status, text = _cli(a, b)
    assert status == DIFFERENT
    assert "DIFFERENCE: time value differs" in text


def test_differing_step_count_is_a_difference(tmp_path: Path) -> None:
    a = _mesh(tmp_path / "a.exo", times=(0.0, 1.0))
    c = _mesh(tmp_path / "c.exo", times=(0.0, 1.0, 2.0))

    result = diff(a, c)
    assert not result.same
    assert any("number of time steps differs" in d for d in result.differences)
    assert _cli(a, c)[0] == DIFFERENT

    # exodiff: an explicit stop step, or an offset with n1 - offset == n2,
    # accounts for the difference.
    assert diff(a, c, DiffOptions(time_selection=TimeSelection(stop=2))).same
    assert diff(
        c, a, DiffOptions(time_selection=TimeSelection(time_step_offset=1))
    ).differences == [
        "time value differs at step 2: 1.000000e+00 != 0.000000e+00",
        "time value differs at step 3: 2.000000e+00 != 1.000000e+00",
    ]


# ---------------------------------------------------------------------------
# Mesh: connectivity, element type, ID maps
# ---------------------------------------------------------------------------


def test_connectivity_difference_is_a_mesh_error(tmp_path: Path) -> None:
    a = _mesh(tmp_path / "a.exo")
    b = _mesh(tmp_path / "b.exo", conn=((1, 2, 5, 4), (2, 3, 6, 4)))

    result = diff(a, b)
    assert not result.same
    assert any("connectivity differs" in e for e in result.errors)
    assert _cli(a, b)[0] == ERROR


@pytest.mark.parametrize(
    ("type1", "type2", "compatible"),
    [
        ("quad4", "shell4", False),
        ("quad4", "QUAD", True),  # close_compare: one name without a node count
        ("quad4", "quad9", False),
        ("HEX8", "hex", True),
    ],
)
def test_element_type_must_agree(tmp_path: Path, type1: str, type2: str, compatible: bool) -> None:
    a = _mesh(tmp_path / "a.exo", etype=type1)
    b = _mesh(tmp_path / "b.exo", etype=type2)

    result = diff(a, b)
    assert result.same is compatible
    if not compatible:
        assert any("element types differ" in e for e in result.errors)


def test_unmatched_node_ids_are_an_error(tmp_path: Path) -> None:
    a = _mesh(tmp_path / "a.exo")
    b = _mesh(tmp_path / "b.exo", node_ids=[101, 102, 103, 104, 105, 106])

    result = diff(a, b)
    assert not result.same
    assert any("unable to match node IDs" in e for e in result.errors)
    assert _cli(a, b)[0] == ERROR

    # With -match_file_order the maps are compared, not used for matching.
    result = diff(a, b, DiffOptions(match_ids=False))
    assert not result.errors
    assert result.differences == ["node ID maps differ (matching by file order)"]
    assert _cli("--match-file-order", a, b)[0] == DIFFERENT


def test_permuted_nodes_with_consistent_ids_are_the_same(tmp_path: Path) -> None:
    # File 2 stores the same nodes in reverse order, with ID maps saying so.
    perm = np.asarray([5, 4, 3, 2, 1, 0], dtype=np.int64)
    inverse = np.argsort(perm)
    conn2 = tuple(tuple(int(inverse[n - 1]) + 1 for n in element) for element in CONN)
    temp = COORDS[:, 0] * 10 + COORDS[:, 1]

    a = _mesh(tmp_path / "a.exo", node_ids=np.arange(1, 7), temp=temp)
    b = _mesh(
        tmp_path / "b.exo", coords=COORDS[perm], conn=conn2, node_ids=perm + 1, temp=temp[perm]
    )

    result = diff(a, b)
    assert result.same, (result.errors, result.differences, result.variable_diffs)

    # Matching by file order instead sees different coordinates.
    assert not diff(a, b, DiffOptions(match_ids=False)).same


def test_permuted_elements_with_consistent_ids_are_the_same(tmp_path: Path) -> None:
    a = _mesh(tmp_path / "a.exo", elem_ids=[1, 2])
    b = _mesh(tmp_path / "b.exo", conn=(CONN[1], CONN[0]), elem_ids=[2, 1])
    assert diff(a, b).same

    # A genuinely different element (same IDs, other nodes) is still caught.
    c = _mesh(tmp_path / "c.exo", conn=((2, 3, 6, 5), (1, 2, 5, 4)), elem_ids=[1, 2])
    assert any("connectivity differs" in e for e in diff(a, c).errors)


def test_block_order_does_not_matter(tmp_path: Path) -> None:
    paths = []
    for name, order in (("a", (10, 20)), ("b", (20, 10))):
        path = tmp_path / f"{name}.exo"
        with ExodusWriter.create(path) as w:
            w.initialize("t", 2, node_count=6, element_count=2, element_blocks=2)
            w.write_coordinates(COORDS)
            for block_id in order:
                w.define_element_block(block_id, "quad4", [list(CONN[0 if block_id == 10 else 1])])
        paths.append(path)

    result = diff(paths[0], paths[1])
    assert not any("block ids differ" in e for e in result.errors)


# ---------------------------------------------------------------------------
# NaN
# ---------------------------------------------------------------------------


def test_nan_in_both_files_is_a_difference(tmp_path: Path) -> None:
    temp = np.arange(6.0)
    temp[2] = np.nan
    a = _mesh(tmp_path / "a.exo", temp=temp)
    b = _mesh(tmp_path / "b.exo", temp=temp)

    result = diff(a, b)
    assert not result.same
    assert _cli(a, b)[0] == DIFFERENT

    assert diff(a, b, DiffOptions(ignore_nans=True)).same
    assert _cli("--ignore-nans", a, b)[0] == SAME

    cmd = tmp_path / "cmd"
    cmd.write_text("IGNORE NANS\nNODAL VARIABLES (all)\n")
    assert _cli("-f", cmd, a, b)[0] == SAME


def test_ignore_nans_still_catches_finite_differences(tmp_path: Path) -> None:
    temp1 = np.arange(6.0)
    temp1[2] = np.nan
    temp2 = temp1.copy()
    temp2[4] += 1.0
    a = _mesh(tmp_path / "a.exo", temp=temp1)
    b = _mesh(tmp_path / "b.exo", temp=temp2)

    result = diff(a, b, DiffOptions(ignore_nans=True))
    assert not result.same
    (vd,) = [v for v in result.variable_diffs if v.name == "T"]
    assert vd.entry_index == 4


# ---------------------------------------------------------------------------
# Command files
# ---------------------------------------------------------------------------


def _globals_and_nodal(tmp_path: Path, name: str, value: float) -> Path:
    return _mesh(tmp_path / name, times=(0.0,), globals_={"T": value})


def test_command_file_unlisted_category_is_not_compared(tmp_path: Path) -> None:
    a = _globals_and_nodal(tmp_path, "a.exo", 1.0)
    b = _globals_and_nodal(tmp_path, "b.exo", 2.0)
    cmd = tmp_path / "cmd"

    cmd.write_text("NODAL VARIABLES\n\tT\n")
    assert _cli("-f", cmd, a, b)[0] == SAME

    cmd.write_text("GLOBAL VARIABLES\n\tT\n")
    assert _cli("-f", cmd, a, b)[0] == DIFFERENT

    # A category header without names (and without "(all)") selects nothing.
    cmd.write_text("GLOBAL VARIABLES relative 1e-6\n")
    assert _cli("-f", cmd, a, b)[0] == SAME

    # Without a command file every category is compared.
    assert _cli(a, b)[0] == DIFFERENT


def test_command_file_exclusion_is_scoped_to_its_category(tmp_path: Path) -> None:
    a = _globals_and_nodal(tmp_path, "a.exo", 1.0)
    b = _globals_and_nodal(tmp_path, "b.exo", 2.0)
    cmd = tmp_path / "cmd"

    # "!T" under NODAL must not drop the global variable T.
    cmd.write_text("GLOBAL VARIABLES (all)\nNODAL VARIABLES\n\t!T\n")
    assert _cli("-f", cmd, a, b)[0] == DIFFERENT

    cmd.write_text("GLOBAL VARIABLES\n\t!T\nNODAL VARIABLES (all)\n")
    assert _cli("-f", cmd, a, b)[0] == SAME


def test_command_file_tolerance_is_scoped_to_its_category(tmp_path: Path) -> None:
    a = _globals_and_nodal(tmp_path, "a.exo", 1.0)
    b = _globals_and_nodal(tmp_path, "b.exo", 2.0)
    cmd = tmp_path / "cmd"

    # The tight nodal override for T must not replace the loose global one.
    cmd.write_text("GLOBAL VARIABLES\n\tT absolute 100\nNODAL VARIABLES\n\tT relative 1e-12\n")
    assert _cli("-f", cmd, a, b)[0] == SAME

    cmd.write_text("GLOBAL VARIABLES\n\tT relative 1e-12\nNODAL VARIABLES\n\tT absolute 100\n")
    assert _cli("-f", cmd, a, b)[0] == DIFFERENT


# ---------------------------------------------------------------------------
# Coordinate matching
# ---------------------------------------------------------------------------

STRIP = np.asarray([[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1]], dtype=float)
STRIP_CONN = [[1, 2, 6, 5], [2, 3, 7, 6], [3, 4, 8, 7]]


def _strip(path: Path, conn, elem_values, coords=STRIP, temp=None) -> Path:
    with ExodusWriter.create(path) as w:
        w.initialize("t", 2, node_count=len(coords), element_count=len(conn), element_blocks=1)
        w.write_coordinates(coords)
        w.define_element_block(10, "quad4", conn)
        w.define_node_variables(["T"])
        w.define_element_variables(["E"])
        w.write_time(0.0)
        w.write_node_values("T", coords[:, 0] if temp is None else temp)
        w.write_element_values("E", elem_values, block_id=10)
    return path


def test_coordinate_matching_element_3_cycle(tmp_path: Path) -> None:
    perm = [1, 2, 0]  # file-2 element j is file-1 element perm[j]
    values = [10.0, 20.0, 30.0]
    a = _strip(tmp_path / "a.exo", STRIP_CONN, values)
    b = _strip(tmp_path / "b.exo", [STRIP_CONN[i] for i in perm], [values[i] for i in perm])

    with ExodusFile.open(a) as exo1, ExodusFile.open(b) as exo2:
        mesh_map = build_mesh_map(exo1, exo2)
    assert mesh_map.elem_map.tolist() == perm

    result = diff(a, b, DiffOptions(coordinate_matching=True))
    assert result.same, (result.errors, result.variable_diffs)


def test_coordinate_matching_free_node_3_cycle(tmp_path: Path) -> None:
    free1 = np.asarray([[10, 10], [11, 10], [12, 10]], dtype=float)
    free2 = free1[[1, 2, 0]]
    coords1 = np.vstack([STRIP, free1])
    coords2 = np.vstack([STRIP, free2])
    a = _strip(tmp_path / "a.exo", STRIP_CONN, [1.0, 2.0, 3.0], coords=coords1)
    b = _strip(tmp_path / "b.exo", STRIP_CONN, [1.0, 2.0, 3.0], coords=coords2)

    with ExodusFile.open(a) as exo1, ExodusFile.open(b) as exo2:
        mesh_map = build_mesh_map(exo1, exo2)
    assert mesh_map.node_map[8:].tolist() == [9, 10, 8]

    result = diff(a, b, DiffOptions(coordinate_matching=True))
    assert result.same, (result.errors, result.variable_diffs)
