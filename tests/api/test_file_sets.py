# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np

from exodusii.api.file import ExodusFile
from exodusii.core.names import AttributeName
from exodusii.core.names import DimensionName
from exodusii.core.names import VariableName
from exodusii.io.netcdf4_backend import NetCDF4Backend


def test_file_reads_all_set_types(tmp_path: Path) -> None:
    path = tmp_path / "sets.exo"
    _write_set_file(path)

    with ExodusFile.open(path) as exo:
        assert exo.node_set_ids().tolist() == [10]
        assert exo.side_set_ids().tolist() == [20]
        assert exo.edge_set_ids().tolist() == [30]
        assert exo.face_set_ids().tolist() == [40]
        assert exo.element_set_ids().tolist() == [50]

        node_set = exo.node_set(10)
        assert node_set.name == "node_set"
        assert np.allclose(node_set.nodes, [1, 2])  # ty: ignore[invalid-argument-type]
        assert np.allclose(node_set.dist_facts, [1.0, 2.0])  # ty: ignore[invalid-argument-type]

        side_set = exo.side_set(20)
        assert side_set.name == "side_set"
        assert np.allclose(side_set.elems, [1])  # ty: ignore[invalid-argument-type]
        assert np.allclose(side_set.sides, [3])  # ty: ignore[invalid-argument-type]
        assert np.allclose(side_set.dist_facts, [3.0])  # ty: ignore[invalid-argument-type]

        edge_set = exo.edge_set(30)
        assert edge_set.name == "edge_set"
        assert np.allclose(edge_set.entries, [1, 2])  # ty: ignore[invalid-argument-type]
        assert np.allclose(edge_set.extra_entries, [1, -1])  # ty: ignore[invalid-argument-type]
        assert np.allclose(edge_set.dist_facts, [4.0, 5.0])  # ty: ignore[invalid-argument-type]

        face_set = exo.face_set(40)
        assert face_set.name == "face_set"
        assert np.allclose(face_set.entries, [1])  # ty: ignore[invalid-argument-type]
        assert np.allclose(face_set.extra_entries, [1])  # ty: ignore[invalid-argument-type]
        assert np.allclose(face_set.dist_facts, [6.0])  # ty: ignore[invalid-argument-type]

        element_set = exo.element_set(50)
        assert element_set.name == "element_set"
        assert np.allclose(element_set.entries, [1])  # ty: ignore[invalid-argument-type]
        assert np.allclose(element_set.dist_facts, [7.0])  # ty: ignore[invalid-argument-type]


def test_file_set_status(tmp_path: Path) -> None:
    path = tmp_path / "sets.exo"
    _write_set_file(path)

    with ExodusFile.open(path) as exo:
        assert exo.set_status("node_set").tolist() == [1]
        assert exo.set_is_active("node_set", 10)
        assert exo.node_set_ids(active_only=True).tolist() == [10]


def _write_set_file(path: Path) -> None:
    with NetCDF4Backend(path, mode="w") as backend:
        backend.set_attribute(AttributeName.TITLE.value, "sets")
        backend.set_attribute(AttributeName.FLOATING_POINT_WORD_SIZE.value, 8)

        backend.create_dimension(DimensionName.TIME.value, None)
        backend.create_dimension(DimensionName.STRING_LENGTH.value, 32)
        backend.create_dimension(DimensionName.NUM_DIMENSIONS.value, 2)
        backend.create_dimension(DimensionName.NUM_NODES.value, 4)
        backend.create_dimension(DimensionName.NUM_ELEMENTS.value, 1)
        backend.create_dimension(DimensionName.NUM_EDGES.value, 2)
        backend.create_dimension(DimensionName.NUM_FACES.value, 1)

        backend.create_dimension(DimensionName.NUM_NODE_SETS.value, 1)
        backend.create_dimension(DimensionName.NUM_SIDE_SETS.value, 1)
        backend.create_dimension(DimensionName.NUM_EDGE_SETS.value, 1)
        backend.create_dimension(DimensionName.NUM_FACE_SETS.value, 1)
        backend.create_dimension(DimensionName.NUM_ELEMENT_SETS.value, 1)

        _write_set_header(
            backend,
            dim_name=DimensionName.NUM_NODE_SETS.value,
            ids_name=VariableName.NODE_SET_IDS.value,
            status_name=VariableName.NODE_SET_STATUS.value,
            names_name=VariableName.NODE_SET_NAMES.value,
            set_id=10,
            name="node_set",
        )
        _write_set_header(
            backend,
            dim_name=DimensionName.NUM_SIDE_SETS.value,
            ids_name=VariableName.SIDE_SET_IDS.value,
            status_name=VariableName.SIDE_SET_STATUS.value,
            names_name=VariableName.SIDE_SET_NAMES.value,
            set_id=20,
            name="side_set",
        )
        _write_set_header(
            backend,
            dim_name=DimensionName.NUM_EDGE_SETS.value,
            ids_name=VariableName.EDGE_SET_IDS.value,
            status_name=VariableName.EDGE_SET_STATUS.value,
            names_name=VariableName.EDGE_SET_NAMES.value,
            set_id=30,
            name="edge_set",
        )
        _write_set_header(
            backend,
            dim_name=DimensionName.NUM_FACE_SETS.value,
            ids_name=VariableName.FACE_SET_IDS.value,
            status_name=VariableName.FACE_SET_STATUS.value,
            names_name=VariableName.FACE_SET_NAMES.value,
            set_id=40,
            name="face_set",
        )
        _write_set_header(
            backend,
            dim_name=DimensionName.NUM_ELEMENT_SETS.value,
            ids_name=VariableName.ELEMENT_SET_IDS.value,
            status_name=VariableName.ELEMENT_SET_STATUS.value,
            names_name=VariableName.ELEMENT_SET_NAMES.value,
            set_id=50,
            name="element_set",
        )

        _write_entries(backend, "num_nod_ns1", "node_ns1", [1, 2], "dist_fact_ns1", [1.0, 2.0])
        _write_entries(
            backend,
            "num_side_ss1",
            "elem_ss1",
            [1],
            "dist_fact_ss1",
            [3.0],
            extra_name="side_ss1",
            extra=[3],
        )
        _write_entries(
            backend,
            "num_edge_es1",
            "edge_es1",
            [1, 2],
            "dist_fact_es1",
            [4.0, 5.0],
            extra_name="ornt_es1",
            extra=[1, -1],
        )
        _write_entries(
            backend,
            "num_face_fs1",
            "face_fs1",
            [1],
            "dist_fact_fs1",
            [6.0],
            extra_name="ornt_fs1",
            extra=[1],
        )
        _write_entries(backend, "num_ele_els1", "elem_els1", [1], "dist_fact_els1", [7.0])


def _write_set_header(
    backend: NetCDF4Backend,
    *,
    dim_name: str,
    ids_name: str,
    status_name: str,
    names_name: str,
    set_id: int,
    name: str,
) -> None:
    from exodusii.core.strings import encode_fixed_width

    backend.create_variable(ids_name, int, (dim_name,))
    backend.create_variable(status_name, int, (dim_name,))
    backend.create_variable(names_name, str, (dim_name, DimensionName.STRING_LENGTH.value))
    backend.write_variable(ids_name, [set_id])
    backend.write_variable(status_name, [1])
    backend.write_variable(names_name, encode_fixed_width([name], width=32))


def _write_entries(
    backend: NetCDF4Backend,
    count_dim: str,
    entry_name: str,
    entries: list[int],
    dist_name: str,
    dist: list[float],
    *,
    extra_name: str | None = None,
    extra: list[int] | None = None,
) -> None:
    backend.create_dimension(count_dim, len(entries))
    backend.create_variable(entry_name, int, (count_dim,))
    backend.create_variable(dist_name, float, (count_dim,))
    backend.write_variable(entry_name, entries)
    backend.write_variable(dist_name, dist)

    if extra_name is not None and extra is not None:
        backend.create_variable(extra_name, int, (count_dim,))
        backend.write_variable(extra_name, extra)
