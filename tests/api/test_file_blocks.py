# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np

from exodusii.api.file import ExodusFile
from exodusii.core.names import AttributeName
from exodusii.core.names import DimensionName
from exodusii.core.names import ExodusNames
from exodusii.core.names import VariableName
from exodusii.io.netcdf4_backend import NetCDF4Backend


def test_file_reads_element_edge_and_face_blocks(tmp_path: Path) -> None:
    path = tmp_path / "blocks.exo"
    _write_block_file(path)

    with ExodusFile.open(path) as exo:
        assert exo.element_block_ids().tolist() == [10]
        assert exo.edge_block_ids().tolist() == [20]
        assert exo.face_block_ids().tolist() == [30]

        elem = exo.element_block(10)
        assert elem.element_type == "QUAD"
        assert elem.count == 1
        assert elem.nodes_per_entity == 4
        assert elem.edges_per_entity == 4
        assert elem.name == "elem_block"

        edge = exo.edge_block(20)
        assert edge.element_type == "EDGE2"
        assert edge.count == 4
        assert edge.nodes_per_entity == 2
        assert edge.name == "edge_block"

        face = exo.face_block(30)
        assert face.element_type == "QUAD"
        assert face.count == 1
        assert face.nodes_per_entity == 4
        assert face.name == "face_block"

        assert np.allclose(exo.element_connectivity(10), [[1, 2, 3, 4]])
        assert np.allclose(exo.edge_connectivity(20), [[1, 2], [2, 3], [3, 4], [4, 1]])
        assert np.allclose(exo.face_connectivity(30), [[1, 2, 3, 4]])

        assert np.allclose(exo.element_edge_connectivity(10), [[1, 2, 3, 4]])


def test_file_block_status(tmp_path: Path) -> None:
    path = tmp_path / "blocks.exo"
    _write_block_file(path)

    with ExodusFile.open(path) as exo:
        assert exo.block_status("element_block").tolist() == [1]
        assert exo.block_is_active("element_block", 10)
        assert exo.element_block_ids(active_only=True).tolist() == [10]


def _write_block_file(path: Path) -> None:
    with NetCDF4Backend(path, mode="w") as backend:
        backend.set_attribute(AttributeName.TITLE.value, "blocks")
        backend.set_attribute(AttributeName.FLOATING_POINT_WORD_SIZE.value, 8)

        backend.create_dimension(DimensionName.TIME.value, None)
        backend.create_dimension(DimensionName.STRING_LENGTH.value, 32)
        backend.create_dimension(DimensionName.NUM_DIMENSIONS.value, 2)
        backend.create_dimension(DimensionName.NUM_NODES.value, 4)
        backend.create_dimension(DimensionName.NUM_ELEMENTS.value, 1)
        backend.create_dimension(DimensionName.NUM_EDGES.value, 4)
        backend.create_dimension(DimensionName.NUM_FACES.value, 1)

        backend.create_dimension(DimensionName.NUM_ELEMENT_BLOCKS.value, 1)
        backend.create_dimension(DimensionName.NUM_EDGE_BLOCKS.value, 1)
        backend.create_dimension(DimensionName.NUM_FACE_BLOCKS.value, 1)

        backend.create_variable(VariableName.TIME.value, float, (DimensionName.TIME.value,))

        backend.create_variable(VariableName.COORD_X.value, float, (DimensionName.NUM_NODES.value,))
        backend.create_variable(VariableName.COORD_Y.value, float, (DimensionName.NUM_NODES.value,))
        backend.write_variable(VariableName.COORD_X.value, [0.0, 1.0, 1.0, 0.0])
        backend.write_variable(VariableName.COORD_Y.value, [0.0, 0.0, 1.0, 1.0])

        _write_block_header(
            backend,
            ids_name=VariableName.ELEMENT_BLOCK_IDS.value,
            status_name=VariableName.ELEMENT_BLOCK_STATUS.value,
            names_name=VariableName.ELEMENT_BLOCK_NAMES.value,
            dim_name=DimensionName.NUM_ELEMENT_BLOCKS.value,
            block_id=10,
            name="elem_block",
        )
        _write_block_header(
            backend,
            ids_name=VariableName.EDGE_BLOCK_IDS.value,
            status_name=VariableName.EDGE_BLOCK_STATUS.value,
            names_name=VariableName.EDGE_BLOCK_NAMES.value,
            dim_name=DimensionName.NUM_EDGE_BLOCKS.value,
            block_id=20,
            name="edge_block",
        )
        _write_block_header(
            backend,
            ids_name=VariableName.FACE_BLOCK_IDS.value,
            status_name=VariableName.FACE_BLOCK_STATUS.value,
            names_name=VariableName.FACE_BLOCK_NAMES.value,
            dim_name=DimensionName.NUM_FACE_BLOCKS.value,
            block_id=30,
            name="face_block",
        )

        backend.create_dimension(ExodusNames.block_count(1), 1)
        backend.create_dimension(ExodusNames.nodes_per_element(1), 4)
        backend.create_dimension(ExodusNames.edges_per_element(1), 4)
        backend.create_variable(
            ExodusNames.element_connectivity(1),
            int,
            (ExodusNames.block_count(1), ExodusNames.nodes_per_element(1)),
        )
        backend.create_variable(
            ExodusNames.element_edge_connectivity(1),
            int,
            (ExodusNames.block_count(1), ExodusNames.edges_per_element(1)),
        )
        backend.set_variable_attribute(
            ExodusNames.element_connectivity(1), AttributeName.ELEMENT_TYPE.value, "QUAD"
        )
        backend.write_variable(ExodusNames.element_connectivity(1), [[1, 2, 3, 4]])
        backend.write_variable(ExodusNames.element_edge_connectivity(1), [[1, 2, 3, 4]])

        backend.create_dimension(ExodusNames.edge_block_count(1), 4)
        backend.create_dimension(ExodusNames.nodes_per_edge(1), 2)
        backend.create_variable(
            ExodusNames.edge_connectivity(1),
            int,
            (ExodusNames.edge_block_count(1), ExodusNames.nodes_per_edge(1)),
        )
        backend.set_variable_attribute(
            ExodusNames.edge_connectivity(1), AttributeName.ELEMENT_TYPE.value, "EDGE2"
        )
        backend.write_variable(ExodusNames.edge_connectivity(1), [[1, 2], [2, 3], [3, 4], [4, 1]])

        backend.create_dimension(ExodusNames.face_block_count(1), 1)
        backend.create_dimension(ExodusNames.nodes_per_face(1), 4)
        backend.create_variable(
            ExodusNames.face_connectivity(1),
            int,
            (ExodusNames.face_block_count(1), ExodusNames.nodes_per_face(1)),
        )
        backend.set_variable_attribute(
            ExodusNames.face_connectivity(1), AttributeName.ELEMENT_TYPE.value, "QUAD"
        )
        backend.write_variable(ExodusNames.face_connectivity(1), [[1, 2, 3, 4]])


def _write_block_header(
    backend: NetCDF4Backend,
    *,
    ids_name: str,
    status_name: str,
    names_name: str,
    dim_name: str,
    block_id: int,
    name: str,
) -> None:
    from exodusii.core.strings import encode_fixed_width

    backend.create_variable(ids_name, int, (dim_name,))
    backend.create_variable(status_name, int, (dim_name,))
    backend.create_variable(names_name, str, (dim_name, DimensionName.STRING_LENGTH.value))
    backend.write_variable(ids_name, [block_id])
    backend.write_variable(status_name, [1])
    backend.write_variable(names_name, encode_fixed_width([name], width=32))
