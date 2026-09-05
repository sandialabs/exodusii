# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pytest

from exodusii import ExodusFile
from exodusii.core.errors import ExodusInvalidEntityError
from exodusii.core.errors import ExodusLookupError
from exodusii.core.names import AttributeName
from exodusii.core.names import DimensionName
from exodusii.core.names import ExodusNames
from exodusii.core.names import VariableName
from exodusii.core.strings import encode_fixed_width
from exodusii.io.netcdf4_backend import NetCDF4Backend


def test_open_basic_file(tmp_path: Path) -> None:
    path = tmp_path / "small.exo"
    _write_small_exodus(path)

    with ExodusFile.open(path) as exo:
        assert exo.path == path
        assert exo.mode == "r"
        assert exo.title == "small mesh"
        assert exo.version == pytest.approx(5.03)
        assert exo.api_version == pytest.approx(5.03)
        assert exo.storage_type == "d"

        assert exo.dimension == 2
        assert exo.node_count == 4
        assert exo.element_count == 1
        assert exo.element_block_count == 1
        assert exo.node_set_count == 1
        assert exo.side_set_count == 1


def test_init_params(tmp_path: Path) -> None:
    path = tmp_path / "small.exo"
    _write_small_exodus(path)

    with ExodusFile.open(path) as exo:
        params = exo.init_params()

    assert params.title == "small mesh"
    assert params.dimension == 2
    assert params.nodes == 4
    assert params.elements == 1
    assert params.element_blocks == 1
    assert params.node_sets == 1
    assert params.side_sets == 1


def test_dimensions_and_variables(tmp_path: Path) -> None:
    path = tmp_path / "small.exo"
    _write_small_exodus(path)

    with ExodusFile.open(path) as exo:
        assert DimensionName.NUM_NODES.value in exo.dimensions()
        assert VariableName.TIME.value in exo.variables()
        assert exo.dimension_size(DimensionName.NUM_NODES.value) == 4
        assert exo.variable(VariableName.TIME.value).shape == (3,)


def test_dimension_size_missing_raises(tmp_path: Path) -> None:
    path = tmp_path / "small.exo"
    _write_small_exodus(path)

    with (
        ExodusFile.open(path) as exo,
        pytest.raises(ExodusLookupError, match="dimension 'missing' not found"),
    ):
        exo.dimension_size("missing")


def test_times(tmp_path: Path) -> None:
    path = tmp_path / "small.exo"
    _write_small_exodus(path)

    with ExodusFile.open(path) as exo:
        assert np.allclose(exo.times(), [0.0, 1.0, 2.0])


def test_coordinate_names_and_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "small.exo"
    _write_small_exodus(path)

    with ExodusFile.open(path) as exo:
        assert exo.coordinate_names().tolist() == ["X", "Y"]
        assert np.allclose(exo.coordinates(), [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def test_displacements_and_displaced_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "small.exo"
    _write_small_exodus(path)

    with ExodusFile.open(path) as exo:
        assert exo.displacement_variable_names() == ("DISPLX", "DISPLY")

        assert np.allclose(
            exo.displacements(time="last"), [[0.2, 0.0], [0.2, 0.0], [0.2, 0.0], [0.2, 0.0]]
        )

        assert np.allclose(
            exo.coordinates(time="last", displaced=True),
            [[0.2, 0.0], [1.2, 0.0], [1.2, 1.0], [0.2, 1.0]],
        )


def test_ids(tmp_path: Path) -> None:
    path = tmp_path / "small.exo"
    _write_small_exodus(path)

    with ExodusFile.open(path) as exo:
        assert exo.ids("node").tolist() == [1, 2, 3, 4]
        assert exo.ids("element").tolist() == [1]
        assert exo.element_block_ids().tolist() == [10]
        assert exo.node_set_ids().tolist() == [100]
        assert exo.side_set_ids().tolist() == [200]


def test_element_block_and_connectivity(tmp_path: Path) -> None:
    path = tmp_path / "small.exo"
    _write_small_exodus(path)

    with ExodusFile.open(path) as exo:
        block = exo.element_block(10)

        assert block.id == 10
        assert block.index == 1
        assert block.element_type == "QUAD"
        assert block.count == 1
        assert block.nodes_per_entity == 4
        assert block.name == "block_10"

        assert np.allclose(exo.element_connectivity(10), [[1, 2, 3, 4]])
        assert np.allclose(exo.element_connectivity(10, zero_based=True), [[0, 1, 2, 3]])


def test_element_block_missing_id_raises(tmp_path: Path) -> None:
    path = tmp_path / "small.exo"
    _write_small_exodus(path)

    with (
        ExodusFile.open(path) as exo,
        pytest.raises(ExodusLookupError, match="element_block ID 999 not found"),
    ):
        exo.element_block(999)


def test_node_set(tmp_path: Path) -> None:
    path = tmp_path / "small.exo"
    _write_small_exodus(path)

    with ExodusFile.open(path) as exo:
        node_set = exo.node_set(100)

        assert node_set.id == 100
        assert node_set.index == 1
        assert node_set.name == "nodeset_100"
        assert np.allclose(node_set.nodes, [1, 4])  # ty: ignore[invalid-argument-type]
        assert np.allclose(node_set.dist_facts, [1.0, 2.0])  # ty: ignore[invalid-argument-type]


def test_side_set(tmp_path: Path) -> None:
    path = tmp_path / "small.exo"
    _write_small_exodus(path)

    with ExodusFile.open(path) as exo:
        side_set = exo.side_set(200)

        assert side_set.id == 200
        assert side_set.index == 1
        assert side_set.name == "sideset_200"
        assert np.allclose(side_set.elems, [1])  # ty: ignore[invalid-argument-type]
        assert np.allclose(side_set.sides, [2])  # ty: ignore[invalid-argument-type]
        assert np.allclose(side_set.dist_facts, [3.0])  # ty: ignore[invalid-argument-type]


def test_variable_names(tmp_path: Path) -> None:
    path = tmp_path / "small.exo"
    _write_small_exodus(path)

    with ExodusFile.open(path) as exo:
        assert exo.variable_names("global") == ("TM_STEP",)
        assert exo.variable_names("node") == ("DISPLX", "DISPLY", "TEMP")
        assert exo.variable_names("element") == ("ENERGY",)


def test_variable_names_rejects_non_variable_entity(tmp_path: Path) -> None:
    path = tmp_path / "small.exo"
    _write_small_exodus(path)

    with (
        ExodusFile.open(path) as exo,
        pytest.raises(ExodusInvalidEntityError, match="not a variable location"),
    ):
        exo.variable_names("element_block")


def test_global_values(tmp_path: Path) -> None:
    path = tmp_path / "small.exo"
    _write_small_exodus(path)

    with ExodusFile.open(path) as exo:
        assert np.allclose(exo.values("TM_STEP", on="global"), [0.0, 1.0, 2.0])
        assert np.allclose(exo.values("TM_STEP", on="global", time="last"), 2.0)
        assert np.allclose(exo.values("tm_step", on="global", time=1), 1.0)


def test_node_values(tmp_path: Path) -> None:
    path = tmp_path / "small.exo"
    _write_small_exodus(path)

    with ExodusFile.open(path) as exo:
        assert np.allclose(exo.values("TEMP", on="node", time="last"), [12.0, 22.0, 32.0, 42.0])
        assert exo.values("TEMP", on="node").shape == (3, 4)


def test_element_values(tmp_path: Path) -> None:
    path = tmp_path / "small.exo"
    _write_small_exodus(path)

    with ExodusFile.open(path) as exo:
        assert np.allclose(exo.values("ENERGY", on="element", block=10), [[0.5], [1.5], [2.5]])
        assert np.allclose(exo.values("ENERGY", on="element", block=10, time="last"), [2.5])
        assert np.allclose(exo.values("ENERGY", on="element", time="last"), [2.5])


def test_values_rejects_missing_variable(tmp_path: Path) -> None:
    path = tmp_path / "small.exo"
    _write_small_exodus(path)

    with (
        ExodusFile.open(path) as exo,
        pytest.raises(ExodusLookupError, match="variable 'missing' not found"),
    ):
        exo.values("missing", on="node")


def test_values_rejects_missing_side_set_variable(tmp_path: Path) -> None:
    path = tmp_path / "small.exo"
    _write_small_exodus(path)

    with ExodusFile.open(path) as exo, pytest.raises(ExodusLookupError):
        exo.values("foo", on="side_set")


def _write_small_exodus(path: Path) -> None:
    with NetCDF4Backend(path, mode="w") as backend:
        backend.set_attribute(AttributeName.TITLE.value, "small mesh")
        backend.set_attribute(AttributeName.VERSION.value, 5.03)
        backend.set_attribute(AttributeName.API_VERSION.value, 5.03)
        backend.set_attribute(AttributeName.FLOATING_POINT_WORD_SIZE.value, 8)

        backend.create_dimension(DimensionName.TIME.value, None)
        backend.create_dimension(DimensionName.STRING_LENGTH.value, 32)
        backend.create_dimension(DimensionName.NAME_LENGTH.value, 32)
        backend.create_dimension(DimensionName.LINE_LENGTH.value, 80)
        backend.create_dimension(DimensionName.FOUR.value, 4)

        backend.create_dimension(DimensionName.NUM_DIMENSIONS.value, 2)
        backend.create_dimension(DimensionName.NUM_NODES.value, 4)
        backend.create_dimension(DimensionName.NUM_ELEMENTS.value, 1)
        backend.create_dimension(DimensionName.NUM_ELEMENT_BLOCKS.value, 1)
        backend.create_dimension(DimensionName.NUM_NODE_SETS.value, 1)
        backend.create_dimension(DimensionName.NUM_SIDE_SETS.value, 1)

        backend.create_variable(VariableName.TIME.value, float, (DimensionName.TIME.value,))
        backend.write_variable(VariableName.TIME.value, 0.0, 0)
        backend.write_variable(VariableName.TIME.value, 1.0, 1)
        backend.write_variable(VariableName.TIME.value, 2.0, 2)

        backend.create_variable(
            VariableName.COORDINATE_NAMES.value,
            str,
            (DimensionName.NUM_DIMENSIONS.value, DimensionName.STRING_LENGTH.value),
        )
        backend.write_variable(
            VariableName.COORDINATE_NAMES.value, encode_fixed_width(["X", "Y"], width=32)
        )

        backend.create_variable(VariableName.COORD_X.value, float, (DimensionName.NUM_NODES.value,))
        backend.create_variable(VariableName.COORD_Y.value, float, (DimensionName.NUM_NODES.value,))
        backend.write_variable(VariableName.COORD_X.value, [0.0, 1.0, 1.0, 0.0])
        backend.write_variable(VariableName.COORD_Y.value, [0.0, 0.0, 1.0, 1.0])

        backend.create_variable(
            VariableName.ELEMENT_BLOCK_IDS.value, int, (DimensionName.NUM_ELEMENT_BLOCKS.value,)
        )
        backend.create_variable(
            VariableName.ELEMENT_BLOCK_STATUS.value, int, (DimensionName.NUM_ELEMENT_BLOCKS.value,)
        )
        backend.create_variable(
            VariableName.ELEMENT_BLOCK_NAMES.value,
            str,
            (DimensionName.NUM_ELEMENT_BLOCKS.value, DimensionName.STRING_LENGTH.value),
        )
        backend.write_variable(VariableName.ELEMENT_BLOCK_IDS.value, [10])
        backend.write_variable(VariableName.ELEMENT_BLOCK_STATUS.value, [1])
        backend.write_variable(
            VariableName.ELEMENT_BLOCK_NAMES.value, encode_fixed_width(["block_10"], width=32)
        )

        backend.create_dimension(ExodusNames.block_count(1), 1)
        backend.create_dimension(ExodusNames.nodes_per_element(1), 4)
        backend.create_variable(
            ExodusNames.element_connectivity(1),
            int,
            (ExodusNames.block_count(1), ExodusNames.nodes_per_element(1)),
        )
        backend.set_variable_attribute(
            ExodusNames.element_connectivity(1), AttributeName.ELEMENT_TYPE.value, "QUAD"
        )
        backend.write_variable(ExodusNames.element_connectivity(1), [[1, 2, 3, 4]])

        backend.create_variable(
            VariableName.NODE_SET_IDS.value, int, (DimensionName.NUM_NODE_SETS.value,)
        )
        backend.create_variable(
            VariableName.NODE_SET_STATUS.value, int, (DimensionName.NUM_NODE_SETS.value,)
        )
        backend.create_variable(
            VariableName.NODE_SET_NAMES.value,
            str,
            (DimensionName.NUM_NODE_SETS.value, DimensionName.STRING_LENGTH.value),
        )
        backend.write_variable(VariableName.NODE_SET_IDS.value, [100])
        backend.write_variable(VariableName.NODE_SET_STATUS.value, [1])
        backend.write_variable(
            VariableName.NODE_SET_NAMES.value, encode_fixed_width(["nodeset_100"], width=32)
        )
        backend.create_dimension(ExodusNames.node_set_count(1), 2)
        backend.create_variable(
            ExodusNames.node_set_nodes(1), int, (ExodusNames.node_set_count(1),)
        )
        backend.create_variable(
            ExodusNames.node_set_distribution_factors(1), float, (ExodusNames.node_set_count(1),)
        )
        backend.write_variable(ExodusNames.node_set_nodes(1), [1, 4])
        backend.write_variable(ExodusNames.node_set_distribution_factors(1), [1.0, 2.0])

        backend.create_variable(
            VariableName.SIDE_SET_IDS.value, int, (DimensionName.NUM_SIDE_SETS.value,)
        )
        backend.create_variable(
            VariableName.SIDE_SET_STATUS.value, int, (DimensionName.NUM_SIDE_SETS.value,)
        )
        backend.create_variable(
            VariableName.SIDE_SET_NAMES.value,
            str,
            (DimensionName.NUM_SIDE_SETS.value, DimensionName.STRING_LENGTH.value),
        )
        backend.write_variable(VariableName.SIDE_SET_IDS.value, [200])
        backend.write_variable(VariableName.SIDE_SET_STATUS.value, [1])
        backend.write_variable(
            VariableName.SIDE_SET_NAMES.value, encode_fixed_width(["sideset_200"], width=32)
        )
        backend.create_dimension(ExodusNames.side_set_count(1), 1)
        backend.create_variable(
            ExodusNames.side_set_elements(1), int, (ExodusNames.side_set_count(1),)
        )
        backend.create_variable(
            ExodusNames.side_set_sides(1), int, (ExodusNames.side_set_count(1),)
        )
        backend.create_variable(
            ExodusNames.side_set_distribution_factors(1), float, (ExodusNames.side_set_count(1),)
        )
        backend.write_variable(ExodusNames.side_set_elements(1), [1])
        backend.write_variable(ExodusNames.side_set_sides(1), [2])
        backend.write_variable(ExodusNames.side_set_distribution_factors(1), [3.0])

        backend.create_dimension(DimensionName.NUM_GLOBAL_VARIABLES.value, 1)
        backend.create_variable(
            VariableName.GLOBAL_VARIABLE_NAMES.value,
            str,
            (DimensionName.NUM_GLOBAL_VARIABLES.value, DimensionName.STRING_LENGTH.value),
        )
        backend.create_variable(
            VariableName.GLOBAL_VARIABLE_VALUES.value,
            float,
            (DimensionName.TIME.value, DimensionName.NUM_GLOBAL_VARIABLES.value),
        )
        backend.write_variable(
            VariableName.GLOBAL_VARIABLE_NAMES.value, encode_fixed_width(["TM_STEP"], width=32)
        )
        backend.write_variable(VariableName.GLOBAL_VARIABLE_VALUES.value, [0.0], 0)
        backend.write_variable(VariableName.GLOBAL_VARIABLE_VALUES.value, [1.0], 1)
        backend.write_variable(VariableName.GLOBAL_VARIABLE_VALUES.value, [2.0], 2)

        backend.create_dimension(DimensionName.NUM_NODE_VARIABLES.value, 3)
        backend.create_variable(
            VariableName.NODE_VARIABLE_NAMES.value,
            str,
            (DimensionName.NUM_NODE_VARIABLES.value, DimensionName.STRING_LENGTH.value),
        )
        backend.write_variable(
            VariableName.NODE_VARIABLE_NAMES.value,
            encode_fixed_width(["DISPLX", "DISPLY", "TEMP"], width=32),
        )
        for index in range(1, 4):
            backend.create_variable(
                ExodusNames.node_variable(index),
                float,
                (DimensionName.TIME.value, DimensionName.NUM_NODES.value),
            )

        backend.write_variable(ExodusNames.node_variable(1), [0.0, 0.0, 0.0, 0.0], 0)
        backend.write_variable(ExodusNames.node_variable(1), [0.1, 0.1, 0.1, 0.1], 1)
        backend.write_variable(ExodusNames.node_variable(1), [0.2, 0.2, 0.2, 0.2], 2)

        backend.write_variable(ExodusNames.node_variable(2), [0.0, 0.0, 0.0, 0.0], 0)
        backend.write_variable(ExodusNames.node_variable(2), [0.0, 0.0, 0.0, 0.0], 1)
        backend.write_variable(ExodusNames.node_variable(2), [0.0, 0.0, 0.0, 0.0], 2)

        backend.write_variable(ExodusNames.node_variable(3), [10.0, 20.0, 30.0, 40.0], 0)
        backend.write_variable(ExodusNames.node_variable(3), [11.0, 21.0, 31.0, 41.0], 1)
        backend.write_variable(ExodusNames.node_variable(3), [12.0, 22.0, 32.0, 42.0], 2)

        backend.create_dimension(DimensionName.NUM_ELEMENT_VARIABLES.value, 1)
        backend.create_variable(
            VariableName.ELEMENT_VARIABLE_NAMES.value,
            str,
            (DimensionName.NUM_ELEMENT_VARIABLES.value, DimensionName.STRING_LENGTH.value),
        )
        backend.write_variable(
            VariableName.ELEMENT_VARIABLE_NAMES.value, encode_fixed_width(["ENERGY"], width=32)
        )
        backend.create_variable(
            ExodusNames.element_variable(1, 1),
            float,
            (DimensionName.TIME.value, ExodusNames.block_count(1)),
        )
        backend.write_variable(ExodusNames.element_variable(1, 1), [0.5], 0)
        backend.write_variable(ExodusNames.element_variable(1, 1), [1.5], 1)
        backend.write_variable(ExodusNames.element_variable(1, 1), [2.5], 2)
