# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pytest

import exodusii
from exodusii.api.file import ExodusFile
from exodusii.api.parallel import ParallelExodusFile
from exodusii.api.writer import ExodusWriter
from exodusii.compat.legacy_parallel import ParallelExodusIIFile
from exodusii.core.errors import ExodusConsistencyError


def test_parallel_file_basic_properties(tmp_path: Path) -> None:
    part0, part1 = _write_parts(tmp_path)

    with ParallelExodusFile.open(part0, part1) as exo:
        assert exo.title == "part mesh"
        assert exo.dimension == 2
        assert exo.node_count == 8
        assert exo.element_count == 2
        assert exo.element_block_count == 1
        assert exo.node_set_count == 1
        assert exo.side_set_count == 1
        assert np.allclose(exo.times(), [0.0, 1.0])
        assert exo.coordinate_names().tolist() == ["X", "Y"]


def test_parallel_coordinates_and_connectivity(tmp_path: Path) -> None:
    part0, part1 = _write_parts(tmp_path)

    with ParallelExodusFile.open(part0, part1) as exo:
        assert np.allclose(exo.coordinates(), np.vstack([_coords0(), _coords1()]))
        assert exo.element_block_ids().tolist() == [10]
        assert np.allclose(exo.element_connectivity(10), [[1, 2, 3, 4], [5, 6, 7, 8]])
        assert np.allclose(
            exo.element_connectivity(10, zero_based=True), [[0, 1, 2, 3], [4, 5, 6, 7]]
        )


def test_parallel_sets(tmp_path: Path) -> None:
    part0, part1 = _write_parts(tmp_path)

    with ParallelExodusFile.open(part0, part1) as exo:
        node_set = exo.node_set(100)
        assert node_set.name == "nodeset_100"
        assert np.allclose(node_set.nodes, [1, 4, 5, 8])  # ty: ignore[invalid-argument-type]
        assert np.allclose(node_set.dist_facts, [1.0, 2.0, 3.0, 4.0])  # ty: ignore[invalid-argument-type]

        side_set = exo.side_set(200)
        assert side_set.name == "sideset_200"
        assert np.allclose(side_set.elems, [1, 2])  # ty: ignore[invalid-argument-type]
        assert np.allclose(side_set.sides, [2, 4])  # ty: ignore[invalid-argument-type]
        assert np.allclose(side_set.dist_facts, [5.0, 6.0])  # ty: ignore[invalid-argument-type]


def test_parallel_values(tmp_path: Path) -> None:
    part0, part1 = _write_parts(tmp_path)

    with ParallelExodusFile.open(part0, part1) as exo:
        assert exo.variable_names("global") == ("TM_STEP",)
        assert exo.variable_names("node") == ("TEMP",)
        assert exo.variable_names("element") == ("ENERGY",)

        assert np.allclose(exo.values("TM_STEP", on="global"), [0.0, 1.0])
        assert np.allclose(
            exo.values("TEMP", on="node", time="last"),
            [11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0, 81.0],
        )
        assert np.allclose(exo.values("ENERGY", on="element", block=10, time="last"), [1.5, 2.5])


def test_parallel_write_serial_file(tmp_path: Path) -> None:
    part0, part1 = _write_parts(tmp_path)
    joined = tmp_path / "joined.exo"

    with ParallelExodusFile.open(part0, part1) as exo:
        result = exo.write(joined)

    assert result == str(joined)

    with ExodusFile.open(joined) as exo:
        assert exo.node_count == 8
        assert exo.element_count == 2
        assert np.allclose(exo.element_connectivity(10), [[1, 2, 3, 4], [5, 6, 7, 8]])
        assert np.allclose(
            exo.values("TEMP", on="node", time="last"),
            [11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0, 81.0],
        )
        assert np.allclose(exo.values("ENERGY", on="element", block=10, time="last"), [1.5, 2.5])


def test_legacy_file_factory_opens_parallel(tmp_path: Path) -> None:
    part0, part1 = _write_parts(tmp_path)

    with exodusii.File(part0, part1) as exo:
        assert isinstance(exo, ParallelExodusIIFile)
        assert isinstance(exo._parallel, ParallelExodusFile)
        assert exo.node_count == 8
        assert exo.element_count == 2


def test_parallel_rejects_inconsistent_times(tmp_path: Path) -> None:
    part0 = tmp_path / "part0.exo"
    part1 = tmp_path / "part1.exo"
    _write_part(part0, _coords0(), [10.0, 20.0, 30.0, 40.0], [11.0, 21.0, 31.0, 41.0], [0.5, 1.5])
    _write_part(
        part1,
        _coords1(),
        [50.0, 60.0, 70.0, 80.0],
        [51.0, 61.0, 71.0, 81.0],
        [1.5, 2.5],
        times=[0.0, 2.0],
    )

    with pytest.raises(ExodusConsistencyError, match="time values"):
        ParallelExodusFile.open(part0, part1)


def _write_parts(tmp_path: Path) -> tuple[Path, Path]:
    part0 = tmp_path / "part0.exo"
    part1 = tmp_path / "part1.exo"

    _write_part(
        part0,
        _coords0(),
        [10.0, 20.0, 30.0, 40.0],
        [11.0, 21.0, 31.0, 41.0],
        [0.5, 1.5],
        node_offset=0,
        element_offset=0,
        node_set_nodes=[1, 4],
        node_set_factors=[1.0, 2.0],
        side_set_sides=[2],
        side_set_factors=[5.0],
    )
    _write_part(
        part1,
        _coords1(),
        [50.0, 60.0, 70.0, 80.0],
        [51.0, 61.0, 71.0, 81.0],
        [1.5, 2.5],
        node_offset=4,
        element_offset=1,
        node_set_nodes=[1, 4],
        node_set_factors=[3.0, 4.0],
        side_set_sides=[4],
        side_set_factors=[6.0],
    )

    return part0, part1


def _write_part(
    path: Path,
    coords: np.ndarray,
    temp0: list[float],
    temp1: list[float],
    energy: list[float],
    *,
    times: list[float] | None = None,
    node_offset: int = 0,
    element_offset: int = 0,
    node_set_nodes: list[int] | None = None,
    node_set_factors: list[float] | None = None,
    side_set_sides: list[int] | None = None,
    side_set_factors: list[float] | None = None,
) -> None:
    times = times or [0.0, 1.0]
    node_set_nodes = node_set_nodes or [1, 4]
    node_set_factors = node_set_factors or [1.0, 2.0]
    side_set_sides = side_set_sides or [2]
    side_set_factors = side_set_factors or [5.0]
    n_nodes = coords.shape[0]

    with ExodusWriter.create(path) as writer:
        writer.initialize("part mesh", 2, n_nodes, 1, element_blocks=1, node_sets=1, side_sets=1)
        writer.write_coordinates(coords)
        # Write id maps so parallel reader does not fall back to sequential numbering.
        writer.write_node_id_map(np.arange(node_offset + 1, node_offset + n_nodes + 1))
        writer.write_element_id_map(np.arange(element_offset + 1, element_offset + 2))
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]], name="block_10")
        writer.define_node_set(
            100, node_set_nodes, distribution_factors=node_set_factors, name="nodeset_100"
        )
        writer.define_side_set(
            200, [1], side_set_sides, distribution_factors=side_set_factors, name="sideset_200"
        )

        writer.define_global_variables(["TM_STEP"])
        writer.define_node_variables(["TEMP"])
        writer.define_element_variables(["ENERGY"])

        writer.write_time(times[0])
        writer.write_global_values([times[0]])
        writer.write_node_values("TEMP", temp0)
        writer.write_element_values("ENERGY", [energy[0]], block_id=10)

        writer.write_time(times[1])
        writer.write_global_values([times[1]])
        writer.write_node_values("TEMP", temp1)
        writer.write_element_values("ENERGY", [energy[1]], block_id=10)


def _coords0() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)


def _coords1() -> np.ndarray:
    return np.asarray([[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]], dtype=float)


def test_parallel_respects_global_node_and_element_ids(tmp_path: Path) -> None:
    part0 = tmp_path / "mapped0.exo"
    part1 = tmp_path / "mapped1.exo"

    _write_mapped_part(
        part0,
        coords=np.asarray(
            [
                [10.0, 0.0],  # gid 10
                [30.0, 0.0],  # gid 30
                [30.0, 1.0],  # gid 30 duplicate-ish not used as duplicate here
                [10.0, 1.0],  # gid 40
            ],
            dtype=float,
        ),
        node_map=[10, 30, 50, 70],
        element_map=[100],
        temp=[10.0, 30.0, 50.0, 70.0],
        energy=[100.0],
    )
    _write_mapped_part(
        part1,
        coords=np.asarray([[20.0, 0.0], [40.0, 0.0], [40.0, 1.0], [20.0, 1.0]], dtype=float),
        node_map=[20, 40, 60, 80],
        element_map=[200],
        temp=[20.0, 40.0, 60.0, 80.0],
        energy=[200.0],
    )

    with ParallelExodusFile.open(part0, part1) as exo:
        # Logical ordering is sorted global node IDs:
        # 10, 20, 30, 40, 50, 60, 70, 80
        assert exo.ids("node").tolist() == [10, 20, 30, 40, 50, 60, 70, 80]
        assert exo.ids("element").tolist() == [100, 200]

        assert np.allclose(
            exo.values("TEMP", on="node", time=0), [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        )

        assert np.allclose(exo.values("ENERGY", on="element", block=10, time=0), [100.0, 200.0])


def _write_mapped_part(
    path: Path,
    *,
    coords: np.ndarray,
    node_map: list[int],
    element_map: list[int],
    temp: list[float],
    energy: list[float],
) -> None:
    from exodusii.core.names import VariableName

    with ExodusWriter.create(path) as writer:
        writer.initialize("mapped", 2, 4, 1, element_blocks=1, node_sets=1, side_sets=1)
        writer.write_coordinates(coords)
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]], name="block_10")
        writer.define_node_variables(["TEMP"])
        writer.define_element_variables(["ENERGY"])
        writer.define_node_set(100, [1, 4], name="nodeset_100")
        writer.define_side_set(200, [1], [2], name="sideset_200")

        # Create explicit global ID maps after initialize.
        writer.backend.create_variable(VariableName.NODE_ID_MAP.value, int, ("num_nodes",))
        writer.backend.write_variable(VariableName.NODE_ID_MAP.value, node_map)
        writer.backend.create_variable(VariableName.ELEMENT_ID_MAP.value, int, ("num_elem",))
        writer.backend.write_variable(VariableName.ELEMENT_ID_MAP.value, element_map)

        writer.write_time(0.0)
        writer.write_node_values("TEMP", temp)
        writer.write_element_values("ENERGY", energy, block_id=10)


def test_parallel_sets_return_global_labels(tmp_path: Path) -> None:
    part0 = tmp_path / "mapped_sets0.exo"
    part1 = tmp_path / "mapped_sets1.exo"

    _write_mapped_part(
        part0,
        coords=_coords0(),
        node_map=[10, 30, 50, 70],
        element_map=[100],
        temp=[10.0, 30.0, 50.0, 70.0],
        energy=[100.0],
    )
    _write_mapped_part(
        part1,
        coords=_coords1(),
        node_map=[20, 40, 60, 80],
        element_map=[200],
        temp=[20.0, 40.0, 60.0, 80.0],
        energy=[200.0],
    )

    with ParallelExodusFile.open(part0, part1) as exo:
        node_set = exo.node_set(100)
        assert np.allclose(node_set.nodes, [10, 20, 70, 80])  # ty: ignore[invalid-argument-type]

        side_set = exo.side_set(200)
        assert np.allclose(side_set.elems, [100, 200])  # ty: ignore[invalid-argument-type]


def test_parallel_legacy_methods(tmp_path: Path) -> None:
    part0, part1 = _write_parts(tmp_path)

    with exodusii.File(part0, part1) as exo:
        assert exo.num_dimensions() == 2
        assert exo.num_nodes() == 8
        assert exo.num_elems() == 2
        assert exo.num_elem_blk() == 1
        assert exo.num_node_sets() == 1
        assert exo.num_side_sets() == 1

        assert exo.get_element_block_ids().tolist() == [10]
        assert exo.get_element_block_id(1) == 10
        assert exo.get_element_block_iid(10) == 1
        assert exo.get_element_block(10).num_block_elems == 2

        assert np.allclose(exo.get_element_conn(10), [[1, 2, 3, 4], [5, 6, 7, 8]])

        assert exo.get_node_set_ids().tolist() == [100]
        assert exo.get_node_set_id(1) == 100
        assert exo.get_node_set_iid(100) == 1
        assert np.allclose(exo.get_node_set_nodes(100), [1, 4, 5, 8])

        assert exo.get_side_set_ids().tolist() == [200]
        assert exo.get_side_set_iid(200) == 1
        assert np.allclose(exo.get_side_set_elems(200), [1, 2])
        assert np.allclose(exo.get_side_set_sides(200), [2, 4])

        assert exo.get_global_variable_names().tolist() == ["TM_STEP"]
        assert exo.get_node_variable_names().tolist() == ["TEMP"]
        assert exo.get_element_variable_names().tolist() == ["ENERGY"]

        assert np.allclose(
            exo.get_node_variable_values("TEMP", time_step=2),
            [11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0, 81.0],
        )
        assert np.allclose(exo.get_element_variable_values(10, "ENERGY", 2), [1.5, 2.5])
        assert np.allclose(exo.get_node_variable_history("TEMP", 6), [60.0, 61.0])
        assert np.allclose(exo.get_element_variable_history("ENERGY", 2), [1.5, 2.5])


def test_exodus_name_decoding_preserves_meaningful_trailing_zero() -> None:
    from exodusii.api.file import _strip_exodus_padding

    assert _strip_exodus_padding("nodeset_100") == "nodeset_100"
    assert _strip_exodus_padding("sideset_200") == "sideset_200"
    assert _strip_exodus_padding("MAT_10") == "MAT_10"
    assert _strip_exodus_padding("DT_HYDRO0000000000000000000000000") == "DT_HYDRO"
