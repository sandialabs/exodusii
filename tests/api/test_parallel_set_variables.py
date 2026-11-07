# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np

from exodusii.api.file import ExodusFile
from exodusii.api.parallel import ParallelExodusFile
from exodusii.api.writer import ExodusWriter


def test_parallel_node_and_side_set_variables(tmp_path: Path) -> None:
    part0 = tmp_path / "part0.exo"
    part1 = tmp_path / "part1.exo"

    _write_part(
        part0, node_map=[10, 20, 30, 40], element_map=[100], ns_values=[1.0, 4.0], ss_values=[10.0]
    )
    _write_part(
        part1, node_map=[50, 60, 70, 80], element_map=[200], ns_values=[5.0, 8.0], ss_values=[20.0]
    )

    with ParallelExodusFile.open(part0, part1) as exo:
        node_set = exo.node_set(10)
        assert np.allclose(node_set.nodes, [10, 40, 50, 80])
        assert np.allclose(
            exo.values("NSVAR", on="node_set", set_id=10, time=0), [1.0, 4.0, 5.0, 8.0]
        )

        side_set = exo.side_set(20)
        assert np.allclose(side_set.elems, [100, 200])
        assert np.allclose(exo.values("SSVAR", on="side_set", set_id=20, time=0), [10.0, 20.0])


def test_parallel_set_variable_write(tmp_path: Path) -> None:
    part0 = tmp_path / "part0.exo"
    part1 = tmp_path / "part1.exo"
    joined = tmp_path / "joined.exo"

    _write_part(
        part0, node_map=[10, 20, 30, 40], element_map=[100], ns_values=[1.0, 4.0], ss_values=[10.0]
    )
    _write_part(
        part1, node_map=[50, 60, 70, 80], element_map=[200], ns_values=[5.0, 8.0], ss_values=[20.0]
    )

    with ParallelExodusFile.open(part0, part1) as exo:
        exo.write(joined)

    with ExodusFile.open(joined) as exo:
        assert exo.variable_names("node_set") == ("NSVAR",)
        assert exo.variable_names("side_set") == ("SSVAR",)
        assert np.allclose(
            exo.values("NSVAR", on="node_set", set_id=10, time=0), [1.0, 4.0, 5.0, 8.0]
        )
        assert np.allclose(exo.values("SSVAR", on="side_set", set_id=20, time=0), [10.0, 20.0])


def _write_part(
    path: Path,
    *,
    node_map: list[int],
    element_map: list[int],
    ns_values: list[float],
    ss_values: list[float],
) -> None:
    from exodusii.core.names import VariableName

    with ExodusWriter.create(path) as writer:
        writer.initialize("set var part", 2, 4, 1, node_sets=1, side_sets=1)
        writer.write_coordinates(_coords())
        writer.define_node_set(10, [1, 4])
        writer.define_side_set(20, [1], [2])

        writer.backend.create_variable(VariableName.NODE_ID_MAP.value, int, ("num_nodes",))
        writer.backend.write_variable(VariableName.NODE_ID_MAP.value, node_map)
        writer.backend.create_variable(VariableName.ELEMENT_ID_MAP.value, int, ("num_elem",))
        writer.backend.write_variable(VariableName.ELEMENT_ID_MAP.value, element_map)

        writer.define_node_set_variables(["NSVAR"])
        writer.define_side_set_variables(["SSVAR"])

        writer.write_time(0.0)
        writer.write_values("NSVAR", ns_values, on=writer_entity("node_set"), set_id=10)
        writer.write_values("SSVAR", ss_values, on=writer_entity("side_set"), set_id=20)


def writer_entity(name: str):
    from exodusii.core.entities import entity

    return entity(name)


def _coords() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
