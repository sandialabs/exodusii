# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Name-table decoding and name-to-index lookups.

Regression tests for:

* trailing zeros being stripped from long names (``kinetic_energy_100`` read
  back as ``kinetic_energy_1``), and
* blank names shifting variable/attribute indices, so that the wrong on-disk
  column is returned.
"""

from pathlib import Path

import numpy as np
import pytest

from exodusii.api.file import ExodusFile
from exodusii.api.parallel import ParallelExodusFile
from exodusii.api.writer import ExodusWriter
from exodusii.core.entities import Entity
from exodusii.core.errors import ExodusLookupError


def _globals_file(path: Path, names: list[str], values: list[float]) -> Path:
    with ExodusWriter.create(path) as w:
        w.initialize("t", 1, node_count=0, element_count=0)
        w.define_global_variables(names)
        w.write_time(0.0)
        w.write_global_values(values)
    return path


@pytest.mark.parametrize(
    "names",
    [
        ["kinetic_energy_100", "kinetic_energy_1"],
        ["pressure_x_1000000", "pressure_x_1"],
        ["surface_200_side00", "surface_200_side"],
    ],
)
def test_long_names_keep_trailing_zeros(tmp_path: Path, names: list[str]) -> None:
    path = _globals_file(tmp_path / "zeros.exo", names, [1.0, 2.0])

    with ExodusFile.open(path) as exo:
        assert exo.variable_names("global") == tuple(names)
        assert np.allclose(exo.values(names[0], on="global"), [1.0])
        assert np.allclose(exo.values(names[1], on="global"), [2.0])


def test_long_block_and_node_variable_names_keep_trailing_zeros(tmp_path: Path) -> None:
    path = tmp_path / "block_zeros.exo"
    with ExodusWriter.create(path) as w:
        w.initialize("t", 1, node_count=2, element_count=1, element_blocks=1)
        w.write_coordinates([[0.0], [1.0]])
        w.define_element_block(1, "bar2", [[1, 2]], name="block_material_100")
        w.define_node_variables(["temperature_at_100"])
        w.write_time(0.0)
        w.write_node_values("temperature_at_100", [3.0, 4.0])

    with ExodusFile.open(path) as exo:
        assert exo.element_block(1).name == "block_material_100"
        assert exo.variable_names("node") == ("temperature_at_100",)
        assert np.allclose(exo.values("temperature_at_100", on="node"), [3.0, 4.0])


def test_blank_variable_name_does_not_shift_indices(tmp_path: Path) -> None:
    path = _globals_file(tmp_path / "blank.exo", ["KE", "", "PE"], [1.0, 2.0, 3.0])

    with ExodusFile.open(path) as exo:
        # Blank names are omitted from the listing ...
        assert exo.variable_names("global") == ("KE", "PE")
        # ... but lookups use on-disk positions.
        assert np.allclose(exo.values("KE", on="global"), [1.0])
        assert np.allclose(exo.values("PE", on="global"), [3.0])
        with pytest.raises(ExodusLookupError):
            exo.values("", on="global")


def test_blank_node_variable_name_does_not_shift_indices(tmp_path: Path) -> None:
    path = tmp_path / "blank_nodal.exo"
    with ExodusWriter.create(path) as w:
        w.initialize("t", 1, node_count=2, element_count=0)
        w.write_coordinates([[0.0], [1.0]])
        w.define_node_variables(["", "U"])
        w.write_time(0.0)
        w.write_values("", [9.0, 9.0], on=Entity.NODE)
        w.write_node_values("U", [1.0, 2.0])

    with ExodusFile.open(path) as exo:
        assert exo.variable_names("node") == ("U",)
        assert np.allclose(exo.values("U", on="node"), [1.0, 2.0])


def test_blank_attribute_name_does_not_shift_columns(tmp_path: Path) -> None:
    path = tmp_path / "attr.exo"
    with ExodusWriter.create(path) as w:
        w.initialize("t", 1, node_count=2, element_count=1, element_blocks=1)
        w.write_coordinates([[0.0], [1.0]])
        w.define_element_block(1, "bar2", [[1, 2]])
        w.write_block_attributes(Entity.ELEMENT_BLOCK, 1, [[1.5, 2.5]], names=["", "thickness"])

    with ExodusFile.open(path) as exo:
        assert exo.attribute_names("element_block", 1) == ("thickness",)
        assert np.allclose(exo.attribute_values("element_block", 1, "thickness"), [2.5])


def test_parallel_blank_variable_name_does_not_shift_indices(tmp_path: Path) -> None:
    paths = []
    for rank in range(2):
        path = tmp_path / f"p.exo.2.{rank}"
        with ExodusWriter.create(path) as w:
            w.initialize("t", 1, node_count=0, element_count=0)
            w.define_global_variables(["KE", "", "PE"])
            w.write_time(0.0)
            w.write_global_values([1.0, 2.0, 3.0])
        paths.append(path)

    pef = ParallelExodusFile(*paths)
    try:
        assert np.allclose(pef.values("PE", on="global"), [3.0])
    finally:
        pef.close()
