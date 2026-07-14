# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from exodusii.api.compare import similar
from exodusii.api.copy import copy_file
from exodusii.api.writer import ExodusWriter
from exodusii.core.entities import Entity


def test_similar_attributes_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.exo"
    target = tmp_path / "target.exo"

    _write_attr_file(source)
    copy_file(source, target)

    assert similar(source, target)


def test_similar_detects_attribute_name_difference(tmp_path: Path) -> None:
    one = tmp_path / "one.exo"
    two = tmp_path / "two.exo"

    _write_attr_file(one, attr_names=["A", "B"])
    _write_attr_file(two, attr_names=["A", "C"])

    with pytest.raises(ValueError, match="attribute names"):
        similar(one, two)


def test_similar_detects_attribute_value_difference(tmp_path: Path) -> None:
    one = tmp_path / "one.exo"
    two = tmp_path / "two.exo"

    _write_attr_file(one, attr_values=[[1.0, 2.0]])
    _write_attr_file(two, attr_values=[[1.0, 3.0]])

    with pytest.raises(ValueError, match="different attributes"):
        similar(one, two)


def _write_attr_file(
    path: Path, *, attr_names: list[str] | None = None, attr_values: list[list[float]] | None = None
) -> None:
    attr_names = attr_names or ["A", "B"]
    attr_values = attr_values or [[1.0, 2.0]]

    with ExodusWriter.create(path) as writer:
        writer.initialize("attrs", 2, 4, 1, element_blocks=1)
        writer.write_coordinates([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        writer.define_element_block(10, "quad", [[1, 2, 3, 4]])
        writer.write_block_attributes(Entity.ELEMENT_BLOCK, 10, attr_values, names=attr_names)
