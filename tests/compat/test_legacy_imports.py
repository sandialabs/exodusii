# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np

import exodusii
import exodusii.region
from exodusii.allclose import allclose
from exodusii.copy import copy
from exodusii.copy import copy_file
from exodusii.element import Quad4
from exodusii.element import factory
from exodusii.file import ExodusIIFile
from exodusii.file import File
from exodusii.file import exodusii_file
from exodusii.file import write_globals
from exodusii.lineout import Lineout
from exodusii.lineout import lineout
from exodusii.similar import similar


def test_legacy_file_imports() -> None:
    assert File is exodusii.File
    assert ExodusIIFile is exodusii.ExodusIIFile
    assert exodusii_file is exodusii.exodusii_file
    assert write_globals is exodusii.write_globals


def test_legacy_compare_imports() -> None:
    assert callable(exodusii.allclose)
    assert callable(exodusii.similar)
    assert callable(allclose)
    assert callable(similar)


def test_legacy_lineout_imports() -> None:
    assert callable(exodusii.lineout)
    assert callable(lineout)
    assert isinstance(Lineout(x="x"), Lineout)


def test_legacy_region_imports() -> None:
    region = exodusii.region.circle([0.0, 0.0], 1.0)

    assert region.contains([0.0, 0.0])
    assert not region.contains([2.0, 0.0])


def test_legacy_element_imports() -> None:
    element = factory("quad", np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))

    assert isinstance(element, Quad4)
    assert element.volume == 1.0


def test_legacy_copy_imports() -> None:
    assert callable(copy)
    assert callable(copy_file)
    assert callable(exodusii.copy_file)
