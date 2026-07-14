# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Legacy element module."""

from exodusii.mesh.elements import Element
from exodusii.mesh.elements import Hex8
from exodusii.mesh.elements import Quad4
from exodusii.mesh.elements import Tet4
from exodusii.mesh.elements import Tri3
from exodusii.mesh.elements import Wedge6
from exodusii.mesh.elements import element_factory

factory = element_factory

__all__ = ["Element", "Hex8", "Quad4", "Tet4", "Tri3", "Wedge6", "element_factory", "factory"]
