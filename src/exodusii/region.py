# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Legacy region module."""

from exodusii.mesh.regions import BoundedTimeDomain
from exodusii.mesh.regions import Circle
from exodusii.mesh.regions import Cylinder
from exodusii.mesh.regions import Quad
from exodusii.mesh.regions import Rectangle
from exodusii.mesh.regions import Ring
from exodusii.mesh.regions import Sphere
from exodusii.mesh.regions import UnboundedTimeDomain
from exodusii.mesh.regions import bound_time_domain
from exodusii.mesh.regions import bounded_time_domain
from exodusii.mesh.regions import circle
from exodusii.mesh.regions import cylinder
from exodusii.mesh.regions import quad
from exodusii.mesh.regions import rectangle
from exodusii.mesh.regions import ring
from exodusii.mesh.regions import sphere
from exodusii.mesh.regions import unbounded_time_domain

__all__ = [
    "BoundedTimeDomain",
    "Circle",
    "Cylinder",
    "Quad",
    "Rectangle",
    "Ring",
    "Sphere",
    "UnboundedTimeDomain",
    "bound_time_domain",
    "bounded_time_domain",
    "circle",
    "cylinder",
    "quad",
    "rectangle",
    "ring",
    "sphere",
    "unbounded_time_domain",
]
