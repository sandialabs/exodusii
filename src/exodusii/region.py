# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Legacy region module."""

from exodusii.mesh.regions import BoundedTimeDomain
from exodusii.mesh.regions import Circle
from exodusii.mesh.regions import Complement
from exodusii.mesh.regions import Cylinder
from exodusii.mesh.regions import Halfspace
from exodusii.mesh.regions import Intersection
from exodusii.mesh.regions import Quad
from exodusii.mesh.regions import Rectangle
from exodusii.mesh.regions import Ring
from exodusii.mesh.regions import Slab
from exodusii.mesh.regions import Sphere
from exodusii.mesh.regions import UnboundedTimeDomain
from exodusii.mesh.regions import Union
from exodusii.mesh.regions import bound_time_domain
from exodusii.mesh.regions import bounded_time_domain
from exodusii.mesh.regions import circle
from exodusii.mesh.regions import complement
from exodusii.mesh.regions import cylinder
from exodusii.mesh.regions import halfspace
from exodusii.mesh.regions import intersection
from exodusii.mesh.regions import quad
from exodusii.mesh.regions import rectangle
from exodusii.mesh.regions import ring
from exodusii.mesh.regions import slab
from exodusii.mesh.regions import sphere
from exodusii.mesh.regions import unbounded_time_domain
from exodusii.mesh.regions import union

__all__ = [
    "BoundedTimeDomain",
    "Circle",
    "Complement",
    "Cylinder",
    "Halfspace",
    "Intersection",
    "Quad",
    "Rectangle",
    "Ring",
    "Slab",
    "Sphere",
    "UnboundedTimeDomain",
    "Union",
    "bound_time_domain",
    "bounded_time_domain",
    "circle",
    "complement",
    "cylinder",
    "halfspace",
    "intersection",
    "quad",
    "rectangle",
    "ring",
    "slab",
    "sphere",
    "unbounded_time_domain",
    "union",
]
