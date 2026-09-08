# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Mesh geometry helpers."""

from exodusii.mesh.elements import Element
from exodusii.mesh.elements import Hex8
from exodusii.mesh.elements import Quad4
from exodusii.mesh.elements import Tet4
from exodusii.mesh.elements import Tri3
from exodusii.mesh.elements import Wedge6
from exodusii.mesh.elements import element_factory
from exodusii.mesh.geometry import bounding_box
from exodusii.mesh.geometry import characteristic_element_length
from exodusii.mesh.geometry import connected_average
from exodusii.mesh.geometry import element_volumes
from exodusii.mesh.geometry import entity_centers
from exodusii.mesh.geometry import nodal_volumes
from exodusii.mesh.matching import MeshMap
from exodusii.mesh.matching import MeshMatchError
from exodusii.mesh.matching import build_mesh_map
from exodusii.mesh.matching import check_sideset_ordinals
from exodusii.mesh.regions import BoundedTimeDomain
from exodusii.mesh.regions import Circle
from exodusii.mesh.regions import Complement
from exodusii.mesh.regions import Cylinder
from exodusii.mesh.regions import Halfspace
from exodusii.mesh.regions import Intersection
from exodusii.mesh.regions import Quad
from exodusii.mesh.regions import Rectangle
from exodusii.mesh.regions import Region
from exodusii.mesh.regions import Slab
from exodusii.mesh.regions import Sphere
from exodusii.mesh.regions import TimeDomain
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
from exodusii.mesh.regions import slab
from exodusii.mesh.regions import sphere
from exodusii.mesh.regions import unbounded_time_domain
from exodusii.mesh.regions import union

__all__ = [
    "BoundedTimeDomain",
    "Circle",
    "Complement",
    "Cylinder",
    "Element",
    "Halfspace",
    "Hex8",
    "Intersection",
    "MeshMap",
    "MeshMatchError",
    "Quad",
    "Quad4",
    "Rectangle",
    "Region",
    "Slab",
    "Sphere",
    "Tet4",
    "TimeDomain",
    "Tri3",
    "UnboundedTimeDomain",
    "Union",
    "Wedge6",
    "bound_time_domain",
    "bounded_time_domain",
    "bounding_box",
    "build_mesh_map",
    "characteristic_element_length",
    "check_sideset_ordinals",
    "circle",
    "complement",
    "connected_average",
    "cylinder",
    "element_factory",
    "element_volumes",
    "entity_centers",
    "halfspace",
    "intersection",
    "nodal_volumes",
    "quad",
    "rectangle",
    "slab",
    "sphere",
    "unbounded_time_domain",
    "union",
]
