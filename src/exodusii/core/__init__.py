# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Core types and utilities for :mod:`exodusii`."""

from exodusii.core.entities import Entity
from exodusii.core.entities import entity
from exodusii.core.models import Block
from exodusii.core.models import InfoRecord
from exodusii.core.models import InitParams
from exodusii.core.models import QARecord
from exodusii.core.models import SetInfo
from exodusii.core.models import VariableInfo
from exodusii.core.schema import BlockSpec
from exodusii.core.schema import MapSpec
from exodusii.core.schema import ObjectSpec
from exodusii.core.schema import SetSpec
from exodusii.core.schema import VariableSpec
from exodusii.core.schema import block_entities
from exodusii.core.schema import block_spec
from exodusii.core.schema import map_spec
from exodusii.core.schema import object_spec
from exodusii.core.schema import set_entities
from exodusii.core.schema import set_spec
from exodusii.core.schema import variable_entities
from exodusii.core.schema import variable_spec
from exodusii.core.schema import variable_value_name
from exodusii.core.selectors import VariableSelector
from exodusii.core.selectors import parse_variable_selector
from exodusii.core.selectors import parse_variable_selectors
from exodusii.core.time import TimeSelection
from exodusii.core.time import nearest_time_index
from exodusii.core.time import resolve_time
from exodusii.core.time import resolve_time_step

__all__ = [
    "Block",
    "BlockSpec",
    "Entity",
    "InfoRecord",
    "InitParams",
    "MapSpec",
    "ObjectSpec",
    "QARecord",
    "SetInfo",
    "SetSpec",
    "TimeSelection",
    "VariableInfo",
    "VariableSelector",
    "VariableSpec",
    "block_entities",
    "block_spec",
    "entity",
    "map_spec",
    "nearest_time_index",
    "object_spec",
    "parse_variable_selector",
    "parse_variable_selectors",
    "resolve_time",
    "resolve_time_step",
    "set_entities",
    "set_spec",
    "variable_entities",
    "variable_spec",
    "variable_value_name",
]
