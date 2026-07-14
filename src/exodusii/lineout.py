# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Legacy callable lineout module."""

import sys
import types
from typing import Any

from exodusii.api.lineout import Lineout
from exodusii.api.lineout import lineout as _lineout

lineout = _lineout


class _CallableModule(types.ModuleType):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _lineout(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule

__all__ = ["Lineout", "lineout"]
