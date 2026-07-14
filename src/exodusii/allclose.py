# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Legacy callable allclose module."""

import sys
import types
from typing import Any

from exodusii.api.compare import allclose as _allclose

allclose = _allclose


class _CallableModule(types.ModuleType):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _allclose(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule

__all__ = ["allclose"]
