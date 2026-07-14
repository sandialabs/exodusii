# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Legacy callable similar module."""

import sys
import types
from typing import Any

from exodusii.api.compare import similar as _similar

similar = _similar


class _CallableModule(types.ModuleType):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _similar(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule

__all__ = ["similar"]
