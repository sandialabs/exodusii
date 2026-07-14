# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Exception hierarchy for :mod:`exodusii`.

The modern API raises Exodus-specific exceptions so callers can distinguish
database, lookup, validation, and unsupported-feature failures from unrelated
Python exceptions.
"""


class ExodusError(Exception):
    """Base class for all ExodusII package exceptions."""


class ExodusIOError(ExodusError):
    """Base class for Exodus file input/output errors."""


class ExodusReadError(ExodusIOError):
    """Raised when an Exodus database cannot be read or decoded."""


class ExodusWriteError(ExodusIOError):
    """Raised when an Exodus database cannot be written."""


class ExodusLookupError(ExodusError, LookupError):
    """Raised when an Exodus entity, variable, dimension, or ID is not found."""


class ExodusDimensionError(ExodusLookupError):
    """Raised when a required Exodus dimension is missing or invalid."""


class ExodusVariableError(ExodusLookupError):
    """Raised when a required Exodus variable is missing or invalid."""


class ExodusInvalidEntityError(ExodusError, ValueError):
    """Raised when an entity name or type is not recognized."""


class ExodusInvalidTimeError(ExodusError, ValueError):
    """Raised when a requested time, time step, cycle, or index is invalid."""


class ExodusInvalidModeError(ExodusError, ValueError):
    """Raised when an unsupported file mode is requested."""


class ExodusUnsupportedFeatureError(ExodusError, NotImplementedError):
    """Raised when an Exodus feature is recognized but not implemented."""


class ExodusConsistencyError(ExodusError):
    """Raised when multiple Exodus databases are inconsistent with each other."""


__all__ = [
    "ExodusConsistencyError",
    "ExodusDimensionError",
    "ExodusError",
    "ExodusIOError",
    "ExodusInvalidEntityError",
    "ExodusInvalidModeError",
    "ExodusInvalidTimeError",
    "ExodusLookupError",
    "ExodusReadError",
    "ExodusUnsupportedFeatureError",
    "ExodusVariableError",
    "ExodusWriteError",
]
