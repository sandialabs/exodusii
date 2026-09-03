# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Modern Python interface for Exodus II finite element databases."""

import importlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

from exodusii.api.compare import ComparisonResult
from exodusii.api.compare import allclose
from exodusii.api.compare import similar
from exodusii.api.copy import copy
from exodusii.api.copy import copy_file
from exodusii.api.diff import DiffOptions
from exodusii.api.diff import DiffResult
from exodusii.api.diff import TimeSelection
from exodusii.api.diff import VariableDiff
from exodusii.api.diff import diff
from exodusii.api.file import ExodusFile
from exodusii.api.lineout import Lineout
from exodusii.api.lineout import lineout
from exodusii.api.parallel import ParallelExodusFile
from exodusii.api.query import QueryResult
from exodusii.api.query import print_query
from exodusii.api.query import query
from exodusii.api.writer import ExodusWriter
from exodusii.compat import ExodusIIFile
from exodusii.compat import File
from exodusii.compat import MFExodusIIFile
from exodusii.compat import ParallelExodusIIFile
from exodusii.compat import exodusii_file
from exodusii.compat import parallel_exodusii_file
from exodusii.compat import write_globals
from exodusii.core.tolerance import Tolerance
from exodusii.core.tolerance import ToleranceMode

region = importlib.import_module("exodusii.region")

exo_file = File

try:
    __version__ = version("exodusii")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = [
    "ComparisonResult",
    "DiffOptions",
    "DiffResult",
    "TimeSelection",
    "ExodusFile",
    "ExodusIIFile",
    "ExodusWriter",
    "File",
    "Lineout",
    "MFExodusIIFile",
    "ParallelExodusFile",
    "ParallelExodusIIFile",
    "QueryResult",
    "Tolerance",
    "ToleranceMode",
    "VariableDiff",
    "__version__",
    "allclose",
    "copy",
    "copy_file",
    "diff",
    "exo_file",
    "exodusii_file",
    "lineout",
    "parallel_exodusii_file",
    "print_query",
    "query",
    "region",
    "similar",
    "write_globals",
]
