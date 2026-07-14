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
from exodusii.compat import exodusii_file
from exodusii.compat import write_globals

region = importlib.import_module("exodusii.region")

parallel_exodusii_file = ParallelExodusFile
MFExodusIIFile = ParallelExodusFile
exo_file = File

try:
    __version__ = version("exodusii")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = [
    "ComparisonResult",
    "ExodusFile",
    "ExodusIIFile",
    "ExodusWriter",
    "File",
    "Lineout",
    "MFExodusIIFile",
    "ParallelExodusFile",
    "QueryResult",
    "__version__",
    "allclose",
    "copy",
    "copy_file",
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
