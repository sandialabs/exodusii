# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Modern public API for Exodus databases."""

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

__all__ = [
    "ComparisonResult",
    "ExodusFile",
    "ExodusWriter",
    "Lineout",
    "ParallelExodusFile",
    "QueryResult",
    "allclose",
    "copy",
    "copy_file",
    "lineout",
    "print_query",
    "query",
    "similar",
]
