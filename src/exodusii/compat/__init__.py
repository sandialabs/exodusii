# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Backward-compatible ExodusII-style API."""

from exodusii.compat.legacy_file import ExodusIIFile
from exodusii.compat.legacy_file import File
from exodusii.compat.legacy_file import exodusii_file
from exodusii.compat.legacy_file import find_element_data_in_region
from exodusii.compat.legacy_file import write_globals
from exodusii.compat.legacy_parallel import MFExodusIIFile
from exodusii.compat.legacy_parallel import ParallelExodusIIFile
from exodusii.compat.legacy_parallel import parallel_exodusii_file

exo_file = File

__all__ = [
    "ExodusIIFile",
    "File",
    "MFExodusIIFile",
    "ParallelExodusIIFile",
    "exo_file",
    "exodusii_file",
    "find_element_data_in_region",
    "parallel_exodusii_file",
    "write_globals",
]
