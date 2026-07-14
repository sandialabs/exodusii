# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Legacy file module."""

from exodusii.compat.legacy_file import ExodusIIFile
from exodusii.compat.legacy_file import File
from exodusii.compat.legacy_file import exodusii_file
from exodusii.compat.legacy_file import write_globals

exo_file = File

__all__ = ["ExodusIIFile", "File", "exo_file", "exodusii_file", "write_globals"]
