# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Legacy parallel file module."""

from exodusii.api.parallel import ParallelExodusFile
from exodusii.compat.legacy_parallel import MFExodusIIFile
from exodusii.compat.legacy_parallel import ParallelExodusIIFile
from exodusii.compat.legacy_parallel import parallel_exodusii_file

__all__ = ["MFExodusIIFile", "ParallelExodusFile", "ParallelExodusIIFile", "parallel_exodusii_file"]
