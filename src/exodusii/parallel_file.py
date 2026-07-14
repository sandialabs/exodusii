# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Legacy parallel file module."""

from exodusii.api.parallel import ParallelExodusFile

parallel_exodusii_file = ParallelExodusFile
MFExodusIIFile = ParallelExodusFile

__all__ = ["MFExodusIIFile", "ParallelExodusFile", "parallel_exodusii_file"]
