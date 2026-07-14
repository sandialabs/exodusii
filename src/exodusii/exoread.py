# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Legacy exoread module."""

from exodusii.cli.exoread import Namespace
from exodusii.cli.exoread import build_parser
from exodusii.cli.exoread import describe
from exodusii.cli.exoread import main

__all__ = ["Namespace", "build_parser", "describe", "main"]
