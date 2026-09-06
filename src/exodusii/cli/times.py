# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""``python -m exodusii times`` — time-step information."""

import argparse
from typing import Any

from exodusii.api.file import ExodusFile
from exodusii.cli._common import time_summary

COMMAND = "times"

__all__ = ["COMMAND", "add_subparser", "times_command"]


def add_subparser(
    subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    """Register the ``times`` subparser."""
    parser = subparsers.add_parser(COMMAND, parents=[common], help="Print time-step information.")
    parser.add_argument("file", help="Exodus database path.")
    parser.add_argument("--all", action="store_true", help="Print all time values.")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum time values to print unless --all is supplied.",
    )
    return parser


def times_command(args: argparse.Namespace) -> dict[str, Any]:
    """Return time information."""
    with ExodusFile.open(args.file) as exo:
        times = exo.times()
        include_count = len(times) if args.all else min(max(args.limit, 0), len(times))

        return {
            "command": "times",
            "file": str(args.file),
            **time_summary(times),
            "returned": include_count,
            "truncated": include_count < len(times),
            "times": [
                {"index": index, "step": index + 1, "value": float(value)}
                for index, value in enumerate(times[:include_count])
            ],
        }
