# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""``python -m exodusii times`` — time-step information."""

import argparse
import sys
from typing import Any
from typing import TextIO

from exodusii.api.file import ExodusFile
from exodusii.cli._command import Command
from exodusii.cli._common import time_summary

__all__ = ["Times"]


class Times(Command):
    """Print time-step information."""

    name = "times"

    @staticmethod
    def setup_parser(parser: argparse.ArgumentParser) -> None:
        """Register arguments on *parser*."""
        parser.add_argument("file", help="Exodus database path.")
        parser.add_argument("--all", action="store_true", help="Print all time values.")
        parser.add_argument(
            "--limit",
            type=int,
            default=100,
            help="Maximum time values to print unless --all is supplied.",
        )
        Command.add_terse_argument(parser)

    def execute(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        *,
        file: TextIO | None = None,
    ) -> int:
        """Return time information as JSON."""
        try:
            with ExodusFile.open(args.file) as exo:
                times = exo.times()
                include_count = len(times) if args.all else min(max(args.limit, 0), len(times))
                payload: dict[str, Any] = {
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
            payload.setdefault("ok", True)
            self.emit_json(payload, terse=bool(getattr(args, "terse", False)), file=file)
            return 0
        except Exception as exc:
            self.emit_json(
                {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}},
                terse=bool(getattr(args, "terse", False)),
                file=file or sys.stderr,
            )
            return 1


def main(argv: list[str] | None = None, *, file=None) -> int:
    """Standalone entry point for the ``times`` subcommand."""
    parser = argparse.ArgumentParser(prog="python -m exodusii times", description=Times.__doc__)
    Times.setup_parser(parser)
    args = parser.parse_args(argv)
    return Times().execute(parser, args)


if __name__ == "__main__":
    raise SystemExit(main())
