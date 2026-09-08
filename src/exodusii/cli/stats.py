# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""``python -m exodusii stats`` — compact numeric summaries for variables."""

import argparse
import sys
from typing import Any
from typing import TextIO

from exodusii.api.file import ExodusFile
from exodusii.cli._command import Command
from exodusii.cli._common import _piece_path
from exodusii.cli._common import resolved_time_payload
from exodusii.cli._common import variable_stats_payload
from exodusii.core.entities import Entity
from exodusii.core.entities import entity
from exodusii.core.selectors import parse_variable_selectors

__all__ = ["Stats"]


class Stats(Command):
    """Print compact numeric summaries for selected variables."""

    name = "stats"

    @staticmethod
    def setup_parser(parser: argparse.ArgumentParser) -> None:
        """Register arguments on *parser*."""
        parser.add_argument("file", help="Exodus database path.")
        parser.add_argument(
            "--select",
            action="append",
            required=True,
            help="Variable selector, e.g. g/TOTAL_ENERGY, n/TEMP, e/ENERGY.",
        )
        parser.add_argument(
            "--time",
            default="last",
            help=(
                "Time selector. Use first, last, a physical time like 0.25, "
                "index:N for zero-based index, or step:N for one-based Exodus step. "
                "[default: last]"
            ),
        )
        parser.add_argument(
            "--by-block",
            action="store_true",
            help="For element/edge/face variables, also report statistics by block.",
        )
        parser.add_argument(
            "--by-set",
            action="store_true",
            help="For set variables, also report statistics by set.",
        )
        parser.add_argument(
            "--piece",
            type=int,
            default=None,
            metavar="N",
            help=(
                "Open only the Nth file as a single-piece reader instead of aggregating. "
                "Useful for reading global scalars from one small decomposed component "
                "without opening the full joined file. Zero-based index into the file argument."
            ),
        )
        Command.add_terse_argument(parser)

    def execute(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        *,
        file: TextIO | None = None,
    ) -> int:
        """Return compact variable statistics as JSON."""
        try:
            selectors = parse_variable_selectors(args.select, require_same_entity=True)
            if not selectors:
                raise ValueError("at least one selector is required")

            location = entity(selectors[0].entity)
            time_selector = self.parse_time_selector(args.time)
            file_path = _piece_path(args.file, getattr(args, "piece", None))

            with ExodusFile.open(file_path) as exo:
                payload: dict[str, Any] = {
                    "command": "stats",
                    "file": str(args.file),
                    "entity": location.value,
                    "variables": {},
                }
                if getattr(args, "piece", None) is not None:
                    payload["piece"] = int(args.piece)
                if location is not Entity.GLOBAL or args.time is not None:
                    payload["time"] = resolved_time_payload(exo, time_selector, requested=args.time)
                for selector in selectors:
                    payload["variables"][selector.name] = variable_stats_payload(
                        exo,
                        selector,
                        time=time_selector,
                        by_block=bool(args.by_block),
                        by_set=bool(args.by_set),
                    )

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
    """Standalone entry point for the ``stats`` subcommand."""
    parser = argparse.ArgumentParser(prog="python -m exodusii stats", description=Stats.__doc__)
    Stats.setup_parser(parser)
    args = parser.parse_args(argv)
    return Stats().execute(parser, args)


if __name__ == "__main__":
    raise SystemExit(main())
