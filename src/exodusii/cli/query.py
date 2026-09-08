# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""``python -m exodusii query`` — query variables and print JSON records."""

import argparse
import sys
from typing import Any
from typing import TextIO

from exodusii.api.file import ExodusFile
from exodusii.api.query import query
from exodusii.cli._command import Command
from exodusii.cli._common import _piece_path
from exodusii.cli._common import resolved_time_payload
from exodusii.cli._common import structured_to_records

__all__ = ["Query"]


class Query(Command):
    """Query variables and print JSON records."""

    name = "query"

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
            default=None,
            help=(
                "Time selector. Use first, last, a physical time like 0.25, "
                "index:N for zero-based index, or step:N for one-based Exodus step."
            ),
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=100,
            help="Maximum result rows to include. Use --limit -1 for all rows.",
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
        query_index = parser.add_mutually_exclusive_group()
        query_index.add_argument(
            "--object-index",
            action="store_true",
            default=True,
            help="Include object index for node/element queries. [default: true]",
        )
        query_index.add_argument(
            "--no-object-index",
            action="store_false",
            dest="object_index",
            help="Do not include object index.",
        )
        Command.add_terse_argument(parser)

    def execute(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        *,
        file: TextIO | None = None,
    ) -> int:
        """Run a structured query and return JSON records."""
        try:
            time_selector = self.parse_time_selector(args.time)
            limit = self.normalize_limit(args.limit)
            file_path = _piece_path(args.file, getattr(args, "piece", None))

            with ExodusFile.open(file_path) as exo:
                result = query(
                    exo, *args.select, time=time_selector, object_index=bool(args.object_index)
                )
                row_count = len(result.data)
                returned_rows = row_count if limit is None else min(row_count, limit)

                payload: dict[str, Any] = {
                    "command": "query",
                    "file": str(args.file),
                    "metadata": result.metadata,
                    "columns": list(result.names),
                    "row_count": row_count,
                    "returned_rows": returned_rows,
                    "truncated": returned_rows < row_count,
                    "data": structured_to_records(result.data, limit=limit),
                }
                if getattr(args, "piece", None) is not None:
                    payload["piece"] = int(args.piece)
                if "time" in result.metadata:
                    payload["time"] = resolved_time_payload(exo, time_selector, requested=args.time)

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
    """Standalone entry point for the ``query`` subcommand."""
    parser = argparse.ArgumentParser(prog="python -m exodusii query", description=Query.__doc__)
    Query.setup_parser(parser)
    args = parser.parse_args(argv)
    return Query().execute(parser, args)


if __name__ == "__main__":
    raise SystemExit(main())
