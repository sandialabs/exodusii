# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""``python -m exodusii tracers`` — query tracer variables by tracer ID."""

import argparse
import sys
from typing import Any
from typing import TextIO

from exodusii.api.file import ExodusFile
from exodusii.cli._command import Command
from exodusii.core.selectors import parse_variable_selectors

__all__ = ["Tracers"]


class Tracers(Command):
    """Query tracer variables by tracer ID."""

    name = "tracers"

    @staticmethod
    def setup_parser(parser: argparse.ArgumentParser) -> None:
        """Register arguments on *parser*."""
        parser.add_argument("file", help="Exodus database path.")
        parser.add_argument(
            "--select",
            required=True,
            metavar="n/NAME",
            help="Nodal variable selector, e.g. n/VELX.",
        )
        parser.add_argument(
            "--ids",
            default=None,
            metavar="ID[,ID...]",
            help="Comma-separated tracer IDs to return. [default: all tracers]",
        )
        parser.add_argument(
            "--time",
            default="last",
            help=(
                "Time selector. Use first, last, a physical time, index:N, or step:N. "
                "[default: last]"
            ),
        )
        parser.add_argument(
            "--id-variable",
            default="ID",
            metavar="NAME",
            help="Name of the nodal variable holding tracer IDs. [default: ID]",
        )
        Command.add_terse_argument(parser)

    def execute(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        *,
        file: TextIO | None = None,
    ) -> int:
        """Return tracer variable values keyed by tracer ID as JSON."""
        try:
            selectors = parse_variable_selectors([args.select], require_same_entity=False)
            if not selectors:
                raise ValueError("--select requires a valid variable selector")
            selector = selectors[0]
            var_name = selector.name

            ids: list[int] | None = None
            if args.ids is not None:
                ids = [int(x.strip()) for x in args.ids.split(",") if x.strip()]

            time_selector = self.parse_time_selector(args.time)
            id_var = getattr(args, "id_variable", "ID")

            with ExodusFile.open(args.file) as exo:
                result = exo.tracer(var_name, ids=ids, time=time_selector, id_variable=id_var)
                all_ids = exo.tracer_ids(id_variable=id_var).tolist()

            data: dict[str, Any] = {str(tid): self.jsonable(v) for tid, v in result.items()}

            payload: dict[str, Any] = {
                "command": "tracers",
                "file": str(args.file),
                "variable": var_name,
                "id_variable": id_var,
                "all_ids": all_ids,
                "requested_ids": ids,
                "time": args.time,
                "data": data,
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
    """Standalone entry point for the ``tracers`` subcommand."""
    parser = argparse.ArgumentParser(prog="python -m exodusii tracers", description=Tracers.__doc__)
    Tracers.setup_parser(parser)
    args = parser.parse_args(argv)
    return Tracers().execute(parser, args)


if __name__ == "__main__":
    raise SystemExit(main())
