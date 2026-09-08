# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""``python -m exodusii sets`` — set metadata."""

import argparse
import sys
from typing import Any
from typing import TextIO

from exodusii.api.file import ExodusFile
from exodusii.cli._command import Command
from exodusii.cli._common import SET_ENTITIES
from exodusii.cli._common import limited_array_payload
from exodusii.cli._common import plural_key
from exodusii.cli._common import set_payload

__all__ = ["Sets"]


class Sets(Command):
    """Print set metadata."""

    name = "sets"

    @staticmethod
    def setup_parser(parser: argparse.ArgumentParser) -> None:
        """Register arguments on *parser*."""
        parser.add_argument("file", help="Exodus database path.")
        parser.add_argument(
            "--entries", action="store_true", help="Include a limited entries preview for each set."
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=20,
            help="Maximum set entries to include when --entries is used.",
        )
        Command.add_terse_argument(parser)

    def execute(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        *,
        file: TextIO | None = None,
    ) -> int:
        """Return set metadata as JSON."""
        try:
            with ExodusFile.open(args.file) as exo:
                payload: dict[str, Any] = {"command": "sets", "file": str(args.file)}
                for set_entity in SET_ENTITIES:
                    key = plural_key(set_entity)
                    items = []
                    for set_id in exo.set_ids(set_entity):
                        set_id_int = int(set_id)
                        item = set_payload(exo, set_entity, set_id_int)
                        if args.entries:
                            set_info = exo.set(set_entity, set_id_int)
                            item["entries"] = limited_array_payload(
                                set_info.entries, limit=args.limit
                            )
                            if set_info.extra_entries is not None:
                                item["extra_entries"] = limited_array_payload(
                                    set_info.extra_entries, limit=args.limit
                                )
                            if set_info.dist_facts is not None:
                                item["distribution_values"] = limited_array_payload(
                                    set_info.dist_facts, limit=args.limit
                                )
                        items.append(item)
                    payload[key] = items
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
    """Standalone entry point for the ``sets`` subcommand."""
    parser = argparse.ArgumentParser(prog="python -m exodusii sets", description=Sets.__doc__)
    Sets.setup_parser(parser)
    args = parser.parse_args(argv)
    return Sets().execute(parser, args)


if __name__ == "__main__":
    raise SystemExit(main())
