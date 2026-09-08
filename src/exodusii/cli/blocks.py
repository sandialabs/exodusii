# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""``python -m exodusii blocks`` — block metadata."""

import argparse
import sys
from typing import Any
from typing import TextIO

from exodusii.api.file import ExodusFile
from exodusii.cli._command import Command
from exodusii.cli._common import BLOCK_ENTITIES
from exodusii.cli._common import block_payload
from exodusii.cli._common import limited_array_payload
from exodusii.cli._common import plural_key

__all__ = ["Blocks"]


class Blocks(Command):
    """Print block metadata."""

    name = "blocks"

    @staticmethod
    def setup_parser(parser: argparse.ArgumentParser) -> None:
        """Register arguments on *parser*."""
        parser.add_argument("file", help="Exodus database path.")
        parser.add_argument(
            "--connectivity",
            action="store_true",
            help="Include a limited connectivity preview for each block.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=5,
            help="Maximum connectivity rows to include when --connectivity is used.",
        )
        Command.add_terse_argument(parser)

    def execute(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        *,
        file: TextIO | None = None,
    ) -> int:
        """Return block metadata as JSON."""
        try:
            with ExodusFile.open(args.file) as exo:
                payload: dict[str, Any] = {"command": "blocks", "file": str(args.file)}
                for block_entity in BLOCK_ENTITIES:
                    key = plural_key(block_entity)
                    items = []
                    for block_id in exo.block_ids(block_entity):
                        block_id_int = int(block_id)
                        item = block_payload(exo, block_entity, block_id_int)
                        if args.connectivity:
                            conn = exo.block_connectivity(block_entity, block_id_int)
                            item["connectivity"] = limited_array_payload(conn, limit=args.limit)
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
    """Standalone entry point for the ``blocks`` subcommand."""
    parser = argparse.ArgumentParser(prog="python -m exodusii blocks", description=Blocks.__doc__)
    Blocks.setup_parser(parser)
    args = parser.parse_args(argv)
    return Blocks().execute(parser, args)


if __name__ == "__main__":
    raise SystemExit(main())
