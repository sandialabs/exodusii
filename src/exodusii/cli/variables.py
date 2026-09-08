# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""``python -m exodusii variables`` — result-variable inventory."""

import argparse
import sys
from typing import Any
from typing import TextIO

from exodusii.api.file import ExodusFile
from exodusii.cli._command import Command
from exodusii.cli._common import VARIABLE_ENTITIES
from exodusii.cli._common import variable_block_location
from exodusii.core.entities import Entity

__all__ = ["Variables"]


class Variables(Command):
    """Print result-variable inventory."""

    name = "variables"

    @staticmethod
    def setup_parser(parser: argparse.ArgumentParser) -> None:
        """Register arguments on *parser*."""
        parser.add_argument("file", help="Exodus database path.")
        parser.add_argument(
            "--truth-tables", action="store_true", help="Include block/set variable truth tables."
        )
        Command.add_terse_argument(parser)

    def execute(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        *,
        file: TextIO | None = None,
    ) -> int:
        """Return variable inventory as JSON."""
        try:
            include_truth_tables = bool(args.truth_tables)
            with ExodusFile.open(args.file) as exo:
                variables: list[dict[str, Any]] = []
                for ent in VARIABLE_ENTITIES:
                    names = exo.variable_names(ent)
                    item: dict[str, Any] = {
                        "entity": ent.value,
                        "selector_prefix": ent.short_name,
                        "names": list(names),
                        "count": len(names),
                    }
                    if ent in {Entity.ELEMENT, Entity.EDGE, Entity.FACE}:
                        location = variable_block_location(ent)
                        item["block_entity"] = location.value
                        item["block_ids"] = exo.block_ids(location).tolist()
                    if ent in {
                        Entity.NODE_SET,
                        Entity.SIDE_SET,
                        Entity.EDGE_SET,
                        Entity.FACE_SET,
                        Entity.ELEMENT_SET,
                    }:
                        item["set_entity"] = ent.value
                        item["set_ids"] = exo.set_ids(ent).tolist()
                    if include_truth_tables:
                        table = exo.variable_truth_table(ent)
                        if table is not None:
                            item["truth_table"] = table.tolist()
                    variables.append(item)

                payload: dict[str, Any] = {
                    "command": "variables",
                    "file": str(args.file),
                    "variables": variables,
                    "agent_hints": {
                        "selector_format": "ENTITY/NAME",
                        "examples": ["g/TOTAL_ENERGY", "n/TEMP", "e/ENERGY"],
                    },
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
    """Standalone entry point for the ``variables`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="python -m exodusii variables", description=Variables.__doc__
    )
    Variables.setup_parser(parser)
    args = parser.parse_args(argv)
    return Variables().execute(parser, args)


if __name__ == "__main__":
    raise SystemExit(main())
