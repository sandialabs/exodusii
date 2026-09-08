# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""``python -m exodusii inspect`` — agent-friendly database summary."""

import argparse
import sys
from typing import Any
from typing import TextIO

from exodusii.api.file import ExodusFile
from exodusii.cli._command import Command
from exodusii.cli._common import VARIABLE_ENTITIES
from exodusii.cli._common import agent_hints
from exodusii.cli._common import time_summary

__all__ = ["Inspect"]


class Inspect(Command):
    """Print an agent-friendly database summary."""

    name = "inspect"

    @staticmethod
    def setup_parser(parser: argparse.ArgumentParser) -> None:
        """Register arguments on *parser*."""
        parser.add_argument("file", help="Exodus database path.")
        Command.add_terse_argument(parser)

    def execute(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        *,
        file: TextIO | None = None,
    ) -> int:
        """Return an agent-friendly database summary as JSON."""
        try:
            with ExodusFile.open(args.file) as exo:
                times = exo.times()
                payload: dict[str, Any] = {
                    "command": "inspect",
                    "file": str(args.file),
                    "title": exo.title,
                    "storage_type": exo.storage_type,
                    "dimension": exo.dimension,
                    "counts": {
                        "nodes": exo.node_count,
                        "edges": exo.edge_count,
                        "faces": exo.face_count,
                        "elements": exo.element_count,
                        "element_blocks": exo.element_block_count,
                        "edge_blocks": len(exo.edge_block_ids()),
                        "face_blocks": len(exo.face_block_ids()),
                        "node_sets": exo.node_set_count,
                        "side_sets": exo.side_set_count,
                        "edge_sets": len(exo.edge_set_ids()),
                        "face_sets": len(exo.face_set_ids()),
                        "element_sets": len(exo.element_set_ids()),
                        "time_steps": len(times),
                    },
                    "coordinates": {
                        "names": exo.coordinate_names().tolist(),
                        "displacement_variables": list(exo.displacement_variable_names()),
                    },
                    "times": time_summary(times),
                    "variables": {
                        ent.value: list(exo.variable_names(ent)) for ent in VARIABLE_ENTITIES
                    },
                    "ids": {
                        "element_blocks": exo.element_block_ids().tolist(),
                        "edge_blocks": exo.edge_block_ids().tolist(),
                        "face_blocks": exo.face_block_ids().tolist(),
                        "node_sets": exo.node_set_ids().tolist(),
                        "side_sets": exo.side_set_ids().tolist(),
                        "edge_sets": exo.edge_set_ids().tolist(),
                        "face_sets": exo.face_set_ids().tolist(),
                        "element_sets": exo.element_set_ids().tolist(),
                    },
                    "agent_hints": agent_hints(),
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
    """Standalone entry point for the ``inspect`` subcommand."""
    parser = argparse.ArgumentParser(prog="python -m exodusii inspect", description=Inspect.__doc__)
    Inspect.setup_parser(parser)
    args = parser.parse_args(argv)
    return Inspect().execute(parser, args)


if __name__ == "__main__":
    raise SystemExit(main())
