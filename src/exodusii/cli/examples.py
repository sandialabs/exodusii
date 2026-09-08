# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""``python -m exodusii examples`` — database-specific Python usage examples."""

import argparse
import sys
from typing import Any
from typing import TextIO

from exodusii.api.file import ExodusFile
from exodusii.cli._command import Command
from exodusii.cli._common import agent_hints
from exodusii.core.entities import Entity

__all__ = ["Examples"]


class Examples(Command):
    """Print database-specific Python usage examples."""

    name = "examples"

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
        """Return database-specific Python examples as JSON."""
        try:
            path = str(args.file)
            with ExodusFile.open(args.file) as exo:
                examples: list[dict[str, str]] = [
                    {
                        "description": "Open the database and print a summary.",
                        "code": (
                            "from exodusii.api.file import ExodusFile\n\n"
                            f"with ExodusFile.open({path!r}) as exo:\n"
                            "    print(exo.title)\n"
                            "    print(exo.dimension)\n"
                            "    print(exo.node_count, exo.element_count)\n"
                            "    print(exo.times())\n"
                        ),
                    },
                    {
                        "description": "List available variables by entity.",
                        "code": (
                            "from exodusii.api.file import ExodusFile\n"
                            "from exodusii.core.entities import Entity\n\n"
                            f"with ExodusFile.open({path!r}) as exo:\n"
                            "    for ent in (Entity.GLOBAL, Entity.NODE, Entity.ELEMENT):\n"
                            "        print(ent.value, exo.variable_names(ent))\n"
                        ),
                    },
                ]

                node_names = exo.variable_names(Entity.NODE)
                if node_names:
                    name = node_names[0]
                    examples.append(
                        {
                            "description": f"Read nodal variable {name!r} at the last time step.",
                            "code": (
                                "from exodusii.api.file import ExodusFile\n\n"
                                f"with ExodusFile.open({path!r}) as exo:\n"
                                f"    values = exo.values({name!r}, on='node', time='last')\n"
                                "    print(values.shape)\n"
                                "    print(values[:10])\n"
                            ),
                        }
                    )

                element_names = exo.variable_names(Entity.ELEMENT)
                block_ids = exo.element_block_ids()
                if element_names and len(block_ids):
                    name = element_names[0]
                    block_id = int(block_ids[0])
                    examples.append(
                        {
                            "description": (
                                f"Read element variable {name!r} on block {block_id} "
                                "at the last time step."
                            ),
                            "code": (
                                "from exodusii.api.file import ExodusFile\n\n"
                                f"with ExodusFile.open({path!r}) as exo:\n"
                                f"    values = exo.values({name!r}, on='element', "
                                f"block_id={block_id}, time='last')\n"
                                "    print(values.shape)\n"
                                "    print(values[:10])\n"
                            ),
                        }
                    )

                global_names = exo.variable_names(Entity.GLOBAL)
                if global_names:
                    name = global_names[0]
                    examples.append(
                        {
                            "description": (
                                f"Read complete time history of global variable {name!r}."
                            ),
                            "code": (
                                "from exodusii.api.file import ExodusFile\n\n"
                                f"with ExodusFile.open({path!r}) as exo:\n"
                                "    times = exo.times()\n"
                                f"    values = exo.values({name!r}, on='global')\n"
                                "    for time, value in zip(times, values, strict=True):\n"
                                "        print(time, value)\n"
                            ),
                        }
                    )

            payload: dict[str, Any] = {
                "command": "examples",
                "file": path,
                "examples": examples,
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
    """Standalone entry point for the ``examples`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="python -m exodusii examples", description=Examples.__doc__
    )
    Examples.setup_parser(parser)
    args = parser.parse_args(argv)
    return Examples().execute(parser, args)


if __name__ == "__main__":
    raise SystemExit(main())
