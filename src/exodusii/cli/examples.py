# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""``python -m exodusii examples`` — database-specific Python usage examples."""

import argparse
from typing import Any

from exodusii.api.file import ExodusFile
from exodusii.cli._common import agent_hints
from exodusii.core.entities import Entity

COMMAND = "examples"

__all__ = ["COMMAND", "add_subparser", "examples_command"]


def add_subparser(
    subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    """Register the ``examples`` subparser."""
    parser = subparsers.add_parser(
        COMMAND, parents=[common], help="Print database-specific Python usage examples."
    )
    parser.add_argument("file", help="Exodus database path.")
    return parser


def examples_command(args: argparse.Namespace) -> dict[str, Any]:
    """Return database-specific Python examples."""
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
                        f"Read element variable {name!r} on block {block_id} at the last time step."
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
                    "description": f"Read complete time history of global variable {name!r}.",
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

    return {"command": "examples", "file": path, "examples": examples, "agent_hints": agent_hints()}
