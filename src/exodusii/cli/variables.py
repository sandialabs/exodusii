# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""``python -m exodusii variables`` — result-variable inventory."""

import argparse
from typing import Any

from exodusii.api.file import ExodusFile
from exodusii.cli._common import VARIABLE_ENTITIES
from exodusii.cli._common import variable_block_location
from exodusii.core.entities import Entity

COMMAND = "variables"

__all__ = ["COMMAND", "add_subparser", "variables_command"]


def add_subparser(
    subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    """Register the ``variables`` subparser."""
    parser = subparsers.add_parser(
        COMMAND, parents=[common], help="Print result-variable inventory."
    )
    parser.add_argument("file", help="Exodus database path.")
    parser.add_argument(
        "--truth-tables", action="store_true", help="Include block/set variable truth tables."
    )
    return parser


def variables_command(args: argparse.Namespace) -> dict[str, Any]:
    """Return variable inventory."""
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

        return {
            "command": "variables",
            "file": str(args.file),
            "variables": variables,
            "agent_hints": {
                "selector_format": "ENTITY/NAME",
                "examples": ["g/TOTAL_ENERGY", "n/TEMP", "e/ENERGY"],
            },
        }
