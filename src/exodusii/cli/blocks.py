# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""``python -m exodusii blocks`` — block metadata."""

import argparse
from typing import Any

from exodusii.api.file import ExodusFile
from exodusii.cli._common import BLOCK_ENTITIES
from exodusii.cli._common import block_payload
from exodusii.cli._common import limited_array_payload
from exodusii.cli._common import plural_key

COMMAND = "blocks"

__all__ = ["COMMAND", "add_subparser", "blocks_command"]


def add_subparser(
    subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    """Register the ``blocks`` subparser."""
    parser = subparsers.add_parser(COMMAND, parents=[common], help="Print block metadata.")
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
    return parser


def blocks_command(args: argparse.Namespace) -> dict[str, Any]:
    """Return block metadata."""
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

        return payload
