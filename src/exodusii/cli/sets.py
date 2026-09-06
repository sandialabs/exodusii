# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""``python -m exodusii sets`` — set metadata."""

import argparse
from typing import Any

from exodusii.api.file import ExodusFile
from exodusii.cli._common import SET_ENTITIES
from exodusii.cli._common import limited_array_payload
from exodusii.cli._common import plural_key
from exodusii.cli._common import set_payload

COMMAND = "sets"

__all__ = ["COMMAND", "add_subparser", "sets_command"]


def add_subparser(
    subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    """Register the ``sets`` subparser."""
    parser = subparsers.add_parser(COMMAND, parents=[common], help="Print set metadata.")
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
    return parser


def sets_command(args: argparse.Namespace) -> dict[str, Any]:
    """Return set metadata."""
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
                    item["entries"] = limited_array_payload(set_info.entries, limit=args.limit)
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

        return payload
