# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""``python -m exodusii stats`` — compact numeric summaries for variables."""

import argparse
from typing import Any

from exodusii.api.file import ExodusFile
from exodusii.cli._common import _piece_path
from exodusii.cli._common import parse_time_selector
from exodusii.cli._common import resolved_time_payload
from exodusii.cli._common import variable_stats_payload
from exodusii.core.entities import Entity
from exodusii.core.entities import entity
from exodusii.core.selectors import parse_variable_selectors

COMMAND = "stats"

__all__ = ["COMMAND", "add_subparser", "stats_command"]


def add_subparser(
    subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    """Register the ``stats`` subparser."""
    parser = subparsers.add_parser(
        COMMAND, parents=[common], help="Print compact numeric summaries for selected variables."
    )
    parser.add_argument("file", help="Exodus database path.")
    parser.add_argument(
        "--select",
        action="append",
        required=True,
        help="Variable selector, e.g. g/TOTAL_ENERGY, n/TEMP, e/ENERGY.",
    )
    parser.add_argument(
        "--time",
        default="last",
        help=(
            "Time selector. Use first, last, a physical time like 0.25, "
            "index:N for zero-based index, or step:N for one-based Exodus step. "
            "Default is last."
        ),
    )
    parser.add_argument(
        "--by-block",
        action="store_true",
        help="For element/edge/face variables, also report statistics by block.",
    )
    parser.add_argument(
        "--by-set", action="store_true", help="For set variables, also report statistics by set."
    )
    parser.add_argument(
        "--piece",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Open only the Nth file as a single-piece reader instead of aggregating. "
            "Useful for reading global scalars from one small decomposed component "
            "without opening the full joined file. Zero-based index into the file argument."
        ),
    )
    return parser


def stats_command(args: argparse.Namespace) -> dict[str, Any]:
    """Return compact variable statistics."""
    selectors = parse_variable_selectors(args.select, require_same_entity=True)
    if not selectors:
        raise ValueError("at least one selector is required")

    location = entity(selectors[0].entity)
    time_selector = parse_time_selector(args.time)
    file_path = _piece_path(args.file, getattr(args, "piece", None))

    with ExodusFile.open(file_path) as exo:
        payload: dict[str, Any] = {
            "command": "stats",
            "file": str(args.file),
            "entity": location.value,
            "variables": {},
        }

        if getattr(args, "piece", None) is not None:
            payload["piece"] = int(args.piece)

        if location is not Entity.GLOBAL or args.time is not None:
            payload["time"] = resolved_time_payload(exo, time_selector, requested=args.time)

        for selector in selectors:
            payload["variables"][selector.name] = variable_stats_payload(
                exo,
                selector,
                time=time_selector,
                by_block=bool(args.by_block),
                by_set=bool(args.by_set),
            )

        return payload
