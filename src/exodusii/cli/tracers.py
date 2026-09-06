# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""``python -m exodusii tracers`` — query tracer variables by tracer ID."""

import argparse
from typing import Any

from exodusii.api.file import ExodusFile
from exodusii.cli._common import jsonable
from exodusii.cli._common import parse_time_selector
from exodusii.core.selectors import parse_variable_selectors

COMMAND = "tracers"

__all__ = ["COMMAND", "add_subparser", "tracers_command"]


def add_subparser(
    subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    """Register the ``tracers`` subparser."""
    parser = subparsers.add_parser(
        COMMAND, parents=[common], help="Query tracer variables by tracer ID."
    )
    parser.add_argument("file", help="Exodus database path.")
    parser.add_argument(
        "--select", required=True, metavar="n/NAME", help="Nodal variable selector, e.g. n/VELX."
    )
    parser.add_argument(
        "--ids",
        default=None,
        metavar="ID[,ID...]",
        help="Comma-separated tracer IDs to return. Default: all tracers.",
    )
    parser.add_argument(
        "--time",
        default="last",
        help=(
            "Time selector. Use first, last, a physical time, index:N, or step:N. Default: last."
        ),
    )
    parser.add_argument(
        "--id-variable",
        default="ID",
        metavar="NAME",
        help="Name of the nodal variable holding tracer IDs. Default: ID.",
    )
    return parser


def tracers_command(args: argparse.Namespace) -> dict[str, Any]:
    """Return tracer variable values keyed by tracer ID."""
    # Parse selector
    selectors = parse_variable_selectors([args.select], require_same_entity=False)
    if not selectors:
        raise ValueError("--select requires a valid variable selector")
    selector = selectors[0]
    var_name = selector.name

    # Parse IDs
    ids: list[int] | None = None
    if args.ids is not None:
        ids = [int(x.strip()) for x in args.ids.split(",") if x.strip()]

    time_selector = parse_time_selector(args.time)
    id_var = getattr(args, "id_variable", "ID")

    with ExodusFile.open(args.file) as exo:
        result = exo.tracer(var_name, ids=ids, time=time_selector, id_variable=id_var)
        all_ids = exo.tracer_ids(id_variable=id_var).tolist()

    # Serialise: ndarray values → lists for JSON
    data = {str(tid): jsonable(v) for tid, v in result.items()}

    return {
        "command": "tracers",
        "file": str(args.file),
        "variable": var_name,
        "id_variable": id_var,
        "all_ids": all_ids,
        "requested_ids": ids,
        "time": args.time,
        "data": data,
    }
