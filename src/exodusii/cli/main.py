# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""JSON-oriented Exodus command-line interface for agents.

This module is the coordinator for the ``python -m exodusii`` CLI.  It builds
the top-level parser by asking each subcommand module (one per file under
:mod:`exodusii.cli`) to register its own subparser, then dispatches to the
matching handler.  Most subcommands return a JSON-serializable ``dict`` that is
emitted via :func:`exodusii.cli._common.emit_json`.

The ``diff`` subcommand is special: it is a full ``exodiff``-style comparison
that manages its own output (text or JSON) and SEACAS-style exit codes
(``0`` same / ``1`` error / ``2`` different), so it is dispatched directly to
:func:`exodusii.cli.exodiff.run_diff` instead of going through the JSON
envelope.

The module intentionally favors stable, machine-readable JSON over compact
human-readable tables.  The ``exoread`` entry point remains available for
table-style output.
"""

import argparse
import sys
from collections.abc import Callable
from typing import Any
from typing import TextIO

from exodusii.cli import blocks as _blocks
from exodusii.cli import examples as _examples
from exodusii.cli import exodiff as _exodiff
from exodusii.cli import inspect as _inspect
from exodusii.cli import learn as _learn
from exodusii.cli import query as _query
from exodusii.cli import region_stats as _region_stats
from exodusii.cli import sets as _sets
from exodusii.cli import stats as _stats
from exodusii.cli import times as _times
from exodusii.cli import tracers as _tracers
from exodusii.cli import variables as _variables
from exodusii.cli._common import emit_json
from exodusii.cli._common import make_common_parser

__all__ = ["build_parser", "dispatch", "main"]

# Subcommand modules whose handlers return a JSON-serializable dict.  Order
# controls help/subcommand listing order and matches the historical layout.
_JSON_COMMANDS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    _inspect.COMMAND: _inspect.inspect_command,
    _variables.COMMAND: _variables.variables_command,
    _blocks.COMMAND: _blocks.blocks_command,
    _sets.COMMAND: _sets.sets_command,
    _times.COMMAND: _times.times_command,
    _query.COMMAND: _query.query_command,
    _stats.COMMAND: _stats.stats_command,
    _region_stats.COMMAND: _region_stats.region_stats_command,
    _examples.COMMAND: _examples.examples_command,
    _tracers.COMMAND: _tracers.tracers_command,
    _learn.COMMAND: _learn.learn_command,
}

# Modules that register a subparser, in listing order.  ``exodiff`` (the
# ``diff`` subcommand) is registered last and dispatched specially.
_SUBPARSER_MODULES = (
    _inspect,
    _variables,
    _blocks,
    _sets,
    _times,
    _query,
    _stats,
    _region_stats,
    _examples,
    _tracers,
    _learn,
    _exodiff,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser from every subcommand module."""
    parser = argparse.ArgumentParser(
        prog="python -m exodusii", description="Machine-readable JSON tools for ExodusII databases."
    )

    # Put --terse on the top-level parser and every subparser so both of these
    # work:
    #
    #   python -m exodusii --terse inspect file.exo
    #   python -m exodusii inspect file.exo --terse
    common = make_common_parser()
    parser.add_argument(
        "--terse",
        action="store_true",
        default=False,
        help="Emit compact JSON with no extra whitespace. Default is indented JSON.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    for module in _SUBPARSER_MODULES:
        module.add_subparser(subparsers, common)

    return parser


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    """Dispatch parsed CLI arguments to a JSON-returning handler.

    Note: the ``diff`` subcommand is *not* handled here; it is dispatched in
    :func:`main` because it manages its own output and exit code.
    """
    handler = _JSON_COMMANDS.get(args.command)
    if handler is None:
        raise ValueError(f"unknown command {args.command!r}")
    return handler(args)


def main(argv: list[str] | None = None, *, file: TextIO | None = None) -> int:
    """Run the JSON-oriented ``python -m exodusii`` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # The diff subcommand is a full exodiff comparison with its own output
    # format(s) and SEACAS-style exit codes; hand it off directly.
    if args.command == _exodiff.COMMAND:
        return _exodiff.run_diff(args, file=file)

    try:
        payload = dispatch(args)
        payload.setdefault("ok", True)
        emit_json(payload, terse=bool(args.terse), file=file)
        return 0
    except Exception as exc:
        emit_json(
            {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}},
            terse=bool(getattr(args, "terse", False)),
            file=file or sys.stderr,
        )
        return 1
