# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""JSON-oriented Exodus command-line interface for agents.

This module is the coordinator for the ``python -m exodusii`` CLI.  It builds
the top-level parser by iterating a registry of :class:`~exodusii.cli._command.Command`
subclasses, asking each to register its own arguments, then dispatches to the
matching instance.

Most subcommands return a JSON-serializable ``dict`` emitted via
:meth:`~exodusii.cli._command.Command.emit_json`.  The ``diff`` subcommand is
special: it manages its own output (text or JSON) and SEACAS-style exit codes
(``0`` same / ``1`` error / ``2`` different).

The module intentionally favors stable, machine-readable JSON over compact
human-readable tables.  The ``exoread`` entry point remains available for
table-style output.
"""

import argparse
from typing import TextIO

from exodusii.cli._command import Command
from exodusii.cli.blocks import Blocks
from exodusii.cli.examples import Examples
from exodusii.cli.exodiff import Diff
from exodusii.cli.exoread import Read
from exodusii.cli.inspect import Inspect
from exodusii.cli.learn import Learn
from exodusii.cli.query import Query
from exodusii.cli.region_stats import RegionStats
from exodusii.cli.sets import Sets
from exodusii.cli.stats import Stats
from exodusii.cli.times import Times
from exodusii.cli.tracers import Tracers
from exodusii.cli.variables import Variables

__all__ = ["main"]

# Ordered list of Command subclasses.  Order controls help / subcommand listing.
_COMMANDS: list[type[Command]] = [
    Inspect,
    Variables,
    Blocks,
    Sets,
    Times,
    Query,
    Stats,
    RegionStats,
    Examples,
    Tracers,
    Learn,
    Read,
    Diff,
]


def main(argv: list[str] | None = None, *, file: TextIO | None = None) -> int:
    """Run the JSON-oriented ``python -m exodusii`` CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m exodusii", description="Machine-readable JSON tools for ExodusII databases."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    registry: dict[str, Command] = {}
    for cls in _COMMANDS:
        cmd = cls()
        name = cls.name or cls.__name__.lower()
        sub = subparsers.add_parser(name, help=cls.__doc__.splitlines()[0] if cls.__doc__ else None)
        cls.setup_parser(sub)
        registry[name] = cmd

    args = parser.parse_args(argv)
    return registry[args.command].execute(parser, args, file=file)
