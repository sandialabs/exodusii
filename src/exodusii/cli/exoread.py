# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Command-line Exodus reader (table-style output).

Provides the ``exoread`` entry point and the ``python -m exodusii read``
subcommand.  Unlike the JSON-oriented subcommands this command writes
human-readable table output via :func:`exodusii.api.query.print_query`.
"""

import argparse
import sys
from typing import Any
from typing import TextIO

from exodusii.api.file import ExodusFile
from exodusii.api.lineout import Lineout
from exodusii.api.query import print_query
from exodusii.cli._command import Command
from exodusii.core.files import expand_files
from exodusii.core.streams import output_stream

__all__ = ["Read"]


class _AccumulatingNamespace(argparse.Namespace):
    """Argument namespace that accumulates variable selectors.

    When ``-g``, ``-e``, ``-f``, ``-d``, or ``-n`` flags are seen, the
    selector is appended to ``self.variables`` with the appropriate
    entity prefix.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.variables: list[str] = []
        super().__init__(**kwargs)

    def __setattr__(self, attr: str, value: Any) -> None:
        if value and attr in {"globalvar", "element", "face", "edge", "node"}:
            prefix = "d" if attr == "edge" else attr[0]
            self.variables.append(f"{prefix}/{value[-1]}")
        super().__setattr__(attr, value)


class Read(Command):
    """Human-readable Exodus reader (table-style output)."""

    name = "read"

    @staticmethod
    def setup_parser(parser: argparse.ArgumentParser) -> None:
        """Register arguments on *parser*."""
        parser.add_argument("-V", "--version", action="version", version="%(prog)s 0.1")

        variables = parser.add_mutually_exclusive_group()
        variables.add_argument(
            "-g",
            "--global",
            action="append",
            dest="globalvar",
            help="Select a global variable name to extract.",
        )
        variables.add_argument(
            "-e", "--element", action="append", help="Select an element variable name to extract."
        )
        variables.add_argument(
            "-f", "--face", action="append", help="Select a face variable name to extract."
        )
        variables.add_argument(
            "-d", "--edge", action="append", help="Select an edge variable name to extract."
        )
        variables.add_argument(
            "-n", "--node", action="append", help="Select a node variable name to extract."
        )

        time = parser.add_mutually_exclusive_group()
        time.add_argument(
            "-t", "--time", type=float, help="Output the variable at the closest time."
        )
        time.add_argument(
            "-i",
            "--index",
            type=int,
            help="Output the variable at this zero-based time index. -1 means last.",
        )
        time.add_argument(
            "-c",
            "--cycle",
            type=int,
            help="Output the variable at this cycle number (global CYCLE/NSTEPS variable).",
        )

        parser.add_argument(
            "--object-index",
            action="store_true",
            default=False,
            help="For non-global variables, include the object index in the output.",
        )
        parser.add_argument(
            "--nolabels",
            action="store_true",
            default=False,
            help="Do not write variable names to the output.",
        )
        parser.add_argument(
            "-L", "--lineout", type=Lineout.from_cli, help="Restrict spatial output to a lineout."
        )
        parser.add_argument(
            "file", help="The ExodusII database file (glob patterns open a parallel set)."
        )

    def execute(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        *,
        file: TextIO | None = None,
    ) -> int:
        """Run the exoread command."""
        # When dispatched from main.py the namespace is plain (no __setattr__ magic),
        # so accumulate the selectors explicitly from the individual flag values.
        variables: list[str] | None = getattr(args, "variables", None)
        if variables is None:
            variables = []
            for attr, prefix in [
                ("globalvar", "g"),
                ("element", "e"),
                ("face", "f"),
                ("edge", "d"),
                ("node", "n"),
            ]:
                val = getattr(args, attr, None)
                if val:
                    variables.extend(f"{prefix}/{v}" for v in val)

        with _open_database(args.file) as exo:
            if variables:
                print_query(
                    exo,
                    *variables,
                    time=_selected_time(args, exo),
                    lineout=args.lineout,
                    object_index=args.object_index,
                    labels=not args.nolabels,
                    file=file,
                )
            else:
                _describe(exo, file=file)
        return 0


def _open_database(pattern: str) -> Any:
    """Open ``pattern`` as one file or, if it expands to several, a parallel set.

    exodusii < 2 globbed the file argument (``exoread 'run.exo.*'``) and
    opened multiple matches as a decomposed parallel database.
    """

    files = expand_files(pattern)
    if len(files) == 1:
        return ExodusFile.open(files[0])
    from exodusii.api.parallel import ParallelExodusFile

    return ParallelExodusFile.open(*files)


def _selected_time(args: argparse.Namespace, exo: Any = None) -> Any:
    if getattr(args, "cycle", None) is not None:
        from exodusii.compat._time import cycle_time_index

        return cycle_time_index(exo, args.cycle)
    if args.time is not None:
        return args.time
    if args.index is not None:
        return args.index
    return None


def describe(exo: ExodusFile, file: Any = None) -> None:
    """Write a compact database description (public alias for backward compatibility).

    ``file`` may be a stream, a path (opened for writing), or ``None`` for
    ``sys.stdout``.
    """
    with output_stream(file) as stream:
        _describe(exo, file=stream)


def _describe(exo: ExodusFile, file: TextIO | None = None) -> None:
    """Write a compact database description."""
    stream = file or sys.stdout
    stream.write(f"Title: {exo.title}\n")
    stream.write(f"Storage type: {exo.storage_type}\n")
    stream.write(f"Dimension: {exo.dimension}\n")
    stream.write(f"Num nodes   : {exo.node_count}\n")
    stream.write(f"Num edges   : {exo.edge_count}\n")
    stream.write(f"Num faces   : {exo.face_count}\n")
    stream.write(f"Num elements: {exo.element_count}\n")
    stream.write(f"Element blocks: {exo.element_block_count}")
    if exo.element_block_count:
        stream.write(" Ids = " + " ".join(str(v) for v in exo.element_block_ids()))
    stream.write("\n")
    stream.write(f"Node sets: {exo.node_set_count}")
    if exo.node_set_count:
        stream.write(" Ids = " + " ".join(str(v) for v in exo.node_set_ids()))
    stream.write("\n")
    stream.write(f"Side sets: {exo.side_set_count}")
    if exo.side_set_count:
        stream.write(" Ids = " + " ".join(str(v) for v in exo.side_set_ids()))
    stream.write("\n")
    _write_variables(stream, "Global", exo.variable_names("global"))
    _write_variables(stream, "Node", exo.variable_names("node"))
    _write_variables(stream, "Element", exo.variable_names("element"))
    times = exo.times()
    stream.write(f"Time steps: {len(times)}\n")
    for index, value in enumerate(times, start=1):
        stream.write(f"  {index} {value}\n")


def _write_variables(stream: TextIO, label: str, names: tuple[str, ...]) -> None:
    stream.write(f"{label} vars: {len(names)}\n")
    for index, name in enumerate(names):
        stream.write(f"  {index} {name}\n")


def main(argv: list[str] | None = None, *, file: TextIO | None = None) -> int:
    """Standalone entry point for the ``exoread`` command."""
    parser = argparse.ArgumentParser(prog="exoread", description=__doc__)
    Read.setup_parser(parser)
    args = parser.parse_args(
        sys.argv[1:] if argv is None else argv, namespace=_AccumulatingNamespace()
    )
    return Read().execute(parser, args, file=file)


if __name__ == "__main__":
    raise SystemExit(main())
