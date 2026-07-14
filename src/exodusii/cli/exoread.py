# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Command-line Exodus reader."""

import argparse
import sys
from typing import Any
from typing import TextIO

from exodusii.api.file import ExodusFile
from exodusii.api.lineout import Lineout
from exodusii.api.query import print_query


class Namespace(argparse.Namespace):
    """Argument namespace that accumulates variable selectors."""

    def __init__(self, **kwargs: Any) -> None:
        self.variables: list[str] = []
        super().__init__(**kwargs)

    def __setattr__(self, attr: str, value):
        if value and attr in {"globalvar", "element", "face", "edge", "node"}:
            prefix = "d" if attr == "edge" else attr[0]
            self.variables.append(f"{prefix}/{value[-1]}")
        super().__setattr__(attr, value)


def main(argv: list[str] | None = None, file: TextIO | None = None) -> int:
    """Run the ``exoread`` command."""

    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv, namespace=Namespace())

    with ExodusFile.open(args.file) as exo:
        if args.variables:
            print_query(
                exo,
                *args.variables,
                time=_selected_time(args),
                lineout=args.lineout,
                object_index=args.object_index,
                labels=not args.nolabels,
                file=file,
            )
        else:
            describe(exo, file=file)

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
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
    time.add_argument("-t", "--time", type=float, help="Output the variable at the closest time.")
    time.add_argument(
        "-i",
        "--index",
        type=int,
        help="Output the variable at this zero-based time index. -1 means last.",
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
    parser.add_argument("file", help="The ExodusII database file.")

    return parser


def describe(exo: ExodusFile, file: TextIO | None = None) -> None:
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
        stream.write(" Ids = " + " ".join(str(value) for value in exo.element_block_ids()))
    stream.write("\n")
    stream.write(f"Node sets: {exo.node_set_count}")
    if exo.node_set_count:
        stream.write(" Ids = " + " ".join(str(value) for value in exo.node_set_ids()))
    stream.write("\n")
    stream.write(f"Side sets: {exo.side_set_count}")
    if exo.side_set_count:
        stream.write(" Ids = " + " ".join(str(value) for value in exo.side_set_ids()))
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


def _selected_time(args: argparse.Namespace):
    if args.time is not None:
        return args.time
    if args.index is not None:
        return args.index
    return None


if __name__ == "__main__":
    raise SystemExit(main())
