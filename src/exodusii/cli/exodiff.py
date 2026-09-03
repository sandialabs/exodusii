# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Command-line interface for exodiff-style comparison.

Provides ``exodiff``-like behavior on top of :mod:`exodusii.api.diff`.
Supports matched mesh ordering (no coordinate-based mesh matching) and offers
both a human-readable and a JSON output mode.  Time-step selection and linear
interpolation (Phase 2) are fully supported.

Return codes follow the SEACAS ``exodiff`` convention:

* ``0`` -- files are the same
* ``1`` -- an error occurred (I/O, structural mismatch)
* ``2`` -- files differ
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import TextIO

from exodusii.api.diff import DiffOptions
from exodusii.api.diff import DiffResult
from exodusii.api.diff import TimeSelection
from exodusii.api.diff import diff
from exodusii.core.tolerance import Tolerance
from exodusii.core.tolerance import ToleranceMode

__all__ = ["build_parser", "main"]

_SAME = 0
_ERROR = 1
_DIFFERENT = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the ``exodiff`` argument parser."""

    parser = argparse.ArgumentParser(
        prog="exodiff",
        description=(
            "Compare two ExodusII databases (matched mesh ordering). "
            "A pure-Python, exodusii-based counterpart to SEACAS exodiff."
        ),
    )
    parser.add_argument("file1", help="First Exodus database path.")
    parser.add_argument("file2", help="Second Exodus database path.")

    tol = parser.add_argument_group("tolerance")
    tol.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=1.0e-6,
        metavar="VALUE",
        help="Default tolerance value (default: 1e-6).",
    )
    mode = tol.add_mutually_exclusive_group()
    mode.add_argument(
        "--relative",
        dest="mode",
        action="store_const",
        const="relative",
        help="Use relative tolerance (default).",
    )
    mode.add_argument(
        "--absolute",
        dest="mode",
        action="store_const",
        const="absolute",
        help="Use absolute tolerance.",
    )
    mode.add_argument(
        "--combined",
        dest="mode",
        action="store_const",
        const="combined",
        help="Use combined tolerance.",
    )
    mode.add_argument(
        "--eigen-relative",
        dest="mode",
        action="store_const",
        const="eigenrel",
        help="Use eigen relative tolerance.",
    )
    mode.add_argument(
        "--eigen-absolute",
        dest="mode",
        action="store_const",
        const="eigenabs",
        help="Use eigen absolute tolerance.",
    )
    mode.add_argument(
        "--eigen-combined",
        dest="mode",
        action="store_const",
        const="eigencom",
        help="Use eigen combined tolerance.",
    )
    mode.add_argument(
        "--ulps-float",
        dest="mode",
        action="store_const",
        const="ulps_float",
        help="Use single-precision ULPs tolerance.",
    )
    mode.add_argument(
        "--ulps-double",
        dest="mode",
        action="store_const",
        const="ulps_double",
        help="Use double-precision ULPs tolerance.",
    )
    tol.add_argument(
        "--floor",
        type=float,
        default=0.0,
        metavar="VALUE",
        help="Floor below which values are treated as equal (default: 0).",
    )
    tol.add_argument(
        "--use-old-floor", action="store_true", help="Use the older floor definition |a-b| < floor."
    )
    tol.add_argument(
        "--coordinate-tolerance",
        type=float,
        default=1.0e-6,
        metavar="VALUE",
        help="Absolute tolerance for nodal coordinates (default: 1e-6).",
    )

    sel = parser.add_argument_group("selection")
    sel.add_argument(
        "-x",
        "--exclude",
        action="append",
        default=[],
        metavar="NAME",
        help="Exclude a variable by name (repeatable).",
    )
    sel.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Match variable names case-sensitively (default: case-insensitive).",
    )
    sel.add_argument(
        "--no-coordinates", action="store_true", help="Do not compare nodal coordinates."
    )
    sel.add_argument(
        "--no-attributes", action="store_true", help="Do not compare block attributes."
    )

    ts = parser.add_argument_group("time-step selection")
    ts.add_argument(
        "-T",
        "--time-step-offset",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Offset: file1_step = file2_step + N.  "
            "Ignored when --start/--stop/--increment are also specified."
        ),
    )
    ts.add_argument(
        "--start",
        type=str,
        default=None,
        metavar="N|LAST",
        help=(
            "First file-2 step to compare (1-based, default 1).  "
            "Use 'LAST' to compare only the final step on each file."
        ),
    )
    ts.add_argument(
        "--stop",
        type=int,
        default=-1,
        metavar="N",
        help="Last file-2 step to compare, inclusive (default: all).",
    )
    ts.add_argument(
        "--increment",
        type=int,
        default=1,
        metavar="N",
        help="Step stride (default 1).",
    )
    ts.add_argument(
        "--exclude-steps",
        type=str,
        default=None,
        metavar="N[,N...]",
        help="Comma-separated 1-based file-2 step numbers to skip.",
    )
    ts.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        metavar="S",
        help="Multiply file-1 time values by S before matching (default 1.0).",
    )
    ts.add_argument(
        "--time-offset",
        type=float,
        default=0.0,
        metavar="O",
        help="Add O to file-1 time values before matching (default 0.0).",
    )
    ts.add_argument(
        "--interpolate",
        action="store_true",
        help=(
            "Interpolate file-2 values to each file-1 time.  "
            "Steps whose file-1 time is outside the file-2 time range are skipped."
        ),
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--format", choices=("text", "json"), default="text", help="Output format (default: text)."
    )
    out.add_argument("--terse", action="store_true", help="For JSON output, emit compact JSON.")
    out.add_argument(
        "--show-all",
        action="store_true",
        help="Report every compared variable, not only those exceeding tolerance.",
    )
    out.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress the per-variable difference listing (text mode).",
    )

    return parser


def _parse_start(value: str | None) -> int:
    """Return the 1-based start step or -1 for LAST."""
    if value is None:
        return 1
    if value.strip().upper() == "LAST":
        return -1
    return int(value)


def _parse_exclude_steps(value: str | None) -> frozenset[int]:
    if not value:
        return frozenset()
    return frozenset(int(s.strip()) for s in value.split(",") if s.strip())


def _options_from_args(args: argparse.Namespace) -> DiffOptions:
    mode = ToleranceMode.parse(args.mode) if args.mode else ToleranceMode.RELATIVE
    default_tol = Tolerance(
        mode=mode,
        value=float(args.tolerance),
        floor=float(args.floor),
        use_old_floor=bool(args.use_old_floor),
    )
    coord_tol = Tolerance(
        mode=ToleranceMode.ABSOLUTE,
        value=float(args.coordinate_tolerance),
        floor=0.0,
        use_old_floor=bool(args.use_old_floor),
    )
    start = _parse_start(args.start)
    exclude_steps = _parse_exclude_steps(args.exclude_steps)
    time_sel = TimeSelection(
        start=start,
        stop=int(args.stop),
        increment=int(args.increment),
        time_step_offset=int(args.time_step_offset),
        exclude_steps=exclude_steps,
        time_value_scale=float(args.time_scale),
        time_value_offset=float(args.time_offset),
        interpolating=bool(args.interpolate),
    )
    return DiffOptions(
        default_tolerance=default_tol,
        coordinate_tolerance=coord_tol,
        exclude=frozenset(args.exclude),
        ignore_case=not args.case_sensitive,
        time_selection=time_sel,
        compare_coordinates=not args.no_coordinates,
        compare_attributes=not args.no_attributes,
        show_all=bool(args.show_all),
    )


def _result_to_json(result: DiffResult) -> dict[str, object]:
    return {
        "same": result.same,
        "file1": result.file1,
        "file2": result.file2,
        "errors": list(result.errors),
        "warnings": list(result.warnings),
        "coordinate_max_delta": result.coordinate_max_delta,
        "variable_diffs": [
            {
                "entity": vd.entity,
                "name": vd.name,
                "max_delta": vd.max_delta,
                "tolerance_mode": vd.tolerance_mode,
                "exceeded": vd.exceeded,
                "time_index": vd.time_index,
                "entry_index": vd.entry_index,
                "block_id": vd.block_id,
                "set_id": vd.set_id,
                "value1": vd.value1,
                "value2": vd.value2,
            }
            for vd in result.variable_diffs
        ],
    }


def _print_text(result: DiffResult, args: argparse.Namespace, out: TextIO) -> None:
    for warning in result.warnings:
        print(f"WARNING: {warning}", file=out)
    for error in result.errors:
        print(f"ERROR: {error}", file=out)

    if not args.quiet:
        exceeded = [vd for vd in result.variable_diffs if vd.exceeded or args.show_all]
        for vd in exceeded:
            loc = []
            if vd.block_id is not None:
                loc.append(f"block {vd.block_id}")
            if vd.set_id is not None:
                loc.append(f"set {vd.set_id}")
            if vd.time_index is not None:
                loc.append(f"step {vd.time_index}")
            if vd.entry_index is not None:
                loc.append(f"entry {vd.entry_index}")
            where = f" ({', '.join(loc)})" if loc else ""
            flag = "" if vd.exceeded else " [within tol]"
            print(
                f"{vd.entity} {vd.name}: max {vd.tolerance_mode} diff ="
                f" {vd.max_delta:.6e}{where}{flag}",
                file=out,
            )

    if result.same:
        print("exodiff: Files are the same", file=out)
    else:
        print("exodiff: Files are different", file=out)


def main(argv: list[str] | None = None, *, file: TextIO | None = None) -> int:
    """Run the ``exodiff`` CLI and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    out = file or sys.stdout

    try:
        options = _options_from_args(args)
        result = diff(args.file1, args.file2, options)
    except Exception as exc:
        if args.format == "json":
            payload = {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
            json.dump(payload, out, indent=None if args.terse else 2)
            out.write("\n")
        else:
            print(f"exodiff: error: {exc}", file=sys.stderr)
        return _ERROR

    if args.format == "json":
        payload = {"ok": True, **_result_to_json(result)}
        json.dump(payload, out, indent=None if args.terse else 2)
        out.write("\n")
    else:
        _print_text(result, args, out)

    if result.errors:
        return _ERROR
    return _SAME if result.same else _DIFFERENT


if __name__ == "__main__":
    raise SystemExit(main())
