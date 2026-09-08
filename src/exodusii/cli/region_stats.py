# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""``python -m exodusii region-stats`` — statistics inside a geometric region."""

import argparse
import sys
from typing import Any
from typing import TextIO

from exodusii.api.file import ExodusFile
from exodusii.cli._command import Command
from exodusii.core.entities import entity
from exodusii.core.selectors import parse_variable_selectors

__all__ = ["RegionStats"]


class RegionStats(Command):
    """Compute statistics of a variable inside a geometric region."""

    name = "region-stats"

    @staticmethod
    def setup_parser(parser: argparse.ArgumentParser) -> None:
        """Register arguments on *parser*."""
        parser.add_argument("file", help="Exodus database path.")
        parser.add_argument(
            "--select",
            required=True,
            metavar="ENTITY/NAME",
            help="Variable selector, e.g. e/YIELD_STRESS_2.",
        )
        parser.add_argument(
            "--block",
            type=int,
            default=None,
            metavar="BLOCK_ID",
            help="Restrict to a single element block ID. Mutually exclusive with --blocks.",
        )
        parser.add_argument(
            "--blocks",
            default=None,
            metavar="MODE",
            help=(
                "Multi-block mode. 'auto' selects only non-empty blocks that define the "
                "requested variable (natural target-material selection). 'all' or omitting "
                "this flag selects all non-empty blocks. Mutually exclusive with --block."
            ),
        )
        parser.add_argument(
            "--time",
            default=None,
            help="Time selector. Use first, last, a physical time, index:N, or step:N. "
            "[default: last]",
        )
        parser.add_argument(
            "--reduce",
            default="mean,max,min,count",
            help=(
                "Comma-separated list of reducers: mean, max, min, sum, count, std. "
                "[default: mean,max,min,count]"
            ),
        )
        parser.add_argument(
            "--where",
            default=None,
            metavar="EXPR",
            help="Field predicate, e.g. 'EQPS_2 > 1.0'. AND-ed with the region mask.",
        )
        parser.add_argument(
            "--symmetry",
            type=float,
            default=1.0,
            metavar="FACTOR",
            help=(
                "Symmetry factor applied to extensive reducers (sum, count). "
                "Use 4.0 for a quarter-symmetry model. [default: 1.0]"
            ),
        )
        region_group = parser.add_mutually_exclusive_group(required=True)
        region_group.add_argument(
            "--cylinder",
            nargs=7,
            type=float,
            metavar=("AX", "AY", "AZ", "BX", "BY", "BZ", "R"),
            help="Cylinder from axis-point A to axis-point B with radius R (3-D).",
        )
        region_group.add_argument(
            "--sphere",
            nargs=4,
            type=float,
            metavar=("CX", "CY", "CZ", "R"),
            help="Sphere centred at (CX, CY, CZ) with radius R (3-D).",
        )
        region_group.add_argument(
            "--circle",
            nargs=3,
            type=float,
            metavar=("CX", "CY", "R"),
            help="Circle centred at (CX, CY) with radius R (2-D).",
        )
        region_group.add_argument(
            "--rectangle",
            nargs=4,
            type=float,
            metavar=("OX", "OY", "W", "H"),
            help="Axis-aligned rectangle with origin (OX, OY), width W, height H (2-D).",
        )
        Command.add_terse_argument(parser)

    def execute(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        *,
        file: TextIO | None = None,
    ) -> int:
        """Compute region statistics and return JSON."""
        try:
            from exodusii.mesh.regions import Circle
            from exodusii.mesh.regions import Cylinder
            from exodusii.mesh.regions import Rectangle
            from exodusii.mesh.regions import Sphere

            if args.cylinder is not None:
                ax, ay, az, bx, by, bz, r = args.cylinder
                region_obj = Cylinder([ax, ay, az], [bx, by, bz], r)
                region_desc = {
                    "type": "cylinder",
                    "p1": [ax, ay, az],
                    "p2": [bx, by, bz],
                    "radius": r,
                }
            elif args.sphere is not None:
                cx, cy, cz, r = args.sphere
                region_obj = Sphere([cx, cy, cz], r)
                region_desc = {"type": "sphere", "center": [cx, cy, cz], "radius": r}
            elif args.circle is not None:
                cx, cy, r = args.circle
                region_obj = Circle([cx, cy], r)
                region_desc = {"type": "circle", "center": [cx, cy], "radius": r}
            elif args.rectangle is not None:
                ox, oy, w, h = args.rectangle
                region_obj = Rectangle([ox, oy], w, h)
                region_desc = {"type": "rectangle", "origin": [ox, oy], "width": w, "height": h}
            else:
                raise ValueError(
                    "a region type must be specified "
                    "(--cylinder, --sphere, --circle, or --rectangle)"
                )

            selectors = parse_variable_selectors([args.select], require_same_entity=False)
            if not selectors:
                raise ValueError("--select requires a valid variable selector")
            selector = selectors[0]
            var_entity = str(entity(selector.entity))
            var_name = selector.name
            reduce_list = [r.strip() for r in args.reduce.split(",") if r.strip()]
            time_selector = self.parse_time_selector(args.time) if args.time is not None else None

            with ExodusFile.open(args.file) as exo:
                result = exo.region_stats(
                    var_name,
                    on=var_entity,
                    block_id=args.block,
                    blocks=getattr(args, "blocks", None),
                    region=region_obj,
                    where=args.where,
                    reduce=reduce_list,
                    time=time_selector,
                    symmetry_factor=args.symmetry,
                )

            payload: dict[str, Any] = {
                "command": "region-stats",
                "file": str(args.file),
                "variable": result.variable,
                "entity": result.entity,
                "block_id": result.block_id,
                "blocks_used": (
                    list(result.blocks_used) if result.blocks_used is not None else None
                ),
                "region": region_desc,
                "where": args.where,
                "time": {"index": result.time_index, "value": result.time_value},
                "count_total": result.count_total,
                "count_selected": result.count_selected,
                "symmetry_factor": result.symmetry_factor,
                "stats": result.stats,
            }
            payload.setdefault("ok", True)
            self.emit_json(payload, terse=bool(getattr(args, "terse", False)), file=file)
            return 0
        except Exception as exc:
            self.emit_json(
                {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}},
                terse=bool(getattr(args, "terse", False)),
                file=file or sys.stderr,
            )
            return 1


def main(argv: list[str] | None = None, *, file=None) -> int:
    """Standalone entry point for the ``region-stats`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="python -m exodusii region-stats", description=RegionStats.__doc__
    )
    RegionStats.setup_parser(parser)
    args = parser.parse_args(argv)
    return RegionStats().execute(parser, args)


if __name__ == "__main__":
    raise SystemExit(main())
