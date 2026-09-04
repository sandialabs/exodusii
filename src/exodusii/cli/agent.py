# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""JSON-oriented Exodus command-line interface for agents.

This module intentionally favors stable, machine-readable JSON over compact
human-readable tables.  The existing ``exoread`` entry point remains available
for table-style output.
"""

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from typing import TextIO

import numpy as np
import numpy.typing as npt

from exodusii.api.file import ExodusFile
from exodusii.api.query import query
from exodusii.cli.learn import learn_command
from exodusii.core.entities import Entity
from exodusii.core.entities import entity
from exodusii.core.selectors import VariableSelector
from exodusii.core.selectors import parse_variable_selectors
from exodusii.core.time import TimeSelector
from exodusii.core.time import resolve_time

VARIABLE_ENTITIES: tuple[Entity, ...] = (
    Entity.GLOBAL,
    Entity.NODE,
    Entity.ELEMENT,
    Entity.EDGE,
    Entity.FACE,
    Entity.NODE_SET,
    Entity.SIDE_SET,
    Entity.EDGE_SET,
    Entity.FACE_SET,
    Entity.ELEMENT_SET,
)

BLOCK_ENTITIES: tuple[Entity, ...] = (Entity.ELEMENT_BLOCK, Entity.EDGE_BLOCK, Entity.FACE_BLOCK)

SET_ENTITIES: tuple[Entity, ...] = (
    Entity.NODE_SET,
    Entity.SIDE_SET,
    Entity.EDGE_SET,
    Entity.FACE_SET,
    Entity.ELEMENT_SET,
)


def main(argv: list[str] | None = None, *, file: TextIO | None = None) -> int:
    """Run the JSON-oriented ``python -m exodusii`` CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

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


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser."""

    parser = argparse.ArgumentParser(
        prog="python -m exodusii", description="Machine-readable JSON tools for ExodusII databases."
    )

    # Put --terse on the top-level parser and every subparser so both of these
    # work:
    #
    #   python -m exodusii --terse inspect file.exo
    #   python -m exodusii inspect file.exo --terse
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--terse",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit compact JSON with no extra whitespace. Default is indented JSON.",
    )
    parser.add_argument(
        "--terse",
        action="store_true",
        default=False,
        help="Emit compact JSON with no extra whitespace. Default is indented JSON.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", parents=[common], help="Print an agent-friendly database summary."
    )
    inspect_parser.add_argument("file", help="Exodus database path.")

    variables_parser = subparsers.add_parser(
        "variables", parents=[common], help="Print result-variable inventory."
    )
    variables_parser.add_argument("file", help="Exodus database path.")
    variables_parser.add_argument(
        "--truth-tables", action="store_true", help="Include block/set variable truth tables."
    )

    blocks_parser = subparsers.add_parser("blocks", parents=[common], help="Print block metadata.")
    blocks_parser.add_argument("file", help="Exodus database path.")
    blocks_parser.add_argument(
        "--connectivity",
        action="store_true",
        help="Include a limited connectivity preview for each block.",
    )
    blocks_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum connectivity rows to include when --connectivity is used.",
    )

    sets_parser = subparsers.add_parser("sets", parents=[common], help="Print set metadata.")
    sets_parser.add_argument("file", help="Exodus database path.")
    sets_parser.add_argument(
        "--entries", action="store_true", help="Include a limited entries preview for each set."
    )
    sets_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum set entries to include when --entries is used.",
    )

    times_parser = subparsers.add_parser(
        "times", parents=[common], help="Print time-step information."
    )
    times_parser.add_argument("file", help="Exodus database path.")
    times_parser.add_argument("--all", action="store_true", help="Print all time values.")
    times_parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum time values to print unless --all is supplied.",
    )

    query_parser = subparsers.add_parser(
        "query", parents=[common], help="Query variables and print JSON records."
    )
    query_parser.add_argument("file", help="Exodus database path.")
    query_parser.add_argument(
        "--select",
        action="append",
        required=True,
        help="Variable selector, e.g. g/TOTAL_ENERGY, n/TEMP, e/ENERGY.",
    )
    query_parser.add_argument(
        "--time",
        default=None,
        help=(
            "Time selector. Use first, last, a physical time like 0.25, "
            "index:N for zero-based index, or step:N for one-based Exodus step."
        ),
    )
    query_parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum result rows to include. Use --limit -1 for all rows.",
    )
    query_parser.add_argument(
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
    query_index = query_parser.add_mutually_exclusive_group()
    query_index.add_argument(
        "--object-index",
        action="store_true",
        default=True,
        help="Include object index for node/element queries. Default.",
    )
    query_index.add_argument(
        "--no-object-index",
        action="store_false",
        dest="object_index",
        help="Do not include object index.",
    )

    stats_parser = subparsers.add_parser(
        "stats", parents=[common], help="Print compact numeric summaries for selected variables."
    )
    stats_parser.add_argument("file", help="Exodus database path.")
    stats_parser.add_argument(
        "--select",
        action="append",
        required=True,
        help="Variable selector, e.g. g/TOTAL_ENERGY, n/TEMP, e/ENERGY.",
    )
    stats_parser.add_argument(
        "--time",
        default="last",
        help=(
            "Time selector. Use first, last, a physical time like 0.25, "
            "index:N for zero-based index, or step:N for one-based Exodus step. "
            "Default is last."
        ),
    )
    stats_parser.add_argument(
        "--by-block",
        action="store_true",
        help="For element/edge/face variables, also report statistics by block.",
    )
    stats_parser.add_argument(
        "--by-set", action="store_true", help="For set variables, also report statistics by set."
    )
    stats_parser.add_argument(
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

    region_stats_parser = subparsers.add_parser(
        "region-stats",
        parents=[common],
        help="Compute statistics of a variable inside a geometric region.",
    )
    region_stats_parser.add_argument("file", help="Exodus database path.")
    region_stats_parser.add_argument(
        "--select",
        required=True,
        metavar="ENTITY/NAME",
        help="Variable selector, e.g. e/YIELD_STRESS_2.",
    )
    region_stats_parser.add_argument(
        "--block",
        type=int,
        default=None,
        metavar="BLOCK_ID",
        help="Restrict to a single element block ID. Default: all blocks.",
    )
    region_stats_parser.add_argument(
        "--time",
        default=None,
        help=(
            "Time selector. Use first, last, a physical time, index:N, or step:N. Default: last."
        ),
    )
    region_stats_parser.add_argument(
        "--reduce",
        default="mean,max,min,count",
        help=(
            "Comma-separated list of reducers: mean, max, min, sum, count, std. "
            "Default: mean,max,min,count."
        ),
    )
    region_stats_parser.add_argument(
        "--where",
        default=None,
        metavar="EXPR",
        help="Field predicate, e.g. 'EQPS_2 > 1.0'. AND-ed with the region mask.",
    )
    region_stats_parser.add_argument(
        "--symmetry",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help=(
            "Symmetry factor applied to extensive reducers (sum, count). "
            "Use 4.0 for a quarter-symmetry model. Default: 1.0."
        ),
    )

    # Region type (mutually exclusive)
    region_group = region_stats_parser.add_mutually_exclusive_group(required=True)
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

    examples_parser = subparsers.add_parser(
        "examples", parents=[common], help="Print database-specific Python usage examples."
    )
    examples_parser.add_argument("file", help="Exodus database path.")

    learn_parser = subparsers.add_parser(
        "learn",
        parents=[common],
        help=(
            "Query exodusii's static capabilities and skills for agent self-learning. "
            "With no topic, prints instructions for using this command."
        ),
    )
    learn_topics = learn_parser.add_subparsers(dest="learn_topic", metavar="topic")

    learn_capabilities = learn_topics.add_parser(
        "capabilities",
        parents=[common],
        aliases=("capability", "caps", "cap"),
        help="Query the static capability database.",
    )
    learn_capabilities.add_argument(
        "query",
        nargs="?",
        default="overview",
        help=(
            "Capability path. Defaults to 'overview'. "
            "Use 'all' (or 'capabilities') for the whole database, a top-level key "
            "like 'query' or 'mesh_geometry', or a nested path like 'python_api.values'."
        ),
    )

    learn_skills = learn_topics.add_parser(
        "skills", parents=[common], aliases=("skill",), help="Query the static skills database."
    )
    learn_skills.add_argument(
        "query",
        nargs="?",
        default="list",
        help=(
            "Skill selector. Defaults to 'list' (skill names). "
            "Use 'all' for every skill, or a skill name like 'exodusii-geometry' "
            "optionally followed by a path, e.g. 'exodusii-querying .body'."
        ),
    )
    learn_skills.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Optional query path below the selected skill, e.g. '.body'.",
    )

    return parser


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    """Dispatch parsed CLI arguments."""

    if args.command == "inspect":
        return inspect_command(args)
    if args.command == "variables":
        return variables_command(args)
    if args.command == "blocks":
        return blocks_command(args)
    if args.command == "sets":
        return sets_command(args)
    if args.command == "times":
        return times_command(args)
    if args.command == "query":
        return query_command(args)
    if args.command == "stats":
        return stats_command(args)
    if args.command == "examples":
        return examples_command(args)
    if args.command == "learn":
        return learn_command(args)
    if args.command == "region-stats":
        return region_stats_command(args)

    raise ValueError(f"unknown command {args.command!r}")


def inspect_command(args: argparse.Namespace) -> dict[str, Any]:
    """Return an agent-friendly database summary."""

    with ExodusFile.open(args.file) as exo:
        times = exo.times()

        return {
            "command": "inspect",
            "file": str(args.file),
            "title": exo.title,
            "storage_type": exo.storage_type,
            "dimension": exo.dimension,
            "counts": {
                "nodes": exo.node_count,
                "edges": exo.edge_count,
                "faces": exo.face_count,
                "elements": exo.element_count,
                "element_blocks": exo.element_block_count,
                "edge_blocks": len(exo.edge_block_ids()),
                "face_blocks": len(exo.face_block_ids()),
                "node_sets": exo.node_set_count,
                "side_sets": exo.side_set_count,
                "edge_sets": len(exo.edge_set_ids()),
                "face_sets": len(exo.face_set_ids()),
                "element_sets": len(exo.element_set_ids()),
                "time_steps": len(times),
            },
            "coordinates": {
                "names": exo.coordinate_names().tolist(),
                "displacement_variables": list(exo.displacement_variable_names()),
            },
            "times": time_summary(times),
            "variables": {ent.value: list(exo.variable_names(ent)) for ent in VARIABLE_ENTITIES},
            "ids": {
                "element_blocks": exo.element_block_ids().tolist(),
                "edge_blocks": exo.edge_block_ids().tolist(),
                "face_blocks": exo.face_block_ids().tolist(),
                "node_sets": exo.node_set_ids().tolist(),
                "side_sets": exo.side_set_ids().tolist(),
                "edge_sets": exo.edge_set_ids().tolist(),
                "face_sets": exo.face_set_ids().tolist(),
                "element_sets": exo.element_set_ids().tolist(),
            },
            "agent_hints": agent_hints(),
        }


def variables_command(args: argparse.Namespace) -> dict[str, Any]:
    """Return variable inventory."""

    include_truth_tables = bool(args.truth_tables)

    with ExodusFile.open(args.file) as exo:
        variables: list[dict[str, Any]] = []

        for ent in VARIABLE_ENTITIES:
            names = exo.variable_names(ent)
            item: dict[str, Any] = {
                "entity": ent.value,
                "selector_prefix": ent.short_name,
                "names": list(names),
                "count": len(names),
            }

            if ent in {Entity.ELEMENT, Entity.EDGE, Entity.FACE}:
                location = variable_block_location(ent)
                item["block_entity"] = location.value
                item["block_ids"] = exo.block_ids(location).tolist()

            if ent in {
                Entity.NODE_SET,
                Entity.SIDE_SET,
                Entity.EDGE_SET,
                Entity.FACE_SET,
                Entity.ELEMENT_SET,
            }:
                item["set_entity"] = ent.value
                item["set_ids"] = exo.set_ids(ent).tolist()

            if include_truth_tables:
                table = exo.variable_truth_table(ent)
                if table is not None:
                    item["truth_table"] = table.tolist()

            variables.append(item)

        return {
            "command": "variables",
            "file": str(args.file),
            "variables": variables,
            "agent_hints": {
                "selector_format": "ENTITY/NAME",
                "examples": ["g/TOTAL_ENERGY", "n/TEMP", "e/ENERGY"],
            },
        }


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


def times_command(args: argparse.Namespace) -> dict[str, Any]:
    """Return time information."""

    with ExodusFile.open(args.file) as exo:
        times = exo.times()
        include_count = len(times) if args.all else min(max(args.limit, 0), len(times))

        return {
            "command": "times",
            "file": str(args.file),
            **time_summary(times),
            "returned": include_count,
            "truncated": include_count < len(times),
            "times": [
                {"index": index, "step": index + 1, "value": float(value)}
                for index, value in enumerate(times[:include_count])
            ],
        }


def query_command(args: argparse.Namespace) -> dict[str, Any]:
    """Run a structured query and return JSON records."""

    time_selector = parse_time_selector(args.time)
    limit = normalize_limit(args.limit)
    file_path = _piece_path(args.file, getattr(args, "piece", None))

    with ExodusFile.open(file_path) as exo:
        result = query(exo, *args.select, time=time_selector, object_index=bool(args.object_index))

        row_count = len(result.data)
        returned_rows = row_count if limit is None else min(row_count, limit)

        payload: dict[str, Any] = {
            "command": "query",
            "file": str(args.file),
            "metadata": result.metadata,
            "columns": list(result.names),
            "row_count": row_count,
            "returned_rows": returned_rows,
            "truncated": returned_rows < row_count,
            "data": structured_to_records(result.data, limit=limit),
        }

        if getattr(args, "piece", None) is not None:
            payload["piece"] = int(args.piece)

        if "time" in result.metadata:
            payload["time"] = resolved_time_payload(exo, time_selector, requested=args.time)

        return payload


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


def examples_command(args: argparse.Namespace) -> dict[str, Any]:
    """Return database-specific Python examples."""

    path = str(args.file)

    with ExodusFile.open(args.file) as exo:
        examples: list[dict[str, str]] = [
            {
                "description": "Open the database and print a summary.",
                "code": (
                    "from exodusii.api.file import ExodusFile\n\n"
                    f"with ExodusFile.open({path!r}) as exo:\n"
                    "    print(exo.title)\n"
                    "    print(exo.dimension)\n"
                    "    print(exo.node_count, exo.element_count)\n"
                    "    print(exo.times())\n"
                ),
            },
            {
                "description": "List available variables by entity.",
                "code": (
                    "from exodusii.api.file import ExodusFile\n"
                    "from exodusii.core.entities import Entity\n\n"
                    f"with ExodusFile.open({path!r}) as exo:\n"
                    "    for ent in (Entity.GLOBAL, Entity.NODE, Entity.ELEMENT):\n"
                    "        print(ent.value, exo.variable_names(ent))\n"
                ),
            },
        ]

        node_names = exo.variable_names(Entity.NODE)
        if node_names:
            name = node_names[0]
            examples.append(
                {
                    "description": f"Read nodal variable {name!r} at the last time step.",
                    "code": (
                        "from exodusii.api.file import ExodusFile\n\n"
                        f"with ExodusFile.open({path!r}) as exo:\n"
                        f"    values = exo.values({name!r}, on='node', time='last')\n"
                        "    print(values.shape)\n"
                        "    print(values[:10])\n"
                    ),
                }
            )

        element_names = exo.variable_names(Entity.ELEMENT)
        block_ids = exo.element_block_ids()
        if element_names and len(block_ids):
            name = element_names[0]
            block_id = int(block_ids[0])
            examples.append(
                {
                    "description": (
                        f"Read element variable {name!r} on block {block_id} at the last time step."
                    ),
                    "code": (
                        "from exodusii.api.file import ExodusFile\n\n"
                        f"with ExodusFile.open({path!r}) as exo:\n"
                        f"    values = exo.values({name!r}, on='element', "
                        f"block_id={block_id}, time='last')\n"
                        "    print(values.shape)\n"
                        "    print(values[:10])\n"
                    ),
                }
            )

        global_names = exo.variable_names(Entity.GLOBAL)
        if global_names:
            name = global_names[0]
            examples.append(
                {
                    "description": f"Read complete time history of global variable {name!r}.",
                    "code": (
                        "from exodusii.api.file import ExodusFile\n\n"
                        f"with ExodusFile.open({path!r}) as exo:\n"
                        "    times = exo.times()\n"
                        f"    values = exo.values({name!r}, on='global')\n"
                        "    for time, value in zip(times, values, strict=True):\n"
                        "        print(time, value)\n"
                    ),
                }
            )

    return {"command": "examples", "file": path, "examples": examples, "agent_hints": agent_hints()}


def region_stats_command(args: argparse.Namespace) -> dict[str, Any]:
    """Compute statistics of a variable inside a geometric region."""
    from exodusii.mesh.regions import Circle
    from exodusii.mesh.regions import Cylinder
    from exodusii.mesh.regions import Rectangle
    from exodusii.mesh.regions import Sphere

    # Build the region object from CLI flags
    if args.cylinder is not None:
        ax, ay, az, bx, by, bz, r = args.cylinder
        region_obj = Cylinder([ax, ay, az], [bx, by, bz], r)
        region_desc = {"type": "cylinder", "p1": [ax, ay, az], "p2": [bx, by, bz], "radius": r}
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
            "a region type must be specified (--cylinder, --sphere, --circle, or --rectangle)"
        )

    # Parse the variable selector
    selectors = parse_variable_selectors([args.select], require_same_entity=False)
    if not selectors:
        raise ValueError("--select requires a valid variable selector")
    selector = selectors[0]
    var_entity = str(entity(selector.entity))
    var_name = selector.name

    # Parse reducers
    reduce_list = [r.strip() for r in args.reduce.split(",") if r.strip()]

    time_selector = parse_time_selector(args.time) if args.time is not None else None

    with ExodusFile.open(args.file) as exo:
        result = exo.region_stats(
            var_name,
            on=var_entity,
            block_id=args.block,
            region=region_obj,
            where=args.where,
            reduce=reduce_list,
            time=time_selector,
            symmetry_factor=args.symmetry,
        )

    return {
        "command": "region-stats",
        "file": str(args.file),
        "variable": result.variable,
        "entity": result.entity,
        "block_id": result.block_id,
        "region": region_desc,
        "where": args.where,
        "time": {"index": result.time_index, "value": result.time_value},
        "count_total": result.count_total,
        "count_selected": result.count_selected,
        "symmetry_factor": result.symmetry_factor,
        "stats": result.stats,
    }


def _piece_path(file_arg: str, piece: int | None) -> str:
    """Return the path to use for a single-piece read.

    When *piece* is ``None``, returns *file_arg* unchanged.  When *piece* is
    given, interprets *file_arg* as a glob pattern or a single path and
    returns the Nth (zero-based) lexicographically sorted match.

    This supports the common pattern of providing one component file directly
    (``--piece 0 mesh.e.96.00``) to avoid opening the full joined file for
    global-scalar queries.
    """
    if piece is None:
        return file_arg

    from pathlib import Path

    path = Path(file_arg)
    if path.exists():
        # Single file supplied directly — the piece index must be 0
        if piece != 0:
            raise ValueError(
                f"--piece {piece}: only piece 0 is valid when a single file path is given"
            )
        return file_arg

    # Try glob expansion
    import glob as _glob

    matches = sorted(_glob.glob(file_arg))
    if not matches:
        raise FileNotFoundError(f"--piece: no files matched {file_arg!r}")
    if piece >= len(matches):
        raise IndexError(
            f"--piece {piece}: only {len(matches)} files matched {file_arg!r} (0-based)"
        )
    return matches[piece]


def variable_stats_payload(
    exo: ExodusFile, selector: VariableSelector, *, time: TimeSelector, by_block: bool, by_set: bool
) -> dict[str, Any]:
    """Return statistics for one variable selector."""

    location = entity(selector.entity)

    if location in {Entity.ELEMENT, Entity.EDGE, Entity.FACE}:
        values = exo.values(selector.name, on=location, time=time)
        payload: dict[str, Any] = {"overall": array_stats(values)}

        if by_block:
            block_location = variable_block_location(location)
            payload["blocks"] = []
            for block_id in exo.block_ids(block_location):
                block_id_int = int(block_id)
                block_values = exo.values(
                    selector.name, on=location, block_id=block_id_int, time=time
                )
                payload["blocks"].append({"block_id": block_id_int, **array_stats(block_values)})

        return payload

    if location in {
        Entity.NODE_SET,
        Entity.SIDE_SET,
        Entity.EDGE_SET,
        Entity.FACE_SET,
        Entity.ELEMENT_SET,
    }:
        values = exo.values(selector.name, on=location, time=time)
        payload = {"overall": array_stats(values)}

        if by_set:
            payload["sets"] = []
            for set_id in exo.set_ids(location):
                set_id_int = int(set_id)
                set_values = exo.values(selector.name, on=location, set_id=set_id_int, time=time)
                payload["sets"].append({"set_id": set_id_int, **array_stats(set_values)})

        return payload

    values = exo.values(selector.name, on=location, time=time)
    return array_stats(values)


def block_payload(exo: ExodusFile, block_entity: Entity, block_id: int) -> dict[str, Any]:
    """Serialize one block."""

    block = exo.block(block_entity, block_id)
    return {
        "id": block.id,
        "index": block.index,
        "entity": entity(block.entity).value,
        "name": block.name,
        "element_type": block.element_type,
        "count": block.count,
        "nodes_per_entity": block.nodes_per_entity,
        "edges_per_entity": block.edges_per_entity,
        "faces_per_entity": block.faces_per_entity,
        "attributes": block.attributes,
        "active": exo.block_is_active(block_entity, block_id),
    }


def set_payload(exo: ExodusFile, set_entity: Entity, set_id: int) -> dict[str, Any]:
    """Serialize one set."""

    set_info = exo.set(set_entity, set_id)
    return {
        "id": set_info.id,
        "index": set_info.index,
        "entity": entity(set_info.entity).value,
        "name": set_info.name,
        "count": set_info.count,
        "distribution_factors": set_info.distribution_factors,
        "active": exo.set_is_active(set_entity, set_id),
    }


def time_summary(times: npt.ArrayLike) -> dict[str, Any]:
    """Return compact time summary."""

    values = np.asarray(times, dtype=np.float64)

    return {
        "count": int(values.size),
        "first": float(values[0]) if values.size else None,
        "last": float(values[-1]) if values.size else None,
        "min": float(np.min(values)) if values.size else None,
        "max": float(np.max(values)) if values.size else None,
    }


def resolved_time_payload(
    exo: ExodusFile, selector: TimeSelector, *, requested: str | None
) -> dict[str, Any]:
    """Return resolved time metadata."""

    selection = resolve_time(exo.times(), selector)
    return {
        "requested": requested,
        "index": selection.index,
        "step": selection.step,
        "value": selection.value,
        "exact": selection.exact,
    }


def parse_time_selector(value: str | None) -> TimeSelector:
    """Parse a CLI time selector.

    Supported forms
    ---------------
    first
    last
    index:N    zero-based Python time index
    step:N     one-based Exodus time step
    0.25       nearest physical time
    """

    if value is None:
        return None

    text = value.strip()
    key = text.lower()

    if key in {"first", "last"}:
        return key

    if key.startswith("index:"):
        index_text = key.split(":", 1)[1]
        return int(index_text)

    if key.startswith("step:"):
        step_text = key.split(":", 1)[1]
        step = int(step_text)
        if step < 1:
            raise ValueError("step:N time selector must use a one-based positive step")
        return step - 1

    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            "time must be 'first', 'last', a physical float time, 'index:N', or 'step:N'"
        ) from exc


def normalize_limit(value: int) -> int | None:
    """Normalize row limit. Negative means unlimited."""

    if value < 0:
        return None
    return value


def array_stats(values: npt.ArrayLike) -> dict[str, Any]:
    """Return JSON-safe numeric statistics."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]

    result: dict[str, Any] = {
        "count": int(array.size),
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
    }

    if finite.size:
        result.update(
            {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
            }
        )
    else:
        result.update({"min": None, "max": None, "mean": None, "std": None})

    return result


def structured_to_records(
    array: npt.NDArray[np.void], *, limit: int | None = None
) -> list[dict[str, Any]]:
    """Convert a structured NumPy array to JSON records."""

    names = array.dtype.names or ()
    rows = array if limit is None else array[:limit]

    return [{name: jsonable(row[name]) for name in names} for row in rows]


def limited_array_payload(values: npt.ArrayLike | None, *, limit: int) -> dict[str, Any]:
    """Return JSON-safe limited preview of an array."""

    if values is None:
        return {"shape": None, "dtype": None, "returned": 0, "truncated": False, "values": None}

    array = np.asarray(values)
    flat_limit = max(limit, 0)

    if array.ndim <= 1:
        preview = array[:flat_limit]
        returned = len(preview)
        total = len(array)
    else:
        preview = array[:flat_limit, ...]
        returned = preview.shape[0]
        total = array.shape[0]

    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "returned": int(returned),
        "truncated": bool(returned < total),
        "values": jsonable(preview),
    }


def variable_block_location(location: Entity) -> Entity:
    """Return block entity for an object variable entity."""

    if location is Entity.ELEMENT:
        return Entity.ELEMENT_BLOCK
    if location is Entity.EDGE:
        return Entity.EDGE_BLOCK
    if location is Entity.FACE:
        return Entity.FACE_BLOCK
    raise ValueError(f"{location.value!r} is not a block-variable entity")


def plural_key(ent: Entity) -> str:
    """Return conventional JSON plural key for an entity."""

    return f"{ent.value}s"


def agent_hints() -> dict[str, Any]:
    """Return stable hints for agents."""

    return {
        "variable_selector_format": "ENTITY/NAME",
        "selector_examples": [
            "g/TOTAL_ENERGY",
            "n/TEMP",
            "n/coordinates",
            "n/displacements",
            "e/ENERGY",
        ],
        "time_selectors": {
            "first": "first time step",
            "last": "last time step",
            "index:N": "zero-based Python time index",
            "step:N": "one-based Exodus time step",
            "float": "nearest physical time value",
        },
        "common_entities": {
            "g": "global",
            "n": "node",
            "e": "element",
            "d": "edge",
            "f": "face",
            "ns": "node_set",
            "ss": "side_set",
            "es": "edge_set",
            "fs": "face_set",
            "els": "element_set",
        },
        "python_api_examples": {
            "open": "with ExodusFile.open('mesh.exo') as exo: ...",
            "times": "exo.times()",
            "node_values": "exo.values('TEMP', on='node', time='last')",
            "element_values": "exo.values('ENERGY', on='element', block_id=1, time='last')",
            "coordinates": "exo.coordinates()",
            "displaced_coordinates": "exo.coordinates(time='last', displaced=True)",
        },
    }


def jsonable(value: Any) -> Any:
    """Convert common Python/NumPy values to JSON-serializable values."""

    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}

    if isinstance(value, tuple | list):
        return [jsonable(item) for item in value]

    return value


def emit_json(payload: dict[str, Any], *, terse: bool, file: TextIO | None = None) -> None:
    """Emit JSON to a stream.

    Default output is indented with two spaces.  ``terse=True`` removes all
    optional whitespace.
    """

    stream = file or sys.stdout
    converted = jsonable(payload)

    if terse:
        json.dump(converted, stream, separators=(",", ":"))
    else:
        json.dump(converted, stream, indent=2)

    stream.write("\n")


__all__ = ["build_parser", "dispatch", "main"]
