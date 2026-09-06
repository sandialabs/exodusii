# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""``python -m exodusii inspect`` — agent-friendly database summary."""

import argparse
from typing import Any

from exodusii.api.file import ExodusFile
from exodusii.cli._common import VARIABLE_ENTITIES
from exodusii.cli._common import agent_hints
from exodusii.cli._common import time_summary

COMMAND = "inspect"

__all__ = ["COMMAND", "add_subparser", "inspect_command"]


def add_subparser(
    subparsers: argparse._SubParsersAction, common: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    """Register the ``inspect`` subparser."""
    parser = subparsers.add_parser(
        COMMAND, parents=[common], help="Print an agent-friendly database summary."
    )
    parser.add_argument("file", help="Exodus database path.")
    return parser


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
