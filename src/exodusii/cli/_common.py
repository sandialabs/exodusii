# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Shared helpers for the ``python -m exodusii`` subcommands.

The JSON-oriented CLI is split into one module per subcommand under
:mod:`exodusii.cli`.  This module holds the entity constants, argument-parsing
helpers, and JSON-serialization utilities that those subcommand modules share,
so no subcommand needs to import another (and ``main`` stays a thin
coordinator).
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
from exodusii.core.entities import Entity
from exodusii.core.entities import entity
from exodusii.core.selectors import VariableSelector
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


def make_common_parser() -> argparse.ArgumentParser:
    """Return the shared parent parser carrying the ``--terse`` flag.

    ``--terse`` is put on both the top-level parser and every subparser (via
    this parent) so both invocation styles work::

        python -m exodusii --terse inspect file.exo
        python -m exodusii inspect file.exo --terse
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--terse",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit compact JSON with no extra whitespace. Default is indented JSON.",
    )
    return common


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
