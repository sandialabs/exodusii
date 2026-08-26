# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Tests for the JSON-oriented exodusii agent CLI."""

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from exodusii.api.writer import ExodusWriter
from exodusii.cli.agent import main
from exodusii.cli.agent import parse_time_selector


@pytest.fixture
def simple_exodus_file(tmp_path: Path) -> Path:
    """Create a small Exodus file with globals, nodal values, elements, and sets."""

    path = tmp_path / "simple.exo"

    with ExodusWriter.create(path) as writer:
        writer.initialize(
            "simple test database", 2, 4, 1, element_blocks=1, node_sets=1, side_sets=1
        )

        writer.write_coordinates(
            np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64),
            names=["X", "Y"],
        )

        writer.define_element_block(
            100, "QUAD4", np.asarray([[1, 2, 3, 4]], dtype=np.int64), name="quad_block"
        )

        writer.define_node_set(10, np.asarray([1, 2], dtype=np.int64), name="fixed_nodes")

        writer.define_side_set(
            20,
            np.asarray([1], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
            distribution_factors=np.asarray([1.0], dtype=np.float64),
            name="loaded_side",
        )

        writer.define_global_variables(["TOTAL_ENERGY"])
        writer.define_node_variables(["TEMP"])
        writer.define_element_variables(["ENERGY"])

        writer.write_time(0.0, step=1)
        writer.write_global_values([10.0], step=1)
        writer.write_node_values("TEMP", [1.0, 2.0, 3.0, 4.0], step=1)
        writer.write_element_values("ENERGY", [100.0], block_id=100, step=1)

        writer.write_time(1.0, step=2)
        writer.write_global_values([20.0], step=2)
        writer.write_node_values("TEMP", [2.0, 3.0, 4.0, 5.0], step=2)
        writer.write_element_values("ENERGY", [200.0], block_id=100, step=2)

    return path


def _run_json(argv: list[str]) -> tuple[int, dict[str, Any]]:
    stream = io.StringIO()
    code = main(argv, file=stream)
    return code, json.loads(stream.getvalue())


def test_inspect_command(simple_exodus_file: Path) -> None:
    code, payload = _run_json(["inspect", str(simple_exodus_file)])

    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "inspect"
    assert payload["file"] == str(simple_exodus_file)
    assert payload["title"] == "simple test database"
    assert payload["dimension"] == 2

    assert payload["counts"]["nodes"] == 4
    assert payload["counts"]["elements"] == 1
    assert payload["counts"]["element_blocks"] == 1
    assert payload["counts"]["node_sets"] == 1
    assert payload["counts"]["side_sets"] == 1
    assert payload["counts"]["time_steps"] == 2

    assert payload["coordinates"]["names"] == ["X", "Y"]
    assert payload["times"]["first"] == 0.0
    assert payload["times"]["last"] == 1.0

    assert payload["variables"]["global"] == ["TOTAL_ENERGY"]
    assert payload["variables"]["node"] == ["TEMP"]
    assert payload["variables"]["element"] == ["ENERGY"]

    assert payload["ids"]["element_blocks"] == [100]
    assert payload["ids"]["node_sets"] == [10]
    assert payload["ids"]["side_sets"] == [20]


def test_variables_command(simple_exodus_file: Path) -> None:
    code, payload = _run_json(["variables", str(simple_exodus_file), "--truth-tables"])

    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "variables"

    by_entity = {item["entity"]: item for item in payload["variables"]}

    assert by_entity["global"]["names"] == ["TOTAL_ENERGY"]
    assert by_entity["node"]["names"] == ["TEMP"]
    assert by_entity["element"]["names"] == ["ENERGY"]
    assert by_entity["element"]["block_ids"] == [100]
    assert by_entity["element"]["truth_table"] == [[1]]


def test_blocks_command_without_connectivity(simple_exodus_file: Path) -> None:
    code, payload = _run_json(["blocks", str(simple_exodus_file)])

    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "blocks"

    assert len(payload["element_blocks"]) == 1

    block = payload["element_blocks"][0]
    assert block["id"] == 100
    assert block["index"] == 1
    assert block["entity"] == "element_block"
    assert block["name"] == "quad_block"
    assert block["element_type"] == "QUAD4"
    assert block["count"] == 1
    assert block["nodes_per_entity"] == 4
    assert block["active"] is True

    assert payload["edge_blocks"] == []
    assert payload["face_blocks"] == []


def test_blocks_command_with_connectivity(simple_exodus_file: Path) -> None:
    code, payload = _run_json(["blocks", str(simple_exodus_file), "--connectivity", "--limit", "1"])

    assert code == 0

    block = payload["element_blocks"][0]
    connectivity = block["connectivity"]

    assert connectivity["shape"] == [1, 4]
    assert connectivity["dtype"].startswith("int")
    assert connectivity["returned"] == 1
    assert connectivity["truncated"] is False
    assert connectivity["values"] == [[1, 2, 3, 4]]


def test_sets_command_without_entries(simple_exodus_file: Path) -> None:
    code, payload = _run_json(["sets", str(simple_exodus_file)])

    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "sets"

    assert len(payload["node_sets"]) == 1
    assert len(payload["side_sets"]) == 1

    node_set = payload["node_sets"][0]
    assert node_set["id"] == 10
    assert node_set["name"] == "fixed_nodes"
    assert node_set["count"] == 2
    assert node_set["active"] is True

    side_set = payload["side_sets"][0]
    assert side_set["id"] == 20
    assert side_set["name"] == "loaded_side"
    assert side_set["count"] == 1
    assert side_set["distribution_factors"] == 1
    assert side_set["active"] is True


def test_sets_command_with_entries(simple_exodus_file: Path) -> None:
    code, payload = _run_json(["sets", str(simple_exodus_file), "--entries", "--limit", "2"])

    assert code == 0

    node_set = payload["node_sets"][0]
    assert node_set["entries"]["shape"] == [2]
    assert node_set["entries"]["values"] == [1, 2]

    side_set = payload["side_sets"][0]
    assert side_set["entries"]["values"] == [1]
    assert side_set["extra_entries"]["values"] == [1]
    assert side_set["distribution_values"]["values"] == [1.0]


def test_times_command_default(simple_exodus_file: Path) -> None:
    code, payload = _run_json(["times", str(simple_exodus_file)])

    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "times"

    assert payload["count"] == 2
    assert payload["first"] == 0.0
    assert payload["last"] == 1.0
    assert payload["returned"] == 2
    assert payload["truncated"] is False
    assert payload["times"] == [
        {"index": 0, "step": 1, "value": 0.0},
        {"index": 1, "step": 2, "value": 1.0},
    ]


def test_query_node_variable_last_time_limited(simple_exodus_file: Path) -> None:
    code, payload = _run_json(
        ["query", str(simple_exodus_file), "--select", "n/TEMP", "--time", "last", "--limit", "2"]
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "query"

    assert payload["row_count"] == 4
    assert payload["returned_rows"] == 2
    assert payload["truncated"] is True
    assert payload["time"]["index"] == 1
    assert payload["time"]["step"] == 2
    assert payload["time"]["value"] == 1.0

    assert len(payload["data"]) == 2
    assert payload["data"][0]["TEMP"] == 2.0
    assert payload["data"][1]["TEMP"] == 3.0


def test_query_global_variable_history(simple_exodus_file: Path) -> None:
    code, payload = _run_json(
        ["query", str(simple_exodus_file), "--select", "g/TOTAL_ENERGY", "--limit", "-1"]
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "query"

    assert payload["row_count"] == 2
    assert payload["returned_rows"] == 2
    assert payload["truncated"] is False

    assert payload["data"][0]["TOTAL_ENERGY"] == 10.0
    assert payload["data"][1]["TOTAL_ENERGY"] == 20.0


def test_stats_node_variable(simple_exodus_file: Path) -> None:
    code, payload = _run_json(
        ["stats", str(simple_exodus_file), "--select", "n/TEMP", "--time", "last"]
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "stats"
    assert payload["entity"] == "node"

    stats = payload["variables"]["TEMP"]
    assert stats["count"] == 4
    assert stats["min"] == 2.0
    assert stats["max"] == 5.0
    assert stats["mean"] == 3.5
    assert stats["nan_count"] == 0
    assert stats["inf_count"] == 0


def test_stats_element_variable_by_block(simple_exodus_file: Path) -> None:
    code, payload = _run_json(
        ["stats", str(simple_exodus_file), "--select", "e/ENERGY", "--time", "last", "--by-block"]
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["entity"] == "element"

    energy = payload["variables"]["ENERGY"]
    assert energy["overall"]["count"] == 1
    assert energy["overall"]["min"] == 200.0
    assert energy["overall"]["max"] == 200.0
    assert energy["overall"]["mean"] == 200.0

    assert len(energy["blocks"]) == 1
    assert energy["blocks"][0]["block_id"] == 100
    assert energy["blocks"][0]["count"] == 1
    assert energy["blocks"][0]["mean"] == 200.0


def test_stats_rejects_mixed_selector_entities(simple_exodus_file: Path) -> None:
    code, payload = _run_json(
        [
            "stats",
            str(simple_exodus_file),
            "--select",
            "n/TEMP",
            "--select",
            "e/ENERGY",
            "--time",
            "last",
        ]
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["error"]["type"] == "ValueError"
    assert "same entity" in payload["error"]["message"]


def test_examples_command(simple_exodus_file: Path) -> None:
    code, payload = _run_json(["examples", str(simple_exodus_file)])

    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "examples"
    assert payload["file"] == str(simple_exodus_file)
    assert payload["examples"]

    descriptions = [example["description"] for example in payload["examples"]]
    assert any("Open the database" in description for description in descriptions)
    assert any("TEMP" in description for description in descriptions)
    assert any("ENERGY" in description for description in descriptions)


def test_command_error_is_json() -> None:
    code, payload = _run_json(["inspect", "does-not-exist.exo"])

    assert code == 1
    assert payload["ok"] is False
    assert "error" in payload
    assert "type" in payload["error"]
    assert "message" in payload["error"]


def test_parse_time_selector_none() -> None:
    assert parse_time_selector(None) is None


def test_parse_time_selector_keywords() -> None:
    assert parse_time_selector("first") == "first"
    assert parse_time_selector("last") == "last"
    assert parse_time_selector(" FIRST ") == "first"


def test_parse_time_selector_index() -> None:
    assert parse_time_selector("index:0") == 0
    assert parse_time_selector("index:3") == 3


def test_parse_time_selector_step() -> None:
    assert parse_time_selector("step:1") == 0
    assert parse_time_selector("step:4") == 3


def test_parse_time_selector_float() -> None:
    assert parse_time_selector("0.25") == 0.25
    assert parse_time_selector("1") == 1.0


def test_parse_time_selector_rejects_bad_step() -> None:
    with pytest.raises(ValueError, match="one-based positive"):
        parse_time_selector("step:0")


def test_parse_time_selector_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="time must be"):
        parse_time_selector("middle")
