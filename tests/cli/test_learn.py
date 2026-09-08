# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Tests for the static exodusii self-learning command."""

import io
import json

import pytest

from exodusii.cli.learn import parse_query
from exodusii.cli.learn import query_capabilities
from exodusii.cli.learn import query_json
from exodusii.cli.learn import query_skills
from exodusii.cli.main import main


def _run_json(argv: list[str]) -> tuple[int, dict]:
    stream = io.StringIO()
    code = main(argv, file=stream)
    return code, json.loads(stream.getvalue())


def test_learn_without_selector_returns_instructions() -> None:
    code, payload = _run_json(["learn"])

    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "learn"
    assert "purpose" in payload
    assert "usage" in payload
    assert "capability_selectors" in payload
    assert "skill_selectors" in payload
    assert "examples" in payload

    examples = [item["command"] for item in payload["examples"]]
    assert "python -m exodusii learn" in examples
    assert "python -m exodusii learn capabilities overview" in examples


def test_learn_unknown_topic_is_rejected() -> None:
    stream = io.StringIO()
    with pytest.raises(SystemExit):
        main(["learn", "bogus-topic"], file=stream)


def test_learn_capability_overview() -> None:
    code, payload = _run_json(["learn", "capabilities", "overview"])

    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "learn"
    assert payload["dataset"] == "capabilities"
    assert payload["selector"] == "overview"
    assert "what_is_exodusii" in payload["result"]


def test_learn_capability_defaults_to_overview() -> None:
    code, payload = _run_json(["learn", "capabilities"])

    assert code == 0
    assert payload["ok"] is True
    assert payload["dataset"] == "capabilities"
    assert payload["selector"] == "overview"
    assert "what_is_exodusii" in payload["result"]


def test_learn_capability_alias() -> None:
    code, payload = _run_json(["learn", "caps", "overview"])

    assert code == 0
    assert payload["ok"] is True
    assert payload["dataset"] == "capabilities"
    assert "what_is_exodusii" in payload["result"]


def test_learn_capability_nested_shortcut() -> None:
    code, payload = _run_json(["learn", "capabilities", "query.selectors"])

    assert code == 0
    assert payload["ok"] is True
    assert payload["result"]["format"] == "ENTITY/NAME"
    assert payload["result"]["entity_prefixes"]["n"] == "node"
    assert payload["result"]["entity_prefixes"]["e"] == "element"


def test_learn_capability_deep_path() -> None:
    code, payload = _run_json(["learn", "capabilities", "query.time_selectors"])

    assert code == 0
    assert payload["ok"] is True
    assert payload["result"]["last"] == "Last time step."
    assert "index:N" in payload["result"]


def test_learn_skill_list() -> None:
    code, payload = _run_json(["learn", "skills", "list"])

    assert code == 0
    assert payload["ok"] is True
    assert payload["dataset"] == "skills"
    assert "exodusii-agent-orientation" in payload["result"]
    assert "exodusii-querying" in payload["result"]


def test_learn_skill_defaults_to_list() -> None:
    code, payload = _run_json(["learn", "skills"])

    assert code == 0
    assert payload["ok"] is True
    assert payload["dataset"] == "skills"
    assert "exodusii-querying" in payload["result"]


def test_learn_skill_body_query() -> None:
    code, payload = _run_json(["learn", "skills", "exodusii-querying", ".body"])

    assert code == 0
    assert payload["ok"] is True
    assert isinstance(payload["result"], str)
    assert "Querying Exodus databases" in payload["result"]
    assert "python -m exodusii query" in payload["result"]


def test_learn_terse_after_subcommand() -> None:
    stream = io.StringIO()
    code = main(["learn", "--terse"], file=stream)

    assert code == 0

    text = stream.getvalue()
    assert text.endswith("\n")
    assert "\n  " not in text
    assert ": " not in text

    payload = json.loads(text)
    assert payload["ok"] is True
    assert payload["command"] == "learn"


def test_learn_terse_before_subcommand() -> None:
    """``python -m exodusii learn --terse`` should emit compact JSON."""

    stream = io.StringIO()
    code = main(["learn", "--terse"], file=stream)

    assert code == 0

    text = stream.getvalue()
    assert text.endswith("\n")
    assert "\n  " not in text
    assert ": " not in text

    payload = json.loads(text)
    assert payload["ok"] is True
    assert payload["command"] == "learn"


def test_learn_capabilities_terse() -> None:
    stream = io.StringIO()
    code = main(["learn", "capabilities", "overview", "--terse"], file=stream)

    assert code == 0
    payload = json.loads(stream.getvalue())
    assert payload["ok"] is True
    assert payload["dataset"] == "capabilities"


def test_query_capabilities_direct() -> None:
    overview = query_capabilities("overview")

    assert overview["what_is_exodusii"].startswith("exodusii is a Python interface")


def test_query_skills_direct_list() -> None:
    skills = query_skills("list")

    assert "exodusii-python-api" in skills
    assert "exodusii-parallel-files" in skills


def test_query_skills_direct_field() -> None:
    description = query_skills("exodusii-python-api", ".description")

    assert isinstance(description, str)
    assert "modern exodusii API" in description


def test_parse_query_dotted_and_bracketed_tokens() -> None:
    assert parse_query(".a.b[0]['c.d']") == ["a", "b", 0, "c.d"]
    assert parse_query("a.b[1]") == ["a", "b", 1]
    assert parse_query('.a["key with spaces"]') == ["a", "key with spaces"]


def test_query_json_whole_object() -> None:
    data = {"a": {"b": [1, 2, 3]}}

    assert query_json(data, ".") is data
    assert query_json(data, "") is data


def test_query_json_nested_access() -> None:
    data = {"a": {"b": [10, {"c.d": 30, "sp ace": 40}]}}

    assert query_json(data, ".a.b[0]") == 10
    assert query_json(data, "a.b[1]['c.d']") == 30
    assert query_json(data, '.a.b[1]["sp ace"]') == 40


def test_query_json_missing_key_message() -> None:
    data = {"alpha": 1, "beta": 2}

    with pytest.raises(KeyError) as excinfo:
        query_json(data, ".gamma")

    message = str(excinfo.value)
    assert "gamma" in message
    assert "alpha" in message
    assert "beta" in message


def test_query_json_type_error_on_key_access_into_list() -> None:
    with pytest.raises(TypeError, match="current value is not an object"):
        query_json([1, 2, 3], ".name")


def test_query_json_type_error_on_index_access_into_object() -> None:
    with pytest.raises(TypeError, match="current value is not an array"):
        query_json({"a": 1}, "[0]")


def test_query_json_index_error() -> None:
    with pytest.raises(IndexError, match="Array length is 2"):
        query_json([1, 2], "[5]")


def test_query_json_invalid_syntax() -> None:
    with pytest.raises(ValueError, match="invalid query syntax"):
        query_json({"a": 1}, "a#bad")


def test_query_json_invalid_bracket_syntax() -> None:
    with pytest.raises(ValueError, match="invalid bracket expression"):
        query_json({"a": 1}, ".a[bad]")
