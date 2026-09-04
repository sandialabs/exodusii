# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Static self-learning datasets for agent-oriented ExodusII usage.

This module backs:

    python -m exodusii learn -c overview
    python -m exodusii learn -c query
    python -m exodusii learn --skill list
    python -m exodusii learn --skill exodusii-querying

The query language is intentionally small and mirrors Canary's lightweight path
syntax: whole object with ".", dotted object keys, bracketed object keys, and
list indexes. It is not jq.
"""

import argparse
import json
import re
from importlib import resources
from pathlib import Path
from typing import Any

_KEY_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*\Z")


def learn_command(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the static learning command and return JSON-safe data."""

    if args.capability:
        data = query_capabilities(args.capability, args.query)
        return {
            "command": "learn",
            "dataset": "capabilities",
            "selector": args.capability,
            "query": args.query,
            "result": data,
        }

    if args.skill:
        data = query_skills(args.skill, args.query)
        return {
            "command": "learn",
            "dataset": "skills",
            "selector": args.skill,
            "query": args.query,
            "result": data,
        }

    if args.query and args.query != ".":
        raise ValueError(
            "learn query paths require a selected dataset. "
            "Use '-c CAPABILITY query' or '--skill SKILL query'."
        )

    return learn_instructions()


def learn_instructions() -> dict[str, Any]:
    """Return instructions for using the learn command."""

    return {
        "command": "learn",
        "purpose": (
            "Self-learning command for AI agents and automation tools. "
            "Use it to query installed static exodusii capability and skill datasets."
        ),
        "usage": {
            "capabilities": "python -m exodusii learn -c CAPABILITY [query]",
            "skills": "python -m exodusii learn --skill SKILL [query]",
            "terse_json": "python -m exodusii learn -c overview --terse",
        },
        "capability_selectors": {
            "overview": "High-level orientation to exodusii.",
            "commands": "Agent-oriented CLI command reference.",
            "query": "Variable selector, time selector, and lineout guidance.",
            "python_api": "Modern Python API usage.",
            "mesh_geometry": "Mesh geometry helpers, geometric region predicates "
            "(Cylinder/Sphere/Circle/Rectangle/Quad), element centers/volumes, "
            "and mass/volume-weighted region reductions.",
            "legacy": "Legacy compatibility API guidance.",
            "limitations": "Known limitations and caveats.",
            "all": "Entire capability database.",
            "capabilities": "Alias for all.",
        },
        "skill_selectors": {
            "list": "List installed skill names.",
            "all": "Return all skill objects.",
            "exodusii-agent-orientation": "General orientation for agents.",
            "exodusii-querying": "How to query Exodus databases and produce JSON.",
            "exodusii-python-api": "How to write Python code using exodusii.",
            "exodusii-geometry": "How to do geometry and region queries, "
            "including selecting entities inside a shape, element centers/volumes, "
            "region mass, and lineout profiles.",
            "exodusii-parallel-files": "How to work with decomposed parallel Exodus files.",
        },
        "query_language": {
            "summary": (
                "The optional query argument selects a path below the selected "
                "capability or skill object. It is intentionally lightweight and is not jq."
            ),
            "syntax": {
                ".": "Return the selected object.",
                "foo.bar": "Access nested object keys. Leading dot is optional.",
                ".foo.bar": "Equivalent to foo.bar.",
                "array[0]": "Access list index.",
                "object['key with spaces']": "Access quoted object key.",
                'object["key.with.dots"]': "Access quoted object key containing punctuation.",
            },
        },
        "examples": [
            {"description": "Show this instruction object.", "command": "python -m exodusii learn"},
            {
                "description": "Read the high-level capability overview.",
                "command": "python -m exodusii learn -c overview",
            },
            {
                "description": "Read the command reference.",
                "command": "python -m exodusii learn -c commands",
            },
            {
                "description": "Read only query selector guidance.",
                "command": "python -m exodusii learn -c query.selectors",
            },
            {
                "description": "Read modern Python value-query guidance.",
                "command": "python -m exodusii learn -c python_api.values",
            },
            {
                "description": "Read geometry/region helper guidance "
                "(Cylinder, element_volumes, etc.).",
                "command": "python -m exodusii learn -c mesh_geometry",
            },
            {
                "description": "Read lineout profile guidance.",
                "command": "python -m exodusii learn -c query.lineouts",
            },
            {
                "description": "List installed skills.",
                "command": "python -m exodusii learn --skill list",
            },
            {
                "description": "Read a skill object.",
                "command": "python -m exodusii learn --skill exodusii-querying",
            },
            {
                "description": "Read only a skill body.",
                "command": "python -m exodusii learn --skill exodusii-querying .body",
            },
            {
                "description": "Emit compact JSON.",
                "command": "python -m exodusii learn -c overview --terse",
            },
        ],
        "recommended_agent_flow": [
            "Run `python -m exodusii learn` if unfamiliar with the self-learning command.",
            "Run `python -m exodusii learn -c overview` for general orientation.",
            "Run `python -m exodusii learn -c commands` to learn available JSON CLI commands.",
            "Run `python -m exodusii inspect FILE.exo` before querying a specific database.",
            "Run `python -m exodusii variables FILE.exo` to discover valid variable selectors.",
            "Use `query` for small extracted tables and `stats` for large numeric arrays.",
        ],
    }


def load_capability_dataset() -> Any:
    """Load exodusii's static capability database."""

    path = resources.files("exodusii").joinpath("data").joinpath("capabilities.json")
    if not path.is_file():
        raise CapabilityDatasetNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))["capabilities"]


def load_skill_dataset() -> Any:
    """Load exodusii's static skills database."""

    path = resources.files("exodusii").joinpath("data").joinpath("skills.json")
    if not path.is_file():
        raise SkillDatasetNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))["skills"]


def query_capabilities(selector: str, query: str = ".") -> Any:
    """Query exodusii's static capability database.

    Examples
    --------
    ``query_capabilities("all")`` returns the full database.

    ``query_capabilities("overview")`` returns ``.overview``.

    ``query_capabilities("python_api.values")`` returns ``.python_api.values``.

    ``query_capabilities("query", ".selectors")`` returns ``.query.selectors``.
    """

    data = load_capability_dataset()
    selector = selector.strip()
    query = query.strip()

    if not selector:
        raise ValueError("capability selector must be non-empty")

    if selector in {"all", "capabilities"}:
        return query_json(data, query)

    shortcut = selector if selector.startswith(".") else f".{selector}"

    if query and query != ".":
        suffix = query[1:] if query.startswith(".") else query
        if suffix:
            shortcut = f"{shortcut}.{suffix}"

    return query_json(data, shortcut)


def query_skills(selector: str, query: str = ".") -> Any:
    """Query exodusii's static skills database.

    Examples
    --------
    ``query_skills("list")`` returns skill names.

    ``query_skills("all")`` returns all skills.

    ``query_skills("exodusii-querying")`` returns that skill object.

    ``query_skills("exodusii-querying", ".body")`` returns the skill body.
    """

    data = load_skill_dataset()
    selector = selector.strip()
    query = query.strip()

    if not selector:
        raise ValueError("skill selector must be non-empty")

    if selector == "all":
        return query_json(data, query)

    if selector == "list":
        return sorted(data.keys())

    try:
        skill = data[selector]
    except KeyError:
        raise KeyError(format_missing_key_message(selector, data)) from None

    return query_json(skill, query)


def query_json(data: Any, query: str) -> Any:
    """Apply a lightweight path query to JSON-like data."""

    query = query.strip()

    if not query or query == ".":
        return data

    if not query.startswith("."):
        query = "." + query

    current = data

    for token in parse_query(query):
        if isinstance(token, str):
            if not isinstance(current, dict):
                raise TypeError(
                    f"cannot access key {token!r} on {type(current).__name__}; "
                    "current value is not an object"
                )

            try:
                current = current[token]
            except KeyError:
                raise KeyError(format_missing_key_message(token, current)) from None

        elif isinstance(token, int):
            if not isinstance(current, list):
                raise TypeError(
                    f"cannot access index {token} on {type(current).__name__}; "
                    "current value is not an array"
                )

            try:
                current = current[token]
            except IndexError:
                raise IndexError(
                    f"no such index: {token}. Array length is {len(current)}."
                ) from None

        else:
            raise TypeError(f"unsupported query token: {token!r}")

    return current


def parse_query(query: str) -> list[str | int]:
    """Parse a simple JSON query path.

    Supported syntax includes:

    - ``.`` for the selected object, handled by ``query_json``
    - ``a.b[0]``
    - ``.a.b[0]``
    - ``a["key.with.dots"]``
    - ``a['key with spaces']``

    Bare dotted keys intentionally accept only a conservative identifier-like
    subset. Use bracket quotes for keys containing spaces, dots, punctuation, or
    other special characters.
    """

    tokens: list[str | int] = []
    i = 0

    while i < len(query):
        ch = query[i]

        if ch == ".":
            i += 1

            # Allow parse_query(".") to return [].
            if i >= len(query):
                break

            start = i
            while i < len(query) and query[i] not in ".[":
                i += 1

            if i > start:
                key_token = query[start:i]
                _validate_bare_key_token(key_token, column=start + 1, query=query)
                tokens.append(key_token)

            continue

        if ch == "[":
            bracket_token, i = parse_bracket(query, i)
            tokens.append(bracket_token)
            continue

        if _is_bare_key_start(ch):
            start = i
            while i < len(query) and query[i] not in ".[":
                i += 1

            key_token = query[start:i]
            _validate_bare_key_token(key_token, column=start + 1, query=query)
            tokens.append(key_token)
            continue

        raise ValueError(f"invalid query syntax at column {i + 1}: {query!r}")

    return tokens


def _is_bare_key_start(ch: str) -> bool:
    """Return true if ``ch`` can start an unquoted key token."""

    return ch.isalpha() or ch == "_"


def _validate_bare_key_token(token: str, *, column: int, query: str) -> None:
    """Validate an unquoted dotted-path key token."""

    if not _KEY_TOKEN_RE.fullmatch(token):
        raise ValueError(f"invalid query syntax at column {column}: {query!r}")


def parse_bracket(query: str, i: int) -> tuple[str | int, int]:
    """Parse one bracket expression from a query path."""

    assert query[i] == "["
    j = i + 1

    if j >= len(query):
        raise ValueError(f"unclosed bracket in query: {query!r}")

    if query[j] in {"'", '"'}:
        quote = query[j]
        j += 1
        chars: list[str] = []

        while j < len(query):
            ch = query[j]

            if ch == "\\":
                if j + 1 >= len(query):
                    raise ValueError(f"invalid escape in query: {query!r}")
                chars.append(query[j + 1])
                j += 2
                continue

            if ch == quote:
                j += 1
                if j >= len(query) or query[j] != "]":
                    raise ValueError(f"expected closing bracket in query: {query!r}")
                return "".join(chars), j + 1

            chars.append(ch)
            j += 1

        raise ValueError(f"unclosed quoted key in query: {query!r}")

    match = re.match(r"-?\d+", query[j:])
    if match:
        value = int(match.group(0))
        j += len(match.group(0))

        if j >= len(query) or query[j] != "]":
            raise ValueError(f"expected closing bracket in query: {query!r}")

        return value, j + 1

    raise ValueError(f"invalid bracket expression in query: {query!r}")


def format_missing_key_message(key: str, current: dict[str, Any]) -> str:
    """Return helpful missing-key text."""

    keys = sorted(str(k) for k in current)

    if not keys:
        return f"no such key: {key!r}. Current object has no keys."

    preview = ", ".join(keys[:24])
    if len(keys) > 24:
        preview += ", ..."

    return f"no such key: {key!r}. Available keys: {preview}"


class CapabilityDatasetNotFoundError(FileNotFoundError):
    """Raised when the installed capability database is missing."""

    def __init__(self, path: Path | Any) -> None:
        self.path = path
        super().__init__(f"exodusii capability database not found: {path}")


class SkillDatasetNotFoundError(FileNotFoundError):
    """Raised when the installed skills database is missing."""

    def __init__(self, path: Path | Any) -> None:
        self.path = path
        super().__init__(f"exodusii skills database not found: {path}")


__all__ = [
    "CapabilityDatasetNotFoundError",
    "SkillDatasetNotFoundError",
    "learn_command",
    "learn_instructions",
    "load_capability_dataset",
    "load_skill_dataset",
    "parse_query",
    "query_capabilities",
    "query_json",
    "query_skills",
]
