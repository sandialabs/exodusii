# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Reader for SEACAS ``exodiff`` *command files* (a.k.a. control files).

A command file is a whitespace-delimited (space/tab) text file of directives
that configure an ``exodiff`` comparison: default and per-category tolerances,
per-variable tolerance overrides and include/exclude lists, time-step
selection, and various behavioral switches.  This module parses such a file
into a :class:`~exodusii.api.diff.DiffOptions`.

The grammar mirrors ``ED_SystemInterface.C`` (``Parse_Command_File`` /
``Parse_Variables``) from SEACAS ``exodiff``:

* One directive per line; blank lines and lines whose first non-blank
  character is ``#`` are ignored.
* Keywords are case-insensitive and may be abbreviated to (generally) their
  first three characters (four for a few, e.g. ``coordinates``, ``global``).
* Tokens are separated by spaces, tabs, ``=`` and ``,``.

Supported directives
--------------------
``DEFAULT TOLERANCE <mode> <value> [FLOOR <f>]``
    Set the default tolerance for result variables.
``COORDINATES [<mode> <value>] [FLOOR <f>]``
    Coordinate comparison tolerance (default absolute 1e-6).
``TIME STEPS [<mode> <value>] [FLOOR <f>]``
    Time-value comparison tolerance.
``FINAL TIME TOLERANCE <value>``
    (accepted; mapped onto the time tolerance value)
``<CATEGORY> VARIABLES [(all)] [<mode> <value> [FLOOR <f>]]`` then an indented
list of ``NAME [<mode> <value>] [FLOOR <f>]`` or ``!NAME`` (exclude) lines.
    ``CATEGORY`` is one of GLOBAL, NODAL, ELEMENT, NODESET, SIDESET,
    EDGEBLOCK, FACEBLOCK; ``ELEMENT ATTRIBUTES`` is also accepted.
``STEP OFFSET automatic|match|<N>``
``EXCLUDE TIMES <list>``
``INTERPOLATE``
``APPLY MATCHING`` / ``NODESET MATCH`` / ``SIDESET MATCH``
``IGNORE CASE`` / ``CASE SENSITIVE``

Accepted-but-inert directives (parsed, then reported as an unsupported-directive
warning on the returned :class:`CommandFileResult`): ``CALCULATE
NORMS/L1NORMS/L2NORMS``, ``IGNORE MAPS/NANS/DUPS/ATTRIBUTES``, ``IGNORE
SIDESET DISTRIBUTION``, ``SHORT BLOCKS`` / ``NO SHORT``, ``PEDANTIC``, ``RETURN
STATUS`` / ``IGNORE STATUS``, ``MAX NAMES``, ``SIDESET DISTRIBUTION``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from exodusii.api.diff import DiffOptions
from exodusii.api.diff import TimeSelection
from exodusii.core.entities import Entity
from exodusii.core.tolerance import Tolerance
from exodusii.core.tolerance import ToleranceMode

__all__ = ["CommandFileError", "CommandFileResult", "read_command_file"]

_TOKEN_SEP = re.compile(r"[ \t=,]+")

# Tolerance-mode tokens recognised in a command file, with the minimum
# abbreviation length SEACAS uses for each.
_MODE_TOKENS: tuple[tuple[str, int, ToleranceMode], ...] = (
    ("relative", 3, ToleranceMode.RELATIVE),
    ("absolute", 3, ToleranceMode.ABSOLUTE),
    ("combine", 3, ToleranceMode.COMBINED),
    ("eigen_relative", 7, ToleranceMode.EIGEN_RELATIVE),
    ("eigen_absolute", 7, ToleranceMode.EIGEN_ABSOLUTE),
    ("eigen_combine", 7, ToleranceMode.EIGEN_COMBINED),
    ("ulps_float", 6, ToleranceMode.ULPS_FLOAT),
    ("ulps_double", 6, ToleranceMode.ULPS_DOUBLE),
    ("ignore", 3, ToleranceMode.IGNORE),
)

# Category keyword -> variable-location Entity.  The int is the minimum
# abbreviation length of the leading keyword.
_CATEGORY_TOKENS: tuple[tuple[str, int, Entity], ...] = (
    ("global", 4, Entity.GLOBAL),
    ("nodal", 4, Entity.NODE),
    ("element", 4, Entity.ELEMENT),
    ("nodeset", 4, Entity.NODE_SET),
    ("sideset", 4, Entity.SIDE_SET),
    ("edgeblock", 4, Entity.EDGE),
    ("faceblock", 4, Entity.FACE),
)


class CommandFileError(ValueError):
    """Raised when a command file cannot be parsed."""


@dataclass
class CommandFileResult:
    """Result of parsing a command file.

    Attributes
    ----------
    options
        The :class:`~exodusii.api.diff.DiffOptions` built from the file.
    warnings
        Human-readable messages for directives that were recognised but have
        no runtime effect in this implementation (accept-and-warn).
    """

    options: DiffOptions
    warnings: list[str] = field(default_factory=list)


def _abbrev(token: str, word: str, minlen: int) -> bool:
    """Return ``True`` if *token* is an accepted abbreviation of *word*.

    Matches SEACAS ``abbreviation``: *token* must be a prefix of *word* at
    least *minlen* characters long (and no longer than *word*).
    """

    token = token.lower()
    if len(token) < minlen or len(token) > len(word):
        return False
    return word.startswith(token)


def _match_mode(token: str) -> ToleranceMode | None:
    for word, minlen, mode in _MODE_TOKENS:
        if _abbrev(token, word, minlen):
            return mode
    return None


def _to_float(token: str, line: str) -> float:
    try:
        return float(token)
    except ValueError as exc:  # pragma: no cover - defensive
        raise CommandFileError(f"expected a number, got {token!r} in line: {line!r}") from exc


@dataclass
class _Tokens:
    """A simple forward token cursor over a single logical line."""

    items: list[str]
    pos: int = 0

    def next(self) -> str | None:
        if self.pos >= len(self.items):
            return None
        tok = self.items[self.pos]
        self.pos += 1
        return tok

    def peek(self) -> str | None:
        return self.items[self.pos] if self.pos < len(self.items) else None


def _split(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped:
        return []
    return [t for t in _TOKEN_SEP.split(stripped) if t]


def _is_comment(line: str) -> bool:
    s = line.lstrip()
    return bool(s) and s[0] == "#"


def _is_indented(line: str) -> bool:
    return bool(line) and line[0] in (" ", "\t")


def _parse_tolerance_spec(toks: _Tokens, base: Tolerance, line: str) -> Tolerance:
    """Parse an optional ``<mode> <value>`` and/or ``FLOOR <f>`` from *toks*.

    Starts from *base* and overrides mode/value/floor as specified.  Leaves the
    cursor positioned after the consumed tokens.
    """

    mode = base.mode
    value = base.value
    floor = base.floor

    tok = toks.peek()
    if tok is not None and _abbrev(tok, "floor", 3):
        toks.next()
        f = toks.next()
        if f is None:
            raise CommandFileError(f"FLOOR specified but no value in line: {line!r}")
        return Tolerance(mode=mode, value=value, floor=_to_float(f, line))

    matched_mode = _match_mode(tok) if tok is not None else None
    if matched_mode is not None:
        toks.next()
        mode = matched_mode
        if mode is ToleranceMode.IGNORE:
            value = 0.0
        else:
            v = toks.next()
            if v is None or _abbrev(v, "floor", 3):
                raise CommandFileError(
                    f"tolerance mode {tok!r} given without a value in line: {line!r}"
                )
            value = _to_float(v, line)

    tok = toks.peek()
    if tok is not None and _abbrev(tok, "floor", 3):
        toks.next()
        f = toks.next()
        if f is None:
            raise CommandFileError(f"FLOOR specified but no value in line: {line!r}")
        floor = _to_float(f, line)

    return Tolerance(mode=mode, value=value, floor=floor)


def read_command_file(path: str | Path) -> CommandFileResult:
    """Parse an ``exodiff`` command file into a :class:`CommandFileResult`.

    Parameters
    ----------
    path
        Path to the command file.

    Returns
    -------
    CommandFileResult
        The parsed :class:`~exodusii.api.diff.DiffOptions` and any
        unsupported-directive warnings.

    Raises
    ------
    CommandFileError
        On a malformed directive.
    OSError
        If the file cannot be read.
    """

    text = Path(path).read_text()
    lines = text.splitlines()

    warnings: list[str] = []

    default_tol = Tolerance(ToleranceMode.RELATIVE, 1.0e-6, 0.0)
    default_specified = False
    coord_tol = Tolerance(ToleranceMode.ABSOLUTE, 1.0e-6, 0.0)
    time_tol = Tolerance(ToleranceMode.RELATIVE, 1.0e-6, 1.0e-15)

    variable_tolerances: dict[str, Tolerance] = {}
    exclude: set[str] = set()
    include_variables: dict[Entity, set[str]] = {}
    all_categories: set[Entity] = set()
    category_defaults: dict[Entity, Tolerance] = {}
    attr_tol: Tolerance | None = None

    ignore_case = True
    coordinate_matching = False
    interpolating = False
    time_step_offset = 0
    exclude_steps: set[int] = set()
    compare_attributes = True

    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        i += 1
        if not raw.strip() or _is_comment(raw):
            continue
        toks = _split(raw)
        if not toks:
            continue
        t1 = toks[0].lower()
        t2 = toks[1].lower() if len(toks) > 1 else ""

        # ── DEFAULT TOLERANCE ────────────────────────────────────────────
        if _abbrev(t1, "default", 3) and _abbrev(t2, "tolerance", 3):
            cur = _Tokens(toks, 2)
            default_tol = _parse_tolerance_spec(cur, default_tol, raw)
            default_specified = True
            continue

        # ── COORDINATES ──────────────────────────────────────────────────
        if _abbrev(t1, "coordinates", 4):
            base = (
                default_tol if default_specified else Tolerance(ToleranceMode.ABSOLUTE, 1.0e-6, 0.0)
            )
            cur = _Tokens(toks, 1)
            coord_tol = _parse_tolerance_spec(cur, base, raw)
            continue

        # ── TIME STEPS ───────────────────────────────────────────────────
        if t1 == "time" and _abbrev(t2, "steps", 4):
            base = default_tol
            cur = _Tokens(toks, 2)
            time_tol = _parse_tolerance_spec(cur, base, raw)
            continue

        # ── FINAL TIME TOLERANCE ─────────────────────────────────────────
        if _abbrev(t1, "final", 3) and _abbrev(t2, "time", 3):
            cur = _Tokens(toks, 2)
            tok = cur.next()
            if tok is None or not _abbrev(tok, "tolerance", 3):
                raise CommandFileError(f"expected TOLERANCE after FINAL TIME in line: {raw!r}")
            v = cur.next()
            if v is None:
                raise CommandFileError(f"FINAL TIME TOLERANCE without a value: {raw!r}")
            time_tol = Tolerance(mode=time_tol.mode, value=_to_float(v, raw), floor=time_tol.floor)
            continue

        # ── STEP OFFSET ──────────────────────────────────────────────────
        if t1 == "step" and t2 == "offset":
            tok = toks[2].lower() if len(toks) > 2 else ""
            if _abbrev(tok, "automatic", 4):
                time_step_offset = -1
                warnings.append(
                    "STEP OFFSET AUTOMATIC is accepted but automatic offset "
                    "detection is not implemented; using match-by-index."
                )
                time_step_offset = 0
            elif _abbrev(tok, "match", 4):
                time_step_offset = 0
            elif tok:
                time_step_offset = int(tok)
            continue

        # ── EXCLUDE TIMES ────────────────────────────────────────────────
        if _abbrev(t1, "exclude", 3) and _abbrev(t2, "times", 3):
            for tok in toks[2:]:
                if tok.startswith("#"):
                    break
                exclude_steps.update(_parse_int_list(tok))
            continue

        # ── matching switches ────────────────────────────────────────────
        if _abbrev(t1, "apply", 3) and _abbrev(t2, "matching", 3):
            coordinate_matching = True
            continue
        if t1 == "nodeset" and _abbrev(t2, "match", 3):
            coordinate_matching = True
            continue
        if t1 == "sideset" and _abbrev(t2, "match", 3):
            coordinate_matching = True
            continue

        if t1 == "interpolate":
            interpolating = True
            continue

        # ── case sensitivity ─────────────────────────────────────────────
        if _abbrev(t1, "ignore", 3) and _abbrev(t2, "case", 3):
            ignore_case = True
            continue
        if _abbrev(t1, "case", 3) and _abbrev(t2, "sensitive", 3):
            ignore_case = False
            continue

        if _abbrev(t1, "ignore", 3) and _abbrev(t2, "attributes", 3):
            compare_attributes = False
            continue

        # ── ELEMENT ATTRIBUTES (a variable block over attributes) ────────
        if _abbrev(t1, "element", 4) and _abbrev(t2, "attributes", 3):
            # Consume the block; map default onto attribute tolerance, per-name
            # tolerances into variable_tolerances (best-effort).
            attr_default, names, tols, _all, i = _consume_variable_block(
                toks, 2, lines, i, default_tol, raw
            )
            for name, tol in zip(names, tols):
                if name.startswith("!"):
                    exclude.add(name[1:])
                else:
                    variable_tolerances[name] = tol
            attr_tol = attr_default
            continue

        # ── SIDESET DISTRIBUTION (inert here) ────────────────────────────
        if _abbrev(t1, "sideset", 4) and _abbrev(t2, "distribution", 4):
            warnings.append(
                "SIDESET DISTRIBUTION tolerance is accepted but not applied "
                "(distribution-factor comparison is not separately configurable)."
            )
            continue

        # ── per-category VARIABLES blocks ────────────────────────────────
        category = _match_category(t1)
        if category is not None and _abbrev(t2, "variables", 3):
            cat_default, names, tols, block_all, i = _consume_variable_block(
                toks, 2, lines, i, default_tol, raw
            )
            includes: set[str] = set()
            for name, tol in zip(names, tols):
                if name.startswith("!"):
                    exclude.add(name[1:])
                    continue
                includes.add(name)
                variable_tolerances[name] = tol
            if block_all or not includes:
                all_categories.add(category)
            if includes:
                include_variables.setdefault(category, set()).update(includes)
            category_defaults[category] = cat_default
            continue

        # ── accepted-but-inert directives ────────────────────────────────
        if _is_inert(t1, t2):
            warnings.append(
                f"directive {' '.join(toks[:2]).upper()!r} is accepted but has "
                "no effect in this implementation."
            )
            continue

        raise CommandFileError(f"unrecognized directive in line: {raw!r}")

    time_selection = TimeSelection(
        time_step_offset=time_step_offset,
        exclude_steps=frozenset(exclude_steps),
        interpolating=interpolating,
    )

    options = DiffOptions(
        default_tolerance=default_tol,
        coordinate_tolerance=coord_tol,
        time_tolerance=time_tol,
        global_tolerance=category_defaults.get(Entity.GLOBAL),
        nodal_tolerance=category_defaults.get(Entity.NODE),
        element_tolerance=category_defaults.get(Entity.ELEMENT),
        edge_tolerance=category_defaults.get(Entity.EDGE),
        face_tolerance=category_defaults.get(Entity.FACE),
        node_set_tolerance=category_defaults.get(Entity.NODE_SET),
        side_set_tolerance=category_defaults.get(Entity.SIDE_SET),
        attribute_tolerance=attr_tol,
        variable_tolerances=dict(variable_tolerances),
        exclude=frozenset(exclude),
        include_variables={k: frozenset(v) for k, v in include_variables.items()},
        all_categories=frozenset(all_categories),
        ignore_case=ignore_case,
        time_selection=time_selection,
        compare_attributes=compare_attributes,
        coordinate_matching=coordinate_matching,
    )

    return CommandFileResult(options=options, warnings=warnings)


def _match_category(token: str) -> Entity | None:
    for word, minlen, ent in _CATEGORY_TOKENS:
        if _abbrev(token, word, minlen):
            return ent
    return None


def _consume_variable_block(
    header_toks: list[str],
    start: int,
    lines: list[str],
    i: int,
    default_tol: Tolerance,
    header_line: str,
) -> tuple[Tolerance, list[str], list[Tolerance], bool, int]:
    """Parse a ``... VARIABLES ...`` header + indented name list.

    Returns ``(category_default_tol, names, tolerances, all_flag, next_i)``.
    ``names`` may include ``!NAME`` exclusion entries (caller splits them out).
    """

    cur = _Tokens(header_toks, start)
    all_flag = False
    tok = cur.peek()
    if tok is not None and tok.lower() in ("(all)", "all"):
        all_flag = True
        cur.next()
    cat_default = _parse_tolerance_spec(cur, default_tol, header_line)

    names: list[str] = []
    tols: list[Tolerance] = []

    n = len(lines)
    while i < n:
        line = lines[i]
        if _is_comment(line):
            i += 1
            continue
        if not _is_indented(line):
            break
        i += 1
        toks = _split(line)
        if not toks:
            continue
        name = toks[0]
        if name.startswith("#"):
            continue
        if name.startswith("!"):
            names.append(name)
            tols.append(cat_default)
            continue
        # optional per-name tolerance
        sub = _Tokens(toks, 1)
        tol = _parse_tolerance_spec(sub, cat_default, line)
        names.append(name)
        tols.append(tol)

    return cat_default, names, tols, all_flag, i


def _parse_int_list(token: str) -> list[int]:
    out: list[int] = []
    for part in token.replace(",", " ").split():
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


_INERT_DIRECTIVES: tuple[tuple[str, int, str, int], ...] = (
    ("calculate", 3, "norms", 3),
    ("calculate", 3, "l1norms", 3),
    ("calculate", 3, "l2norms", 3),
    ("ignore", 3, "maps", 3),
    ("ignore", 3, "nans", 3),
    ("ignore", 3, "dups", 3),
    ("short", 3, "blocks", 3),
    ("pedantic", 8, "", 0),
    ("return", 3, "status", 3),
    ("ignore", 3, "status", 3),
    ("max", 3, "names", 3),
)


def _is_inert(t1: str, t2: str) -> bool:
    if t1 == "no" and _abbrev(t2, "short", 3):
        return True
    for w1, m1, w2, m2 in _INERT_DIRECTIVES:
        if _abbrev(t1, w1, m1) and (not w2 or _abbrev(t2, w2, m2)):
            return True
    return False
