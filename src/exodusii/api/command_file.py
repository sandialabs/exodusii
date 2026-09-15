# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Readers and writer for exodiff *command files* (a.k.a. control files).

Two on-disk formats are supported for configuring a :func:`~exodusii.api.diff.diff`:

* **YAML** (preferred, modern) -- a structured, self-documenting format with
  full feature parity with the SEACAS exodiff command file.  See
  :class:`YamlCommandFileReader` and :func:`diff_options_to_yaml`.
* **SEACAS exodiff text** (legacy, for backward compatibility) -- the
  whitespace-delimited directive grammar of ``exodiff``.  See
  :class:`ExodiffCommandFileReader`.

Use the format-detecting factory :func:`read_command_file` (or
:func:`command_file_reader`) to parse either; YAML is preferred when the format
is ambiguous.  A :class:`~exodusii.api.diff.DiffOptions` can be serialized back
to YAML with :func:`diff_options_to_yaml` / :meth:`DiffOptions.to_yaml`.

Legacy exodiff grammar
----------------------
* One directive per line; blank lines and lines whose first non-blank
  character is ``#`` are ignored.
* Keywords are case-insensitive and may be abbreviated to (generally) their
  first three characters (four for a few, e.g. ``coordinates``, ``global``).
* Tokens are separated by spaces, tabs, ``=`` and ``,``.

Supported exodiff directives: ``DEFAULT TOLERANCE``, ``COORDINATES``, ``TIME
STEPS``, ``FINAL TIME TOLERANCE``, per-category ``<GLOBAL|NODAL|ELEMENT|
NODESET|SIDESET|EDGEBLOCK|FACEBLOCK> VARIABLES`` blocks (with ``(all)``,
per-variable tolerances and ``!NAME`` exclusions), ``ELEMENT ATTRIBUTES``,
``STEP OFFSET``, ``EXCLUDE TIMES``, ``INTERPOLATE``, ``APPLY MATCHING`` /
``NODESET MATCH`` / ``SIDESET MATCH``, and ``IGNORE CASE`` / ``CASE
SENSITIVE``.

Accepted-but-inert exodiff directives (parsed, then reported as a warning on
the returned :class:`CommandFileResult`): ``CALCULATE NORMS/L1NORMS/L2NORMS``,
``IGNORE MAPS/NANS/DUPS/ATTRIBUTES``, ``IGNORE SIDESET DISTRIBUTION``, ``SHORT
BLOCKS`` / ``NO SHORT``, ``PEDANTIC``, ``RETURN STATUS`` / ``IGNORE STATUS``,
``MAX NAMES``, ``SIDESET DISTRIBUTION``.
"""

from __future__ import annotations

import re
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import yaml

from exodusii.api.diff import DiffOptions
from exodusii.api.diff import TimeSelection
from exodusii.core.entities import Entity
from exodusii.core.tolerance import Tolerance
from exodusii.core.tolerance import ToleranceMode

__all__ = [
    "CommandFileError",
    "CommandFileReader",
    "CommandFileResult",
    "ExodiffCommandFileReader",
    "YamlCommandFileReader",
    "command_file_reader",
    "diff_options_to_yaml",
    "read_command_file",
    "write_command_file",
]

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


class CommandFileReader(ABC):
    """Abstract base class for command-file readers.

    A reader is constructed from a path and produces a
    :class:`CommandFileResult` via :meth:`read`.  Concrete subclasses parse a
    specific on-disk format:

    * :class:`YamlCommandFileReader` -- the modern, preferred YAML format.
    * :class:`ExodiffCommandFileReader` -- the legacy SEACAS exodiff text
      format, retained for backward compatibility.

    Use :func:`command_file_reader` to pick the right reader for a file.
    """

    #: Human-readable name of the format this reader parses.
    format_name: str = ""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    @abstractmethod
    def sniff(cls, path: Path, head: str) -> bool:
        """Return ``True`` if *path* appears to be in this reader's format.

        Parameters
        ----------
        path
            The file path (its suffix may be inspected).
        head
            The first portion of the file's text content.
        """

    @abstractmethod
    def read(self) -> CommandFileResult:
        """Parse the file and return a :class:`CommandFileResult`."""


def command_file_reader(path: str | Path) -> CommandFileReader:
    """Return a :class:`CommandFileReader` for *path*, auto-detecting format.

    YAML is the preferred format: a ``.yaml``/``.yml`` suffix, or content that
    parses as a YAML mapping, selects :class:`YamlCommandFileReader`.  The
    legacy exodiff text format is used otherwise.

    Raises
    ------
    OSError
        If the file cannot be read.
    """

    p = Path(path)
    head = p.read_text()[:4096]
    # YAML is preferred; try it first, then fall back to the exodiff text form.
    if YamlCommandFileReader.sniff(p, head):
        return YamlCommandFileReader(p)
    return ExodiffCommandFileReader(p)


def read_command_file(path: str | Path) -> CommandFileResult:
    """Parse a command file (YAML or exodiff text) into a result.

    The format is auto-detected via :func:`command_file_reader` (YAML
    preferred).  This is the top-level entry point most callers want.

    Parameters
    ----------
    path
        Path to the command file.

    Returns
    -------
    CommandFileResult
        The parsed :class:`~exodusii.api.diff.DiffOptions` and any warnings.

    Raises
    ------
    CommandFileError
        On a malformed file.
    OSError
        If the file cannot be read.
    """

    return command_file_reader(path).read()


def write_command_file(options: DiffOptions, path: str | Path) -> None:
    """Write *options* to *path* as a YAML command file (full/explicit)."""

    Path(path).write_text(diff_options_to_yaml(options))


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


class ExodiffCommandFileReader(CommandFileReader):
    """Reader for the legacy SEACAS ``exodiff`` command-file text grammar.

    Retained for backward compatibility; new files should use the YAML format
    (:class:`YamlCommandFileReader`).  Mirrors ``ED_SystemInterface.C``
    (``Parse_Command_File`` / ``Parse_Variables``).
    """

    format_name = "exodiff"

    @classmethod
    def sniff(cls, path: Path, head: str) -> bool:
        # This reader is the fallback; it accepts anything that is not YAML.
        return True

    def read(self) -> CommandFileResult:
        """Parse the exodiff command file into a :class:`CommandFileResult`."""

        return _parse_exodiff_lines(self.path.read_text().splitlines())


def _parse_exodiff_lines(lines: list[str]) -> CommandFileResult:
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


# ═══════════════════════════════════════════════════════════════════════════
# YAML command file (modern, preferred)
# ═══════════════════════════════════════════════════════════════════════════

#: YAML category key  ->  (variable-location Entity, DiffOptions tolerance field)
#
# These keys are used both under ``tolerances.categories`` (per-category
# default tolerances) and under ``variables.include`` (per-category include
# lists / ``all``).
_YAML_CATEGORIES: dict[str, tuple[Entity, str]] = {
    "global": (Entity.GLOBAL, "global_tolerance"),
    "nodal": (Entity.NODE, "nodal_tolerance"),
    "element": (Entity.ELEMENT, "element_tolerance"),
    "edge": (Entity.EDGE, "edge_tolerance"),
    "face": (Entity.FACE, "face_tolerance"),
    "node_set": (Entity.NODE_SET, "node_set_tolerance"),
    "side_set": (Entity.SIDE_SET, "side_set_tolerance"),
    "edge_set": (Entity.EDGE_SET, "edge_set_tolerance"),
    "face_set": (Entity.FACE_SET, "face_set_tolerance"),
    "element_set": (Entity.ELEMENT_SET, "element_set_tolerance"),
}
_ENTITY_TO_YAML_CATEGORY: dict[Entity, str] = {
    ent: key for key, (ent, _) in _YAML_CATEGORIES.items()
}


def _tolerance_from_yaml(spec: Any, where: str) -> Tolerance:
    """Build a :class:`Tolerance` from a YAML node.

    Accepts:

    * a mapping ``{mode: <str>, value: <num>, floor: <num>, use_old_floor: bool}``
      (any subset; ``mode`` defaults to ``relative``, ``value``/``floor`` to 0);
    * a shorthand string ``"<mode> <value>"`` or ``"<value>"`` (mode defaults
      to relative), e.g. ``"absolute 1e-8"`` or ``"1e-6"``;
    * a bare number (relative tolerance of that value).
    """

    if isinstance(spec, dict):
        mode = spec.get("mode", "relative")
        try:
            mode_enum = ToleranceMode.parse(mode)
        except ValueError as exc:
            raise CommandFileError(f"{where}: {exc}") from exc
        return Tolerance(
            mode=mode_enum,
            value=float(spec.get("value", 0.0)),
            floor=float(spec.get("floor", 0.0)),
            use_old_floor=bool(spec.get("use_old_floor", False)),
        )
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return Tolerance(mode=ToleranceMode.RELATIVE, value=float(spec), floor=0.0)
    if isinstance(spec, str):
        parts = spec.split()
        if len(parts) == 1:
            try:
                return Tolerance(ToleranceMode.RELATIVE, float(parts[0]), 0.0)
            except ValueError:
                mode_enum = _parse_yaml_mode(parts[0], where)
                return Tolerance(mode=mode_enum, value=0.0, floor=0.0)
        if len(parts) == 2:
            mode_enum = _parse_yaml_mode(parts[0], where)
            return Tolerance(mode=mode_enum, value=float(parts[1]), floor=0.0)
        raise CommandFileError(f"{where}: cannot parse tolerance shorthand {spec!r}")
    raise CommandFileError(f"{where}: invalid tolerance specification {spec!r}")


def _parse_yaml_mode(token: str, where: str) -> ToleranceMode:
    try:
        return ToleranceMode.parse(token)
    except ValueError as exc:
        raise CommandFileError(f"{where}: {exc}") from exc


def _tolerance_to_yaml(tol: Tolerance) -> dict[str, Any]:
    """Serialize a :class:`Tolerance` to an explicit YAML mapping."""

    out: dict[str, Any] = {
        "mode": tol.mode.value,
        "value": float(tol.value),
        "floor": float(tol.floor),
    }
    if tol.use_old_floor:
        out["use_old_floor"] = True
    return out


class YamlCommandFileReader(CommandFileReader):
    """Reader for the modern YAML command-file format (preferred).

    The document is a mapping with these optional top-level sections::

        version: 1
        tolerances:
          default:    {mode: relative, value: 1.0e-6, floor: 0.0}
          coordinate: {mode: absolute, value: 1.0e-6}
          time:       {mode: relative, value: 1.0e-6, floor: 1.0e-15}
          attribute:  {mode: relative, value: 1.0e-6}
          categories:                 # per-category default tolerances
            nodal:   {mode: absolute, value: 1.0e-7}
          variables:                  # per-variable overrides (highest priority)
            DISPLX:  {mode: relative, value: 1.0e-9}
        variables:
          ignore_case: true
          exclude: [VELZ]
          include:                    # per-category include lists
            nodal: [DISPLX, DISPLY]
            global: all               # 'all' (or true) => compare all
        coordinates:  {compare: true}
        attributes:   {compare: true}
        mesh_matching:
          enabled: false
          tolerance: 1.0e-6
          require_unique: true
        time:
          start: 1
          stop: -1
          increment: 1
          step_offset: 0
          exclude_steps: [2, 4]
          value_scale: 1.0
          value_offset: 0.0
          interpolate: false
        report: {show_all: false}

    Any tolerance may be given as an explicit mapping, a shorthand string
    (``"absolute 1e-8"`` / ``"1e-6"``), or a bare number.
    """

    format_name = "yaml"

    @classmethod
    def sniff(cls, path: Path, head: str) -> bool:
        if path.suffix.lower() in (".yaml", ".yml"):
            return True
        # Content sniff: parses as a YAML mapping with a known top-level key.
        try:
            doc = yaml.safe_load(head)
        except yaml.YAMLError:
            return False
        if not isinstance(doc, dict):
            return False
        known = {
            "version",
            "tolerances",
            "variables",
            "coordinates",
            "attributes",
            "mesh_matching",
            "time",
            "report",
        }
        return bool(known.intersection(doc))

    def read(self) -> CommandFileResult:
        text = self.path.read_text()
        try:
            doc = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise CommandFileError(f"invalid YAML in {self.path}: {exc}") from exc
        if doc is None:
            doc = {}
        if not isinstance(doc, dict):
            raise CommandFileError(
                f"{self.path}: top-level YAML must be a mapping, got {type(doc).__name__}"
            )
        return _diff_options_from_yaml(doc)


def _diff_options_from_yaml(doc: dict[str, Any]) -> CommandFileResult:
    warnings: list[str] = []
    kwargs: dict[str, Any] = {}

    tol_sec = doc.get("tolerances") or {}
    if not isinstance(tol_sec, dict):
        raise CommandFileError("'tolerances' must be a mapping")
    if "default" in tol_sec and tol_sec["default"] is not None:
        kwargs["default_tolerance"] = _tolerance_from_yaml(tol_sec["default"], "tolerances.default")
    if "coordinate" in tol_sec and tol_sec["coordinate"] is not None:
        kwargs["coordinate_tolerance"] = _tolerance_from_yaml(
            tol_sec["coordinate"], "tolerances.coordinate"
        )
    if "time" in tol_sec and tol_sec["time"] is not None:
        kwargs["time_tolerance"] = _tolerance_from_yaml(tol_sec["time"], "tolerances.time")
    if "attribute" in tol_sec and tol_sec["attribute"] is not None:
        kwargs["attribute_tolerance"] = _tolerance_from_yaml(
            tol_sec["attribute"], "tolerances.attribute"
        )
    categories = tol_sec.get("categories") or {}
    if not isinstance(categories, dict):
        raise CommandFileError("'tolerances.categories' must be a mapping")
    for key, spec in categories.items():
        if spec is None:
            continue
        entry = _YAML_CATEGORIES.get(str(key).lower())
        if entry is None:
            raise CommandFileError(f"unknown tolerance category {key!r}")
        _ent, field_name = entry
        kwargs[field_name] = _tolerance_from_yaml(spec, f"tolerances.categories.{key}")
    variable_tolerances: dict[str, Tolerance] = {}
    var_tols = tol_sec.get("variables") or {}
    if not isinstance(var_tols, dict):
        raise CommandFileError("'tolerances.variables' must be a mapping")
    for name, spec in var_tols.items():
        variable_tolerances[str(name)] = _tolerance_from_yaml(spec, f"tolerances.variables.{name}")
    if variable_tolerances:
        kwargs["variable_tolerances"] = variable_tolerances

    var_sec = doc.get("variables") or {}
    if not isinstance(var_sec, dict):
        raise CommandFileError("'variables' must be a mapping")
    if "ignore_case" in var_sec:
        kwargs["ignore_case"] = bool(var_sec["ignore_case"])
    if "exclude" in var_sec:
        exclude = var_sec["exclude"] or []
        if not isinstance(exclude, list):
            raise CommandFileError("'variables.exclude' must be a list")
        kwargs["exclude"] = frozenset(str(x) for x in exclude)
    include = var_sec.get("include") or {}
    if not isinstance(include, dict):
        raise CommandFileError("'variables.include' must be a mapping")
    include_variables: dict[Entity, frozenset[str]] = {}
    all_categories: set[Entity] = set()
    for key, names in include.items():
        entry = _YAML_CATEGORIES.get(str(key).lower())
        if entry is None:
            raise CommandFileError(f"unknown include category {key!r}")
        ent = entry[0]
        if names in (True, "all", "ALL", "(all)"):
            all_categories.add(ent)
        elif isinstance(names, list):
            include_variables[ent] = frozenset(str(x) for x in names)
        else:
            raise CommandFileError(f"'variables.include.{key}' must be a list of names or 'all'")
    if include_variables:
        kwargs["include_variables"] = include_variables
    if all_categories:
        kwargs["all_categories"] = frozenset(all_categories)

    coord_sec = doc.get("coordinates") or {}
    if not isinstance(coord_sec, dict):
        raise CommandFileError("'coordinates' must be a mapping")
    if "compare" in coord_sec:
        kwargs["compare_coordinates"] = bool(coord_sec["compare"])

    attr_sec = doc.get("attributes") or {}
    if not isinstance(attr_sec, dict):
        raise CommandFileError("'attributes' must be a mapping")
    if "compare" in attr_sec:
        kwargs["compare_attributes"] = bool(attr_sec["compare"])

    mm_sec = doc.get("mesh_matching") or {}
    if not isinstance(mm_sec, dict):
        raise CommandFileError("'mesh_matching' must be a mapping")
    if "enabled" in mm_sec:
        kwargs["coordinate_matching"] = bool(mm_sec["enabled"])
    if "tolerance" in mm_sec:
        kwargs["matching_tolerance"] = float(mm_sec["tolerance"])
    if "require_unique" in mm_sec:
        kwargs["require_unique_mapping"] = bool(mm_sec["require_unique"])

    time_sec = doc.get("time") or {}
    if not isinstance(time_sec, dict):
        raise CommandFileError("'time' must be a mapping")
    if time_sec:
        exclude_steps = time_sec.get("exclude_steps") or []
        if not isinstance(exclude_steps, list):
            raise CommandFileError("'time.exclude_steps' must be a list")
        selection = TimeSelection(
            start=int(time_sec.get("start", 1)),
            stop=int(time_sec.get("stop", -1)),
            increment=int(time_sec.get("increment", 1)),
            time_step_offset=int(time_sec.get("step_offset", 0)),
            exclude_steps=frozenset(int(s) for s in exclude_steps),
            time_value_scale=float(time_sec.get("value_scale", 1.0)),
            time_value_offset=float(time_sec.get("value_offset", 0.0)),
            interpolating=bool(time_sec.get("interpolate", False)),
        )
        # Leave time_selection unset (None) when it is entirely default, so a
        # default DiffOptions round-trips exactly.  _effective_time_selection()
        # treats the two identically at diff time.
        if selection != TimeSelection():
            kwargs["time_selection"] = selection

    report_sec = doc.get("report") or {}
    if not isinstance(report_sec, dict):
        raise CommandFileError("'report' must be a mapping")
    if "show_all" in report_sec:
        kwargs["show_all"] = bool(report_sec["show_all"])

    return CommandFileResult(options=DiffOptions(**kwargs), warnings=warnings)


def diff_options_to_yaml(options: DiffOptions) -> str:
    """Serialize *options* to a YAML command-file string (full/explicit).

    Every field is written with its current value (self-documenting).  The
    output round-trips through :func:`read_command_file` /
    :class:`YamlCommandFileReader`.
    """

    ts = options._effective_time_selection()

    categories: dict[str, Any] = {}
    for key, (_ent, field_name) in _YAML_CATEGORIES.items():
        tol = getattr(options, field_name)
        if tol is not None:
            categories[key] = _tolerance_to_yaml(tol)

    variables_tols = {
        name: _tolerance_to_yaml(tol) for name, tol in options.variable_tolerances.items()
    }

    include: dict[str, Any] = {}
    for ent in sorted(options.all_categories, key=lambda e: e.value):
        include[_ENTITY_TO_YAML_CATEGORY.get(ent, ent.value)] = "all"
    for ent, names in options.include_variables.items():
        include[_ENTITY_TO_YAML_CATEGORY.get(ent, ent.value)] = sorted(names)

    doc: dict[str, Any] = {
        "version": 1,
        "tolerances": {
            "default": _tolerance_to_yaml(options.default_tolerance),
            "coordinate": _tolerance_to_yaml(options.coordinate_tolerance),
            "time": _tolerance_to_yaml(options.time_tolerance),
            "attribute": (
                _tolerance_to_yaml(options.attribute_tolerance)
                if options.attribute_tolerance is not None
                else None
            ),
            "categories": categories,
            "variables": variables_tols,
        },
        "variables": {
            "ignore_case": options.ignore_case,
            "exclude": sorted(options.exclude),
            "include": include,
        },
        "coordinates": {"compare": options.compare_coordinates},
        "attributes": {"compare": options.compare_attributes},
        "mesh_matching": {
            "enabled": options.coordinate_matching,
            "tolerance": float(options.matching_tolerance),
            "require_unique": options.require_unique_mapping,
        },
        "time": {
            "start": ts.start,
            "stop": ts.stop,
            "increment": ts.increment,
            "step_offset": ts.time_step_offset,
            "exclude_steps": sorted(ts.exclude_steps),
            "value_scale": ts.time_value_scale,
            "value_offset": ts.time_value_offset,
            "interpolate": ts.interpolating,
        },
        "report": {"show_all": options.show_all},
    }

    header = (
        "# exodusii diff options (YAML command file)\n"
        "# Preferred, modern replacement for the SEACAS exodiff command file.\n"
        "# Read with exodusii.read_command_file(); emit with DiffOptions.to_yaml().\n"
    )
    return header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
