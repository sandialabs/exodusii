# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Write repo-local agent guidance files and skills for exodusii.

This backs ``python -m exodusii agent bootstrap`` and mirrors
``canary agent bootstrap``.  Three target layouts are supported:

``generic``
    ``.agent/exodusii/BOOTSTRAP.md``, ``.agent/exodusii/INDEX.md`` and
    ``.agent/exodusii/skills/<skill>.md``.
``opencode``
    The generic layout plus ``.opencode/skills/<skill>/SKILL.md``.
``claude``
    ``CLAUDE.md`` plus ``.claude/skills/<skill>/SKILL.md``.

``brief=True`` writes only the concise bootstrap file (``BOOTSTRAP.md`` or
``CLAUDE.md``).  Skill content comes from the installed static skills dataset
(the same data served by ``python -m exodusii learn skills``).

Files previously written by this command are refreshed in place.  Any other
existing file is left alone unless ``force=True``.  ``append=True`` (claude
target only) inserts or refreshes a delimited exodusii guidance block in an
existing ``CLAUDE.md`` without touching the rest of the file.
"""

import re
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "BOOTSTRAP_TARGETS",
    "CLAUDE_GUIDANCE_BEGIN",
    "CLAUDE_GUIDANCE_END",
    "CORE_BOOTSTRAP_SKILLS",
    "bootstrap_file_map",
    "render_bootstrap_markdown",
    "render_claude_guidance_markdown",
    "render_index_markdown",
    "skill_to_markdown",
    "write_agent_bootstrap",
]

BOOTSTRAP_TARGETS = ("generic", "opencode", "claude")

CORE_BOOTSTRAP_SKILLS = (
    "exodusii-agent-orientation",
    "exodusii-querying",
    "exodusii-python-api",
    "exodusii-comparison",
)

CLAUDE_GUIDANCE_PATH = Path("CLAUDE.md")
CLAUDE_GUIDANCE_BEGIN = "<!-- BEGIN EXODUSII GUIDANCE -->"
CLAUDE_GUIDANCE_END = "<!-- END EXODUSII GUIDANCE -->"

GENERIC_ROOT = Path(".agent") / "exodusii"
BOOTSTRAP_TITLE = "# exodusii Agent Bootstrap\n"
INDEX_TITLE = "# exodusii Agent Guide\n"

_TEMPLATE_TOKEN_RE = re.compile(r"@@[A-Z0-9_]+@@")


def write_agent_bootstrap(
    target: str,
    *,
    brief: bool = False,
    force: bool = False,
    append: bool = False,
    root: str | Path = ".",
) -> list[Path]:
    """Write the agent bootstrap files for *target* below *root*.

    Args:
        target: One of :data:`BOOTSTRAP_TARGETS`.
        brief: Write only the concise bootstrap file.
        force: Overwrite existing files that were not written by this command.
        append: (claude only) Insert or refresh the exodusii guidance block in
            an existing ``CLAUDE.md`` instead of refusing to touch it.
        root: Directory to write into.  Defaults to the current directory.

    Returns:
        The written paths, relative to *root*.

    Raises:
        ValueError: If *target* is unknown, or *append* is used with a target
            other than ``claude``.
        FileExistsError: If writing would clobber unmanaged files and
            *force* is not set.
    """
    if append and target != "claude":
        raise ValueError("--append is only supported for the 'claude' target")
    base = Path(root)
    files = bootstrap_file_map(target, brief=brief)
    ensure_bootstrap_targets_available(base, files, force=force, append=append)

    for relpath, text in files.items():
        path = base / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        if relpath == CLAUDE_GUIDANCE_PATH and append:
            append_or_refresh_claude_guidance(path, text)
        else:
            path.write_text(text, encoding="utf-8")
    return list(files)


def bootstrap_file_map(target: str, *, brief: bool) -> dict[Path, str]:
    """Return ``{relative_path: text}`` for every file *target* would write."""
    if target == "generic":
        return generic_bootstrap_files(brief=brief)
    if target == "opencode":
        if brief:
            return generic_bootstrap_files(brief=True)
        files = generic_bootstrap_files(brief=False)
        files.update(render_skill_dirs(Path(".opencode") / "skills"))
        return files
    if target == "claude":
        files = {CLAUDE_GUIDANCE_PATH: render_claude_guidance_markdown()}
        if not brief:
            files.update(render_skill_dirs(Path(".claude") / "skills"))
        return files
    choices = ", ".join(BOOTSTRAP_TARGETS)
    raise ValueError(f"unsupported agent bootstrap target {target!r}; expected one of: {choices}")


def generic_bootstrap_files(*, brief: bool) -> dict[Path, str]:
    files: dict[Path, str] = {GENERIC_ROOT / "BOOTSTRAP.md": render_bootstrap_markdown()}
    if brief:
        return files
    files[GENERIC_ROOT / "INDEX.md"] = render_index_markdown()
    for skill in curated_skills():
        files[GENERIC_ROOT / "skills" / f"{skill['name']}.md"] = skill_to_markdown(skill)
    return files


def render_skill_dirs(root: Path) -> dict[Path, str]:
    """Render skills in the ``<root>/<skill-name>/SKILL.md`` layout."""
    return {
        root / skill["name"] / "SKILL.md": skill_to_markdown(skill) for skill in curated_skills()
    }


def ensure_bootstrap_targets_available(
    base: Path, files: dict[Path, str], *, force: bool, append: bool = False
) -> None:
    claude_md = base / CLAUDE_GUIDANCE_PATH
    if CLAUDE_GUIDANCE_PATH in files and claude_md.exists() and not append and not force:
        raise FileExistsError(
            "CLAUDE.md exists. To overwrite it, re-run with --force. "
            "To append or refresh exodusii guidance, re-run with --append."
        )

    conflicts = [
        base / relpath
        for relpath, text in files.items()
        if relpath != CLAUDE_GUIDANCE_PATH
        and (base / relpath).exists()
        and not can_refresh_managed_file(base / relpath, text)
    ]
    if conflicts and not force:
        lines = ["exodusii agent bootstrap would overwrite existing files:"]
        lines.extend(f"  {path}" for path in conflicts)
        lines.append("Re-run with --force to overwrite them.")
        raise FileExistsError("\n".join(lines))


def can_refresh_managed_file(path: Path, new_text: str) -> bool:
    """True if *path* was written by this command and may be refreshed in place."""
    try:
        existing = path.read_text(encoding="utf-8")
    except OSError:
        return False

    if existing == new_text:
        return True
    if path.name == "BOOTSTRAP.md":
        return existing.startswith(BOOTSTRAP_TITLE)
    if path.name == "INDEX.md":
        return existing.startswith(INDEX_TITLE)

    expected_name = extract_skill_name(new_text)
    if expected_name is None:
        return False
    return extract_skill_name(existing) == expected_name


def extract_skill_name(text: str) -> str | None:
    match = re.search(r"^name:\s+(.+)$", text, flags=re.MULTILINE)
    if match is None:
        return None
    return match.group(1).strip()


def append_or_refresh_claude_guidance(path: Path, guidance: str) -> None:
    """Insert *guidance* into *path*, replacing an existing delimited block."""
    if not path.exists():
        path.write_text(guidance, encoding="utf-8")
        return

    existing = path.read_text(encoding="utf-8")
    start = existing.find(CLAUDE_GUIDANCE_BEGIN)
    end = existing.find(CLAUDE_GUIDANCE_END)

    if start != -1 and end != -1 and start < end:
        block_end = end + len(CLAUDE_GUIDANCE_END)
        while block_end < len(existing) and existing[block_end] in " \t":
            block_end += 1
        if block_end < len(existing) and existing[block_end] == "\n":
            block_end += 1
        updated = existing[:start] + guidance + existing[block_end:]
    else:
        separator = "" if existing.endswith("\n") else "\n"
        updated = f"{existing}{separator}\n{guidance}"

    path.write_text(updated, encoding="utf-8")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def curated_skills() -> list[dict[str, Any]]:
    """Return the installed skill objects sorted by name."""
    from exodusii.cli.learn import load_skill_dataset

    data = load_skill_dataset()
    return [data[name] for name in sorted(data)]


def skill_to_markdown(skill: dict[str, Any]) -> str:
    """Convert a skill object to ``SKILL.md`` Markdown with YAML frontmatter."""
    body = skill.get("body", "")
    if not isinstance(body, str):
        raise ValueError("skill object must contain a string field: body")
    frontmatter = {key: value for key, value in skill.items() if key != "body"}
    name = frontmatter.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("skill object must contain a non-empty string field: name")
    if not isinstance(frontmatter.get("description"), str):
        raise ValueError("skill object must contain a string field: description")

    frontmatter_text = yaml.safe_dump(
        frontmatter, sort_keys=False, allow_unicode=True, default_flow_style=False
    ).rstrip()
    markdown = f"---\n{frontmatter_text}\n---\n\n{body}"
    if not markdown.endswith("\n"):
        markdown += "\n"
    return markdown


def render_template(name: str, replacements: dict[str, str]) -> str:
    template = (
        resources.files("exodusii")
        .joinpath("data")
        .joinpath("agent")
        .joinpath(name)
        .read_text(encoding="utf-8")
    )
    for key, value in replacements.items():
        template = template.replace(f"@@{key}@@", value)

    remaining = sorted(set(_TEMPLATE_TOKEN_RE.findall(template)))
    if remaining:
        raise ValueError(f"unresolved agent template token(s) in {name}: {', '.join(remaining)}")
    if not template.endswith("\n"):
        template += "\n"
    return template


def render_skill_map() -> str:
    return "\n".join(f"- `{s['name']}`: {s['description']}" for s in curated_skills())


def render_core_bootstrap_skills() -> str:
    return "\n".join(f"- `{name}`" for name in CORE_BOOTSTRAP_SKILLS)


def _replacements() -> dict[str, str]:
    return {
        "SKILL_MAP": render_skill_map(),
        "CORE_BOOTSTRAP_SKILLS": render_core_bootstrap_skills(),
    }


def render_bootstrap_markdown() -> str:
    return render_template("generic-bootstrap.md.in", _replacements())


def render_index_markdown() -> str:
    return render_template("generic-index.md.in", {"SKILL_MAP": render_skill_map()})


def render_claude_guidance_markdown() -> str:
    return render_template("claude.md.in", _replacements())
