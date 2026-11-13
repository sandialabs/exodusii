# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""``python -m exodusii agent`` -- bootstrap agent-facing guidance and skills.

Usage::

    python -m exodusii agent bootstrap                # .agent/exodusii/...
    python -m exodusii agent bootstrap opencode       # + .opencode/skills/...
    python -m exodusii agent bootstrap claude         # CLAUDE.md + .claude/skills/...
    python -m exodusii agent bootstrap claude --append  # merge into existing CLAUDE.md
"""

import argparse
import sys
from pathlib import Path
from typing import TextIO

from exodusii.agent.bootstrap import BOOTSTRAP_TARGETS
from exodusii.agent.bootstrap import write_agent_bootstrap
from exodusii.cli._command import Command

__all__ = ["Agent", "main"]


class Agent(Command):
    """Bootstrap agent-facing exodusii guidance and skills."""

    name = "agent"

    @staticmethod
    def setup_parser(parser: argparse.ArgumentParser) -> None:
        """Register the ``agent`` subparser and its ``bootstrap`` action."""
        actions = parser.add_subparsers(dest="agent_command", metavar="SUBCMD")

        bootstrap = actions.add_parser("bootstrap", help="Write repo-local agent guidance files.")
        bootstrap.add_argument(
            "target",
            nargs="?",
            choices=BOOTSTRAP_TARGETS,
            default="generic",
            help="Bootstrap target layout. [default: generic]",
        )
        bootstrap.add_argument(
            "--brief", action="store_true", help="Write only the concise bootstrap file."
        )
        bootstrap.add_argument(
            "--force", action="store_true", help="Overwrite existing unmanaged files."
        )
        bootstrap.add_argument(
            "--append",
            action="store_true",
            help="Append or refresh exodusii guidance in an existing CLAUDE.md (claude only).",
        )
        bootstrap.add_argument(
            "-C",
            "--directory",
            default=".",
            help="Directory to write into. [default: current directory]",
        )
        Command.add_terse_argument(bootstrap)

        Command.add_terse_argument(parser)

    def execute(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        *,
        file: TextIO | None = None,
    ) -> int:
        """Run ``agent bootstrap`` and emit a JSON summary of the written files."""
        terse = bool(getattr(args, "terse", False))
        if getattr(args, "agent_command", None) != "bootstrap":
            self.emit_json(
                {
                    "ok": True,
                    "command": "agent",
                    "purpose": "Bootstrap agent-facing exodusii guidance and skills.",
                    "usage": {
                        "bootstrap": (
                            "python -m exodusii agent bootstrap [generic|opencode|claude] "
                            "[--brief] [--force] [--append] [-C DIR]"
                        )
                    },
                },
                terse=terse,
                file=file,
            )
            return 0
        try:
            written = write_agent_bootstrap(
                args.target,
                brief=args.brief,
                force=args.force,
                append=args.append,
                root=args.directory,
            )
        except Exception as exc:
            self.emit_json(
                {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}},
                terse=terse,
                file=file or sys.stderr,
            )
            return 1
        self.emit_json(
            {
                "ok": True,
                "command": "agent bootstrap",
                "target": args.target,
                "directory": str(Path(args.directory).absolute()),
                "files": [str(p) for p in written],
            },
            terse=terse,
            file=file,
        )
        return 0


def main(argv: list[str] | None = None, *, file: TextIO | None = None) -> int:
    """Standalone entry point for the ``agent`` subcommand."""
    parser = argparse.ArgumentParser(prog="python -m exodusii agent", description=Agent.__doc__)
    Agent.setup_parser(parser)
    args = parser.parse_args(argv)
    return Agent().execute(parser, args, file=file)


if __name__ == "__main__":
    raise SystemExit(main())
