# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Base class for all ``python -m exodusii`` subcommands.

Each subcommand subclasses :class:`Command`, sets ``name`` (defaults to the
lowercase class name), implements :meth:`setup_parser` and :meth:`execute`,
and optionally provides a module-level ``main()`` function for standalone use::

    def main(argv=None, *, file=None):
        parser = argparse.ArgumentParser(prog="exodusii-inspect")
        Inspect.setup_parser(parser)
        args = parser.parse_args(argv)
        return Inspect().execute(parser, args)

The class also carries the small set of CLI utilities that every subcommand
needs: JSON emission, value coercion, and time-selector parsing.  Domain-
specific payload builders (block/set/variable stats, etc.) stay in
:mod:`exodusii.cli._common`.
"""

import argparse
import json
import sys
from typing import Any
from typing import TextIO

import numpy as np
import numpy.typing as npt

from exodusii.core.time import TimeSelector

__all__ = ["Command"]


class Command:
    """Base class for exodusii CLI subcommands.

    Subclasses set :attr:`name` and implement :meth:`setup_parser` and
    :meth:`execute`.  The class-level utility methods (:meth:`emit_json`,
    :meth:`jsonable`, :meth:`parse_time_selector`, :meth:`normalize_limit`,
    :meth:`array_stats`, :meth:`add_terse_argument`) are available to all
    subcommands without any import from ``_common``.
    """

    #: Subcommand name as it appears on the CLI.  Defaults to the lowercase
    #: class name when ``None``.
    name: str | None = None

    @staticmethod
    def setup_parser(parser: argparse.ArgumentParser) -> None:
        """Register arguments on *parser*.

        Called by :func:`exodusii.cli.main.main` with an already-created
        subparser, and by standalone ``main()`` functions with a freshly
        constructed ``ArgumentParser``.  Should not call
        ``parser.add_subparsers`` or ``subparsers.add_parser`` — that is the
        coordinator's responsibility.
        """

    def execute(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        *,
        file: TextIO | None = None,
    ) -> int:
        """Execute the subcommand.  Return an integer exit code."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared CLI utilities
    # ------------------------------------------------------------------

    @staticmethod
    def add_terse_argument(parser: argparse.ArgumentParser) -> None:
        """Add ``--terse`` to *parser*.

        Call this at the end of :meth:`setup_parser` for any subcommand that
        supports compact JSON output::

            @staticmethod
            def setup_parser(parser):
                parser.add_argument("file", ...)
                Command.add_terse_argument(parser)
        """
        parser.add_argument(
            "--terse",
            action="store_true",
            default=False,
            help="Emit compact JSON with no extra whitespace. [default: indented JSON]",
        )

    @staticmethod
    def emit_json(payload: dict[str, Any], *, terse: bool, file: TextIO | None = None) -> None:
        """Emit *payload* as JSON to *file* (default ``stdout``).

        ``terse=True`` removes all optional whitespace; the default is
        two-space-indented JSON.
        """
        stream = file or sys.stdout
        converted = Command.jsonable(payload)
        if terse:
            json.dump(converted, stream, separators=(",", ":"))
        else:
            json.dump(converted, stream, indent=2)
        stream.write("\n")

    @staticmethod
    def jsonable(value: Any) -> Any:
        """Recursively convert Python/NumPy values to JSON-serializable form."""
        from collections.abc import Mapping
        from pathlib import Path

        if isinstance(value, np.ndarray):
            return Command.jsonable(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(k): Command.jsonable(v) for k, v in value.items()}
        if isinstance(value, tuple | list):
            return [Command.jsonable(item) for item in value]
        return value

    @staticmethod
    def parse_time_selector(value: str | None) -> TimeSelector:
        """Parse a CLI time-selector string.

        Supported forms::

            first          first time step
            last           last time step
            index:N        zero-based Python time index
            step:N         one-based Exodus time step
            0.25           nearest physical time value
        """
        if value is None:
            return None

        text = value.strip()
        key = text.lower()

        if key in {"first", "last"}:
            return key  # type: ignore[return-value]

        if key.startswith("index:"):
            return int(key.split(":", 1)[1])

        if key.startswith("step:"):
            step = int(key.split(":", 1)[1])
            if step < 1:
                raise ValueError("step:N time selector must use a one-based positive step")
            return step - 1

        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(
                "time must be 'first', 'last', a physical float time, 'index:N', or 'step:N'"
            ) from exc

    @staticmethod
    def normalize_limit(value: int) -> int | None:
        """Normalize a row-limit value.  Negative means unlimited (``None``)."""
        if value < 0:
            return None
        return value

    @staticmethod
    def array_stats(values: npt.ArrayLike) -> dict[str, Any]:
        """Return JSON-safe numeric statistics for an array."""
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
