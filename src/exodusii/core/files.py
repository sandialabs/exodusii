# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""File-name expansion shared by the CLI and legacy factories."""

import glob
import os
from pathlib import Path


def expand_files(*files: str | os.PathLike[str]) -> list[str]:
    """Expand shell-style patterns the way exodusii < 2 did.

    Each argument is passed through :func:`glob.glob`; an argument that names
    an existing file is kept even if it contains glob metacharacters.  The
    matches of each pattern are sorted (``glob`` order is filesystem
    dependent) and concatenated in argument order.

    Raises
    ------
    FileNotFoundError
        If an argument matches nothing.
    """

    found: list[str] = []
    for file in files:
        text = os.fspath(file)
        if Path(text).exists():
            found.append(text)
            continue
        matches = sorted(glob.glob(text))
        if not matches:
            raise FileNotFoundError(text)
        found.extend(matches)
    return found


__all__ = ["expand_files"]
