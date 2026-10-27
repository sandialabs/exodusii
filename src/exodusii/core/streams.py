# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Output stream helpers."""

import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import TextIO

OutputTarget = TextIO | str | os.PathLike[str] | None
"""A writable text stream, a path to open for writing, or ``None`` for stdout."""


@contextmanager
def output_stream(file: OutputTarget = None) -> Generator[TextIO]:
    """Yield a writable text stream for ``file``.

    ``None`` selects ``sys.stdout``; a ``str`` or path-like value is opened
    for writing (and closed on exit), matching exodusii < 2, which accepted
    a filename for ``print(file=...)`` and ``describe(file=...)``.  Any other
    object is assumed to be a writable stream and is left open.
    """

    if file is None:
        yield sys.stdout
    elif isinstance(file, str | os.PathLike):
        with open(file, "w", encoding="utf-8") as stream:
            yield stream
    else:
        yield file


__all__ = ["OutputTarget", "output_stream"]
