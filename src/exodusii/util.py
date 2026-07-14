# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Legacy utility helpers.

These functions preserve the historically public ``exodusii.util`` module while
delegating modern string and geometry operations to the refreshed implementation
where possible.
"""

import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import TextIO

import numpy as np
import numpy.typing as npt

from exodusii.core.strings import decode_text
from exodusii.core.strings import stringify
from exodusii.mesh.geometry import connected_average

string_kinds = ("U", "S")
string_types = (str,)


def decode(value: Any) -> str:
    """Decode one text-like value."""

    return decode_text(value)


def index(array: Sequence[Any] | npt.ArrayLike, value: Any) -> int:
    """Return the zero-based index of ``value`` in ``array``."""

    if isinstance(array, list | tuple):
        return array.index(value)

    values = np.asarray(array)
    matches = np.nonzero(values == value)[0]
    if not len(matches):
        raise ValueError(f"{value} is not in array")

    return int(matches[0])


def is_exe(path: str | os.PathLike[str]) -> bool:
    """Return true if ``path`` is an executable file."""

    return os.path.isfile(path) and os.access(path, os.X_OK)


def which(*names: str) -> str:
    """Return the first executable found in ``PATH``."""

    for name in names:
        found = shutil.which(name)
        if found is not None:
            return found

    if not names:
        raise ValueError("no executable name provided")

    raise ValueError(f"Required executable {names[0]} not found. Make sure it is in your path")


@contextmanager
def working_dir(dirname: str | os.PathLike[str]) -> Iterator[None]:
    """Temporarily change the working directory."""

    cwd = Path.cwd()
    os.chdir(dirname)
    try:
        yield
    finally:
        os.chdir(cwd)


def epu(*files: str | os.PathLike[str]) -> str | None:
    """Concatenate Exodus files using the external ``epu`` executable.

    This is retained for compatibility. The refreshed Python-native parallel
    writer should be preferred when possible.
    """

    if not files:
        return None
    if len(files) == 1:
        return str(files[0])

    epu_exe = which("epu")
    workdir = Path(files[0]).parent
    if not workdir.is_dir():
        raise ValueError(f"{workdir} is not a directory")

    with working_dir(workdir):
        basenames = [Path(file).name for file in files]
        for basename in basenames:
            if not Path(basename).exists():
                raise ValueError(f"{basename} is not a file")

        parts = basenames[0].split(".")
        try:
            base, suffix, _part, _count = parts
        except ValueError as exc:
            raise ValueError("Expected files `base.suf.#p.#n`") from exc

        log = Path(".epu.log")
        with log.open("w") as stream:
            process = subprocess.run(
                [epu_exe, "-auto", basenames[0]],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )

        if process.returncode != 0:
            visible_log = Path("epu.log")
            log.rename(visible_log)
            raise SystemExit(f"Exodus file concatenation failed, see {visible_log}")

        log.unlink(missing_ok=True)

    joined = workdir / f"{base}.{suffix}"
    target = Path(f"{base}.{suffix}")
    if joined.resolve() != target.resolve():
        joined.rename(target)

    return str(target)


def compute_connected_average(
    conn: npt.ArrayLike, values: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Legacy alias for :func:`exodusii.mesh.geometry.connected_average`."""

    return connected_average(conn, values)


def streamify(file: str | os.PathLike[str] | TextIO | None) -> tuple[TextIO | None, bool]:
    """Return ``(stream, owned)`` for a path or stream."""

    if file is None:
        return None, False
    if isinstance(file, str | os.PathLike):
        return open(file, "w", encoding="utf-8"), True
    return file, False


def fmt_join(*, fmt: str, items: Sequence[Any], sep: str = " ") -> str:
    """Join formatted items."""

    return sep.join(fmt % item for item in items)


def fuzzy_compare(arg1: str, arg2: str) -> bool:
    """Compare strings ignoring case and treating ``-``/``_`` as spaces."""

    regex = re.compile(r"[-_]")
    transform = lambda value: " ".join(regex.sub(" ", value).split()).lower()
    return transform(arg1) == transform(arg2)


def find_index(sequence: Sequence[str], arg: str, strict: bool = True) -> int | None:
    """Find ``arg`` in ``sequence`` using strict or fuzzy comparison."""

    for i, item in enumerate(sequence):
        if strict and arg == item:
            return i
        if not strict and fuzzy_compare(arg, item):
            return i

    return None


def find_nearest(array: npt.ArrayLike, value: float) -> tuple[int, float]:
    """Return index and value nearest to ``value``."""

    values = np.asarray(array, dtype=np.float64)
    idx = int(np.abs(values - value).argmin())
    return idx, float(values[idx])


def check_bounds(array: npt.ArrayLike, value: float, tol: float = 1.0e-12) -> bool:
    """Return true if value is strictly inside array bounds with tolerance."""

    values = np.asarray(array, dtype=np.float64)
    if values.size == 0:
        return False

    lower = float(np.amin(values))
    upper = float(np.amax(values))
    return lower + tol < value < upper - tol


def contains(array: npt.ArrayLike, value: Any) -> bool:
    """Return true if ``value`` is contained in ``array``."""

    return value in np.asarray(array).tolist()


__all__ = [
    "check_bounds",
    "compute_connected_average",
    "contains",
    "decode",
    "epu",
    "find_index",
    "find_nearest",
    "fmt_join",
    "fuzzy_compare",
    "index",
    "is_exe",
    "streamify",
    "string_kinds",
    "string_types",
    "stringify",
    "which",
    "working_dir",
]
