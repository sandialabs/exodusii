# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Glob patterns for legacy File()/exo_file() and exoread (exodusii < 2)."""

import shutil
from io import StringIO
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import exodusii
from exodusii.cli.exoread import main as exoread_main
from exodusii.compat import ExodusIIFile
from exodusii.compat import ParallelExodusIIFile
from exodusii.core.files import expand_files


def _copy_noh(tmp_path: Path, datadir: Path) -> Path:
    for src in sorted(datadir.glob("noh.exo*")):
        shutil.copy(src, tmp_path / src.name)
    return tmp_path


def test_expand_files_sorted_and_missing(tmp_path: Path, datadir: Path) -> None:
    root = _copy_noh(tmp_path, datadir)

    names = expand_files(str(root / "noh.exo.3.*"))

    assert [Path(n).name for n in names] == ["noh.exo.3.0", "noh.exo.3.1", "noh.exo.3.2"]
    with pytest.raises(FileNotFoundError):
        expand_files(str(root / "missing.*"))


def test_file_glob_opens_parallel(tmp_path: Path, datadir: Path) -> None:
    root = _copy_noh(tmp_path, datadir)

    pattern = exodusii.File(str(root / "noh.exo.3.*"))
    explicit = exodusii.exo_file(*sorted(str(p) for p in root.glob("noh.exo.3.*")))
    single = exodusii.File(str(root / "noh.exo"))

    assert isinstance(pattern, ParallelExodusIIFile)
    assert isinstance(single, ExodusIIFile)
    assert pattern.num_elems() == explicit.num_elems() == single.num_elems()


def test_exoread_glob_describes_parallel_set(tmp_path: Path, datadir: Path) -> None:
    root = _copy_noh(tmp_path, datadir)
    serial, merged = StringIO(), StringIO()

    exoread_main([str(root / "noh.exo")], file=serial)
    exoread_main([str(root / "noh.exo.3.*")], file=merged)

    def count(text: str) -> str:
        return next(line for line in text.splitlines() if line.startswith("Num elements"))

    assert count(merged.getvalue()) == count(serial.getvalue())


def test_parallel_block_ids_ignore_unwritten_global_ids(tmp_path: Path, datadir: Path) -> None:
    # alegra per-rank files allocate el_blk_ids_global but leave it zero.
    for src in sorted(datadir.glob("edges.exo.4.*")):
        shutil.copy(src, tmp_path / src.name)
    files = sorted(tmp_path.glob("edges.exo.4.*"))
    for path in files:
        with netCDF4.Dataset(path, "a") as ds:
            ds["el_blk_ids_global"][:] = 0

    exo = exodusii.File(str(tmp_path / "edges.exo.4.*"))

    assert exo.get_element_block_ids().tolist() == [1, 2]
    assert np.asarray(exodusii.compute_element_centers(exo)).shape[0] == exo.num_elems()
