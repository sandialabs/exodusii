# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Modern Python interface for Exodus II finite element databases.

exodusii provides a clean, NumPy-based API for reading, writing, querying,
comparing, and inspecting Sandia Exodus II finite-element databases, which are
NetCDF files produced by simulation codes such as Sierra, Alegra, and many
open-source solvers.

Primary entry points
--------------------
:class:`ExodusFile`
    Read an Exodus database: coordinates, connectivity, result variables,
    sets, blocks, and time history.
:class:`ExodusWriter`
    Create a new Exodus database from NumPy arrays.
:class:`ParallelExodusFile`
    Aggregate decomposed parallel (Nemesis) component files into a single
    logical database by merging global ID maps.
:func:`diff`
    Compare two databases with exodiff-style per-variable tolerance checking.
:func:`query`
    Extract result variables as structured NumPy arrays.
:func:`allclose` / :func:`similar`
    Quick data-equality and layout-similarity checks.
:func:`copy` / :func:`copy_file`
    Copy database contents between files.

Quick start
-----------
Read a database and extract values::

    import exodusii

    with exodusii.ExodusFile.open("mesh.exo") as exo:
        times  = exo.times()
        coords = exo.coordinates()
        temp   = exo.values("TEMP", on="node", time="last")

Write a minimal mesh::

    import numpy as np
    import exodusii

    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    with exodusii.ExodusWriter.create("out.exo") as w:
        w.initialize("title", 2, 4, 1, element_blocks=1)
        w.write_coordinates(coords)
        w.define_element_block(10, "quad", [[1, 2, 3, 4]])
        w.define_node_variables(["TEMP"])
        w.write_time(0.0)
        w.write_node_values("TEMP", [100.0, 200.0, 300.0, 400.0])

Compare two databases::

    result = exodusii.diff("gold.exo", "test.exo")
    if not result:
        for vd in result.variable_diffs:
            print(f"{vd.entity}/{vd.name}: max_delta={vd.max_delta:.3e}")

Notes
-----
Requires Python >= 3.13 and the ``netCDF4`` package.

Exodus II is a finite-element data model developed at Sandia National
Laboratories.  For the binary format specification see the SEACAS project
at https://github.com/sandialabs/seacas.
"""

import importlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

from exodusii.api.compare import ComparisonResult
from exodusii.api.compare import allclose
from exodusii.api.compare import similar
from exodusii.api.copy import copy
from exodusii.api.copy import copy_file
from exodusii.api.diff import DiffOptions
from exodusii.api.diff import DiffResult
from exodusii.api.diff import TimeSelection
from exodusii.api.diff import VariableDiff
from exodusii.api.diff import diff
from exodusii.api.file import ExodusFile
from exodusii.api.lineout import Lineout
from exodusii.api.lineout import lineout
from exodusii.api.parallel import ParallelExodusFile
from exodusii.api.query import QueryResult
from exodusii.api.query import print_query
from exodusii.api.query import query
from exodusii.api.region_reduce import RegionMassResult
from exodusii.api.region_reduce import RegionStatsResult
from exodusii.api.region_reduce import region_mass
from exodusii.api.region_reduce import region_stats
from exodusii.api.writer import ExodusWriter
from exodusii.compat import ExodusIIFile
from exodusii.compat import File
from exodusii.compat import MFExodusIIFile
from exodusii.compat import ParallelExodusIIFile
from exodusii.compat import exodusii_file
from exodusii.compat import parallel_exodusii_file
from exodusii.compat import write_globals
from exodusii.core.tolerance import Tolerance
from exodusii.core.tolerance import ToleranceMode

region = importlib.import_module("exodusii.region")

exo_file = File

try:
    __version__ = version("exodusii")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = [
    "ComparisonResult",
    "DiffOptions",
    "DiffResult",
    "ExodusFile",
    "ExodusIIFile",
    "ExodusWriter",
    "File",
    "Lineout",
    "MFExodusIIFile",
    "ParallelExodusFile",
    "ParallelExodusIIFile",
    "QueryResult",
    "RegionMassResult",
    "RegionStatsResult",
    "TimeSelection",
    "Tolerance",
    "ToleranceMode",
    "VariableDiff",
    "__version__",
    "allclose",
    "copy",
    "copy_file",
    "diff",
    "exo_file",
    "exodusii_file",
    "lineout",
    "parallel_exodusii_file",
    "print_query",
    "query",
    "region",
    "region_mass",
    "region_stats",
    "similar",
    "write_globals",
]
