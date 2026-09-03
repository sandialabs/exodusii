Overview
========

What is Exodus II?
------------------

`Exodus II <https://sandialabs.github.io/seacas-docs/sphinx/html/index.html>`_
is a finite-element data model and file format developed at Sandia National
Laboratories.  Exodus files are `NetCDF
<https://www.unidata.ucar.edu/software/netcdf/>`_ files that store:

* **Mesh topology** — nodal coordinates, element blocks with connectivity
  tables, and optional edge and face blocks.
* **Sets** — node sets, side sets, edge sets, face sets, and element sets,
  each with optional distribution factors.
* **Result variables** — global, nodal, element, edge, face, and set
  variables at one or more time steps.
* **Maps** — optional node and element numbering maps used by parallel
  (Nemesis) decompositions.

Exodus files are produced by Sierra, Alegra, OpenFOAM (via converter), and
many other simulation codes, and are consumed by VisIt, ParaView, SEACAS
tools, and — now — exodusii.

What does exodusii provide?
----------------------------

exodusii gives Python users a clean, NumPy-based API for every major Exodus
workflow:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Task
     - API entry point
   * - **Read** a database
     - :class:`~exodusii.ExodusFile`
   * - **Write** a new database
     - :class:`~exodusii.ExodusWriter`
   * - **Aggregate** decomposed parallel files
     - :class:`~exodusii.ParallelExodusFile`
   * - **Compare** two databases (exodiff-style)
     - :func:`~exodusii.diff`, :class:`~exodusii.DiffOptions`
   * - **Query** variables as NumPy arrays
     - :func:`~exodusii.query`
   * - **Copy** between files
     - :func:`~exodusii.copy_file`
   * - **Check** data equality / layout similarity
     - :func:`~exodusii.allclose`, :func:`~exodusii.similar`
   * - **Mesh geometry** (centers, volumes)
     - :mod:`exodusii.mesh`

Design principles
-----------------

* **No compiled extensions.** exodusii is pure Python and relies on the
  ``netCDF4`` package for file I/O.
* **NumPy everywhere.** Coordinates, connectivity, and result arrays are
  always returned as ``numpy.ndarray``.
* **Context-manager safety.**  :class:`~exodusii.ExodusFile` and
  :class:`~exodusii.ExodusWriter` implement ``__enter__`` / ``__exit__``
  so files are always closed even on error.
* **Matched semantics with SEACAS.**  Indexing conventions (1-based
  connectivity, 1-based entity IDs, truth-table-aware variable access) follow
  the Exodus specification and the SEACAS reference implementation.

Key concepts
------------

Entity types
~~~~~~~~~~~~

Every piece of data in an Exodus file belongs to an *entity type*:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Name
     - Description
   * - ``global``
     - One value per time step (e.g. total energy).
   * - ``node``
     - One value per node per time step (e.g. temperature).
   * - ``element``
     - One value per element per time step, stored by element block.
   * - ``edge`` / ``face``
     - Stored by edge / face block when those entities are present.
   * - ``node_set``
     - Values defined on a subset of nodes.
   * - ``side_set``
     - Values defined on element-face pairs.

Entity-name aliases
~~~~~~~~~~~~~~~~~~~

All methods that accept an entity string are flexible.  ``'n'``,
``'node'``, ``'nodal'`` all mean the same thing.  ``'e'``, ``'el'``,
``'elem'``, ``'element'`` are equivalent.  See
:func:`~exodusii.core.entities.entity` for the full alias table.

Indexing conventions
~~~~~~~~~~~~~~~~~~~~

* **Connectivity** is stored 1-based (Exodus convention).  Pass
  ``zero_based=True`` to any connectivity method to get Python-friendly
  0-based arrays.
* **Time selectors** in the modern API accept ``None`` (full history),
  ``'first'``, ``'last'``, an ``int`` (zero-based Python index), or a
  ``float`` (nearest physical time value).
* **Entity IDs** (block IDs, set IDs) are arbitrary user integers — they
  are not guaranteed to be contiguous or zero-based.
