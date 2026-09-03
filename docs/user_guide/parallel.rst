.. _parallel:

Parallel (Nemesis) files
========================

When a simulation uses domain decomposition, each MPI rank writes its own
Exodus file.  The collection of these *component files* is called a Nemesis
decomposition.  :class:`~exodusii.ParallelExodusFile` aggregates them into
a single logical database by merging the global ID maps stored in each file.

How aggregation works
---------------------

Each component file contains:

* Local mesh data (coordinates, connectivity, sets, variables).
* ``node_num_map`` — maps local node indices (1-based) to global node IDs.
* ``elem_num_map`` — maps local element indices to global element IDs.

:class:`~exodusii.ParallelExodusFile` reads these maps and builds a unified
global ordering.  Shared nodes (nodes that appear in multiple component
files with the same global ID) are deduplicated automatically.

.. warning::

   If a component file does not contain ``node_num_map``, a sequential
   fallback is used and a ``UserWarning`` is emitted.  The fallback is
   correct **only** for non-overlapping partitions where no nodes are
   shared between ranks.  Real Nemesis files produced by standard
   decomposers (``nem_slice``) must include the map; files without it
   will produce incorrect global coordinates and connectivity.

Opening parallel files
----------------------

Pass all component file paths to
:meth:`~exodusii.ParallelExodusFile.open`::

    import exodusii

    paths = [f"mesh.e.4.{i}" for i in range(4)]

    with exodusii.ParallelExodusFile.open(*paths) as exo:
        print(f"Global nodes:    {exo.node_count}")
        print(f"Global elements: {exo.element_count}")

You can also use glob patterns to collect component files::

    import glob, exodusii

    paths = sorted(glob.glob("mesh.e.*.*"))
    with exodusii.ParallelExodusFile.open(*paths) as exo:
        ...

Reading aggregated data
-----------------------

The API mirrors :class:`~exodusii.ExodusFile` exactly::

    with exodusii.ParallelExodusFile.open(*paths) as exo:
        # Coordinates aggregated in global-ID order
        coords = exo.coordinates()           # (n_global_nodes, dim)

        # Values aggregated across all component files
        temp = exo.values("TEMP", on="node", time="last")

        # Element values for one global block
        stress = exo.values("STRESS", on="element",
                             block_id=1, time="last")

        # Node sets — shared nodes are deduplicated
        ns = exo.node_set(100)
        print(ns.nodes)                       # global node IDs

        # Side sets — duplicate (elem, side) pairs are deduplicated
        ss = exo.side_set(200)

Writing a joined serial file
-----------------------------

To produce a single serial Exodus file from a parallel decomposition::

    with exodusii.ParallelExodusFile.open(*paths) as exo:
        exo.write("joined.exo")

The joined file contains the fully aggregated mesh and all result variables
in global-ID order.

Legacy API
----------

Older code that expects the historical method-style interface can use::

    from exodusii.compat.legacy_parallel import ParallelExodusIIFile

    with ParallelExodusIIFile.open(*paths) as exo:
        coords = exo.get_coords()
        temp   = exo.get_node_variable_values("TEMP", time_step=1)

Edge and face blocks
--------------------

Edge and face entities follow the same pattern as element blocks.  If
``edge_num_map`` or ``face_num_map`` is absent from a component file, a
sequential fallback is used only when the local edge/face count is
non-zero::

    with exodusii.ParallelExodusFile.open(*paths) as exo:
        edge_count = exo.edge_count
        edge_ids   = exo.edge_block_ids()
