.. _writing:

Writing databases
=================

:class:`~exodusii.ExodusWriter` creates new Exodus II databases.  It always
writes **large-model** format (separate ``coordx`` / ``coordy`` / ``coordz``
NetCDF variables), which is the modern default in SEACAS.

.. important::

   The counts passed to :meth:`~exodusii.ExodusWriter.initialize` — the
   number of element blocks, node sets, side sets, etc. — must **exactly
   match** the number of blocks and sets subsequently defined.  NetCDF4
   dimensions cannot be resized after the file is created.  Passing the
   wrong count raises :class:`~exodusii.core.errors.ExodusWriteError`.

Required workflow
-----------------

The seven-step pattern must be followed in order:

1. :meth:`~exodusii.ExodusWriter.initialize` — declare all counts.
2. :meth:`~exodusii.ExodusWriter.write_coordinates` — write nodal positions.
3. :meth:`~exodusii.ExodusWriter.define_element_block` (one call per block).
4. :meth:`~exodusii.ExodusWriter.define_node_set` /
   :meth:`~exodusii.ExodusWriter.define_side_set` (one call per set).
5. ``define_*_variables`` — declare variable names (before any time step).
6. :meth:`~exodusii.ExodusWriter.write_time` — advance to the next step.
7. ``write_*_values`` — write result data for that step.

Repeat steps 6–7 for each time step.

Minimal example
---------------

A single quad element, one node variable, two time steps:

.. code-block:: python

    import numpy as np
    import exodusii

    coords = np.array([[0., 0.],
                       [1., 0.],
                       [1., 1.],
                       [0., 1.]])

    with exodusii.ExodusWriter.create("out.exo") as w:
        # Step 1: declare structure
        w.initialize("unit quad", dimension=2, node_count=4,
                     element_count=1, element_blocks=1)

        # Step 2: coordinates — shape (node_count, dimension)
        w.write_coordinates(coords)

        # Step 3: define element block
        #   connectivity is 1-based by default; use zero_based=True for 0-based
        w.define_element_block(block_id=10, element_type="quad",
                               connectivity=[[1, 2, 3, 4]])

        # Step 5: declare variables
        w.define_node_variables(["TEMP"])
        w.define_global_variables(["TIME_STEP"])

        for step, (t, T) in enumerate([(0.0, 300.0), (1.0, 350.0)], start=1):
            # Step 6: write time value
            w.write_time(t)
            # Step 7: write result data
            w.write_global_values([float(step)])
            w.write_node_values("TEMP", np.full(4, T))

Element types
-------------

The ``element_type`` string passed to
:meth:`~exodusii.ExodusWriter.define_element_block` accepts a variety of
aliases:

.. list-table::
   :header-rows: 1

   * - Canonical
     - Aliases
   * - ``Quad4``
     - ``quad``, ``quad4``, ``shell4``
   * - ``Hex8``
     - ``hex``, ``hex8``
   * - ``Tri3``
     - ``tri3``, ``tri``, ``triangle``
   * - ``Tet4``
     - ``tet4``, ``tet``, ``tetra``
   * - ``Wedge6``
     - ``wedge6``, ``wedge``

Node sets and side sets
-----------------------

.. code-block:: python

    # Node set: list of 1-based node IDs
    w.define_node_set(set_id=100,
                      nodes=[1, 2, 3],
                      distribution_factors=[0.5, 1.0, 0.5],
                      name="left_boundary")

    # Side set: matching 1-based element IDs and side ordinals
    w.define_side_set(set_id=200,
                      elements=[1, 1],
                      sides=[1, 2],
                      distribution_factors=[1.0, 1.0],
                      name="top_face")

Element and set variables
-------------------------

.. code-block:: python

    # Element variables with optional truth table
    # truth_table shape: (n_blocks, n_vars) — 1 if present, 0 if absent
    w.define_element_variables(["STRESS", "STRAIN"])

    # Node-set variables
    w.define_node_set_variables(["PRESSURE"])

    # Writing set variable values at a time step
    w.write_node_set_values("PRESSURE", [100.0, 200.0, 100.0], set_id=100)

Element attributes
------------------

Block element attributes are time-independent::

    # values shape: (n_elements, n_attributes)
    w.write_block_attributes(
        on="element_block",
        block_id=10,
        values=[[7800.0, 200e9]],     # one element, two attributes
        names=["DENSITY", "YOUNGS_MODULUS"],
    )

ID maps (for parallel-ready files)
------------------------------------

Writing explicit 1-based global ID maps makes files compatible with
:class:`~exodusii.ParallelExodusFile`::

    w.write_node_id_map(np.arange(1, node_count + 1))
    w.write_element_id_map(np.arange(1, element_count + 1))

Metadata
--------

.. code-block:: python

    w.write_info_records(["Created by my simulation code"])
    w.write_qa_records([["SIM_CODE", "1.0", "2024-01-01", "12:00:00"]])
    w.define_property("element_block", "MATERIAL_ID", [42])

Copying databases
-----------------

To copy an existing database (possibly with modifications)::

    # File-to-file copy
    exodusii.copy_file("input.exo", "output.exo")

    # Programmatic copy (ExodusFile → ExodusWriter)
    with exodusii.ExodusFile.open("input.exo") as src, \
         exodusii.ExodusWriter.create("output.exo") as dst:
        exodusii.copy(src, dst)
