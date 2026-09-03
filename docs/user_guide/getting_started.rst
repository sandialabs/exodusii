Getting started
===============

This page walks through the most common exodusii tasks on a real database.
All examples use the :class:`~exodusii.ExodusFile` context manager to ensure
the file is closed automatically.

Opening a file
--------------

.. code-block:: python

    import exodusii

    with exodusii.ExodusFile.open("mesh.exo") as exo:
        print(exo.title)        # database title string
        print(exo.dimension)    # spatial dimension: 1, 2, or 3
        print(exo.node_count)   # total number of nodes
        print(exo.element_count)  # total number of elements

Inspecting metadata
-------------------

.. code-block:: python

    with exodusii.ExodusFile.open("mesh.exo") as exo:
        # Time axis
        times = exo.times()          # shape (n_steps,), dtype float64
        print(f"{len(times)} steps, t_final = {times[-1]:.4g}")

        # Coordinate names
        print(exo.coordinate_names())     # e.g. ['X', 'Y', 'Z']

        # Result variable inventory
        print(exo.variable_names("node"))     # nodal variables
        print(exo.variable_names("element"))  # element variables
        print(exo.variable_names("global"))   # global variables

        # Block and set IDs
        print(exo.element_block_ids())  # e.g. [1, 2, 5]
        print(exo.node_set_ids())
        print(exo.side_set_ids())

Reading coordinates
-------------------

.. code-block:: python

    with exodusii.ExodusFile.open("mesh.exo") as exo:
        coords = exo.coordinates()       # shape (n_nodes, dim)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        # Displaced coordinates at the last time step
        disp = exo.coordinates(time="last", displaced=True)

Reading result variables
------------------------

.. code-block:: python

    with exodusii.ExodusFile.open("mesh.exo") as exo:

        # Full time history for a global variable
        ke = exo.values("KINETIC_ENERGY", on="global")  # shape (n_steps,)

        # Nodal values at the last time step
        temp = exo.values("TEMP", on="node", time="last")  # shape (n_nodes,)

        # Nodal full history
        temp_hist = exo.values("TEMP", on="node")  # shape (n_steps, n_nodes)

        # Element values in one block at one step
        stress = exo.values("VON_MISES", on="element",
                             block_id=1, time="last")

        # Element values across all blocks, concatenated
        stress_all = exo.values("VON_MISES", on="element", time="last")

        # Node-set variable
        ns_val = exo.values("PRESSURE", on="node_set", set_id=100, time="last")

.. tip::

   The ``on=`` parameter and ``time=`` selector accept flexible strings.
   ``on='e'``, ``on='elem'``, and ``on='element'`` are all equivalent.
   ``time='last'`` and ``time=-1`` both select the final step.

Reading connectivity
--------------------

.. code-block:: python

    with exodusii.ExodusFile.open("mesh.exo") as exo:
        for block_id in exo.element_block_ids():
            block = exo.element_block(int(block_id))
            print(f"Block {block_id}: {block.element_type}, "
                  f"{block.count} elements")

            # 1-based connectivity (Exodus convention)
            conn = exo.element_connectivity(int(block_id))

            # 0-based connectivity (Python convention)
            conn0 = exo.element_connectivity(int(block_id), zero_based=True)

Reading sets
------------

.. code-block:: python

    with exodusii.ExodusFile.open("mesh.exo") as exo:
        ns = exo.node_set(100)
        print(ns.nodes)        # 1-based node IDs, shape (count,)
        print(ns.dist_facts)   # distribution factors or None

        ss = exo.side_set(200)
        print(ss.elems)        # 1-based element IDs
        print(ss.sides)        # side ordinals (1-based face index)

Writing a new database
----------------------

See :ref:`writing` for the full workflow.  Here is the minimal pattern:

.. code-block:: python

    import numpy as np
    import exodusii

    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    conn   = [[1, 2, 3, 4]]   # 1-based quad connectivity

    with exodusii.ExodusWriter.create("out.exo") as w:
        # Step 1: declare structure (must be first)
        w.initialize("my mesh", dimension=2, node_count=4,
                     element_count=1, element_blocks=1)
        # Step 2: coordinates
        w.write_coordinates(coords)
        # Step 3: define blocks/sets
        w.define_element_block(10, "quad", conn)
        # Step 4: define variables (before any time step)
        w.define_node_variables(["TEMP"])
        # Step 5: write time steps
        w.write_time(0.0)
        w.write_node_values("TEMP", [100.0, 200.0, 300.0, 400.0])

Comparing two databases
-----------------------

.. code-block:: python

    result = exodusii.diff("gold.exo", "test.exo")
    if result.same:
        print("files are identical within tolerance")
    else:
        for vd in result.variable_diffs:
            if vd.exceeded:
                print(f"  {vd.entity}/{vd.name}: "
                      f"max_delta={vd.max_delta:.3e} "
                      f"(block_id={vd.block_id})")
        for err in result.errors:
            print(f"  ERROR: {err}")

See :ref:`comparison` for tolerance options and time-step selection.

Using the command-line tools
----------------------------

exodusii ships two console scripts:

``exodiff``
    Compare two databases with exodiff-style tolerances::

        exodiff gold.exo test.exo
        exodiff --absolute -t 1e-8 gold.exo test.exo
        exodiff --start LAST gold.exo test.exo
        exodiff --format json gold.exo test.exo

``exoread``
    Extract variables interactively::

        exoread mesh.exo -n TEMP -t last
        exoread mesh.exo -e ENERGY -g TOTAL_KE

``python -m exodusii``
    Agent-oriented JSON CLI::

        python -m exodusii inspect mesh.exo
        python -m exodusii variables mesh.exo
        python -m exodusii stats mesh.exo --select n/TEMP --time last
        python -m exodusii query mesh.exo --select e/ENERGY --time step:5
