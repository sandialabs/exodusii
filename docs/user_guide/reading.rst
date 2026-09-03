.. _reading:

Reading databases
=================

:class:`~exodusii.ExodusFile` is the primary interface for reading Exodus
databases.  It wraps a ``netCDF4`` backend with caching, lazy loading, and
a clean NumPy API.

Opening and closing
-------------------

Always use the context manager::

    with exodusii.ExodusFile.open("mesh.exo") as exo:
        ...   # exo is closed automatically on exit

For exploratory work you can also manage the lifecycle manually::

    exo = exodusii.ExodusFile.open("mesh.exo")
    try:
        coords = exo.coordinates()
    finally:
        exo.close()

Metadata properties
-------------------

All of these are cached on first access:

.. list-table::
   :header-rows: 1

   * - Property
     - Type
     - Description
   * - ``exo.title``
     - ``str``
     - Database title string.
   * - ``exo.dimension``
     - ``int``
     - Spatial dimension (1, 2, or 3).
   * - ``exo.node_count``
     - ``int``
     - Number of nodes.
   * - ``exo.element_count``
     - ``int``
     - Total number of elements across all blocks.
   * - ``exo.element_block_count``
     - ``int``
     - Number of element blocks.
   * - ``exo.node_set_count``
     - ``int``
     - Number of node sets.
   * - ``exo.side_set_count``
     - ``int``
     - Number of side sets.

Time steps
----------

.. code-block:: python

    times = exo.times()          # ndarray, shape (n_steps,), float64
    n     = len(times)
    t0, tf = times[0], times[-1]

Time selectors
~~~~~~~~~~~~~~

Methods that accept a ``time=`` keyword use the following selector types:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Meaning
   * - ``None`` (default)
     - Return all time steps (full history).
   * - ``'first'``
     - First time step (index 0).
   * - ``'last'``
     - Last time step.
   * - ``int`` (zero-based)
     - A specific step by Python index.  ``0`` = first, ``-1`` = last.
   * - ``float``
     - Nearest step by physical time value.

Coordinates
-----------

.. code-block:: python

    # Material (reference) coordinates, shape (n_nodes, dim)
    coords = exo.coordinates()

    # Displaced coordinates at a time step
    disp_coords = exo.coordinates(time="last", displaced=True)

    # Raw displacements
    delta = exo.displacements(time="last")   # shape (n_nodes, dim)

Coordinate names::

    names = exo.coordinate_names()  # e.g. ('X', 'Y', 'Z')

Element blocks
--------------

.. code-block:: python

    for bid in exo.element_block_ids():
        block = exo.element_block(int(bid))  # returns a Block dataclass
        print(block.id, block.element_type, block.count,
              block.nodes_per_entity)

Connectivity
~~~~~~~~~~~~

Connectivity arrays store node indices for each element.  By default they
are **1-based** (Exodus convention).  Pass ``zero_based=True`` for Python
0-based arrays::

    # shape (n_elems, nodes_per_elem), 1-based
    conn = exo.element_connectivity(block_id=1)

    # shape (n_elems, nodes_per_elem), 0-based
    conn0 = exo.element_connectivity(block_id=1, zero_based=True)

Result variables
----------------

The :meth:`~exodusii.ExodusFile.values` method is the primary interface for
reading result data.  All parameters after ``name`` are **keyword-only**.

.. code-block:: python

    # Global variable — full history, shape (n_steps,)
    ke = exo.values("KINETIC_ENERGY", on="global")

    # Nodal variable at one step — shape (n_nodes,)
    temp = exo.values("TEMP", on="node", time="last")

    # Nodal variable full history — shape (n_steps, n_nodes)
    temp_hist = exo.values("TEMP", on="node")

    # Element variable, one block, one step
    stress = exo.values("STRESS", on="element", block_id=1, time=0.5)

    # Element variable, all blocks concatenated, one step
    stress_all = exo.values("STRESS", on="element", time="last")

    # Set variable
    pres = exo.values("PRESSURE", on="node_set", set_id=100, time="last")

Variable names::

    print(exo.variable_names("node"))     # tuple of strings
    print(exo.variable_names("element"))

Node sets and side sets
-----------------------

.. code-block:: python

    # Node set — returns SetInfo dataclass
    ns = exo.node_set(100)
    node_ids = ns.nodes        # 1-based node IDs, shape (count,)
    df = ns.dist_facts         # distribution factors or None

    # Side set
    ss = exo.side_set(200)
    elem_ids = ss.elems        # 1-based element IDs
    side_nums = ss.sides       # side ordinals

Element attributes
------------------

Block element attributes (e.g. material density) are distinct from result
variables and are time-independent::

    names = exo.attribute_names("element_block", block_id=1)
    # shape (n_elements, n_attrs) — all attributes for a block
    vals  = exo.attributes("element_block", block_id=1)
    # shape (n_elements,) — one named attribute
    dens  = exo.attribute_values("element_block", block_id=1, "DENSITY")

Block properties
----------------

User-defined integer properties on blocks and sets::

    prop_names = exo.property_names("element_block")
    mat_ids    = exo.property_values("element_block", "MATERIAL_ID")
    mat_id_1   = exo.property_value("element_block", block_id=1, "MATERIAL_ID")

Truth tables
------------

For element variables and set variables, an optional *truth table* records
which (variable, block/set) combinations have data.  exodusii derives the
truth table dynamically when it is not stored in the file::

    # Full table: shape (n_blocks, n_vars)
    table = exo.variable_truth_table("element")

    # Single block row: shape (n_vars,)
    row = exo.variable_truth_table("element", id=block_id)

Parallel (decomposed) databases
--------------------------------

For decomposed Nemesis files see :ref:`parallel`.
