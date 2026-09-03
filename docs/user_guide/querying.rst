.. _querying:

Querying and summarising
========================

exodusii provides two higher-level interfaces for extracting and summarising
result data: :func:`~exodusii.query` for structured NumPy output and the
agent-oriented JSON CLI.

The query function
------------------

:func:`~exodusii.query` extracts one or more result variables and returns
a :class:`~exodusii.QueryResult` with a ``data`` array and ``names`` tuple.

Variable selector format
~~~~~~~~~~~~~~~~~~~~~~~~

Selectors use the pattern ``ENTITY/NAME``:

.. list-table::
   :header-rows: 1

   * - Prefix
     - Entity
   * - ``g``
     - global
   * - ``n``
     - node
   * - ``e``
     - element
   * - ``d``
     - edge
   * - ``f``
     - face
   * - ``ns``
     - node set
   * - ``ss``
     - side set

Special node selectors: ``n/coordinates`` and ``n/displacements``.

Examples
~~~~~~~~

.. code-block:: python

    import exodusii

    with exodusii.ExodusFile.open("mesh.exo") as exo:

        # Global variable — returns QueryResult with data shape (n_steps,)
        r = exodusii.query(exo, "g/TOTAL_ENERGY")
        print(r.names, r.data)

        # Nodal variable at last step — shape (n_nodes,)
        r = exodusii.query(exo, "n/TEMP", time="last")

        # Multiple selectors — data shape (n_nodes, 2)
        r = exodusii.query(exo, ["n/TEMP", "n/PRESSURE"], time="last")

        # Element variable — shape (n_elements,)
        r = exodusii.query(exo, "e/ENERGY", time="last")

        # Nodal coordinates
        r = exodusii.query(exo, "n/coordinates")

Printing
~~~~~~~~

:func:`~exodusii.print_query` prints a tabular summary to stdout::

    exodusii.print_query(exo, "n/TEMP", time="last", limit=20)

Command-line interface
----------------------

The ``python -m exodusii`` subcommand CLI prints JSON output and is designed
for use by agents and scripts.

Inspect a database::

    python -m exodusii inspect mesh.exo

List result variables::

    python -m exodusii variables mesh.exo
    python -m exodusii variables mesh.exo --truth-tables

List time steps::

    python -m exodusii times mesh.exo

List blocks and sets::

    python -m exodusii blocks mesh.exo
    python -m exodusii sets mesh.exo --entries --limit 20

Query values::

    python -m exodusii query mesh.exo --select n/TEMP --time last
    python -m exodusii query mesh.exo --select e/ENERGY --time step:5 --limit 10

Statistics (min, max, mean, std, L2)::

    python -m exodusii stats mesh.exo --select n/TEMP --time last
    python -m exodusii stats mesh.exo --select e/ENERGY --time last --by-block
    python -m exodusii stats mesh.exo --select ns/NSVAR --time last --by-set

Print database-specific Python examples::

    python -m exodusii examples mesh.exo

Time selectors in the CLI::

    python -m exodusii query mesh.exo --select n/TEMP --time first
    python -m exodusii query mesh.exo --select n/TEMP --time last
    python -m exodusii query mesh.exo --select n/TEMP --time index:0
    python -m exodusii query mesh.exo --select n/TEMP --time step:1
    python -m exodusii query mesh.exo --select n/TEMP --time 0.25

Output format::

    # Default: indented JSON (indent=2)
    python -m exodusii query mesh.exo --select n/TEMP --time last

    # Compact JSON
    python -m exodusii query mesh.exo --select n/TEMP --time last --terse

exoread
-------

The ``exoread`` console script is a traditional variable extractor::

    exoread mesh.exo -n TEMP -t last
    exoread mesh.exo -e ENERGY
    exoread mesh.exo -g TOTAL_ENERGY

Lineout
-------

:class:`~exodusii.Lineout` restricts a nodal result to nodes along a line
parallel to one of the coordinate axes::

    from exodusii import Lineout, lineout

    # Extract TEMP along the line X=0.5 (Y varies)
    with exodusii.ExodusFile.open("mesh.exo") as exo:
        lo = Lineout("X=0.5")
        result = lo.apply(exo, "TEMP", time="last")

    # Or using the convenience function
    with exodusii.ExodusFile.open("mesh.exo") as exo:
        result = lineout(exo, "TEMP", "X=0.5", time="last")

Self-learning CLI
-----------------

Use ``python -m exodusii learn`` to query the built-in capability
documentation::

    python -m exodusii learn -c overview
    python -m exodusii learn -c python_api
    python -m exodusii learn -c commands
    python -m exodusii learn --skill list
    python -m exodusii learn --skill exodusii-comparison .body
