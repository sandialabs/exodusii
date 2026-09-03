.. _comparison:

Comparing databases
===================

exodusii provides two comparison interfaces:

* :func:`~exodusii.diff` — the full Python API, returning a structured
  :class:`~exodusii.DiffResult`.
* ``exodiff`` — the registered console script with SEACAS-compatible exit
  codes (``0`` same, ``1`` error, ``2`` different).

Both implement the same tolerance semantics as the SEACAS ``exodiff`` tool,
and support both **matched mesh ordering** (the default) and
**coordinate-based mesh matching** (``--match-coordinates`` /
``DiffOptions(coordinate_matching=True)``).

Quick comparison
----------------

.. code-block:: python

    import exodusii

    result = exodusii.diff("gold.exo", "test.exo")

    if result.same:
        print("files are identical within tolerance")
    else:
        for vd in result.variable_diffs:
            if vd.exceeded:
                print(f"  {vd.entity}/{vd.name}: "
                      f"max_delta={vd.max_delta:.3e}")
        for err in result.errors:
            print(f"  ERROR: {err}")

:class:`~exodusii.DiffResult` is falsey when ``same=False``::

    if not exodusii.diff("gold.exo", "test.exo"):
        raise AssertionError("regression detected")

Tolerance modes
---------------

:class:`~exodusii.Tolerance` pairs a
:class:`~exodusii.ToleranceMode` with a value and optional floor:

.. list-table::
   :header-rows: 1

   * - Mode
     - String alias
     - Definition
   * - ``RELATIVE``
     - ``"relative"``
     - ``|a - b| > tol * max(|a|, |b|)``
   * - ``ABSOLUTE``
     - ``"absolute"``
     - ``|a - b| > tol``
   * - ``COMBINED``
     - ``"combined"``
     - ``|a - b| >= tol * max(1, max(|a|, |b|))``
   * - ``IGNORE``
     - ``"ignore"``
     - Always considered equal.
   * - ``EIGEN_RELATIVE``
     - ``"eigenrel"``
     - Relative comparison of magnitudes (|a| vs |b|).
   * - ``EIGEN_ABSOLUTE``
     - ``"eigenabs"``
     - Absolute comparison of magnitudes.
   * - ``ULPS_FLOAT``
     - ``"ulps_float"``
     - ULPs distance at single precision.
   * - ``ULPS_DOUBLE``
     - ``"ulps_double"``
     - ULPs distance at double precision.

The default tolerance is ``RELATIVE`` with value ``1e-6``.

Controlling tolerances
----------------------

Use :class:`~exodusii.DiffOptions` to customise:

.. code-block:: python

    from exodusii import diff, DiffOptions, Tolerance, ToleranceMode

    # Absolute tolerance for everything
    opts = DiffOptions(
        default_tolerance=Tolerance(ToleranceMode.ABSOLUTE, 1e-8)
    )

    # Per-category tolerances
    opts = DiffOptions(
        nodal_tolerance=Tolerance(ToleranceMode.RELATIVE, 1e-6),
        element_tolerance=Tolerance(ToleranceMode.ABSOLUTE, 1e-5),
        coordinate_tolerance=Tolerance(ToleranceMode.ABSOLUTE, 1e-10),
    )

    # Per-variable override (highest priority)
    opts = DiffOptions(
        variable_tolerances={
            "PRESSURE": Tolerance(ToleranceMode.COMBINED, 1e-4),
        }
    )

    result = diff("gold.exo", "test.exo", opts)

Tolerance priority (high → low):

1. Per-variable override in ``variable_tolerances``.
2. Per-category tolerance (``nodal_tolerance``, ``element_tolerance``, …).
3. ``default_tolerance``.

Excluding variables
-------------------

.. code-block:: python

    opts = DiffOptions(
        exclude=frozenset({"TIME", "STEP_NUMBER"}),
        compare_coordinates=False,   # skip coordinate comparison
        compare_attributes=False,    # skip block element attributes
    )

Coordinate-based mesh matching (Phase 4)
-----------------------------------------

By default ``diff`` requires both files to have nodes and elements in the
same order.  When comparing files produced by different mesh generators or
reordering tools, enable coordinate-based matching:

.. code-block:: python

    from exodusii import diff, DiffOptions

    opts = DiffOptions(
        coordinate_matching=True,
        matching_tolerance=1e-8,   # per-axis spatial proximity tolerance
    )
    result = diff("gold.exo", "reordered.exo", opts)

The algorithm (mirroring SEACAS ``exodiff map.C``):

1. **Element centroid matching** — centroids are sorted along the
   max-spread coordinate axis; binary search identifies candidates within
   ``matching_tolerance``.
2. **Node map derivation** — matched element pairs yield node-to-node
   correspondences from local node coordinates.
3. **Free-node fallback** — nodes not reachable via element matching
   (e.g. on disconnected sub-meshes) are matched by direct coordinate
   proximity.

After the map is built, all variable arrays (nodal, element, set) are
reordered before comparison so that element-wise deltas are physically
meaningful.

.. note::

   **Sideset face ordinals** are checked after element remapping.  When an
   element's node order differs between the two files, the local face
   (side) number may change.  Such mismatches are emitted as **warnings**
   in :attr:`~exodusii.DiffResult.warnings` but do not constitute a fatal
   error.  Full face-ordinal correction from connectivity permutation is
   not currently performed.

.. list-table:: Coordinate-matching options
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``coordinate_matching``
     - Enable matching (default ``False``).
   * - ``matching_tolerance``
     - Per-axis spatial tolerance for building the map (default ``1e-6``).
       Independent of ``coordinate_tolerance``.
   * - ``require_unique_mapping``
     - Raise :class:`~exodusii.mesh.matching.MeshMatchError` when any
       node/element has no unique match (default ``True``).  Set to
       ``False`` to warn and continue with a partial map.

Time-step selection
-------------------

Use :class:`~exodusii.TimeSelection` for fine-grained step control.
All step numbers are **1-based** and refer to *file-2* steps.

.. code-block:: python

    from exodusii import TimeSelection

    # Compare steps 3 through 10, every other step
    ts = TimeSelection(start=3, stop=10, increment=2)

    # Compare only the final step on each file (LAST sentinel)
    ts = TimeSelection(start=-1)

    # Skip specific steps
    ts = TimeSelection(exclude_steps=frozenset({2, 5}))

    # File-2 step N corresponds to file-1 step N+1
    ts = TimeSelection(time_step_offset=1)

    # Interpolate file-2 to match file-1 time axis
    ts = TimeSelection(interpolating=True)

    # File-1 times are on a 2× coarser grid; scale before matching
    ts = TimeSelection(interpolating=True, time_value_scale=0.5)

    opts = DiffOptions(time_selection=ts)
    result = diff("gold.exo", "test.exo", opts)

Inspecting results
------------------

:class:`~exodusii.DiffResult` fields:

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - ``same``
     - ``bool``
     - ``True`` when no errors and no exceeded tolerances.
   * - ``errors``
     - ``list[str]``
     - Fatal structural mismatches (node count, missing variables, …).
   * - ``warnings``
     - ``list[str]``
     - Non-fatal notices (time-step count mismatch, sideset ordinal
       changes after mesh reordering, …).
   * - ``variable_diffs``
     - ``list[VariableDiff]``
     - One record per variable/location that exceeded tolerance.
   * - ``coordinate_max_delta``
     - ``float | None``
     - Worst coordinate difference.
   * - ``mesh_map_built``
     - ``bool``
     - ``True`` when coordinate-based matching was performed.
   * - ``unmatched_nodes``
     - ``int``
     - Count of unmatched nodes (0 unless ``require_unique_mapping=False``).
   * - ``unmatched_elems``
     - ``int``
     - Count of unmatched elements.

Each :class:`~exodusii.VariableDiff` record carries ``entity``, ``name``,
``max_delta``, ``exceeded``, and location information (``time_index``,
``entry_index``, ``block_id``, ``set_id``, ``value1``, ``value2``).

Show all variables (not just exceeded ones)::

    opts = DiffOptions(show_all=True)
    result = diff("gold.exo", "test.exo", opts)
    for vd in result.variable_diffs:
        print(f"{vd.name}: {vd.max_delta:.3e} {'EXCEEDED' if vd.exceeded else 'ok'}")

Command-line interface
----------------------

The ``exodiff`` console script mirrors the Python API::

    # Basic comparison (default: relative 1e-6)
    exodiff gold.exo test.exo

    # Absolute tolerance
    exodiff --absolute -t 1e-8 gold.exo test.exo

    # Exclude variables, skip coordinates
    exodiff -x TIME --no-coordinates gold.exo test.exo

    # Compare only the final step
    exodiff --start LAST gold.exo test.exo

    # Compare every other step
    exodiff --start 1 --increment 2 gold.exo test.exo

    # Interpolation mode
    exodiff --interpolate --time-scale 0.5 gold.exo test.exo

    # Coordinate-based mesh matching
    exodiff --match-coordinates gold.exo reordered.exo
    exodiff --match-coordinates --matching-tolerance 1e-8 gold.exo reordered.exo
    exodiff --match-coordinates --allow-partial-match gold.exo reordered.exo

    # JSON output (useful for scripting)
    exodiff --format json gold.exo test.exo | python -m json.tool

Exit codes: ``0`` same, ``1`` I/O error or structural mismatch, ``2`` different.

Simple equality checks
----------------------

For quick data-equality tests without per-variable reporting::

    # True if all arrays agree within tolerances
    ok = exodusii.allclose("gold.exo", "test.exo", rtol=1e-6, atol=0.0)

    # True if mesh and variable layout are identical
    ok = exodusii.similar("gold.exo", "test.exo")


.. code-block:: python

    import exodusii

    result = exodusii.diff("gold.exo", "test.exo")

    if result.same:
        print("files are identical within tolerance")
    else:
        for vd in result.variable_diffs:
            if vd.exceeded:
                print(f"  {vd.entity}/{vd.name}: "
                      f"max_delta={vd.max_delta:.3e}")
        for err in result.errors:
            print(f"  ERROR: {err}")

:class:`~exodusii.DiffResult` is falsey when ``same=False``::

    if not exodusii.diff("gold.exo", "test.exo"):
        raise AssertionError("regression detected")

Tolerance modes
---------------

:class:`~exodusii.Tolerance` pairs a
:class:`~exodusii.ToleranceMode` with a value and optional floor:

.. list-table::
   :header-rows: 1

   * - Mode
     - String alias
     - Definition
   * - ``RELATIVE``
     - ``"relative"``
     - ``|a - b| > tol * max(|a|, |b|)``
   * - ``ABSOLUTE``
     - ``"absolute"``
     - ``|a - b| > tol``
   * - ``COMBINED``
     - ``"combined"``
     - ``|a - b| >= tol * max(1, max(|a|, |b|))``
   * - ``IGNORE``
     - ``"ignore"``
     - Always considered equal.
   * - ``EIGEN_RELATIVE``
     - ``"eigenrel"``
     - Relative comparison of magnitudes (|a| vs |b|).
   * - ``EIGEN_ABSOLUTE``
     - ``"eigenabs"``
     - Absolute comparison of magnitudes.
   * - ``ULPS_FLOAT``
     - ``"ulps_float"``
     - ULPs distance at single precision.
   * - ``ULPS_DOUBLE``
     - ``"ulps_double"``
     - ULPs distance at double precision.

The default tolerance is ``RELATIVE`` with value ``1e-6``.

Controlling tolerances
----------------------

Use :class:`~exodusii.DiffOptions` to customise:

.. code-block:: python

    from exodusii import diff, DiffOptions, Tolerance, ToleranceMode

    # Absolute tolerance for everything
    opts = DiffOptions(
        default_tolerance=Tolerance(ToleranceMode.ABSOLUTE, 1e-8)
    )

    # Per-category tolerances
    opts = DiffOptions(
        nodal_tolerance=Tolerance(ToleranceMode.RELATIVE, 1e-6),
        element_tolerance=Tolerance(ToleranceMode.ABSOLUTE, 1e-5),
        coordinate_tolerance=Tolerance(ToleranceMode.ABSOLUTE, 1e-10),
    )

    # Per-variable override (highest priority)
    opts = DiffOptions(
        variable_tolerances={
            "PRESSURE": Tolerance(ToleranceMode.COMBINED, 1e-4),
        }
    )

    result = diff("gold.exo", "test.exo", opts)

Tolerance priority (high → low):

1. Per-variable override in ``variable_tolerances``.
2. Per-category tolerance (``nodal_tolerance``, ``element_tolerance``, …).
3. ``default_tolerance``.

Excluding variables
-------------------

.. code-block:: python

    opts = DiffOptions(
        exclude=frozenset({"TIME", "STEP_NUMBER"}),
        compare_coordinates=False,   # skip coordinate comparison
        compare_attributes=False,    # skip block element attributes
    )

Time-step selection
-------------------

Use :class:`~exodusii.TimeSelection` for fine-grained step control.
All step numbers are **1-based** and refer to *file-2* steps.

.. code-block:: python

    from exodusii import TimeSelection

    # Compare steps 3 through 10, every other step
    ts = TimeSelection(start=3, stop=10, increment=2)

    # Compare only the final step on each file (LAST sentinel)
    ts = TimeSelection(start=-1)

    # Skip specific steps
    ts = TimeSelection(exclude_steps=frozenset({2, 5}))

    # File-2 step N corresponds to file-1 step N+1
    ts = TimeSelection(time_step_offset=1)

    # Interpolate file-2 to match file-1 time axis
    ts = TimeSelection(interpolating=True)

    # File-1 times are on a 2× coarser grid; scale before matching
    ts = TimeSelection(interpolating=True, time_value_scale=0.5)

    opts = DiffOptions(time_selection=ts)
    result = diff("gold.exo", "test.exo", opts)

Inspecting results
------------------

:class:`~exodusii.DiffResult` fields:

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - ``same``
     - ``bool``
     - ``True`` when no errors and no exceeded tolerances.
   * - ``errors``
     - ``list[str]``
     - Fatal structural mismatches (node count, missing variables, …).
   * - ``warnings``
     - ``list[str]``
     - Non-fatal notices (time-step count mismatch, time value differences).
   * - ``variable_diffs``
     - ``list[VariableDiff]``
     - One record per variable/location that exceeded tolerance.
   * - ``coordinate_max_delta``
     - ``float | None``
     - Worst coordinate difference.

Each :class:`~exodusii.VariableDiff` record carries ``entity``, ``name``,
``max_delta``, ``exceeded``, and location information (``time_index``,
``entry_index``, ``block_id``, ``set_id``, ``value1``, ``value2``).

Show all variables (not just exceeded ones)::

    opts = DiffOptions(show_all=True)
    result = diff("gold.exo", "test.exo", opts)
    for vd in result.variable_diffs:
        print(f"{vd.name}: {vd.max_delta:.3e} {'EXCEEDED' if vd.exceeded else 'ok'}")

Command-line interface
----------------------

The ``exodiff`` console script mirrors the Python API::

    # Basic comparison (default: relative 1e-6)
    exodiff gold.exo test.exo

    # Absolute tolerance
    exodiff --absolute -t 1e-8 gold.exo test.exo

    # Exclude variables, skip coordinates
    exodiff -x TIME --no-coordinates gold.exo test.exo

    # Compare only the final step
    exodiff --start LAST gold.exo test.exo

    # Compare every other step
    exodiff --start 1 --increment 2 gold.exo test.exo

    # Interpolation mode
    exodiff --interpolate --time-scale 0.5 gold.exo test.exo

    # JSON output (useful for scripting)
    exodiff --format json gold.exo test.exo | python -m json.tool

Exit codes: ``0`` same, ``1`` I/O error or structural mismatch, ``2`` different.

Simple equality checks
----------------------

For quick data-equality tests without per-variable reporting::

    # True if all arrays agree within tolerances
    ok = exodusii.allclose("gold.exo", "test.exo", rtol=1e-6, atol=0.0)

    # True if mesh and variable layout are identical
    ok = exodusii.similar("gold.exo", "test.exo")
