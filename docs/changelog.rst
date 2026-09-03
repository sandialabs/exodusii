Changelog
=========

0.1.0a0 (in development)
--------------------------

* Initial public release of the modern API.
* :class:`~exodusii.ExodusFile` — read Exodus II databases.
* :class:`~exodusii.ExodusWriter` — write Exodus II databases.
* :class:`~exodusii.ParallelExodusFile` — aggregate decomposed Nemesis files.
* :func:`~exodusii.diff`, :class:`~exodusii.DiffOptions`,
  :class:`~exodusii.TimeSelection` — exodiff-style comparison with full
  time-step selection and linear interpolation.
* :func:`~exodusii.query`, :func:`~exodusii.print_query` — structured variable
  extraction.
* :mod:`exodusii.mesh` — mesh geometry helpers (centers, volumes, nodal volumes).
* Agent-oriented JSON CLI (``python -m exodusii``).
* ``exodiff`` and ``exoread`` console scripts.
* Compatibility layer for legacy ``ExodusIIFile`` API.
* Correctness fixes against SEACAS reference:

  - :meth:`~exodusii.ExodusFile.coordinates` now falls back to the combined
    ``coord`` variable for normal-model (``EX_NORMAL_MODEL``) files.
  - :meth:`~exodusii.ExodusFile.variable_truth_table` derives the table
    dynamically when not stored, matching ``ex_get_truth_table.c``.
  - :class:`~exodusii.ParallelExodusFile` emits ``UserWarning`` when
    ``node_num_map`` / ``elem_num_map`` are absent.
  - :meth:`~exodusii.ParallelExodusFile.side_set` deduplicates shared
    ``(element, side)`` pairs.
  - :meth:`~exodusii.mesh.elements.Tri3.volume` fixed for NumPy ≥ 2.5
    (``np.cross`` now requires 3D vectors; use explicit scalar formula).
