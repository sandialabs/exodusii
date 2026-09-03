.. _mesh_geometry:

Mesh geometry
=============

The :mod:`exodusii.mesh` sub-package provides NumPy-based helpers for
computing geometric quantities from mesh data.

Element types
-------------

Five concrete element classes are provided, each implementing the
:class:`~exodusii.mesh.elements.Element` protocol:

.. list-table::
   :header-rows: 1

   * - Class
     - Type strings
     - Dimension
   * - :class:`~exodusii.mesh.elements.Quad4`
     - ``quad``, ``quad4``, ``shell4``
     - 2D (area)
   * - :class:`~exodusii.mesh.elements.Hex8`
     - ``hex``, ``hex8``
     - 3D (volume)
   * - :class:`~exodusii.mesh.elements.Tri3`
     - ``tri3``, ``tri``, ``triangle``
     - 2D (area)
   * - :class:`~exodusii.mesh.elements.Tet4`
     - ``tet4``, ``tet``, ``tetra``
     - 3D (volume)
   * - :class:`~exodusii.mesh.elements.Wedge6`
     - ``wedge6``, ``wedge``
     - 3D (volume)

Instantiate an element from its nodal coordinates::

    import numpy as np
    from exodusii.mesh.elements import Quad4, element_factory

    # 2D unit square
    coord = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    q = Quad4(coord)
    print(q.dimension)    # 2
    print(q.center)       # [0.5, 0.5]
    print(q.volume)       # 1.0

Create an element from a type string::

    elem = element_factory("quad4", coord)

Subdivision
~~~~~~~~~~~

All element classes support uniform refinement via the ``subdiv`` family of
methods::

    centers = q.subdiv(2)    # shape (n_sub, dim) — sub-element centres
    coords  = q.subcoord(2)  # shape (n_sub_nodes, dim)
    conn    = q.subconn(2)   # shape (n_sub, nodes_per_elem), 0-based
    vols    = q.subvols(2)   # shape (n_sub,) — sub-element areas/volumes

Array-based geometry helpers
-----------------------------

The :mod:`exodusii.mesh.geometry` module provides functions that operate
directly on connectivity and coordinate arrays rather than individual element
objects.  All connectivity arrays are expected to be **zero-based**.

Element centres
~~~~~~~~~~~~~~~

.. code-block:: python

    from exodusii.mesh.geometry import entity_centers

    with exodusii.ExodusFile.open("mesh.exo") as exo:
        coords = exo.coordinates()
        for bid in exo.element_block_ids():
            conn0 = exo.element_connectivity(int(bid), zero_based=True)
            centers = entity_centers(conn0, coords)  # (n_elems, dim)

Element volumes
~~~~~~~~~~~~~~~

.. code-block:: python

    from exodusii.mesh.geometry import element_volumes

    with exodusii.ExodusFile.open("mesh.exo") as exo:
        coords = exo.coordinates()
        conn0  = exo.element_connectivity(1, zero_based=True)
        vols   = element_volumes("quad", conn0, coords)  # (n_elems,)

Nodal volumes
~~~~~~~~~~~~~

Distribute element volumes to nodes (useful for nodal mass matrices)::

    from exodusii.mesh.geometry import nodal_volumes

    node_vols = nodal_volumes(
        element_volumes=vols,
        connectivity=conn0,     # 0-based, shape (n_elems, nodes_per_elem)
        node_count=exo.node_count,
    )  # shape (n_nodes,)

Connectivity-weighted average
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Average nodal values over element connectivity (element-to-node projection)::

    from exodusii.mesh.geometry import connected_average

    elem_vals  = exo.values("ENERGY", on="element", block_id=1, time="last")
    node_avg   = connected_average(
        values=elem_vals,
        connectivity=conn0,
        node_count=exo.node_count,
    )

Bounding box
~~~~~~~~~~~~

.. code-block:: python

    from exodusii.mesh.geometry import bounding_box

    with exodusii.ExodusFile.open("mesh.exo") as exo:
        lo, hi = bounding_box(exo.coordinates())
        print(f"X: [{lo[0]:.4g}, {hi[0]:.4g}]")

Characteristic element length
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    from exodusii.mesh.geometry import characteristic_element_length

    h = characteristic_element_length("quad", conn0, coords)  # scalar
