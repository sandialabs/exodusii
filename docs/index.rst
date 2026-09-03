exodusii
========

**Version:** |version| — **License:** BSD 3-Clause

*A modern pure-Python interface for Sandia Exodus II finite-element databases.*

exodusii lets you read, write, query, compare, and inspect
`Exodus II <https://sandialabs.github.io/seacas-docs/sphinx/html/index.html>`_
databases — the NetCDF-based file format used by Sierra, Alegra, and many
open-source finite-element codes — entirely from Python with no compiled
extensions required.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: user_guide/installation
      :link-type: doc

      Install exodusii, open your first database, and extract results in
      under five minutes.

   .. grid-item-card:: User guide
      :link: user_guide/index
      :link-type: doc

      Step-by-step guides: reading, writing, parallel files, comparison,
      querying, and mesh geometry.

   .. grid-item-card:: API reference
      :link: api/index
      :link-type: doc

      Full autodoc API reference for every public class, function, and
      module.

   .. grid-item-card:: Changelog
      :link: changelog
      :link-type: doc

      What changed in each release.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting started

   user_guide/installation
   user_guide/overview
   user_guide/getting_started

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User guide

   user_guide/reading
   user_guide/writing
   user_guide/parallel
   user_guide/comparison
   user_guide/querying
   user_guide/mesh_geometry

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API reference

   api/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Project

   changelog
