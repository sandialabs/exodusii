# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Exodus II finite-element database writer.

This module provides :class:`ExodusWriter`, a writer-oriented API for creating
Exodus II databases backed by a NetCDF4 file.  The required workflow order is:

1. Create a writer with :meth:`ExodusWriter.create` (or as a context manager).
2. Call :meth:`ExodusWriter.initialize` **once** to declare mesh dimensions and
   upfront counts for every block and set type.
3. Write mesh geometry (:meth:`write_coordinates`) and topology
   (:meth:`define_element_block`, :meth:`define_node_set`, …).
4. Optionally write ID maps and block/set attributes.
5. Define result variables (:meth:`define_global_variables`, etc.) — must be
   called *before* the first :meth:`write_time`.
6. Iterate over time steps: call :meth:`write_time` then write result values.
7. Close the database (:meth:`close`, or rely on the context manager).

**Upfront-count constraint** — the ``element_blocks``, ``node_sets``, and
``side_sets`` counts passed to :meth:`initialize` are written as NetCDF4
dimension sizes and **cannot be changed** after initialisation.  The exact
number of blocks/sets subsequently defined via ``define_*`` calls must equal
the counts declared up front.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from exodusii.core.entities import Entity
from exodusii.core.errors import ExodusLookupError
from exodusii.core.errors import ExodusWriteError
from exodusii.core.names import AttributeName
from exodusii.core.names import DimensionName
from exodusii.core.names import ExodusNames
from exodusii.core.names import VariableName
from exodusii.core.schema import block_spec
from exodusii.core.schema import set_spec
from exodusii.core.schema import variable_spec
from exodusii.core.schema import variable_value_name
from exodusii.core.strings import encode_fixed_width
from exodusii.io.netcdf4_backend import NetCDF4Backend


class ExodusWriter:
    """Write an Exodus II finite-element database.

    :class:`ExodusWriter` wraps a :class:`~exodusii.io.netcdf4_backend.NetCDF4Backend`
    and exposes a clean, write-only API.  Every public method corresponds to a
    logical step in the Exodus file-creation workflow.  Legacy ``put_*`` methods
    are implemented in a separate compatibility adapter.

    Parameters
    ----------
    backend : NetCDF4Backend
        An open, writable NetCDF4 backend.  Use :meth:`create` to construct
        both the backend and the writer in one step.

    Notes
    -----
    **Context manager** — the recommended usage pattern is::

        with ExodusWriter.create("mesh.exo") as w:
            w.initialize(...)
            ...

    The context manager calls :meth:`close` on exit, which flushes and closes
    the underlying NetCDF4 file.

    **Upfront-count constraint** — the ``element_blocks``, ``node_sets``, and
    related counts passed to :meth:`initialize` are committed as NetCDF4
    dimension sizes at initialisation time and **cannot be altered** afterwards.
    You must declare exactly as many blocks/sets as you subsequently define via
    ``define_*`` calls.  Declaring more than you define leaves uninitialised
    slots in the file; defining more than you declared raises
    :exc:`~exodusii.core.errors.ExodusWriteError`.

    Examples
    --------
    Minimal 4-node quadrilateral mesh with one time step:

    >>> import numpy as np
    >>> from exodusii.api.writer import ExodusWriter
    >>> coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    >>> conn = np.array([[1, 2, 3, 4]])  # one quad element, 1-based nodes
    >>> with ExodusWriter.create("/tmp/quad.exo") as w:
    ...     w.initialize("quad mesh", 2, node_count=4, element_count=1,
    ...                  element_blocks=1)
    ...     w.write_coordinates(coords)
    ...     w.define_element_block(1, "quad4", conn)
    ...     w.define_node_variables(["disp_x", "disp_y"])
    ...     w.write_time(0.0)
    ...     w.write_node_values("disp_x", np.zeros(4))
    ...     w.write_node_values("disp_y", np.zeros(4))
    """

    def __init__(self, backend: NetCDF4Backend) -> None:
        self._backend = backend
        self._initialized = False
        self._time_step = 0

        self._element_block_counter = 0
        self._node_set_counter = 0
        self._side_set_counter = 0

        self._element_block_indices: dict[int, int] = {}
        self._node_set_indices: dict[int, int] = {}
        self._side_set_indices: dict[int, int] = {}

        self._edge_block_counter = 0
        self._face_block_counter = 0
        self._edge_set_counter = 0
        self._face_set_counter = 0
        self._element_set_counter = 0

        self._edge_block_indices: dict[int, int] = {}
        self._face_block_indices: dict[int, int] = {}
        self._edge_set_indices: dict[int, int] = {}
        self._face_set_indices: dict[int, int] = {}
        self._element_set_indices: dict[int, int] = {}

    @classmethod
    def create(cls, path: str | Path) -> "ExodusWriter":
        """Create a new Exodus database file and return a writer for it.

        Parameters
        ----------
        path : str or Path
            Filesystem path for the new Exodus file.  Any existing file at
            this path will be overwritten.

        Returns
        -------
        ExodusWriter
            A writer ready for :meth:`initialize`.

        Examples
        --------
        >>> w = ExodusWriter.create("/tmp/mesh.exo")
        >>> w.initialize("test", 3, node_count=8, element_count=1,
        ...              element_blocks=1)
        >>> w.close()
        """

        return cls(NetCDF4Backend(path, mode="w"))

    @property
    def path(self) -> Path:
        """Database path."""

        return self._backend.path

    @property
    def backend(self) -> NetCDF4Backend:
        """Underlying backend."""

        return self._backend

    def close(self) -> None:
        """Close the database and flush all pending writes.

        Notes
        -----
        Prefer the context-manager form (``with ExodusWriter.create(...) as w``)
        over calling :meth:`close` explicitly, as the context manager guarantees
        cleanup even when an exception is raised.
        """

        self._backend.close()

    def sync(self) -> None:
        """Flush pending writes to disk without closing the file.

        Notes
        -----
        Useful for monitoring long-running simulations that write incrementally.
        For normal usage the context manager calls :meth:`close`, which also
        flushes.
        """

        self._backend.sync()

    def __enter__(self) -> "ExodusWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def initialize(
        self,
        title: str,
        dimension: int,
        node_count: int,
        element_count: int,
        *,
        element_blocks: int = 0,
        node_sets: int = 0,
        side_sets: int = 0,
        edge_count: int = 0,
        edge_blocks: int = 0,
        face_count: int = 0,
        face_blocks: int = 0,
        edge_sets: int = 0,
        face_sets: int = 0,
        element_sets: int = 0,
        floating_point_word_size: int = 8,
    ) -> None:
        """Initialize the Exodus database with mesh dimensions and entity counts.

        This method **must** be called exactly once before any other write
        operation.  It sets global file attributes, creates the standard NetCDF4
        dimensions, allocates ID/status/name arrays for every block and set
        type, and creates the time and coordinate variables.

        Parameters
        ----------
        title : str
            Descriptive title stored in the file (maximum ~80 characters for
            broad compatibility).
        dimension : int
            Spatial dimensionality of the mesh: ``1``, ``2``, or ``3``.
        node_count : int
            Total number of nodes in the mesh.
        element_count : int
            Total number of elements across all element blocks.
        element_blocks : int, optional
            Number of element blocks that will be defined.  **Must** equal the
            number of subsequent :meth:`define_element_block` calls.
            Default is ``0``.
        node_sets : int, optional
            Number of node sets that will be defined.  Default is ``0``.
        side_sets : int, optional
            Number of side sets that will be defined.  Default is ``0``.
        edge_count : int, optional
            Total number of edges (required only for edge-block meshes).
            Default is ``0``.
        edge_blocks : int, optional
            Number of edge blocks that will be defined.  Default is ``0``.
        face_count : int, optional
            Total number of faces (required only for face-block meshes).
            Default is ``0``.
        face_blocks : int, optional
            Number of face blocks that will be defined.  Default is ``0``.
        edge_sets : int, optional
            Number of edge sets that will be defined.  Default is ``0``.
        face_sets : int, optional
            Number of face sets that will be defined.  Default is ``0``.
        element_sets : int, optional
            Number of element sets that will be defined.  Default is ``0``.
        floating_point_word_size : int, optional
            Word size in bytes for floating-point data stored in the file.
            Must be ``4`` (single) or ``8`` (double).  Default is ``8``.

        Raises
        ------
        ExodusWriteError
            If the database has already been initialised.
        ValueError
            If ``dimension`` is not ``1``, ``2``, or ``3``.

        Notes
        -----
        **Upfront-count constraint** — the block and set counts (e.g.
        ``element_blocks``, ``node_sets``) are written as fixed NetCDF4
        dimension sizes and **cannot be changed** after this call.  Define
        exactly as many blocks/sets as declared; exceeding the declared count
        raises :exc:`~exodusii.core.errors.ExodusWriteError`.

        Examples
        --------
        Two-dimensional mesh with two element blocks and one node set:

        >>> with ExodusWriter.create("/tmp/mesh2.exo") as w:
        ...     w.initialize(
        ...         "two-block 2-D mesh",
        ...         dimension=2,
        ...         node_count=9,
        ...         element_count=4,
        ...         element_blocks=2,
        ...         node_sets=1,
        ...     )
        """

        if self._initialized:
            raise ExodusWriteError("database is already initialized")
        if dimension not in {1, 2, 3}:
            raise ValueError("dimension must be 1, 2, or 3")

        self._backend.set_attribute(AttributeName.TITLE.value, title)
        self._backend.set_attribute(AttributeName.API_VERSION.value, 5.03)
        self._backend.set_attribute(AttributeName.VERSION.value, 5.03)
        self._backend.set_attribute(
            AttributeName.FLOATING_POINT_WORD_SIZE.value, floating_point_word_size
        )
        self._backend.set_attribute(AttributeName.FILE_SIZE.value, 1)

        self._create_standard_dimensions()
        self._backend.create_dimension(DimensionName.NUM_DIMENSIONS.value, dimension)

        if node_count:
            self._backend.create_dimension(DimensionName.NUM_NODES.value, node_count)
        if element_count:
            self._backend.create_dimension(DimensionName.NUM_ELEMENTS.value, element_count)
        if edge_count:
            self._backend.create_dimension(DimensionName.NUM_EDGES.value, edge_count)
        if face_count:
            self._backend.create_dimension(DimensionName.NUM_FACES.value, face_count)

        self._create_time_variable()

        if node_count:
            self._create_coordinate_variables(dimension)

        self._allocate_blocks(
            count=element_blocks,
            dimension_name=DimensionName.NUM_ELEMENT_BLOCKS.value,
            ids_name=VariableName.ELEMENT_BLOCK_IDS.value,
            status_name=VariableName.ELEMENT_BLOCK_STATUS.value,
            names_name=VariableName.ELEMENT_BLOCK_NAMES.value,
        )
        self._allocate_blocks(
            count=edge_blocks,
            dimension_name=DimensionName.NUM_EDGE_BLOCKS.value,
            ids_name=VariableName.EDGE_BLOCK_IDS.value,
            status_name=VariableName.EDGE_BLOCK_STATUS.value,
            names_name=VariableName.EDGE_BLOCK_NAMES.value,
        )
        self._allocate_blocks(
            count=face_blocks,
            dimension_name=DimensionName.NUM_FACE_BLOCKS.value,
            ids_name=VariableName.FACE_BLOCK_IDS.value,
            status_name=VariableName.FACE_BLOCK_STATUS.value,
            names_name=VariableName.FACE_BLOCK_NAMES.value,
        )
        self._allocate_sets(
            count=node_sets,
            dimension_name=DimensionName.NUM_NODE_SETS.value,
            ids_name=VariableName.NODE_SET_IDS.value,
            status_name=VariableName.NODE_SET_STATUS.value,
            names_name=VariableName.NODE_SET_NAMES.value,
        )
        self._allocate_sets(
            count=side_sets,
            dimension_name=DimensionName.NUM_SIDE_SETS.value,
            ids_name=VariableName.SIDE_SET_IDS.value,
            status_name=VariableName.SIDE_SET_STATUS.value,
            names_name=VariableName.SIDE_SET_NAMES.value,
        )
        self._allocate_sets(
            count=edge_sets,
            dimension_name=DimensionName.NUM_EDGE_SETS.value,
            ids_name=VariableName.EDGE_SET_IDS.value,
            status_name=VariableName.EDGE_SET_STATUS.value,
            names_name=VariableName.EDGE_SET_NAMES.value,
        )
        self._allocate_sets(
            count=face_sets,
            dimension_name=DimensionName.NUM_FACE_SETS.value,
            ids_name=VariableName.FACE_SET_IDS.value,
            status_name=VariableName.FACE_SET_STATUS.value,
            names_name=VariableName.FACE_SET_NAMES.value,
        )
        self._allocate_sets(
            count=element_sets,
            dimension_name=DimensionName.NUM_ELEMENT_SETS.value,
            ids_name=VariableName.ELEMENT_SET_IDS.value,
            status_name=VariableName.ELEMENT_SET_STATUS.value,
            names_name=VariableName.ELEMENT_SET_NAMES.value,
        )

        self._initialized = True

    def write_info_records(self, records: Sequence[str]) -> None:
        """Write free-text information records to the database.

        Information records are arbitrary single-line strings stored verbatim
        in the file.  They are typically used to embed provenance data such as
        the generating code name, version, and run date.

        Parameters
        ----------
        records : sequence of str
            Lines of text to store.  Each line is padded or truncated to 80
            characters.  An empty sequence is a no-op.

        Examples
        --------
        >>> with ExodusWriter.create("/tmp/info.exo") as w:
        ...     w.initialize("demo", 2, node_count=4, element_count=1,
        ...                  element_blocks=1)
        ...     w.write_info_records(["Generated by MyCode v1.0", "Run: 2026-09-03"])
        """

        if not records:
            return

        if not self._backend.has_dimension(DimensionName.NUM_INFO_RECORDS.value):
            self._backend.create_dimension(DimensionName.NUM_INFO_RECORDS.value, len(records))

        if not self._backend.has_variable(VariableName.INFO_RECORDS.value):
            self._backend.create_variable(
                VariableName.INFO_RECORDS.value,
                str,
                (DimensionName.NUM_INFO_RECORDS.value, DimensionName.LINE_LENGTH.value),
            )

        self._backend.write_variable(
            VariableName.INFO_RECORDS.value, encode_fixed_width(records, width=80)
        )

    def write_qa_records(self, records: Sequence[Sequence[str]]) -> None:
        """Write quality-assurance (QA) records to the database.

        QA records identify the code that produced the file.  Each record
        contains exactly four fields: code name, code version, run date, and
        run time.  Records with fewer than four fields are right-padded with
        empty strings; extra fields are silently dropped.

        Parameters
        ----------
        records : sequence of sequence of str
            Outer sequence is one entry per QA record; inner sequence contains
            up to four string fields ``[name, version, date, time]``.  Each
            field is padded or truncated to 32 characters.  An empty sequence
            is a no-op.

        Examples
        --------
        >>> with ExodusWriter.create("/tmp/qa.exo") as w:
        ...     w.initialize("qa demo", 2, node_count=4, element_count=1,
        ...                  element_blocks=1)
        ...     w.write_qa_records([["MyCode", "1.0", "2026-09-03", "12:00:00"]])
        """

        if not records:
            return

        normalized = []
        for record in records:
            fields = [str(field) for field in record]
            fields = fields[:4] + [""] * max(0, 4 - len(fields))
            normalized.append(fields[:4])

        if not self._backend.has_dimension(DimensionName.NUM_QA_RECORDS.value):
            self._backend.create_dimension(DimensionName.NUM_QA_RECORDS.value, len(normalized))

        if not self._backend.has_variable(VariableName.QA_RECORDS.value):
            self._backend.create_variable(
                VariableName.QA_RECORDS.value,
                str,
                (
                    DimensionName.NUM_QA_RECORDS.value,
                    DimensionName.FOUR.value,
                    DimensionName.STRING_LENGTH.value,
                ),
            )

        encoded = np.asarray([encode_fixed_width(record, width=32) for record in normalized])
        self._backend.write_variable(VariableName.QA_RECORDS.value, encoded)

    def define_property(self, on: Entity, name: str, values: npt.ArrayLike) -> None:
        """Define and write a named integer property array for a block or set entity.

        Properties are per-block or per-set integer scalars used to annotate
        topology groupings (e.g. material IDs, ownership flags for parallel
        decompositions).

        Parameters
        ----------
        on : Entity
            The entity type to attach the property to.  Must be a block entity
            (e.g. ``Entity.ELEMENT_BLOCK``) or a set entity
            (e.g. ``Entity.NODE_SET``).
        name : str
            Property name stored as a variable attribute.  The Exodus standard
            reserves ``"ID"`` for the block/set ID property.
        values : array_like of int
            Integer property values, one entry per block or set.  Length must
            equal the number of allocated blocks or sets of the given type.

        Raises
        ------
        ValueError
            If ``on`` is neither a block nor a set entity, or if the length of
            ``values`` does not match the allocated count.

        Examples
        --------
        >>> with ExodusWriter.create("/tmp/prop.exo") as w:
        ...     w.initialize("prop demo", 2, node_count=4, element_count=2,
        ...                  element_blocks=2)
        ...     # ... define blocks ...
        ...     w.define_property(Entity.ELEMENT_BLOCK, "material_id", [101, 202])
        """

        values_array = np.asarray(values, dtype=np.int64)

        property_factory = None
        dimension = ""

        if on.is_block:
            block_schema = block_spec(on)
            property_factory = block_schema.property_variable
            dimension = block_schema.count_dimension
        elif on.is_set:
            set_schema = set_spec(on)
            property_factory = set_schema.property_variable
            dimension = set_schema.count_dimension
        else:
            raise ValueError(f"{on.value} does not have properties")

        if property_factory is None:
            raise ValueError(f"{on.value} does not have properties")

        count = self._dimension_size(dimension)
        if len(values_array) != count:
            raise ValueError(f"property value count must be {count}")

        property_index = self._next_property_index(property_factory)
        variable_name = property_factory(property_index)

        self._backend.create_variable(variable_name, int, (dimension,))
        self._backend.set_variable_attribute(variable_name, AttributeName.PROPERTY_NAME.value, name)
        self._backend.write_variable(variable_name, values_array)

    def _next_property_index(self, property_factory) -> int:
        index = 1
        while self._backend.has_variable(property_factory(index)):
            index += 1
        return index

    def define_variables(
        self, on: Entity, names: Sequence[str], *, truth_table: npt.ArrayLike | None = None
    ) -> None:
        """Define result variables for any supported entity type.

        This is the generic entry point used by the type-specific
        ``define_*_variables`` convenience methods.  It creates the variable
        count dimension, the names variable, and—for block/set entities—the
        per-location value variables gated by the truth table.

        Parameters
        ----------
        on : Entity
            Entity type for which to define variables (e.g. ``Entity.NODE``,
            ``Entity.ELEMENT``, ``Entity.NODE_SET``).
        names : sequence of str
            Variable names, each padded or truncated to 32 characters.
        truth_table : array_like of int, optional
            Boolean mask of shape ``(n_locations, n_variables)`` where
            ``n_locations`` is the number of blocks or sets of the
            corresponding location entity and ``n_variables`` is
            ``len(names)``.  A value of ``1`` means the variable is active for
            that block/set; ``0`` means it is inactive (no storage allocated).
            If ``None``, all entries default to ``1`` (all active).

        Raises
        ------
        ExodusWriteError
            If the database is not initialised, or if the location blocks/sets
            have not yet been defined when required.
        ValueError
            If ``on`` does not support variables or if ``truth_table`` has the
            wrong shape.

        Notes
        -----
        For ``Entity.GLOBAL`` and ``Entity.NODE`` variables the ``truth_table``
        parameter is ignored.  This method must be called **before** the first
        :meth:`write_time` call so that the necessary NetCDF4 variables are
        created before any data is written.

        Examples
        --------
        >>> # Element variables with a sparse truth table (block 1 only has "stress")
        >>> truth = [[1, 0], [0, 1]]  # shape (2 blocks, 2 vars)
        >>> w.define_variables(Entity.ELEMENT, ["stress", "strain"],
        ...                    truth_table=truth)
        """

        self._require_initialized()
        spec = variable_spec(on)
        if on not in {Entity.GLOBAL, Entity.NODE}:
            if spec.location_entity is None:
                raise ValueError(f"{on.value} variables do not have a location entity")

            location_ids = self._ids_for_variable_location(spec.location_entity)
            if len(location_ids) == 0:
                label = spec.location_entity.value.replace("_", " ")
                raise ExodusWriteError(f"define {label}s before defining {on.value} variables")

        self._backend.create_dimension(spec.count_dimension, len(names))
        self._backend.create_variable(
            spec.names_variable, str, (spec.count_dimension, DimensionName.STRING_LENGTH.value)
        )
        self._backend.write_variable(spec.names_variable, encode_fixed_width(names, width=32))

        if on is Entity.GLOBAL:
            self._backend.create_variable(
                variable_value_name(on, 1), float, (DimensionName.TIME.value, spec.count_dimension)
            )
            return

        if on is Entity.NODE:
            for variable_index in range(1, len(names) + 1):
                self._backend.create_variable(
                    variable_value_name(on, variable_index),
                    float,
                    (DimensionName.TIME.value, DimensionName.NUM_NODES.value),
                )
            return

        location = spec.location_entity
        if location is None:
            raise ValueError(f"{on.value} variables do not have a location entity")

        location_ids = self._ids_for_variable_location(location)
        location_dimension = self._location_count_dimension(location)

        if truth_table is not None:
            table = np.asarray(truth_table, dtype=np.int64)
            expected_shape = (len(location_ids), len(names))
            if table.shape != expected_shape:
                raise ValueError(f"truth_table shape must be {expected_shape}")
        else:
            table = np.ones((len(location_ids), len(names)), dtype=np.int64)

        if spec.truth_table_variable is not None:
            self._backend.create_variable(
                spec.truth_table_variable, int, (location_dimension, spec.count_dimension)
            )
            self._backend.write_variable(spec.truth_table_variable, table)

        for location_position, location_id in enumerate(location_ids):
            location_index = self._location_index(location, int(location_id))
            count_dim = self._value_count_dimension(location, location_index)

            for variable_index in range(1, len(names) + 1):
                if table[location_position, variable_index - 1]:
                    self._backend.create_variable(
                        variable_value_name(on, variable_index, location_index),
                        float,
                        (DimensionName.TIME.value, count_dim),
                    )

    def write_block_attributes(
        self,
        on: Entity,
        block_id: int,
        values: npt.ArrayLike,
        *,
        names: Sequence[str] | None = None,
    ) -> None:
        """Write per-element attributes for a block.

        Attributes are floating-point scalars attached to each element (or edge
        or face) in a block.  Common uses include material constants and
        cross-sectional properties.

        Parameters
        ----------
        on : Entity
            Block entity type: ``Entity.ELEMENT_BLOCK``, ``Entity.EDGE_BLOCK``,
            or ``Entity.FACE_BLOCK``.
        block_id : int
            ID of the target block (as supplied to the corresponding
            ``define_*_block`` call).
        values : array_like of float
            Attribute values.  May be one-dimensional (a single attribute per
            element, shape ``(n_elems,)``) or two-dimensional (multiple
            attributes, shape ``(n_elems, n_attrs)``).  ``n_elems`` must equal
            the element count of the specified block.
        names : sequence of str, optional
            Attribute names, one per column of ``values``.  Length must equal
            ``n_attrs``.  If ``None``, names are not written.

        Raises
        ------
        ValueError
            If ``on`` does not support attributes, if ``values`` has the wrong
            shape, or if ``len(names)`` does not match the attribute count.

        Examples
        --------
        >>> # Two-attribute block: thickness and Young's modulus
        >>> attrs = np.array([[0.01, 210e9], [0.01, 210e9]])  # shape (2 elems, 2 attrs)
        >>> w.write_block_attributes(Entity.ELEMENT_BLOCK, 1, attrs,
        ...                          names=["thickness", "youngs_modulus"])
        """

        spec = block_spec(on)
        if spec.attributes_dimension is None or spec.attributes_variable is None:
            raise ValueError(f"{on.value} does not support attributes")

        block_index = self._location_index(on, block_id)
        array = np.asarray(values, dtype=np.float64)

        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("attribute values must be one- or two-dimensional")

        block_count = self._dimension_size(spec.object_count_dimension(block_index))
        if array.shape[0] != block_count:
            raise ValueError(f"attribute row count must be {block_count}")

        attr_count = array.shape[1]
        attr_dim = spec.attributes_dimension(block_index)
        attr_var = spec.attributes_variable(block_index)

        if not self._backend.has_dimension(attr_dim):
            self._backend.create_dimension(attr_dim, attr_count)
        if not self._backend.has_variable(attr_var):
            self._backend.create_variable(
                attr_var, float, (spec.object_count_dimension(block_index), attr_dim)
            )

        self._backend.write_variable(attr_var, array)

        if names is not None:
            if spec.attribute_names_variable is None:
                raise ValueError(f"{on.value} does not support attribute names")
            if len(names) != attr_count:
                raise ValueError("attribute name count must match attribute column count")

            name_var = spec.attribute_names_variable(block_index)
            if not self._backend.has_variable(name_var):
                self._backend.create_variable(
                    name_var, str, (attr_dim, DimensionName.STRING_LENGTH.value)
                )
            self._backend.write_variable(name_var, encode_fixed_width(names, width=32))

    def set_block_status(self, entity: Entity, block_id: int, active: bool) -> None:
        """Set the active/inactive status flag for a block.

        Parameters
        ----------
        entity : Entity
            Block entity type (e.g. ``Entity.ELEMENT_BLOCK``).
        block_id : int
            ID of the target block.
        active : bool
            ``True`` to mark the block as active; ``False`` to mark it
            inactive.

        Examples
        --------
        >>> w.set_block_status(Entity.ELEMENT_BLOCK, 10, active=False)
        """

        spec = block_spec(entity)
        block_index = self._location_index(spec.entity, block_id)
        self._backend.write_variable(spec.status_variable, 1 if active else 0, block_index - 1)

    def set_set_status(self, entity: Entity, set_id: int, active: bool) -> None:
        """Set the active/inactive status flag for a set.

        Parameters
        ----------
        entity : Entity
            Set entity type (e.g. ``Entity.NODE_SET``, ``Entity.SIDE_SET``).
        set_id : int
            ID of the target set.
        active : bool
            ``True`` to mark the set as active; ``False`` to mark it inactive.

        Examples
        --------
        >>> w.set_set_status(Entity.NODE_SET, 5, active=False)
        """

        spec = set_spec(entity)
        set_index = self._location_index(spec.entity, set_id)
        self._backend.write_variable(spec.status_variable, 1 if active else 0, set_index - 1)

    def write_coordinates(
        self, coords: npt.ArrayLike, *, names: Sequence[str] | None = None
    ) -> None:
        """Write nodal coordinates.

        Parameters
        ----------
        coords : array_like of float, shape (node_count, dimension)
            Node coordinates in row-major order.  Each row is one node; each
            column corresponds to a spatial axis.  The number of rows must
            equal the ``node_count`` passed to :meth:`initialize`; the number
            of columns must equal ``dimension``.
        names : sequence of str, optional
            Axis labels, length ``dimension``.  Defaults to ``["X"]``,
            ``["X", "Y"]``, or ``["X", "Y", "Z"]`` for dimensions 1-3.

        Raises
        ------
        ExodusWriteError
            If the database has not been initialised.
        ValueError
            If ``coords`` is not two-dimensional, if the column count does not
            match the initialised dimension, if the row count does not match
            ``node_count``, or if ``len(names)`` does not equal ``dimension``.

        Examples
        --------
        >>> coords = np.array([[0.0, 0.0], [1.0, 0.0],
        ...                    [1.0, 1.0], [0.0, 1.0]])
        >>> w.write_coordinates(coords, names=["x", "y"])
        """

        self._require_initialized()

        array = np.asarray(coords, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("coords must be a two-dimensional array")

        dimension = array.shape[1]
        expected_dimension = self._backend.dimension(DimensionName.NUM_DIMENSIONS.value)
        if dimension != expected_dimension:
            raise ValueError(
                f"coords dimension {dimension} does not match "
                f"initialized dimension {expected_dimension}"
            )

        expected_nodes = self._backend.dimension(DimensionName.NUM_NODES.value, 0)
        if array.shape[0] != expected_nodes:
            raise ValueError(
                f"coords node count {array.shape[0]} does not match "
                f"initialized node count {expected_nodes}"
            )

        coordinate_names = list(names) if names is not None else list("XYZ"[:dimension])
        if len(coordinate_names) != dimension:
            raise ValueError("coordinate name count must match dimension")

        self._backend.write_variable(
            VariableName.COORDINATE_NAMES.value, encode_fixed_width(coordinate_names, width=32)
        )

        for axis in range(dimension):
            self._backend.write_variable(ExodusNames.coordinate(axis), array[:, axis])

    def write_id_map(self, on: Entity, values: npt.ArrayLike) -> None:
        """Write a global object ID map for nodes, elements, edges, or faces.

        An ID map translates the local (file-internal) 1-based indices to
        application-level global IDs.  This is critical for parallel
        decompositions where each partition file stores a subset of the mesh.

        Parameters
        ----------
        on : Entity
            Entity type for which to write the map: ``Entity.NODE``,
            ``Entity.ELEMENT``, ``Entity.EDGE``, or ``Entity.FACE``.
        values : array_like of int
            Global IDs, 1-based, one entry per object.  Length must equal the
            corresponding count declared in :meth:`initialize`.

        Raises
        ------
        ValueError
            If ``on`` does not have an ID map, or if the length of ``values``
            does not match the entity count.

        Notes
        -----
        When assembling a parallel result database, the ID maps in each
        partition file must be consistent: together they should enumerate every
        global ID exactly once.

        Examples
        --------
        >>> # Four nodes mapped to global IDs 101-104
        >>> w.write_id_map(Entity.NODE, [101, 102, 103, 104])
        """

        array = np.asarray(values, dtype=np.int64)

        if on is Entity.NODE:
            variable = VariableName.NODE_ID_MAP.value
            dimension = DimensionName.NUM_NODES.value
        elif on is Entity.ELEMENT:
            variable = VariableName.ELEMENT_ID_MAP.value
            dimension = DimensionName.NUM_ELEMENTS.value
        elif on is Entity.EDGE:
            variable = VariableName.EDGE_ID_MAP.value
            dimension = DimensionName.NUM_EDGES.value
        elif on is Entity.FACE:
            variable = VariableName.FACE_ID_MAP.value
            dimension = DimensionName.NUM_FACES.value
        else:
            raise ValueError(f"{on.value} does not have an object ID map")

        expected = self._dimension_size(dimension)
        if len(array) != expected:
            raise ValueError(f"{on.value} ID map length must be {expected}")

        if not self._backend.has_variable(variable):
            self._backend.create_variable(variable, int, (dimension,))

        self._backend.write_variable(variable, array)

    def write_node_id_map(self, values: npt.ArrayLike) -> None:
        """Write the node global ID map.

        Parameters
        ----------
        values : array_like of int
            Global node IDs, 1-based, length ``node_count``.

        Notes
        -----
        Delegates to :meth:`write_id_map` with ``on=Entity.NODE``.  In
        parallel workflows the concatenation of ID maps across all partition
        files must cover the full global node set without duplicates.

        Examples
        --------
        >>> w.write_node_id_map([1, 2, 3, 4])
        """

        self.write_id_map(Entity.NODE, values)

    def write_element_id_map(self, values: npt.ArrayLike) -> None:
        """Write the element global ID map.

        Parameters
        ----------
        values : array_like of int
            Global element IDs, 1-based, length ``element_count``.

        Notes
        -----
        Delegates to :meth:`write_id_map` with ``on=Entity.ELEMENT``.  In
        parallel workflows the concatenation of ID maps across all partition
        files must cover the full global element set without duplicates.

        Examples
        --------
        >>> w.write_element_id_map([1])
        """

        self.write_id_map(Entity.ELEMENT, values)

    def write_edge_id_map(self, values: npt.ArrayLike) -> None:
        """Write the edge global ID map."""

        self.write_id_map(Entity.EDGE, values)

    def write_face_id_map(self, values: npt.ArrayLike) -> None:
        """Write the face global ID map."""

        self.write_id_map(Entity.FACE, values)

    def define_block(
        self,
        entity: Entity,
        block_id: int,
        element_type: str,
        connectivity: npt.ArrayLike,
        *,
        name: str = "",
        zero_based: bool = False,
        edge_connectivity: npt.ArrayLike | None = None,
        face_connectivity: npt.ArrayLike | None = None,
        active: bool = True,
    ) -> None:
        """Define an element, edge, or face block.

        This is the generic entry point used by the type-specific
        ``define_element_block``, ``define_edge_block``, and
        ``define_face_block`` convenience methods.

        Parameters
        ----------
        entity : Entity
            Block entity type: ``Entity.ELEMENT_BLOCK``, ``Entity.EDGE_BLOCK``,
            or ``Entity.FACE_BLOCK``.
        block_id : int
            User-assigned integer ID for this block.  Must be unique among
            blocks of the same type.
        element_type : str
            Exodus element type string, stored as an upper-cased variable
            attribute.  Common aliases include ``'quad'`` / ``'quad4'`` (4-node
            quadrilateral), ``'tri'`` / ``'tri3'`` (3-node triangle),
            ``'hex'`` / ``'hex8'`` (8-node hexahedron), ``'tet'`` / ``'tet4'``
            (4-node tetrahedron), ``'bar'`` / ``'bar2'`` (2-node bar/beam).
        connectivity : array_like of int, shape (n_elems, nodes_per_elem)
            Node connectivity for each element.  By default, indices are
            1-based (matching Exodus convention).  Pass ``zero_based=True`` to
            use 0-based indices, which will be converted internally.
        name : str, optional
            Human-readable block name.  Default is ``""`` (empty).
        zero_based : bool, optional
            If ``True``, interpret connectivity indices as 0-based and
            increment them by 1 before writing.  Default is ``False``.
        edge_connectivity : array_like of int, optional
            Element-to-edge connectivity, shape
            ``(n_elems, edges_per_elem)``.  Only valid for element blocks.
        face_connectivity : array_like of int, optional
            Element-to-face connectivity, shape
            ``(n_elems, faces_per_elem)``.  Only valid for element blocks.
        active : bool, optional
            Initial block status flag.  Default is ``True``.

        Raises
        ------
        ExodusWriteError
            If the database is not initialised, or if the number of defined
            blocks would exceed the count declared in :meth:`initialize`.
        ValueError
            If ``connectivity`` is not two-dimensional, or if
            ``edge_connectivity`` / ``face_connectivity`` are provided for an
            entity type that does not support them.

        Examples
        --------
        >>> conn = np.array([[1, 2, 3, 4], [3, 4, 5, 6]])  # two quad4 elements
        >>> w.define_block(Entity.ELEMENT_BLOCK, 1, "quad4", conn, name="surface")
        """

        self._require_initialized()
        spec = block_spec(entity)

        conn = np.asarray(connectivity, dtype=np.int64)
        if conn.ndim != 2:
            raise ValueError("connectivity must be a two-dimensional array")
        if zero_based:
            conn = conn + 1

        block_index = self._next_block_index(spec.entity, block_id)

        count_dim = spec.object_count_dimension(block_index)
        nodes_dim = spec.nodes_per_object_dimension(block_index)
        conn_name = spec.connectivity_variable(block_index)

        self._backend.create_dimension(count_dim, conn.shape[0])
        self._backend.create_dimension(nodes_dim, conn.shape[1])
        self._backend.create_variable(conn_name, int, (count_dim, nodes_dim))
        self._backend.set_variable_attribute(
            conn_name, AttributeName.ELEMENT_TYPE.value, element_type.upper()
        )
        self._backend.write_variable(conn_name, conn)

        if edge_connectivity is not None:
            if spec.edges_per_object_dimension is None or spec.edge_connectivity_variable is None:
                raise ValueError(f"{spec.entity.value} does not support edge connectivity")
            edge_conn = np.asarray(edge_connectivity, dtype=np.int64)
            if zero_based:
                edge_conn = edge_conn + 1
            edge_dim = spec.edges_per_object_dimension(block_index)
            self._backend.create_dimension(edge_dim, edge_conn.shape[1])
            self._backend.create_variable(
                spec.edge_connectivity_variable(block_index), int, (count_dim, edge_dim)
            )
            self._backend.write_variable(spec.edge_connectivity_variable(block_index), edge_conn)

        if face_connectivity is not None:
            if spec.faces_per_object_dimension is None or spec.face_connectivity_variable is None:
                raise ValueError(f"{spec.entity.value} does not support face connectivity")
            face_conn = np.asarray(face_connectivity, dtype=np.int64)
            if zero_based:
                face_conn = face_conn + 1
            face_dim = spec.faces_per_object_dimension(block_index)
            self._backend.create_dimension(face_dim, face_conn.shape[1])
            self._backend.create_variable(
                spec.face_connectivity_variable(block_index), int, (count_dim, face_dim)
            )
            self._backend.write_variable(spec.face_connectivity_variable(block_index), face_conn)

        self._backend.write_variable(spec.ids_variable, int(block_id), block_index - 1)
        self._backend.write_variable(spec.status_variable, 1 if active else 0, block_index - 1)
        self._backend.write_variable(
            spec.names_variable, encode_fixed_width(name, width=32), block_index - 1
        )

    def define_element_block(
        self,
        block_id: int,
        element_type: str,
        connectivity: npt.ArrayLike,
        *,
        name: str = "",
        zero_based: bool = False,
        edge_connectivity: npt.ArrayLike | None = None,
        face_connectivity: npt.ArrayLike | None = None,
        active: bool = True,
    ) -> None:
        """Define an element block.

        Parameters
        ----------
        block_id : int
            User-assigned integer ID for this block.  Must be unique among
            element blocks.
        element_type : str
            Exodus element type string (case-insensitive).  Common values:
            ``'quad4'`` or ``'quad'``, ``'tri3'`` or ``'tri'``,
            ``'hex8'`` or ``'hex'``, ``'tet4'`` or ``'tet'``,
            ``'bar2'`` or ``'bar'``, ``'shell4'``, ``'wedge6'``.
        connectivity : array_like of int, shape (n_elems, nodes_per_elem)
            Node indices for each element.  Indices are 1-based by default;
            use ``zero_based=True`` for 0-based input.
        name : str, optional
            Human-readable block name.  Default is ``""`` (empty).
        zero_based : bool, optional
            If ``True``, increment connectivity indices by 1 before writing.
            Default is ``False``.
        edge_connectivity : array_like of int, optional
            Element-to-edge connectivity, shape ``(n_elems, edges_per_elem)``.
        face_connectivity : array_like of int, optional
            Element-to-face connectivity, shape ``(n_elems, faces_per_elem)``.
        active : bool, optional
            Initial block status.  Default is ``True``.

        Raises
        ------
        ExodusWriteError
            If the database is not initialised, or if the declared
            ``element_blocks`` count in :meth:`initialize` is exceeded.
        ValueError
            If ``connectivity`` is not two-dimensional.

        Examples
        --------
        >>> conn = np.array([[1, 2, 3, 4]])  # single quad4 element
        >>> w.define_element_block(1, "quad4", conn, name="domain")
        """

        self.define_block(
            Entity.ELEMENT_BLOCK,
            block_id,
            element_type,
            connectivity,
            name=name,
            zero_based=zero_based,
            edge_connectivity=edge_connectivity,
            face_connectivity=face_connectivity,
            active=active,
        )

    def define_edge_block(
        self,
        block_id: int,
        element_type: str,
        connectivity: npt.ArrayLike,
        *,
        name: str = "",
        zero_based: bool = False,
        active: bool = True,
    ) -> None:
        """Define an edge block.

        Parameters
        ----------
        block_id : int
            User-assigned integer ID for this edge block.
        element_type : str
            Exodus element type string for the edge topology (e.g. ``'bar2'``).
        connectivity : array_like of int, shape (n_edges, nodes_per_edge)
            Node indices for each edge, 1-based by default.
        name : str, optional
            Human-readable block name.  Default is ``""`` (empty).
        zero_based : bool, optional
            If ``True``, increment connectivity indices by 1.  Default is
            ``False``.
        active : bool, optional
            Initial block status.  Default is ``True``.

        Raises
        ------
        ExodusWriteError
            If the database is not initialised or the declared ``edge_blocks``
            count is exceeded.
        ValueError
            If ``connectivity`` is not two-dimensional.

        Examples
        --------
        >>> conn = np.array([[1, 2], [2, 3]])  # two bar2 edges
        >>> w.define_edge_block(10, "bar2", conn)
        """

        self.define_block(
            Entity.EDGE_BLOCK,
            block_id,
            element_type,
            connectivity,
            name=name,
            zero_based=zero_based,
            active=active,
        )

    def define_face_block(
        self,
        block_id: int,
        element_type: str,
        connectivity: npt.ArrayLike,
        *,
        name: str = "",
        zero_based: bool = False,
        active: bool = True,
    ) -> None:
        """Define a face block.

        Parameters
        ----------
        block_id : int
            User-assigned integer ID for this face block.
        element_type : str
            Exodus element type string for the face topology (e.g. ``'quad4'``,
            ``'tri3'``).
        connectivity : array_like of int, shape (n_faces, nodes_per_face)
            Node indices for each face, 1-based by default.
        name : str, optional
            Human-readable block name.  Default is ``""`` (empty).
        zero_based : bool, optional
            If ``True``, increment connectivity indices by 1.  Default is
            ``False``.
        active : bool, optional
            Initial block status.  Default is ``True``.

        Raises
        ------
        ExodusWriteError
            If the database is not initialised or the declared ``face_blocks``
            count is exceeded.
        ValueError
            If ``connectivity`` is not two-dimensional.

        Examples
        --------
        >>> conn = np.array([[1, 2, 3, 4]])  # one quad4 face
        >>> w.define_face_block(20, "quad4", conn)
        """

        self.define_block(
            Entity.FACE_BLOCK,
            block_id,
            element_type,
            connectivity,
            name=name,
            zero_based=zero_based,
            active=active,
        )

    def define_set(
        self,
        entity: Entity,
        set_id: int,
        entries: npt.ArrayLike,
        *,
        extra_entries: npt.ArrayLike | None = None,
        distribution_factors: npt.ArrayLike | None = None,
        name: str = "",
        active: bool = True,
    ) -> None:
        """Define a generic set (node set, side set, edge set, etc.).

        This is the generic entry point used by the type-specific
        ``define_*_set`` convenience methods.

        Parameters
        ----------
        entity : Entity
            Set entity type: ``Entity.NODE_SET``, ``Entity.SIDE_SET``,
            ``Entity.EDGE_SET``, ``Entity.FACE_SET``, or
            ``Entity.ELEMENT_SET``.
        set_id : int
            User-assigned integer ID for this set.  Must be unique among sets
            of the same type.
        entries : array_like of int, shape (n_entries,)
            Primary entries for the set.  For node sets these are 1-based node
            indices; for side sets these are 1-based element indices (sides are
            supplied via ``extra_entries``).
        extra_entries : array_like of int, optional
            Secondary entries, same length as ``entries``.  For side sets these
            are the local side numbers (1-based) within each element; for edge
            and face sets these are optional orientation values.
        distribution_factors : array_like of float, optional
            Per-entry distribution factors used to distribute boundary
            conditions across the set.
        name : str, optional
            Human-readable set name.  Default is ``""`` (empty).
        active : bool, optional
            Initial set status.  Default is ``True``.

        Raises
        ------
        ExodusWriteError
            If the database is not initialised, or if the declared set count
            in :meth:`initialize` is exceeded.
        ValueError
            If ``entries`` is not one-dimensional, or if ``extra_entries`` is
            provided for an entity type that does not support it.

        Examples
        --------
        >>> w.define_set(Entity.NODE_SET, 1, [1, 2, 3, 4])
        """

        self._require_initialized()
        spec = set_spec(entity)

        entry_array = np.asarray(entries, dtype=np.int64)
        if entry_array.ndim != 1:
            raise ValueError("entries must be a one-dimensional array")

        set_index = self._next_set_index(spec.entity, set_id)
        count_dim = spec.entry_count_dimension(set_index)

        self._backend.create_dimension(count_dim, len(entry_array))
        self._backend.create_variable(spec.entries_variable(set_index), int, (count_dim,))
        self._backend.write_variable(spec.entries_variable(set_index), entry_array)

        if extra_entries is not None:
            if spec.extra_entries_variable is None:
                raise ValueError(f"{spec.entity.value} does not support extra entries")
            extra_array = np.asarray(extra_entries, dtype=np.int64)
            if extra_array.shape != entry_array.shape:
                raise ValueError("extra_entries must have the same length as entries")
            self._backend.create_variable(spec.extra_entries_variable(set_index), int, (count_dim,))
            self._backend.write_variable(spec.extra_entries_variable(set_index), extra_array)

        if distribution_factors is not None:
            factors = np.asarray(distribution_factors, dtype=np.float64)
            if len(factors):
                dist_dim = spec.dist_factor_count_dimension(set_index)
                self._backend.create_dimension(dist_dim, len(factors))
                self._backend.create_variable(
                    spec.dist_factors_variable(set_index), float, (dist_dim,)
                )
                self._backend.write_variable(spec.dist_factors_variable(set_index), factors)

        self._backend.write_variable(spec.ids_variable, int(set_id), set_index - 1)
        self._backend.write_variable(spec.status_variable, 1 if active else 0, set_index - 1)
        self._backend.write_variable(
            spec.names_variable, encode_fixed_width(name, width=32), set_index - 1
        )

    def define_node_set(
        self,
        set_id: int,
        nodes: npt.ArrayLike,
        *,
        distribution_factors: npt.ArrayLike | None = None,
        name: str = "",
        active: bool = True,
    ) -> None:
        """Define a node set.

        Parameters
        ----------
        set_id : int
            User-assigned integer ID for this node set.  Must be unique among
            node sets.
        nodes : array_like of int, shape (n_nodes,)
            1-based node indices belonging to this set.
        distribution_factors : array_like of float, optional
            Per-node distribution factors, same length as ``nodes``.  Used to
            apportion boundary-condition values across nodes.
        name : str, optional
            Human-readable set name.  Default is ``""`` (empty).
        active : bool, optional
            Initial set status.  Default is ``True``.

        Raises
        ------
        ExodusWriteError
            If the database is not initialised or the declared ``node_sets``
            count in :meth:`initialize` is exceeded.

        Examples
        --------
        >>> # Fix the left edge (nodes 1 and 4)
        >>> w.define_node_set(1, [1, 4], name="left_edge",
        ...                   distribution_factors=[1.0, 1.0])
        """

        self.define_set(
            Entity.NODE_SET,
            set_id,
            nodes,
            distribution_factors=distribution_factors,
            name=name,
            active=active,
        )

    def define_side_set(
        self,
        set_id: int,
        elements: npt.ArrayLike,
        sides: npt.ArrayLike,
        *,
        distribution_factors: npt.ArrayLike | None = None,
        name: str = "",
        active: bool = True,
    ) -> None:
        """Define a side set.

        A side set identifies boundary faces of a mesh by specifying, for each
        entry, the element that owns the face and the local face number within
        that element.

        Parameters
        ----------
        set_id : int
            User-assigned integer ID for this side set.  Must be unique among
            side sets.
        elements : array_like of int, shape (n_sides,)
            1-based element indices, one per side.
        sides : array_like of int, shape (n_sides,)
            1-based local side numbers within each element, same length as
            ``elements``.
        distribution_factors : array_like of float, optional
            Per-side-node distribution factors.  Length must equal the total
            number of nodes on the side set faces.
        name : str, optional
            Human-readable set name.  Default is ``""`` (empty).
        active : bool, optional
            Initial set status.  Default is ``True``.

        Raises
        ------
        ExodusWriteError
            If the database is not initialised or the declared ``side_sets``
            count in :meth:`initialize` is exceeded.

        Examples
        --------
        >>> # Right boundary: side 2 of element 1
        >>> w.define_side_set(10, elements=[1], sides=[2], name="right_wall")
        """

        self.define_set(
            Entity.SIDE_SET,
            set_id,
            elements,
            extra_entries=sides,
            distribution_factors=distribution_factors,
            name=name,
            active=active,
        )

    def define_edge_set(
        self,
        set_id: int,
        edges: npt.ArrayLike,
        *,
        orientations: npt.ArrayLike | None = None,
        distribution_factors: npt.ArrayLike | None = None,
        name: str = "",
        active: bool = True,
    ) -> None:
        """Define an edge set."""

        self.define_set(
            Entity.EDGE_SET,
            set_id,
            edges,
            extra_entries=orientations,
            distribution_factors=distribution_factors,
            name=name,
            active=active,
        )

    def define_face_set(
        self,
        set_id: int,
        faces: npt.ArrayLike,
        *,
        orientations: npt.ArrayLike | None = None,
        distribution_factors: npt.ArrayLike | None = None,
        name: str = "",
        active: bool = True,
    ) -> None:
        """Define a face set."""

        self.define_set(
            Entity.FACE_SET,
            set_id,
            faces,
            extra_entries=orientations,
            distribution_factors=distribution_factors,
            name=name,
            active=active,
        )

    def define_element_set(
        self,
        set_id: int,
        elements: npt.ArrayLike,
        *,
        distribution_factors: npt.ArrayLike | None = None,
        name: str = "",
        active: bool = True,
    ) -> None:
        """Define an element set."""

        self.define_set(
            Entity.ELEMENT_SET,
            set_id,
            elements,
            distribution_factors=distribution_factors,
            name=name,
            active=active,
        )

    def define_global_variables(self, names: Sequence[str]) -> None:
        """Define global result variable names.

        Global variables hold one scalar value per time step for the entire
        mesh (e.g. total energy, reaction force sum).

        Parameters
        ----------
        names : sequence of str
            Variable names, each padded or truncated to 32 characters.  The
            order determines the index used when writing values with
            :meth:`write_global_values`.

        Notes
        -----
        Must be called **before** the first :meth:`write_time` call.  Calling
        after writing a time step results in undefined behaviour.

        Examples
        --------
        >>> w.define_global_variables(["total_energy", "kinetic_energy"])
        """

        self.define_variables(Entity.GLOBAL, names)

    def define_node_variables(self, names: Sequence[str]) -> None:
        """Define nodal result variable names.

        Nodal variables hold one value per node per time step (e.g.
        displacement components, temperature).

        Parameters
        ----------
        names : sequence of str
            Variable names, each padded or truncated to 32 characters.

        Notes
        -----
        Must be called **before** the first :meth:`write_time` call.

        Examples
        --------
        >>> w.define_node_variables(["disp_x", "disp_y", "disp_z"])
        """

        self.define_variables(Entity.NODE, names)

    def define_element_variables(
        self, names: Sequence[str], *, truth_table: npt.ArrayLike | None = None
    ) -> None:
        """Define element result variable names.

        Element variables hold one value per element per time step for each
        block in which they are active (e.g. stress, strain energy density).

        Parameters
        ----------
        names : sequence of str
            Variable names, each padded or truncated to 32 characters.
        truth_table : array_like of int, optional
            Activation mask of shape ``(n_element_blocks, len(names))``.  A
            value of ``1`` means the variable is stored for that block; ``0``
            means no storage is allocated.  If ``None``, all variables are
            active for all blocks.

        Notes
        -----
        Must be called **before** the first :meth:`write_time` call.  Element
        blocks must be defined before this method is called.

        Examples
        --------
        >>> # "stress" active only in block 0; "strain" active in all blocks
        >>> truth = [[1, 1], [0, 1]]
        >>> w.define_element_variables(["stress", "strain"], truth_table=truth)
        """

        self.define_variables(Entity.ELEMENT, names, truth_table=truth_table)

    def define_edge_variables(
        self, names: Sequence[str], *, truth_table: npt.ArrayLike | None = None
    ) -> None:
        """Define edge result variable names.

        Parameters
        ----------
        names : sequence of str
            Variable names, each padded or truncated to 32 characters.
        truth_table : array_like of int, optional
            Activation mask of shape ``(n_edge_blocks, len(names))``.  If
            ``None``, all variables are active for all edge blocks.

        Notes
        -----
        Edge blocks must be defined before this method is called.  Must be
        called before the first :meth:`write_time`.

        Examples
        --------
        >>> w.define_edge_variables(["axial_force"])
        """

        self.define_variables(Entity.EDGE, names, truth_table=truth_table)

    def define_face_variables(
        self, names: Sequence[str], *, truth_table: npt.ArrayLike | None = None
    ) -> None:
        """Define face result variable names.

        Parameters
        ----------
        names : sequence of str
            Variable names, each padded or truncated to 32 characters.
        truth_table : array_like of int, optional
            Activation mask of shape ``(n_face_blocks, len(names))``.  If
            ``None``, all variables are active for all face blocks.

        Notes
        -----
        Face blocks must be defined before this method is called.  Must be
        called before the first :meth:`write_time`.

        Examples
        --------
        >>> w.define_face_variables(["pressure"])
        """

        self.define_variables(Entity.FACE, names, truth_table=truth_table)

    def define_node_set_variables(
        self, names: Sequence[str], *, truth_table: npt.ArrayLike | None = None
    ) -> None:
        """Define node-set result variable names.

        Parameters
        ----------
        names : sequence of str
            Variable names, each padded or truncated to 32 characters.
        truth_table : array_like of int, optional
            Activation mask of shape ``(n_node_sets, len(names))``.  If
            ``None``, all variables are active for all node sets.

        Notes
        -----
        Node sets must be defined before this method is called.  Must be
        called before the first :meth:`write_time`.

        Examples
        --------
        >>> w.define_node_set_variables(["reaction_x", "reaction_y"])
        """

        self.define_variables(Entity.NODE_SET, names, truth_table=truth_table)

    def define_side_set_variables(
        self, names: Sequence[str], *, truth_table: npt.ArrayLike | None = None
    ) -> None:
        """Define side-set result variable names.

        Parameters
        ----------
        names : sequence of str
            Variable names, each padded or truncated to 32 characters.
        truth_table : array_like of int, optional
            Activation mask of shape ``(n_side_sets, len(names))``.  If
            ``None``, all variables are active for all side sets.

        Notes
        -----
        Side sets must be defined before this method is called.  Must be
        called before the first :meth:`write_time`.

        Examples
        --------
        >>> w.define_side_set_variables(["heat_flux"])
        """

        self.define_variables(Entity.SIDE_SET, names, truth_table=truth_table)

    def define_edge_set_variables(
        self, names: Sequence[str], *, truth_table: npt.ArrayLike | None = None
    ) -> None:
        """Define edge-set result variables."""

        self.define_variables(Entity.EDGE_SET, names, truth_table=truth_table)

    def define_face_set_variables(
        self, names: Sequence[str], *, truth_table: npt.ArrayLike | None = None
    ) -> None:
        """Define face-set result variables."""

        self.define_variables(Entity.FACE_SET, names, truth_table=truth_table)

    def define_element_set_variables(
        self, names: Sequence[str], *, truth_table: npt.ArrayLike | None = None
    ) -> None:
        """Define element-set result variables."""

        self.define_variables(Entity.ELEMENT_SET, names, truth_table=truth_table)

    def write_time(self, value: float, *, step: int | None = None) -> int:
        """Write a time value and advance (or set) the current time step.

        This method must be called **before** writing any result values for
        a given step.  Result-value methods (:meth:`write_global_values`,
        :meth:`write_node_values`, etc.) default to writing at the most
        recently written step when ``step`` is not supplied.

        Parameters
        ----------
        value : float
            Physical time value to record.
        step : int, optional
            Explicit 1-based time-step index.  If ``None`` (default), the
            internal counter is incremented and the next sequential step is
            used.  If provided, the internal counter is updated to
            ``max(current_step, step)``.

        Returns
        -------
        int
            The 1-based time-step index at which ``value`` was written.

        Raises
        ------
        ExodusWriteError
            If the database has not been initialised.
        ValueError
            If ``step`` is provided and is less than 1.

        Notes
        -----
        Each call increments the unlimited ``time`` dimension in the NetCDF4
        file by one slot (when ``step`` is ``None`` or a new maximum).
        Calling with an existing step index overwrites that step's time value.

        Examples
        --------
        >>> step = w.write_time(0.0)      # step == 1
        >>> step = w.write_time(0.1)      # step == 2
        >>> step = w.write_time(1.0, step=10)  # step == 10
        """

        self._require_initialized()

        if step is None:
            self._time_step += 1
            step = self._time_step
        elif step < 1:
            raise ValueError("step must be one-based and positive")
        else:
            self._time_step = max(self._time_step, step)

        self._backend.write_variable(VariableName.TIME.value, float(value), step - 1)
        return step

    def write_values(
        self,
        name: str,
        values: npt.ArrayLike,
        *,
        on: Entity,
        step: int | None = None,
        block_id: int | None = None,
        set_id: int | None = None,
    ) -> None:
        """Write result values for a named variable at a time step.

        This is the generic entry point used by the type-specific
        ``write_*_values`` convenience methods.

        Parameters
        ----------
        name : str
            Variable name, matched case-insensitively against the names
            registered with the corresponding ``define_*_variables`` call.
        values : array_like of float
            Result values to write.  Shape depends on ``on``:

            - ``Entity.GLOBAL`` — shape ``(n_global_vars,)`` (all global
              values at once; see also :meth:`write_global_values`).
            - ``Entity.NODE`` — shape ``(node_count,)``.
            - ``Entity.ELEMENT`` — shape ``(n_elems_in_block,)``; requires
              ``block_id``.
            - ``Entity.EDGE`` — shape ``(n_edges_in_block,)``; requires
              ``block_id``.
            - ``Entity.FACE`` — shape ``(n_faces_in_block,)``; requires
              ``block_id``.
            - ``Entity.NODE_SET`` — shape ``(n_nodes_in_set,)``; requires
              ``set_id``.
            - ``Entity.SIDE_SET`` — shape ``(n_sides_in_set,)``; requires
              ``set_id``.
        on : Entity
            Entity type to which the variable belongs.
        step : int, optional
            1-based time-step index.  If ``None`` (default), the current step
            (set by the most recent :meth:`write_time` call) is used.
        block_id : int, optional
            Required when ``on`` is a block entity type.
        set_id : int, optional
            Required when ``on`` is a set entity type.

        Raises
        ------
        ExodusWriteError
            If no time step has been written yet and ``step`` is ``None``.
        ExodusLookupError
            If ``name`` is not found among the registered variable names.
        ValueError
            If ``block_id`` or ``set_id`` is required but not provided.

        Examples
        --------
        >>> w.write_time(1.0)
        >>> w.write_values("temperature", temp_array, on=Entity.NODE)
        >>> w.write_values("stress", stress_array, on=Entity.ELEMENT, block_id=1)
        """

        step = self._current_or_requested_step(step)
        spec = variable_spec(on)
        variable_index = self._name_index(spec.names_variable, name)

        if on is Entity.GLOBAL:
            self._backend.write_variable(
                variable_value_name(on, variable_index),
                np.asarray(values, dtype=np.float64),
                step - 1,
            )
            return

        if on is Entity.NODE:
            self._backend.write_variable(
                variable_value_name(on, variable_index),
                np.asarray(values, dtype=np.float64),
                step - 1,
            )
            return

        if spec.location_entity is None:
            raise ValueError(f"{on.value} variables do not have a location entity")

        if spec.location_entity.is_block:
            if block_id is None:
                raise ValueError("block_id is required")
            location_index = self._location_index(spec.location_entity, block_id)
        elif spec.location_entity.is_set:
            if set_id is None:
                raise ValueError("set_id is required")
            location_index = self._location_index(spec.location_entity, set_id)
        else:
            raise ValueError(f"unsupported variable location {spec.location_entity.value}")

        self._backend.write_variable(
            variable_value_name(on, variable_index, location_index),
            np.asarray(values, dtype=np.float64),
            step - 1,
        )

    def write_element_edge_connectivity(
        self, block_id: int, connectivity: npt.ArrayLike, *, zero_based: bool = False
    ) -> None:
        """Write element-to-edge connectivity for an existing element block."""

        block_index = self._location_index(Entity.ELEMENT_BLOCK, block_id)
        spec = block_spec(Entity.ELEMENT_BLOCK)

        if spec.edges_per_object_dimension is None or spec.edge_connectivity_variable is None:
            raise ValueError("element blocks do not support edge connectivity")

        conn = np.asarray(connectivity, dtype=np.int64)
        if conn.ndim != 2:
            raise ValueError("edge connectivity must be two-dimensional")
        if zero_based:
            conn = conn + 1

        count_dim = spec.object_count_dimension(block_index)
        edge_dim = spec.edges_per_object_dimension(block_index)
        variable = spec.edge_connectivity_variable(block_index)

        if not self._backend.has_dimension(edge_dim):
            self._backend.create_dimension(edge_dim, conn.shape[1])
        if not self._backend.has_variable(variable):
            self._backend.create_variable(variable, int, (count_dim, edge_dim))

        self._backend.write_variable(variable, conn)

    def write_element_face_connectivity(
        self, block_id: int, connectivity: npt.ArrayLike, *, zero_based: bool = False
    ) -> None:
        """Write element-to-face connectivity for an existing element block."""

        block_index = self._location_index(Entity.ELEMENT_BLOCK, block_id)
        spec = block_spec(Entity.ELEMENT_BLOCK)

        if spec.faces_per_object_dimension is None or spec.face_connectivity_variable is None:
            raise ValueError("element blocks do not support face connectivity")

        conn = np.asarray(connectivity, dtype=np.int64)
        if conn.ndim != 2:
            raise ValueError("face connectivity must be two-dimensional")
        if zero_based:
            conn = conn + 1

        count_dim = spec.object_count_dimension(block_index)
        face_dim = spec.faces_per_object_dimension(block_index)
        variable = spec.face_connectivity_variable(block_index)

        if not self._backend.has_dimension(face_dim):
            self._backend.create_dimension(face_dim, conn.shape[1])
        if not self._backend.has_variable(variable):
            self._backend.create_variable(variable, int, (count_dim, face_dim))

        self._backend.write_variable(variable, conn)

    def write_global_values(self, values: npt.ArrayLike, *, step: int | None = None) -> None:
        """Write all global variable values at a time step.

        Parameters
        ----------
        values : array_like of float, shape (n_global_vars,)
            Values for **all** global variables in the order they were
            registered with :meth:`define_global_variables`.
        step : int, optional
            1-based time-step index.  If ``None`` (default), the current step
            (set by the most recent :meth:`write_time` call) is used.

        Raises
        ------
        ExodusWriteError
            If no time step has been written yet and ``step`` is ``None``.

        Examples
        --------
        >>> w.write_time(1.0)
        >>> w.write_global_values([1.234e6, 0.456e6])  # total_energy, kinetic_energy
        """

        step = self._current_or_requested_step(step)
        self._backend.write_variable(
            VariableName.GLOBAL_VARIABLE_VALUES.value,
            np.asarray(values, dtype=np.float64),
            step - 1,
        )

    def write_node_values(
        self, name: str, values: npt.ArrayLike, *, step: int | None = None
    ) -> None:
        """Write one nodal variable for all nodes at a time step.

        Parameters
        ----------
        name : str
            Variable name as registered with :meth:`define_node_variables`.
            Matched case-insensitively.
        values : array_like of float, shape (node_count,)
            One value per node in the same order as the nodal coordinates.
        step : int, optional
            1-based time-step index.  If ``None`` (default), the current step
            is used.

        Raises
        ------
        ExodusWriteError
            If no time step has been written yet and ``step`` is ``None``.
        ExodusLookupError
            If ``name`` is not a registered nodal variable.

        Examples
        --------
        >>> w.write_time(0.5)
        >>> w.write_node_values("temperature", np.linspace(300, 400, node_count))
        """

        self.write_values(name, values, on=Entity.NODE, step=step)

    def write_element_values(
        self, name: str, values: npt.ArrayLike, *, block_id: int, step: int | None = None
    ) -> None:
        """Write one element variable for all elements in a block at a time step.

        Parameters
        ----------
        name : str
            Variable name as registered with :meth:`define_element_variables`.
            Matched case-insensitively.
        values : array_like of float, shape (n_elems_in_block,)
            One value per element in the specified block.
        block_id : int
            ID of the element block to write into.
        step : int, optional
            1-based time-step index.  If ``None`` (default), the current step
            is used.

        Raises
        ------
        ExodusWriteError
            If no time step has been written yet and ``step`` is ``None``.
        ExodusLookupError
            If ``name`` is not a registered element variable or ``block_id``
            does not correspond to a defined element block.

        Examples
        --------
        >>> w.write_time(1.0)
        >>> w.write_element_values("von_mises", stress_array, block_id=1)
        """

        self.write_values(name, values, on=Entity.ELEMENT, block_id=block_id, step=step)

    def write_edge_values(
        self, name: str, values: npt.ArrayLike, *, block_id: int, step: int | None = None
    ) -> None:
        """Write one edge variable for all edges in a block at a time step.

        Parameters
        ----------
        name : str
            Variable name as registered with :meth:`define_edge_variables`.
            Matched case-insensitively.
        values : array_like of float, shape (n_edges_in_block,)
            One value per edge in the specified edge block.
        block_id : int
            ID of the edge block to write into.
        step : int, optional
            1-based time-step index.  If ``None`` (default), the current step
            is used.

        Raises
        ------
        ExodusWriteError
            If no time step has been written yet and ``step`` is ``None``.
        ExodusLookupError
            If ``name`` is not a registered edge variable or ``block_id``
            does not correspond to a defined edge block.

        Examples
        --------
        >>> w.write_time(0.1)
        >>> w.write_edge_values("axial_force", force_array, block_id=10)
        """

        self.write_values(name, values, on=Entity.EDGE, block_id=block_id, step=step)

    def write_face_values(
        self, name: str, values: npt.ArrayLike, *, block_id: int, step: int | None = None
    ) -> None:
        """Write one face variable for all faces in a block at a time step.

        Parameters
        ----------
        name : str
            Variable name as registered with :meth:`define_face_variables`.
            Matched case-insensitively.
        values : array_like of float, shape (n_faces_in_block,)
            One value per face in the specified face block.
        block_id : int
            ID of the face block to write into.
        step : int, optional
            1-based time-step index.  If ``None`` (default), the current step
            is used.

        Raises
        ------
        ExodusWriteError
            If no time step has been written yet and ``step`` is ``None``.
        ExodusLookupError
            If ``name`` is not a registered face variable or ``block_id``
            does not correspond to a defined face block.

        Examples
        --------
        >>> w.write_time(0.2)
        >>> w.write_face_values("pressure", pressure_array, block_id=20)
        """

        self.write_values(name, values, on=Entity.FACE, block_id=block_id, step=step)

    def write_node_set_values(
        self, name: str, values: npt.ArrayLike, *, set_id: int, step: int | None = None
    ) -> None:
        """Write one node-set variable for all nodes in a set at a time step.

        Parameters
        ----------
        name : str
            Variable name as registered with :meth:`define_node_set_variables`.
            Matched case-insensitively.
        values : array_like of float, shape (n_nodes_in_set,)
            One value per node in the specified node set.
        set_id : int
            ID of the node set to write into.
        step : int, optional
            1-based time-step index.  If ``None`` (default), the current step
            is used.

        Raises
        ------
        ExodusWriteError
            If no time step has been written yet and ``step`` is ``None``.
        ExodusLookupError
            If ``name`` is not a registered node-set variable or ``set_id``
            does not correspond to a defined node set.

        Examples
        --------
        >>> w.write_time(1.0)
        >>> w.write_node_set_values("reaction_x", rxn_array, set_id=1)
        """

        self.write_values(name, values, on=Entity.NODE_SET, set_id=set_id, step=step)

    def _ids_for_variable_location(self, location: Entity) -> npt.NDArray[np.int64]:
        if location is Entity.ELEMENT_BLOCK:
            return np.asarray(list(self._element_block_indices), dtype=np.int64)
        if location is Entity.EDGE_BLOCK:
            return np.asarray(list(self._edge_block_indices), dtype=np.int64)
        if location is Entity.FACE_BLOCK:
            return np.asarray(list(self._face_block_indices), dtype=np.int64)
        if location is Entity.NODE_SET:
            return np.asarray(list(self._node_set_indices), dtype=np.int64)
        if location is Entity.SIDE_SET:
            return np.asarray(list(self._side_set_indices), dtype=np.int64)
        if location is Entity.EDGE_SET:
            return np.asarray(list(self._edge_set_indices), dtype=np.int64)
        if location is Entity.FACE_SET:
            return np.asarray(list(self._face_set_indices), dtype=np.int64)
        if location is Entity.ELEMENT_SET:
            return np.asarray(list(self._element_set_indices), dtype=np.int64)
        raise ValueError(f"unsupported variable location {location.value}")

    def _location_index(self, location: Entity, id_value: int) -> int:
        try:
            if location is Entity.ELEMENT_BLOCK:
                return self._element_block_indices[id_value]
            if location is Entity.EDGE_BLOCK:
                return self._edge_block_indices[id_value]
            if location is Entity.FACE_BLOCK:
                return self._face_block_indices[id_value]
            if location is Entity.NODE_SET:
                return self._node_set_indices[id_value]
            if location is Entity.SIDE_SET:
                return self._side_set_indices[id_value]
            if location is Entity.EDGE_SET:
                return self._edge_set_indices[id_value]
            if location is Entity.FACE_SET:
                return self._face_set_indices[id_value]
            if location is Entity.ELEMENT_SET:
                return self._element_set_indices[id_value]
        except KeyError as exc:
            label = location.value.replace("_", " ")
            raise ExodusLookupError(f"{label} ID {id_value} not found") from exc

        raise ValueError(f"unsupported variable location {location.value}")

    def _location_count_dimension(self, location: Entity) -> str:
        if location is Entity.ELEMENT_BLOCK:
            return DimensionName.NUM_ELEMENT_BLOCKS.value
        if location is Entity.EDGE_BLOCK:
            return DimensionName.NUM_EDGE_BLOCKS.value
        if location is Entity.FACE_BLOCK:
            return DimensionName.NUM_FACE_BLOCKS.value
        if location is Entity.NODE_SET:
            return DimensionName.NUM_NODE_SETS.value
        if location is Entity.SIDE_SET:
            return DimensionName.NUM_SIDE_SETS.value
        if location is Entity.EDGE_SET:
            return DimensionName.NUM_EDGE_SETS.value
        if location is Entity.FACE_SET:
            return DimensionName.NUM_FACE_SETS.value
        if location is Entity.ELEMENT_SET:
            return DimensionName.NUM_ELEMENT_SETS.value
        raise ValueError(f"unsupported variable location {location.value}")

    def _value_count_dimension(self, location: Entity, location_index: int) -> str:
        if location is Entity.ELEMENT_BLOCK:
            return ExodusNames.block_count(location_index)
        if location is Entity.EDGE_BLOCK:
            return ExodusNames.edge_block_count(location_index)
        if location is Entity.FACE_BLOCK:
            return ExodusNames.face_block_count(location_index)
        if location is Entity.NODE_SET:
            return ExodusNames.node_set_count(location_index)
        if location is Entity.SIDE_SET:
            return ExodusNames.side_set_count(location_index)
        if location is Entity.EDGE_SET:
            return f"num_edge_es{location_index}"
        if location is Entity.FACE_SET:
            return f"num_face_fs{location_index}"
        if location is Entity.ELEMENT_SET:
            return f"num_ele_els{location_index}"
        raise ValueError(f"unsupported variable location {location.value}")

    def _create_standard_dimensions(self) -> None:
        self._backend.create_dimension(DimensionName.TIME.value, None)
        self._backend.create_dimension(DimensionName.STRING_LENGTH.value, 32)
        self._backend.create_dimension(DimensionName.NAME_LENGTH.value, 256)
        self._backend.create_dimension(DimensionName.LINE_LENGTH.value, 80)
        self._backend.create_dimension(DimensionName.FOUR.value, 4)

    def _create_time_variable(self) -> None:
        self._backend.create_variable(VariableName.TIME.value, float, (DimensionName.TIME.value,))

    def _create_coordinate_variables(self, dimension: int) -> None:
        self._backend.create_variable(
            VariableName.COORDINATE_NAMES.value,
            str,
            (DimensionName.NUM_DIMENSIONS.value, DimensionName.STRING_LENGTH.value),
        )
        for axis in range(dimension):
            self._backend.create_variable(
                ExodusNames.coordinate(axis), float, (DimensionName.NUM_NODES.value,)
            )

    def _allocate_blocks(
        self, *, count: int, dimension_name: str, ids_name: str, status_name: str, names_name: str
    ) -> None:
        if not count:
            return

        self._backend.create_dimension(dimension_name, count)
        self._backend.create_variable(ids_name, int, (dimension_name,))
        self._backend.create_variable(status_name, int, (dimension_name,))
        self._backend.create_variable(
            names_name, str, (dimension_name, DimensionName.STRING_LENGTH.value)
        )
        self._backend.write_variable(ids_name, np.zeros(count, dtype=np.int32))
        self._backend.write_variable(status_name, np.zeros(count, dtype=np.int32))
        self._backend.write_variable(names_name, encode_fixed_width([""] * count, width=32))
        self._backend.set_variable_attribute(ids_name, AttributeName.PROPERTY_NAME.value, "ID")

    def _allocate_sets(
        self, *, count: int, dimension_name: str, ids_name: str, status_name: str, names_name: str
    ) -> None:
        self._allocate_blocks(
            count=count,
            dimension_name=dimension_name,
            ids_name=ids_name,
            status_name=status_name,
            names_name=names_name,
        )

    def _next_block_index(self, entity: Entity, block_id: int) -> int:
        if entity is Entity.ELEMENT_BLOCK:
            self._element_block_counter += 1
            counter = self._element_block_counter
            allocated = self._dimension_size(DimensionName.NUM_ELEMENT_BLOCKS.value)
            indices = self._element_block_indices
            label = "element blocks"
        elif entity is Entity.EDGE_BLOCK:
            self._edge_block_counter += 1
            counter = self._edge_block_counter
            allocated = self._dimension_size(DimensionName.NUM_EDGE_BLOCKS.value)
            indices = self._edge_block_indices
            label = "edge blocks"
        elif entity is Entity.FACE_BLOCK:
            self._face_block_counter += 1
            counter = self._face_block_counter
            allocated = self._dimension_size(DimensionName.NUM_FACE_BLOCKS.value)
            indices = self._face_block_indices
            label = "face blocks"
        else:
            raise ValueError(f"{entity.value} is not a block entity")

        if counter > allocated:
            raise ExodusWriteError(f"allocated number of {label} exceeded")

        indices[block_id] = counter
        return counter

    def _next_set_index(self, entity: Entity, set_id: int) -> int:
        if entity is Entity.NODE_SET:
            self._node_set_counter += 1
            counter = self._node_set_counter
            allocated = self._dimension_size(DimensionName.NUM_NODE_SETS.value)
            indices = self._node_set_indices
            label = "node sets"
        elif entity is Entity.SIDE_SET:
            self._side_set_counter += 1
            counter = self._side_set_counter
            allocated = self._dimension_size(DimensionName.NUM_SIDE_SETS.value)
            indices = self._side_set_indices
            label = "side sets"
        elif entity is Entity.EDGE_SET:
            self._edge_set_counter += 1
            counter = self._edge_set_counter
            allocated = self._dimension_size(DimensionName.NUM_EDGE_SETS.value)
            indices = self._edge_set_indices
            label = "edge sets"
        elif entity is Entity.FACE_SET:
            self._face_set_counter += 1
            counter = self._face_set_counter
            allocated = self._dimension_size(DimensionName.NUM_FACE_SETS.value)
            indices = self._face_set_indices
            label = "face sets"
        elif entity is Entity.ELEMENT_SET:
            self._element_set_counter += 1
            counter = self._element_set_counter
            allocated = self._dimension_size(DimensionName.NUM_ELEMENT_SETS.value)
            indices = self._element_set_indices
            label = "element sets"
        else:
            raise ValueError(f"{entity.value} is not a set entity")

        if counter > allocated:
            raise ExodusWriteError(f"allocated number of {label} exceeded")

        indices[set_id] = counter
        return counter

    def _next_element_block_index(self, block_id: int) -> int:
        self._element_block_counter += 1
        allocated = self._dimension_size(DimensionName.NUM_ELEMENT_BLOCKS.value)
        if self._element_block_counter > allocated:
            raise ExodusWriteError("allocated number of element blocks exceeded")

        self._element_block_indices[block_id] = self._element_block_counter
        return self._element_block_counter

    def _next_node_set_index(self, set_id: int) -> int:
        self._node_set_counter += 1
        allocated = self._dimension_size(DimensionName.NUM_NODE_SETS.value)
        if self._node_set_counter > allocated:
            raise ExodusWriteError("allocated number of node sets exceeded")

        self._node_set_indices[set_id] = self._node_set_counter
        return self._node_set_counter

    def _next_side_set_index(self, set_id: int) -> int:
        self._side_set_counter += 1
        allocated = self._dimension_size(DimensionName.NUM_SIDE_SETS.value)
        if self._side_set_counter > allocated:
            raise ExodusWriteError("allocated number of side sets exceeded")

        self._side_set_indices[set_id] = self._side_set_counter
        return self._side_set_counter

    def _current_or_requested_step(self, step: int | None) -> int:
        if step is None:
            if self._time_step < 1:
                raise ExodusWriteError("write a time value before writing result values")
            return self._time_step

        if step < 1:
            raise ValueError("step must be one-based and positive")
        return step

    def _name_index(self, variable_name: str, requested: str) -> int:
        names = self._backend.variable(variable_name, default=[])
        decoded = [str(name) for name in np.asarray(names).reshape(-1)]
        for index, name in enumerate(decoded, start=1):
            if name == requested or name.lower() == requested.lower():
                return index

        raise ExodusLookupError(f"variable {requested!r} not found")

    def _dimension_size(self, name: str, default: int = 0) -> int:
        value = self._backend.dimension(name, default)
        if value is None:
            return default
        return int(value)

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise ExodusWriteError("database is not initialized")


__all__ = ["ExodusWriter"]
