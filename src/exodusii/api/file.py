# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Modern read API for Exodus databases.

This module provides :class:`ExodusFile`, the primary interface for reading
Exodus II finite-element database files.  It wraps a pluggable NetCDF backend
and exposes typed accessors for mesh topology, result variables, sets, blocks,
attributes, and metadata.  The class supports the context-manager protocol so
files are closed automatically when used in a ``with`` statement.
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from exodusii.core.entities import Entity
from exodusii.core.entities import entity
from exodusii.core.errors import ExodusInvalidEntityError
from exodusii.core.errors import ExodusLookupError
from exodusii.core.models import Block
from exodusii.core.models import InitParams
from exodusii.core.models import SetInfo
from exodusii.core.names import AttributeName
from exodusii.core.names import DimensionName
from exodusii.core.names import ExodusNames
from exodusii.core.names import VariableName
from exodusii.core.schema import VariableSpec
from exodusii.core.schema import block_spec
from exodusii.core.schema import set_spec
from exodusii.core.schema import variable_spec
from exodusii.core.schema import variable_value_name
from exodusii.core.strings import decode_text
from exodusii.core.strings import string_array
from exodusii.core.time import TimeSelector
from exodusii.core.time import resolve_time
from exodusii.io.backend import FileMode
from exodusii.io.backend import NetCDFBackend
from exodusii.io.netcdf4_backend import NetCDF4Backend


class ExodusFile:
    """Modern Exodus database reader.

    Provides read (and limited write) access to an Exodus II finite-element
    database through a pluggable :class:`~exodusii.io.backend.NetCDFBackend`.
    Immutable metadata such as time values, variable-name tables, and block/set
    ID arrays are cached for the lifetime of the instance and invalidated on
    :meth:`close` or :meth:`sync`.

    Parameters
    ----------
    backend : NetCDFBackend
        NetCDF backend implementing
        :class:`exodusii.io.backend.NetCDFBackend`.

    Notes
    -----
    The preferred usage pattern is as a context manager, which guarantees the
    underlying file handle is closed even if an exception occurs::

        with ExodusFile.open("results.exo") as f:
            coords = f.coordinates()
            temps = f.values("temperature", on="node")

    For one-off interactive use :meth:`open` / :meth:`close` may also be called
    directly.

    Examples
    --------
    Open a file, read nodal coordinates, and close explicitly:

    >>> f = ExodusFile.open("results.exo")
    >>> coords = f.coordinates()
    >>> f.close()

    Use as a context manager (recommended):

    >>> with ExodusFile.open("results.exo") as f:
    ...     t = f.times()
    ...     temp = f.values("temperature", on="node", time=t[-1])
    """

    def __init__(self, backend: NetCDFBackend) -> None:
        self._backend = backend
        # Cache for immutable metadata reads (times, variable-name tables,
        # set/block id arrays, and index lookups).  ExodusFile is a pure
        # reader (all writes go through ExodusWriter with its own backend),
        # so these are stable for the lifetime of the instance.  The cache
        # is cleared on close()/sync() as a safety measure.
        self._cache: dict[str, Any] = {}

    @classmethod
    def open(cls, path: str | Path, mode: str = "r") -> "ExodusFile":
        """Open an Exodus database file and return an :class:`ExodusFile`.

        Parameters
        ----------
        path : str or Path
            Filesystem path to the ``.exo`` / ``.e`` / ``.g`` database file.
        mode : str, optional
            File open mode.  Use ``"r"`` (default) for read-only access or
            ``"r+"`` / ``"w"`` when write access is required.

        Returns
        -------
        ExodusFile
            A new :class:`ExodusFile` instance backed by a
            :class:`~exodusii.io.netcdf4_backend.NetCDF4Backend`.

        Examples
        --------
        >>> f = ExodusFile.open("results.exo")
        >>> f.node_count
        1024
        >>> f.close()
        """

        return cls(NetCDF4Backend(path, mode=mode))

    @property
    def backend(self) -> NetCDFBackend:
        """Underlying NetCDF backend."""

        return self._backend

    @property
    def path(self) -> Path:
        """Database path."""

        return self._backend.path

    @property
    def mode(self) -> FileMode:
        """Open mode."""

        return self._backend.mode

    def close(self) -> None:
        """Close the database and release the file handle.

        Notes
        -----
        The internal metadata cache is cleared before the backend is closed.
        When using the context-manager protocol the file is closed
        automatically; explicit calls to :meth:`close` are only needed for
        non-context-manager usage.

        Examples
        --------
        >>> f = ExodusFile.open("results.exo")
        >>> f.close()
        """

        self._cache.clear()
        self._backend.close()

    def sync(self) -> None:
        """Flush pending writes to disk.

        Notes
        -----
        Clears the internal metadata cache as a safety measure so that any
        data written through a companion writer is visible on the next read.
        Prefer using the context-manager protocol over explicit
        :meth:`sync` / :meth:`close` calls.

        Examples
        --------
        >>> with ExodusFile.open("results.exo", mode="r+") as f:
        ...     f.sync()
        """

        self._cache.clear()
        self._backend.sync()

    def __enter__(self) -> "ExodusFile":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def title(self) -> str:
        """Database title string stored in the file header.

        Returns
        -------
        str
            The title attribute of the Exodus database, or an empty string if
            no title was recorded.
        """

        return str(self._backend.attribute(AttributeName.TITLE.value, ""))

    @property
    def version(self) -> float | None:
        """Exodus database format version.

        Returns
        -------
        float or None
            The ``version`` global attribute cast to ``float``, or ``None`` if
            the attribute is absent from the file.
        """

        value = self._backend.attribute(AttributeName.VERSION.value, None)
        return None if value is None else float(value)

    @property
    def api_version(self) -> float | None:
        """Exodus API version used to write the file.

        Returns
        -------
        float or None
            The ``api_version`` global attribute cast to ``float``, or
            ``None`` if absent.
        """

        value = self._backend.attribute(AttributeName.API_VERSION.value, None)
        return None if value is None else float(value)

    @property
    def storage_type(self) -> str:
        """Floating-point storage type code.

        Returns
        -------
        str
            ``"f"`` when nodal/element data are stored as 32-bit floats,
            ``"d"`` when stored as 64-bit doubles.
        """

        word_size = self._backend.attribute(AttributeName.FLOATING_POINT_WORD_SIZE.value, None)
        if word_size is None:
            word_size = self._backend.attribute(
                AttributeName.FLOATING_POINT_WORD_SIZE_LEGACY.value, 8
            )

        return "f" if int(word_size) == 4 else "d"

    @property
    def dimension(self) -> int:
        """Spatial dimension of the mesh (1, 2, or 3).

        Returns
        -------
        int
            Number of spatial coordinates per node.  Returns ``0`` if the
            dimension is not recorded in the file.
        """

        return self.dimension_size(DimensionName.NUM_DIMENSIONS.value, default=0)

    @property
    def node_count(self) -> int:
        """Total number of nodes in the mesh.

        Returns
        -------
        int
            Number of nodes, or ``0`` if not present in the file.
        """

        return self.dimension_size(DimensionName.NUM_NODES.value, default=0)

    @property
    def edge_count(self) -> int:
        """Total number of edges in the mesh.

        Returns
        -------
        int
            Number of edges, or ``0`` if not present in the file.
        """

        return self.dimension_size(DimensionName.NUM_EDGES.value, default=0)

    @property
    def face_count(self) -> int:
        """Total number of faces in the mesh.

        Returns
        -------
        int
            Number of faces, or ``0`` if not present in the file.
        """

        return self.dimension_size(DimensionName.NUM_FACES.value, default=0)

    @property
    def element_count(self) -> int:
        """Total number of elements across all blocks.

        Returns
        -------
        int
            Number of elements, or ``0`` if not present in the file.
        """

        return self.dimension_size(DimensionName.NUM_ELEMENTS.value, default=0)

    @property
    def element_block_count(self) -> int:
        """Number of element blocks.

        Returns
        -------
        int
            Count of element blocks, or ``0`` if none are present.
        """

        return self.dimension_size(DimensionName.NUM_ELEMENT_BLOCKS.value, default=0)

    @property
    def node_set_count(self) -> int:
        """Number of node sets.

        Returns
        -------
        int
            Count of node sets, or ``0`` if none are present.
        """

        return self.dimension_size(DimensionName.NUM_NODE_SETS.value, default=0)

    @property
    def side_set_count(self) -> int:
        """Number of side sets.

        Returns
        -------
        int
            Count of side sets, or ``0`` if none are present.
        """

        return self.dimension_size(DimensionName.NUM_SIDE_SETS.value, default=0)

    def info_records(self) -> tuple[str, ...]:
        """Return Exodus information records stored in the file.

        Returns
        -------
        tuple of str
            Each element is one information record string with trailing
            whitespace and null bytes stripped.  Returns an empty tuple if no
            information records are present.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     for line in f.info_records():
        ...         print(line)
        """

        values = self._backend.variable(VariableName.INFO_RECORDS.value, default=None)
        if values is None:
            return ()

        decoded = string_array(values)
        return tuple(str(value).rstrip(" \x00") for value in decoded)

    def qa_records(self) -> tuple[tuple[str, str, str, str], ...]:
        """Return Exodus QA records stored in the file.

        Returns
        -------
        tuple of tuple of str
            Each inner tuple contains four strings:
            ``(code_name, code_qa, date, time)``.  Returns an empty tuple if
            no QA records are present.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     for code, qa, date, time in f.qa_records():
        ...         print(code, date)
        """

        values = self._backend.variable(VariableName.QA_RECORDS.value, default=None)
        if values is None:
            return ()

        array = np.asarray(values)
        if array.size == 0:
            return ()

        records: list[tuple[str, str, str, str]] = []

        # Backend/string decoding may have flattened 3-D char arrays to strings
        # of four fields. Preserve that form if already decoded.
        if array.ndim == 1:
            for item in array:
                fields = _four_fields(str(item).split())
                records.append(fields)
            return tuple(records)

        if array.ndim == 3:
            for record in array:
                fields = _four_fields(list(_decode_name_table(record, expected_count=4)))
                records.append(fields)
            return tuple(records)

        return ()

    def set_ids(self, on: Entity | str, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return set IDs for a set entity type.

        Parameters
        ----------
        on : Entity or str
            Set entity type.  Accepts :class:`~exodusii.core.entities.Entity`
            values or string aliases such as ``"node_set"``, ``"side_set"``,
            ``"ns"``, ``"ss"``, etc.
        active_only : bool, optional
            When ``True``, return only IDs whose status flag is non-zero.
            Default is ``False``.

        Returns
        -------
        ndarray of int64
            Sorted array of set IDs.  Returns an empty array if no sets of
            the requested type exist.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     ids = f.set_ids("node_set")
        ...     print(ids)
        [1 2 5]
        """

        spec = set_spec(on)
        cache_key = f"set_ids:{spec.entity.value}"
        ids = self._cache.get(cache_key)
        if ids is None:
            values = self._backend.variable(spec.ids_variable, default=None)
            if values is None:
                ids = np.asarray([], dtype=np.int64)
            else:
                ids = np.asarray(values, dtype=np.int64)
            ids.setflags(write=False)
            self._cache[cache_key] = ids

        if active_only:
            status = self.set_status(spec.entity)
            ids = ids[status.astype(bool)]

        return ids

    def set_status(self, on: Entity | str) -> npt.NDArray[np.int64]:
        """Return set status flags."""

        spec = set_spec(on)
        count = self.dimension_size(spec.count_dimension, default=0)
        values = self._backend.variable(spec.status_variable, default=None)

        if values is None:
            return np.ones(count, dtype=np.int64)

        return np.asarray(values, dtype=np.int64)

    def set_is_active(self, on: Entity | str, set_id: int) -> bool:
        """Return true if a set is active."""

        index = self._set_index(on, set_id)
        status = self.set_status(on)
        if index - 1 >= len(status):
            return False
        return bool(status[index - 1])

    def set(self, on: Entity | str, set_id: int) -> SetInfo:
        """Return metadata and entry arrays for a set.

        Parameters
        ----------
        on : Entity or str
            Set entity type.  Accepts :class:`~exodusii.core.entities.Entity`
            values or string aliases such as ``"node_set"`` or ``"side_set"``.
        set_id : int
            Exodus set ID (one-based, as stored in the file).

        Returns
        -------
        SetInfo
            A frozen dataclass with the following attributes:

            * ``id`` — the Exodus set ID.
            * ``index`` — one-based position in the file's set list.
            * ``entity`` — the normalized :class:`~exodusii.core.entities.Entity`.
            * ``count`` — number of entries.
            * ``distribution_factors`` — number of distribution factor values.
            * ``name`` — optional string name of the set.
            * ``entries`` — ``int64`` array of primary entry IDs (node IDs for
              node sets, element IDs for side sets).
            * ``extra_entries`` — ``int64`` array of secondary entry IDs (side
              numbers for side sets), or ``None``.
            * ``distribution_values`` — ``float64`` distribution factor array,
              or ``None``.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     ns = f.set("node_set", 10)
        ...     print(ns.nodes)
        [3 7 12 ...]
        """

        spec = set_spec(on)
        set_index = self._set_index(spec.entity, set_id)

        entries = np.asarray(
            self._backend.variable(spec.entries_variable(set_index), default=[]), dtype=np.int64
        )

        extra_entries = None
        if spec.extra_entries_variable is not None:
            values = self._backend.variable(spec.extra_entries_variable(set_index), default=None)
            if values is not None:
                extra_entries = np.asarray(values, dtype=np.int64)

        dist_facts = self._backend.variable(spec.dist_factors_variable(set_index), default=None)
        dist_array = None if dist_facts is None else np.asarray(dist_facts, dtype=np.float64)

        return SetInfo(
            id=set_id,
            index=set_index,
            entity=spec.entity,
            count=len(entries),
            distribution_factors=0 if dist_array is None else len(dist_array),
            name=self._entity_name(spec.entity, set_index),
            entries=entries,
            extra_entries=extra_entries,
            distribution_values=dist_array,
        )

    def init_params(self) -> InitParams:
        """Return top-level initialization parameters for the database.

        Returns
        -------
        InitParams
            A frozen dataclass whose fields mirror the Exodus initialization
            structure: ``title``, ``dimension``, ``nodes``, ``elements``,
            ``element_blocks``, ``node_sets``, ``side_sets``, ``edges``,
            ``edge_blocks``, ``edge_sets``, ``faces``, ``face_blocks``,
            ``face_sets``, ``element_sets``, ``node_maps``, ``element_maps``,
            ``edge_maps``, and ``face_maps``.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     p = f.init_params()
        ...     print(p.nodes, p.elements, p.dimension)
        1024 512 3
        """

        return InitParams(
            title=self.title,
            dimension=self.dimension,
            nodes=self.node_count,
            elements=self.element_count,
            element_blocks=self.element_block_count,
            node_sets=self.node_set_count,
            side_sets=self.side_set_count,
            edges=self.edge_count,
            edge_blocks=self.dimension_size(DimensionName.NUM_EDGE_BLOCKS.value, default=0),
            edge_sets=self.dimension_size(DimensionName.NUM_EDGE_SETS.value, default=0),
            faces=self.face_count,
            face_blocks=self.dimension_size(DimensionName.NUM_FACE_BLOCKS.value, default=0),
            face_sets=self.dimension_size(DimensionName.NUM_FACE_SETS.value, default=0),
            element_sets=self.dimension_size(DimensionName.NUM_ELEMENT_SETS.value, default=0),
            node_maps=self.dimension_size(DimensionName.NUM_NODE_MAPS.value, default=0),
            element_maps=self.dimension_size(DimensionName.NUM_ELEMENT_MAPS.value, default=0),
            edge_maps=self.dimension_size(DimensionName.NUM_EDGE_MAPS.value, default=0),
            face_maps=self.dimension_size(DimensionName.NUM_FACE_MAPS.value, default=0),
        )

    def dimensions(self) -> tuple[str, ...]:
        """Return all NetCDF dimension names."""

        return self._backend.dimensions()

    def variables(self) -> tuple[str, ...]:
        """Return all NetCDF variable names."""

        return self._backend.variables()

    def dimension_size(self, name: str, *, default: int | None = None) -> int:
        """Return a dimension size."""

        value = self._backend.dimension(name, default)
        if value is None:
            raise ExodusLookupError(f"dimension {name!r} not found")
        return int(value)

    def variable(self, name: str, *, default: Any = None, raw: bool = False) -> Any:
        """Return a raw NetCDF variable value."""

        return self._backend.variable(name, default=default, raw=raw)

    def times(self) -> npt.NDArray[np.float64]:
        """Return all simulation time values stored in the database.

        Results are cached after the first call.

        Returns
        -------
        ndarray of float64, shape (n_steps,)
            Monotonically increasing time values, one per output step.
            Returns an empty array if the file has no time steps.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     t = f.times()
        ...     print(t)
        [0.   0.1  0.2  0.5  1.0]
        """

        cached = self._cache.get("times")
        if cached is None:
            values = self._backend.variable(VariableName.TIME.value, default=[])
            cached = np.asarray(values, dtype=np.float64)
            cached.setflags(write=False)
            self._cache["times"] = cached
        return cached

    def coordinate_names(self) -> npt.NDArray[np.str_]:
        """Return the coordinate axis names stored in the file.

        Returns
        -------
        ndarray of str, shape (dimension,)
            Axis labels in file order (e.g. ``["x", "y", "z"]`` for a 3-D
            mesh).  Falls back to ``["X", "Y", "Z"]`` when no names are
            stored.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     print(f.coordinate_names())
        ['x' 'y' 'z']
        """

        default = np.asarray(["X", "Y", "Z"][: self.dimension], dtype=object)
        values = self._backend.variable(VariableName.COORDINATE_NAMES.value, default=default)
        names = _decode_name_table(values, expected_count=self.dimension)
        return np.asarray(names[: self.dimension], dtype=str)

    def coordinates(
        self, *, time: TimeSelector = None, displaced: bool = False
    ) -> npt.NDArray[np.float64]:
        """Return nodal coordinates, optionally displaced at a time step.

        Supports both large-model files (separate ``coordx``/``coordy``/
        ``coordz`` variables, the modern default) and normal-model files (a
        combined 2-D ``coord`` variable with shape ``(num_dim, num_nodes)``),
        matching the ``ex_large_model`` branching in the SEACAS C library
        (``ex_get_coord.c``).

        If ``displaced`` is ``True``, displacement variables are added at the
        selected time step.

        Parameters
        ----------
        time : TimeSelector, optional
            Time step selector used only when ``displaced=True``.  May be a
            float (physical time), an integer step index, or ``None`` to use
            the last available step.
        displaced : bool, optional
            When ``True``, add nodal displacement values (looked up via
            :meth:`displacements`) to the reference coordinates before
            returning.  Default is ``False``.

        Returns
        -------
        ndarray of float64, shape (node_count, dimension)
            Coordinate matrix with one row per node and one column per spatial
            dimension.

        Raises
        ------
        ValueError
            If neither the per-component (``coordx``, …) nor the combined
            ``coord`` variable is found in the file.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     xyz = f.coordinates()
        ...     xyz.shape
        (1024, 3)

        Displaced coordinates at the last time step:

        >>> with ExodusFile.open("results.exo") as f:
        ...     xyz = f.coordinates(displaced=True)
        """

        # Try large-model format first (coordx / coordy / coordz).
        coord_names = [ExodusNames.coordinate(i) for i in range(self.dimension)]
        components = [self._backend.variable(name, default=None) for name in coord_names]

        if all(c is not None for c in components):
            coords = np.column_stack(components).astype(np.float64)
        else:
            # Fall back to normal-model combined ``coord`` variable
            # (shape: num_dim x num_nodes, stored row-major).
            combined = self._backend.variable(VariableName.COORDINATES.value, default=None)
            if combined is None:
                raise ValueError(
                    "nodal coordinates not found: neither 'coordx'/'coordy'/'coordz' "
                    "nor the combined 'coord' variable is present in the file"
                )
            # combined shape is (num_dim, num_nodes); transpose to (num_nodes, num_dim)
            coords = np.asarray(combined, dtype=np.float64).T

        if displaced:
            coords = coords + self.displacements(time=time)

        return coords

    def displacement_variable_names(self) -> tuple[str, ...]:
        """Return the names of nodal displacement result variables.

        Searches the nodal variable list for names matching common patterns
        (``displx``/``disply``/``displz``, ``dispx``/``dispy``/``dispz``,
        ``displ_x``/``displ_y``/``displ_z``).

        Returns
        -------
        tuple of str
            Ordered displacement variable names, one per spatial dimension
            (e.g. ``("DISPLX", "DISPLY", "DISPLZ")`` for a 3-D mesh).
            Returns an empty tuple when no displacement variables are found.

        Raises
        ------
        ValueError
            If displacement-like variable names are found but their count does
            not equal :attr:`dimension`.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     print(f.displacement_variable_names())
        ('DISPLX', 'DISPLY', 'DISPLZ')
        """

        candidates = {
            f"{base}{axis}"
            for base in ("displ", "disp", "displ_")
            for axis in "xyz"[: self.dimension]
        }
        names = self.variable_names(Entity.NODE)
        found = tuple(name for name in names if name.lower() in candidates)

        if not found:
            return ()
        if len(found) != self.dimension:
            raise ValueError("incorrect number of displacement variable names found")

        return found

    def displacements(self, *, time: TimeSelector = None) -> npt.NDArray[np.float64]:
        """Return nodal displacement vectors at a selected time step.

        Parameters
        ----------
        time : TimeSelector, optional
            Time step selector.  May be a float (physical time), an integer
            step index, or ``None`` to select the last available step.

        Returns
        -------
        ndarray of float64, shape (node_count, dimension)
            Displacement matrix with one row per node and one column per
            spatial dimension.  Returns an all-zeros array if no displacement
            variables are present in the file.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     d = f.displacements(time=0.5)
        ...     d.shape
        (1024, 3)
        """

        names = self.displacement_variable_names()
        if not names:
            return np.zeros((self.node_count, self.dimension), dtype=np.float64)

        return np.column_stack([self.values(name, on=Entity.NODE, time=time) for name in names])

    def ids(self, on: Entity | str) -> npt.NDArray[np.int64]:
        """Return Exodus IDs for a mesh object or map entity.

        For ``"node"`` and ``"element"`` entities the file may store explicit
        node/element maps; if the map variable is absent a contiguous
        ``[1, …, count]`` array is synthesised.  For block and set entities
        the block/set ID arrays are returned.

        Parameters
        ----------
        on : Entity or str
            Entity type.  Accepts :class:`~exodusii.core.entities.Entity`
            values or string aliases such as ``"node"``, ``"element"``,
            ``"element_block"``, ``"node_set"``, etc.

        Returns
        -------
        ndarray of int64
            Array of Exodus IDs for the requested entity type.  Returns an
            empty array when none exist.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     node_ids = f.ids("node")
        ...     elem_ids = f.ids("element")
        """

        ent = entity(on)
        name = ExodusNames.ids(ent)

        default: npt.NDArray[np.int64] | None
        if ent is Entity.NODE:
            default = np.arange(1, self.node_count + 1, dtype=np.int64)
        elif ent is Entity.ELEMENT:
            default = np.arange(1, self.element_count + 1, dtype=np.int64)
        elif ent is Entity.EDGE:
            default = np.arange(1, self.edge_count + 1, dtype=np.int64)
        elif ent is Entity.FACE:
            default = np.arange(1, self.face_count + 1, dtype=np.int64)
        else:
            default = None

        values = self._backend.variable(name, default=default)
        if values is None:
            return np.asarray([], dtype=np.int64)

        return np.asarray(values, dtype=np.int64)

    def element_block_ids(self, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return all element block IDs.

        Parameters
        ----------
        active_only : bool, optional
            When ``True``, return only IDs whose block status flag is
            non-zero.  Default is ``False``.

        Returns
        -------
        ndarray of int64
            Array of element block IDs.  Returns an empty array if no element
            blocks are present.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     print(f.element_block_ids())
        [1 2 3]
        """

        return self.block_ids(Entity.ELEMENT_BLOCK, active_only=active_only)

    def block_ids(self, on: Entity | str, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return block IDs for an element, edge, or face block entity.

        Results are cached after the first call for each entity type.

        Parameters
        ----------
        on : Entity or str
            Block entity type.  Accepts
            :class:`~exodusii.core.entities.Entity` values or string aliases
            such as ``"element_block"``, ``"edge_block"``, ``"face_block"``,
            ``"eb"``, etc.
        active_only : bool, optional
            When ``True``, return only IDs whose block status flag is
            non-zero.  Default is ``False``.

        Returns
        -------
        ndarray of int64
            Array of block IDs in file order.  Returns an empty array if no
            blocks of the requested type exist.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     print(f.block_ids("element_block"))
        [10 20 30]
        """

        spec = block_spec(on)
        cache_key = f"block_ids:{spec.entity.value}"
        ids = self._cache.get(cache_key)
        if ids is None:
            values = self._backend.variable(spec.ids_variable, default=None)
            if values is None:
                ids = np.asarray([], dtype=np.int64)
            else:
                ids = np.asarray(values, dtype=np.int64)
            ids.setflags(write=False)
            self._cache[cache_key] = ids

        if active_only:
            status = self.block_status(spec.entity)
            ids = ids[status.astype(bool)]

        return ids

    def block_status(self, on: Entity | str) -> npt.NDArray[np.int64]:
        """Return block status flags."""

        spec = block_spec(on)
        count = self.dimension_size(spec.count_dimension, default=0)
        values = self._backend.variable(spec.status_variable, default=None)

        if values is None:
            return np.ones(count, dtype=np.int64)

        return np.asarray(values, dtype=np.int64)

    def block_is_active(self, on: Entity | str, block_id: int) -> bool:
        """Return true if a block is active."""

        index = self._block_index(on, block_id)
        status = self.block_status(on)
        if index - 1 >= len(status):
            return False
        return bool(status[index - 1])

    def block(self, on: Entity | str, block_id: int) -> Block:
        """Return metadata for a single block.

        Parameters
        ----------
        on : Entity or str
            Block entity type.  Accepts
            :class:`~exodusii.core.entities.Entity` values or string aliases
            such as ``"element_block"``, ``"edge_block"``, ``"face_block"``.
        block_id : int
            Exodus block ID as stored in the file.

        Returns
        -------
        Block
            A frozen dataclass with the following attributes:

            * ``id`` — the Exodus block ID.
            * ``index`` — one-based position in the file's block list.
            * ``entity`` — normalized :class:`~exodusii.core.entities.Entity`.
            * ``element_type`` — element type string (e.g. ``"HEX8"``).
            * ``count`` — number of elements (or edges/faces) in the block.
            * ``nodes_per_entity`` — nodes per element.
            * ``edges_per_entity`` — edges per element (0 if not stored).
            * ``faces_per_entity`` — faces per element (0 if not stored).
            * ``attributes`` — number of per-element attributes.
            * ``name`` — optional string name.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     b = f.block("element_block", 1)
        ...     print(b.element_type, b.count, b.nodes_per_entity)
        HEX8 512 8
        """

        spec = block_spec(on)
        block_index = self._block_index(spec.entity, block_id)
        conn_name = spec.connectivity_variable(block_index)

        element_type = ""
        if self._backend.has_variable(conn_name):
            element_type = str(
                self._backend.variable_attribute(
                    conn_name, AttributeName.ELEMENT_TYPE.value, default=""
                )
            )

        attributes = (
            self.dimension_size(spec.attributes_dimension(block_index), default=0)
            if spec.attributes_dimension is not None
            else 0
        )
        edges_per_object = (
            self.dimension_size(spec.edges_per_object_dimension(block_index), default=0)
            if spec.edges_per_object_dimension is not None
            else 0
        )
        faces_per_object = (
            self.dimension_size(spec.faces_per_object_dimension(block_index), default=0)
            if spec.faces_per_object_dimension is not None
            else 0
        )

        return Block(
            id=block_id,
            index=block_index,
            entity=spec.entity,
            element_type=element_type,
            count=self.dimension_size(spec.object_count_dimension(block_index), default=0),
            nodes_per_entity=self.dimension_size(
                spec.nodes_per_object_dimension(block_index), default=0
            ),
            edges_per_entity=edges_per_object,
            faces_per_entity=faces_per_object,
            attributes=attributes,
            name=self._entity_name(spec.entity, block_index),
        )

    def block_connectivity(
        self, on: Entity | str, block_id: int, *, zero_based: bool = False
    ) -> npt.NDArray[np.int64]:
        """Return nodal connectivity for a block.

        Parameters
        ----------
        on : Entity or str
            Block entity type (``"element_block"``, ``"edge_block"``, or
            ``"face_block"``).
        block_id : int
            Exodus block ID as stored in the file.
        zero_based : bool, optional
            When ``True``, subtract 1 from all node indices so that the
            returned array uses 0-based indexing compatible with NumPy array
            indexing.  Default is ``False`` (Exodus 1-based convention).

        Returns
        -------
        ndarray of int64, shape (n_elems, nodes_per_elem)
            Connectivity table.  Each row lists the node IDs (1-based by
            default) for one element.  Returns an empty ``(0, 0)`` array if
            the block has no connectivity data.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     conn = f.block_connectivity("element_block", 1, zero_based=True)
        ...     conn.shape
        (512, 8)
        """

        spec = block_spec(on)
        block_index = self._block_index(spec.entity, block_id)
        values = self._backend.variable(spec.connectivity_variable(block_index), default=None)
        if values is None:
            return np.empty((0, 0), dtype=np.int64)
        array = np.asarray(values, dtype=np.int64)
        return array - 1 if zero_based else array

    def element_edge_connectivity(
        self, block_id: int, *, zero_based: bool = False
    ) -> npt.NDArray[np.int64] | None:
        """Return element-to-edge connectivity for an element block."""

        spec = block_spec(Entity.ELEMENT_BLOCK)
        block_index = self._block_index(Entity.ELEMENT_BLOCK, block_id)

        if spec.edge_connectivity_variable is None:
            return None

        name = spec.edge_connectivity_variable(block_index)
        values = self._backend.variable(name, default=None)
        if values is None:
            return None

        array = np.asarray(values, dtype=np.int64)
        return array - 1 if zero_based else array

    def element_face_connectivity(
        self, block_id: int, *, zero_based: bool = False
    ) -> npt.NDArray[np.int64] | None:
        """Return element-to-face connectivity for an element block."""

        spec = block_spec(Entity.ELEMENT_BLOCK)
        block_index = self._block_index(Entity.ELEMENT_BLOCK, block_id)

        if spec.face_connectivity_variable is None:
            return None

        name = spec.face_connectivity_variable(block_index)
        values = self._backend.variable(name, default=None)
        if values is None:
            return None

        array = np.asarray(values, dtype=np.int64)
        return array - 1 if zero_based else array

    def node_set_ids(self, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return all node set IDs.

        Parameters
        ----------
        active_only : bool, optional
            When ``True``, return only IDs whose status flag is non-zero.
            Default is ``False``.

        Returns
        -------
        ndarray of int64
            Array of node set IDs.  Returns an empty array if none exist.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     print(f.node_set_ids())
        [1 2]
        """

        return self.set_ids(Entity.NODE_SET, active_only=active_only)

    def side_set_ids(self, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return all side set IDs.

        Parameters
        ----------
        active_only : bool, optional
            When ``True``, return only IDs whose status flag is non-zero.
            Default is ``False``.

        Returns
        -------
        ndarray of int64
            Array of side set IDs.  Returns an empty array if none exist.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     print(f.side_set_ids())
        [10 20]
        """

        return self.set_ids(Entity.SIDE_SET, active_only=active_only)

    def edge_set_ids(self, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return all edge set IDs."""

        return self.set_ids(Entity.EDGE_SET, active_only=active_only)

    def face_set_ids(self, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return all face set IDs."""

        return self.set_ids(Entity.FACE_SET, active_only=active_only)

    def element_set_ids(self, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return all element set IDs."""

        return self.set_ids(Entity.ELEMENT_SET, active_only=active_only)

    def element_block(self, block_id: int) -> Block:
        """Return metadata for an element block.

        Parameters
        ----------
        block_id : int
            Exodus element block ID.

        Returns
        -------
        Block
            Metadata dataclass; see :meth:`block` for field descriptions.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     b = f.element_block(1)
        ...     print(b.element_type, b.count)
        HEX8 512
        """

        return self.block(Entity.ELEMENT_BLOCK, block_id)

    def edge_block(self, block_id: int) -> Block:
        """Return metadata for an edge block.

        Parameters
        ----------
        block_id : int
            Exodus edge block ID.

        Returns
        -------
        Block
            Metadata dataclass; see :meth:`block` for field descriptions.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     b = f.edge_block(1)
        ...     print(b.count)
        128
        """

        return self.block(Entity.EDGE_BLOCK, block_id)

    def face_block(self, block_id: int) -> Block:
        """Return metadata for a face block.

        Parameters
        ----------
        block_id : int
            Exodus face block ID.

        Returns
        -------
        Block
            Metadata dataclass; see :meth:`block` for field descriptions.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     b = f.face_block(1)
        ...     print(b.count)
        256
        """

        return self.block(Entity.FACE_BLOCK, block_id)

    def element_connectivity(
        self, block_id: int, *, zero_based: bool = False
    ) -> npt.NDArray[np.int64]:
        """Return the nodal connectivity table for an element block.

        Parameters
        ----------
        block_id : int
            Exodus element block ID.
        zero_based : bool, optional
            When ``True``, node indices are shifted to 0-based so the array
            can be used directly as NumPy indices.  Default is ``False``
            (Exodus 1-based convention).

        Returns
        -------
        ndarray of int64, shape (n_elems, nodes_per_elem)
            Connectivity table; each row contains the node IDs of one element.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     conn = f.element_connectivity(1, zero_based=True)
        ...     print(conn.shape)
        (512, 8)
        """

        return self.block_connectivity(Entity.ELEMENT_BLOCK, block_id, zero_based=zero_based)

    def edge_connectivity(
        self, block_id: int, *, zero_based: bool = False
    ) -> npt.NDArray[np.int64]:
        """Return edge-block nodal connectivity."""

        return self.block_connectivity(Entity.EDGE_BLOCK, block_id, zero_based=zero_based)

    def face_connectivity(
        self, block_id: int, *, zero_based: bool = False
    ) -> npt.NDArray[np.int64]:
        """Return face-block nodal connectivity."""

        return self.block_connectivity(Entity.FACE_BLOCK, block_id, zero_based=zero_based)

    def node_set(self, set_id: int) -> SetInfo:
        """Return metadata and entries for a node set.

        Parameters
        ----------
        set_id : int
            Exodus node set ID.

        Returns
        -------
        SetInfo
            Frozen dataclass with the following useful attributes:

            * ``nodes`` — ``int64`` array of node IDs belonging to the set.
            * ``dist_facts`` — ``float64`` distribution factor array, or
              ``None`` if no distribution factors are stored.
            * ``count`` — number of nodes.
            * ``name`` — optional string name of the set.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     ns = f.node_set(1)
        ...     print(ns.nodes[:5])
        [3 7 12 18 25]
        """

        return self.set(Entity.NODE_SET, set_id)

    def side_set(self, set_id: int) -> SetInfo:
        """Return metadata and entries for a side set.

        Parameters
        ----------
        set_id : int
            Exodus side set ID.

        Returns
        -------
        SetInfo
            Frozen dataclass with the following useful attributes:

            * ``elems`` — ``int64`` array of element IDs for each side.
            * ``sides`` — ``int64`` array of local side numbers (one per
              entry in ``elems``).
            * ``dist_facts`` — ``float64`` distribution factor array, or
              ``None`` if absent.
            * ``count`` — number of sides.
            * ``name`` — optional string name of the set.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     ss = f.side_set(10)
        ...     print(ss.elems[:3], ss.sides[:3])
        [5 6 7] [2 2 3]
        """

        return self.set(Entity.SIDE_SET, set_id)

    def edge_set(self, set_id: int) -> SetInfo:
        """Return edge-set metadata and entries."""

        return self.set(Entity.EDGE_SET, set_id)

    def face_set(self, set_id: int) -> SetInfo:
        """Return face-set metadata and entries."""

        return self.set(Entity.FACE_SET, set_id)

    def element_set(self, set_id: int) -> SetInfo:
        """Return element-set metadata and entries."""

        return self.set(Entity.ELEMENT_SET, set_id)

    def edge_block_ids(self, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return all edge block IDs.

        Parameters
        ----------
        active_only : bool, optional
            When ``True``, return only IDs whose status flag is non-zero.
            Default is ``False``.

        Returns
        -------
        ndarray of int64
            Array of edge block IDs.  Returns an empty array if none exist.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     print(f.edge_block_ids())
        [1]
        """

        return self.block_ids(Entity.EDGE_BLOCK, active_only=active_only)

    def face_block_ids(self, *, active_only: bool = False) -> npt.NDArray[np.int64]:
        """Return all face block IDs.

        Parameters
        ----------
        active_only : bool, optional
            When ``True``, return only IDs whose status flag is non-zero.
            Default is ``False``.

        Returns
        -------
        ndarray of int64
            Array of face block IDs.  Returns an empty array if none exist.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     print(f.face_block_ids())
        [1]
        """

        return self.block_ids(Entity.FACE_BLOCK, active_only=active_only)

    def property_names(self, on: Entity | str) -> tuple[str, ...]:
        """Return the names of user-defined properties for a block or set.

        Properties are integer scalars attached to each block or set and are
        accessed by name.  The first property is always ``"ID"`` (the Exodus
        ID itself).

        Parameters
        ----------
        on : Entity or str
            Block or set entity type.  Must be a block (``"element_block"``,
            ``"edge_block"``, ``"face_block"``) or set (``"node_set"``,
            ``"side_set"``, etc.) entity.

        Returns
        -------
        tuple of str
            Property names in file order.  Returns an empty tuple when no
            property variables are found or the entity type does not support
            properties.

        Raises
        ------
        ExodusInvalidEntityError
            If *on* is not a block or set entity type.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     print(f.property_names("element_block"))
        ('ID', 'MATL', 'REGION')
        """

        ent = entity(on)

        property_factory = None

        if ent.is_block:
            block_schema = block_spec(ent)
            property_factory = block_schema.property_variable
        elif ent.is_set:
            set_schema = set_spec(ent)
            property_factory = set_schema.property_variable
        else:
            raise ExodusInvalidEntityError(f"{ent.value!r} does not have properties")

        if property_factory is None:
            return ()

        names: list[str] = []
        index = 1
        while True:
            variable_name = property_factory(index)
            if not self._backend.has_variable(variable_name):
                break
            name = self._backend.variable_attribute(
                variable_name, AttributeName.PROPERTY_NAME.value, default=""
            )
            names.append(str(name))
            index += 1

        if names and not names[0]:
            names[0] = "ID"

        return tuple(names)

    def property_values(self, on: Entity | str, name: str) -> npt.NDArray[np.int64]:
        """Return all values for a named property across all blocks or sets.

        Parameters
        ----------
        on : Entity or str
            Block or set entity type.
        name : str
            Property name (case-insensitive).

        Returns
        -------
        ndarray of int64
            One value per block (or set), in file order.

        Raises
        ------
        ExodusInvalidEntityError
            If *on* is not a block or set entity.
        ExodusLookupError
            If *name* does not match any property name.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     print(f.property_values("element_block", "MATL"))
        [101 102 102]
        """

        ent = entity(on)
        property_index = self._property_index(ent, name)

        if ent.is_block:
            block_schema = block_spec(ent)
            property_factory = block_schema.property_variable
        elif ent.is_set:
            set_schema = set_spec(ent)
            property_factory = set_schema.property_variable
        else:
            raise ExodusInvalidEntityError(f"{ent.value!r} does not have properties")

        if property_factory is None:
            raise ExodusInvalidEntityError(f"{ent.value!r} does not have properties")

        variable_name = property_factory(property_index)
        return np.asarray(self._backend.variable(variable_name, default=[]), dtype=np.int64)

    def property_value(self, on: Entity | str, id_value: int, name: str) -> int:
        """Return the value of a named property for a single block or set ID.

        Parameters
        ----------
        on : Entity or str
            Block or set entity type.
        id_value : int
            Exodus block or set ID.
        name : str
            Property name (case-insensitive).

        Returns
        -------
        int
            The integer property value for the specified block/set ID.

        Raises
        ------
        ExodusInvalidEntityError
            If *on* is not a block or set entity.
        ExodusLookupError
            If *name* or *id_value* is not found.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     matl = f.property_value("element_block", 1, "MATL")
        ...     print(matl)
        101
        """

        ent = entity(on)
        values = self.property_values(ent, name)

        if ent.is_block:
            index = self._block_index(ent, id_value)
        elif ent.is_set:
            index = self._set_index(ent, id_value)
        else:
            raise ExodusInvalidEntityError(f"{ent.value!r} does not have properties")

        return int(values[index - 1])

    def _property_index(self, ent: Entity, name: str) -> int:
        requested = name.lower()
        for index, property_name in enumerate(self.property_names(ent), start=1):
            if property_name.lower() == requested:
                return index

        raise ExodusLookupError(f"property {name!r} not found for {ent.value}")

    def variable_names(self, on: Entity | str) -> tuple[str, ...]:
        """Return result variable names for a given entity location.

        Parameters
        ----------
        on : Entity or str
            Entity location.  Must be a variable location: ``"global"``,
            ``"node"``, ``"element"``, ``"edge"``, ``"face"``,
            ``"node_set"``, ``"side_set"``, ``"edge_set"``, ``"face_set"``,
            or ``"element_set"``.

        Returns
        -------
        tuple of str
            Variable names in file order, empty strings stripped.  Returns an
            empty tuple when no variables of that type exist.

        Raises
        ------
        ExodusInvalidEntityError
            If *on* is not a valid variable location.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     print(f.variable_names("node"))
        ('temperature', 'pressure', 'DISPLX', 'DISPLY', 'DISPLZ')
        """

        ent = entity(on)
        if not ent.is_variable_location:
            raise ExodusInvalidEntityError(f"{ent.value!r} is not a variable location")

        cache_key = f"variable_names:{ent.value}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        spec = variable_spec(ent)
        expected_count = self.dimension_size(spec.count_dimension, default=0)
        if expected_count == 0:
            self._cache[cache_key] = ()
            return ()

        values = self._backend.variable(spec.names_variable, default=[])
        names = _decode_name_table(values, expected_count=expected_count)
        result = tuple(name for name in names if name)
        self._cache[cache_key] = result
        return result

    def variable_truth_table(
        self, on: Entity | str, *, id: int | None = None
    ) -> npt.NDArray[np.int64] | None:
        """Return the variable truth table for block or set result variables.

        When no explicit truth table is stored in the file, the table is
        derived dynamically by probing whether each per-block/set result
        variable exists in the NetCDF file — matching the behaviour of
        ``ex_get_truth_table`` in the SEACAS C library
        (``ex_get_truth_table.c:162-178``).

        Returns ``None`` only when the entity type does not support a truth
        table (e.g. ``Entity.GLOBAL``).

        Parameters
        ----------
        on : Entity or str
            Block or set variable location (e.g. ``"element"``,
            ``"node_set"``).
        id : int, optional
            When provided, return only the single row of the truth table
            corresponding to the block or set with this Exodus ID.  When
            omitted the full 2-D table is returned.

        Returns
        -------
        ndarray of int64 or None
            * Full table: shape ``(n_blocks_or_sets, n_vars)``, values 0 or 1.
            * Single row (when *id* is given): shape ``(n_vars,)``.
            * ``None`` when the entity type has no truth table.

        Examples
        --------
        Return the full element truth table:

        >>> with ExodusFile.open("results.exo") as f:
        ...     tt = f.variable_truth_table("element")
        ...     print(tt.shape)
        (3, 5)

        Return the row for a single element block:

        >>> with ExodusFile.open("results.exo") as f:
        ...     row = f.variable_truth_table("element", id=2)
        ...     print(row)
        [1 1 0 1 1]
        """

        ent = entity(on)
        spec = variable_spec(ent)
        if spec.truth_table_variable is None:
            return None

        table = self._backend.variable(spec.truth_table_variable, default=None)
        if table is not None:
            array = np.asarray(table, dtype=np.int64)
        else:
            # Derive dynamically: probe whether vals_*_varN*M exists for every
            # (variable_index, location_index) pair — mirrors ex_get_truth_table.c.
            array = self._derive_truth_table(spec)
            if array is None:
                return None

        if id is None:
            return array

        if spec.location_entity is None:
            return array

        if spec.location_entity.is_block:
            index = self._block_index(spec.location_entity, id)
        elif spec.location_entity.is_set:
            index = self._set_index(spec.location_entity, id)
        else:
            return array

        return array[index - 1]

    def _derive_truth_table(self, spec: VariableSpec) -> npt.NDArray[np.int64] | None:
        """Derive truth table by probing per-block/set variable existence.

        Reproduces the dynamic derivation in ``ex_get_truth_table.c`` when no
        explicit truth table variable is stored.  Returns a ``(num_loc, num_var)``
        int64 array, or ``None`` if there are no variables of this type.
        """
        num_var = self.dimension_size(spec.count_dimension, default=0)
        if num_var == 0:
            return None

        if spec.location_entity is None:
            return None

        if spec.location_entity.is_block:
            ids = self.block_ids(spec.location_entity)
        elif spec.location_entity.is_set:
            ids = self.set_ids(spec.location_entity)
        else:
            return None

        num_loc = len(ids)
        if num_loc == 0:
            return None

        table = np.zeros((num_loc, num_var), dtype=np.int64)
        for loc_idx in range(num_loc):
            for var_idx in range(num_var):
                # variable_index and location_index are both 1-based
                nc_var = spec.values_variable(var_idx + 1, loc_idx + 1)
                if self._backend.has_variable(nc_var):
                    table[loc_idx, var_idx] = 1

        return table

    def values(
        self,
        name: str,
        *,
        on: Entity | str,
        time: TimeSelector = None,
        block: int | None = None,
        block_id: int | None = None,
        set_id: int | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return result variable values for any entity location.

        This is the primary method for reading simulation output.  It
        dispatches to the appropriate internal reader based on *on* and
        returns either a full time history or a snapshot at a single step.

        Parameters
        ----------
        name : str
            Variable name as it appears in the file (case-insensitive lookup
            is attempted after an exact-match failure).
        on : Entity or str
            Entity location.  Accepted values: ``"global"``, ``"node"``,
            ``"element"``, ``"edge"``, ``"face"``, ``"node_set"``,
            ``"side_set"``, ``"edge_set"``, ``"face_set"``,
            ``"element_set"``.
        time : TimeSelector, optional
            Time step selector.  May be:

            * ``None`` (default) — return values for **all** time steps.
            * A ``float`` — select the step whose time is nearest to this
              physical value.
            * An ``int`` — select by 0-based step index.
        block_id : int, optional
            For element/edge/face variables, restrict output to a single
            block with this Exodus ID.  When omitted, values from all blocks
            are concatenated along the entity axis.
        block : int, optional
            Alias for *block_id*.  If both are supplied, *block_id* takes
            precedence.
        set_id : int, optional
            For set variables (node set, side set, etc.), restrict output to
            a single set with this Exodus ID.  When omitted, values from all
            sets are concatenated.

        Returns
        -------
        ndarray of float64
            Shape depends on the combination of arguments:

            * **Global** — ``(n_steps,)`` (full history) or scalar-like
              ``(1,)`` / ``float`` (single step).
            * **Nodal / per-block / per-set with** ``time=None`` —
              ``(n_steps, n_entities)`` where *n_entities* is the node count,
              per-block element count, or set entry count.
            * **Nodal / per-block / per-set at a single time** —
              ``(n_entities,)``.
            * **Multi-block concatenation** (no *block_id*) with
              ``time=None`` — ``(n_steps, total_entities)``.
            * **Multi-block concatenation** at a single time —
              ``(total_entities,)``.

        Notes
        -----
        *block* is a historical alias for *block_id*; new code should prefer
        *block_id*.  All parameters after *name* are keyword-only.

        Raises
        ------
        ExodusLookupError
            If *name* is not found in the variable list for *on*.
        NotImplementedError
            If values for the requested entity type are not yet implemented.

        Examples
        --------
        Global variable — full history:

        >>> with ExodusFile.open("results.exo") as f:
        ...     ke = f.values("kinetic_energy", on="global")
        ...     ke.shape
        (50,)

        Nodal temperature at a specific time:

        >>> with ExodusFile.open("results.exo") as f:
        ...     temp = f.values("temperature", on="node", time=0.5)
        ...     temp.shape
        (1024,)

        Element stress in a single block, full history:

        >>> with ExodusFile.open("results.exo") as f:
        ...     sig = f.values("stress_xx", on="element", block_id=1)
        ...     sig.shape
        (50, 512)

        Element stress concatenated across all blocks at time 1.0:

        >>> with ExodusFile.open("results.exo") as f:
        ...     sig = f.values("stress_xx", on="element", time=1.0)
        ...     sig.shape
        (2048,)

        Node-set variable at a single time step:

        >>> with ExodusFile.open("results.exo") as f:
        ...     flux = f.values("heat_flux", on="node_set", set_id=1, time=0.1)
        ...     flux.shape
        (64,)
        """

        ent = entity(on)

        if ent is Entity.GLOBAL:
            return self._global_values(name, time=time)
        if ent is Entity.NODE:
            return self._node_values(name, time=time)
        if ent in {Entity.ELEMENT, Entity.EDGE, Entity.FACE}:
            return self._object_block_values(
                name, on=ent, time=time, block_id=block_id if block_id is not None else block
            )
        if ent in {
            Entity.NODE_SET,
            Entity.SIDE_SET,
            Entity.EDGE_SET,
            Entity.FACE_SET,
            Entity.ELEMENT_SET,
        }:
            return self._set_values(name, on=ent, time=time, set_id=set_id)

        raise NotImplementedError(f"values for {ent.value!r} are not implemented yet")

    def attribute_names(self, on: Entity | str, block_id: int) -> tuple[str, ...]:
        """Return the names of per-element attributes for a block.

        Parameters
        ----------
        on : Entity or str
            Block entity type (``"element_block"``, ``"edge_block"``, or
            ``"face_block"``).
        block_id : int
            Exodus block ID.

        Returns
        -------
        tuple of str
            Attribute names in file order, empty strings omitted.  Returns an
            empty tuple when no attribute name variable is present or the
            block has no attributes.

        Raises
        ------
        ExodusInvalidEntityError
            If *on* is not a block entity.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     print(f.attribute_names("element_block", 1))
        ('density', 'youngs_modulus')
        """

        ent = entity(on)
        if not ent.is_block:
            raise ExodusInvalidEntityError(f"{ent.value!r} is not a block entity")

        spec = block_spec(ent)
        if spec.attribute_names_variable is None:
            return ()

        block_index = self._block_index(ent, block_id)
        values = self._backend.variable(spec.attribute_names_variable(block_index), default=None)
        if values is None:
            return ()

        expected = self.dimension_size(
            spec.attributes_dimension(block_index) if spec.attributes_dimension is not None else "",
            default=0,
        )
        if expected == 0:
            return ()

        return tuple(name for name in _decode_name_table(values, expected_count=expected) if name)

    def attributes(self, on: Entity | str, block_id: int) -> npt.NDArray[np.float64] | None:
        """Return all per-element attributes for a block as a 2-D array.

        Parameters
        ----------
        on : Entity or str
            Block entity type (``"element_block"``, ``"edge_block"``, or
            ``"face_block"``).
        block_id : int
            Exodus block ID.

        Returns
        -------
        ndarray of float64, shape (entity_count, attribute_count) or None
            Attribute matrix where rows correspond to elements (or
            edges/faces) and columns correspond to named attributes.  Returns
            ``None`` when the block has no attribute data.

        Raises
        ------
        ExodusInvalidEntityError
            If *on* is not a block entity.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     attrs = f.attributes("element_block", 1)
        ...     attrs.shape
        (512, 2)
        """

        ent = entity(on)
        if not ent.is_block:
            raise ExodusInvalidEntityError(f"{ent.value!r} is not a block entity")

        spec = block_spec(ent)
        if spec.attributes_variable is None:
            return None

        block_index = self._block_index(ent, block_id)
        values = self._backend.variable(spec.attributes_variable(block_index), default=None)
        if values is None:
            return None

        return np.asarray(values, dtype=np.float64)

    def attribute_values(
        self, on: Entity | str, block_id: int, name: str
    ) -> npt.NDArray[np.float64]:
        """Return the values of a single named attribute for all elements in a block.

        Parameters
        ----------
        on : Entity or str
            Block entity type.
        block_id : int
            Exodus block ID.
        name : str
            Attribute name (case-insensitive).

        Returns
        -------
        ndarray of float64, shape (entity_count,)
            One value per element (or edge/face) in the block.

        Raises
        ------
        ExodusLookupError
            If *name* does not match any attribute name for the block.

        Examples
        --------
        >>> with ExodusFile.open("results.exo") as f:
        ...     rho = f.attribute_values("element_block", 1, "density")
        ...     rho.shape
        (512,)
        """

        names = self.attribute_names(on, block_id)
        requested = name.lower()

        for index, attr_name in enumerate(names):
            if attr_name.lower() == requested:
                attrs = self.attributes(on, block_id)
                if attrs is None:
                    return np.asarray([], dtype=np.float64)
                return np.asarray(attrs[:, index], dtype=np.float64)

        raise ExodusLookupError(f"attribute {name!r} not found")

    def _object_block_values(
        self, name: str, *, on: Entity, time: TimeSelector = None, block_id: int | None = None
    ) -> npt.NDArray[np.float64]:
        spec = variable_spec(on)
        variable_index = self._variable_index(on, name)

        if spec.location_entity is None:
            raise ExodusInvalidEntityError(f"{on.value!r} does not have block variable locations")

        if block_id is None:
            block_ids = self.block_ids(spec.location_entity)
            chunks = [
                self._object_block_values(name, on=on, time=time, block_id=int(current_id))
                for current_id in block_ids
            ]
            if not chunks:
                return np.asarray([], dtype=np.float64)
            return np.concatenate(chunks, axis=0 if time is not None else 1)

        block_index = self._block_index(spec.location_entity, block_id)
        variable_name = variable_value_name(on, variable_index, block_index)
        values = self._backend.variable(variable_name, default=None)

        if values is None:
            count = self.block(spec.location_entity, block_id).count
            if time is None:
                return np.zeros((len(self.times()), count), dtype=np.float64)
            return np.zeros(count, dtype=np.float64)

        array = np.asarray(values, dtype=np.float64)
        if time is None:
            return array

        selection = resolve_time(self.times(), time)
        return np.asarray(array[selection.index], dtype=np.float64)

    def _set_values(
        self, name: str, *, on: Entity, time: TimeSelector = None, set_id: int | None = None
    ) -> npt.NDArray[np.float64]:
        spec = variable_spec(on)
        variable_index = self._variable_index(on, name)

        if spec.location_entity is None:
            raise ExodusInvalidEntityError(f"{on.value!r} does not have set variable locations")

        if set_id is None:
            set_ids = self.set_ids(spec.location_entity)
            chunks = [
                self._set_values(name, on=on, time=time, set_id=int(current_id))
                for current_id in set_ids
            ]
            if not chunks:
                return np.asarray([], dtype=np.float64)
            return np.concatenate(chunks, axis=0 if time is not None else 1)

        index = self._set_index(spec.location_entity, set_id)
        variable_name = variable_value_name(on, variable_index, index)
        values = self._backend.variable(variable_name, default=None)

        set_info = self.set(spec.location_entity, set_id)
        if values is None:
            if time is None:
                return np.zeros((len(self.times()), set_info.count), dtype=np.float64)
            return np.zeros(set_info.count, dtype=np.float64)

        array = np.asarray(values, dtype=np.float64)
        if time is None:
            return array

        selection = resolve_time(self.times(), time)
        return np.asarray(array[selection.index], dtype=np.float64)

    def _block_index(self, on: Entity | str, block_id: int) -> int:
        spec = block_spec(on)
        ids = self.block_ids(spec.entity)
        matches = np.nonzero(ids == block_id)[0]
        if not len(matches):
            raise ExodusLookupError(f"{spec.entity.value} ID {block_id} not found")
        return int(matches[0]) + 1

    def _set_index(self, on: Entity | str, set_id: int) -> int:
        spec = set_spec(on)
        ids = self.set_ids(spec.entity)
        matches = np.nonzero(ids == set_id)[0]
        if not len(matches):
            raise ExodusLookupError(f"{spec.entity.value} ID {set_id} not found")
        return int(matches[0]) + 1

    def _global_values(self, name: str, *, time: TimeSelector = None) -> npt.NDArray[np.float64]:
        variable_index = self._variable_index(Entity.GLOBAL, name)
        values = np.asarray(
            self._backend.variable(VariableName.GLOBAL_VARIABLE_VALUES.value), dtype=np.float64
        )[:, variable_index - 1]

        if time is None:
            return values

        selection = resolve_time(self.times(), time)
        return np.asarray(values[selection.index], dtype=np.float64)

    def _node_values(self, name: str, *, time: TimeSelector = None) -> npt.NDArray[np.float64]:
        variable_index = self._variable_index(Entity.NODE, name)
        values = np.asarray(
            self._backend.variable(ExodusNames.node_variable(variable_index)), dtype=np.float64
        )

        if time is None:
            return values

        selection = resolve_time(self.times(), time)
        return np.asarray(values[selection.index], dtype=np.float64)

    def _element_values(
        self, name: str, *, time: TimeSelector = None, block: int | None = None
    ) -> npt.NDArray[np.float64]:
        return self._object_block_values(name, on=Entity.ELEMENT, time=time, block_id=block)

    def _variable_index(self, ent: Entity, name: str) -> int:
        cache_key = f"variable_index:{ent.value}:{name}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        names = self.variable_names(ent)
        index = _one_based_name_index(names, name)
        self._cache[cache_key] = index
        return index

    def _id_index(self, ent: Entity, id_value: int) -> int:
        ids = self.ids(ent)
        matches = np.nonzero(ids == id_value)[0]
        if not len(matches):
            raise ExodusLookupError(f"{ent.value} ID {id_value} not found")
        return int(matches[0]) + 1

    def _entity_name(self, ent: Entity, one_based_index: int) -> str:
        values = self._backend.variable(ExodusNames.names(ent), default=[])
        names = _decode_name_table(values)
        zero_based_index = one_based_index - 1

        if zero_based_index < 0 or zero_based_index >= len(names):
            return ""

        return names[zero_based_index]


def _one_based_name_index(names: tuple[str, ...], requested: str) -> int:
    for index, name in enumerate(names, start=1):
        if name == requested:
            return index

    requested_lower = requested.lower()
    for index, name in enumerate(names, start=1):
        if name.lower() == requested_lower:
            return index

    available = ", ".join(names)
    raise ExodusLookupError(f"variable {requested!r} not found; available variables: {available}")


def _four_fields(values: list[str]) -> tuple[str, str, str, str]:
    padded = values[:4] + [""] * max(0, 4 - len(values))
    return (padded[0], padded[1], padded[2], padded[3])


def _decode_name_table(values: Any, *, expected_count: int | None = None) -> tuple[str, ...]:
    """Decode an Exodus fixed-width name table.

    Handles raw NetCDF character arrays and already-decoded netCDF4 arrays.
    Legacy files may pad fixed-width strings with ASCII ``"0"`` bytes.
    """

    array = np.asarray(values)

    if (
        expected_count is not None
        and array.ndim == 1
        and len(array) == expected_count
        and array.dtype.kind in {"U", "S", "O"}
    ):
        return tuple(_strip_exodus_padding(decode_text(item)) for item in array)

    decoded = string_array(values)
    return tuple(_strip_exodus_padding(str(item)) for item in decoded)


def _strip_exodus_padding(value: str) -> str:
    text = value.rstrip(" \x00")

    # Some legacy files appear with fixed-width strings padded using ASCII "0"
    # characters.  Do not strip meaningful zeros from ordinary names like
    # "nodeset_100".  Only treat zeros as padding when the decoded field still
    # looks fixed-width.
    without_zeros = text.rstrip("0")
    zero_count = len(text) - len(without_zeros)

    if zero_count >= 2 and len(text) >= 16:
        return without_zeros

    return text


__all__ = ["ExodusFile"]
