# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Parallel/multi-file Exodus reader using Exodus global ID maps."""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from exodusii.api.file import ExodusFile
from exodusii.api.writer import ExodusWriter
from exodusii.core.entities import Entity
from exodusii.core.entities import entity
from exodusii.core.errors import ExodusConsistencyError
from exodusii.core.errors import ExodusInvalidEntityError
from exodusii.core.errors import ExodusLookupError
from exodusii.core.models import Block
from exodusii.core.models import SetInfo
from exodusii.core.names import DimensionName
from exodusii.core.names import VariableName
from exodusii.core.schema import block_spec
from exodusii.core.schema import variable_spec
from exodusii.core.schema import variable_value_name
from exodusii.core.time import TimeSelector


@dataclass(frozen=True, slots=True)
class _FileMaps:
    node_lid_to_gid: npt.NDArray[np.int64]
    element_lid_to_gid: npt.NDArray[np.int64]
    edge_lid_to_gid: npt.NDArray[np.int64]
    face_lid_to_gid: npt.NDArray[np.int64]


class ParallelExodusFile:
    """Aggregate multiple decomposed Exodus files as one logical Exodus database.

    Each component file represents a spatial partition of the mesh.
    Global identity of nodes and elements is recovered from the per-file
    ``node_num_map`` (``node_id_map``) and ``elem_num_map``
    (``element_id_map``) variables written by Nemesis decomposers such as
    ``nem_slice``.  When a map is absent a `UserWarning` is emitted and a
    sequential fallback is used, which is only correct for non-overlapping
    partitions without shared border nodes.

    The logical output ordering of nodes and elements is determined by
    sorting their unique global IDs and re-labelling them contiguously
    starting from 1, matching the layout of a serial joined Exodus file.

    Parameters
    ----------
    *files : str or Path or ExodusFile
        Component files to aggregate.  Plain path arguments are sorted
        lexicographically before opening (preserving the behaviour of the
        legacy ``parallel_exodusii_file`` API).  Already-open
        :class:`~exodusii.api.file.ExodusFile` instances are used as-is and
        are *not* closed when the aggregator is closed.

    Raises
    ------
    ValueError
        If no files are supplied.
    ExodusConsistencyError
        If spatial dimension, time values, coordinate names, or global/nodal/
        element variable names differ across component files, or if a map
        length does not match the declared per-file entity count.

    Examples
    --------
    Open two Nemesis component files and read global coordinates:

    >>> pef = ParallelExodusFile("mesh.e.2.0", "mesh.e.2.1")
    >>> coords = pef.coordinates()
    >>> coords.shape
    (1024, 3)
    >>> pef.close()

    Using the context-manager form:

    >>> with ParallelExodusFile.open("mesh.e.2.0", "mesh.e.2.1") as pef:
    ...     print(pef.node_count)
    1024
    """

    def __init__(self, *files: str | Path | ExodusFile) -> None:
        if not files:
            raise ValueError("at least one file is required")

        self._owned: list[ExodusFile] = []
        self._files: list[ExodusFile] = []

        # Legacy parallel_exodusii_file sorted filenames before opening.  This
        # matters because old callers may inspect exof.files[0] and expect a
        # deterministic component file.
        ordered_files: tuple[str | Path | ExodusFile, ...]
        if all(not isinstance(file, ExodusFile) for file in files):
            ordered_files = tuple(sorted(files, key=lambda value: str(value)))
        else:
            ordered_files = files

        for file in ordered_files:
            if isinstance(file, ExodusFile):
                self._files.append(file)
            else:
                opened = ExodusFile.open(file)
                self._owned.append(opened)
                self._files.append(opened)

        self._check_consistency()
        self._maps = self._build_file_maps()
        self._node_gid_to_index = _contiguous_gid_map(
            np.concatenate([maps.node_lid_to_gid for maps in self._maps])
        )
        self._element_gid_to_index = _contiguous_gid_map(
            np.concatenate([maps.element_lid_to_gid for maps in self._maps])
        )
        self._edge_gid_to_index = _contiguous_gid_map_or_empty(
            [maps.edge_lid_to_gid for maps in self._maps]
        )
        self._face_gid_to_index = _contiguous_gid_map_or_empty(
            [maps.face_lid_to_gid for maps in self._maps]
        )
        self._validate_global_metadata()

    @classmethod
    def open(cls, *files: str | Path | ExodusFile) -> "ParallelExodusFile":
        """Open and aggregate multiple Exodus component files.

        This is the preferred factory method.  It is equivalent to calling
        the constructor directly but reads more naturally in application code.

        Parameters
        ----------
        *files : str or Path or ExodusFile
            Paths to Exodus/Nemesis component files, or already-open
            :class:`~exodusii.api.file.ExodusFile` instances.  Path arguments
            are sorted lexicographically before opening.

        Returns
        -------
        ParallelExodusFile
            Aggregated logical database backed by the supplied component files.

        Examples
        --------
        >>> pef = ParallelExodusFile.open("mesh.e.4.0", "mesh.e.4.1",
        ...                               "mesh.e.4.2", "mesh.e.4.3")
        >>> pef.node_count
        4096
        >>> pef.close()
        """

        return cls(*files)

    @property
    def files(self) -> tuple[ExodusFile, ...]:
        """Component files backing this aggregated database.

        Returns
        -------
        tuple of ExodusFile
            Ordered tuple of :class:`~exodusii.api.file.ExodusFile` instances,
            one per component file, in the order they were opened (lexicographic
            for path arguments).
        """

        return tuple(self._files)

    @property
    def path(self) -> str:
        """Comma-separated paths of all component files.

        Returns
        -------
        str
            A single string of the form ``"part0.exo,part1.exo,..."`` built
            from the :attr:`~exodusii.api.file.ExodusFile.path` attribute of
            each component file.
        """

        return ",".join(str(file.path) for file in self._files)

    @property
    def filename(self) -> str:
        """Legacy alias for :attr:`path`.

        Returns
        -------
        str
            Same value as :attr:`path`.
        """

        return self.path

    @property
    def title(self) -> str:
        """Logical title of the aggregated database.

        Returns
        -------
        str
            The longest title string found among all component files.  When
            all component files carry the same title this is that common title.
        """

        return max((file.title for file in self._files), key=len)

    @property
    def dimension(self) -> int:
        """Spatial dimension of the mesh (2 or 3).

        Returns
        -------
        int
            Spatial dimension taken from the first component file.  All
            component files are required to share the same spatial dimension;
            a :exc:`~exodusii.core.errors.ExodusConsistencyError` is raised
            during construction if they differ.
        """

        return self._files[0].dimension

    @property
    def node_count(self) -> int:
        """Number of unique global nodes across all component files.

        Returns
        -------
        int
            When the Nemesis global dimension ``num_nodes_global`` is present
            in the first component file that value is returned directly.
            Otherwise the count is derived from the number of distinct global
            node IDs assembled from all ``node_num_map`` arrays.

        Notes
        -----
        Shared border nodes that appear in more than one component file are
        counted only once.
        """

        value = self._global_dimension(DimensionName.NUM_NODES_GLOBAL.value)
        return int(value) if value is not None else len(self._node_gid_to_index)

    @property
    def edge_count(self) -> int:
        """Number of unique global edges across all component files.

        Returns
        -------
        int
            Number of distinct global edge IDs assembled from all component
            file ``edge_num_map`` arrays.  Returns 0 when no component file
            contains edge data.

        Notes
        -----
        When ``edge_num_map`` is absent from a component file that has edges a
        `UserWarning` is emitted and a sequential fallback is used.  The
        fallback count is only correct for non-overlapping partitions.
        """
        return len(self._edge_gid_to_index)

    @property
    def face_count(self) -> int:
        """Number of unique global faces across all component files.

        Returns
        -------
        int
            Number of distinct global face IDs assembled from all component
            file ``face_num_map`` arrays.  Returns 0 when no component file
            contains face data.

        Notes
        -----
        When ``face_num_map`` is absent from a component file that has faces a
        `UserWarning` is emitted and a sequential fallback is used.  The
        fallback count is only correct for non-overlapping partitions.
        """
        return len(self._face_gid_to_index)

    @property
    def element_count(self) -> int:
        """Number of unique global elements across all component files.

        Returns
        -------
        int
            When the Nemesis global dimension ``num_elem_global`` is present in
            the first component file that value is returned directly.  Otherwise
            the count is derived from the number of distinct global element IDs
            assembled from all ``elem_num_map`` arrays.
        """

        value = self._global_dimension(DimensionName.NUM_ELEMENTS_GLOBAL.value)
        return int(value) if value is not None else len(self._element_gid_to_index)

    @property
    def element_block_count(self) -> int:
        """Number of unique element blocks across all component files.

        Returns
        -------
        int
            When the Nemesis global dimension ``num_el_blk_global`` is present
            in the first component file that value is returned directly.
            Otherwise ``len(self.element_block_ids())`` is used.
        """

        value = self._global_dimension(DimensionName.NUM_ELEMENT_BLOCKS_GLOBAL.value)
        return int(value) if value is not None else len(self.element_block_ids())

    @property
    def node_set_count(self) -> int:
        """Number of unique node sets across all component files.

        Returns
        -------
        int
            When the Nemesis global dimension ``num_ns_global`` is present in
            the first component file that value is returned directly.  Otherwise
            ``len(self.node_set_ids())`` is used.
        """

        value = self._global_dimension(DimensionName.NUM_NODE_SETS_GLOBAL.value)
        return int(value) if value is not None else len(self.node_set_ids())

    @property
    def side_set_count(self) -> int:
        """Number of unique side sets across all component files.

        Returns
        -------
        int
            When the Nemesis global dimension ``num_ss_global`` is present in
            the first component file that value is returned directly.  Otherwise
            ``len(self.side_set_ids())`` is used.
        """

        value = self._global_dimension(DimensionName.NUM_SIDE_SETS_GLOBAL.value)
        return int(value) if value is not None else len(self.side_set_ids())

    @property
    def edge_block_count(self) -> int:
        """Number of unique edge blocks across all component files.

        Returns
        -------
        int
            ``len(self.edge_block_ids())``.  Returns 0 when no component file
            contains edge-block data.
        """
        return len(self.edge_block_ids())

    @property
    def face_block_count(self) -> int:
        """Number of unique face blocks across all component files.

        Returns
        -------
        int
            ``len(self.face_block_ids())``.  Returns 0 when no component file
            contains face-block data.
        """
        return len(self.face_block_ids())

    @property
    def storage_type(self) -> str:
        """Floating-point storage type."""

        return self._files[0].storage_type

    def close(self) -> None:
        """Close component files that were opened by this instance.

        Notes
        -----
        Only files opened internally (i.e. constructed from path arguments) are
        closed.  :class:`~exodusii.api.file.ExodusFile` instances that were
        passed in directly by the caller are left open.

        This method is called automatically when the instance is used as a
        context manager (``with`` statement).
        """

        for file in self._owned:
            file.close()

    def __enter__(self) -> "ParallelExodusFile":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def info_records(self) -> tuple[str, ...]:
        """Return informational text records from the first component file.

        Returns
        -------
        tuple of str
            Each element is one informational record string, as stored in the
            ``info_records`` variable of the first component file.  Returns an
            empty tuple when no records are present.
        """

        return self._files[0].info_records()

    def qa_records(self) -> tuple[tuple[str, str, str, str], ...]:
        """Return quality-assurance records from the first component file.

        Returns
        -------
        tuple of tuple of str
            Each inner tuple contains four strings:
            ``(code_name, code_version, date, time)``.  Returns an empty tuple
            when no QA records are present.
        """

        return self._files[0].qa_records()

    def _variable_index(self, on: Entity, name: str) -> int:
        """Return one-based variable index with caching."""

        cache = getattr(self, "_variable_index_cache", None)
        if cache is None:
            cache = {}
            self._variable_index_cache = cache

        key = (on, name.lower())
        if key in cache:
            return int(cache[key])

        for index, variable_name in enumerate(self.variable_names(on), start=1):
            if variable_name == name or variable_name.lower() == name.lower():
                cache[key] = index
                return index

        raise ExodusLookupError(f"variable {name!r} not found")

    def _global_dimension(self, name: str) -> int | None:
        return self._files[0].backend.dimension(name, None)

    def _global_variable(self, name: str) -> npt.NDArray[np.int64] | None:
        values = self._files[0].variable(name, default=None)
        if values is None:
            return None
        return np.asarray(values, dtype=np.int64)

    def block_ids(self, on: Entity | str) -> npt.NDArray[np.int64]:
        """Return unique block IDs for the requested block entity type.

        Parameters
        ----------
        on : Entity or str
            Block entity type.  Must be one of ``Entity.ELEMENT_BLOCK``,
            ``Entity.EDGE_BLOCK``, or ``Entity.FACE_BLOCK`` (or the
            corresponding string values).

        Returns
        -------
        ndarray of int64
            Sorted array of unique block IDs found across all component files.

        Raises
        ------
        ExodusInvalidEntityError
            If *on* does not refer to a block entity type.
        """
        location = entity(on)
        if location is Entity.ELEMENT_BLOCK:
            return self.element_block_ids()
        if location is Entity.EDGE_BLOCK:
            return self.edge_block_ids()
        if location is Entity.FACE_BLOCK:
            return self.face_block_ids()
        raise ExodusInvalidEntityError(f"{location.value!r} is not a block entity")

    def edge_block_ids(self) -> npt.NDArray[np.int64]:
        """Return unique edge block IDs across all component files.

        Returns
        -------
        ndarray of int64
            Sorted array of unique edge block IDs.  Returns an empty array of
            shape ``(0,)`` when no component file contains edge blocks.
        """
        ids = [file.edge_block_ids() for file in self._files if file.edge_count]
        return _unique_sorted(np.concatenate(ids)) if ids else np.asarray([], dtype=np.int64)

    def face_block_ids(self) -> npt.NDArray[np.int64]:
        """Return unique face block IDs across all component files.

        Returns
        -------
        ndarray of int64
            Sorted array of unique face block IDs.  Returns an empty array of
            shape ``(0,)`` when no component file contains face blocks.
        """
        ids = [file.face_block_ids() for file in self._files if file.face_count]
        return _unique_sorted(np.concatenate(ids)) if ids else np.asarray([], dtype=np.int64)

    def block(self, on: Entity | str, block_id: int) -> Block:
        """Return the logical block descriptor for the given entity type and ID.

        The returned :class:`~exodusii.core.models.Block` reflects the global
        aggregated state: its ``count`` is the number of unique global objects
        (elements, edges, or faces) in the block, and its ``index`` is the
        one-based position of *block_id* within :meth:`block_ids`.

        Parameters
        ----------
        on : Entity or str
            Block entity type (``Entity.ELEMENT_BLOCK``, ``Entity.EDGE_BLOCK``,
            or ``Entity.FACE_BLOCK``).
        block_id : int
            Global block ID.

        Returns
        -------
        Block
            Aggregated block descriptor with global object count.

        Raises
        ------
        ExodusLookupError
            If *block_id* is not found in any component file.
        """
        location = entity(on)
        blocks = [
            file.block(location, block_id)
            for file in self._files
            if self._file_block_is_active(file, location, block_id)
        ]
        if not blocks:
            blocks = [
                file.block(location, block_id)
                for file in self._files
                if _contains(file.block_ids(location), block_id)
            ]
        if not blocks:
            raise ExodusLookupError(f"{location.value} ID {block_id} not found")

        first = blocks[0]
        gids = self._block_object_gids(location, block_id)

        return Block(
            id=block_id,
            index=_one_based_index(self.block_ids(location), block_id),
            entity=location,
            element_type=first.element_type,
            count=len(gids),
            nodes_per_entity=first.nodes_per_entity,
            edges_per_entity=first.edges_per_entity,
            faces_per_entity=first.faces_per_entity,
            attributes=first.attributes,
            name=first.name,
        )

    def block_connectivity(
        self, on: Entity | str, block_id: int, *, zero_based: bool = False, labels: bool = False
    ) -> npt.NDArray[np.int64]:
        """Return the nodal connectivity for a block assembled from all component files.

        Local node IDs from each component file are translated to global node
        IDs via the per-file ``node_num_map``, then mapped to contiguous
        one-based indices that correspond to the row ordering of
        :meth:`coordinates`.

        Parameters
        ----------
        on : Entity or str
            Block entity type (``Entity.ELEMENT_BLOCK``, ``Entity.EDGE_BLOCK``,
            or ``Entity.FACE_BLOCK``).
        block_id : int
            Global block ID.
        zero_based : bool, optional
            If ``True``, return zero-based node indices.  Ignored when
            *labels* is ``True``.  Default is ``False``.
        labels : bool, optional
            If ``True``, return raw global node IDs (labels) instead of
            contiguous positional indices.  Default is ``False``.

        Returns
        -------
        ndarray of int64, shape (n_objects, nodes_per_entity)
            Connectivity table where rows correspond to global objects ordered
            by ascending global object ID and columns are node indices (or
            labels when *labels* is ``True``).
        """
        location = entity(on)
        spec = block_spec(location)

        block_gids = self._block_object_gids(location, block_id)
        block_gid_to_row = {gid: row for row, gid in enumerate(block_gids)}
        block = self.block(location, block_id)
        connectivity = np.zeros((len(block_gids), block.nodes_per_entity), dtype=np.int64)

        for file_index, file in enumerate(self._files):
            if not self._file_block_is_active(file, location, block_id):
                continue
            local_conn = file.block_connectivity(location, block_id, zero_based=False)
            local_object_gids = self._block_object_gids_for_file(file_index, location, block_id)
            node_map = self._maps[file_index].node_lid_to_gid

            for local_row, object_gid in enumerate(local_object_gids):
                output_row = block_gid_to_row[int(object_gid)]
                local_node_ids = local_conn[local_row]

                if labels:
                    connectivity[output_row] = [
                        int(node_map[int(local_node_id) - 1]) for local_node_id in local_node_ids
                    ]
                else:
                    connectivity[output_row] = [
                        self._node_gid_to_index[int(node_map[int(local_node_id) - 1])]
                        for local_node_id in local_node_ids
                    ]

        return connectivity - 1 if zero_based and not labels else connectivity

    def edge_connectivity(
        self, block_id: int, *, zero_based: bool = False, labels: bool = False
    ) -> npt.NDArray[np.int64]:
        """Return the nodal connectivity for an edge block.

        Delegates to :meth:`block_connectivity` with ``on=Entity.EDGE_BLOCK``.

        Parameters
        ----------
        block_id : int
            Global edge block ID.
        zero_based : bool, optional
            If ``True``, return zero-based node indices.  Ignored when
            *labels* is ``True``.  Default is ``False``.
        labels : bool, optional
            If ``True``, return raw global node IDs instead of contiguous
            positional indices.  Default is ``False``.

        Returns
        -------
        ndarray of int64, shape (n_edges, nodes_per_edge)
            Connectivity table for the requested edge block.
        """
        return self.block_connectivity(
            Entity.EDGE_BLOCK, block_id, zero_based=zero_based, labels=labels
        )

    def face_connectivity(
        self, block_id: int, *, zero_based: bool = False, labels: bool = False
    ) -> npt.NDArray[np.int64]:
        """Return the nodal connectivity for a face block.

        Delegates to :meth:`block_connectivity` with ``on=Entity.FACE_BLOCK``.

        Parameters
        ----------
        block_id : int
            Global face block ID.
        zero_based : bool, optional
            If ``True``, return zero-based node indices.  Ignored when
            *labels* is ``True``.  Default is ``False``.
        labels : bool, optional
            If ``True``, return raw global node IDs instead of contiguous
            positional indices.  Default is ``False``.

        Returns
        -------
        ndarray of int64, shape (n_faces, nodes_per_face)
            Connectivity table for the requested face block.
        """
        return self.block_connectivity(
            Entity.FACE_BLOCK, block_id, zero_based=zero_based, labels=labels
        )

    def edge_block(self, block_id: int) -> Block:
        """Return the logical block descriptor for an edge block.

        Delegates to :meth:`block` with ``on=Entity.EDGE_BLOCK``.

        Parameters
        ----------
        block_id : int
            Global edge block ID.

        Returns
        -------
        Block
            Aggregated edge block descriptor with global edge count.

        Raises
        ------
        ExodusLookupError
            If *block_id* is not found in any component file.
        """
        return self.block(Entity.EDGE_BLOCK, block_id)

    def face_block(self, block_id: int) -> Block:
        """Return the logical block descriptor for a face block.

        Delegates to :meth:`block` with ``on=Entity.FACE_BLOCK``.

        Parameters
        ----------
        block_id : int
            Global face block ID.

        Returns
        -------
        Block
            Aggregated face block descriptor with global face count.

        Raises
        ------
        ExodusLookupError
            If *block_id* is not found in any component file.
        """
        return self.block(Entity.FACE_BLOCK, block_id)

    def times(self) -> npt.NDArray[np.float64]:
        """Return the simulation time values shared by all component files.

        Returns
        -------
        ndarray of float64, shape (n_steps,)
            Array of simulation times, one entry per time step.  All component
            files are required to carry identical time sequences; a
            :exc:`~exodusii.core.errors.ExodusConsistencyError` is raised
            during construction when they differ.

        Examples
        --------
        >>> with ParallelExodusFile.open("mesh.e.2.0", "mesh.e.2.1") as pef:
        ...     t = pef.times()
        ...     print(t)
        [0.   0.1  0.2  0.3]
        """

        return self._files[0].times()

    def coordinate_names(self) -> npt.NDArray[np.str_]:
        """Return the coordinate axis names.

        Returns
        -------
        ndarray of str_, shape (dimension,)
            Axis labels such as ``["x", "y", "z"]`` or
            ``["coordx", "coordy", "coordz"]``.

        Examples
        --------
        >>> with ParallelExodusFile.open("mesh.e.2.0", "mesh.e.2.1") as pef:
        ...     pef.coordinate_names()
        array(['x', 'y', 'z'], dtype='<U1')
        """

        return self._files[0].coordinate_names()

    def coordinates(
        self, *, time: TimeSelector = None, displaced: bool = False
    ) -> npt.NDArray[np.float64]:
        """Return nodal coordinates ordered by contiguous global node index.

        Local node positions from each component file are placed into the
        output array using the per-file ``node_num_map`` to resolve global
        IDs, then the global-to-contiguous-index map to determine output rows.

        Parameters
        ----------
        time : TimeSelector, optional
            Time at which to evaluate displaced coordinates.  Only relevant
            when *displaced* is ``True``.  ``None`` returns reference
            (undeformed) coordinates.
        displaced : bool, optional
            If ``True``, add displacement variable values to the reference
            coordinates to return the deformed configuration.  Requires
            displacement variables to be present.  Default is ``False``.

        Returns
        -------
        ndarray of float64, shape (node_count, dimension)
            Nodal coordinate array where row *i* corresponds to the node with
            contiguous global index *i+1*.

        Examples
        --------
        >>> with ParallelExodusFile.open("mesh.e.2.0", "mesh.e.2.1") as pef:
        ...     xyz = pef.coordinates()
        ...     xyz.shape
        (1024, 3)
        """

        coords = np.zeros((self.node_count, self.dimension), dtype=np.float64)

        for file_index, file in enumerate(self._files):
            local_coords = file.coordinates(time=time, displaced=displaced)
            node_map = self._maps[file_index].node_lid_to_gid

            for local_index, gid in enumerate(node_map, start=1):
                global_index = self._node_gid_to_index[int(gid)] - 1
                coords[global_index] = local_coords[local_index - 1]

        return coords

    def displacement_variable_names(self) -> tuple[str, ...]:
        """Return the names of recognized displacement variables.

        Returns
        -------
        tuple of str
            Variable names (e.g. ``("disp_x", "disp_y", "disp_z")``) that the
            first component file identifies as displacement variables.  Returns
            an empty tuple when no displacement variables are present.
        """

        return self._files[0].displacement_variable_names()

    def displacements(self, *, time: TimeSelector = None) -> npt.NDArray[np.float64]:
        """Return aggregate nodal displacement vectors.

        Parameters
        ----------
        time : TimeSelector, optional
            Time selector.  ``None`` returns the full time history as a
            3-D array of shape ``(n_steps, node_count, dimension)``.
            A specific time value or step index returns a 2-D array of shape
            ``(node_count, dimension)``.

        Returns
        -------
        ndarray of float64, shape (node_count, dimension) or (n_steps, node_count, dimension)
            Displacement vectors assembled from nodal displacement variables.
            Returns a zero array when no displacement variables are found.
        """

        names = self.displacement_variable_names()
        if not names:
            return np.zeros((self.node_count, self.dimension), dtype=np.float64)

        return np.column_stack([self.values(name, on=Entity.NODE, time=time) for name in names])

    def variable_names(self, on: Entity | str) -> tuple[str, ...]:
        """Return the result variable names defined for an entity type.

        Parameters
        ----------
        on : Entity or str
            Entity type whose variable names are requested (e.g.
            ``Entity.NODE``, ``Entity.ELEMENT``, ``"node"``, ``"element"``).

        Returns
        -------
        tuple of str
            Variable names in the order they are stored in the first component
            file.  All component files are required to define the same
            global, nodal, and element variable names; a
            :exc:`~exodusii.core.errors.ExodusConsistencyError` is raised
            during construction when they differ.

        Examples
        --------
        >>> with ParallelExodusFile.open("mesh.e.2.0", "mesh.e.2.1") as pef:
        ...     pef.variable_names("node")
        ('disp_x', 'disp_y', 'disp_z', 'velocity_x', 'velocity_y')
        """

        return self._files[0].variable_names(on)

    def ids(self, on: Entity | str) -> npt.NDArray[np.int64]:
        """Return logical global IDs for nodes, elements, blocks, or sets.

        Parameters
        ----------
        on : Entity or str
            Entity type.  Supported values: ``Entity.NODE``,
            ``Entity.ELEMENT``, ``Entity.ELEMENT_BLOCK``,
            ``Entity.NODE_SET``, ``Entity.SIDE_SET``, ``Entity.EDGE``,
            ``Entity.FACE``.

        Returns
        -------
        ndarray of int64
            Sorted array of global IDs.  For nodes and elements these are the
            raw global IDs from the ``node_num_map`` / ``elem_num_map``
            variables.  For blocks and sets these are the unique block/set IDs
            across all component files.

        Raises
        ------
        ExodusInvalidEntityError
            If *on* is an entity type for which parallel ID assembly is not
            implemented.
        """

        location = entity(on)

        if location is Entity.NODE:
            return np.asarray(sorted(self._node_gid_to_index), dtype=np.int64)
        if location is Entity.ELEMENT:
            return np.asarray(sorted(self._element_gid_to_index), dtype=np.int64)
        if location is Entity.ELEMENT_BLOCK:
            return self.element_block_ids()
        if location is Entity.NODE_SET:
            return self.node_set_ids()
        if location is Entity.SIDE_SET:
            return self.side_set_ids()
        if location is Entity.EDGE:
            return np.asarray(sorted(self._edge_gid_to_index), dtype=np.int64)
        if location is Entity.FACE:
            return np.asarray(sorted(self._face_gid_to_index), dtype=np.int64)

        raise ExodusInvalidEntityError(f"parallel IDs for {location.value!r} are not implemented")

    def element_block_ids(self) -> npt.NDArray[np.int64]:
        """Return unique element block IDs across all component files.

        Returns
        -------
        ndarray of int64
            Sorted array of unique element block IDs.  When the Nemesis global
            variable ``eb_prop1_global`` is present it is returned directly;
            otherwise block IDs are collected from all component files and
            deduplicated.

        Examples
        --------
        >>> with ParallelExodusFile.open("mesh.e.2.0", "mesh.e.2.1") as pef:
        ...     pef.element_block_ids()
        array([1, 2, 3], dtype=int64)
        """

        global_ids = self._global_variable(VariableName.ELEMENT_BLOCK_IDS_GLOBAL.value)
        if global_ids is not None:
            return global_ids

        ids = [file.element_block_ids() for file in self._files if file.element_block_count]
        return _unique_sorted(np.concatenate(ids)) if ids else np.asarray([], dtype=np.int64)

    def node_set_ids(self) -> npt.NDArray[np.int64]:
        """Return unique node set IDs across all component files.

        Returns
        -------
        ndarray of int64
            Sorted array of unique node set IDs.  When the Nemesis global
            variable ``ns_prop1_global`` is present it is returned directly.

        Examples
        --------
        >>> with ParallelExodusFile.open("mesh.e.2.0", "mesh.e.2.1") as pef:
        ...     pef.node_set_ids()
        array([10, 20], dtype=int64)
        """

        global_ids = self._global_variable(VariableName.NODE_SET_IDS_GLOBAL.value)
        if global_ids is not None:
            return global_ids

        ids = [file.node_set_ids() for file in self._files if file.node_set_count]
        return _unique_sorted(np.concatenate(ids)) if ids else np.asarray([], dtype=np.int64)

    def side_set_ids(self) -> npt.NDArray[np.int64]:
        """Return unique side set IDs across all component files.

        Returns
        -------
        ndarray of int64
            Sorted array of unique side set IDs.  When the Nemesis global
            variable ``ss_prop1_global`` is present it is returned directly.

        Examples
        --------
        >>> with ParallelExodusFile.open("mesh.e.2.0", "mesh.e.2.1") as pef:
        ...     pef.side_set_ids()
        array([100], dtype=int64)
        """

        global_ids = self._global_variable(VariableName.SIDE_SET_IDS_GLOBAL.value)
        if global_ids is not None:
            return global_ids

        ids = [file.side_set_ids() for file in self._files if file.side_set_count]
        return _unique_sorted(np.concatenate(ids)) if ids else np.asarray([], dtype=np.int64)

    def element_block(self, block_id: int) -> Block:
        """Return the logical block descriptor for an element block.

        Delegates to :meth:`block` with ``on=Entity.ELEMENT_BLOCK``.

        Parameters
        ----------
        block_id : int
            Global element block ID.

        Returns
        -------
        Block
            Aggregated element block descriptor with global element count.

        Raises
        ------
        ExodusLookupError
            If *block_id* is not found in any component file.
        """
        return self.block(Entity.ELEMENT_BLOCK, block_id)

    def element_connectivity(
        self, block_id: int, *, zero_based: bool = False, labels: bool = False
    ) -> npt.NDArray[np.int64]:
        """Return the nodal connectivity for an element block.

        Delegates to :meth:`block_connectivity` with
        ``on=Entity.ELEMENT_BLOCK``.

        Parameters
        ----------
        block_id : int
            Global element block ID.
        zero_based : bool, optional
            If ``True``, return zero-based node indices.  Ignored when
            *labels* is ``True``.  Default is ``False``.
        labels : bool, optional
            If ``True``, return raw global node IDs instead of contiguous
            positional indices.  Default is ``False``.

        Returns
        -------
        ndarray of int64, shape (n_elements, nodes_per_element)
            Connectivity table for the requested element block.
        """
        return self.block_connectivity(
            Entity.ELEMENT_BLOCK, block_id, zero_based=zero_based, labels=labels
        )

    def node_set(self, set_id: int) -> SetInfo:
        """Return the aggregated node set using global node labels.

        Node IDs from each component file are translated to global IDs via the
        per-file ``node_num_map``.  Global IDs that appear in more than one
        component file (shared border nodes) are deduplicated; the distribution
        factor for the first occurrence of each global node ID is retained.

        Parameters
        ----------
        set_id : int
            Global node set ID.

        Returns
        -------
        SetInfo
            Aggregated node set.  ``SetInfo.entries`` contains the sorted
            unique global node IDs.  ``SetInfo.distribution_values`` is
            ``None`` when no component file stores distribution factors.

        Notes
        -----
        Shared border nodes that appear in multiple component files are counted
        and reported only once.

        Raises
        ------
        ExodusLookupError
            If *set_id* is not found in any component file.

        Examples
        --------
        >>> with ParallelExodusFile.open("mesh.e.2.0", "mesh.e.2.1") as pef:
        ...     ns = pef.node_set(10)
        ...     ns.count
        64
        """

        node_gids: list[int] = []
        factors_by_gid: dict[int, float] = {}
        name = ""

        for file_index, file in enumerate(self._files):
            if not self._file_set_is_active(file, Entity.NODE_SET, set_id):
                continue

            node_set = file.node_set(set_id)
            name = name or node_set.name
            if node_set.nodes is None:
                continue

            node_map = self._maps[file_index].node_lid_to_gid
            for position, local_node_id in enumerate(node_set.nodes):
                gid = int(node_map[int(local_node_id) - 1])
                node_gids.append(gid)

                if node_set.dist_facts is not None:
                    factors_by_gid[gid] = float(node_set.dist_facts[position])

        unique_gids = sorted(set(node_gids))
        if not unique_gids:
            raise ExodusLookupError(f"node_set ID {set_id} not found")

        nodes = np.asarray(unique_gids, dtype=np.int64)
        dist_facts = (
            np.asarray([factors_by_gid[gid] for gid in unique_gids], dtype=np.float64)
            if factors_by_gid
            else None
        )

        return SetInfo(
            id=set_id,
            index=_one_based_index(self.node_set_ids(), set_id),
            entity=Entity.NODE_SET,
            count=len(nodes),
            distribution_factors=0 if dist_facts is None else len(dist_facts),
            name=name,
            entries=nodes,
            distribution_values=dist_facts,
        )

    def side_set(self, set_id: int) -> SetInfo:
        """Return the aggregated side set using global element labels.

        Element IDs from each component file are translated to global IDs via
        the per-file ``elem_num_map``.  Duplicate ``(element_gid, side)`` pairs
        that arise from border elements shared across processor boundaries are
        deduplicated; the distribution factor for the first occurrence of each
        unique pair is retained.

        Parameters
        ----------
        set_id : int
            Global side set ID.

        Returns
        -------
        SetInfo
            Aggregated side set.  ``SetInfo.entries`` contains the global
            element IDs and ``SetInfo.extra_entries`` the corresponding side
            ordinals.  ``SetInfo.distribution_values`` is ``None`` when no
            component file stores distribution factors.

        Notes
        -----
        Duplicate ``(element_gid, side)`` pairs that occur when a boundary
        element is present in more than one component file are removed,
        keeping the entry from the first component file that contains it.
        This is consistent with the deduplication performed by :meth:`node_set`
        for shared border nodes.

        Raises
        ------
        ExodusLookupError
            If *set_id* is not found in any component file.

        Examples
        --------
        >>> with ParallelExodusFile.open("mesh.e.2.0", "mesh.e.2.1") as pef:
        ...     ss = pef.side_set(100)
        ...     ss.count
        32
        """

        # Use an ordered dict to deduplicate (elem_gid, side) pairs while
        # preserving encounter order and associating distribution factors.
        seen: dict[tuple[int, int], float | None] = {}
        has_factors = False
        name = ""

        for file_index, file in enumerate(self._files):
            if not self._file_set_is_active(file, Entity.SIDE_SET, set_id):
                continue

            side_set = file.side_set(set_id)
            name = name or side_set.name
            if side_set.elems is None or side_set.sides is None:
                continue

            element_map = self._maps[file_index].element_lid_to_gid

            for position, local_element_id in enumerate(side_set.elems):
                gid = int(element_map[int(local_element_id) - 1])
                side = int(side_set.sides[position])
                key = (gid, side)
                if key not in seen:
                    df: float | None = None
                    if side_set.dist_facts is not None:
                        has_factors = True
                        df = float(side_set.dist_facts[position])
                    seen[key] = df

        if not seen:
            raise ExodusLookupError(f"side_set ID {set_id} not found")

        elements = np.asarray([k[0] for k in seen], dtype=np.int64)
        sides = np.asarray([k[1] for k in seen], dtype=np.int64)
        dist_facts = np.asarray(list(seen.values()), dtype=np.float64) if has_factors else None

        return SetInfo(
            id=set_id,
            index=_one_based_index(self.side_set_ids(), set_id),
            entity=Entity.SIDE_SET,
            count=len(elements),
            distribution_factors=0 if dist_facts is None else len(dist_facts),
            name=name,
            entries=elements,
            extra_entries=sides,
            distribution_values=dist_facts,
        )

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
        """Return aggregate result variable values.

        Parameters
        ----------
        name : str
            Result variable name.
        on : Entity or str
            Variable location (e.g. ``Entity.NODE``, ``Entity.ELEMENT``,
            ``"node"``, ``"element"``).
        time : TimeSelector, optional
            Time selector.  ``None`` returns the full time history.  An
            integer is interpreted as a one-based time-step index.  A float
            is matched to the nearest available time value.
        block : int, optional
            Backward-compatible alias for *block_id*.  Ignored when *block_id*
            is also supplied.
        block_id : int, optional
            Block ID for element, edge, or face variables.  When ``None`` and
            the location is a block-based entity, values are concatenated
            across all blocks in block-ID order.
        set_id : int, optional
            Set ID for node-set, side-set, edge-set, face-set, or element-set
            variables.  When ``None`` values are concatenated across all sets.

        Returns
        -------
        ndarray of float64
            For a specific time step: shape ``(n_objects,)`` for nodal /
            element / set variables, or ``(n_time_steps, n_objects)`` when
            *time* is ``None``.

        Examples
        --------
        Read all nodal ``disp_x`` values at the final time step:

        >>> with ParallelExodusFile.open("mesh.e.2.0", "mesh.e.2.1") as pef:
        ...     u = pef.values("disp_x", on="node", time=-1)
        ...     u.shape
        (1024,)

        Read the complete time history of element stress in block 1:

        >>> with ParallelExodusFile.open("mesh.e.2.0", "mesh.e.2.1") as pef:
        ...     s = pef.values("stress_xx", on="element", block_id=1)
        ...     s.shape
        (50, 256)
        """

        location = entity(on)
        requested_block_id = block_id if block_id is not None else block

        if location is Entity.GLOBAL:
            return self._files[0].values(name, on=location, time=time)

        if location is Entity.NODE:
            return self._node_values(name, time=time)

        if location in {Entity.ELEMENT, Entity.EDGE, Entity.FACE}:
            block_location = _variable_block_location(location)

            if requested_block_id is None:
                chunks = [
                    self.values(name, on=location, time=time, block_id=int(current_block_id))
                    for current_block_id in self.block_ids(block_location)
                ]
                if not chunks:
                    return np.asarray([], dtype=np.float64)
                return np.concatenate(chunks, axis=0 if time is not None else 1)

            return self._block_object_values(
                name, on=location, block=int(requested_block_id), time=time
            )

        if location in {
            Entity.NODE_SET,
            Entity.SIDE_SET,
            Entity.EDGE_SET,
            Entity.FACE_SET,
            Entity.ELEMENT_SET,
        }:
            return self._set_values(name, on=location, set_id=set_id, time=time)

        raise ExodusInvalidEntityError(f"values for {location.value!r} are not implemented")

    def _global_set_distribution_factor_count(self, on: Entity, set_id: int) -> int | None:
        """Return global distribution-factor count for a set if Nemesis metadata exists."""

        variable_names = {
            Entity.NODE_SET: VariableName.NODE_SET_DF_COUNT_GLOBAL.value,
            Entity.SIDE_SET: VariableName.SIDE_SET_DF_COUNT_GLOBAL.value,
            Entity.EDGE_SET: "es_df_cnt_global",
            Entity.FACE_SET: "fs_df_cnt_global",
            Entity.ELEMENT_SET: "els_df_cnt_global",
        }

        variable_name = variable_names.get(on)
        if variable_name is None:
            return None

        counts = self._global_variable(variable_name)
        if counts is None:
            return None

        set_ids = self.set_ids(on)
        index = _one_based_index(set_ids, set_id) - 1
        return int(counts[index])

    def _node_values(self, name: str, *, time: TimeSelector) -> npt.NDArray[np.float64]:
        """Fast aggregate nodal variable values without serial name decoding."""

        variable_index = self._variable_index(Entity.NODE, name)
        variable_name = variable_value_name(Entity.NODE, variable_index)

        first_values = self._files[0].variable(variable_name)
        first_array = np.asarray(first_values, dtype=np.float64)

        output: npt.NDArray[np.float64]
        if time is None:
            output = np.zeros((first_array.shape[0], self.node_count), dtype=np.float64)
            selection_index = None
        else:
            output = np.zeros(self.node_count, dtype=np.float64)
            selection_index = self._resolve_time_index(time)

        for file_index, file in enumerate(self._files):
            local_values = np.asarray(file.variable(variable_name), dtype=np.float64)
            node_map = self._maps[file_index].node_lid_to_gid

            for local_index, gid in enumerate(node_map):
                output_index = self._node_gid_to_index[int(gid)] - 1
                if selection_index is None:
                    output[:, output_index] = local_values[:, local_index]
                else:
                    output[output_index] = local_values[selection_index, local_index]

        return output

    def _set_values(
        self, name: str, *, on: Entity, set_id: int | None, time: TimeSelector
    ) -> npt.NDArray[np.float64]:
        if set_id is None:
            chunks = [
                self._set_values(name, on=on, set_id=int(current_id), time=time)
                for current_id in self.set_ids(on)
            ]
            if not chunks:
                return np.asarray([], dtype=np.float64)
            return np.concatenate(chunks, axis=0 if time is not None else 1)

        set_info = self.set(on, set_id)
        keys = _set_value_keys(set_info, on)

        output: npt.NDArray[np.float64]
        if time is None:
            first_chunk = next(
                file.values(name, on=on, set_id=set_id, time=time)
                for file in self._files
                if self._file_set_is_active(file, on, set_id)
            )
            output = np.zeros((first_chunk.shape[0], len(keys)), dtype=np.float64)
        else:
            output = np.zeros(len(keys), dtype=np.float64)

        key_to_row = {key: row for row, key in enumerate(keys)}
        for file_index, file in enumerate(self._files):
            if not self._file_set_is_active(file, on, set_id):
                continue

            local_set = file.set(on, set_id)
            local_labels = self._local_set_global_labels(file_index, local_set, on)
            local_keys = _set_value_keys_from_local(local_labels, local_set, on)
            local_values = file.values(name, on=on, set_id=set_id, time=time)

            for local_index, key in enumerate(local_keys):
                output_index = key_to_row[key]
                if time is None:
                    output[:, output_index] = local_values[:, local_index]
                else:
                    output[output_index] = local_values[local_index]

        return output

    def set_ids(self, on: Entity | str) -> npt.NDArray[np.int64]:
        """Return unique set IDs for the requested set entity type.

        Parameters
        ----------
        on : Entity or str
            Set entity type.  Must be one of ``Entity.NODE_SET``,
            ``Entity.SIDE_SET``, ``Entity.EDGE_SET``, ``Entity.FACE_SET``,
            or ``Entity.ELEMENT_SET`` (or the corresponding string values).

        Returns
        -------
        ndarray of int64
            Sorted array of unique set IDs found across all component files.

        Raises
        ------
        ExodusInvalidEntityError
            If *on* does not refer to a set entity type.
        """
        location = entity(on)

        if location is Entity.NODE_SET:
            return self.node_set_ids()
        if location is Entity.SIDE_SET:
            return self.side_set_ids()
        if location is Entity.EDGE_SET:
            return self.edge_set_ids()
        if location is Entity.FACE_SET:
            return self.face_set_ids()
        if location is Entity.ELEMENT_SET:
            return self.element_set_ids()

        raise ExodusInvalidEntityError(f"{location.value!r} is not a set entity")

    def edge_set_ids(self) -> npt.NDArray[np.int64]:
        """Return unique edge set IDs across all component files.

        Returns
        -------
        ndarray of int64
            Sorted array of unique edge set IDs.  Returns an empty array of
            shape ``(0,)`` when no component file contains edge sets.
        """
        ids = [file.edge_set_ids() for file in self._files if len(file.edge_set_ids())]
        return _unique_sorted(np.concatenate(ids)) if ids else np.asarray([], dtype=np.int64)

    def face_set_ids(self) -> npt.NDArray[np.int64]:
        """Return unique face set IDs across all component files.

        Returns
        -------
        ndarray of int64
            Sorted array of unique face set IDs.  Returns an empty array of
            shape ``(0,)`` when no component file contains face sets.
        """
        ids = [file.face_set_ids() for file in self._files if len(file.face_set_ids())]
        return _unique_sorted(np.concatenate(ids)) if ids else np.asarray([], dtype=np.int64)

    def element_set_ids(self) -> npt.NDArray[np.int64]:
        """Return unique element set IDs across all component files.

        Returns
        -------
        ndarray of int64
            Sorted array of unique element set IDs.  Returns an empty array of
            shape ``(0,)`` when no component file contains element sets.
        """
        ids = [file.element_set_ids() for file in self._files if len(file.element_set_ids())]
        return _unique_sorted(np.concatenate(ids)) if ids else np.asarray([], dtype=np.int64)

    def edge_set(self, set_id: int) -> SetInfo:
        """Return the aggregated edge set for the given ID.

        Parameters
        ----------
        set_id : int
            Global edge set ID.

        Returns
        -------
        SetInfo
            Aggregated edge set with global edge IDs in ``SetInfo.entries``.

        Raises
        ------
        ExodusLookupError
            If *set_id* is not found in any component file.
        """
        return self._object_set(Entity.EDGE_SET, set_id)

    def face_set(self, set_id: int) -> SetInfo:
        """Return the aggregated face set for the given ID.

        Parameters
        ----------
        set_id : int
            Global face set ID.

        Returns
        -------
        SetInfo
            Aggregated face set with global face IDs in ``SetInfo.entries``.

        Raises
        ------
        ExodusLookupError
            If *set_id* is not found in any component file.
        """
        return self._object_set(Entity.FACE_SET, set_id)

    def element_set(self, set_id: int) -> SetInfo:
        """Return the aggregated element set for the given ID.

        Parameters
        ----------
        set_id : int
            Global element set ID.

        Returns
        -------
        SetInfo
            Aggregated element set with global element IDs in
            ``SetInfo.entries``.

        Raises
        ------
        ExodusLookupError
            If *set_id* is not found in any component file.
        """
        return self._object_set(Entity.ELEMENT_SET, set_id)

    def element_edge_connectivity(
        self, block_id: int, *, zero_based: bool = False, labels: bool = False
    ) -> npt.NDArray[np.int64] | None:
        """Return aggregate element-to-edge connectivity."""

        return self._element_subobject_connectivity(
            block_id, subobject=Entity.EDGE, zero_based=zero_based, labels=labels
        )

    def element_face_connectivity(
        self, block_id: int, *, zero_based: bool = False, labels: bool = False
    ) -> npt.NDArray[np.int64] | None:
        """Return aggregate element-to-face connectivity."""

        return self._element_subobject_connectivity(
            block_id, subobject=Entity.FACE, zero_based=zero_based, labels=labels
        )

    def _element_subobject_connectivity(
        self, block_id: int, *, subobject: Entity, zero_based: bool, labels: bool
    ) -> npt.NDArray[np.int64] | None:
        block_gids = self._block_object_gids(Entity.ELEMENT_BLOCK, block_id)
        block_gid_to_row = {gid: row for row, gid in enumerate(block_gids)}

        local_arrays: list[npt.NDArray[np.int64]] = []
        local_gids_by_file: list[tuple[int, npt.NDArray[np.int64]]] = []

        for file_index, file in enumerate(self._files):
            if not self._file_block_is_active(file, Entity.ELEMENT_BLOCK, block_id):
                continue

            if subobject is Entity.EDGE:
                local = file.element_edge_connectivity(block_id, zero_based=False)
                object_map = self._maps[file_index].edge_lid_to_gid
            elif subobject is Entity.FACE:
                local = file.element_face_connectivity(block_id, zero_based=False)
                object_map = self._maps[file_index].face_lid_to_gid
            else:
                raise ExodusInvalidEntityError(f"{subobject.value!r} is not supported")

            if local is None:
                continue

            local_arrays.append(local)
            local_gids_by_file.append(
                (
                    file_index,
                    self._block_object_gids_for_file(file_index, Entity.ELEMENT_BLOCK, block_id),
                )
            )

        if not local_arrays:
            return None

        width = local_arrays[0].shape[1]
        output = np.zeros((len(block_gids), width), dtype=np.int64)

        for local, (file_index, element_gids) in zip(local_arrays, local_gids_by_file, strict=True):
            if subobject is Entity.EDGE:
                object_map = self._maps[file_index].edge_lid_to_gid
                contiguous = self._edge_gid_to_index
            else:
                object_map = self._maps[file_index].face_lid_to_gid
                contiguous = self._face_gid_to_index

            for local_row, element_gid in enumerate(element_gids):
                output_row = block_gid_to_row[int(element_gid)]
                local_object_ids = local[local_row]

                if labels:
                    output[output_row] = [
                        int(object_map[int(local_id) - 1]) for local_id in local_object_ids
                    ]
                else:
                    output[output_row] = [
                        contiguous[int(object_map[int(local_id) - 1])]
                        for local_id in local_object_ids
                    ]

        return output - 1 if zero_based and not labels else output

    def _object_set(self, on: Entity, set_id: int) -> SetInfo:
        all_labels: list[int] = []
        all_extras: list[int] = []
        all_factors: list[float] = []
        has_extras = False
        has_factors = False
        name = ""

        for file_index, file in enumerate(self._files):
            if not self._file_set_is_active(file, on, set_id):
                continue

            local_set = file.set(on, set_id)
            name = name or local_set.name
            local_labels = self._local_set_global_labels(file_index, local_set, on)

            for position, label in enumerate(local_labels):
                all_labels.append(int(label))

                if local_set.extra_entries is not None:
                    has_extras = True
                    all_extras.append(int(local_set.extra_entries[position]))
                if local_set.dist_facts is not None:
                    has_factors = True
                    all_factors.append(float(local_set.dist_facts[position]))

        if not all_labels:
            raise ExodusLookupError(f"{on.value} ID {set_id} not found")

        # Do NOT deduplicate: element/edge/face sets may legitimately contain
        # duplicate entries (the same object referenced multiple times).  The
        # raw Exodus API (ex_get_set) returns entries without deduplication.
        entries = np.asarray(all_labels, dtype=np.int64)
        extra = np.asarray(all_extras, dtype=np.int64) if has_extras else None
        dist = np.asarray(all_factors, dtype=np.float64) if has_factors else None

        return SetInfo(
            id=set_id,
            index=_one_based_index(self.set_ids(on), set_id),
            entity=on,
            count=len(entries),
            distribution_factors=0 if dist is None else len(dist),
            name=name,
            entries=entries,
            extra_entries=extra,
            distribution_values=dist,
        )

    def _distribution_factors_for_write(
        self, on: Entity, set_info: SetInfo
    ) -> npt.NDArray[np.float64] | None:
        """Return distribution factors to write to a joined serial file.

        User-facing set reads should expose factors found in component files.
        For joined-file writing, Nemesis global distribution-factor counts are
        authoritative when present. If the global count is zero, suppress local
        factors so we do not create ``num_df_*`` dimensions absent from the
        expected joined file.
        """

        global_count = self._global_set_distribution_factor_count(on, set_info.id)
        if global_count == 0:
            return None

        return set_info.dist_facts if set_info.distribution_factors else None

    def _local_set_global_labels(
        self, file_index: int, local_set: SetInfo, on: Entity
    ) -> npt.NDArray[np.int64]:
        if local_set.entries is None:
            return np.asarray([], dtype=np.int64)

        local_entries = np.asarray(local_set.entries, dtype=np.int64)

        if on is Entity.NODE_SET:
            object_map = self._maps[file_index].node_lid_to_gid
        elif on is Entity.SIDE_SET:
            object_map = self._maps[file_index].element_lid_to_gid
        elif on is Entity.EDGE_SET:
            object_map = self._maps[file_index].edge_lid_to_gid
        elif on is Entity.FACE_SET:
            object_map = self._maps[file_index].face_lid_to_gid
        elif on is Entity.ELEMENT_SET:
            object_map = self._maps[file_index].element_lid_to_gid
        else:
            raise ExodusInvalidEntityError(f"{on.value!r} is not a set entity")

        return np.asarray(
            [object_map[int(local_id) - 1] for local_id in local_entries], dtype=np.int64
        )

    def set(self, on: Entity | str, set_id: int) -> SetInfo:
        """Return the aggregated set for the given entity type and ID.

        Dispatches to the appropriate typed set method based on *on*.

        Parameters
        ----------
        on : Entity or str
            Set entity type.  Must be one of ``Entity.NODE_SET``,
            ``Entity.SIDE_SET``, ``Entity.EDGE_SET``, ``Entity.FACE_SET``,
            or ``Entity.ELEMENT_SET``.
        set_id : int
            Global set ID.

        Returns
        -------
        SetInfo
            Aggregated set descriptor with global object IDs.

        Raises
        ------
        ExodusInvalidEntityError
            If *on* does not refer to a set entity type.
        ExodusLookupError
            If *set_id* is not found in any component file.
        """
        location = entity(on)

        if location is Entity.NODE_SET:
            return self.node_set(set_id)
        if location is Entity.SIDE_SET:
            return self.side_set(set_id)
        if location is Entity.EDGE_SET:
            return self.edge_set(set_id)
        if location is Entity.FACE_SET:
            return self.face_set(set_id)
        if location is Entity.ELEMENT_SET:
            return self.element_set(set_id)

        raise ExodusInvalidEntityError(f"{location.value!r} is not a set entity")

    def region_stats(
        self,
        name: str,
        *,
        on: Any = "element",
        block_id: int | None = None,
        blocks: str | None = None,
        region: Any,
        where: str | None = None,
        reduce: Any,
        time: TimeSelector = None,
        symmetry_factor: float = 1.0,
    ) -> Any:
        """Compute statistics of a result variable inside a geometric region.

        Delegates to :func:`exodusii.api.region_reduce.region_stats` using
        this :class:`ParallelExodusFile` as the data source.  The parallel
        aggregation (coordinates and values assembled from all component
        files) is handled transparently.

        Parameters
        ----------
        name : str
            Result variable name.
        on : Entity or str, optional
            Entity location.  Default ``"element"``.
        block_id : int or None, optional
            Restrict to one element block.  Mutually exclusive with *blocks*.
        blocks : str or None, optional
            Multi-block mode.  ``"auto"`` selects only non-empty blocks that
            define *name*.  ``"all"`` or ``None`` selects all non-empty
            blocks.  Mutually exclusive with *block_id*.
        region : Region
            Geometric region predicate.
        where : str or None, optional
            Field-threshold predicate, e.g. ``"EQPS_2 > 1.0"``.
        reduce : str or list of str
            Reducer names: ``"mean"``, ``"max"``, ``"min"``, ``"sum"``,
            ``"count"``, ``"std"``.
        time : TimeSelector, optional
            Time step selector.  ``None`` selects the last available step.
        symmetry_factor : float, optional
            Scale factor for extensive reducers.  Default ``1.0``.

        Returns
        -------
        RegionStatsResult
        """
        from exodusii.api.region_reduce import region_stats as _region_stats
        from exodusii.core.entities import entity as _entity

        return _region_stats(
            self,  # type: ignore[arg-type]
            name,
            on=str(_entity(on)),
            block_id=block_id,
            blocks=blocks,
            region=region,
            where=where,
            reduce=reduce,
            time=time,
            symmetry_factor=symmetry_factor,
        )

    def region_mass(
        self,
        *,
        block_id: int,
        region: Any,
        density_name: str = "DENSITY",
        volfrac_name: str | None = None,
        where: str | None = None,
        time: TimeSelector = None,
        symmetry_factor: float = 1.0,
    ) -> Any:
        """Compute the mass of material inside a geometric region.

        Delegates to :func:`exodusii.api.region_reduce.region_mass` using
        this :class:`ParallelExodusFile` as the data source.

        Parameters
        ----------
        block_id : int
            Element block ID.
        region : Region
            Geometric region predicate.
        density_name : str, optional
            Element density variable name.  Default ``"DENSITY"``.
        volfrac_name : str or None, optional
            Optional volume-fraction variable name.
        where : str or None, optional
            Field-threshold predicate, e.g. ``"EQPS_2 > 1.0"``.
        time : TimeSelector, optional
            Time step selector.  ``None`` selects the last available step.
        symmetry_factor : float, optional
            Symmetry scaling factor.  Default ``1.0``.

        Returns
        -------
        RegionMassResult
        """
        from exodusii.api.region_reduce import region_mass as _region_mass

        return _region_mass(
            self,  # type: ignore[arg-type]
            block_id=block_id,
            region=region,
            density_name=density_name,
            volfrac_name=volfrac_name,
            where=where,
            time=time,
            symmetry_factor=symmetry_factor,
        )

    def write(self, filename: str | Path) -> str:
        """Write the aggregated logical database to a serial Exodus file.

        All blocks, sets, coordinate data, and result variable histories are
        assembled from the component files and written to a single joined
        Exodus file.  The output file layout matches what a serial Exodus
        reader would produce for the equivalent non-decomposed mesh.

        Parameters
        ----------
        filename : str or Path
            Destination path for the joined serial Exodus file.

        Returns
        -------
        str
            String form of *filename*, identical to ``str(filename)``.

        Examples
        --------
        >>> with ParallelExodusFile.open("mesh.e.4.0", "mesh.e.4.1",
        ...                              "mesh.e.4.2", "mesh.e.4.3") as pef:
        ...     out = pef.write("joined.exo")
        ...     print(out)
        joined.exo
        """

        with ExodusWriter.create(filename) as writer:
            self._write_to(writer)
        return str(filename)

    def get_mapping(
        self, name: Any, invert: bool = False, contiguous: bool = False
    ) -> dict[Any, Any]:
        """Return a legacy local-to-global ID mapping dictionary.

        Parameters
        ----------
        name : enum or str
            Mapping name.  Recognised values (as ``name.name`` or the plain
            string):

            ``"node_local_to_global"``
                Maps ``(file_index, local_node_index)`` to the global node ID.
            ``"elem_local_to_global"``
                Maps ``(file_index, local_element_index)`` to the global
                element ID.
            ``"elem_block_elem_local_to_global"``
                Maps ``(file_index, block_id, local_index_within_block)`` to
                the global element ID.
        invert : bool, optional
            If ``True``, return the inverse mapping (global ID → local key).
            Default is ``False``.
        contiguous : bool, optional
            If ``True``, replace raw global IDs with contiguous one-based
            indices before returning (or inverting).  Default is ``False``.

        Returns
        -------
        dict
            Mapping dictionary.  Keys are tuples as described above; values
            are global IDs (or contiguous indices when *contiguous* is
            ``True``).  When *invert* is ``True`` keys and values are swapped.

        Notes
        -----
        This method exists to support legacy callers that expect the dict-based
        mapping interface of the old ``parallel_exodusii_file`` API.  New code
        should prefer :attr:`files` and the per-file ``node_num_map`` /
        ``elem_num_map`` arrays directly.

        Raises
        ------
        ValueError
            If *name* is not one of the recognised mapping names.
        """

        map_name = getattr(name, "name", str(name))

        mapping: dict[Any, Any]
        if map_name == "node_local_to_global":
            mapping = {
                (file_index, local_index): int(gid)
                for file_index, maps in enumerate(self._maps)
                for local_index, gid in enumerate(maps.node_lid_to_gid, start=1)
            }
        elif map_name == "elem_local_to_global":
            mapping = {
                (file_index, local_index): int(gid)
                for file_index, maps in enumerate(self._maps)
                for local_index, gid in enumerate(maps.element_lid_to_gid, start=1)
            }
        elif map_name == "elem_block_elem_local_to_global":
            mapping = {}
            for file_index, file in enumerate(self._files):
                for block_id in file.element_block_ids():
                    for local_index, gid in enumerate(
                        self._block_element_gids_for_file(file_index, int(block_id)), start=1
                    ):
                        mapping[(file_index, int(block_id), local_index)] = int(gid)
        else:
            raise ValueError(f"invalid map name {name!r}")

        if contiguous:
            gids = sorted(set(mapping.values()))
            contiguous_map = {gid: index for index, gid in enumerate(gids, start=1)}
            mapping = {key: contiguous_map[gid] for key, gid in mapping.items()}

        if invert:
            return {value: key for key, value in mapping.items()}

        return mapping

    def _resolve_time_index(self, time: TimeSelector) -> int:
        from exodusii.core.time import resolve_time

        return resolve_time(self.times(), time).index

    def _element_values(
        self, name: str, *, block: int, time: TimeSelector
    ) -> npt.NDArray[np.float64]:
        return self._block_object_values(name, on=Entity.ELEMENT, block=block, time=time)

    def _write_to(self, writer: ExodusWriter) -> None:
        writer.initialize(
            self.title,
            self.dimension,
            self.node_count,
            self.element_count,
            element_blocks=self.element_block_count,
            node_sets=self.node_set_count,
            side_sets=self.side_set_count,
            edge_count=self.edge_count,
            edge_blocks=self.edge_block_count,
            edge_sets=len(self.edge_set_ids()),
            face_count=self.face_count,
            face_blocks=self.face_block_count,
            face_sets=len(self.face_set_ids()),
            element_sets=len(self.element_set_ids()),
        )

        if self.node_count:
            writer.write_coordinates(self.coordinates(), names=self.coordinate_names().tolist())

        for block_id in self.element_block_ids():
            block = self.element_block(int(block_id))

            nodal_conn = self.element_connectivity(block.id)
            if nodal_conn.size == 0:
                # Cannot define a conventional serial element block without
                # nodal connectivity. Preserve block metadata with a zero-width
                # placeholder is not NetCDF-safe, so fail clearly.
                raise NotImplementedError(
                    f"element block {block.id} has no nodal connectivity; "
                    "edge/face-only element block writing is not implemented yet"
                )

            edge_conn = self.element_edge_connectivity(block.id)
            face_conn = self.element_face_connectivity(block.id)

            writer.define_element_block(
                block.id, block.element_type, self.element_connectivity(block.id), name=block.name
            )

            if edge_conn is not None:
                writer.write_element_edge_connectivity(block.id, edge_conn)

            if face_conn is not None:
                writer.write_element_face_connectivity(block.id, face_conn)

        for block_id in self.edge_block_ids():
            block = self.edge_block(int(block_id))
            writer.define_edge_block(
                block.id, block.element_type, self.edge_connectivity(block.id), name=block.name
            )

        for block_id in self.face_block_ids():
            block = self.face_block(int(block_id))
            writer.define_face_block(
                block.id, block.element_type, self.face_connectivity(block.id), name=block.name
            )

        for set_id in self.node_set_ids():
            node_set = self.node_set(int(set_id))
            nodes = (
                _labels_to_contiguous(node_set.nodes, self._node_gid_to_index)
                if node_set.nodes is not None
                else np.asarray([], dtype=np.int64)
            )
            writer.define_node_set(
                node_set.id,
                nodes,
                distribution_factors=self._distribution_factors_for_write(
                    Entity.NODE_SET, node_set
                ),
                name=node_set.name,
            )
        for set_id in self.side_set_ids():
            side_set = self.side_set(int(set_id))
            elements = (
                _labels_to_contiguous(side_set.elems, self._element_gid_to_index)
                if side_set.elems is not None
                else np.asarray([], dtype=np.int64)
            )
            writer.define_side_set(
                side_set.id,
                elements,
                side_set.sides if side_set.sides is not None else [],
                distribution_factors=self._distribution_factors_for_write(
                    Entity.SIDE_SET, side_set
                ),
                name=side_set.name,
            )
        global_names = self.variable_names(Entity.GLOBAL)
        node_names = self.variable_names(Entity.NODE)
        element_names = self.variable_names(Entity.ELEMENT)
        edge_names = self.variable_names(Entity.EDGE) if len(self.edge_block_ids()) else ()
        face_names = self.variable_names(Entity.FACE) if len(self.face_block_ids()) else ()
        node_set_names = self.variable_names(Entity.NODE_SET)
        side_set_names = self.variable_names(Entity.SIDE_SET)
        edge_set_names = self.variable_names(Entity.EDGE_SET) if len(self.edge_set_ids()) else ()
        face_set_names = self.variable_names(Entity.FACE_SET) if len(self.face_set_ids()) else ()
        element_set_names = (
            self.variable_names(Entity.ELEMENT_SET) if len(self.element_set_ids()) else ()
        )

        if global_names:
            writer.define_global_variables(global_names)
        if node_names:
            writer.define_node_variables(node_names)
        if element_names:
            writer.define_element_variables(element_names)
        if edge_names:
            writer.define_edge_variables(edge_names)
        if face_names:
            writer.define_face_variables(face_names)
        if node_set_names:
            writer.define_node_set_variables(node_set_names)
        if side_set_names:
            writer.define_side_set_variables(side_set_names)
        if edge_set_names:
            writer.define_edge_set_variables(edge_set_names)
        if face_set_names:
            writer.define_face_set_variables(face_set_names)
        if element_set_names:
            writer.define_element_set_variables(element_set_names)

        for step, time in enumerate(self.times(), start=1):
            writer.write_time(float(time), step=step)

        self._write_all_variable_histories(
            writer,
            global_names=global_names,
            node_names=node_names,
            element_names=element_names,
            edge_names=edge_names,
            face_names=face_names,
            node_set_names=node_set_names,
            side_set_names=side_set_names,
            edge_set_names=edge_set_names,
            face_set_names=face_set_names,
            element_set_names=element_set_names,
        )

    def _write_all_variable_histories(
        self,
        writer: ExodusWriter,
        *,
        global_names: tuple[str, ...],
        node_names: tuple[str, ...],
        element_names: tuple[str, ...],
        edge_names: tuple[str, ...],
        face_names: tuple[str, ...],
        node_set_names: tuple[str, ...],
        side_set_names: tuple[str, ...],
        edge_set_names: tuple[str, ...],
        face_set_names: tuple[str, ...],
        element_set_names: tuple[str, ...],
    ) -> None:
        """Write all variable histories in bulk.

        This avoids repeated per-time-step parallel aggregation, which is very
        slow for large decomposed legacy fixtures.
        """

        if global_names:
            values = np.column_stack(
                [self.values(name, on=Entity.GLOBAL, time=None) for name in global_names]
            )
            writer.backend.write_variable(variable_value_name(Entity.GLOBAL, 1), values)

        for name in node_names:
            variable_index = writer._name_index(variable_spec(Entity.NODE).names_variable, name)
            writer.backend.write_variable(
                variable_value_name(Entity.NODE, variable_index),
                self.values(name, on=Entity.NODE, time=None),
            )

        self._write_block_variable_histories(
            writer, Entity.ELEMENT, element_names, self.element_block_ids()
        )
        self._write_block_variable_histories(writer, Entity.EDGE, edge_names, self.edge_block_ids())
        self._write_block_variable_histories(writer, Entity.FACE, face_names, self.face_block_ids())

        self._write_set_variable_histories(
            writer, Entity.NODE_SET, node_set_names, self.node_set_ids()
        )
        self._write_set_variable_histories(
            writer, Entity.SIDE_SET, side_set_names, self.side_set_ids()
        )
        self._write_set_variable_histories(
            writer, Entity.EDGE_SET, edge_set_names, self.edge_set_ids()
        )
        self._write_set_variable_histories(
            writer, Entity.FACE_SET, face_set_names, self.face_set_ids()
        )
        self._write_set_variable_histories(
            writer, Entity.ELEMENT_SET, element_set_names, self.element_set_ids()
        )

    def _write_block_variable_histories(
        self, writer: ExodusWriter, on: Entity, names: tuple[str, ...], block_ids: npt.ArrayLike
    ) -> None:
        if not names:
            return

        spec = variable_spec(on)
        if spec.location_entity is None:
            return

        block_id_array = np.asarray(block_ids, dtype=np.int64).reshape(-1)

        for name in names:
            variable_index = writer._name_index(spec.names_variable, name)
            for block_id_value in block_id_array:
                block_id_int = int(block_id_value)
                location_index = writer._location_index(spec.location_entity, block_id_int)
                writer.backend.write_variable(
                    variable_value_name(on, variable_index, location_index),
                    self.values(name, on=on, block_id=block_id_int, time=None),
                )

    def _write_set_variable_histories(
        self, writer: ExodusWriter, on: Entity, names: tuple[str, ...], set_ids: npt.ArrayLike
    ) -> None:
        if not names:
            return

        spec = variable_spec(on)
        if spec.location_entity is None:
            return

        set_id_array = np.asarray(set_ids, dtype=np.int64).reshape(-1)

        for name in names:
            variable_index = writer._name_index(spec.names_variable, name)
            for set_id_value in set_id_array:
                set_id_int = int(set_id_value)
                location_index = writer._location_index(spec.location_entity, set_id_int)
                writer.backend.write_variable(
                    variable_value_name(on, variable_index, location_index),
                    self.values(name, on=on, set_id=set_id_int, time=None),
                )

    def _file_block_is_active(self, file: ExodusFile, location: Entity, block_id: int) -> bool:
        if not _contains(file.block_ids(location), block_id):
            return False
        return file.block_is_active(location, block_id)

    def _file_set_is_active(self, file: ExodusFile, location: Entity, set_id: int) -> bool:
        if not _contains(file.set_ids(location), set_id):
            return False
        return file.set_is_active(location, set_id)

    def _build_file_maps(self) -> list[_FileMaps]:
        maps: list[_FileMaps] = []
        node_offset = 0
        element_offset = 0
        edge_offset = 0
        face_offset = 0

        for file in self._files:
            node_map = file.variable(VariableName.NODE_ID_MAP.value, default=None)
            if node_map is None:
                warnings.warn(
                    f"{file.path}: 'node_num_map' not found; using sequential fallback. "
                    "This is only correct for non-overlapping partitions with no shared nodes. "
                    "Real Nemesis files with shared border/external nodes must include "
                    "node_num_map.",
                    stacklevel=2,
                )
                node_map = np.arange(node_offset + 1, node_offset + file.node_count + 1)
            node_map = np.asarray(node_map, dtype=np.int64)

            element_map = file.variable(VariableName.ELEMENT_ID_MAP.value, default=None)
            if element_map is None:
                warnings.warn(
                    f"{file.path}: 'elem_num_map' not found; using sequential fallback. "
                    "This is only correct for non-overlapping partitions.",
                    stacklevel=2,
                )
                element_map = np.arange(element_offset + 1, element_offset + file.element_count + 1)
            element_map = np.asarray(element_map, dtype=np.int64)

            edge_map = file.variable(VariableName.EDGE_ID_MAP.value, default=None)
            if edge_map is None:
                if file.edge_count > 0:
                    warnings.warn(
                        f"{file.path}: 'edge_num_map' not found; using sequential fallback.",
                        stacklevel=2,
                    )
                edge_map = np.arange(edge_offset + 1, edge_offset + file.edge_count + 1)
            edge_map = np.asarray(edge_map, dtype=np.int64)

            face_map = file.variable(VariableName.FACE_ID_MAP.value, default=None)
            if face_map is None:
                if file.face_count > 0:
                    warnings.warn(
                        f"{file.path}: 'face_num_map' not found; using sequential fallback.",
                        stacklevel=2,
                    )
                face_map = np.arange(face_offset + 1, face_offset + file.face_count + 1)
            face_map = np.asarray(face_map, dtype=np.int64)

            if len(node_map) != file.node_count:
                raise ExodusConsistencyError(f"node map length mismatch in {file.path}")
            if len(element_map) != file.element_count:
                raise ExodusConsistencyError(f"element map length mismatch in {file.path}")
            if len(edge_map) != file.edge_count:
                raise ExodusConsistencyError(f"edge map length mismatch in {file.path}")
            if len(face_map) != file.face_count:
                raise ExodusConsistencyError(f"face map length mismatch in {file.path}")

            maps.append(
                _FileMaps(
                    node_lid_to_gid=node_map,
                    element_lid_to_gid=element_map,
                    edge_lid_to_gid=edge_map,
                    face_lid_to_gid=face_map,
                )
            )

            node_offset += file.node_count
            element_offset += file.element_count
            edge_offset += file.edge_count
            face_offset += file.face_count

        return maps

    def _block_object_values(
        self, name: str, *, on: Entity, block: int, time: TimeSelector
    ) -> npt.NDArray[np.float64]:
        """Fast aggregate element/edge/face block variable values.

        This intentionally avoids calling ``file.values(...)`` for each component
        file because that path repeatedly decodes variable-name character arrays
        and is too slow for large decomposed legacy fixtures.
        """

        block_location = _variable_block_location(on)
        block_gids = self._block_object_gids(block_location, block)
        block_gid_to_row = {gid: row for row, gid in enumerate(block_gids)}
        variable_index = self._variable_index(on, name)

        first_array: npt.NDArray[np.float64] | None = None

        for file in self._files:
            if not self._file_block_is_active(file, block_location, block):
                continue

            local_block_index = file._block_index(block_location, block)
            variable_name = variable_value_name(on, variable_index, local_block_index)
            local_values = file.variable(variable_name, default=None)
            if local_values is not None:
                first_array = np.asarray(local_values, dtype=np.float64)
                break

        if first_array is None:
            if time is None:
                return np.zeros((len(self.times()), len(block_gids)), dtype=np.float64)
            return np.zeros(len(block_gids), dtype=np.float64)

        output: npt.NDArray[np.float64]
        if time is None:
            output = np.zeros((first_array.shape[0], len(block_gids)), dtype=np.float64)
            selection_index = None
        else:
            output = np.zeros(len(block_gids), dtype=np.float64)
            selection_index = self._resolve_time_index(time)

        for file_index, file in enumerate(self._files):
            if not self._file_block_is_active(file, block_location, block):
                continue

            local_block_index = file._block_index(block_location, block)
            variable_name = variable_value_name(on, variable_index, local_block_index)
            local_values = file.variable(variable_name, default=None)
            if local_values is None:
                continue

            local_array = np.asarray(local_values, dtype=np.float64)
            local_gids = self._block_object_gids_for_file(file_index, block_location, block)

            for local_index, gid in enumerate(local_gids):
                output_index = block_gid_to_row[int(gid)]
                if selection_index is None:
                    output[:, output_index] = local_array[:, local_index]
                else:
                    output[output_index] = local_array[selection_index, local_index]

        return output

    def _block_object_gids(self, location: Entity, block_id: int) -> list[int]:
        gids: list[int] = []
        for file_index, file in enumerate(self._files):
            if self._file_block_is_active(file, location, block_id):
                gids.extend(
                    self._block_object_gids_for_file(file_index, location, block_id).tolist()
                )

        if not gids:
            for file_index, file in enumerate(self._files):
                if _contains(file.block_ids(location), block_id):
                    gids.extend(
                        self._block_object_gids_for_file(file_index, location, block_id).tolist()
                    )

        if not gids:
            raise ExodusLookupError(f"{location.value} ID {block_id} not found")

        return sorted(set(gids))

    def _block_object_gids_for_file(
        self, file_index: int, location: Entity, block_id: int, *, require_active: bool = False
    ) -> npt.NDArray[np.int64]:
        """Return the global object IDs for a block from one component file.

        .. note:: **Specification assumption**

           This method maps local object indices to global IDs by accumulating
           per-block element counts across blocks in the order returned by
           ``file.block_ids(location)``.  This relies on the conventional
           Nemesis/Exodus layout where ``elem_num_map`` (or ``edge_num_map`` /
           ``face_num_map``) lists all objects contiguously per block, in the
           same order as the block ID list.

           Standard decomposers (e.g., ``nem_slice``) write files in this
           layout, but the Exodus/Nemesis specification does not formally
           guarantee it.  Files produced by non-standard decomposers that store
           objects in a different order within the map will produce incorrect
           GID assignments here.
        """
        file = self._files[file_index]
        spec = block_spec(location)

        if require_active and not self._file_block_is_active(file, location, block_id):
            return np.asarray([], dtype=np.int64)

        if location is Entity.ELEMENT_BLOCK:
            object_map = self._maps[file_index].element_lid_to_gid
        elif location is Entity.EDGE_BLOCK:
            object_map = self._maps[file_index].edge_lid_to_gid
        elif location is Entity.FACE_BLOCK:
            object_map = self._maps[file_index].face_lid_to_gid
        else:
            raise ExodusInvalidEntityError(f"{location.value!r} is not a block entity")

        start = 0
        local_block_ids = file.block_ids(location)

        for local_block_position, current_block_id in enumerate(local_block_ids, start=1):
            # Do not call file.block(...) here.  In parallel/Nemesis files, inactive
            # local blocks may have status metadata but no connectivity variable.
            count = file.dimension_size(
                spec.object_count_dimension(local_block_position), default=0
            )
            stop = start + count

            if int(current_block_id) == block_id:
                return object_map[start:stop]

            start = stop

        return np.asarray([], dtype=np.int64)

    def _block_element_gids(self, block_id: int) -> list[int]:
        return self._block_object_gids(Entity.ELEMENT_BLOCK, block_id)

    def _block_element_gids_for_file(self, file_index: int, block_id: int) -> npt.NDArray[np.int64]:
        return self._block_object_gids_for_file(file_index, Entity.ELEMENT_BLOCK, block_id)

    def _check_consistency(self) -> None:
        dimension = self._files[0].dimension
        times = self._files[0].times()
        coord_names = self._files[0].coordinate_names().tolist()

        for file in self._files[1:]:
            if file.dimension != dimension:
                raise ExodusConsistencyError("spatial dimension not consistent across files")
            if not np.allclose(file.times(), times):
                raise ExodusConsistencyError("time values not consistent across files")
            if file.coordinate_names().tolist() != coord_names:
                raise ExodusConsistencyError("coordinate names not consistent across files")

            for location in (Entity.GLOBAL, Entity.NODE, Entity.ELEMENT):
                if file.variable_names(location) != self._files[0].variable_names(location):
                    raise ExodusConsistencyError(
                        f"{location.value} variable names not consistent across files"
                    )

    def _validate_global_metadata(self) -> None:
        expected_nodes = self._global_dimension(DimensionName.NUM_NODES_GLOBAL.value)
        if expected_nodes is not None and expected_nodes != len(self._node_gid_to_index):
            raise ExodusConsistencyError(
                f"computed global node count {len(self._node_gid_to_index)} "
                f"does not match {DimensionName.NUM_NODES_GLOBAL.value}={expected_nodes}"
            )

        expected_elements = self._global_dimension(DimensionName.NUM_ELEMENTS_GLOBAL.value)
        if expected_elements is not None and expected_elements != len(self._element_gid_to_index):
            raise ExodusConsistencyError(
                f"computed global element count {len(self._element_gid_to_index)} "
                f"does not match {DimensionName.NUM_ELEMENTS_GLOBAL.value}={expected_elements}"
            )

        expected_blocks = self._global_dimension(DimensionName.NUM_ELEMENT_BLOCKS_GLOBAL.value)
        if expected_blocks is not None and expected_blocks != len(self.element_block_ids()):
            raise ExodusConsistencyError(
                f"computed global element block count {len(self.element_block_ids())} "
                f"does not match {DimensionName.NUM_ELEMENT_BLOCKS_GLOBAL.value}={expected_blocks}"
            )

        expected_node_sets = self._global_dimension(DimensionName.NUM_NODE_SETS_GLOBAL.value)
        if expected_node_sets is not None and expected_node_sets != len(self.node_set_ids()):
            raise ExodusConsistencyError(
                f"computed global node set count {len(self.node_set_ids())} "
                f"does not match {DimensionName.NUM_NODE_SETS_GLOBAL.value}={expected_node_sets}"
            )

        expected_side_sets = self._global_dimension(DimensionName.NUM_SIDE_SETS_GLOBAL.value)
        if expected_side_sets is not None and expected_side_sets != len(self.side_set_ids()):
            raise ExodusConsistencyError(
                f"computed global side set count {len(self.side_set_ids())} "
                f"does not match {DimensionName.NUM_SIDE_SETS_GLOBAL.value}={expected_side_sets}"
            )

    def get_time_step(self, target: float, pcttol: float = 1.0e-5) -> int:
        """Return the one-based index of the time step nearest to *target*.

        Parameters
        ----------
        target : float
            Desired simulation time value.
        pcttol : float, optional
            Relative tolerance for warning about a mismatch between *target*
            and the nearest available time.  A logging warning is emitted when
            ``abs(t_nearest - target) / abs(target) > pcttol``.
            Default is ``1e-5``.

        Returns
        -------
        int
            One-based index of the time step whose time value is closest to
            *target*.

        Raises
        ------
        ValueError
            If the database contains no time steps.
        """

        times = np.asarray(self.times(), dtype=np.float64)
        if times.size == 0:
            raise ValueError("no time steps found")

        index = int(np.abs(times - target).argmin())

        if abs(target) > 0.0:
            relerr = abs(float(times[index]) - target) / abs(target)
            if relerr > pcttol:
                import logging

                logging.warning("Solution time differs significantly from desired solution time.")

        return index + 1


def _contains(values: npt.ArrayLike, value: int) -> bool:
    return bool(np.any(np.asarray(values) == value))


def _unique_sorted(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
    return np.asarray(sorted(set(np.asarray(values, dtype=np.int64).tolist())), dtype=np.int64)


def _contiguous_gid_map(values: npt.ArrayLike) -> dict[int, int]:
    gids = sorted(set(np.asarray(values, dtype=np.int64).tolist()))
    return {int(gid): index for index, gid in enumerate(gids, start=1)}


def _labels_to_contiguous(
    labels: npt.ArrayLike, label_to_index: dict[int, int]
) -> npt.NDArray[np.int64]:
    return np.asarray(
        [label_to_index[int(label)] for label in np.asarray(labels).reshape(-1)], dtype=np.int64
    )


def _one_based_index(values: npt.ArrayLike, value: int) -> int:
    matches = np.nonzero(np.asarray(values) == value)[0]
    if not len(matches):
        raise ExodusLookupError(f"ID {value} not found")
    return int(matches[0]) + 1


def _contiguous_gid_map_or_empty(values: list[npt.NDArray[np.int64]]) -> dict[int, int]:
    nonempty = [value for value in values if len(value)]
    if not nonempty:
        return {}
    return _contiguous_gid_map(np.concatenate(nonempty))


def _variable_block_location(location: Entity) -> Entity:
    if location is Entity.ELEMENT:
        return Entity.ELEMENT_BLOCK
    if location is Entity.EDGE:
        return Entity.EDGE_BLOCK
    if location is Entity.FACE:
        return Entity.FACE_BLOCK
    raise ExodusInvalidEntityError(f"{location.value!r} is not a block variable entity")


def _set_primary_labels(set_info: SetInfo, on: Entity) -> npt.NDArray[np.int64]:
    if set_info.entries is None:
        return np.asarray([], dtype=np.int64)
    return np.asarray(set_info.entries, dtype=np.int64)


def _set_value_keys(set_info: SetInfo, on: Entity) -> list[tuple[int, ...]]:
    if set_info.entries is None:
        return []

    entries = np.asarray(set_info.entries, dtype=np.int64)

    if on is Entity.SIDE_SET and set_info.extra_entries is not None:
        extra = np.asarray(set_info.extra_entries, dtype=np.int64)
        return [(int(entry), int(side)) for entry, side in zip(entries, extra, strict=True)]

    return [(int(entry),) for entry in entries]


def _set_value_keys_from_local(
    labels: npt.NDArray[np.int64], local_set: SetInfo, on: Entity
) -> list[tuple[int, ...]]:
    if on is Entity.SIDE_SET and local_set.extra_entries is not None:
        extra = np.asarray(local_set.extra_entries, dtype=np.int64)
        return [(int(label), int(side)) for label, side in zip(labels, extra, strict=True)]

    return [(int(label),) for label in labels]


__all__ = ["ParallelExodusFile"]
