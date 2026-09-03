# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Copy Exodus database contents between modern API objects.

This module provides two public entry points:

* :func:`copy` — copies supported Exodus contents from any source object
  that exposes the standard exodusii read API into an open
  :class:`~exodusii.api.writer.ExodusWriter`.
* :func:`copy_file` — convenience wrapper that opens a source
  :class:`~exodusii.api.file.ExodusFile` (or accepts an already-open one),
  creates a new :class:`~exodusii.api.writer.ExodusWriter` at *target*, calls
  :func:`copy`, and returns the target path string.

Copied content includes nodal coordinates, element/edge/face block
connectivity and attributes, node sets and side sets (and the higher-order
edge-set, face-set, and element-set families when present), all result
variables (global, nodal, element, edge, face, and set variables) across
every time step, QA records, info records, block/set properties, and
block/set active-status arrays.
"""

from pathlib import Path
from typing import Any

import numpy as np

from exodusii.api.file import ExodusFile
from exodusii.api.writer import ExodusWriter
from exodusii.core.entities import Entity
from exodusii.core.schema import variable_spec


def copy(source: Any, target: ExodusWriter) -> None:
    """Copy supported Exodus contents from *source* into *target*.

    Reads all mesh topology and result-variable data from *source* using the
    standard exodusii read API and writes it to the already-opened *target*
    writer.  The writer must not yet have been initialized; :func:`copy`
    calls ``target.initialize`` internally.

    Parameters
    ----------
    source : any
        Any object that exposes the standard exodusii read API, including
        :class:`~exodusii.api.file.ExodusFile` and legacy
        ``ExodusIIFile`` instances.  The following attributes and methods
        are required: ``title``, ``dimension``, ``node_count``,
        ``element_count``, ``element_block_count``, ``node_set_count``,
        ``side_set_count``, ``edge_count``, ``face_count``,
        ``coordinates()``, ``coordinate_names()``,
        ``element_block_ids()``, ``node_set_ids()``, ``side_set_ids()``,
        ``variable_names(entity)``, ``variable_truth_table(entity)``,
        ``values(name, on, ...)``, and ``times()``.
    target : ExodusWriter
        An open, not-yet-initialized writer object.  Typically obtained
        via :meth:`ExodusWriter.create`.

    Notes
    -----
    The following data are copied:

    * Mesh initialization parameters (title, dimension, counts).
    * Nodal coordinates and coordinate names.
    * Node, element, edge, and face ID maps (when present).
    * Element, edge, and face block definitions (type, connectivity,
      per-block attributes, and attribute names).
    * Node sets, side sets, edge sets, face sets, and element sets
      (including entries, distribution factors, and names).
    * Variable definitions and truth tables for all entity types.
    * Complete time-step history for all result variables.
    * Block and set active-status arrays.
    * Info records and QA records.
    * Block and set user-defined properties.

    Examples
    --------
    >>> from exodusii.api.file import ExodusFile
    >>> from exodusii.api.writer import ExodusWriter
    >>> with ExodusFile.open("source.exo") as src:
    ...     with ExodusWriter.create("dest.exo") as dst:
    ...         copy(src, dst)
    """

    target.initialize(
        source.title,
        source.dimension,
        source.node_count,
        source.element_count,
        element_blocks=source.element_block_count,
        node_sets=source.node_set_count,
        side_sets=source.side_set_count,
        edge_count=source.edge_count,
        edge_blocks=len(source.edge_block_ids()) if hasattr(source, "edge_block_ids") else 0,
        edge_sets=len(source.edge_set_ids()) if hasattr(source, "edge_set_ids") else 0,
        face_count=source.face_count,
        face_blocks=len(source.face_block_ids()) if hasattr(source, "face_block_ids") else 0,
        face_sets=len(source.face_set_ids()) if hasattr(source, "face_set_ids") else 0,
        element_sets=len(source.element_set_ids()) if hasattr(source, "element_set_ids") else 0,
    )

    if source.node_count:
        target.write_coordinates(source.coordinates(), names=source.coordinate_names().tolist())

    _copy_id_maps(source, target)
    _copy_blocks(source, target)
    _copy_sets(source, target)
    _copy_variables(source, target)
    _copy_history(source, target)
    _copy_status(source, target)
    _copy_records(source, target)
    _copy_properties(source, target)


def copy_file(source: str | Path | ExodusFile, target: str | Path) -> str:
    """Copy a source Exodus file to *target* and return the target path string.

    Opens *source* if it is not already an :class:`~exodusii.api.file.ExodusFile`,
    creates a new Exodus file at *target*, copies all supported data using
    :func:`copy`, and returns ``str(target)``.

    Parameters
    ----------
    source : str or Path or ExodusFile
        Source Exodus file.  A path string or :class:`~pathlib.Path` is
        opened automatically; an already-open :class:`ExodusFile` is used
        directly and is *not* closed by this function.
    target : str or Path
        Destination path for the new Exodus file.  The file is created (or
        overwritten) by :meth:`ExodusWriter.create`.

    Returns
    -------
    str
        String representation of *target*.

    Examples
    --------
    >>> out = copy_file("original.exo", "/tmp/copy.exo")
    >>> print(out)
    /tmp/copy.exo
    """

    close_source = False
    if isinstance(source, ExodusFile):
        source_file = source
    else:
        source_file = ExodusFile.open(source)
        close_source = True

    try:
        with ExodusWriter.create(target) as writer:
            copy(source_file, writer)
    finally:
        if close_source:
            source_file.close()

    return str(target)


def _copy_id_maps(source: Any, target: ExodusWriter) -> None:
    if _has_variable(source, "node_num_map"):
        target.write_node_id_map(source.ids(Entity.NODE))
    if _has_variable(source, "elem_num_map"):
        target.write_element_id_map(source.ids(Entity.ELEMENT))
    if _has_variable(source, "edge_num_map"):
        target.write_edge_id_map(source.ids(Entity.EDGE))
    if _has_variable(source, "face_num_map"):
        target.write_face_id_map(source.ids(Entity.FACE))


def _copy_blocks(source: Any, target: ExodusWriter) -> None:
    for block_id in source.element_block_ids():
        block = source.element_block(int(block_id))
        target.define_element_block(
            block.id, block.element_type, source.element_connectivity(block.id), name=block.name
        )
        _copy_block_attributes(source, target, Entity.ELEMENT_BLOCK, block.id)

    if hasattr(source, "edge_block_ids"):
        for block_id in source.edge_block_ids():
            block = source.edge_block(int(block_id))
            target.define_edge_block(
                block.id, block.element_type, source.edge_connectivity(block.id), name=block.name
            )
            _copy_block_attributes(source, target, Entity.EDGE_BLOCK, block.id)

    if hasattr(source, "face_block_ids"):
        for block_id in source.face_block_ids():
            block = source.face_block(int(block_id))
            target.define_face_block(
                block.id, block.element_type, source.face_connectivity(block.id), name=block.name
            )
            _copy_block_attributes(source, target, Entity.FACE_BLOCK, block.id)


def _copy_sets(source: Any, target: ExodusWriter) -> None:
    for set_id in source.node_set_ids():
        node_set = source.node_set(int(set_id))
        target.define_node_set(
            node_set.id,
            node_set.nodes if node_set.nodes is not None else [],
            distribution_factors=node_set.dist_facts if node_set.distribution_factors else None,
            name=node_set.name,
        )

    for set_id in source.side_set_ids():
        side_set = source.side_set(int(set_id))
        target.define_side_set(
            side_set.id,
            side_set.elems if side_set.elems is not None else [],
            side_set.sides if side_set.sides is not None else [],
            distribution_factors=side_set.dist_facts if side_set.distribution_factors else None,
            name=side_set.name,
        )

    if hasattr(source, "edge_set_ids"):
        for set_id in source.edge_set_ids():
            edge_set = source.edge_set(int(set_id))
            target.define_edge_set(
                edge_set.id,
                edge_set.entries if edge_set.entries is not None else [],
                orientations=edge_set.extra_entries,
                distribution_factors=edge_set.dist_facts if edge_set.distribution_factors else None,
                name=edge_set.name,
            )

    if hasattr(source, "face_set_ids"):
        for set_id in source.face_set_ids():
            face_set = source.face_set(int(set_id))
            target.define_face_set(
                face_set.id,
                face_set.entries if face_set.entries is not None else [],
                orientations=face_set.extra_entries,
                distribution_factors=face_set.dist_facts if face_set.distribution_factors else None,
                name=face_set.name,
            )

    if hasattr(source, "element_set_ids"):
        for set_id in source.element_set_ids():
            element_set = source.element_set(int(set_id))
            target.define_element_set(
                element_set.id,
                element_set.entries if element_set.entries is not None else [],
                distribution_factors=element_set.dist_facts
                if element_set.distribution_factors
                else None,
                name=element_set.name,
            )


def _copy_variables(source: Any, target: ExodusWriter) -> None:
    _define_variables_if_present(source, target, Entity.GLOBAL)
    _define_variables_if_present(source, target, Entity.NODE)
    _define_variables_if_present(source, target, Entity.ELEMENT)
    _define_variables_if_present(source, target, Entity.EDGE)
    _define_variables_if_present(source, target, Entity.FACE)
    _define_variables_if_present(source, target, Entity.NODE_SET)
    _define_variables_if_present(source, target, Entity.SIDE_SET)
    _define_variables_if_present(source, target, Entity.EDGE_SET)
    _define_variables_if_present(source, target, Entity.FACE_SET)
    _define_variables_if_present(source, target, Entity.ELEMENT_SET)


def _define_variables_if_present(source: Any, target: ExodusWriter, on: Entity) -> None:
    names = source.variable_names(on)
    if not names:
        return

    truth_table = None
    spec = variable_spec(on)
    if spec.truth_table_variable is not None:
        truth_table = source.variable_truth_table(on)

    if on is Entity.GLOBAL:
        target.define_global_variables(names)
    elif on is Entity.NODE:
        target.define_node_variables(names)
    elif on is Entity.ELEMENT:
        target.define_element_variables(names, truth_table=truth_table)
    elif on is Entity.EDGE:
        target.define_edge_variables(names, truth_table=truth_table)
    elif on is Entity.FACE:
        target.define_face_variables(names, truth_table=truth_table)
    elif on is Entity.NODE_SET:
        target.define_node_set_variables(names, truth_table=truth_table)
    elif on is Entity.SIDE_SET:
        target.define_side_set_variables(names, truth_table=truth_table)
    elif on is Entity.EDGE_SET:
        target.define_edge_set_variables(names, truth_table=truth_table)
    elif on is Entity.FACE_SET:
        target.define_face_set_variables(names, truth_table=truth_table)
    elif on is Entity.ELEMENT_SET:
        target.define_element_set_variables(names, truth_table=truth_table)


def _copy_history(source: Any, target: ExodusWriter) -> None:
    global_names = source.variable_names(Entity.GLOBAL)
    node_names = source.variable_names(Entity.NODE)
    element_names = source.variable_names(Entity.ELEMENT)
    edge_names = source.variable_names(Entity.EDGE)
    face_names = source.variable_names(Entity.FACE)
    node_set_names = source.variable_names(Entity.NODE_SET)
    side_set_names = source.variable_names(Entity.SIDE_SET)
    edge_set_names = source.variable_names(Entity.EDGE_SET)
    face_set_names = source.variable_names(Entity.FACE_SET)
    element_set_names = source.variable_names(Entity.ELEMENT_SET)

    for step, time in enumerate(source.times(), start=1):
        time_index = step - 1
        target.write_time(float(time), step=step)

        if global_names:
            target.write_global_values(
                np.asarray(
                    [
                        source.values(name, on=Entity.GLOBAL, time=time_index)
                        for name in global_names
                    ],
                    dtype=np.float64,
                ),
                step=step,
            )

        for name in node_names:
            target.write_node_values(
                name, source.values(name, on=Entity.NODE, time=time_index), step=step
            )

        _copy_block_variable_history_at_step(
            source,
            target,
            Entity.ELEMENT,
            element_names,
            source.element_block_ids(),
            time_index,
            step,
        )
        _copy_block_variable_history_at_step(
            source,
            target,
            Entity.EDGE,
            edge_names,
            source.edge_block_ids() if hasattr(source, "edge_block_ids") else [],
            time_index,
            step,
        )
        _copy_block_variable_history_at_step(
            source,
            target,
            Entity.FACE,
            face_names,
            source.face_block_ids() if hasattr(source, "face_block_ids") else [],
            time_index,
            step,
        )

        _copy_set_variable_history_at_step(
            source, target, Entity.NODE_SET, node_set_names, source.node_set_ids(), time_index, step
        )
        _copy_set_variable_history_at_step(
            source, target, Entity.SIDE_SET, side_set_names, source.side_set_ids(), time_index, step
        )
        _copy_set_variable_history_at_step(
            source,
            target,
            Entity.EDGE_SET,
            edge_set_names,
            source.edge_set_ids() if hasattr(source, "edge_set_ids") else [],
            time_index,
            step,
        )
        _copy_set_variable_history_at_step(
            source,
            target,
            Entity.FACE_SET,
            face_set_names,
            source.face_set_ids() if hasattr(source, "face_set_ids") else [],
            time_index,
            step,
        )
        _copy_set_variable_history_at_step(
            source,
            target,
            Entity.ELEMENT_SET,
            element_set_names,
            source.element_set_ids() if hasattr(source, "element_set_ids") else [],
            time_index,
            step,
        )


def _copy_block_variable_history_at_step(
    source: Any,
    target: ExodusWriter,
    on: Entity,
    names: tuple[str, ...],
    block_ids: Any,
    time_index: int,
    step: int,
) -> None:
    for name in names:
        for block_id in block_ids:
            values = source.values(name, on=on, block_id=int(block_id), time=time_index)

            if on is Entity.ELEMENT:
                target.write_element_values(name, values, block_id=int(block_id), step=step)
            elif on is Entity.EDGE:
                target.write_edge_values(name, values, block_id=int(block_id), step=step)
            elif on is Entity.FACE:
                target.write_face_values(name, values, block_id=int(block_id), step=step)


def _copy_set_variable_history_at_step(
    source: Any,
    target: ExodusWriter,
    on: Entity,
    names: tuple[str, ...],
    set_ids: Any,
    time_index: int,
    step: int,
) -> None:
    for name in names:
        for set_id in set_ids:
            target.write_values(
                name,
                source.values(name, on=on, set_id=int(set_id), time=time_index),
                on=on,
                set_id=int(set_id),
                step=step,
            )


def _copy_status(source: Any, target: ExodusWriter) -> None:
    _copy_block_status(source, target, Entity.ELEMENT_BLOCK)
    _copy_block_status(source, target, Entity.EDGE_BLOCK)
    _copy_block_status(source, target, Entity.FACE_BLOCK)

    _copy_set_status(source, target, Entity.NODE_SET)
    _copy_set_status(source, target, Entity.SIDE_SET)
    _copy_set_status(source, target, Entity.EDGE_SET)
    _copy_set_status(source, target, Entity.FACE_SET)
    _copy_set_status(source, target, Entity.ELEMENT_SET)


def _copy_block_status(source: Any, target: ExodusWriter, entity: Entity) -> None:
    if not hasattr(source, "block_ids") or not hasattr(source, "block_is_active"):
        return

    for block_id in source.block_ids(entity):
        target.set_block_status(
            entity, int(block_id), bool(source.block_is_active(entity, int(block_id)))
        )


def _copy_set_status(source: Any, target: ExodusWriter, entity: Entity) -> None:
    if not hasattr(source, "set_ids") or not hasattr(source, "set_is_active"):
        return

    for set_id in source.set_ids(entity):
        target.set_set_status(entity, int(set_id), bool(source.set_is_active(entity, int(set_id))))


def _copy_records(source: Any, target: ExodusWriter) -> None:
    if hasattr(source, "info_records"):
        records = source.info_records()
        if records:
            target.write_info_records(records)

    if hasattr(source, "qa_records"):
        records = source.qa_records()
        if records:
            target.write_qa_records(records)


def _copy_properties(source: Any, target: ExodusWriter) -> None:
    for on in (
        Entity.ELEMENT_BLOCK,
        Entity.EDGE_BLOCK,
        Entity.FACE_BLOCK,
        Entity.NODE_SET,
        Entity.SIDE_SET,
        Entity.EDGE_SET,
        Entity.FACE_SET,
        Entity.ELEMENT_SET,
    ):
        if not hasattr(source, "property_names"):
            continue

        for name in source.property_names(on):
            if name.upper() == "ID":
                continue
            target.define_property(on, name, source.property_values(on, name))


def _copy_block_attributes(source: Any, target: ExodusWriter, on: Entity, block_id: int) -> None:
    if not hasattr(source, "attributes"):
        return

    attrs = source.attributes(on, block_id)
    if attrs is None:
        return

    names = source.attribute_names(on, block_id) if hasattr(source, "attribute_names") else None
    target.write_block_attributes(on, block_id, attrs, names=names)


def _has_variable(source: Any, name: str) -> bool:
    if hasattr(source, "variables"):
        variables = source.variables
        if callable(variables):
            return name in variables()
        return name in variables

    if hasattr(source, "backend"):
        return source.backend.has_variable(name)

    return False


__all__ = ["copy", "copy_file"]
