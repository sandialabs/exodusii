import numpy as np
import multiprocessing
from .util import compute_connected_average
from .element import factory as element_factory

__all__ = [
    "compute_element_centers",
    "compute_edge_centers",
    "compute_face_centers",
    "compute_element_length",
    "compute_element_volumes",
    "compute_volume_averaged_elem_variable",
    "compute_node_volumes",
    "compute_node_variable_values_at_element_center",
]


def compute_element_centers(file, block_id=None, time_step=None):
    """Computes the element geometric center.

    Parameters
    ----------
    block_id : int
        element block ID (not INDEX)
    time_step : int
        1-based index of time step

    Returns
    -------
    centers : ndarray of float

    Note
    ----
    If `time_step` is not None, the center of the element displacement is computed

    """
    if block_id is None:
        # compute centers for all blocks
        centers = []
        for id in file.get_element_block_ids():
            centers.append(compute_element_centers(file, id, time_step=time_step))
        return np.concatenate(centers, axis=0)
    else:
        conn = file.get_element_conn(block_id) - 1
        coords = file.get_coords(time_step=time_step)
        return compute_connected_average(conn, coords)


def compute_element_length(file, time):
    """Calculate the characteristic element length.

    The length of a 3D element is the cube root of the volume; for a 2D element
    it is the square root of the area.

    The characteristic element length is the average element length over the mesh.
    """

    time_step = file.get_time_step(time)

    ndim = file.num_dimensions()
    dexp = 1.0 / ndim

    length = 0.0
    for block_id in file.get_element_block_ids():
        vols = compute_element_volumes(file, block_id, time_step=time_step)
        length += np.sum(np.power(np.abs(vols), dexp))

    return length / file.num_elems()


def compute_edge_centers(file, block_id=None, time_step=None):
    """Computes the edge geometric center.

    Parameters
    ----------
    block_id : int
        edge block ID (not INDEX)
    time_step : int
        1-based index of time step

    Returns
    -------
    centers : ndarray of float

    Note
    ----
    If `time_step` is not None, the center of the edge displacement is computed

    """
    if block_id is None:
        # compute centers for all blocks
        centers = []
        for id in file.get_edge_block_ids():
            centers.append(compute_edge_centers(file, id, time_step=time_step))
        return np.concatenate(centers, axis=0)
    else:
        conn = file.get_edge_block_conn(block_id) - 1
        coords = file.get_coords(time_step=time_step)
        return compute_connected_average(conn, coords)


def compute_face_centers(file, block_id=None, time_step=None):
    """Computes the face geometric center.

    Parameters
    ----------
    block_id : int
        face block ID (not INDEX)
    time_step : int
        1-based index of time step

    Returns
    -------
    centers : ndarray of float

    Note
    ----
    If `time_step` is not None, the center of the face displacement is computed

    """
    if block_id is None:
        # compute centers for all blocks
        centers = []
        for id in file.get_face_block_ids():
            centers.append(compute_face_centers(file, id, time_step=time_step))
        return np.concatenate(centers, axis=0)
    else:
        conn = file.get_face_block_conn(block_id) - 1
        coords = file.get_coords(time_step=time_step)
        return compute_connected_average(conn, coords)


def compute_node_variable_values_at_element_center(
    file, block_id, var_name, time_step=None
):
    """Computes the value of a node variable at an element's center

    Parameters
    ----------
    block_id : int
        element block ID (not INDEX)
    var_name : str
        The nodal variable name
    time_step : int
        1-based index of time step

    Returns
    -------
    data : ndarray of float

    """
    if block_id is None:
        # compute for all blocks
        data = []
        for id in file.get_element_block_ids():
            x = compute_node_variable_values_at_element_center(
                file, id, var_name, time_step=time_step
            )
            data.append(x)
        return np.concatenate(data, axis=0)
    else:
        if var_name == "coordinates":
            nvars = file.get_coords()
        elif var_name == "displacements":
            nvars = file.get_displ(time_step)
        else:
            nvars = file.get_node_variable_values(var_name, time_step=time_step)
        conn = file.get_element_conn(block_id) - 1
        return compute_connected_average(conn, nvars)


def compute_node_volumes(file, time_step=None):
    """Get the node volumes at the time index specified. The time index is
    1-based. If provided, vol_array must be an array.array object of type
    storageType(), which is filled with the values; otherwise it is created."""

    # Basic strategy: For each block, first get the element volumes
    # then distribute the element volumes to the nodes.
    vol = np.zeros(file.num_nodes())

    for block_id in file.get_element_block_ids():
        blk = file.get_element_block(block_id)

        nodes_per_i = 1.0 / blk.num_elem_nodes

        # Get the element volumes for a block, then partition the volume
        # to the element's nodes
        element_volumes = compute_element_volumes(file, block_id, time_step)

        conn = file.get_element_conn(block_id) - 1
        # Now, partition the element volume and distribute it to the nodes.
        for element in range(blk.num_block_elems):
            node_vol_part = nodes_per_i * element_volumes[element]
            vol[conn[element]] += node_vol_part

    return vol


def compute_volume_averaged_elem_variable(
    file, block_id, time_step, func, intervals=5, zfill=None, processes=None
):
    """Get the cell-average of a variable for block block_id at time_step.

    If the exoobj mesh is 2D and zfill is provided, zfill is appended to the x
    and y values in restructured_coords for all nodes.

    """
    processes = processes or 1

    # Get the time that matches the solution time_index (which
    # might not be the same as the test_time)
    exact_time = file.get_time(time_step)

    elem_blk = file.get_element_block(block_id)
    elem_type = elem_blk.elem_type

    coord = file.get_coords(time_step=time_step)
    if file.num_dimensions() == 2 and zfill is not None:
        coord = np.column_stack((coord, np.zeros(coord.shape[0])))

    conn = file.get_element_conn(block_id) - 1
    if processes <= 2:
        averaged = _compute_ave(elem_type, func, exact_time, conn, coord, intervals)
    else:
        count = elem_blk.num_block_elems
        nproc = processes - 1
        pipes = [(None, None) for i in range(nproc)]
        procs = [None for i in range(nproc)]
        for procno in range(nproc):
            start = int((procno * count) / nproc)
            end = int(((procno + 1) * count) / nproc)
            pipes[procno] = multiprocessing.Pipe(False)
            p = multiprocessing.Process(
                target=_compute_ave,
                args=(
                    elem_type,
                    func,
                    exact_time,
                    conn[start:end],
                    coord,
                    intervals,
                    pipes[procno][1],
                ),
            )
            procs[procno] = p
            p.start()
        averaged = np.zeros(count)
        for procno in range(nproc):
            p = procs[procno]
            start = int((procno * count) / nproc)
            end = int(((procno + 1) * count) / nproc)
            pipe = pipes[procno][0]
            averaged[start:end] = pipe.recv()
            pipe.close()
            p.join()

    return averaged


def _compute_ave(elem_type, fun, time, conn, coord, intervals, pipe=None):
    averaged = np.zeros(len(conn))
    for (iel, ix) in enumerate(conn):
        el = element_factory(elem_type, coord[ix])
        centers = el.subdiv(intervals)
        vols = el.subvols(intervals)
        exact = np.array([fun(x, time) for x in centers])
        averaged[iel] = np.sum(vols * exact) / np.sum(vols)
    if pipe is None:
        return averaged
    else:
        pipe.send(averaged)
        pipe.close()


def compute_element_volumes(file, block_id, time_step=None):
    """Computes the element volumes.

    Parameters
    ----------
    block_id : int
        element block ID (not INDEX)
    time_step : int
        1-based index of time step

    Returns
    -------
    volumes : ndarray of float

    Note
    ----
    If `time_step` is not None, the volume of the displaced element displacement

    """
    coords = file.get_coords(time_step=time_step)
    elem_blk = file.get_element_block(block_id)
    efactory = lambda x: element_factory(elem_blk.elem_type, x)

    # Connectivity is 1 based
    conn = file.get_element_conn(block_id) - 1

    # Now, compute the volumes.
    vol = np.zeros(elem_blk.num_block_elems)
    for (iel, ix) in enumerate(conn):
        el = efactory(coords[ix])
        vol[iel] = el.volume

    return vol
