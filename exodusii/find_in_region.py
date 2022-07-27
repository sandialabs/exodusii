import logging
import numpy as np
from collections import OrderedDict as ordered_dict

from .file import exodusii_file
from .region import bounded_time_region
from .extension import compute_element_centers


def find_node_data_in_region(files, vars, region, time_region=None):
    """Finds element data in a finite element mesh region

    Parameters
    ----------
    files : list of str
        List of ExodusII files to read
    vars : list of str
        Node variables to read from files
    region : object
        An object implementing a `contains` method that takes as input an array of
        coordinates and returns a boolean array of the same length containing True if
        the point is in the region and False otherwise.
    time_region : callable
        Takes an array of times as input and returns a boolean array of the same
        length containg True if the time should be queried and False otherwise.

    Returns
    -------
    data : dict
        data[cycle] = cycle_data
        where cycle_data is a dictionary containing ndarrays for the nodal
        coordinates and each node variable.

    Examples
    --------
    >>> region = cylinder((0, 12.5e-6), (None, 12.5e-6), 6e-6)
    >>> time_region = bound_time_region(0, None)
    >>> vars = ("FORCEX", "FORCEY")
    >>> data = find_node_data_in_region(files, vars, region, time_reigon=time_region)
    >>> data[0]["time"]
    0.000
    >>> data[0]["cycle"]
    0
    >>> data[0]["FORCEX"]
    ndarray([3.4529e+3, ..., 5.3914e+4])

    """

    if isinstance(files, str):
        files = [files]

    data = ordered_dict()
    time_region = time_region or bounded_time_region(0, None)

    for (i, filename) in enumerate(files):

        logging.info(f"Processing file {i + 1} of {len(files)} files")

        file = exodusii_file(filename, mode="r")

        if i == 0:
            times = file.get_times()
            cycles = time_region(times).nonzero()[0]

        xc = file.get_coords()
        dimension = 1 if xc.ndim == 1 else xc.shape[1]
        if dimension != region.dimension:
            raise ValueError("Coordinate dimension does not match region dimension")

        ix = region.contains(xc)
        if not np.any(ix):
            continue

        for cycle in cycles:
            xd = [xc[ix]]
            for var in vars:
                elem_data = file.get_node_variable_values(var, time_step=cycle + 1)
                xd.append(elem_data[ix])
            xd = np.column_stack(xd)
            if cycle in data:
                data[cycle] = np.row_stack((data[cycle], xd))
            else:
                data[cycle] = xd

    for (cycle, xd) in data.items():
        # Sort by coordinate
        ix = np.argsort(xd[:, 0])
        cycle_data = {"cycle": cycle, "time": times[cycle]}
        for (i, dim) in enumerate("XYZ"[:dimension]):
            cycle_data[dim] = xd[ix, i]
        for (i, var) in enumerate(vars, start=dimension):
            cycle_data[var] = xd[ix, i]
        data[cycle] = cycle_data

    return data


def find_element_data_in_region(files, vars, region, time_region=None):
    """Finds element data in a finite element mesh region

    Parameters
    ----------
    files : list of str
        List of ExodusII files to read
    vars : list of str
        Element variables to read from files
    region : object
        An object implementing a `contains` method that takes as input an array of
        coordinates and returns a boolean array of the same length containing True if
        the point is in the region and False otherwise.
    tr : callable
        Takes an array of times as input and returns a boolean array of the same
        length containg True if the time should be queried and False otherwise.

    Returns
    -------
    data : dict
        data[cycle] = cycle_data
        where cycle_data is a dictionary containing ndarrays for the nodal
        coordinates and each node variable.

    Examples
    --------
    >>> region = cylinder((0, 12.5e-6), (None, 12.5e-6), 6e-6)
    >>> time_region = bound_time_region(0, None)
    >>> vars = ("BE_MAG", "VOID_FRC")
    >>> data = find_element_data_in_region(files, vars, region, time_reigon=time_region)
    >>> data[0]["time"]
    0.000
    >>> data[0]["cycle"]
    0
    >>> data[0]["BE_MAG"]
    ndarray([3.4529e+3, ..., 5.3914e+4])

    """

    if isinstance(files, str):
        files = [files]

    time_region = time_region or bounded_time_region(0, None)

    data = ordered_dict()
    for (i, filename) in enumerate(files):

        logging.info(f"Processing file {i + 1} of {len(files)} files")

        file = exodusii_file(filename, mode="r")

        if i == 0:
            times = file.get_times()
            cycles = time_region(times).nonzero()[0]

        block_ids = file.get_element_block_ids()
        for block_id in block_ids:
            num_elem = file.num_elems_in_blk(block_id)
            if not num_elem:
                continue
            xe = compute_element_centers(file, block_id)
            dimension = 1 if xe.ndim == 1 else xe.shape[1]
            if dimension != region.dimension:
                raise ValueError("Coordinate dimension does not match region dimension")

            ix = region.contains(xe)
            if not np.any(ix):
                continue
            for cycle in cycles:
                xd = [np.ones(len(ix.nonzero()[0])) * times[cycle], xe[ix]]
                for var in vars:
                    elem_data = file.get_element_variable_values(
                        block_id, var, time_step=cycle + 1
                    )
                    xd.append(elem_data[ix])
                xd = np.column_stack(xd)
                if cycle in data:
                    data[cycle] = np.row_stack((data[cycle], xd))
                else:
                    data[cycle] = xd

    for (cycle, xd) in data.items():
        # Sort by coordinate
        ix = np.argsort(xd[:, 0])
        cycle_data = {"cycle": cycle, "time": times[cycle]}
        for (i, dim) in enumerate("XYZ"[:dimension]):
            cycle_data[dim] = xd[ix, i]
        for (i, var) in enumerate(vars, start=dimension):
            cycle_data[var] = xd[ix, i]
        data[cycle] = cycle_data

    return data
