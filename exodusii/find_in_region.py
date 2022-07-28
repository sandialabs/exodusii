import logging
import numpy as np
from collections import OrderedDict as ordered_dict

from .file import exodusii_file
from .region import unbounded_time_domain
from .extension import compute_element_centers
from .parallel_file import parallel_exodusii_file


def find_node_data_in_region(
    files, vars, region, time_domain=None, use_displaced_coords=False
):
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
    time_domain : object
        An object implementing a `contains` method that take an array of times
        as input and returns a boolean array of the same length containg True if
        the time should be queried and False otherwise.
    use_displaced_coords : bool
        Use displaced coordinates for determining geometric region

    Returns
    -------
    data : dict
        data[cycle] = cycle_data
        where cycle_data is a dictionary containing ndarrays for the nodal
        coordinates and each node variable.

        cycle_data["cycle"] = int
        cycle_data["time"] = float
        cycle_data["X"] = ndarray
        cycle_data["Y"] = ndarray
        cycle_data["Z"] = ndarray  [only if 3d]
        cycle_data["var1"] = ndarray
        ...
        cycle_data["varn"] = ndarray

    Examples
    --------
    >>> region = cylinder((0, 12.5e-6), (None, 12.5e-6), 6e-6)
    >>> time_domain = bound_time_domain(0, None)
    >>> vars = ("FORCEX", "FORCEY")
    >>> data = find_node_data_in_region(files, vars, region, time_domain=time_domain)
    >>> data[0]["time"]
    0.000
    >>> data[0]["cycle"]
    0
    >>> data[0]["FORCEX"]
    ndarray([3.4529e+3, ..., 5.3914e+4])

    """

    if isinstance(files, (str, exodusii_file, parallel_exodusii_file)):
        files = [files]

    data = ordered_dict()
    time_domain = time_domain or unbounded_time_domain()

    for (i, file) in enumerate(files):

        logging.info(f"Processing file {i + 1} of {len(files)} files")

        if not isinstance(file, (exodusii_file, parallel_exodusii_file)):
            file = exodusii_file(file, mode="r")

        if i == 0:
            times = file.get_times()
            cycles = time_domain.contains(times).nonzero()[0]

        if not use_displaced_coords:
            # Precompute element centers for all cycles
            xc = file.get_coords()
            dimension = 1 if xc.ndim == 1 else xc.shape[1]
            if dimension != region.dimension:
                raise ValueError("Coordinate dimension does not match region")
            ix = region.contains(xc)
            if not np.any(ix):
                continue

        for cycle in cycles:

            if use_displaced_coords:
                xc = file.get_coords(time_step=cycle + 1)
                dimension = 1 if xc.ndim == 1 else xc.shape[1]
                if dimension != region.dimension:
                    raise ValueError("Coordinate dimension does not match region")
                ix = region.contains(xc)
                if not np.any(ix):
                    continue

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


def find_element_data_in_region(
    files, vars, region, time_domain=None, block_ids=None, use_displaced_coords=False
):
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
    time_domain : object
        An object implementing a `contains` method that take an array of times
        as input and returns a boolean array of the same length containg True if
        the time should be queried and False otherwise.
    block_ids : list of int
        Get element data only from these blocks.  If None, get data from all blocks
    use_displaced_coords : bool
        Use displaced coordinates for determining geometric region

    Returns
    -------
    data : dict
        data[cycle] = cycle_data
        where cycle_data is a dictionary containing ndarrays for the nodal
        coordinates and each node variable.

        cycle_data["cycle"] = int
        cycle_data["time"] = float
        cycle_data["X"] = ndarray
        cycle_data["Y"] = ndarray
        cycle_data["Z"] = ndarray  [only if 3d]
        cycle_data["var1"] = ndarray
        ...
        cycle_data["varn"] = ndarray

    Examples
    --------
    >>> region = cylinder((0, 12.5e-6), (None, 12.5e-6), 6e-6)
    >>> time_domain = bound_time_domain(0, None)
    >>> vars = ("BE_MAG", "VOID_FRC")
    >>> data = find_element_data_in_region(files, vars, region, time_domain=time_domain)
    >>> data[0]["time"]
    0.000
    >>> data[0]["cycle"]
    0
    >>> data[0]["BE_MAG"]
    ndarray([3.4529e+3, ..., 5.3914e+4])

    """

    if isinstance(files, (str, exodusii_file, parallel_exodusii_file)):
        files = [files]

    _block_ids = block_ids
    time_domain = time_domain or unbounded_time_domain()

    data = ordered_dict()
    for (i, file) in enumerate(files):

        logging.info(f"Processing file {i + 1} of {len(files)} files")

        if not isinstance(file, (exodusii_file, parallel_exodusii_file)):
            file = exodusii_file(file, mode="r")

        if i == 0:
            times = file.get_times()
            cycles = time_domain.contains(times).nonzero()[0]

        block_ids = file.get_element_block_ids() if _block_ids is None else _block_ids
        for block_id in block_ids:
            num_elem = file.num_elems_in_blk(block_id)
            if not num_elem:
                continue

            if not use_displaced_coords:
                # Precompute element centers for all cycles
                xe = compute_element_centers(file, block_id)
                dimension = 1 if xe.ndim == 1 else xe.shape[1]
                if dimension != region.dimension:
                    raise ValueError("Coordinate dimension does not match region")
                ix = region.contains(xe)
                if not np.any(ix):
                    continue

            for cycle in cycles:

                if use_displaced_coords:
                    xe = compute_element_centers(file, block_id, time_step=cycle + 1)
                    dimension = 1 if xe.ndim == 1 else xe.shape[1]
                    if dimension != region.dimension:
                        raise ValueError("Coordinate dimension does not match region")
                    ix = region.contains(xe)
                    if not np.any(ix):
                        continue

                xd = [xe[ix]]
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
