import os
import sys
import logging
import numpy as np
from functools import wraps
from types import SimpleNamespace

from . import nc
from .util import stringify, streamify, find_index, check_bounds, find_nearest, contains
from .ex_params import ex_init_params
from .extension import compute_node_variable_values_at_element_center
from .util import index
from . import exodus_h as ex


def requires_write_mode(fun):
    @wraps(fun)
    def inner(self, *args, **kwargs):
        if self.mode not in ("w", "a"):
            raise UnsupportedOperation(f"{self.filename}: not writable")
        return fun(self, *args, **kwargs)

    return inner


class exodusii_file:
    """
    A file object for Exodus data.

    Exodus is a model developed to store and retrieve data for finite element
    analyses. It is used for preprocessing (problem definition), postprocessing
    (results visualization), as well as code to code data transfer. An Exodus
    data file is a random access, machine independent, binary file that is
    written and read via C, C++, or Fortran library routines which comprise the
    Application Programming Interface. Exodus uses NetCDF Library as the on-disk
    storage format.  See
    https://gsjaardema.github.io/seacas-docs/sphinx/html/index.html for the
    official Exodus documentation.

    Parameters
    ----------
    filename : string or file-like
        string -> filename
    mode : {'r', 'w', 'a'}, optional
        read-write-append mode, default is 'r'

    Notes
    -----
    Exodus provides its own Python interface, documented at
    https://gsjaardema.github.io/seacas-docs/exodus.html, called `exodus.py`.
    `exodus.py` accessing file data using the Exodus library and requires,
    therefore, building the Exodus shared object libraries.  This interface
    strives to be API compatible with `exodus.py` but differs by accessing the
    underlying NetCDF objects directly and does not require installing the
    Exodus libraries.

    Examples
    --------
    To create a ExodusII file:

    >>> import sys
    >>> import exodusii
    >>> f = exodusii.exodusii_file('file.exo', 'r')
    >>> f.describe(file=sys.stdout)
    ...

    """  # noqa: E501

    def __init__(self, filename, mode="r"):
        assert mode in "raw"
        self.mode = mode
        self.filename = filename
        if self.mode == "r":
            if not os.path.isfile(self.filename):
                raise FileNotFoundError(self.filename)
        self.files = [self.open(self.filename, mode=self.mode)]

        if self.mode == "w":
            self.initw()

        self.exinit = ex_init_params(self.files[0])

    def __repr__(self):
        files = ",".join([self._filename(f) for f in self.files])
        return f"exodusii_file({files})"

    def __contains__(self, name):
        return name in self.files[0].variables

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @property
    def fh(self):
        return self.files[0]

    def _filename(self, file):
        return nc.filename(file)

    def get_filename(self, file):
        return nc.filename(file)

    def _open(self, filename, mode):
        return nc.open(filename, mode=mode)

    def open(self, filename, mode="r"):
        return self._open(filename, mode)

    def close(self):
        if self.mode == "w":
            return nc.close(self.files[0])

    def is_edge_variable(self, variable):
        variables = self.get_edge_variable_names()
        if variables is None:
            return False
        return variable in variables

    def is_face_variable(self, variable):
        variables = self.get_face_variable_names()
        if variables is None:
            return False
        return variable in variables

    def is_element_variable(self, variable):
        variables = self.get_element_variable_names()
        if variables is None:
            return False
        return variable in variables

    def is_node_variable(self, variable):
        variables = self.get_node_variable_names()
        if variables is None:
            return False
        return variable in variables

    def _get_dimension(self, file, name, default):
        return nc.get_dimension(file, name, default=default)

    def get_dimension(self, name, default=None):
        return self._get_dimension(self.files[0], name, default)

    def _get_variable(self, file, name, default, raw):
        return nc.get_variable(file, name, default=default, raw=raw)

    def get_variable(self, name, default=None, raw=False):
        return self._get_variable(self.files[0], name, default=default, raw=raw)

    def variables(self):
        return self.files[0].variables

    def dimensions(self):
        return self.files[0].dimensions

    def variable_names(self):
        return list(self.variables().keys())

    def dimension_names(self):
        return list(self.dimensions().keys())

    def storage_type(self):
        word_size = self.files[0].floating_point_word_size
        return "f" if word_size == 4 else "d"

    def parse_variable_args(self, args, lineout, namespace=None):
        """Parse variable inputs to File.get

        Parameters
        ----------
        args : list

        Returns
        -------
        namespace : Namespace
          namspace.variables[i] is (TYPE, VARNAME) of the ith requested
          variable.

        """
        namespace = namespace or SimpleNamespace()
        namespace.variables = []
        types = set()
        for arg in args:
            if arg in ("displacements", "coordinates"):
                namespace.variables.append(arg)
                continue
            type, *remainder = arg.split("/")
            if type not in "gefdn":
                raise ValueError(f"Unexpected data type in {arg}")
            types.add(type)
            if len(remainder) != 1:
                raise ValueError(
                    f"Unexpected variable format {arg}, expected {type}/NAME"
                )
            namespace.variables.append(remainder[0])
        if len(types) > 1:
            types = ", ".join(list(types))
            raise ValueError(f"Types {types} are mutually exclusive")
        namespace.variable_type = list(types)[0]
        if lineout is not None and namespace.variable_type != "g":
            namespace.variables.insert(0, "coordinates")
            if lineout.needs_displacements:
                namespace.variables.insert(0, "displacements")
        return namespace

    def parse_history_args(self, time, cycle, index, namespace=None):
        namespace = namespace or SimpleNamespace()
        if len([1 for _ in (time, cycle, index) if _ is not None]) > 1:
            raise ValueError("time, cycle, and index are mutually exclusive")
        times = self.get_times()
        if time is not None:
            if not check_bounds(times, time):
                raise ValueError(
                    f"Time {time} is outside the bounds in {self.filename}"
                )
            index, _ = find_nearest(times, time)
        elif cycle is not None:
            # first, need the global variable for storing the cycle number
            names = self.get_global_variable_names()
            for name in ["cycle", "CYCLE", "nsteps", "NSTEPS"]:
                if name in names:
                    break
            else:
                raise ValueError("Could not find cycle number variable")
            # then search the cycles for a match
            cycles = self.get_global_variable_values(name)
            if not contains(cycles, cycle):
                raise ValueError(f"Cycle {cycle} not found in {self.filename}")
            index, _ = find_nearest(cycles, cycle)
        elif index is not None:
            if index == -1:
                index = len(times) - 1
            else:
                if not contains(range(len(times)), index):
                    raise ValueError(f"Index {index} not found in {self.filename}")
        namespace.index = index
        return namespace

    def describe(self, file=None):
        """Writes out the number of objects and the variable names to the text
        stream `file`.

        Parameters
        ----------
        file : str or file-object
            Text stream.  If a `str`, a file with that name will be opened in `w` mode.

        """

        stream, fown = streamify(file or sys.stdout)

        stream = stream or sys.stdout
        stream.write(f"Title: {self.title()}\n")
        stream.write(f"Storage type: {self.storage_type()}\n")
        stream.write("Num info strings: 0\n")  # FIXME
        stream.write(f"Dimension: {self.num_dimensions()}\n")
        stream.write(f"Num nodes   : {self.num_nodes()}\n")
        stream.write(f"Num edges   : {self.num_edges()}\n")
        stream.write(f"Num faces   : {self.num_faces()}\n")
        stream.write(f"Num elements: {self.num_elems()}\n")

        self._summarize_block("elem", self.get_element_block_ids(), stream)
        self._summarize_block("face", self.get_face_block_ids(), stream)  # FIXME
        self._summarize_block("edge", self.get_edge_block_ids(), stream)  # FIXME
        self._summarize_set("node", self.get_node_set_ids(), stream)
        self._summarize_set("side", self.get_side_set_ids(), stream)
        self._summarize_set("edge", self.get_edge_set_ids(), stream)  # FIXME
        self._summarize_set("face", self.get_face_set_ids(), stream)  # FIXME
        self._summarize_set("elem", self.get_element_set_ids(), stream)  # FIXME

        self._summarize_vars("global", self.get_global_variable_names(), stream)
        self._summarize_vars("elem", self.get_element_variable_names(), stream)
        self._summarize_vars("face", self.get_face_variable_names(), stream)  # FIXME
        self._summarize_vars("edge", self.get_edge_variable_names(), stream)  # FIXME
        self._summarize_vars("node", self.get_node_variable_names(), stream)
        self._summarize_vars("node set", self.get_node_set_names(), stream)  # FIXME
        self._summarize_vars("side set", self.get_side_set_names(), stream)  # FIXME
        self._summarize_vars("edge set", self.get_edge_set_names(), stream)  # FIXME
        self._summarize_vars("face set", self.get_face_set_names(), stream)  # FIXME
        self._summarize_vars("elem set", self.get_element_set_names(), stream)  # FIXME

        times = self.get_times()
        stream.write(f"Time steps: {len(times)}\n")
        for (i, time) in enumerate(times, start=1):
            stream.write(f"  {i} {time}\n")

    def _summarize_set(self, name, items, stream, prefix="{name:s} sets: {num:d}"):
        num = 0 if items is None else len(items)
        stream.write(prefix.format(name=name.title(), num=num))
        if num:
            ids = " ".join(str(_) for _ in items)
            stream.write(f" Ids = {ids}")
        stream.write("\n")

    def _summarize_block(self, name, items, stream):
        self._summarize_set(name, items, stream, prefix="{name:s} blocks: {num:d}")

    def _summarize_vars(self, entity, names, stream):
        n = 0 if names is None else len(names)
        stream.write(f"{entity.title()} vars: {n}\n")
        if names is not None:
            for (i, name) in enumerate(names):
                stream.write(f"  {i} {name}\n")

    def print(
        self,
        *variables,
        time=None,
        cycle=None,
        index=None,
        file=None,
        labels=True,
        lineout=None,
    ):
        data = self.get(
            *variables, time=time, cycle=cycle, index=index, lineout=lineout
        )
        stream, fown = streamify(file or sys.stdout)
        names = data.dtype.names
        if labels:
            labels = [data.dtype.metadata["map"].get(_, _) for _ in names]
            row = " " + " ".join([f"{_:>23s}" for _ in labels])
            stream.write(row + "\n")
        for row in data:
            values = [row[name] for name in names]
            line = " " + " ".join([f"{value: 20.16e}" for value in values])
            stream.write(line + "\n")

        if fown:
            stream.close()

    def get(
        self,
        *variables,
        time=None,
        index=None,
        cycle=None,
        lineout=None,
    ):
        """Finds the variables in the file and returns the values in a list of rows,
        where each row is a list of values.  The first row contains the labels.

        Parameters
        ----------
        variables : list of str
            a list of variable names specified as TYPE/NAME where TYPE is one of
            g, n, e, f, d

        time : float
            The time slab to extract from the file
        index : int
            The output index to extract
        cycle : int
            The output cycle to extract

        Notes
        -----
        1. The returned table is of the form:

             [ time, value1, value2, ... ]

           for global variables, or

             [ value1, value2, ... ]

           for spatial variables.

        2. There are two special variable names:
           - `coordinates`: reads the nodal coordinates for type 'n' or computes the
             centers for types 'e', 'f', and 'd'
           - `displacements`: reads the nodal displacements for type 'n' or computes the
             average dislacements for types 'e', 'f', and 'd'

        Examples
        --------
        >>> f = exodusii.exodusii_file()
        sxx = f.get

        """

        times = self.get_times()
        if len(times) == 0:
            raise ValueError("no time steps found in file")

        args = self.parse_variable_args(variables, lineout)
        args = self.parse_history_args(time, cycle, index, namespace=args)

        if args.variable_type == "g":
            available_var_names = self.get_global_variable_names()
        elif args.variable_type == "e":
            available_var_names = self.get_element_variable_names()
        elif args.variable_type == "f":
            available_var_names = self.get_face_variable_names()
        elif args.variable_type == "d":
            available_var_names = self.get_edge_variable_names()
        else:
            available_var_names = self.get_node_variable_names()

        names = []  # variable name list (as stored in the file)
        metadata = {"map": {}}
        for (i, variable) in enumerate(args.variables):

            if args.variable_type != "g" and variable == "coordinates":
                name = "coordinates"

            elif args.variable_type != "g" and variable == "displacements":
                name = "displacements"

            elif args.variable_type == "n" and variable.lower() in ("x", "y", "z"):
                name = variable
                metadata["map"][f"n/{variable}"] = name

            else:
                idx = find_index(available_var_names, variable, strict=False)
                if idx is None:
                    avail = ", ".join(available_var_names)
                    raise ValueError(
                        f"{variable!r} not found in available names {avail}"
                    )
                name = available_var_names[idx]
                metadata["map"][f"{args.variable_type}/{variable}"] = name
            names.append(name)

        vals = None
        if args.variable_type == "g":
            names.insert(0, "TIME")
            data = np.column_stack(
                [times] + [self.get_global_variable_values(name) for name in names[1:]]
            )
            if args.index is not None:
                data = np.expand_dims(data[args.index], axis=0)
        else:
            coor_names = [_.upper() for _ in self.get_coord_variable_names()]
            displ_names = self.get_displ_variable_names(default=[])
            # if no time slab specified, select the last one
            if args.index is None:
                args.index = len(times) - 1
            data = []
            metadata["TIME"] = times[args.index]
            for name in names:
                if name in ("coordinates", "displacements"):
                    i = names.index(name)
                    if args.variable_type == "n":
                        vals = (
                            self.get_coords()
                            if name == "coordinates"
                            else self.get_displ(args.index + 1)
                        )
                    else:
                        vals = compute_node_variable_values_at_element_center(
                            self,
                            None,
                            name,
                            time_step=None if name == "coordinates" else args.index + 1,
                        )
                    data.extend([_ for _ in vals.T])
                    subs = coor_names if name == "coordinates" else displ_names
                    names = names[:i] + subs + names[i + 1 :]
                    continue
                elif args.variable_type == "n" and name.lower() in ("x", "y", "z"):
                    x = self.get_coords()
                    if name in "XYZ":
                        x += self.get_displ(args.index + 1)
                    vals = x[:, "xyz".index(name.lower())]
                elif args.variable_type == "e":
                    vals = self.get_element_variable_values(None, name, args.index + 1)
                elif args.variable_type == "f":
                    vals = self.get_face_variable_values(None, name, args.index + 1)
                elif args.variable_type == "d":
                    vals = self.get_edge_variable_values(None, name, args.index + 1)
                elif args.variable_type == "n":
                    vals = self.get_node_variable_values(name, args.index + 1)
                data.append(vals)

            data = np.column_stack(data)

        if lineout is not None:
            names, data = lineout.apply(names, data)

        map = dict([(b, a) for (a, b) in metadata["map"].items()])
        if args.index is not None:
            metadata["@time"] = times[args.index]
        formats = [(map.get(name, name), "f8") for name in names]
        dtype = np.dtype(formats, metadata=metadata)
        exodata = np.array(list(zip(*data.T)), dtype=dtype)

        return exodata

    def title(self):
        """Get the database title

        Returns
        -------
        title : str
        """
        return stringify(self.files[0].title)

    def version_num(self):
        """Get exodus version number used to create the database

        Returns
        -------
        version : str
            string representation of version number
        """
        return f"{self.files[0].version:1.3}"

    def _get_element_info(self, elem_id):
        """Get the element block ID and element index within the block for elem_id

        Note
        ----
        This is an implementation detail.

        """
        start = 0
        elem_num_map = self.get_element_id_map()
        for block_id in self.get_element_block_ids():
            end = start + self.num_elems_in_blk(block_id)
            if elem_id in elem_num_map[start:end]:
                elem_iid = self.get_iid(elem_num_map[start:end], elem_id)
                block_iid = self.get_element_block_iid(block_id)
                return SimpleNamespace(
                    id=elem_id, iid=elem_iid, blk_id=block_id, blk_iid=block_iid
                )
            start = end
        raise ValueError(f"Unable to determine element block for element {elem_id}")

    @requires_write_mode
    def create_dimension(self, name, value):
        return nc.create_dimension(self.fh, name, value)

    @requires_write_mode
    def create_variable(self, id, type, shape):
        return nc.create_variable(self.fh, id, type, shape)

    @requires_write_mode
    def fill_variable(self, name, *args):
        return nc.fill_variable(self.fh, name, *args)

    def get_all_global_variable_values(self, time_step=None):
        """Get all global variable values (one for each global variable
        name, and in the order given by exo.get_global_variable_names())
        at a specified time step

        Parameters
        ----------
        time_step : int
            1-based index of time step

        Returns
        -------
        gvar_vals : ndarray of float
        """
        values = self.get_variable(ex.VAR_GLO_VAR)
        if values is None:
            return None
        if time_step is None:
            return values
        return values[time_step - 1]

    def get_all_node_set_params(self):
        """Get total number of nodes and distribution factors (e.g. nodal
        'weights') combined among all node sets

        Returns
        -------
        tot_num_nodes : int
        tot_num_dist_facts : int
        """
        raise NotImplementedError

    def get_all_side_set_params(self):
        """Get total number of sides, nodes, and distribution factors
        (e.g. nodal 'weights') combined among all side sets

        Returns
        -------
        tot_num_ss_sides : int
        tot_num_ss_nodes : int
        tot_num_ss_dist_facts : int

        Note:
        -----
        The number of nodes (and distribution factors) in a side set is
        the sum of all face nodes.  A single node can be counted more
        than once, i.e. once for each face it belongs to in the side set.
        """
        raise NotImplementedError

    def get_coord(self, node_num):
        """Get model coordinates of a single node

        Parameters
        ----------
        node_num : int
            the 1-based node index (indexing is from 1 to exo.num_nodes())

        Returns
        -------
        x_coord : double
            global x-direction coordinate
        y_coord : double
            global y-direction coordinate
        z_coord : double
            global z-direction coordinate

        """
        i = node_num if node_num < 0 else node_num - 1
        coords = self.get_coords()
        return coords[i]

    def get_coord_names(self):
        """Get a list of length exo.num_dimensions() that has the name of each model
        coordinate direction

        Returns
        -------
        coord_names : list of str

        """
        return self.get_variable(ex.VAR_NAME_COORD, default=list("xyz"))

    def get_coord_variable_names(self):
        """Get a list of length exo.num_dimensions() that has the name of each model
        coordinate variable name

        Returns
        -------
        coord_names : list of str

        """
        coor_names = self.get_coord_names()
        if "coor" in coor_names[0]:
            candidates = coor_names
        else:
            candidates = [f"{ex.VAR_COORD}{_.lower()}" for _ in coor_names]
        variables = self.variables()
        names = [_ for _ in candidates if _.lower() in variables]
        if len(names) != self.num_dimensions():
            raise ValueError("Incorrect number of coord names found")
        return names

    def get_coords(self, time_step=None):
        """Get model coordinates of all nodes; for each coordinate direction, a
        length exo.num_nodes() list is returned

        Returns
        -------
        coords : ndarray of float
        """
        coord_names = self.get_coord_variable_names()
        coords = np.column_stack([self.get_variable(_) for _ in coord_names])
        if time_step is not None:
            displ = self.get_displ(time_step)
            if displ is not None:
                coords += displ
        return coords

    def get_displ(self, time_step, default=None):
        """Get model displacements of all nodes; for each coordinate direction

        Parameters
        ----------
        time_step : int
            1-based index of time step
        default : Any
            Object to return in the case that no displacement names are found.
            Default is `None`.

        Returns
        -------
        displ : ndarray of float
        """
        displ_names = self.get_displ_variable_names()
        if not displ_names:
            return default
        displ = [self.get_node_variable_values(_, time_step) for _ in displ_names]
        return np.column_stack(displ)

    def get_displ_variable_names(self, default=None):
        bases = ["displ", "disp", "displ_"]
        candidates = [f"{base}{x}" for base in bases for x in "xyz"]
        names = [_ for _ in self.get_node_variable_names() if _.lower() in candidates]
        if not names:
            return default
        if len(names) != self.num_dimensions():
            raise ValueError("Incorrect number of displ names found")
        return names

    def get_edge_block_id(self, blk_iid):
        """get exodus edge block id from internal id (1 based index)

        Returns
        -------
        blk_iid : int

        """
        ids = self.get_edge_block_ids()
        return ids[blk_iid - 1]

    def get_edge_block_ids(self):
        """get mapping of exodus edge block index to user - or application-defined
        edge block id.

        edge_block_ids is ordered by the edge block INDEX ordering,
        a 1-based system going from 1 to exo.num_blks(), used by exodus for storage
        and input/output of array data stored on the edge blocks a user or
        application can optionally use a separate edge block ID numbering system,
        so the edge_block_ids array points to the edge block ID for each edge
        block INDEX

        Returns
        -------
        edge_block_ids : ndarray of int

        """
        return self.get_variable(ex.VAR_ID_EDGE_BLK)

    def get_edge_block_iid(self, edge_block_id):
        """get exodus edge block index from id

        Returns
        -------
        edge_block_id : int

        """
        ids = self.get_edge_block_ids()
        return self.get_iid(ids, edge_block_id)

    def get_edge_block(self, block_id):
        """Get the edge block info

        Parameters
        ----------
        block_id : int
            Edge block ID (not INDEX)

        Returns
        -------
        elem_type : str
            Element type, e.g. 'HEX8'
        num_block_edges : int
            number of edges in the block
        num_edge_nodes : int
            number of nodes per edge
        num_edge_attrs : int
            number of attributes per edge
        """
        if block_id not in self.get_edge_block_ids():
            raise None
        elem_type = self.get_edge_block_elem_type(block_id)
        if elem_type is None:
            return None
        info = SimpleNamespace(
            id=block_id,
            iid=self.get_edge_block_iid(block_id),
            elem_type=elem_type,
            num_block_edges=self.num_edges_in_blk(block_id),
            num_edge_nodes=self.num_nodes_per_edge(block_id),
            num_edge_attrs=self.num_edge_attr(block_id),
        )
        return info

    def get_edge_block_name(self, edge_block_id):
        """Get the edge block name

        Parameters
        ----------
        edge_block_id : int
            edge block *ID* (not *INDEX*)

        Returns
        -------
        edge_block_name : string
        """
        block_ids = self.get_edge_block_ids()
        blk_iid = self.get_iid(block_ids, edge_block_id)
        names = self.get_edge_block_names()
        return names[blk_iid - 1]

    def get_edge_block_names(self):
        """Get a list of all edge block names ordered by block *INDEX*; (see
        `exodus.get_ids` for explanation of the difference between block *ID* and
        block *INDEX*)

        Returns
        -------
        edge_block_names : list of str
        """
        return self.get_variable(ex.VAR_NAME_EDGE_BLK, default=np.array([], dtype=str))

    def get_edge_block_conn(self, block_id):
        """Get the nodal connectivity for a single edge block

        Parameters
        ----------
        block_id : int
            edge block *ID* (not *INDEX*)

        Returns
        -------
        edge_conn : ndarray of int
            define the connectivity of each edge in the block; the list cycles
            through all nodes of the first edge, then all nodes of the second
            edge, etc. (see `exodus.get_id_map` for explanation of node *INDEX*
            versus node *ID*)

        """
        block_iid = self.get_edge_block_iid(block_id)
        return self.get_variable(ex.VAR_EDGE_BLK_CONN(block_iid))

    def get_edge_id_map(self):
        """Get mapping of exodus edge index to user- or application- defined
        edge id; edge_id_map is ordered by the edgeent *INDEX* ordering, a 1-based
        system going from 1 to exo.num_edges(), used by exodus for storage and
        input/output of array data stored on the edgeents; a user or application can
        optionally use a separate edgeent *ID* numbering system, so the edge_id_map
        points to the edgeent *ID* for each edgeent *INDEX*

        Returns
        -------
        edge_id_map : ndarray of int

        """
        map = self.get_variable(ex.VAR_EDGE_NUM_MAP)
        if map is None:
            map = np.arange(self.num_edges(), dtype=int) + 1
        return map

    def get_edge_variable_truth_table(self, block_id=None):
        """gets a truth table indicating which variables are defined for a block; if
        block_id is not passed, then a concatenated truth table for all blocks is
        returned with variable index cycling faster than block index

        Parameters
        ----------
        block_id : int
            edge block *ID* (not *INDEX*)

        Returns
        -------
        evar_truth_tab : list of bool
            True for variable defined in block, False otherwise

        """
        truth_table = self.get_variable(ex.VAR_EDGE_BLK_TAB)
        if block_id is not None:
            block_iid = self.get_edge_block_iid(block_id)
            truth_table = truth_table[block_iid - 1]
        return truth_table

    def get_edge_set(self, set_id):
        ns = self.get_edge_set_params(set_id)
        ns.id = set_id
        ns.name = self.get_edge_set_name(set_id)
        ns.edges = self.get_edge_set_edges(set_id)
        ns.dist_facts = self.get_edge_set_dist_facts(set_id)
        num_df = None if ns.dist_facts is None else len(ns.dist_facts)
        if num_df is not None:
            assert num_df == ns.num_nodes
        return ns

    def get_edge_set_iid(self, set_id):
        ids = self.get_edge_set_ids()
        return index(ids, set_id)

    def get_edge_set_ids(self):
        """Get mapping of exodus edge set index to user- or application- defined edge
        set id; edge_set_ids is ordered by the *INDEX* ordering, a 1-based system
        going from 1 to exo.num_edge_sets(), used by exodus for storage and
        input/output of array data stored on the edge sets; a user or application can
        optionally use a separate edge set *ID* numbering system, so the edge_set_ids
        array points to the edge set *ID* for each edge set *INDEX*

        Returns
        -------
        edge_set_ids : ndarray of int
        """
        return self.get_variable(ex.VAR_EDGE_SET_IDS)

    def get_edge_set_name(self, set_id):
        """get the name of a edge set

        Parameters
        ----------
        set_id : int
            edge set *ID* (not *INDEX*)

        Returns
        -------
        edge_set_name : str
        """
        set_ids = self.get_edge_set_ids()
        set_iid = self.get_iid(set_ids, set_id)
        names = self.get_edge_set_names()
        return names[set_iid - 1]

    def get_edge_set_names(self):
        """Get a list of all edge set names ordered by edge set *INDEX*; (see
        description of get_edge_set_ids() for explanation of the difference between
        edge set *ID* and edge set *INDEX*)

        Returns
        -------
        edge_set_names : list of str
        """
        return self.get_variable(ex.VAR_NAME_EDGE_SET, default=np.array([], dtype=str))

    def get_edge_set_edges(self, set_id):
        """Get the list of edge *INDICES* in a edge set (see `exodus.get_id_map` for
        explanation of edge *INDEX* versus edge *ID*)

        Parameters
        ----------
        set_id : int
            edge set *ID* (not *INDEX*)

        Returns
        -------
        es_edges : ndarray of int

        """
        set_iid = self.get_edge_set_iid(set_id)
        return self.get_variable(ex.VAR_EDGE_EDGE_SET(set_iid))

    def get_edge_set_params(self, set_id):
        """Get number of edges and distribution factors (e.g. nodal 'weights') in a
        edge set

        Parameters
        ----------
        set_id : int
            edge set *ID* (not *INDEX*)

        Returns
        -------
        num_es_edges : int
        num_es_dist_facts : int
        """
        if set_id not in self.get_edge_set_ids():
            raise ValueError(f"{set_id} is not a valid edge set ID")
        info = SimpleNamespace(
            num_edges=self.num_edges_in_edge_set(set_id),
            num_nodes_per_edge=self.num_nodes_per_edge(set_id),
            num_dist_facts=self.num_edge_set_dist_fact(set_id),
        )
        return info

    def get_edge_set_property_names(self):
        """Get the list of edge set property names for all edge sets in the model

        Returns
        -------
        nsprop_names : list of str
        """
        raise NotImplementedError

    def get_edge_set_property_value(self, set_id, name):
        """Get edge set property value (an integer) for a specified edge
        set and edge set property name

        Parameters
        ----------
        set_id: int
            edge set *ID* (not *INDEX*)
        nsprop_name : string

        Returns
        -------
        nsprop_val : int
        """
        raise NotImplementedError

    def get_edge_set_variable_names(self):
        """Get the list of edge set variable names in the model

        Returns
        -------
        var_names : list of str
        """
        return self.get_variable(
            ex.VAR_NAME_EDGE_SET_VAR, default=np.array([], dtype=str)
        )

    def get_edge_set_variable_number(self):
        """Get the number of edge set variables in the model

        Returns
        -------
        num_nsvars : int
        """
        return self.get_dimension(ex.DIM_NUM_EDGE_SET_VAR, default=0)

    def get_edge_set_variable_truth_table(self, set_id=None):
        """Gets a truth table indicating which variables are defined for a edge set;
        if set_id is not passed, then a concatenated truth table for all edge
        sets is returned with variable index cycling faster than edge set index

        Parameters
        ----------
        set_id : int
            edge set *ID* (not *INDEX*)

        Returns
        -------
        nsvar_truth_tab : list of bool
            True if variable is defined in a edge set, False otherwise
        """
        truth_table = self.get_variable(ex.VAR_EDGE_SET_TAB)
        if set_id is not None:
            set_iid = self.get_edge_set_iid(set_id)
            truth_table = truth_table[set_iid - 1]
        return truth_table

    def get_edge_set_variable_values(self, set_id, var_name, time_step):
        """Get list of edge set variable values for a specified edge set, edge set
        variable name, and time step; the list has one variable value per edge in the
        set

        Parameters
        ----------
        set_id :  int
            edge set *ID* (not *INDEX*)
        var_name : str
            name of edge set variable
        time_step : int
            1-based index of time step

        Returns
        -------
        nsvar_vals : ndarray of float

        """
        raise NotImplementedError

    def get_edge_variable_names(self):
        """Get the list of edge variable names in the model

        Returns
        -------
        var_names : list of str
        """
        return self.get_variable(ex.VAR_NAME_EDGE_VAR, default=np.array([], dtype=str))

    def get_edge_variable_number(self):
        """Get the number of edge variables in the model

        Returns
        -------
        num_evars : int
        """
        return self.get_dimension(ex.DIM_NUM_EDGE_VAR)

    def get_edge_variable_values(self, edge_block_id, var_name, time_step=None):
        """Get list of edge variable values for a specified edge block, edge
        variable name, and time step

        Parameters
        ----------
        edge_block_id : int
            edge block *ID* (not *INDEX*)
        var_name : str
            name of edge variable
        time_step : int
            1-based index of time step

        Returns
        -------
        evar_vals : ndarray of float
        """
        if edge_block_id is None:
            return self.get_edge_variable_values_across_blocks(var_name, time_step)
        names = self.get_edge_variable_names()
        var_iid = self.get_iid(names, var_name)
        block_iid = self.get_edge_block_iid(edge_block_id)
        values = self.get_variable(ex.VAR_EDGE_VAR(var_iid, block_iid))
        if values is None:
            return None
        if time_step is None:
            return values
        elif time_step < 0:
            time_step = len(values)
        return values[time_step - 1]

    def get_edge_variable_values_across_blocks(self, var_name, time_step):
        values = []
        for id in self.get_edge_block_ids():
            x = self.get_edge_variable_values(id, var_name, time_step)
            if x is not None:
                values.extend(x)
            else:
                values.extend(np.zeros(self.num_edges_in_blk(id)))
        return np.array(values)

    def get_element_attr(self, block_id):
        """Get all attributes for each element in a block

        Parameters
        ----------
        block_id : int
            element block *ID* (not *INDEX*)

        Returns
        -------
        elem_attrs : ndarray of int
            list of attribute values for all elements in the block; the list cycles
            through all attributes of the first element, then all attributes of the
            second element, etc. Attributes are ordered by the ordering of the names
            returned by exo.get_element_attribute_names()
        """
        raise NotImplementedError

    def get_element_attr_values(self, block_id, elem_attr_name):
        """Get an attribute for each element in a block

        Parameters
        ----------
        block_id : int
            element block *ID* (not *INDEX*)
        elem_attr_name : str
            element attribute name

        Returns
        -------
        values : ndarray of int
            array of values for the requested attribute. Array has dimensions of 1 x
            num_elem, where num_elem is the number of elements on the element block.
        """
        raise NotImplementedError

    def get_element_block_id(self, blk_iid):
        """get exodus element block id from internal id (1 based index)

        Returns
        -------
        blk_iid : int

        """
        ids = self.get_element_block_ids()
        return ids[blk_iid - 1]

    def get_element_block_ids(self):
        """get mapping of exodus element block index to user - or application-defined
        element block id.

        block_ids is ordered by the element block INDEX ordering, a 1-based system
        going from 1 to exo.num_blks(), used by exodus for storage and input/output
        of array data stored on the element blocks; a user or application can
        optionally use a separate element block ID numbering system, so the block_ids
        array points to the element block ID for each element block INDEX

        Returns
        -------
        block_ids : ndarray of int

        """
        return self.get_variable(ex.VAR_ID_ELEM_BLK)

    def get_element_block_iid(self, block_id):
        """get exodus element block index from id

        Returns
        -------
        block_id : int

        """
        ids = self.get_element_block_ids()
        return self.get_iid(ids, block_id)

    def get_element_block(self, block_id):
        """Get the element block info

        Parameters
        ----------
        block_id : int
            element block ID (not INDEX)

        Returns
        -------
        elem_type : str
            element type, e.g. 'HEX8'
        num_block_elems : int
            number of elements in the block
        num_elem_nodes : int
            number of nodes per element
        num_elem_attrs : int
            number of attributes per element
        """
        if block_id not in self.get_element_block_ids():
            raise None
        elem_type = self.get_element_block_elem_type(block_id)
        if elem_type is None:
            return None
        info = SimpleNamespace(
            id=block_id,
            elem_type=elem_type,
            name=self.get_element_block_name(block_id),
            num_block_elems=self.num_elems_in_blk(block_id),
            num_elem_nodes=self.num_nodes_per_elem(block_id),
            num_elem_edges=self.num_edges_per_elem(block_id),
            num_elem_faces=self.num_faces_per_elem(block_id),
            num_elem_attrs=self.num_attr(block_id),
        )
        return info

    def get_element_block_name(self, block_id):
        """Get the element block name

        Parameters
        ----------
        block_id : int
            element block *ID* (not *INDEX*)

        Returns
        -------
        elem_block_name : string
        """
        block_ids = self.get_element_block_ids()
        blk_iid = self.get_iid(block_ids, block_id)
        names = self.get_element_block_names()
        return names[blk_iid - 1]

    def get_element_block_names(self):
        """Get a list of all element block names ordered by block *INDEX*; (see
        `exodus.get_ids` for explanation of the difference between block *ID* and
        block *INDEX*)

        Returns
        -------
        elem_block_names : list of str
        """
        return self.get_variable(ex.VAR_NAME_ELEM_BLK, default=np.array([], dtype=str))

    def get_element_conn(self, block_id, type=ex.types.node):
        """Get the nodal connectivity for a single block

        Parameters
        ----------
        block_id : int
            element block *ID* (not *INDEX*)

        Returns
        -------
        elem_conn : ndarray of int
            define the connectivity of each element in the block; the list cycles
            through all nodes of the first element, then all nodes of the second
            element, etc. (see `exodus.get_id_map` for explanation of node *INDEX*
            versus node *ID*)
        """
        block_iid = self.get_element_block_iid(block_id)
        if type == ex.types.node:
            var = ex.VAR_ELEM_BLK_CONN(block_iid)
        elif type == ex.types.edge:
            var = ex.VAR_EDGE_CONN(block_iid)
        elif type == ex.types.face:
            var = ex.VAR_FACE_CONN(block_iid)
        else:
            raise ValueError(f"Invalid element connectivity type {type!r}")
        return self.get_variable(var)

    def get_element_connectivity(self, block_id):
        """Get the nodal connectivity, number of elements, and number of nodes per
        element for a single block

        Parameters
        ----------
        block_id : int
            element block *ID* (not *INDEX*)

        Returns
        -------
        elem_conn : ndarray of int
            define the connectivity of each element in the block; the list cycles
            through all nodes of the first element, then all nodes of the second
            element, etc. (see `exodus.get_id_map` for explanation of node *INDEX*
            versus node *ID*)
        num_block_elems : int
            number of elements in the block
        num_elem_nodes : int
            number of nodes per element

        """
        info = SimpleNamespace(
            elem_conn=self.get_element_conn(block_id),
            num_block_elems=self.num_elems_in_blk(block_id),
            num_elem_nodes=self.num_nodes_per_elem(block_id),
        )
        return info

    def get_element_id_map(self):
        """Get mapping of exodus element index to user- or application- defined
        element id; elem_id_map is ordered by the element *INDEX* ordering, a 1-based
        system going from 1 to exo.num_elems(), used by exodus for storage and
        input/output of array data stored on the elements; a user or application can
        optionally use a separate element *ID* numbering system, so the elem_id_map
        points to the element *ID* for each element *INDEX*

        Returns
        -------
        elem_id_map : ndarray of int

        """
        map = self.get_variable(ex.VAR_ELEM_NUM_MAP)
        if map is None:
            map = np.arange(self.num_elems(), dtype=int) + 1
        return map

    def get_element_order_map(self):
        """Get mapping of exodus element index to application-defined optimal
        ordering; elem_order_map is ordered by the element index ordering used by
        exodus for storage and input/output of array data stored on the elements; a
        user or application can optionally use a separate element ordering, e.g. for
        optimal solver performance, so the elem_order_map points to the index used by
        the application for each exodus element index

        Returns
        -------
        elem_order_map : ndarray of int

        """
        raise NotImplementedError

    def get_element_set_ids(self):
        """Get mapping of exodus elem set index to user- or application- defined elem
        set id; elem_set_ids is ordered by the *INDEX* ordering, a 1-based system
        going from 1 to exo.num_elem_sets(), used by exodus for storage and
        input/output of array data stored on the elem sets; a user or application can
        optionally use a separate elem set *ID* numbering system, so the elem_set_ids
        array points to the elem set *ID* for each elem set *INDEX*

        Returns
        -------
        elem_set_ids : ndarray of int
        """
        return self.get_variable(ex.VAR_ELEM_SET_IDS)

    def get_element_set_names(self):
        """Get a list of all element set names ordered by element set *INDEX*; (see
        description of get_element_set_ids() for explanation of the difference between
        element set *ID* and element set *INDEX*)

        Returns
        -------
        element_set_names : list of str
        """
        return self.get_variable(ex.VAR_NAME_ELEM_SET, default=np.array([], dtype=str))

    def get_element_block_elem_type(self, block_id):
        """Get the element type, e.g. "HEX8", for an element block"""
        block_iid = self.get_element_block_iid(block_id)
        var = self.get_variable(ex.VAR_ELEM_BLK_CONN(block_iid), raw=True)
        return None if var is None else var.elem_type

    def get_element_attribute_names(self, block_id):
        """Get the list of element attribute names for a block

        Parameters
        ----------
        block_id : int
            element block *ID* (not *INDEX*)

        Returns
        -------
        attr_names : list of str
        """
        raise NotImplementedError

    def get_element_property_names(self):
        """Get the list of element property names for all element blocks in the model

        Returns
        -------
        eprop_names : list of str
        """
        raise NotImplementedError

    def get_element_property_value(self, block_id, eprop_name):
        """Get element property value (an integer) for a specified element block and
        element property name

        Parameters
        ----------
        block_id : int
            element block *ID* (not *INDEX*)
        eprop_name : str

        Returns
        -------
        eprop_val : int
        """
        raise NotImplementedError

    def get_element_variable_history(self, var_name, elem_id):
        """Get element variable values for a element variable var_name at element
        `elem_id` for all time

        Parameters
        ----------
        var_name : str
            name of element variable
        elem_id : int
            node ID not *INDEX*

        Returns
        -------
        evar_vals : ndarray of float

        """
        names = self.get_element_variable_names()
        var_iid = self.get_iid(names, var_name)
        e = self._get_element_info(elem_id)
        values = self.get_variable(ex.VAR_ELEM_VAR(var_iid, e.blk_iid))
        if values is None:
            return None
        return values[:, e.iid - 1]

    def get_element_variable_names(self):
        """Get the list of element variable names in the model

        Returns
        -------
        var_names : list of str
        """
        return self.get_variable(ex.VAR_NAME_ELEM_VAR, default=np.array([], dtype=str))

    def get_element_variable_number(self):
        """Get the number of element variables in the model

        Returns
        -------
        num_evars : int
        """
        return self.get_dimension(ex.DIM_NUM_ELEM_VAR)

    def get_element_variable_truth_table(self, block_id=None):
        """gets a truth table indicating which variables are defined for a block; if
        block_id is not passed, then a concatenated truth table for all blocks is
        returned with variable index cycling faster than block index

        Parameters
        ----------
        block_id : int
            element block *ID* (not *INDEX*)

        Returns
        -------
        evar_truth_tab : list of bool
            True for variable defined in block, False otherwise
        """
        truth_table = self.get_variable(ex.VAR_ELEM_TAB)
        if block_id is not None:
            block_iid = self.get_element_block_iid(block_id)
            truth_table = truth_table[block_iid - 1]
        return truth_table

    def get_element_variable_values(self, block_id, var_name, time_step=None):
        """Get list of element variable values for a specified element block, element
        variable name, and time step

        Parameters
        ----------
        block_id : int
            element block *ID* (not *INDEX*)
        var_name : str
            name of element variable
        time_step : int
            1-based index of time step

        Returns
        -------
        evar_vals : ndarray of float
        """
        if block_id is None:
            return self.get_element_variable_values_across_blocks(var_name, time_step)

        names = self.get_element_variable_names()
        var_iid = self.get_iid(names, var_name)
        block_iid = self.get_element_block_iid(block_id)
        values = self.get_variable(ex.VAR_ELEM_VAR(var_iid, block_iid))
        if values is None:
            return None
        if time_step is None:
            return values
        elif time_step < 0:
            time_step = len(values)
        return values[time_step - 1]

    def get_element_variable_values_across_blocks(self, var_name, time_step=None):
        values = []
        for id in self.get_element_block_ids():
            x = self.get_element_variable_values(id, var_name, time_step)
            if x is not None:
                values.extend(x)
            else:
                values.extend(np.zeros(self.num_elems_in_blk(id)))
        return np.array(values)

    def get_face_block_elem_type(self, block_id):
        """Get the element type, e.g. "HEX8", for a face block"""
        block_iid = self.get_face_block_iid(block_id)
        var = self.get_variable(ex.VAR_FACE_BLK_CONN(block_iid), raw=True)
        return None if var is None else var.elem_type

    def get_face_block_id(self, blk_iid):
        """get exodus face block id from internal id (1 based index)

        Returns
        -------
        blk_iid : int

        """
        ids = self.get_face_block_ids()
        return ids[blk_iid - 1]

    def get_face_block_ids(self):
        """get mapping of exodus face block index to user - or application-defined
        face block id.

        block_ids is ordered by the face block INDEX ordering,
        a 1-based system going from 1 to exo.num_blks(), used by exodus for storage
        and input/output of array data stored on the face blocks a user or
        application can optionally use a separate face block ID numbering system,
        so the block_ids array points to the face block ID for each face
        block INDEX

        Returns
        -------
        block_ids : ndarray of int

        """
        return self.get_variable(ex.VAR_ID_FACE_BLK)

    def get_face_block_iid(self, block_id):
        """get exodus face block index from id

        Returns
        -------
        block_iid : int

        """
        ids = self.get_face_block_ids()
        return self.get_iid(ids, block_id)

    def get_face_block(self, block_id):
        """Get the face block info

        Parameters
        ----------
        block_id : int
            Face block ID (not INDEX)

        Returns
        -------
        elem_type : str
            Element type, e.g. 'HEX8'
        num_block_faces : int
            number of faces in the block
        num_face_nodes : int
            number of nodes per face
        num_face_attrs : int
            number of attributes per face
        """
        if block_id not in self.get_face_block_ids():
            raise None
        elem_type = self.get_face_block_elem_type(block_id)
        if elem_type is None:
            return None
        info = SimpleNamespace(
            id=block_id,
            iid=self.get_face_block_iid(block_id),
            elem_type=elem_type,
            num_block_faces=self.num_faces_in_blk(block_id),
            num_face_nodes=self.num_nodes_per_face(block_id),
            num_face_attrs=self.num_face_attr(block_id),
        )
        return info

    def get_face_block_name(self, block_id):
        """Get the face block name

        Parameters
        ----------
        block_id : int
            face block *ID* (not *INDEX*)

        Returns
        -------
        face_block_name : string
        """
        block_ids = self.get_face_block_ids()
        blk_iid = self.get_iid(block_ids, block_id)
        names = self.get_face_block_names()
        return names[blk_iid - 1]

    def get_face_block_names(self):
        """Get a list of all face block names ordered by block *INDEX*; (see
        `exodus.get_ids` for explanation of the difference between block *ID* and
        block *INDEX*)

        Returns
        -------
        face_block_names : list of str
        """
        return self.get_variable(ex.VAR_NAME_FACE_BLK, default=np.array([], dtype=str))

    def get_face_block_conn(self, block_id):
        """Get the nodal connectivity for a single block

        Parameters
        ----------
        block_id : int
            Face block *ID* (not *INDEX*)

        Returns
        -------
        face_conn : ndarray of int
            define the connectivity of each face in the block; the list cycles
            through all nodes of the first face, then all nodes of the second
            face, etc. (see `exodus.get_id_map` for explanation of node *INDEX*
            versus node *ID*)
        """
        block_iid = self.get_face_block_iid(block_id)
        return self.get_variable(ex.VAR_FACE_BLK_CONN(block_iid))

    def get_face_variable_truth_table(self, block_id=None):
        """gets a truth table indicating which variables are defined for a block; if
        block_id is not passed, then a concatenated truth table for all blocks is
        returned with variable index cycling faster than block index

        Parameters
        ----------
        block_id : int
            face block *ID* (not *INDEX*)

        Returns
        -------
        evar_truth_tab : list of bool
            True for variable defined in block, False otherwise

        """
        truth_table = self.get_variable(ex.VAR_FACE_BLK_TAB)
        if block_id is not None:
            block_iid = self.get_face_block_iid(block_id)
            truth_table = truth_table[block_iid - 1]
        return truth_table

    def get_face_set(self, set_id):
        ns = self.get_face_set_params(set_id)
        ns.id = set_id
        ns.name = self.get_face_set_name(set_id)
        ns.faces = self.get_face_set_faces(set_id)
        ns.dist_facts = self.get_face_set_dist_facts(set_id)
        return ns

    def get_face_set_dist_facts(self, set_id):
        """Get the list of distribution factors for faces in a face set

        Parameters
        ----------
        set_id : int
            face set *ID* (not *INDEX*)

        Returns
        -------
        fs_dist_facts : ndarray of float
            a list of distribution factors, e.g. face 'weights'
        """
        set_iid = self.get_face_set_iid(set_id)
        return self.get_variable(ex.VAR_DF_FACE_SET(set_iid))

    def get_face_set_iid(self, set_id):
        ids = self.get_face_set_ids()
        return self.get_iid(ids, set_id)

    def get_face_set_ids(self):
        """Get mapping of exodus face set index to user- or application- defined face
        set id; face_set_ids is ordered by the *INDEX* ordering, a 1-based system
        going from 1 to exo.num_face_sets(), used by exodus for storage and
        input/output of array data stored on the face sets; a user or application can
        optionally use a separate face set *ID* numbering system, so the face_set_ids
        array points to the face set *ID* for each face set *INDEX*

        Returns
        -------
        face_set_ids : ndarray of int
        """
        return self.get_variable(ex.VAR_FACE_SET_IDS)

    def get_face_set_names(self):
        """Get a list of all face set names ordered by face set *INDEX*; (see
        description of get_face_set_ids() for explanation of the difference between
        face set *ID* and face set *INDEX*)

        Returns
        -------
        face_set_names : list of str
        """
        return self.get_variable(ex.VAR_NAME_FACE_SET, default=np.array([], dtype=str))

    def get_face_set_faces(self, set_id):
        """Get the list of face *INDICES* in a face set (see `exodus.get_id_map` for
        explanation of face *INDEX* versus face *ID*)

        Parameters
        ----------
        set_id : int
            face set *ID* (not *INDEX*)

        Returns
        -------
        fs_faces : ndarray of int

        """
        set_iid = self.get_face_set_iid(set_id)
        return self.get_variable(ex.VAR_FACE_FACE_SET(set_iid))

    def get_face_set_params(self, set_id):
        """Get number of faces and distribution factors (e.g. nodal 'weights') in a
        face set

        Parameters
        ----------
        set_id : int
            face set *ID* (not *INDEX*)

        Returns
        -------
        num_fs_faces : int
        num_fs_dist_facts : int
        """
        if set_id not in self.get_face_set_ids():
            raise ValueError(f"{set_id} is not a valid face set ID")
        info = SimpleNamespace(
            num_faces=self.num_faces_in_face_set(set_id),
            num_nodes_per_face=self.num_nodes_per_face(set_id),
            num_dist_facts=self.num_face_set_dist_fact(set_id),
        )
        return info

    def get_face_set_property_names(self):
        """Get the list of face set property names for all face sets in the model

        Returns
        -------
        nsprop_names : list of str
        """
        raise NotImplementedError

    def get_face_set_property_value(self, set_id, name):
        """Get face set property value (an integer) for a specified face
        set and face set property name

        Parameters
        ----------
        set_id: int
            face set *ID* (not *INDEX*)
        nsprop_name : string

        Returns
        -------
        nsprop_val : int
        """
        raise NotImplementedError

    def get_face_set_variable_names(self):
        """Get the list of face set variable names in the model

        Returns
        -------
        var_names : list of str
        """
        return self.get_variable(
            ex.VAR_NAME_FACE_SET_VAR, default=np.array([], dtype=str)
        )

    def get_face_set_variable_number(self):
        """Get the number of face set variables in the model

        Returns
        -------
        num_nsvars : int
        """
        return self.get_dimension(ex.DIM_NUM_FACE_SET_VAR, default=0)

    def get_face_set_variable_truth_table(self, set_id=None):
        """Gets a truth table indicating which variables are defined for a face set;
        if set_id is not passed, then a concatenated truth table for all face
        sets is returned with variable index cycling faster than face set index

        Parameters
        ----------
        set_id : int
            face set *ID* (not *INDEX*)

        Returns
        -------
        nsvar_truth_tab : list of bool
            True if variable is defined in a face set, False otherwise
        """
        truth_table = self.get_variable(ex.VAR_FACE_SET_TAB)
        if set_id is not None:
            set_iid = self.get_face_set_iid(set_id)
            truth_table = truth_table[set_iid - 1]
        return truth_table

    def get_face_set_variable_values(self, set_id, var_name, time_step):
        """Get list of face set variable values for a specified face set, face set
        variable name, and time step; the list has one variable value per face in the
        set

        Parameters
        ----------
        set_id :  int
            face set *ID* (not *INDEX*)
        var_name : str
            name of face set variable
        time_step : int
            1-based index of time step

        Returns
        -------
        nsvar_vals : ndarray of float

        """
        raise NotImplementedError

    def get_face_variable_names(self):
        """Get the list of face variable names in the model

        Returns
        -------
        var_names : list of str
        """
        return self.get_variable(ex.VAR_NAME_FACE_VAR, default=np.array([], dtype=str))

    def get_face_variable_number(self):
        """Get the number of face variables in the model

        Returns
        -------
        num_evars : int
        """
        return self.get_dimension(ex.DIM_NUM_FACE_VAR)

    def get_face_variable_values(self, block_id, var_name, time_step=None):
        """Get list of face variable values for a specified face block, face
        variable name, and time step

        Parameters
        ----------
        block_id : int
            face block *ID* (not *INDEX*)
        var_name : str
            name of face variable
        time_step : int
            1-based index of time step

        Returns
        -------
        fvar_vals : ndarray of float
        """
        if block_id is None:
            return self.get_face_variable_values_across_blocks(var_name, time_step)
        names = self.get_face_variable_names()
        var_iid = self.get_iid(names, var_name)
        block_iid = self.get_face_block_iid(block_id)
        values = self.get_variable(ex.VAR_FACE_VAR(var_iid, block_iid))
        if values is None:
            return None
        if time_step is None:
            return values
        elif time_step < 0:
            time_step = len(values)
        return values[time_step - 1]

    def get_face_variable_values_across_blocks(self, var_name, time_step):
        values = []
        for id in self.get_face_block_ids():
            x = self.get_face_variable_values(id, var_name, time_step)
            if x is not None:
                values.extend(x)
            else:
                values.extend(np.zeros(self.num_faces_in_blk(id)))
        return np.array(values)

    def get_global_variable_iid(self, var_name):
        names = self.get_global_variable_names()
        return self.get_iid(names, var_name)

    def get_global_variable_names(self):
        """Get the list of global variable names in the model

        Returns
        -------
        var_names : list of str
        """
        return self.get_variable(ex.VAR_NAME_GLO_VAR, default=np.array([], dtype=str))

    def get_global_variable_number(self):
        """Get the number of global variables in the model

        Returns
        -------
        num_gvars : int
        """
        return self.get_dimension(ex.DIM_NUM_GLO_VAR)

    def get_global_variable_value(self, var_name, time_step):
        """Get a global variable value for a specified global variable name and time
        step

        Parameters
        ----------
        var_name : str
            name of global variable
        time_step : int
            1-based index of time step

        Returns
        -------
        gvar_val : float
        """
        values = self.get_global_variable_values(var_name)
        return values[time_step - 1]

    def get_global_variable_values(self, var_name):
        """Get global variable values over all time steps for one global variable
        name

        Parameters
        ----------
        var_name : str
            name of global variable

        Returns
        -------
        gvar_vals : ndarray of float

        """
        var_iid = self.get_global_variable_iid(var_name)
        values = self.get_variable(ex.VAR_GLO_VAR)
        return values[:, var_iid - 1]

    def get_iid(self, container, item):
        if item not in container:
            return None
        return index(container, item) + 1

    def get_info_records(self):
        """Get a list info records where each entry in the list is one info record,
        e.g. a line of an input deck

        Returns
        -------
        info_recs : list of str
        """
        if not self.num_info_records():
            return None
        return self.get_variable(ex.VAR_INFO)

    def get_node_set(self, set_id):
        ns = self.get_node_set_params(set_id)
        ns.id = set_id
        ns.name = self.get_node_set_name(set_id)
        ns.nodes = self.get_node_set_nodes(set_id)
        ns.dist_facts = self.get_node_set_dist_facts(set_id)
        return ns

    def get_node_id_map(self):
        """Get mapping of exodus node index to user- or application- defined node id;
        node_id_map is ordered the same as the nodal coordinate arrays returned by
        exo.get_coords() -- this ordering follows the exodus node *INDEX* order, a
        1-based system going from 1 to exo.num_nodes(); a user or application can
        optionally use a separate node *ID* numbering system, so the node_id_map
        points to the node *ID* for each node *INDEX*

        Returns
        -------
        node_id_map : ndarray of int
        """
        map = self.get_variable(ex.VAR_NODE_NUM_MAP)
        if map is None:
            map = np.arange(self.num_nodes(), dtype=int) + 1
        return map

    def get_node_set_dist_facts(self, set_id):
        """Get the list of distribution factors for nodes in a node set

        Parameters
        ----------
        set_id : int
            node set *ID* (not *INDEX*)

        Returns
        -------
        ns_dist_facts : ndarray of float
            a list of distribution factors, e.g. nodal 'weights'
        """
        set_iid = self.get_node_set_iid(set_id)
        return self.get_variable(ex.VAR_DF_NODE_SET(set_iid))

    def get_node_set_id(self, set_iid):
        """Get exodus node set id from internal ID

        Returns
        -------
        set_id : int
        """
        ids = self.get_node_set_ids()
        return ids[set_iid - 1]

    def get_node_set_ids(self):
        """Get mapping of exodus node set index to user- or application- defined node
        set id; set_ids is ordered by the *INDEX* ordering, a 1-based system
        going from 1 to exo.num_node_sets(), used by exodus for storage and
        input/output of array data stored on the node sets; a user or application can
        optionally use a separate node set *ID* numbering system, so the set_ids
        array points to the node set *ID* for each node set *INDEX*

        Returns
        -------
        set_ids : ndarray of int
        """
        return self.get_variable(ex.VAR_NODE_SET_IDS)

    def get_node_set_iid(self, set_id):
        """Get exodus node set index from id

        Returns
        -------
        set_id : int
        """
        ids = self.get_node_set_ids()
        return self.get_iid(ids, set_id)

    def get_node_set_name(self, set_id):
        """get the name of a node set

        Parameters
        ----------
        set_id : int
            node set *ID* (not *INDEX*)

        Returns
        -------
        node_set_name : str
        """
        set_ids = self.get_node_set_ids()
        set_iid = self.get_iid(set_ids, set_id)
        names = self.get_node_set_names()
        return names[set_iid - 1]

    def get_node_set_names(self):
        """Get a list of all node set names ordered by node set *INDEX*; (see
        description of get_node_set_ids() for explanation of the difference between
        node set *ID* and node set *INDEX*)

        Returns
        -------
        node_set_names : list of str
        """
        return self.get_variable(ex.VAR_NAME_NODE_SET, default=np.array([], dtype=str))

    def get_node_set_nodes(self, set_id):
        """Get the list of node *INDICES* in a node set (see `exodus.get_id_map` for
        explanation of node *INDEX* versus node *ID*)

        Parameters
        ----------
        set_id : int
            node set *ID* (not *INDEX*)

        Returns
        -------
        ns_nodes : ndarray of int

        """
        set_iid = self.get_node_set_iid(set_id)
        return self.get_variable(ex.VAR_NODE_NODE_SET(set_iid))

    def get_node_set_params(self, set_id):
        """Get number of nodes and distribution factors (e.g. nodal 'weights') in a
        node set

        Parameters
        ----------
        set_id : int
            node set *ID* (not *INDEX*)

        Returns
        -------
        num_nodes : int
        num_dist_facts : int
        """
        if set_id not in self.get_node_set_ids():
            raise ValueError(f"{set_id} is not a valid node set ID")
        info = SimpleNamespace(
            num_nodes=self.num_nodes_in_node_set(set_id),
            num_dist_facts=self.num_node_set_dist_fact(set_id),
        )
        return info

    def get_node_set_property_names(self):
        """Get the list of node set property names for all node sets in the model

        Returns
        -------
        nsprop_names : list of str
        """
        raise NotImplementedError

    def get_node_set_property_value(self, set_id, name):
        """Get node set property value (an integer) for a specified node
        set and node set property name

        Parameters
        ----------
        set_id: int
            node set *ID* (not *INDEX*)
        nsprop_name : string

        Returns
        -------
        nsprop_val : int
        """
        raise NotImplementedError

    def get_node_set_variable_names(self):
        """Get the list of node set variable names in the model

        Returns
        -------
        var_names : list of str
        """
        return self.get_variable(
            ex.VAR_NAME_NODE_SET_VAR, default=np.array([], dtype=str)
        )

    def get_node_set_variable_number(self):
        """Get the number of node set variables in the model

        Returns
        -------
        num_nsvars : int
        """
        return self.get_dimension(ex.DIM_NUM_NODE_SET_VAR, default=0)

    def get_node_set_variable_truth_table(self, set_id=None):
        """Gets a truth table indicating which variables are defined for a node set;
        if set_id is not passed, then a concatenated truth table for all node
        sets is returned with variable index cycling faster than node set index

        Parameters
        ----------
        set_id : int
            node set *ID* (not *INDEX*)

        Returns
        -------
        nsvar_truth_tab : list of bool
            True if variable is defined in a node set, False otherwise
        """
        truth_table = self.get_variable(ex.VAR_NODE_SET_TAB)
        if set_id is not None:
            set_iid = self.get_node_set_iid(set_id)
            truth_table = truth_table[set_iid - 1]
        return truth_table

    def get_node_set_variable_values(self, set_id, var_name, time_step):
        """Get list of node set variable values for a specified node set, node set
        variable name, and time step; the list has one variable value per node in the
        set

        Parameters
        ----------
        set_id :  int
            node set *ID* (not *INDEX*)
        var_name : str
            name of node set variable
        time_step : int
            1-based index of time step

        Returns
        -------
        nsvar_vals : ndarray of float

        """
        raise NotImplementedError

    def get_node_variable_history(self, var_name, node_id):
        """Get nodal variable values for a nodal variable var_name at node `node_id`
        for all time

        Parameters
        ----------
        var_name : str
            name of nodal variable
        node_id : int
            node ID not *INDEX*

        Returns
        -------
        nvar_vals : ndarray of float
        """
        values = self.get_node_variable_values(var_name, time_step=None)
        if values is None:
            return values
        node_num_map = self.get_node_id_map()
        node_iid = self.get_iid(node_num_map, node_id)
        return values[:, node_iid - 1]

    def get_node_variable_names(self):
        """Get the list of nodal variable names in the model

        Returns
        -------
        var_names : list of str
        """
        return self.get_variable(ex.VAR_NAME_NODE_VAR, default=np.array([], dtype=str))

    def get_node_variable_number(self):
        """Get the number of node variables in the model

        Returns
        -------
        num_evars : int
        """
        return self.get_dimension(ex.DIM_NUM_NODE_VAR)

    def get_node_variable_values(self, var_name, time_step=None):
        """Get list of nodal variable values for a nodal variable var_name and time
        step

        Parameters
        ----------
        var_name : str
            name of nodal variable
        time_step : int
            1-based index of time step

        Returns
        -------
        nvar_vals : ndarray of float
        """
        names = self.get_node_variable_names()
        var_iid = self.get_iid(names, var_name)
        values = self.get_variable(ex.VAR_NODE_VAR(var_iid))
        if values is None:
            return None
        if time_step is None:
            return values
        elif time_step < 0:
            return values[-1]
        return values[time_step - 1]

    def get_qa_records(self):
        """Get a list of QA records where each QA record is a length-4 tuple of
        strings:

              1) the software name that accessed/modified the database
              2) the software descriptor, e.g. version
              3) additional software data
              4) time stamp

        Returns
        qa_recs : list of tuple of string
        """
        return self.get_variable(ex.VAR_QA_TITLE)

    def get_side_set(self, set_id):
        """Get the lists of element and side indices in a side set; the two lists
        correspond: together, ss_elems[i] and ss_sides[i] define the face of an
        element

        Parameters
        ----------
        set_id : int
            side set *ID* (not *INDEX*)

        Returns
        -------
        ss_elems : ndarray of int
        ss_sides : ndarray of int
        """
        ns = self.get_side_set_params(set_id)
        ns.id = set_id
        ns.name = self.get_side_set_name(set_id)
        ns.sides = self.get_side_set_sides(set_id)
        ns.elems = self.get_side_set_elems(set_id)
        ns.dist_facts = self.get_side_set_dist_facts(set_id)
        return ns

    def get_side_set_dist_facts(self, set_id):
        """Get the list of distribution factors for sides in a side set

        Parameters
        ----------
        set_id : int
            side set *ID* (not *INDEX*)

        Returns
        -------
        ss_dist_facts : ndarray of float
            A list of distribution factors, e.g. nodal 'weights'

        Note
        ----
        The number of sides (and distribution factors) in a side set is the sum of
        all face sides. A single side can be counted more than once, i.e. once for
        each face it belongs to in the side set.
        """
        set_iid = self.get_side_set_iid(set_id)
        if set_iid is None:
            return None
        return self.get_variable(ex.VAR_DF_SIDE_SET(set_iid))

    def get_side_set_ids(self):
        """Get mapping of exodus side set index to user- or application- defined side
        set id; set_ids is ordered by the *INDEX* ordering, a 1-based system
        going from 1 to exo.num_side_sets(), used by exodus for storage and
        input/output of array data stored on the side sets; a user or application can
        optionally use a separate side set *ID* numbering system, so the set_ids
        array points to the side set *ID* for each side set *INDEX*

        Returns
        -------
        set_ids : ndarray of int
        """
        return self.get_variable(ex.VAR_SIDE_SET_IDS)

    def get_side_set_iid(self, set_id):
        ids = self.get_side_set_ids()
        return self.get_iid(ids, set_id)

    def get_side_set_name(self, set_id):
        """Get the name of a side set

        Parameters
        ----------
        set_id : int
            side set *ID* (not *INDEX*)

        Returns
        -------
        side_set_name : str
        """
        set_ids = self.get_side_set_ids()
        set_iid = self.get_iid(set_ids, set_id)
        names = self.get_side_set_names()
        return names[set_iid - 1]

    def get_side_set_names(self):
        """Get a list of all side set names ordered by side set *INDEX*; (see
        description of get_side_set_ids() for explanation of the difference between
        side set *ID* and side set *INDEX*)

        Returns
        -------
        side_set_names : list of str
        """
        return self.get_variable(ex.VAR_NAME_SIDE_SET, default=np.array([], dtype=str))

    def get_side_set_node_list(self, id):
        """Get two lists:
            1. number of nodes for each side in the set
            2. concatenation of the nodes for each side in the set

        Parameters
        ----------
        set_id : int
            side set *ID* (not *INDEX*)

        Returns
        -------
        ss_num_side_nodes : ndarray of int
        ss_nodes : ndarray of int

        Note
        ----
        The number of nodes (and distribution factors) in a side set is the sum of
        the entries in ss_num_nodes_per_side. A single node can be counted more than
        once, i.e. once for each face it belongs to in the side set.
        """
        raise NotImplementedError

    def get_side_set_params(self, set_id):
        """Get number of sides and distribution factors (e.g. nodal 'weights') in a
        side set

        Parameters
        ----------
        set_id : int
            side set *ID* (not *INDEX*)

        Returns
        -------
        num_ss_sides : int
        num_ss_dist_facts : int

        Note
        ----
        The number of nodes (and distribution factors) in a side set is the sum of
        all face nodes. A single node can be counted more than once, i.e. once for
        each face it belongs to in the side set.
        """
        if set_id not in self.get_side_set_ids():
            raise ValueError(f"{set_id} is not a valid side set ID")
        info = SimpleNamespace(
            num_sides=self.num_sides_in_side_set(set_id),
            num_dist_facts=self.num_side_set_dist_fact(set_id),
        )
        return info

    def get_side_set_property_names(self):
        """Get the list of side set property names for all side sets in the model

        Returns
        -------
        ssprop_names : list of str
        """
        raise NotImplementedError

    def get_side_set_property_value(self, set_id, ssprop_name):
        """Get side set property value (an integer) for a specified side
        set and side set property name

        Parameters
        ----------
        set_id: int
            side set *ID* (not *INDEX*)
        ssprop_name : string

        Returns
        -------
        nsprop_val : int
        """
        raise NotImplementedError

    def get_side_set_elems(self, set_id):
        """Get the list of side *INDICES* in a side set (see `exodus.get_id_map` for
        explanation of side *INDEX* versus side *ID*)

        Parameters
        ----------
        set_id : int
            side set *ID* (not *INDEX*)

        Returns
        -------
        ss_sides : ndarray of int

        """
        set_iid = self.get_side_set_iid(set_id)
        return self.get_variable(ex.VAR_ELEM_SIDE_SET(set_iid))

    def get_side_set_sides(self, set_id):
        """Get the list of side *INDICES* in a side set (see `exodus.get_id_map` for
        explanation of side *INDEX* versus side *ID*)

        Parameters
        ----------
        set_id : int
            side set *ID* (not *INDEX*)

        Returns
        -------
        ss_sides : ndarray of int

        """
        set_iid = self.get_side_set_iid(set_id)
        return self.get_variable(ex.VAR_SIDE_SIDE_SET(set_iid))

    def get_side_set_variable_names(self):
        """Get the list of side set variable names in the model

        Returns
        -------
        var_names : list of str
        """
        raise NotImplementedError

    def get_side_set_variable_number(self):
        """Get the number of side set variables in the model

        Returns
        -------
        num_nsvars : int
        """
        raise NotImplementedError

    def get_side_set_variable_truth_table(self, set_id=None):
        """Gets a truth table indicating which variables are defined for a side set;
        if set_id is not passed, then a concatenated truth table for all side
        sets is returned with variable index cycling faster than side set index

        Parameters
        ----------
        set_id : int
            side set *ID* (not *INDEX*)

        Returns
        -------
        nsvar_truth_tab : list of bool
            True if variable is defined in a side set, False otherwise
        """
        truth_table = self.get_variable(ex.VAR_SIDE_SET_TAB)
        if set_id is not None:
            set_iid = self.get_side_set_iid(set_id)
            truth_table = truth_table[set_iid - 1]
        return truth_table

    def get_side_set_variable_values(self, set_id, var_name, time_step):
        """Get list of side set variable values for a specified side set, side set
        variable name, and time step; the list has one variable value per side in the
        set

        Parameters
        ----------
        set_id :  int
            side set *ID* (not *INDEX*)
        var_name : str
            name of side set variable
        time_step : int
            1-based index of time step

        Returns
        -------
        nsvar_vals : ndarray of float

        """
        raise NotImplementedError

    def get_time(self, time_step):
        """Get the time values"""
        return self.get_times()[time_step - 1]

    def get_time_step(self, target, pcttol=1.0e-5):
        """Given an exodus file object and a specified time, return the time index.

        The time index is 1-based, i.e. the index of the first time step is 1.
        """
        times = self.get_times()
        for (itime, time) in enumerate(times):
            if time > target:
                if (time - target) > (target - times[itime - 1]):
                    itime -= 1
                    break

        # If there are a lot of closely spaced time steps a better test
        # would have, e.g. (times[itime] - times[itime - 1]) in the
        # denominator; eveutually add this as an option
        if abs(target) > 0.0:
            if (abs(target - times[itime]) / target) > pcttol:
                logging.warn(
                    "Solution time differs significantly from desired solution time."
                )

        return itime + 1

    def get_times(self):
        """Get the time values"""
        return self.get_variable(ex.VAR_WHOLE_TIME)

    def get_variable_type(self, name):
        """Determines the variable type of `name`"""
        if name in self.get_global_variable_names():
            return "g"
        elif name in self.get_element_variable_names():
            return "e"
        elif name in self.get_node_variable_names():
            return "n"
        elif name in self.get_edge_variable_names():
            return "d"
        elif name in self.get_face_variable_names():
            return "f"
        return None

    def get_variable_values(self, type, name, time_step=None):
        """Determines the variable type of `name`"""
        if type == "g":
            values = self.get_global_variable_values(name)
        elif type == "e":
            values = self.get_element_variable_values(None, name)
        elif type == "n":
            values = self.get_node_variable_values(name)
        elif type == "d":
            values = self.get_edge_variable_values(None, name)
        elif type == "f":
            values = self.get_face_variable_values(None, name)
        else:
            return None
        if time_step is None:
            return values
        elif time_step < 0:
            time_step = len(values)
        return values[time_step - 1]

    @requires_write_mode
    def initw(self):
        version = 5.0300002
        self.setattr(ex.ATT_API_VERSION, version)
        self.setattr(ex.ATT_VERSION, version)
        self.setattr(ex.ATT_FLT_WORDSIZE, 4)
        self.setattr(ex.ATT_FILESIZE, 1)

        self.create_dimension(ex.DIM_TIME, None)
        self.create_variable(ex.VAR_WHOLE_TIME, float, (ex.DIM_TIME,))

        # standard ExodusII dimensioning
        self.create_dimension(ex.DIM_NAME, 32)
        self.create_dimension(ex.DIM_STR, 32)
        self.create_dimension(ex.DIM_LIN, 80)
        self.create_dimension(ex.DIM_N4, 4)

        self._counter = {}

    def num_attr(self, block_id):
        """get the number of attributes per element for an element block

        Parameters
        ----------
        block_id: int
            element block ID(not INDEX)

        Returns
        -------
            num_elem_attrs : int

        """
        block_iid = self.get_element_block_iid(block_id)
        return self.get_dimension(ex.DIM_NUM_ATT_IN_ELEM_BLK(block_iid), default=0)

    def num_blks(self):
        """Get the number of element blocks in the model

        Returns
        -------
        num_elem_blks : int
        """
        return self.exinit.num_el_blk

    def num_element_blocks(self):
        return self.num_blks()

    def num_dimensions(self):
        return self.exinit.num_dim

    def num_edge_blk(self):
        """Number of model edge blocks"""
        return self.exinit.num_edge_blk

    def num_edge_maps(self):
        """Number of model edge maps"""
        return self.exinit.num_edge_maps

    def num_edge_sets(self):
        """Number of model edge sets"""
        return self.exinit.num_edge_sets

    def num_edges(self):
        """Number of model edges"""
        return self.get_dimension(ex.DIM_NUM_EDGE, default=0)

    def num_edges_per_elem(self, block_id):
        """Get the number of edges per element for an element block

        Parameters
        ----------
        block_id: int
            element block ID(not INDEX)

        Returns
        -------
        num_elem_edges : int

        """
        block_iid = self.get_element_block_iid(block_id)
        return self.get_dimension(ex.DIM_NUM_EDGE_PER_ELEM(block_iid))

    def num_elem_blk(self):
        """Get the number of element blocks in the model

        Returns
        -------
        num_elem_blks : int
        """
        return self.exinit.num_el_blk

    def num_elem_maps(self):
        """Number of model elem maps"""
        return self.exinit.num_elem_maps

    def num_elem_sets(self):
        """Number of model elem sets"""
        return self.exinit.num_elem_sets

    def num_elems(self):
        """Get the number of elements in the model

        Returns
        -------
        num_elems : int

        """
        return self.get_dimension(ex.DIM_NUM_ELEM, default=0)

    def num_elems_in_all_blks(self):
        """Get the number of elements in all element blocks

        Parameters
        ----------
        block_id : int
            element block ID (not INDEX)

        Returns
        -------
        num_block_elems : int

        """
        num_elems = []
        for block_id in self.get_element_block_ids():
            num_elems.append(self.num_elems_in_blk(block_id))
        return np.array(num_elems, dtype=int)

    def num_elems_in_blk(self, block_id):
        """Get the number of elements in an element block

        Parameters
        ----------
        block_id : int
            element block ID (not INDEX)

        Returns
        -------
        num_block_elems : int

        """
        block_iid = self.get_element_block_iid(block_id)
        return self.get_dimension(ex.DIM_NUM_ELEM_IN_ELEM_BLK(block_iid))

    def num_face_attr(self, block_id):
        """get the number of attributes per face for an face block

        Parameters
        ----------
        block_id: int
            Face block ID(not INDEX)

        Returns
        -------
            num_face_attrs : int

        """
        block_iid = self.get_face_block_iid(block_id)
        return self.get_dimension(ex.DIM_NUM_ATT_IN_FACE_BLK(block_iid))

    def num_face_blk(self):
        """Number of model face blocks"""
        return self.exinit.num_face_blk

    def num_face_maps(self):
        """Number of model face maps"""
        return self.exinit.num_face_maps

    def num_face_sets(self):
        """Number of model face sets"""
        return self.exinit.num_face_sets

    def num_faces(self):
        """Number of model faces"""
        return self.get_dimension(ex.DIM_NUM_FACE, 0)

    def num_faces_in_blk(self, block_id):
        """Get the number of faces in a face block

        Parameters
        ----------
        block_id : int
            Face block ID (not INDEX)

        Returns
        -------
        num_block_faces : int

        """
        block_iid = self.get_face_block_iid(block_id)
        return self.get_dimension(ex.DIM_NUM_FACE_IN_FACE_BLK(block_iid))

    def num_faces_in_side_set(self, set_id):
        """Get the number of faces in a side set

        Parameters
        ----------
        set_id : int
            side set *ID* (not *INDEX*)

        Returns
        -------
        num_ss_faces : int
        """
        raise NotImplementedError

    def num_faces_per_elem(self, block_id):
        """Get the number of faces per element for an element block

        Parameters
        ----------
        block_id: int
            element block ID(not INDEX)

        Returns
        -------
        num_elem_faces : int

        """
        block_iid = self.get_element_block_iid(block_id)
        return self.get_dimension(ex.DIM_NUM_FACE_PER_ELEM(block_iid))

    def num_info_records(self):
        """Get the number of info records

        Returns
        -------
        num_info_recs : int
        """
        return self.get_dimension(ex.DIM_NUM_INFO)

    def num_node_maps(self):
        """Number of model node maps"""
        return self.exinit.num_node_maps

    def num_edge_set_dist_fact(self, set_id):
        set_iid = self.get_node_set_iid(set_id)
        return self.get_dimension(ex.DIM_NUM_EDGE_EDGE_SET(set_iid))

    def num_face_set_dist_fact(self, set_id):
        df = self.get_face_set_dist_facts(set_id)
        if df is not None:
            return len(df)

    def num_node_set_dist_fact(self, set_id):
        df = self.get_node_set_dist_facts(set_id)
        if df is not None:
            return len(df)

    def num_side_set_dist_fact(self, set_id):
        df = self.get_side_set_dist_facts(set_id)
        if df is not None:
            return len(df)

    def num_node_sets(self):
        """Get the number of node sets in the model

        Returns
        -------
        num_node_sets : int

        """
        return self.exinit.num_node_sets

    def num_nodes(self):
        """Get the number of nodes in the model

        Returns
        -------
        num_nodes : int
        """
        return self.get_dimension(ex.DIM_NUM_NODES, default=0)

    def num_nodes_in_node_set(self, set_id):
        """Get the number of nodes in a node set

        Parameters
        ----------
        set_id : int
            node set *ID* (not *INDEX*)

        Returns
        -------
        num_ns_nodes : int
        """
        node_set_nodes = self.get_node_set_nodes(set_id)
        return len(node_set_nodes)

    def num_nodes_per_edge(self, block_id):
        """Get the number of nodes per edge for a edge block

        Parameters
        ----------
        block_id: int
            Edge block ID(not INDEX)

        Returns
        -------
        num_edge_nodes : int

        """
        block_iid = self.get_edge_block_iid(block_id)
        return self.get_dimension(ex.DIM_NUM_NODE_PER_EDGE(block_iid))

    def num_nodes_per_elem(self, block_id):
        """Get the number of nodes per element for an element block

        Parameters
        ----------
        block_id: int
            element block ID(not INDEX)

        Returns
        -------
        num_elem_nodes : int

        """
        block_iid = self.get_element_block_iid(block_id)
        return self.get_dimension(ex.DIM_NUM_NODE_PER_ELEM(block_iid))

    def num_nodes_per_face(self, block_id):
        """Get the number of nodes per face for an face block

        Parameters
        ----------
        block_id: int
            Face block ID(not INDEX)

        Returns
        -------
        num_face_nodes : int

        """
        block_iid = self.get_face_block_iid(block_id)
        return self.get_dimension(ex.DIM_NUM_NODE_PER_FACE(block_iid))

    def num_qa_records(self):
        """Get the number of qa records

        Returns
        -------
        num_qa_recs : int
        """
        return self.get_dimension(ex.DIM_NUM_QA)

    def num_side_sets(self):
        """Get the number of side sets in the model

        Returns
        -------
        num_side_sets : int
        """
        return self.exinit.num_side_sets

    def num_sides_in_side_set(self, set_id):
        """Get the number of sides in a side set

        Parameters
        ----------
        set_id : int
            side set *ID* (not *INDEX*)

        Returns
        -------
        num_ss_sides : int
        """
        side_set_sides = self.get_side_set_sides(set_id)
        return len(side_set_sides)

    def num_times(self):
        """Get the number of time steps

        Returns
        -------
        num_times : int
        """
        return len(self.get_times())

    @requires_write_mode
    def put_coord(self, *coords):
        """Write the names of the coordinate arrays

        Parameters
        ----------
        coords: x, y, z : each array_like
            x, y, z coordinates

        """
        num_dim = self.get_dimension(ex.DIM_NUM_DIM)
        for i in range(num_dim):
            coord = (ex.VAR_COORD_X, ex.VAR_COORD_Y, ex.VAR_COORD_Z)[i]
            self.fill_variable(coord, coords[i])

    @requires_write_mode
    def put_coords(self, coords):
        """Write the names of the coordinate arrays

        Parameters
        ----------
        coords: ndarray
            x, y, z coordinates

        """
        num_dim = self.get_dimension(ex.DIM_NUM_DIM)
        for i in range(num_dim):
            coord = (ex.VAR_COORD_X, ex.VAR_COORD_Y, ex.VAR_COORD_Z)[i]
            self.fill_variable(coord, coords[:, i])

    @requires_write_mode
    def put_coord_names(self, coord_names):
        """Writes the names of the coordinate arrays to the database.

        Parameters
        ----------
        coord_names : array_like
            Array containing num_dim names (of length MAX_STR_LENGTH) of the
            nodal coordinate arrays.

        """
        num_dim = self.get_dimension(ex.DIM_NUM_DIM)
        for i in range(num_dim):
            self.fill_variable(
                ex.VAR_NAME_COORD, i, "{0:{1}s}".format(coord_names[i], 32)
            )

    @requires_write_mode
    def put_edge_block(
        self, block_id, elem_type, num_block_edges, num_nodes_per_edge, num_attr
    ):
        """Write parameters used to describe an edge block

        Parameters
        ----------
        block_id : int
            The edge block ID.
        elem_type : str
            The type of elements in the edge block. The maximum length of
            this string is MAX_STR_LENGTH. For historical reasons, this
            string should be all upper case.
        num_block_edges : int
            The number of edges in the edge block.
        num_nodes_per_edge : int
            Number of nodes per edge in the edge block
        num_attr : int
            The number of attributes per edge in the edge block.

        """
        counter = "ed"
        num_blocks = self.get_dimension(ex.DIM_NUM_EDGE_BLK)
        num_blocks_counted = self._counter.setdefault(counter, 0)
        block_iid = num_blocks_counted + 1
        if block_iid > num_blocks:
            raise ValueError("Allocated number of edge blocks exceeded")

        nums = (num_block_edges, num_nodes_per_edge)
        dims = (
            ex.DIM_NUM_EDGE_IN_EDGE_BLK(block_iid),
            ex.DIM_NUM_NODE_PER_EDGE(block_iid),
        )

        self.fh.variables[ex.VAR_ID_EDGE_BLK][num_blocks_counted] = block_id
        self.create_dimension(dims[0], nums[0])
        self.create_dimension(dims[1], nums[1])

        # set up the edge block connectivity
        var = ex.VAR_EDGE_BLK_CONN(block_iid)
        self.create_variable(var, int, (dims[0], dims[1]))
        self.setncattr(var, ex.ATT_NAME_ELEM_TYPE, elem_type.upper())

        conn = np.zeros(nums)
        self.fill_variable(var, conn)

        # edge block attributes
        if num_attr:
            dim = ex.DIM_NUM_ATT_IN_EDGE_BLK(block_iid)
            var = ex.VAR_EDGE_BLK_ATTRIB(block_iid)
            self.create_dimension(dim, num_attr)
            self.create_variable(var, float, (dims[0], dim))
            self.fill_variable(var, np.zeros(num_attr))

            var = ex.VAR_NAME_EDGE_BLK_ATTRIB(block_iid)
            self.create_variable(var, str, (dims[1], ex.DIM_STR))
            self.fill_variable(var, " " * 32)

        self._counter[counter] += 1

    @requires_write_mode
    def put_edge_conn(self, block_id, connect):
        """writes the connectivity array for a edge block

        Parameters
        ----------
        block_id : int
            The edge block ID
        connect : array_like
            Connectivity array, list of nodes that define each edge in the
            block.

        """
        block_iid = self.get_edge_block_iid(block_id)
        num_row = self.get_dimension(ex.DIM_NUM_EDGE_IN_EDGE_BLK(block_iid))
        num_col = self.get_dimension(ex.DIM_NUM_NODE_PER_EDGE(block_iid))

        if connect.shape != (num_row, num_col):
            raise ValueError(f"incorrect edge connect shape for block {block_id}")

        # connectivity
        self.fill_variable(ex.VAR_EDGE_BLK_CONN(block_iid), connect)

    @requires_write_mode
    def put_edge_id_map(self, edge_num_map):
        """Writes out the optional edge order map to the database

        Parameters
        ----------
        edge_map : array_like
            The edge map

        Note
        ----
        The following code generates a default edge order map and outputs
        it to an open EXODUS II file. This is a trivial case and included just
        for illustration. Since this map is optional, it should be written out
        only if it contains something other than the default map.

        edge_map = []
        for i in range(num_edge):
            edge_map.append(i+1)

        """
        num_edge = self.get_dimension(ex.DIM_NUM_EDGE)
        if len(edge_num_map) > num_edge:
            raise ValueError("len(edge_map) > num_edge")
        self.create_variable(ex.VAR_EDGE_NUM_MAP, int, (ex.DIM_NUM_EDGE,))
        self.fill_variable(ex.VAR_EDGE_NUM_MAP, edge_num_map)

    @requires_write_mode
    def put_edge_variable_truth_table(self, edge_var_tab):
        """Writes the EXODUS II edge variable truth table to the database.

        The edge variable truth table indicates whether a particular
        edge result is written for the edges in a particular edge
        block. A 0 (zero) entry indicates that no results will be output for
        that edge variable for that edge block. A non-zero entry
        indicates that the appropriate results will be output.

        Parameters
        ----------
        edge_var_tab : array_like, (num_edge_blk, num_edge_var)
             A 2-dimensional array containing the edge variable truth
             table.

        Note
        ----
        Although writing the edge variable truth table is optional, it is
        encouraged because it creates at one time all the necessary netCDF
        variables in which to hold the EXODUS edge variable values. This
        results in significant time savings. See Appendix A for a discussion
        of efficiency issues. Calling the function put_var_tab with an
        object type of "E" results in the same behavior as calling this
        function.

        The function put_var_param (or EXPVP for Fortran) must be called
        before this routine in order to define the number of edge
        variables.

        """
        num_edge_blk, num_edge_var = edge_var_tab.shape
        if num_edge_blk != self.get_dimension(ex.DIM_NUM_EDGE_BLK):
            raise ValueError("Incorrect number of edge blocks")
        if num_edge_var != self.get_dimension(ex.DIM_NUM_EDGE_VAR):
            raise ValueError("Incorrect number of edge variables")
        self.create_variable(
            ex.VAR_EDGE_BLK_TAB, int, (ex.DIM_NUM_EDGE_BLK, ex.DIM_NUM_EDGE_VAR)
        )
        self.fill_variable(ex.VAR_EDGE_BLK_TAB, edge_var_tab)

    @requires_write_mode
    def put_edge_variable_names(self, names):
        """Writes the names of the edge results variables to the database.

        Note
        ----
        The function put_edge_variable_params must be called before this function is
        invoked.
        """
        names = ["{0:{1}s}".format(x, 32)[:32] for x in names]
        for (i, name) in enumerate(names):
            self.fill_variable(ex.VAR_NAME_EDGE_VAR, i, name)

    @requires_write_mode
    def put_edge_variable_params(self, num_vars):
        """Writes the number of edge variables that will be written to the database.

        Parameters
        ----------
        num_vars : int
            The number of node variables that will be written to the database.
        """
        self.create_dimension(ex.DIM_NUM_EDGE_VAR, num_vars)
        self.create_variable(
            ex.VAR_NAME_EDGE_VAR, str, (ex.DIM_NUM_EDGE_VAR, ex.DIM_STR)
        )
        for block_id in self.get_edge_block_ids():
            block_iid = self.get_edge_block_iid(block_id)
            for i in range(num_vars):
                name = ex.VAR_EDGE_VAR(i + 1, block_iid)
                shape = (ex.DIM_TIME, ex.DIM_NUM_EDGE_IN_EDGE_BLK(block_iid))
                self.create_variable(name, float, shape)

    @requires_write_mode
    def put_edge_variable_values(self, time_step, block_id, name, values):
        """Writes the values of a single edge variable for a single time
        step.

        Parameters
        ----------
        time_step : int
            The time step number, as described under put_time. This is
            essentially a counter that is incremented when results variables
            are output. The first time step is 1.
        block_id : int
            The edge block ID
        name : str
            The name of the variable
        values : array_like
            Array of num_edge_this_blk values of the edge_var_indexth edge
            variable for the edge block with ID of block_id at the
            time_stepth time step

        Note
        ----
        The function put_edge_variable_params must be invoked before this call is
        made.

        It is recommended, but not required, to write the edge variable
        truth table before this function is invoked for better efficiency.

        """
        names = self.get_edge_variable_names()
        var_iid = self.get_iid(names, name)
        block_iid = self.get_edge_block_iid(block_id)
        key = ex.VAR_EDGE_VAR(var_iid, block_iid)
        if time_step is None:
            self.fh.variables[key][:] = values
        else:
            self.fh.variables[key][time_step - 1, : len(values)] = values

    @requires_write_mode
    def put_element_attr(self, block_id, attr):
        """writes the attribute to the

        Parameters
        ----------
        block_id : int
            The element block ID

        attr : array_like, (num_elem_this_block, num_attr)
            List of attributes for the element block

        """
        block_iid = self.get_element_block_iid(block_id)
        self.fill_variable(ex.VAR_ELEM_ATTRIB(block_iid), attr)

    @requires_write_mode
    def put_element_block(
        self,
        block_id,
        elem_type,
        num_block_elems,
        num_nodes_per_elem,
        num_faces_per_elem=None,
        num_edges_per_elem=None,
        num_attr=None,
    ):
        """Write parameters used to describe an element block

        Parameters
        ----------
        block_id : int
            The element block ID.
        elem_type : str
            The type of elements in the element block. The maximum length of
            this string is MAX_STR_LENGTH. For historical reasons, this
            string should be all upper case.
        num_block_elems : int
            The number of elements in the element block.
        num_nodes_per_elem : int
            Number of nodes per element in the element block
        num_faces_per_elem : int
            Number of faces per element in the element block
        num_edges_per_elem : int
            Number of edges per element in the element block
        num_attr : int
            The number of attributes per element in the element block.

        """
        counter = "eb"
        num_blocks = self.get_dimension(ex.DIM_NUM_ELEM_BLK)
        num_blocks_counted = self._counter.setdefault(counter, 0)
        block_iid = num_blocks_counted + 1
        if block_iid > num_blocks:
            raise ValueError("Allocated number of element blocks exceeded")

        nums = (
            num_block_elems,
            num_nodes_per_elem,
            num_edges_per_elem,
            num_faces_per_elem,
        )
        dims = (
            ex.DIM_NUM_ELEM_IN_ELEM_BLK(block_iid),
            ex.DIM_NUM_NODE_PER_ELEM(block_iid),
            ex.DIM_NUM_EDGE_PER_ELEM(block_iid),
            ex.DIM_NUM_FACE_PER_ELEM(block_iid),
        )

        self.fh.variables[ex.VAR_ID_ELEM_BLK][num_blocks_counted] = block_id
        for (i, dim) in enumerate(dims):
            if nums[i] is not None:
                self.create_dimension(dim, nums[i])

        # set up the element block connectivity
        var = ex.VAR_ELEM_BLK_CONN(block_iid)
        self.create_variable(var, int, (dims[0], dims[1]))
        self.setncattr(var, ex.ATT_NAME_ELEM_TYPE, elem_type.upper())
        self.fill_variable(var, np.zeros((nums[0], nums[1]), dtype=int))

        if num_edges_per_elem:
            var = ex.VAR_EDGE_CONN(block_iid)
            self.create_variable(var, int, (dims[0], dims[2]))
            self.fill_variable(var, np.zeros((nums[0], nums[2]), dtype=int))

        if num_faces_per_elem:
            var = ex.VAR_FACE_CONN(block_iid)
            self.create_variable(var, int, (dims[0], dims[3]))
            self.fill_variable(var, np.zeros((nums[0], nums[3]), dtype=int))

        # element block attributes
        if num_attr:
            dim = ex.DIM_NUM_ATT_IN_ELEM_BLK(block_iid)
            var = ex.VAR_ELEM_BLK_ATTRIB(block_iid)
            self.create_dimension(dim, num_attr)
            self.create_variable(var, float, (dims[0], dim))
            self.fill_variable(var, np.zeros(num_attr))

            var = ex.VAR_NAME_ELEM_BLK_ATTRIB(block_iid)
            self.create_variable(var, str, (dims[1], ex.DIM_STR))
            self.fill_variable(var, " " * 32)

        self._counter[counter] += 1

    @requires_write_mode
    def put_element_block_names(self, names):
        """store a list of all element block names ordered by block INDEX
        (see get_element_block_ids for an explanation of the difference
        between block id and index)

        Returns
        -------
        elem_blk_names : list<string>
        
        """
        names = ["{0:{1}s}".format(x, 32)[:32] for x in names]
        block_ids = self.get_element_block_ids()

        for (block_id, name) in zip(block_ids, names):
            block_iid = self.get_element_block_iid(block_id)
            self.fill_variable(ex.VAR_NAME_ELEM_BLK, block_iid-1, name)

    @requires_write_mode
    def put_element_block_name(self, block_id, name):
        """store the element block name

        Parameters
        ----------
        block_id : ex_entity_id
            element block *ID* (not *INDEX*)
        name : string
        
        """
        name = "{0:{1}s}".format(name, 32)[:32]
        block_iid = self.get_element_block_iid(block_id)
        self.fill_variable(ex.VAR_NAME_ELEM_BLK, block_iid-1, name)

    @requires_write_mode
    def put_element_conn(self, block_id, connect, type=ex.types.node):
        """writes the connectivity array for an element block

        Parameters
        ----------
        block_id : int
            The element block ID
        connect : array_like
            Connectivity array, list of nodes that define each element in the
            block.

        """
        block_iid = self.get_element_block_iid(block_id)
        num_row = self.get_dimension(ex.DIM_NUM_ELEM_IN_ELEM_BLK(block_iid))
        if type == ex.types.node:
            var = ex.VAR_ELEM_BLK_CONN(block_iid)
            num_col = self.get_dimension(ex.DIM_NUM_NODE_PER_ELEM(block_iid))
        elif type == ex.types.edge:
            var = ex.VAR_EDGE_CONN(block_iid)
            num_col = self.get_dimension(ex.DIM_NUM_EDGE_PER_ELEM(block_iid))
        elif type == ex.types.face:
            var = ex.VAR_FACE_CONN(block_iid)
            num_col = self.get_dimension(ex.DIM_NUM_FACE_PER_ELEM(block_iid))
        else:
            raise ValueError(f"Invalid element connectivity type {type!r}")

        if connect.shape != (num_row, num_col):
            raise ValueError(f"incorrect element connect shape for block {block_id}")

        # connectivity
        self.fill_variable(var, connect)

    @requires_write_mode
    def put_element_id_map(self, elem_num_map):
        """Writes out the optional element order map to the database

        Parameters
        ----------
        elem_map : array_like
            The element map

        Note
        ----
        The following code generates a default element order map and outputs
        it to an open EXODUS II file. This is a trivial case and included just
        for illustration. Since this map is optional, it should be written out
        only if it contains something other than the default map.

        elem_map = []
        for i in range(num_elem):
            elem_map.append(i+1)

        """
        num_elem = self.get_dimension(ex.DIM_NUM_ELEM)
        if len(elem_num_map) > num_elem:
            raise ValueError("len(elem_map) > num_elem")
        self.create_variable(ex.VAR_ELEM_NUM_MAP, int, (ex.DIM_NUM_ELEM,))
        self.fill_variable(ex.VAR_ELEM_NUM_MAP, elem_num_map)

    @requires_write_mode
    def put_element_variable_truth_table(self, elem_var_tab):
        """Writes the EXODUS II element variable truth table to the database.

        The element variable truth table indicates whether a particular
        element result is written for the elements in a particular element
        block. A 0 (zero) entry indicates that no results will be output for
        that element variable for that element block. A non-zero entry
        indicates that the appropriate results will be output.

        Parameters
        ----------
        elem_var_tab : array_like, (num_elem_blk, num_elem_var)
             A 2-dimensional array containing the element variable truth
             table.

        Note
        ----
        Although writing the element variable truth table is optional, it is
        encouraged because it creates at one time all the necessary netCDF
        variables in which to hold the EXODUS element variable values. This
        results in significant time savings. See Appendix A for a discussion
        of efficiency issues. Calling the function put_var_tab with an
        object type of "E" results in the same behavior as calling this
        function.

        The function put_var_param (or EXPVP for Fortran) must be called
        before this routine in order to define the number of element
        variables.

        """
        num_elem_blk, num_elem_var = elem_var_tab.shape
        if num_elem_blk != self.get_dimension(ex.DIM_NUM_ELEM_BLK):
            raise ValueError("Incorrect number of element blocks")
        if num_elem_var != self.get_dimension(ex.DIM_NUM_ELEM_VAR):
            raise ValueError("Incorrect number of element variables")
        self.create_variable(
            ex.VAR_ELEM_TAB, int, (ex.DIM_NUM_ELEM_BLK, ex.DIM_NUM_ELEM_VAR)
        )
        self.fill_variable(ex.VAR_ELEM_TAB, elem_var_tab)

    @requires_write_mode
    def put_element_variable_names(self, names):
        """Writes the names of the element results variables to the database.

        Note
        ----
        The function put_element_variable_params must be called before this function is
        invoked.
        """
        names = ["{0:{1}s}".format(x, 32)[:32] for x in names]
        for (i, name) in enumerate(names):
            self.fill_variable(ex.VAR_NAME_ELEM_VAR, i, name)

    @requires_write_mode
    def put_element_variable_params(self, num_vars):
        """Writes the number of node variables that will be written to the database.

        Parameters
        ----------
        num_vars : int
            The number of element variables that will be written to the database.
        """
        self.create_dimension(ex.DIM_NUM_ELEM_VAR, num_vars)
        self.create_variable(
            ex.VAR_NAME_ELEM_VAR, str, (ex.DIM_NUM_ELEM_VAR, ex.DIM_STR)
        )
        for block_id in self.get_element_block_ids():
            block_iid = self.get_element_block_iid(block_id)
            for i in range(num_vars):
                name = ex.VAR_ELEM_VAR(i + 1, block_iid)
                shape = (ex.DIM_TIME, ex.DIM_NUM_ELEM_IN_ELEM_BLK(block_iid))
                self.create_variable(name, float, shape)

    @requires_write_mode
    def put_element_variable_values(self, time_step, block_id, name, values):
        """Writes the values of a single elemental variable for a single time
        step.

        Parameters
        ----------
        time_step : int
            The time step number, as described under put_time. This is
            essentially a counter that is incremented when results variables
            are output. The first time step is 1.
        block_id : int
            The element block ID
        name : str
            The name of the variable
        values : array_like
            Array of num_elem_this_blk values of the elem_var_indexth element
            variable for the element block with ID of block_id at the
            time_stepth time step

        Note
        ----
        The function put_element_variable_params must be invoked before this call is
        made.

        It is recommended, but not required, to write the element variable
        truth table before this function is invoked for better efficiency.

        """
        names = self.get_element_variable_names()
        var_iid = self.get_iid(names, name)
        block_iid = self.get_element_block_iid(block_id)
        key = ex.VAR_ELEM_VAR(var_iid, block_iid)
        if time_step is None:
            self.fh.variables[key][:] = values
        else:
            self.fh.variables[key][time_step - 1, : len(values)] = values

    @requires_write_mode
    def put_face_block(
        self, block_id, elem_type, num_block_faces, num_nodes_per_face, num_attr
    ):
        """Write parameters used to describe an face block

        Parameters
        ----------
        block_id : int
            The face block ID.
        elem_type : str
            The type of elements in the face block. The maximum length of
            this string is MAX_STR_LENGTH. For historical reasons, this
            string should be all upper case.
        num_block_faces : int
            The number of faces in the face block.
        num_nodes_per_face : int
            Number of nodes per face in the face block
        num_attr : int
            The number of attributes per face in the face block.

        """
        counter = "fa"
        num_blocks = self.get_dimension(ex.DIM_NUM_FACE_BLK)
        num_blocks_counted = self._counter.setdefault(counter, 0)
        block_iid = num_blocks_counted + 1
        if block_iid > num_blocks:
            raise ValueError("Allocated number of face blocks exceeded")

        nums = (num_block_faces, num_nodes_per_face)
        dims = (
            ex.DIM_NUM_FACE_IN_FACE_BLK(block_iid),
            ex.DIM_NUM_NODE_PER_FACE(block_iid),
        )

        self.fh.variables[ex.VAR_ID_FACE_BLK][num_blocks_counted] = block_id
        self.create_dimension(dims[0], nums[0])
        self.create_dimension(dims[1], nums[1])

        # set up the face block connectivity
        var = ex.VAR_FACE_BLK_CONN(block_iid)
        self.create_variable(var, int, (dims[0], dims[1]))
        self.setncattr(var, ex.ATT_NAME_ELEM_TYPE, elem_type.upper())

        conn = np.zeros(nums)
        self.fill_variable(var, conn)

        # face block attributes
        if num_attr:
            dim = ex.DIM_NUM_ATT_IN_FACE_BLK(block_iid)
            var = ex.VAR_FACE_ATTRIB(block_iid)
            self.create_dimension(dim, num_attr)
            self.create_variable(var, float, (dims[0], dim))
            self.fill_variable(var, np.zeros(num_attr))

            var = ex.VAR_NAME_FACE_BLK_ATTRIB(block_iid)
            self.create_variable(var, str, (dims[1], ex.DIM_STR))
            self.fill_variable(var, " " * 32)

        self._counter[counter] += 1

    @requires_write_mode
    def put_face_conn(self, block_id, connect):
        """writes the connectivity array for a face block

        Parameters
        ----------
        block_id : int
            The face block ID
        connect : array_like
            Connectivity array, list of nodes that define each face in the
            block.

        """
        block_iid = self.get_face_block_iid(block_id)
        num_row = self.get_dimension(ex.DIM_NUM_FACE_IN_FACE_BLK(block_iid))
        num_col = self.get_dimension(ex.DIM_NUM_NODE_PER_FACE(block_iid))

        if connect.shape != (num_row, num_col):
            raise ValueError(f"incorrect face connect shape for block {block_id}")

        # connectivity
        self.fill_variable(ex.VAR_FACE_BLK_CONN(block_iid), connect)

    @requires_write_mode
    def put_face_id_map(self, face_num_map):
        """Writes out the optional face order map to the database

        Parameters
        ----------
        face_map : array_like
            The face map

        Note
        ----
        The following code generates a default face order map and outputs
        it to an open EXODUS II file. This is a trivial case and included just
        for illustration. Since this map is optional, it should be written out
        only if it contains something other than the default map.

        face_map = []
        for i in range(num_face):
            face_map.append(i+1)

        """
        num_face = self.get_dimension(ex.DIM_NUM_FACE)
        if len(face_num_map) > num_face:
            raise ValueError("len(face_map) > num_face")
        self.create_variable(ex.VAR_FACE_NUM_MAP, int, (ex.DIM_NUM_FACE,))
        self.fill_variable(ex.VAR_FACE_NUM_MAP, face_num_map)

    @requires_write_mode
    def put_face_variable_truth_table(self, face_var_tab):
        """Writes the EXODUS II face variable truth table to the database.

        The face variable truth table indicates whether a particular
        face result is written for the faces in a particular face
        block. A 0 (zero) entry indicates that no results will be output for
        that face variable for that face block. A non-zero entry
        indicates that the appropriate results will be output.

        Parameters
        ----------
        face_var_tab : array_like, (num_face_blk, num_face_var)
             A 2-dimensional array containing the face variable truth
             table.

        Note
        ----
        Although writing the face variable truth table is optional, it is
        encouraged because it creates at one time all the necessary netCDF
        variables in which to hold the EXODUS face variable values. This
        results in significant time savings. See Appendix A for a discussion
        of efficiency issues. Calling the function put_var_tab with an
        object type of "E" results in the same behavior as calling this
        function.

        The function put_var_param (or EXPVP for Fortran) must be called
        before this routine in order to define the number of face
        variables.

        """
        num_face_blk, num_face_var = face_var_tab.shape
        if num_face_blk != self.get_dimension(ex.DIM_NUM_FACE_BLK):
            raise ValueError("Incorrect number of face blocks")
        if num_face_var != self.get_dimension(ex.DIM_NUM_FACE_VAR):
            raise ValueError("Incorrect number of face variables")
        self.create_variable(
            ex.VAR_FACE_BLK_TAB, int, (ex.DIM_NUM_FACE_BLK, ex.DIM_NUM_FACE_VAR)
        )
        self.fill_variable(ex.VAR_FACE_BLK_TAB, face_var_tab)

    @requires_write_mode
    def put_face_variable_names(self, names):
        """Writes the names of the face results variables to the database.

        Note
        ----
        The function put_face_variable_params must be called before this function is
        invoked.
        """
        names = ["{0:{1}s}".format(x, 32)[:32] for x in names]
        for (i, name) in enumerate(names):
            self.fill_variable(ex.VAR_NAME_FACE_VAR, i, name)

    @requires_write_mode
    def put_face_variable_params(self, num_vars):
        """Writes the number of face variables that will be written to the database.

        Parameters
        ----------
        num_vars : int
            The number of node variables that will be written to the database.
        """
        self.create_dimension(ex.DIM_NUM_FACE_VAR, num_vars)
        self.create_variable(
            ex.VAR_NAME_FACE_VAR, str, (ex.DIM_NUM_FACE_VAR, ex.DIM_STR)
        )
        for block_id in self.get_face_block_ids():
            block_iid = self.get_face_block_iid(block_id)
            for i in range(num_vars):
                name = ex.VAR_FACE_VAR(i + 1, block_iid)
                shape = (ex.DIM_TIME, ex.DIM_NUM_FACE_IN_FACE_BLK(block_iid))
                self.create_variable(name, float, shape)

    @requires_write_mode
    def put_face_variable_values(self, time_step, block_id, name, values):
        """Writes the values of a single face variable for a single time
        step.

        Parameters
        ----------
        time_step : int
            The time step number, as described under put_time. This is
            essentially a counter that is incremented when results variables
            are output. The first time step is 1.
        block_id : int
            The face block ID
        name : str
            The name of the variable
        values : array_like
            Array of num_face_this_blk values of the face_var_indexth face
            variable for the face block with ID of block_id at the
            time_stepth time step

        Note
        ----
        The function put_face_variable_params must be invoked before this call is
        made.

        It is recommended, but not required, to write the face variable
        truth table before this function is invoked for better efficiency.

        """
        names = self.get_face_variable_names()
        var_iid = self.get_iid(names, name)
        block_iid = self.get_face_block_iid(block_id)
        key = ex.VAR_FACE_VAR(var_iid, block_iid)
        if time_step is None:
            self.fh.variables[key][:] = values
        else:
            self.fh.variables[key][time_step - 1, : len(values)] = values

    @requires_write_mode
    def put_global_variable_names(self, names):
        """Writes the names of the global results variables to the database.

        Note
        ----
        The function put_global_variable_params must be called before this function is
        invoked.
        """
        for (i, name) in enumerate(names):
            self.fill_variable(ex.VAR_NAME_GLO_VAR, i, "{0:{1}s}".format(name, 32))

    @requires_write_mode
    def put_global_variable_params(self, num_vars):
        """Writes the number of global variables that will be written to the database.

        Parameters
        ----------
        num_vars : int
            The number of global variables that will be written to the database.
        """
        self.create_dimension(ex.DIM_NUM_GLO_VAR, num_vars)
        self.create_variable(
            ex.VAR_NAME_GLO_VAR, str, (ex.DIM_NUM_GLO_VAR, ex.DIM_NAME)
        )
        self.create_variable(ex.VAR_GLO_VAR, float, (ex.DIM_TIME, ex.DIM_NUM_GLO_VAR))

    @requires_write_mode
    def put_global_variable_values(self, time_step, vals_glo_var):
        """Writes the values of all the global variables for a single time step.

        time_step : int
            The time step number, as described under put_time.
            This is essentially a counter that is incremented when results
            variables are output. The first time step is 1.
        glob_var_vals : array_like
            Array of num_glob_vars global variable values for the time_stepth
            time step.

        Note
        ----
        The function put_global_variable_names must be invoked before this call is
        made.

        """
        num_glo_var = len(vals_glo_var)
        if time_step is None:
            self.fh.variables[ex.VAR_GLO_VAR][:] = vals_glo_var
        else:
            self.fh.variables[ex.VAR_GLO_VAR][
                time_step - 1, :num_glo_var
            ] = vals_glo_var

    @requires_write_mode
    def put_info(self, num_info, info):
        """Writes information records to the database. The records are
        MAX_LINE_LENGTH-character strings.

        Parameters
        ----------
        info : array_like, (num_info, )
            Array containing the information records

        """
        """Reads/writes information records to the database"""
        num_info = len(info)
        self.create_dimension(ex.DIM_NUM_INFO, num_info)
        self.create_variable(ex.VAR_INFO, str, (ex.DIM_NUM_INFO, ex.DIM_LIN))
        for (i, info_record) in enumerate(info):
            self.fh.variables[ex.VAR_INFO][i] = "{0:{1}s}".format(info_record, 80)

    @requires_write_mode
    def put_init(
        self,
        title,
        num_dim,
        num_nodes,
        num_elem,
        num_elem_blk,
        num_node_sets,
        num_side_sets,
        num_edge=None,
        num_edge_blk=None,
        num_face=None,
        num_face_blk=None,
    ):
        """Writes the initialization parameters to the EXODUS II file

        Parameters
        ----------
        title : str
            Title
        num_dim : int
            Number of spatial dimensions [1, 2, 3]
        num_nodes : int
            Number of nodes
        num_elem : int
            Number of elements
        num_elem_blk : int
            Number of element blocks
        num_node_sets : int
            Number of node sets
        num_side_sets : int
            Number of side sets
        """

        self.setattr(ex.ATT_TITLE, title or "")

        # Create required dimensions
        self.create_dimension(ex.DIM_NUM_DIM, num_dim)
        if num_nodes > 0:
            self.create_dimension(ex.DIM_NUM_NODES, num_nodes)
        if num_elem > 0:
            self.create_dimension(ex.DIM_NUM_ELEM, num_elem)

        # element block meta data
        self.allocate_element_blocks(num_elem_blk)

        # node set meta data
        self.allocate_node_sets(num_node_sets)

        # side set meta data
        self.allocate_side_sets(num_side_sets)

        if num_edge:
            self.create_dimension(ex.DIM_NUM_EDGE, num_edge)
            self.allocate_edge_blocks(num_edge_blk)

        if num_face:
            self.create_dimension(ex.DIM_NUM_FACE, num_face)
            self.allocate_face_blocks(num_face_blk)

        if num_nodes:
            self.create_variable(ex.VAR_NAME_COORD, str, (ex.DIM_NUM_DIM, ex.DIM_STR))
            coords = (ex.VAR_COORD_X, ex.VAR_COORD_Y, ex.VAR_COORD_Z)
            for i in range(num_dim):
                self.create_variable(coords[i], float, (ex.DIM_NUM_NODES,))
            self.put_coord_names([coords[i] for i in range(num_dim)])

    @requires_write_mode
    def put_map(self, elem_map):
        """Writes out the optional element order map to the database

        Parameters
        ----------
        elem_map : array_like
            The element map

        Note
        ----
        The following code generates a default element order map and outputs
        it to an open EXODUS II file. This is a trivial case and included just
        for illustration. Since this map is optional, it should be written out
        only if it contains something other than the default map.

        elem_map = []
        for i in range(num_elem):
            elem_map.append(i+1)

        """
        num_elem = self.get_dimension(ex.DIM_NUM_ELEM)
        if len(elem_map) > num_elem:
            raise ValueError("len(elem_map) > num_elem")
        self.create_variable(ex.VAR_ELEM_MAP, int, (ex.DIM_NUM_ELEM,))
        self.fill_variable(ex.VAR_ELEM_MAP, elem_map)

    @requires_write_mode
    def put_element_map(self, *args):
        return self.put_map(*args)

    @requires_write_mode
    def put_node_id_map(self, node_num_map):
        """Writes out the optional node order map to the database

        Parameters
        ----------
        node_num_map : array_like
            The node map

        Note
        ----
        The following code generates a default node order map and outputs
        it to an open EXODUS II file. This is a trivial case and included just
        for illustration. Since this map is optional, it should be written out
        only if it contains something other than the default map.

        node_map = []
        for i in range(num_node):
            node_map.append(i+1)

        """
        num_node = self.get_dimension(ex.DIM_NUM_NODES)
        if len(node_num_map) > num_node:
            raise ValueError("len(node_map) > num_node")
        self.create_variable(ex.VAR_NODE_NUM_MAP, int, (ex.DIM_NUM_NODES,))
        self.fill_variable(ex.VAR_NODE_NUM_MAP, node_num_map)

    @requires_write_mode
    def put_node_set(self, set_id, nodes):
        """Writes the node set node list for a single node set

        Parameters
        ----------
        set_id : int
            The side set ID.
        nodes : array_like
            Array containing the nodes in the node set. Internal node
            IDs are used in this list

        Note
        ----
        The function put_node_set_param must be called before this routine is
        invoked.

        """
        self.put_node_set_nodes(set_id, nodes)

    @requires_write_mode
    def put_node_set_name(self, set_id, name):
        """Put the name of a node set

        Parameters
        ----------
        set_id : int
            node set *ID* (not *INDEX*)
        name : str
            The node set name
        """
        set_ids = self.get_node_set_ids()
        set_iid = self.get_iid(set_ids, set_id)
        self.fill_variable(ex.VAR_NAME_NODE_SET, set_iid - 1, name)

    @requires_write_mode
    def put_node_set_variable_names(self, names):
        """Writes the names of the node set results variables to the database.

        Note
        ----
        The function put_node_set_variable_params must be called before this function
        is invoked.
        """
        if names is None:
            return
        names = ["{0:{1}s}".format(x, 32)[:32] for x in names]
        for (i, name) in enumerate(names):
            self.fill_variable(ex.VAR_NAME_NODE_SET_VAR, i, name)

    @requires_write_mode
    def put_node_set_variable_params(self, num_vars):
        """Writes the number of node set variables that will be written to the database.

        Parameters
        ----------
        num_vars : int
            The number of node variables that will be written to the database.
        """
        if not num_vars:
            return
        self.create_dimension(ex.DIM_NODE_SET_VAR, num_vars)
        self.create_variable(
            ex.VAR_NAME_NODE_SET_VAR, str, (ex.DIM_NODE_SET_VAR, ex.DIM_STR)
        )
        for set_id in self.get_node_set_ids():
            set_iid = self.get_node_set_iid(set_id)
            for i in range(num_vars):
                name = ex.VAR_NODE_SET_VAR(i + 1, set_iid)
                self.create_variable(name, float, (ex.DIM_TIME, "num_node"))

    @requires_write_mode
    def put_node_set_variable_values(self, time_step, set_id, name, values):
        """Writes the values of a single node set variable for a single time
        step.

        Parameters
        ----------
        time_step : int
            The time step number, as described under put_time. This is
            essentially a counter that is incremented when results variables
            are output. The first time step is 1.
        set_id : int
            The node set ID
        name : str
            The name of the variable
        values : array_like
            Array of num_edge_this_blk values of the edge_var_indexth edge
            variable for the edge block with ID of set_id at the
            time_stepth time step

        Note
        ----
        The function put_node_set_variable_params must be invoked before this call is
        made.

        """
        names = self.get_node_set_variable_names()
        var_iid = self.get_iid(names, name)
        set_iid = self.get_edge_block_iid(set_id)
        key = ex.VAL_NODE_SET_VAR(var_iid, set_iid)
        self.fh.variables[key][time_step - 1, : len(values)] = values

    @requires_write_mode
    def put_node_set_nodes(self, set_id, node_set_nodes):
        """Writes the node list for a single node set.

        Parameters
        ----------
        node_ set_id : int
            The node set ID.
        node_set_nodes : array_like
            Array containing the node list for the node set. Internal node IDs
            are used in this list.

        Note
        ----
        The function put_node_set_param must be called before this routine is
        invoked.

        """
        set_iid = self.get_node_set_iid(set_id)
        self.fill_variable(ex.VAR_NODE_NODE_SET(set_iid), node_set_nodes)

    @requires_write_mode
    def put_node_set_dist_fact(self, set_id, node_set_dist_fact):
        """Writes distribution factors for a single node set

        Parameters
        ----------
        node_ set_id : int
            The node set ID.

        node_set_dist_fact : array_like
            Array containing the distribution factors for each node in the set

        Note
        ----
        The function put_node_set_param must be called before this routine is
        invoked.

        """
        set_iid = self.get_node_set_iid(set_id)
        dim = self.get_dimension(ex.DIM_NUM_NODE_NODE_SET(set_iid))
        if len(node_set_dist_fact) != dim:
            raise ValueError(
                f"len(node_set_dist_fact) = {len(node_set_dist_fact)} != {dim}"
            )
        self.fill_variable(ex.VAR_DF_NODE_SET(set_iid), node_set_dist_fact)

    @requires_write_mode
    def put_node_set_param(self, set_id, num_nodes_in_set, num_dist_fact_in_set=0):
        """Writes the node set ID, the number of nodes which describe a single
        node set, and the number of distribution factors for the node set.

        Parameters
        ----------
        set_id : int
            The node set ID

        num_nodes_in_set : int
            Number of nodes in set

        num_dist_fact_in_set : int
            The number of distribution factors in the node set. This should be
            either 0 (zero) for no factors, or should equal num_nodes_in_set.

        """
        num_node_sets = self.get_dimension(ex.DIM_NUM_NODE_SET)
        num_counted_ns = self._counter.setdefault("ns", 0)
        set_iid = num_counted_ns + 1
        if set_iid > num_node_sets:
            raise ValueError("Allocated number of node sets exceeded")

        self.fh.variables[ex.VAR_NODE_SET_IDS][num_counted_ns] = int(set_id)
        self.fh.variables[ex.VAR_NODE_SET_STAT][num_counted_ns] = 1

        dim = ex.DIM_NUM_NODE_NODE_SET(set_iid)
        self.create_dimension(dim, num_nodes_in_set)
        self.create_variable(ex.VAR_NODE_NODE_SET(set_iid), int, (dim,))

        if num_dist_fact_in_set:
            assert num_dist_fact_in_set == num_nodes_in_set
            self.create_variable(ex.VAR_DF_NODE_SET(set_iid), float, (dim,))
            self.fill_variable(ex.VAR_DF_NODE_SET(set_iid), np.ones(num_nodes_in_set))

        self._counter["ns"] += 1

    @requires_write_mode
    def put_node_variable_names(self, names):
        """Writes the names of the node results variables to the database.

        Note
        ----
        The function put_node_variable_params must be called before this function is
        invoked.
        """
        names = ["{0:{1}s}".format(x, 32)[:32] for x in names]
        for (i, name) in enumerate(names):
            self.fill_variable(ex.VAR_NAME_NODE_VAR, i, name)

    @requires_write_mode
    def put_node_variable_params(self, num_vars):
        """Writes the number of node variables that will be written to the database.

        Parameters
        ----------
        num_vars : int
            The number of node variables that will be written to the database.
        """
        self.create_dimension(ex.DIM_NUM_NODE_VAR, num_vars)
        self.create_variable(
            ex.VAR_NAME_NODE_VAR, str, (ex.DIM_NUM_NODE_VAR, ex.DIM_STR)
        )
        for i in range(num_vars):
            self.create_variable(
                ex.VAR_NODE_VAR(i + 1), float, (ex.DIM_TIME, ex.DIM_NUM_NODES)
            )

    @requires_write_mode
    def put_node_variable_values(self, time_step, name, values):
        """Writes the values of a single node variable for a single time
        step.

        Parameters
        ----------
        time_step : int
            The time step number, as described under put_time. This is
            essentially a counter that is incremented when results variables
            are output. The first time step is 1.
        name : str
            The node variable name
        values : array_like
            Array of num_nodes values of the node_var_indexth node variable
            for the time_stepth time step.

        Note
        ----
        The function put_var_param must be invoked before this call is made.

        """
        names = self.get_node_variable_names()
        var_iid = self.get_iid(names, name)
        key = ex.VAR_NODE_VAR(var_iid)
        if time_step is None:
            self.fh.variables[key][:] = values
        else:
            self.fh.variables[key][time_step - 1, : len(values)] = values
        return

    @requires_write_mode
    def put_qa(self, num_qa_records, qa_records):
        """Writes the QA records to the database.

        Parameters
        ----------
        num_qa_records : int
            Then number of QA records

        qa_record : array_like, (num_qa_records, 4)
            Array containing the QA records

        Note
        ----
        Each QA record contains for MAX_STR_LENGTH-byte character strings. The
        character strings are

          1) the analysis code name
          2) the analysis code QA descriptor
          3) the analysis date
          4) the analysis time

        """
        self.create_dimension(ex.DIM_NUM_QA, num_qa_records)
        self.create_variable(
            ex.VAR_QA_TITLE, str, (ex.DIM_NUM_QA, ex.DIM_N4, ex.DIM_STR)
        )
        for (i, qa_record) in enumerate(qa_records):
            self.fill_variable(ex.VAR_QA_TITLE, i, 0, qa_record[0])
            self.fill_variable(ex.VAR_QA_TITLE, i, 1, qa_record[1])
            self.fill_variable(ex.VAR_QA_TITLE, i, 2, qa_record[2])
            self.fill_variable(ex.VAR_QA_TITLE, i, 3, qa_record[3])

    @requires_write_mode
    def put_side_set_sides(self, set_id, side_set_elems, side_set_sides):
        """Writes the side set element list and side set side (face on 3-d
        element types; edge on 2-d element types) list for a single side set.

        Parameters
        ----------
        set_id : int
            The side set ID.
        side_set_elems : array_like
            Array containing the elements in the side set. Internal element
            IDs are used in this list
        side_set_sides : array_like
            Array containing the side in the side set

        Note
        ----
        The function put_side_set_param must be called before this routine is
        invoked.

        """
        set_iid = self.get_side_set_iid(set_id)
        if set_iid is None:
            raise ValueError(f"{set_id} is not a side set ID")
        self.fill_variable(ex.VAR_SIDE_SIDE_SET(set_iid), side_set_sides)
        self.fill_variable(ex.VAR_ELEM_SIDE_SET(set_iid), side_set_elems)

    @requires_write_mode
    def put_side_set_name(self, set_id, name):
        """Put the name of a side set

        Parameters
        ----------
        set_id : int
            side set *ID* (not *INDEX*)
        name : str
            The side set name
        """
        name = "{0:{1}s}".format(name, 32)[:32]
        set_ids = self.get_side_set_ids()
        set_iid = self.get_iid(set_ids, set_id)
        self.fill_variable(ex.VAR_NAME_SIDE_SET, set_iid - 1, name)

    @requires_write_mode
    def put_side_set_dist_fact(self, set_id, side_set_dist_fact):
        """Writes distribution factors for a single side set

        Parameters
        ----------
        side_ set_id : int
            The side set ID.

        side_set_dist_fact : array_like
            Array containing the distribution factors for each side in the set

        Note
        ----
        The function put_side_set_param must be called before this routine is
        invoked.

        """
        set_iid = self.get_side_set_iid(set_id)
        if set_iid is None:
            raise ValueError(f"{set_id} is not a side set ID")
        dim = self.get_dimension(ex.DIM_NUM_SIDE_SIDE_SET(set_iid))
        if len(side_set_dist_fact) != dim:
            raise ValueError("len(side_set_dist_fact) incorrect")
        self.fill_variable(ex.VAR_DF_SIDE_SET(set_iid), side_set_dist_fact)

    @requires_write_mode
    def put_side_set_param(self, set_id, num_sides_in_set, num_dist_fact_in_set=0):
        """Writes the side set ID, the number of sides (faces on 3-d element,
        edges on 2-d) which describe a single side set, and the number of
        distribution factors on the side set.

        Parameters
        ----------
        set_id : int
            The side set ID

        num_sides_in_set : int
            Number of sides in set

        num_dist_fact_in_set : int
            The number of distribution factors in the side set. This should be
            either 0 (zero) for no factors, or should equal num_sides_in_set.

        """
        num_side_sets = self.get_dimension(ex.DIM_NUM_SIDE_SET)
        num_counted_ss = self._counter.setdefault("ss", 0)
        set_iid = num_counted_ss + 1
        if set_iid > num_side_sets:
            raise ValueError("Allocated number of side sets exceeded")

        self.fh.variables[ex.VAR_SIDE_SET_IDS][num_counted_ss] = int(set_id)
        self.fh.variables[ex.VAR_SIDE_SET_STAT][num_counted_ss] = 1

        dim = ex.DIM_NUM_SIDE_SIDE_SET(set_iid)
        self.create_dimension(dim, num_sides_in_set)
        self.create_variable(ex.VAR_SIDE_SIDE_SET(set_iid), int, (dim,))
        self.create_variable(ex.VAR_ELEM_SIDE_SET(set_iid), int, (dim,))

        if num_dist_fact_in_set:
            assert num_dist_fact_in_set == num_sides_in_set
            self.create_variable(ex.VAR_DF_SIDE_SET(set_iid), float, (dim,))
            self.fill_variable(ex.VAR_DF_SIDE_SET(set_iid), np.ones(num_sides_in_set))

        self._counter["ss"] += 1

    @requires_write_mode
    def put_side_set_variable_names(self, names):
        """Writes the names of the side set results variables to the database.

        Note
        ----
        The function put_side_set_variable_params must be called before this function
        is invoked.
        """
        if names is None:
            return
        names = ["{0:{1}s}".format(x, 32)[:32] for x in names]
        for (i, name) in enumerate(names):
            self.fill_variable(ex.VAR_NAME_SIDE_SET_VAR, i, name)

    @requires_write_mode
    def put_side_set_variable_params(self, num_vars):
        """Writes the number of side set variables that will be written to the database.

        Parameters
        ----------
        num_vars : int
            The number of side set variables that will be written to the database.
        """
        if not num_vars:
            return
        self.create_dimension(ex.DIM_NUM_SIDE_SET_VAR, num_vars)
        self.create_variable(
            ex.VAR_NAME_SIDE_SET_VAR, str, (ex.DIM_NUM_SIDE_SET_VAR, ex.DIM_STR)
        )
        for set_id in self.get_side_set_ids():
            set_iid = self.get_side_set_iid(set_id)
            for i in range(num_vars):
                name = ex.VAR_SIDE_SET_VAR(i + 1, set_iid)
                self.create_variable(name, float, (ex.DIM_TIME, ex.DIM_NUM_SIDE))

    @requires_write_mode
    def put_side_set_variable_values(self, time_step, set_id, name, values):
        """Writes the values of a single side set variable for a single time
        step.

        Parameters
        ----------
        time_step : int
            The time step number, as described under put_time. This is
            essentially a counter that is incremented when results variables
            are output. The first time step is 1.
        set_id : int
            The side set ID
        name : str
            The name of the variable
        values : array_like
            Array of num_edge_this_blk values of the edge_var_indexth edge
            variable for the edge block with ID of set_id at the
            time_stepth time step

        Note
        ----
        The function put_side_set_variable_params must be invoked before this call is
        made.

        """
        names = self.get_side_set_variable_names()
        var_iid = self.get_iid(names, name)
        set_iid = self.get_edge_block_iid(set_id)
        key = ex.VAR_SIDE_SET_VAR(var_iid, set_iid)
        self.fh.variables[key][time_step - 1, : len(values)] = values

    @requires_write_mode
    def put_time(self, time_step, time_value):
        """Writes the time value for a specified time step.

        Parameters
        ----------
        time_step : int
            The time step number.
            This is essentially a counter that is incremented only when
            results variables are output to the data file. The first time step
            is 1.
        time_value : float
            The time at the specified time step.

        """
        self.fh.variables[ex.VAR_WHOLE_TIME][time_step - 1] = time_value

    @requires_write_mode
    def setattr(self, name, value):
        setattr(self.fh, name, value)

    @requires_write_mode
    def setncattr(self, variable, name, value):
        return nc.setncattr(self.fh, variable, name, value)

    @requires_write_mode
    def allocate_edge_blocks(self, num_edge_blk):

        if not num_edge_blk:
            return

        # block/set meta data
        self.create_dimension(ex.DIM_NUM_EDGE_BLK, num_edge_blk)

        prop1 = np.arange(num_edge_blk, dtype=np.int32)
        self.create_variable(ex.VAR_ID_EDGE_BLK, int, (ex.DIM_NUM_EDGE_BLK,))
        self.fh.variables[ex.VAR_ID_EDGE_BLK][:] = prop1
        self.setncattr(ex.VAR_ID_EDGE_BLK, ex.ATT_PROP_NAME, "ID")

        status = np.ones(num_edge_blk, dtype=np.int32)
        self.create_variable(ex.VAR_STAT_EDGE_BLK, int, (ex.DIM_NUM_EDGE_BLK,))
        self.fill_variable(ex.VAR_STAT_EDGE_BLK, status)

        names = np.array([" " * 32 for _ in prop1])
        self.create_variable(
            ex.VAR_NAME_EDGE_BLK, str, (ex.DIM_NUM_EDGE_BLK, ex.DIM_STR)
        )
        for (i, name) in enumerate(names):
            self.fill_variable(ex.VAR_NAME_EDGE_BLK, i, name)

    @requires_write_mode
    def allocate_element_blocks(self, num_elem_blk):

        if not num_elem_blk:
            return

        # block/set meta data
        self.create_dimension(ex.DIM_NUM_ELEM_BLK, num_elem_blk)

        prop1 = np.arange(num_elem_blk, dtype=np.int32)
        self.create_variable(ex.VAR_ID_ELEM_BLK, int, (ex.DIM_NUM_ELEM_BLK,))
        self.fh.variables[ex.VAR_ID_ELEM_BLK][:] = prop1
        self.setncattr(ex.VAR_ID_ELEM_BLK, ex.ATT_PROP_NAME, "ID")

        status = np.ones(num_elem_blk, dtype=np.int32)
        self.create_variable(ex.VAR_STAT_ELEM_BLK, int, (ex.DIM_NUM_ELEM_BLK,))
        self.fill_variable(ex.VAR_STAT_ELEM_BLK, status)

        names = np.array([" " * 32 for _ in prop1])
        self.create_variable(
            ex.VAR_NAME_ELEM_BLK, str, (ex.DIM_NUM_ELEM_BLK, ex.DIM_STR)
        )
        for (i, name) in enumerate(names):
            self.fill_variable(ex.VAR_NAME_ELEM_BLK, i, name)

    @requires_write_mode
    def allocate_face_blocks(self, num_face_blk):

        if not num_face_blk:
            return

        # block/set meta data
        self.create_dimension(ex.DIM_NUM_FACE_BLK, num_face_blk)

        prop1 = np.arange(num_face_blk, dtype=np.int32)
        self.create_variable(ex.VAR_ID_FACE_BLK, int, (ex.DIM_NUM_FACE_BLK,))
        self.fh.variables[ex.VAR_ID_FACE_BLK][:] = prop1
        self.setncattr(ex.VAR_ID_FACE_BLK, ex.ATT_PROP_NAME, "ID")

        status = np.ones(num_face_blk, dtype=np.int32)
        self.create_variable(ex.VAR_STAT_FACE_BLK, int, (ex.DIM_NUM_FACE_BLK,))
        self.fill_variable(ex.VAR_STAT_FACE_BLK, status)

        names = np.array([" " * 32 for _ in prop1])
        self.create_variable(
            ex.VAR_NAME_FACE_BLK, str, (ex.DIM_NUM_FACE_BLK, ex.DIM_STR)
        )
        for (i, name) in enumerate(names):
            self.fill_variable(ex.VAR_NAME_FACE_BLK, i, name)

    @requires_write_mode
    def allocate_node_sets(self, num_node_sets):

        if not num_node_sets:
            return

        # block/set meta data
        self.create_dimension(ex.DIM_NUM_NODE_SET, num_node_sets)

        prop1 = np.zeros(num_node_sets, dtype=np.int32)
        self.create_variable(ex.VAR_NODE_SET_IDS, int, (ex.DIM_NUM_NODE_SET,))
        self.fh.variables[ex.VAR_NODE_SET_IDS][:] = prop1
        self.setncattr(ex.VAR_NODE_SET_IDS, ex.ATT_PROP_NAME, "ID")

        status = np.zeros(num_node_sets, dtype=np.int32)
        self.create_variable(ex.VAR_NODE_SET_STAT, int, (ex.DIM_NUM_NODE_SET,))
        self.fill_variable(ex.VAR_NODE_SET_STAT, status)

        names = np.array([" " * 32 for _ in prop1])
        self.create_variable(
            ex.VAR_NAME_NODE_SET, str, (ex.DIM_NUM_NODE_SET, ex.DIM_STR)
        )
        for (i, name) in enumerate(names):
            self.fill_variable(ex.VAR_NAME_NODE_SET, i, name)

    @requires_write_mode
    def allocate_side_sets(self, num_side_sets):

        if not num_side_sets:
            return

        # block/set meta data
        self.create_dimension(ex.DIM_NUM_SIDE_SET, num_side_sets)

        prop1 = np.zeros(num_side_sets, dtype=np.int32)
        self.create_variable(ex.VAR_SIDE_SET_IDS, int, (ex.DIM_NUM_SIDE_SET,))
        self.fh.variables[ex.VAR_SIDE_SET_IDS][:] = prop1
        self.setncattr(ex.VAR_SIDE_SET_IDS, ex.ATT_PROP_NAME, "ID")

        status = np.zeros(num_side_sets, dtype=np.int32)
        self.create_variable(ex.VAR_SIDE_SET_STAT, int, (ex.DIM_NUM_SIDE_SET,))
        self.fill_variable(ex.VAR_SIDE_SET_STAT, status)

        names = np.array([" " * 32 for _ in prop1])
        self.create_variable(
            ex.VAR_NAME_SIDE_SET, str, (ex.DIM_NUM_SIDE_SET, ex.DIM_STR)
        )
        for (i, name) in enumerate(names):
            self.fill_variable(ex.VAR_NAME_SIDE_SET, i, name)

    def edge_blocks(self):
        edge_block_ids = self.get_edge_block_ids()
        if edge_block_ids is None:
            edge_block_ids = []
        for block_id in edge_block_ids:
            yield self.get_edge_block(block_id)

    def elem_blocks(self):
        elem_block_ids = self.get_element_block_ids()
        if elem_block_ids is None:
            elem_block_ids = []
        for block_id in elem_block_ids:
            yield self.get_element_block(block_id)

    def face_blocks(self):
        face_block_ids = self.get_face_block_ids()
        if face_block_ids is None:
            face_block_ids = []
        for block_id in face_block_ids:
            yield self.get_face_block(block_id)

    def node_sets(self):
        node_set_ids = self.get_node_set_ids()
        if node_set_ids is None:
            node_set_ids = []
        for set_id in node_set_ids:
            yield self.get_node_set(set_id)

    def side_sets(self):
        side_set_ids = self.get_side_set_ids()
        if side_set_ids is None:
            side_set_ids = []
        for set_id in side_set_ids:
            yield self.get_side_set(set_id)

    def elem_sets(self):
        elem_set_ids = self.get_element_set_ids()
        if elem_set_ids is None:
            elem_set_ids = []
        for set_id in elem_set_ids:
            yield self.get_element_set(set_id)

    def edge_sets(self):
        edge_set_ids = self.get_edge_set_ids()
        if edge_set_ids is None:
            edge_set_ids = []
        for set_id in edge_set_ids:
            yield self.get_edge_set(set_id)

    def face_sets(self):
        face_set_ids = self.get_face_set_ids()
        if face_set_ids is None:
            face_set_ids = []
        for set_id in face_set_ids:
            yield self.get_face_set(set_id)

    @staticmethod
    def make_contiguous(l2g_map):
        """Make a contiguous map from `l2g_map`

        Parameters
        ----------
        l2g_map : dict
            Mapping from (file ID, local ID) -> global ID

        Returns
        -------
        continuous_map : dict
            Mapping from global ID -> contiguous global ID

        Note
        ----
        The GIDs in l2g_map may or may not be contiguous.  They may be more
        appropriately thought of as labels.  They may, for example, not start at 1
        (Exodus convention), have gaps in numbering, etc.  The contiguous map is simply
        the mapping from the (potentially) non-contigous GID (label) to the globally
        contiguous GID.  The globally contigous GIDs are appropriate for use as indexes
        (1-based indexing) in to arrays containing data read in across files.

        Examples
        --------
        >>> l2g_map = {(0, 1): 2, (0, 2): 5, (1, 2): 7, (1, 2): 3}
        >>> MFExodusFile.make_contiguous(l2g_map)
        {2: 1, 3: 2, 5: 3, 7: 4}

        """
        gids = sorted(set(list(l2g_map.values())))
        contigous = dict([(gid, index) for (index, gid) in enumerate(gids, start=1)])
        return contigous


ExodusIIFile = exodusii_file


def write_globals(data, times, title=None, filename="Globals.exo"):
    """Write an exodus file which contains only global variables.

    Parameters
    ----------
    data : dict of ndarray
        data[key] = val, where key is the variable name and val the corresponding values
    times : ndarray
        times (same for all variables in data).

    """

    numtime = len(times)

    # Make the list of variables so we can put them in the header
    # Also, minimal checks on data
    variables = []
    for variable in data:
        # Check that variables are 32 chars or less
        if len(variable) > 32:
            raise ValueError(
                f"Variable name {variable} exceeds the Exodus limit of 32 characters"
            )
        else:
            variables.append(variable)

        # Check that there are the right number of values
        numval = len(data[variable])
        if not numval == numtime:
            raise ValueError(f"len({variable}) = {numval}, not {numtime}")

    # Initial header info.
    with exodusii_file(filename, mode="w") as fh:
        fh.put_init(title or "", 1, 0, 0, 0, 0, 0)
        fh.put_global_variable_params(len(variables))
        fh.put_global_variable_names(variables)

        for (i, time) in enumerate(times):
            time_step = i + 1
            fh.put_time(time_step, time)
            a = np.array([data[variable][i] for variable in variables])
            fh.put_global_variable_values(time_step, a)

    return filename


class UnsupportedOperation(Exception):
    pass
