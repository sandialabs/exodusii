import os
import numpy as np
from argparse import Namespace

from .copy import copy
from .file import exodusii_file
from .ex_params import ex_init_params
from .util import stringify
from .config import config
from . import exodus_h as ex


class parallel_exodusii_file(exodusii_file):
    """Parallel multi-file ExodusII file"""

    def __init__(self, *files):
        self.mode = "r"
        self.files = self.open(*files)

        maxlen = lambda list_like: max(list_like, key=len)
        self._title = maxlen([stringify(_.title) for _ in self.files])
        qa_records = self.get_variable(self.files[0], ex.VAR_QA_TITLE)
        for file in self.files:
            qa_records_this_file = self.get_variable(file, ex.VAR_QA_TITLE)
            if qa_records_this_file is None:
                continue
            elif qa_records is None:
                qa_records = qa_records_this_file
            else:
                qa_records = max([qa_records, qa_records_this_file], key=len)
        self._qa_records = qa_records

        self.exinit = ex_init_params(self.files[0])
        self.cache = {}

    def __contains__(self, name):
        return name in self.files[0].variables

    @property
    def filename(self):
        return ",".join(self.get_filename(f) for f in self.files)

    def open(self, *files):
        self.files = [self._open(file, "r") for file in sorted(files)]
        self.check_consistency(self.files)

        return self.files

    def close(self):
        pass

    def write(self, filename):
        """Write the parallel Exodus file to a single file"""
        target = exodusii_file(filename, mode="w")
        copy(source=self, target=target)
        target.close()
        return target.filename

    def get_dimension(self, *args, **kwargs):
        default = kwargs.get("default", None)
        if len(args) == 1:
            file, name = self.files[0], args[0]
        elif len(args) == 2:
            file, name = args
        else:
            n = len(args)
            raise TypeError(
                f"get_dimension() takes 1 or 2 positional arguments but {n} were given"
            )
        assert isinstance(name, (str, bytes))
        return self._get_dimension(file, name, default)

    def get_variable(self, *args, **kwargs):
        default = kwargs.get("default", None)
        raw = kwargs.get("raw", False)
        if len(args) == 1:
            file, name = self.files[0], args[0]
        elif len(args) == 2:
            file, name = args
        else:
            n = len(args)
            raise TypeError(
                f"get_variable() takes 1 or 2 positional arguments but {n} were given"
            )
        return self._get_variable(file, name, default, raw)

    def variables(self):
        return self.files[0].variables

    def dimensions(self):
        return self.files[0].dimensions

    def storage_type(self):
        word_size = self.files[0].floating_point_word_size
        return "f" if word_size == 4 else "d"

    def title(self):
        """Get the database title

        Returns
        -------
        title : str
        """
        return self._title

    def version_num(self):
        """Get exodus version number used to create the database

        Returns
        -------
        version : str
            string representation of version number
        """
        return f"{self.files[0].version:1.3}"

    def get_coords(self, time_step=None):
        """Get model coordinates of all nodes; for each coordinate direction, a
        length exo.num_nodes() list is returned

        Returns
        -------
        coords : ndarray of float
        """
        coord_names = self.get_coord_variable_names()
        coords = np.zeros((self.num_nodes(), self.num_dimensions()), dtype=float)

        lid_to_gid = self.get_mapping(ex.maps.node_local_to_global, contiguous=True)
        for (fid, file) in enumerate(self.files):
            num_nodes = self.get_dimension(file, ex.DIM_NUM_NODES)
            gids = [lid_to_gid[(fid, lid)] - 1 for lid in range(1, num_nodes + 1)]
            x = np.column_stack([self.get_variable(file, _) for _ in coord_names])
            assert x.shape[0] == num_nodes
            coords[gids, :] = x
        if time_step is not None:
            coords += self.get_displ(time_step)
        return coords

    def get_edge_block_elem_type(self, file, block_id):
        """Get the element type, e.g. "HEX8", for an element block"""
        block_iid = self.f_get_edge_block_iid(file, block_id)
        if not self.edge_block_is_active(file, block_iid):
            return None
        var = self.get_variable(file, ex.VAR_EDGE_BLK_CONN(block_iid), raw=True)
        return None if var is None else var.elem_type

    def get_edge_block_ids(self):
        """get mapping of exodus edge block index to user - or application-defined
        edge block id.

        block_ids is ordered by the edge block INDEX ordering,
        a 1-based system going from 1 to exo.num_blks(), used by exodus for storage
        and input/output of array data stored on the edge blocks a user or
        application can optionally use a separate edge block ID numbering system,
        so the block_ids array points to the edge block ID for each edge
        block INDEX

        Returns
        -------
        block_ids : ndarray of int

        """
        block_ids = []
        for file in self.files:
            file_local_ids = self.get_variable(file, ex.VAR_ID_EDGE_BLK)
            if file_local_ids:
                block_ids.extend(file_local_ids)
        return sorted(set(block_ids))

    def f_get_edge_block_iid(self, file, block_id):
        """get exodus edge block index from id

        Returns
        -------
        block_iid : int

        """
        block_ids = self.get_variable(file, ex.VAR_ID_EDGE_BLK)
        return self.get_iid(block_ids, block_id)

    def get_edge_block(self, block_id):
        """Get the edge block info

        Parameters
        ----------
        block_id : int
            edge block ID (not INDEX)

        Returns
        -------
        elem_type : str
            element type, e.g. 'HEX8'
        num_block_edges : int
            number of edges in the block
        num_edge_nodes : int
            number of nodes per edge
        num_edge_attrs : int
            number of attributes per edge
        """
        if block_id not in self.get_edge_block_ids():
            return None
        for (fid, file) in enumerate(self.files):
            block_iid = self.f_get_edge_block_iid(file, block_id)
            if not self.edge_block_is_active(file, block_iid):
                continue
            elem_type = self.get_edge_block_elem_type(file, block_id)
            info = Namespace(
                id=block_id,
                fid=fid,
                iid=block_iid,
                elem_type=elem_type,
                num_block_edges=self.num_edges_in_blk(block_id),
                num_edge_nodes=self.num_nodes_per_edge(block_id),
                num_edge_attrs=self.num_attr(block_id),
            )
            return info

    def get_edge_block_conn(self, block_id):
        """Get the nodal connectivity for a single block

        Parameters
        ----------
        block_id : int
            Edge block *ID* (not *INDEX*)

        Returns
        -------
        edge_conn : ndarray of int
            define the connectivity of each edge in the block; the list cycles
            through all nodes of the first edge, then all nodes of the second
            edge, etc. (see `exodus.get_id_map` for explanation of node *INDEX*
            versus node *ID*)
        """
        shape = (
            self.num_edges_in_blk(block_id),
            self.num_nodes_per_edge(block_id),
        )
        conn = np.zeros(shape, dtype=int)
        lid_to_gid = self.get_mapping(ex.maps.node_local_to_global, contiguous=True)
        map1 = self.get_mapping(ex.maps.edge_block_edge_local_to_global)
        map2 = self.get_mapping(ex.maps.edge_block_edge_global_to_local)
        for (fid, file) in enumerate(self.files):
            block_iid = self.f_get_edge_block_iid(file, block_id)
            if not self.edge_block_is_active(file, block_iid):
                continue
            file_local_conn = self.get_variable(file, ex.VAR_EDGE_BLK_CONN(block_iid))
            for (lid, nodes) in enumerate(file_local_conn, start=1):
                gid = map1[(fid, block_id, lid)]
                idx = map2[(block_id, gid)] - 1
                conn[idx, :] = [lid_to_gid[(fid, nid)] for nid in nodes]
        return conn

    def get_edge_variable_values(self, block_id, var_name, time_step=None):
        """Get list of edge variable values for a specified edge block, edge
        variable name, and time step

        Parameters
        ----------
        block_id : int
            edge block *ID* (not *INDEX*)
        var_name : str
            name of edge variable
        time_step : int
            1-based index of time step

        Returns
        -------
        evar_vals : ndarray of float
        """
        if block_id is None:
            return self.get_edge_variable_values_across_blocks(
                var_name, time_step=time_step
            )

        names = self.get_edge_variable_names()
        var_iid = self.get_iid(names, var_name)

        shape = (self.num_times(), self.num_edges_in_blk(block_id))
        values = np.zeros(shape, dtype=float)

        map1 = self.get_mapping(ex.maps.edge_block_edge_local_to_global)
        map2 = self.get_mapping(ex.maps.edge_block_edge_global_to_local)
        for (fid, file) in enumerate(self.files):
            block_iid = self.f_get_edge_block_iid(file, block_id)
            if not self.edge_block_is_active(file, block_iid):
                continue
            local_vals = self.get_variable(file, ex.VAR_EDGE_VAR(var_iid, block_iid))
            cols = np.zeros(local_vals.shape[1], dtype=int)
            for iid in range(len(cols)):
                gid = map1[(fid, block_id, iid + 1)]
                cols[iid] = map2[(block_id, gid)] - 1
            values[:, cols] = local_vals
        if time_step is None:
            return values
        elif time_step < 0:
            time_step = len(values)
        return values[time_step - 1]

    def get_edge_variable_values_across_blocks(self, var_name, time_step=None):
        values = []
        for id in self.get_edge_block_ids():
            x = self.get_edge_variable_values(id, var_name, time_step)
            if x is not None:
                values.extend(x)
            else:
                values.extend(np.zeros(self.num_elems_in_blk(id)))
        return np.array(values)

    def get_element_block_ids(self):
        """get mapping of exodus element block index to user - or application-defined
        element block id.

        block_ids is ordered by the element block INDEX ordering,
        a 1-based system going from 1 to exo.num_blks(), used by exodus for storage
        and input/output of array data stored on the element blocks a user or
        application can optionally use a separate element block ID numbering system,
        so the block_ids array points to the element block ID for each element
        block INDEX

        Returns
        -------
        block_ids : ndarray of int

        """
        return self.get_variable(self.files[0], ex.VAR_ID_ELEM_BLK)

    def f_get_element_block_iid(self, file, block_id):
        """get exodus element block index from id

        Returns
        -------
        block_id : int

        """
        block_ids = self.get_variable(file, ex.VAR_ID_ELEM_BLK)
        return self.get_iid(block_ids, block_id)

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
        for (fid, file) in enumerate(self.files):
            block_iid = self.f_get_element_block_iid(file, block_id)
            if not self.elem_block_is_active(file, block_iid):
                continue
            elem_type = self.get_element_block_elem_type(file, block_id)
            info = Namespace(
                id=block_id,
                fid=fid,
                iid=block_iid,
                elem_type=elem_type,
                num_block_elems=self.num_elems_in_blk(block_id),
                num_elem_nodes=self._num_nodes_per_elem(file, block_iid),
                num_elem_edges=self._num_edges_per_elem(file, block_iid),
                num_elem_faces=self._num_faces_per_elem(file, block_iid),
                num_elem_attrs=self.num_attr(block_id),
            )
            return info

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
        num_row = self.num_elems_in_blk(block_id)
        if type == ex.types.node:
            var = lambda iid: ex.VAR_ELEM_BLK_CONN(iid)
            num_col = self.num_nodes_per_elem(block_id)
            lid_to_gid = self.get_mapping(ex.maps.node_local_to_global, contiguous=True)
        elif type == ex.types.edge:
            var = lambda iid: ex.VAR_EDGE_CONN(iid)
            num_col = self.num_edges_per_elem(block_id)
            lid_to_gid = self.get_mapping(ex.maps.edge_local_to_global, contiguous=True)
        elif type == ex.types.face:
            var = lambda iid: ex.VAR_FACE_CONN(iid)
            num_col = self.num_faces_per_elem(block_id)
            lid_to_gid = self.get_mapping(ex.maps.face_local_to_global, contiguous=True)
        else:
            raise ValueError(f"Invalid element connectivity type {type!r}")

        if num_col is None:
            return None

        shape = (num_row, num_col)
        conn = np.zeros(shape, dtype=int)

        map1 = self.get_mapping(ex.maps.elem_block_elem_local_to_global)
        map2 = self.get_mapping(ex.maps.elem_block_elem_global_to_local)
        for (fid, file) in enumerate(self.files):
            block_iid = self.f_get_element_block_iid(file, block_id)
            if not self.elem_block_is_active(file, block_iid):
                continue
            file_local_conn = self.get_variable(file, var(block_iid))
            for (iid, lids) in enumerate(file_local_conn, start=1):
                # Map file local block element index to global block element index
                gid = map1[(fid, block_id, iid)]
                idx = map2[(block_id, gid)] - 1
                conn[idx] = [lid_to_gid[(fid, lid)] for lid in lids]
        return conn

    def f_get_element_id_map(self, file):
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
        map = self.get_variable(file, ex.VAR_ELEM_NUM_MAP)
        if map is not None:
            return map
        start = 0
        for f in self.files:
            num_elem_this_file = self.get_dimension(f, ex.DIM_NUM_ELEM)
            if os.path.samefile(self.get_filename(f), self.get_filename(file)):
                return np.arange(start, start + num_elem_this_file, dtype=int) + 1
            start += num_elem_this_file
        raise ValueError("Cannot determine element number map")

    def get_element_id_map(self, file=None):
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
        map = None if file is None else self.get_variable(file, ex.VAR_ELEM_NUM_MAP)
        if map is None:
            map = np.arange(self.num_elems(), dtype=int) + 1
        return map

    def get_element_block_elem_type(self, file, block_id):
        """Get the element type, e.g. "HEX8", for an element block"""
        block_iid = self.f_get_element_block_iid(file, block_id)
        if not self.elem_block_is_active(file, block_iid):
            return None
        var = self.get_variable(file, ex.VAR_ELEM_BLK_CONN(block_iid), raw=True)
        return None if var is None else var.elem_type

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
        values = self.get_variable(
            self.files[e.fid], ex.VAR_ELEM_VAR(var_iid, e.blk_iid)
        )
        if values is None:
            return None
        return values[:, e.iid - 1]

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
            return self.get_element_variable_values_across_blocks(
                var_name, time_step=time_step
            )

        names = self.get_element_variable_names()
        var_iid = self.get_iid(names, var_name)

        shape = (self.num_times(), self.num_elems_in_blk(block_id))
        values = np.zeros(shape, dtype=float)

        map1 = self.get_mapping(
            ex.maps.elem_block_elem_local_to_global, contiguous=True
        )
        map2 = self.get_mapping(ex.maps.elem_block_elem_global_to_local)
        for (fid, file) in enumerate(self.files):
            block_iid = self.f_get_element_block_iid(file, block_id)
            if not self.elem_block_is_active(file, block_iid):
                continue
            local_vals = self.get_variable(file, ex.VAR_ELEM_VAR(var_iid, block_iid))
            cols = np.zeros(local_vals.shape[1], dtype=int)
            for iid in range(len(cols)):
                gid = map1[(fid, block_id, iid + 1)]
                cols[iid] = map2[(block_id, gid)] - 1
            values[:, cols] = local_vals
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
                values.append(x)
        if time_step is None:
            return np.column_stack(values)
        return np.array([_ for x in values for _ in x])

    def get_face_block_elem_type(self, file, block_id):
        """Get the element type, e.g. "HEX8", for a face block"""
        block_iid = self.f_get_face_block_iid(file, block_id)
        if not self.face_block_is_active(file, block_iid):
            return None
        var = self.get_variable(file, ex.VAR_FACE_BLK_CONN(block_iid), raw=True)
        return None if var is None else var.elem_type

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
        block_ids = []
        for file in self.files:
            file_local_ids = self.get_variable(file, ex.VAR_ID_FACE_BLK)
            if file_local_ids:
                block_ids.extend(file_local_ids)
        return sorted(set(block_ids))

    def f_get_face_block_iid(self, file, block_id):
        """get exodus face block index from id

        Returns
        -------
        block_iid : int

        """
        block_ids = self.get_variable(file, ex.VAR_ID_FACE_BLK)
        return self.get_iid(block_ids, block_id)

    def get_face_block(self, block_id):
        """Get the face block info

        Parameters
        ----------
        block_id : int
            face block ID (not INDEX)

        Returns
        -------
        elem_type : str
            element type, e.g. 'HEX8'
        num_block_faces : int
            number of faces in the block
        num_face_nodes : int
            number of nodes per face
        num_face_attrs : int
            number of attributes per face
        """
        if block_id not in self.get_face_block_ids():
            raise None
        for (fid, file) in enumerate(self.files):
            block_iid = self.f_get_face_block_iid(file, block_id)
            if not self.face_block_is_active(file, block_iid):
                continue
            elem_type = self.get_face_block_elem_type(file, block_id)
            info = Namespace(
                id=block_id,
                fid=fid,
                iid=block_iid,
                elem_type=elem_type,
                num_block_faces=self.num_faces_in_blk(block_id),
                num_face_nodes=self.num_nodes_per_face(block_id),
                num_face_attrs=self.num_attr(block_id),
            )
            return info

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
        shape = (
            self.num_faces_in_blk(block_id),
            self.num_nodes_per_face(block_id),
        )
        conn = np.zeros(shape, dtype=int)
        lid_to_gid = self.get_mapping(ex.maps.node_local_to_global, contiguous=True)
        map1 = self.get_mapping(ex.maps.face_block_face_local_to_global)
        map2 = self.get_mapping(ex.maps.face_block_face_global_to_local)
        for (fid, file) in enumerate(self.files):
            block_iid = self.f_get_face_block_iid(file, block_id)
            if not self.face_block_is_active(file, block_iid):
                continue
            file_local_conn = self.get_variable(file, ex.VAR_FACE_BLK_CONN(block_iid))
            for (lid, nodes) in enumerate(file_local_conn, start=1):
                gid = map1[(fid, block_id, lid)]
                idx = map2[(block_id, gid)] - 1
                conn[idx, :] = [lid_to_gid[(fid, nid)] for nid in nodes]
        return conn

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
        evar_vals : ndarray of float
        """
        if block_id is None:
            return self.get_face_variable_values_across_blocks(
                var_name, time_step=time_step
            )

        names = self.get_face_variable_names()
        var_iid = self.get_iid(names, var_name)

        shape = (self.num_times(), self.num_faces_in_blk(block_id))
        values = np.zeros(shape, dtype=float)

        map1 = self.get_mapping(ex.maps.face_block_face_local_to_global)
        map2 = self.get_mapping(ex.maps.face_block_face_global_to_local)
        for (fid, file) in enumerate(self.files):
            block_iid = self.f_get_face_block_iid(file, block_id)
            if not self.face_block_is_active(file, block_iid):
                continue
            local_vals = self.get_variable(file, ex.VAR_FACE_VAR(var_iid, block_iid))
            cols = np.zeros(local_vals.shape[1], dtype=int)
            for iid in range(len(cols)):
                gid = map1[(fid, block_id, iid + 1)]
                cols[iid] = map2[(block_id, gid)] - 1
            values[:, cols] = local_vals
        if time_step is None:
            return values
        elif time_step < 0:
            time_step = len(values)
        return values[time_step - 1]

    def get_face_variable_values_across_blocks(self, var_name, time_step=None):
        values = []
        for id in self.get_face_block_ids():
            x = self.get_face_variable_values(id, var_name, time_step)
            if x is not None:
                values.extend(x)
            else:
                values.extend(np.zeros(self.num_elems_in_blk(id)))
        return np.array(values)

    def f_get_node_id_map(self, file):
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
        return self.get_variable(file, ex.VAR_NODE_NUM_MAP)

    def get_node_id_map(self, file=None):
        # FIXME
        map = None if file is None else self.get_variable(ex.VAR_NODE_NUM_MAP)
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
        cache = self.cache.setdefault("node_ns_df", {})
        if set_id in cache:
            return cache[set_id]

        node_set_nodes = self.get_node_set_nodes(set_id)
        if node_set_nodes is None:
            return None

        # Making a mapping from gid to node set local lid
        mapping = dict([(gid, lid) for (lid, gid) in enumerate(node_set_nodes)])

        node_map = self.get_mapping(ex.maps.node_local_to_global)
        ns_dist_facts = None
        for (fid, file) in enumerate(self.files):
            set_iid = self.f_get_node_set_iid(file, set_id)
            if not self.node_set_is_active(file, set_iid):
                continue
            file_local_ns_df = self.get_variable(file, ex.VAR_DF_NODE_SET(set_iid))
            if file_local_ns_df is None:
                continue
            if ns_dist_facts is None:
                ns_dist_facts = np.zeros(len(mapping))
            file_local_ns_nodes = self.get_variable(file, ex.VAR_NODE_NODE_SET(set_iid))
            for (i, dist_fact) in enumerate(file_local_ns_df):
                gid = node_map[(fid, file_local_ns_nodes[i])]
                ns_dist_facts[mapping[gid]] = dist_fact
        cache[set_id] = ns_dist_facts
        if ns_dist_facts is None:
            return np.ones(len(mapping))
        return ns_dist_facts

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
        return self.get_variable(self.files[0], ex.VAR_NODE_SET_IDS_GLOBAL)

    def f_get_node_set_iid(self, file, set_id):
        set_ids = self.get_variable(file, ex.VAR_NODE_SET_IDS)
        return self.get_iid(set_ids, set_id)

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
        cache = self.cache.setdefault("node_ns", {})
        if set_id in cache:
            return cache[set_id]

        set_ids = self.get_node_set_ids()
        if set_id not in set_ids:
            return None

        num_nodes_in_ns = self.num_nodes_in_node_set(set_id)
        if not num_nodes_in_ns:
            return None

        ns_nodes = []
        node_map = self.get_mapping(ex.maps.node_local_to_global)
        for (fid, file) in enumerate(self.files):
            set_iid = self.f_get_node_set_iid(file, set_id)
            if not self.node_set_is_active(file, set_iid):
                continue
            lids = self.get_variable(file, ex.VAR_NODE_NODE_SET(set_iid))
            ns_nodes.extend([node_map[(fid, lid)] for lid in lids])

        ns_nodes = np.array(sorted(set(ns_nodes)), dtype=int)
        cache[set_id] = ns_nodes

        return ns_nodes

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
        return values[:, node_id - 1]

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
        if var_iid is None:
            return None
        values = np.zeros((self.num_times(), self.num_nodes()), dtype=float)
        node_map = self.get_mapping(ex.maps.node_local_to_global)
        for (fid, file) in enumerate(self.files):
            num_nodes = self.get_dimension(file, ex.DIM_NUM_NODES)
            cols = [node_map[(fid, lid)] - 1 for lid in range(1, num_nodes + 1)]
            values[:, cols] = self.get_variable(file, ex.VAR_NODE_VAR(var_iid))
        if time_step is None:
            return values
        elif time_step < 0:
            time_step = len(values)
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
        return self._qa_records

    def get_side_set_data(self, set_id):
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
        cache = self.cache.setdefault("ss", {})
        if set_id in cache:
            return cache[set_id]

        set_ids = self.get_side_set_ids()
        if set_id not in set_ids:
            return None

        num_sides = self.num_sides_in_side_set(set_id)
        elems, sides = np.zeros(num_sides, dtype=int), np.zeros(num_sides, dtype=int)

        side = 0
        lid_to_gid = self.get_mapping(ex.maps.elem_local_to_global)
        for (fid, file) in enumerate(self.files):
            set_iid = self.get_side_set_iid(file, set_id)
            if not self.side_set_is_active(file, set_iid):
                continue
            file_local_ss_elems = self.get_variable(file, ex.VAR_ELEM_SIDE_SET(set_iid))
            file_local_ss_sides = self.get_variable(file, ex.VAR_SIDE_SIDE_SET(set_iid))
            for (i, local_elem) in enumerate(file_local_ss_elems):
                elems[side] = lid_to_gid[(fid, local_elem)]
                sides[side] = file_local_ss_sides[i]
                side += 1

        return elems, sides

    def get_side_set_elems(self, set_id):
        return self.get_side_set_data(set_id)[0]

    def get_side_set_sides(self, set_id):
        return self.get_side_set_data(set_id)[1]

    def get_side_set_dist_facts(self, set_id):
        """Get the list of distribution factors for nodes in a side set

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
        The number of nodes (and distribution factors) in a side set is the sum of
        all face nodes. A single node can be counted more than once, i.e. once for
        each face it belongs to in the side set.
        """
        cache = self.cache.setdefault("ss_df", {})
        if set_id in cache:
            return cache[set_id]

        set_ids = self.get_side_set_ids()
        if set_id not in set_ids:
            return None

        dist_facts = []
        for (fid, file) in enumerate(self.files):
            set_iid = self.get_side_set_iid(file, set_id)
            if not self.side_set_is_active(file, set_iid):
                continue
            file_local_ss_dist_facts = self.get_variable(
                file, ex.VAR_DF_SIDE_SET(set_iid)
            )
            if file_local_ss_dist_facts is not None:
                dist_facts.extend(file_local_ss_dist_facts)
        dist_facts = np.array(dist_facts)
        cache[set_id] = dist_facts

        return dist_facts

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
        return self.get_variable(self.files[0], ex.VAR_SIDE_SET_IDS_GLOBAL)

    def get_side_set_iid(self, file, set_id):
        set_ids = self.get_variable(file, ex.VAR_SIDE_SET_IDS)
        return self.get_iid(set_ids, set_id)

    def edge_block_is_active(self, file, block_iid):
        status = self.get_variable(file, ex.VAR_STAT_EDGE_BLK)
        return bool(status[block_iid - 1])

    def elem_block_is_active(self, file, block_iid):
        status = self.get_variable(file, ex.VAR_STAT_ELEM_BLK)
        return bool(status[block_iid - 1])

    def face_block_is_active(self, file, block_iid):
        status = self.get_variable(file, ex.VAR_STAT_FACE_BLK)
        return bool(status[block_iid - 1])

    def node_set_is_active(self, file, set_iid):
        status = self.get_variable(file, ex.VAR_NODE_SET_STAT)
        return bool(status[set_iid - 1])

    def side_set_is_active(self, file, set_iid):
        status = self.get_variable(file, ex.VAR_SIDE_SET_STAT)
        return bool(status[set_iid - 1])

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
        block_iid = self.get_iid(self.get_element_block_ids(), block_id)
        return self._num_attr(self.files[0], block_iid)

    def _num_attr(self, file, block_iid):
        return self.get_dimension(file, ex.DIM_NUM_ATTR(block_iid), default=0)

    def num_blks(self):
        """Get the number of element blocks in the model

        Returns
        -------
        num_elem_blks : int
        """
        return self.get_dimension(self.files[0], ex.DIM_NUM_ELEM_BLK_GLOBAL, default=0)

    def num_edge_blk(self):
        return len(self.get_edge_block_ids())

    def num_edges(self):
        """Number of model edges"""
        return len(self.get_mapping(ex.maps.edge_local_to_global, invert=True))

    def num_faces(self):
        """Number of model faces"""
        return len(self.get_mapping(ex.maps.face_local_to_global, invert=True))

    def f_get_edge_id_map(self, file):
        """Get mapping of exodus edge index to user- or application- defined
        edge id; edge_id_map is ordered by the edge *INDEX* ordering, a 1-based
        system going from 1 to exo.num_edges(), used by exodus for storage and
        input/output of array data stored on the edges; a user or application can
        optionally use a separate edge *ID* numbering system, so the edge_id_map
        points to the edge *ID* for each edge *INDEX*

        Returns
        -------
        edge_id_map : ndarray of int

        """
        return self.get_variable(file, ex.VAR_EDGE_NUM_MAP)

    def get_edge_id_map(self):
        # FIXME
        return np.arange(self.num_edges(), dtype=int) + 1

    def get_edges_in_blk(self, block_id):
        """Get the edges in a edge block

        Parameters
        ----------
        block_id : int
            edge block ID (not INDEX)

        Returns
        -------
        edges : ndarray of int

        """
        edges = []
        for file in self.files:
            start = 0
            edge_map = self.get_variable(file, ex.VAR_EDGE_NUM_MAP)
            block_iid = self.f_get_edge_block_iid(file, block_id)
            if not self.edge_block_is_active(file, block_iid):
                continue
            conn = self.get_variable(file, ex.VAR_EDGE_BLK_CONN(block_iid))
            end = start + len(conn)
            edges.extend(edge_map[start:end])
            start = end
        return np.array(sorted(set(edges)), dtype=int)

    def f_get_face_id_map(self, file):
        """Get mapping of exodus face index to user- or application- defined
        face id; face_id_map is ordered by the face *INDEX* ordering, a 1-based
        system going from 1 to exo.num_faces(), used by exodus for storage and
        input/output of array data stored on the faces; a user or application can
        optionally use a separate face *ID* numbering system, so the face_id_map
        points to the face *ID* for each face *INDEX*

        Returns
        -------
        face_id_map : ndarray of int

        """
        return self.get_variable(file, ex.VAR_FACE_NUM_MAP)

    def get_face_id_map(self):
        # FIXME
        return np.arange(self.num_faces(), dtype=int) + 1

    def get_faces_in_blk(self, block_id):
        """Get the faces in a face block

        Parameters
        ----------
        block_id : int
            face block ID (not INDEX)

        Returns
        -------
        faces : ndarray of int

        """
        faces = []
        for file in self.files:
            start = 0
            block_iid = self.f_get_face_block_iid(file, block_id)
            if not self.face_block_is_active(file, block_iid):
                continue
            face_map = self.get_variable(file, ex.VAR_FACE_NUM_MAP)
            num_faces = self.num__faces_in_blk(file, block_id)
            faces.extend(face_map[start : start + num_faces])
            start += num_faces
        return np.array(sorted(set(faces)), dtype=int)

    def num_edges_in_blk(self, block_id):
        """Get the number of edges in a edge block

        Parameters
        ----------
        block_id : int
            edge block ID (not INDEX)

        Returns
        -------
        num_block_edges : int

        """
        return len(self.get_edges_in_blk(block_id))

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
        for (fid, file) in enumerate(self.files):
            block_iid = self.f_get_element_block_iid(file, block_id)
            if not self.elem_block_is_active(file, block_iid):
                continue
            return self._num_edges_per_elem(file, block_iid)

    def _num_edges_per_elem(self, file, block_iid):
        return self.get_dimension(file, ex.DIM_NUM_EDGE_PER_ELEM(block_iid))

    def num_elems(self):
        """Get the number of elements in the model

        Returns
        -------
        num_elems : int
        """
        return self.get_dimension(self.files[0], ex.DIM_NUM_ELEM_GLOBAL, default=0)

    def num_elems_in_all_blks(self):
        """Get the number of elements in an element block

        Parameters
        ----------
        block_id : int
            element block ID (not INDEX)

        Returns
        -------
        num_block_elems : int

        """
        return self.get_variable(self.files[0], ex.VAR_ELEM_BLK_COUNT_GLOBAL)

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
        block_count = self.get_variable(self.files[0], ex.VAR_ELEM_BLK_COUNT_GLOBAL)
        for (i, bid) in enumerate(self.get_element_block_ids()):
            if block_id == bid:
                return block_count[i]

    def _num_elems_in_blk(self, file, block_iid):
        return self.get_dimension(file, ex.DIM_NUM_ELEM_IN_ELEM_BLK(block_iid))

    def num_face_blk(self):
        return len(self.get_face_block_ids())

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
        return len(self.get_faces_in_blk(block_id))

    def num__faces_in_blk(self, file, block_id):
        """Get the number of faces in a face block

        Parameters
        ----------
        block_id : int
            Face block ID (not INDEX)

        Returns
        -------
        num_block_faces : int

        """
        block_iid = self.f_get_face_block_iid(file, block_id)
        return self.get_dimension(file, ex.DIM_NUM_FACE_IN_FACE_BLK(block_iid))

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
        for (fid, file) in enumerate(self.files):
            block_iid = self.f_get_element_block_iid(file, block_id)
            if not self.elem_block_is_active(file, block_iid):
                continue
            return self._num_faces_per_elem(file, block_iid)

    def _num_faces_per_elem(self, file, block_iid):
        return self.get_dimension(file, ex.DIM_NUM_FACE_PER_ELEM(block_iid))

    def num_nodes_per_edge(self, block_id):
        """Get the number of nodes per edge for an edge block

        Parameters
        ----------
        block_id: int
            edge block ID(not INDEX)

        Returns
        -------
        num_edge_nodes : int

        """
        for file in self.files:
            block_iid = self.f_get_edge_block_iid(file, block_id)
            if not self.edge_block_is_active(file, block_iid):
                continue
            return self.get_dimension(file, ex.DIM_NUM_NODE_PER_EDGE(block_iid))

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
        for file in self.files:
            block_iid = self.f_get_face_block_iid(file, block_id)
            if not self.face_block_is_active(file, block_iid):
                continue
            return self.get_dimension(file, ex.DIM_NUM_NODE_PER_FACE(block_iid))

    def num_node_sets(self):
        """Get the number of node sets in the model

        Returns
        -------
        num_node_sets : int

        """
        return self.get_dimension(self.files[0], ex.DIM_NUM_NODE_SET_GLOBAL, default=0)

    def num_nodes(self):
        """Get the number of nodes in the model

        Returns
        -------
        num_nodes : int
        """
        num_nodes = self.get_dimension(self.files[0], ex.DIM_NUM_NODE_GLOBAL, default=0)
        if config.debug:
            nodes = []
            for file in self.files:
                file_local_map = self.get_variable(file, ex.VAR_NODE_NUM_MAP)
                nodes.extend(file_local_map)
            nodes = sorted(set(nodes))
            if len(nodes) != num_nodes:
                raise ValueError(
                    f"The number of nodes across files ({len(nodes)}) is not the "
                    f"same as the number reported by the num_node_global dimension "
                    f"({num_nodes})"
                )
        return num_nodes

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
        ns_node_count = self.get_variable(
            self.files[0], ex.VAR_NODE_SET_NODE_COUNT_GLOBAL
        )
        for (i, ns_id) in enumerate(self.get_node_set_ids()):
            if set_id == ns_id:
                return ns_node_count[i]

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
        for (fid, file) in enumerate(self.files):
            block_iid = self.f_get_element_block_iid(file, block_id)
            if not self.elem_block_is_active(file, block_iid):
                continue
            return self._num_nodes_per_elem(file, block_iid)

    def _num_nodes_per_elem(self, file, block_iid):
        return self.get_dimension(file, ex.DIM_NUM_NODE_PER_ELEM(block_iid))

    def num_side_sets(self):
        """Get the number of side sets in the model

        Returns
        -------
        num_side_sets : int

        """
        return self.get_dimension(self.files[0], ex.DIM_NUM_SIDE_SET_GLOBAL, default=0)

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
        ss_side_count = self.get_variable(
            self.files[0], ex.VAR_SIDE_SET_SIDE_COUNT_GLOBAL
        )
        for (i, ss_id) in enumerate(self.get_side_set_ids()):
            if set_id == ss_id:
                return ss_side_count[i]
        return None

    def get_mapping(self, name, invert=False, contiguous=False):
        """Return the mapping for `name`

        Parameters
        ----------
        name : enum
            The mapping type

        Note
        ----
        This is an implementation detail

        """
        assert isinstance(name, ex.maps)
        mappings = {
            ex.maps.elem_block_elem_local_to_global: self.elem_blk_elem_l2g,
            ex.maps.elem_block_elem_global_to_local: self.elem_blk_elem_g2l,
            ex.maps.edge_block_edge_local_to_global: self.edge_blk_edge_l2g,
            ex.maps.edge_block_edge_global_to_local: self.edge_blk_edge_g2l,
            ex.maps.face_block_face_local_to_global: self.face_blk_face_l2g,
            ex.maps.face_block_face_global_to_local: self.face_blk_face_g2l,
            ex.maps.elem_local_to_global: self.elem_l2g,
            ex.maps.node_local_to_global: self.node_l2g,
            ex.maps.edge_local_to_global: self.edge_l2g,
            ex.maps.face_local_to_global: self.face_l2g,
        }
        fun = mappings.get(name)
        if fun is None:
            raise ValueError(f"Invalid map name {name}")
        if name not in self.cache:
            self.cache[name] = fun()
        mapping = self.cache[name]
        if invert:
            mapping = dict(zip(list(mapping.values()), list(mapping.keys())))
        if contiguous:
            contiguous_map = self.make_contiguous(mapping)
            return dict([(key, contiguous_map[gid]) for (key, gid) in mapping.items()])
        return mapping

    def elem_blk_elem_l2g(self):
        """Return mapping from (file, element block, iid) to global element ID"""
        mapping = {}
        block_ids = self.get_element_block_ids()
        for (fid, file) in enumerate(self.files):
            start = 0
            elem_num_map = self.f_get_element_id_map(file)
            for block_id in block_ids:
                block_iid = self.f_get_element_block_iid(file, block_id)
                if not self.elem_block_is_active(file, block_iid):
                    continue
                file_local_conn = self.get_variable(
                    file, ex.VAR_ELEM_BLK_CONN(block_iid)
                )
                end = start + len(file_local_conn)
                assert end <= len(elem_num_map), f"{end}, {len(elem_num_map)}"
                for (iid, id) in enumerate(elem_num_map[start:end], start=1):
                    mapping[(fid, block_id, iid)] = id
                start = end
        return mapping

    def elem_blk_elem_g2l(self):
        num_elems = self.num_elems()
        counts = self.num_elems_in_all_blks()
        assert sum(counts) == num_elems
        el = 1
        mapping = {}
        for (i, block_id) in enumerate(self.get_element_block_ids()):
            for j in range(counts[i]):
                mapping[(block_id, el)] = j + 1
                el += 1
        return mapping

    def edge_blk_edge_l2g(self):
        """Return mapping from (file, edge block, iid) to global edge ID"""
        mapping = {}
        for (fid, file) in enumerate(self.files):
            start = 0
            edge_num_map = self.f_get_edge_id_map(file)
            for block_id in self.get_edge_block_ids():
                block_iid = self.f_get_edge_block_iid(file, block_id)
                if not self.edge_block_is_active(file, block_iid):
                    continue
                file_local_conn = self.get_variable(
                    file, ex.VAR_EDGE_BLK_CONN(block_iid)
                )
                end = start + len(file_local_conn)
                for (iid, id) in enumerate(edge_num_map[start:end], start=1):
                    mapping[(fid, block_id, iid)] = id
                start = end
        return mapping

    def edge_blk_edge_g2l(self):
        num_edges = self.num_edges()
        counts = [self.num_edges_in_blk(_) for _ in self.get_edge_block_ids()]
        assert sum(counts) == num_edges
        ed = 1
        mapping = {}
        for (i, block_id) in enumerate(self.get_edge_block_ids()):
            for j in range(counts[i]):
                mapping[(block_id, ed)] = j + 1
                ed += 1
        return mapping

    def face_blk_face_l2g(self):
        """Return mapping from (file, face block, iid) to global face ID"""
        mapping = {}
        for (fid, file) in enumerate(self.files):
            start = 0
            face_num_map = self.f_get_face_id_map(file)
            for block_id in self.get_face_block_ids():
                block_iid = self.f_get_face_block_iid(file, block_id)
                if not self.face_block_is_active(file, block_iid):
                    continue
                file_local_conn = self.get_variable(
                    file, ex.VAR_FACE_BLK_CONN(block_iid)
                )
                end = start + len(file_local_conn)
                for (iid, id) in enumerate(face_num_map[start:end], start=1):
                    mapping[(fid, block_id, iid)] = id
                start = end
        return mapping

    def face_blk_face_g2l(self):
        num_faces = self.num_faces()
        counts = [self.num_faces_in_blk(_) for _ in self.get_face_block_ids()]
        assert sum(counts) == num_faces
        fb = 1
        mapping = {}
        for (i, block_id) in enumerate(self.get_face_block_ids()):
            for j in range(counts[i]):
                mapping[(block_id, fb)] = j + 1
                fb += 1
        return mapping

    def edge_l2g(self):
        """Return mapping from file local edge ID to global edge ID"""
        mapping = {}
        for (fid, file) in enumerate(self.files):
            file_local_map = self.get_variable(file, ex.VAR_EDGE_NUM_MAP)
            if file_local_map is None:
                continue
            for (lid, gid) in enumerate(file_local_map, start=1):
                mapping[(fid, lid)] = gid - 1
        return mapping

    def elem_l2g(self):
        """Return mapping from file local element ID to global element ID"""
        mapping = {}
        for (fid, file) in enumerate(self.files):
            file_local_map = self.get_variable(file, ex.VAR_ELEM_NUM_MAP)
            for (lid, gid) in enumerate(file_local_map, start=1):
                mapping[(fid, lid)] = gid
        return mapping

    def face_l2g(self):
        """Return mapping from file local face ID to global face ID"""
        mapping = {}
        for (fid, file) in enumerate(self.files):
            file_local_map = self.get_variable(file, ex.VAR_FACE_NUM_MAP)
            if file_local_map is None:
                continue
            for (lid, gid) in enumerate(file_local_map, start=1):
                mapping[(fid, lid)] = gid - 1
        return mapping

    def node_l2g(self):
        """Return mapping from file local node ID to global node ID"""
        mapping = {}
        for (fid, file) in enumerate(self.files):
            file_local_map = self.f_get_node_id_map(file)
            for (lid, gid) in enumerate(file_local_map, start=1):
                mapping[(fid, lid)] = gid
        return mapping

    def check_consistency(self, files):

        times = self.get_variable(files[0], ex.VAR_WHOLE_TIME)
        word_size = files[0].floating_point_word_size
        ex_init = ex_init_params(files[0])

        for fh in files[1:]:

            if fh.floating_point_word_size != word_size:
                raise TypeError("floating point storage not consistent across files")

            if len(times) != len(self.get_variable(fh, ex.VAR_WHOLE_TIME)):
                raise TypeError("number of time steps not consistent across files")

            if ex_init != ex_init_params(fh):
                raise TypeError("initialization parameters not consistent across files")

        nblks = self.get_dimension(files[0], ex.DIM_NUM_ELEM_BLK_GLOBAL, default=1)
        for file in files:
            counted = 0
            elem_num_map = self.f_get_element_id_map(file)
            for block_iid in range(1, nblks + 1):
                if not self.elem_block_is_active(file, block_iid):
                    continue
                conn = self.get_variable(file, ex.VAR_ELEM_BLK_CONN(block_iid))
                counted += len(conn)
            if counted != len(elem_num_map):
                raise ValueError(
                    f"Expected {len(elem_num_map)} elements across all blocks in "
                    f"{self.get_filename(file)}, counted {counted}"
                )

        return

    def _get_element_info(self, elem_id):
        """Get the element block ID and element index within the block for elem_id

        Note
        ----
        This is an implementation detail.

        """
        for (fid, file) in enumerate(self.files):
            start = 0
            elem_num_map = self.f_get_element_id_map(file)
            block_ids = self.get_element_block_ids()
            for block_iid in range(1, self.num_element_blocks() + 1):
                if not self.elem_block_is_active(file, block_iid):
                    continue
                file_local_conn = self.get_variable(
                    file, ex.VAR_ELEM_BLK_CONN(block_iid)
                )
                end = start + len(file_local_conn)
                if elem_id in elem_num_map[start:end]:
                    elem_iid = self.get_iid(elem_num_map[start:end], elem_id)
                    return Namespace(
                        fid=fid,
                        id=elem_id,
                        iid=elem_iid,
                        blk_id=block_ids[block_iid - 1],
                        blk_iid=block_iid,
                    )
                start = end
        raise ValueError(f"Unable to determine element block for element {elem_id}")

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


# Backward compat
MFExodusIIFile = parallel_exodusii_file
