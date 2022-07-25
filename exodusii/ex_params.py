from . import nc


class ex_init_params:
    def __init__(self, fh):
        self.fh = fh

    def __eq__(self, other):
        if not isinstance(other, ex_init_params):
            return False
        for attr in dir(self):
            if not attr.startswith("num_"):
                continue
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def getdim(self, name, default=None):
        return nc.get_dimension(self.fh, name, default=default)

    @property
    def num_dim(self):
        return self.getdim("num_dim", 0)

    @property
    def num_edge_blk(self):
        """Number of model edge blocks"""
        return self.getdim("num_edge_blk", 0)

    @property
    def num_edge_maps(self):
        """Number of model edge maps"""
        return self.getdim("num_edge_maps", 0)

    @property
    def num_edge_sets(self):
        """Number of model edge sets"""
        return self.getdim("num_edge_sets", 0)

    @property
    def num_el_blk(self):
        """Number of model element blocks"""
        return self.getdim("num_el_blk", 0)

    @property
    def num_elem_maps(self):
        """Number of model elem maps"""
        return self.getdim("num_elem_maps", 0)

    @property
    def num_elem_sets(self):
        """Number of model elem sets"""
        return self.getdim("num_elem_sets", 0)

    @property
    def num_face_blk(self):
        """Number of model face blocks"""
        return self.getdim("num_face_blk", 0)

    @property
    def num_face_maps(self):
        """Number of model face maps"""
        return self.getdim("num_face_maps", 0)

    @property
    def num_face_sets(self):
        """Number of model face sets"""
        return self.getdim("num_face_sets", 0)

    @property
    def num_glo_var(self):
        return self.getdim("num_glo_var", 0)

    @property
    def num_node_maps(self):
        """Number of model node maps"""
        return self.getdim("num_node_maps", 0)

    @property
    def num_node_sets(self):
        """Number of model node sets"""
        return self.getdim("num_node_sets", 0)

    @property
    def num_side_sets(self):
        """Number of model side sets"""
        return self.getdim("num_side_sets", 0)
