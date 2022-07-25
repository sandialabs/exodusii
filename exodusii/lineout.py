import numpy as np
from functools import cmp_to_key

from .util import string_types


class lineout:
    """Restrict the nodes/elements/edges/faces to those whose coordinates are along a
    line parallel to an axis.

    """

    def __init__(self, *, x=None, y=None, z=None, tol=None):
        self.x = x
        self.y = y
        self.z = z
        self.tol = tol
        self.needs_displacements = self.x == "X" or self.y == "Y" or self.z == "Z"

    @property
    def spec(self):
        return [self.x, self.y, self.z]

    @classmethod
    def from_cli(cls, arg):
        """Instantiate the lineout object using the command line interface. Use lower
        case x/y/z for original or material coordinates and upper case for displaced
        or Lagrangian coordinates. In 2D, both an X and a Y entry are required and
        for 3D, a Z entry. So "x/1.0" in 2D would be a lineout along x with y=1.0,
        while "2.0/y" would be a lineout in y with x=2.0. The last number with a "T"
        in front is an optional tolerance used in selecting the coordinate locations.
        A 3D example is "1.0/Y/3.0/T0.1" which is a lineout in y using displaced
        coordinates with x in the range (0.9,1.1) and z in (2.9,3.1).

        """
        parts = [_.strip() for _ in arg.split("/") if _.split()]

        tol = None
        if parts[-1].startswith(("t", "T")):
            try:
                tol = float(parts[-1][1:])
            except (ValueError, TypeError, SyntaxError):
                raise ValueError("lineout: time parameter must be a float") from None
            parts = parts[:-1]

        if len(parts) > 3:
            raise ValueError("lineout: expected at most 3 spatial specifiers")

        x = cls.read_spatial_spec(parts[0], "x")
        y = cls.read_spatial_spec(None if len(parts) < 2 else parts[1], "y")
        z = cls.read_spatial_spec(None if len(parts) < 3 else parts[2], "z")

        return cls(x=x, y=y, z=z, tol=tol)

    @staticmethod
    def read_spatial_spec(spec, coord):
        if spec is None:
            return None
        try:
            return float(spec)
        except (ValueError, TypeError, SyntaxError):
            pass
        if spec.lower() != coord:
            raise ValueError(
                f"lineout: expected specifier "
                f"{spec!r} to be {coord!r}, {coord.upper()!r}, or a float"
            )
        return spec

    def apply(self, *args):
        """Removes points that are not along a line.

        If self.needs_displacements, the displacements are added to the coordinates
        first. In this case, the columns are expected to be:

                0     1      2      3      4      5      6
        1D   index DISPLX COORDX
        2D   index DISPLX DISPLY COORDX COORDY
        3D   index DISPLX DISPLY DISPLZ COORDX COORDY COORDZ

        Otherwise, the columns are expected to be:

                0     1      2      3
        1D   index COORDX
        2D   index COORDX COORDY
        3D   index COORDX COORDY COORDZ

        """
        isstructured = False
        if len(args) == 1:
            # Structured array
            isstructured = True
            header = list(args[0].dtype.names)
            data = np.array(args[0].tolist())
        elif len(args) == 2:
            header, data = args
        else:
            raise TypeError(
                f"apply() takes 1 or 2 positional arguments but {len(args)} were given"
            )
        if isinstance(data, np.ndarray) and data.dtype.names is not None:
            # Don't work with structured arrays
            data = np.asarray(data.tolist())
        start = 1 if header[0].lower() == "index" else 0
        dim = 3 if "COORDZ" in header else 2 if "COORDY" in header else 1
        pairs = [(start + i, start + dim + i) for i in range(dim)]
        dim = len(pairs)

        if self.needs_displacements:
            assert "DISPLX" in header
            # add the displacements to the coordinates, then remove the displacements
            for a, b in pairs:
                data[:, b] += data[:, a]
            cols = [a for a, _ in pairs]
            header = np.delete(header, cols, axis=0).tolist()
            data = np.delete(data, cols, axis=1)
            header[start : start + len(pairs)] = [
                f"LOCATION{'XYZ'[i]}" for i in range(len(pairs))
            ]

        if self.tol is None and len(data):
            self.tol = self.compute_tol_from_bounding_box(data[:, start : start + dim])

        sort_order = []
        indices_to_remove = []
        for i in range(dim):
            idx = start + i
            if isinstance(self.spec[i], float):
                indices_to_remove.insert(0, idx)  # want largest index first
                # compute new list excluding points that are not within tolerance
                filtered = []
                for row in data:
                    if abs(self.spec[i] - row[idx]) < self.tol:
                        filtered.append(row)
                data = np.array(filtered)
            else:
                sort_order.append(idx)

        if sort_order:
            # sort by the free column(s)
            data = sorted(data, key=cmp_to_key(self.linecmp(sort_order)))

        if indices_to_remove:
            # remove the LOCATION columns of restricted coordinates
            header = np.delete(header, indices_to_remove, axis=0).tolist()
            if len(data):
                data = np.delete(data, indices_to_remove, axis=1)
            else:
                data = np.empty((0, len(header)))

        data = np.asarray(data)

        if isstructured:
            formats = [(name, "f8") for name in header]
            dtype = np.dtype(formats)
            return np.array(list(zip(*data.T)), dtype=dtype)

        return header, data

    def compute_tol_from_bounding_box(self, data):
        # tolerance not given; first compute mesh bounding box
        dim = data.shape[1]
        bbox = [None] * 6
        bbox[0:2] = [np.min(data[:, 0]), np.max(data[:, 0])]
        if dim >= 2:
            bbox[2:4] = [np.min(data[:, 1]), np.max(data[:, 1])]
        if dim > 2:
            bbox[4:6] = [np.min(data[:, 2]), np.max(data[:, 2])]

        # from the direction(s) being restricted, compute tolerance
        def tol_from_bbox(x, bx, by):
            fac, puny = 1.0e-4, 1.0e-30
            return None if not isinstance(x, float) else fac * max(by - bx, puny)

        xtol = tol_from_bbox(self.x, bbox[0], bbox[1])
        ytol = None if dim < 2 else tol_from_bbox(self.y, bbox[2], bbox[3])
        ztol = None if dim < 3 else tol_from_bbox(self.z, bbox[4], bbox[5])
        return min([xtol or 1, ytol or 1, ztol or 1])

    def linecmp(self, sortidx):
        if len(sortidx) == 1:
            i1 = sortidx[0]

            def linecmp(a, b):
                if isinstance(a[0], string_types):
                    return -1
                if isinstance(b[0], string_types):
                    return 1
                return cmp(a[i1], b[i1])

        elif len(sortidx) == 2:
            i1, i2 = sortidx[0:2]

            def linecmp(a, b):
                if isinstance(a[0], string_types):
                    return -1
                if isinstance(b[0], string_types):
                    return 1
                c = cmp(a[i1], b[i1])
                if c == 0:
                    return cmp(a[i2], b[i2])
                return c

        else:
            i1, i2, i3 = sortidx[0:3]

            def linecmp(a, b):
                if isinstance(a[0], string_types):
                    return -1
                if isinstance(b[0], string_types):
                    return 1
                c = cmp(a[i1], b[i1])
                if c == 0:
                    c = cmp(a[i2], b[i2])
                    if c == 0:
                        return cmp(a[i3], b[i3])
                return c

        return linecmp


def cmp(x, y):
    """
    Replacement for built-in function cmp that was removed in Python 3

    Compare the two objects x and y and return an integer according to
    the outcome. The return value is negative if x < y, zero if x == y
    and strictly positive if x > y.
    """
    return bool(x > y) - bool(x < y)
