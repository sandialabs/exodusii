import numpy as np


def factory(elem_type, elem_coord):
    """Determine the element type and return a dictionary containing
    functions for that element.

    """
    if isinstance(elem_type, bytes):
        elem_type = elem_type.decode("ascii")
    etype = elem_type.lower()
    if etype in ("quad", "quad4"):
        return Quad4(elem_coord)
    elif etype in ("hex", "hex8"):
        return Hex8(elem_coord)
    elif etype in ("tri3", "triangle", "triangle3"):
        return Tri3(elem_coord)
    elif etype in ("tet", "tet4", "tetra", "tetra4"):
        return Tet4(elem_coord)
    elif etype in ("wedge", "wedge6"):
        return Wedge6(elem_coord)
    raise ValueError(f"==> Error: unknown element type {elem_type!r}")


class Quad4:
    """A QUAD4 exodus element object

    Parameters
    ----------
    coord : array_like
        coordinates of element's nodes, assuming exodus node order convention -
        (counter clockwise around the element)

    """

    dim = 2
    name = "QUAD4"
    nnode = 4

    def __init__(self, coord):
        self.coord = np.array(coord)

    @property
    def volume(self):
        x, y = self.coord[:, 0], self.coord[:, 1]
        return 0.5 * ((x[0] - x[2]) * (y[1] - y[3]) + (x[1] - x[3]) * (y[2] - y[0]))

    @property
    def center(self):
        """Compute the coordinates of the center of the element.

        Note
        ----
        - Simple average in physical space.
        - The result is the same as for subdiv with intervals=1.

        """
        xc = np.average(self.coord[:, 0]) / 4.0
        return np.append(xc, 0.0)

    def subdiv(self, intervals):
        """Compute an equispaced subdivision of a quad4 element.

        Parameters
        ----------
        intervals : int
            The element will be subdivided into nx*ny equispaced subelements, where
            nx = ny = intervals

        Note
        ----
        Quadrature points are equispaced, rather than Gaussian; improved
        order of accuracy of Gaussian quadrature is not achieved for
        discontinuous data.

        """

        x, y = self.coord[:, 0], self.coord[:, 1]
        coord = []
        for jj in range(intervals):
            j = (0.5 + jj) / intervals
            for ii in range(intervals):
                i = (0.5 + ii) / intervals
                xp = (
                    (1 - j) * (1 - i) * x[0]
                    + (1 - j) * i * x[1]
                    + j * i * x[2]
                    + j * (1 - i) * x[3]
                )
                yp = (
                    (1 - j) * (1 - i) * y[0]
                    + (1 - j) * i * y[1]
                    + j * i * y[2]
                    + j * (1 - i) * y[3]
                )
                coord.append([xp, yp, 0.0])

        return np.array(coord)

    def subcoord(self, intervals):
        """Compute an equispaced subdivision of a quad4 element.

        Parameters
        ----------
        intervals : int
            The element will be subdivided into nx*ny equispaced subelements, where
            nx = ny = intervals

        Returns
        -------
        coord : ndarray
            the nodal coordinates of the subelements.

        """
        if intervals <= 0:
            return np.array(self.coord)

        x, y = self.coord[:, 0], self.coord[:, 1]
        coord = []
        for jj in range(intervals + 1):
            j = float(jj) / intervals
            for ii in range(intervals + 1):
                i = float(ii) / intervals
                xp = (
                    (1 - j) * (1 - i) * x[0]
                    + (1 - j) * i * x[1]
                    + j * i * x[2]
                    + j * (1 - i) * x[3]
                )
                yp = (
                    (1 - j) * (1 - i) * y[0]
                    + (1 - j) * i * y[1]
                    + j * i * y[2]
                    + j * (1 - i) * y[3]
                )
                coord.append([xp, yp])

        return np.array(coord)

    def subconn(self, intervals):
        """Compute the connectivity matrix relating the subelements to the
        nodes produced by subcoord.

        Parameters
        ----------
        intervals : int
            The element will be subdivided into nx*ny equispaced subelements, where
            nx = ny = intervals

        Returns
        -------
        conn : ndarray
            Element connectivity

        Note
        ----
        Node indices are 0-based.

        """
        conn = []
        n = intervals + 1
        for j in range(intervals):
            jn = j * n
            j1n = (j + 1) * n
            for i in range(intervals):
                i1 = i + 1
                # 0-based element number is j*intervals + i
                # local node numbers n1, n2, n3, n4
                n1, n2, n3, n4 = i + jn, i1 + jn, i1 + j1n, i + j1n
                conn.append([n1, n2, n3, n4])

        return np.array(conn, dtype=int)

    def subvols(self, intervals):
        """Compute the subelement volumes of an equispaced subdivision of
        a quad4 element.

        Parameters
        ----------
        intervals : int
            The element will be subdivided into nx*ny equispaced subelements, where
            nx = ny = intervals

        Returns
        -------
        vols : ndarray
            Volumes of the subelements

        """

        coord = self.subcoord(intervals)
        conn = self.subconn(intervals)

        m = intervals * intervals
        vols = np.zeros(m)
        for subel in range(m):
            loc_coords = coord[conn[subel]]
            vols[subel] = Quad4(loc_coords).volume

        return vols


class Hex8:
    dim = 3
    name = "HEX8"
    nnode = 8

    def __init__(self, coord):
        self.coord = np.array(coord)

    @property
    def volume(self):
        x1, x2, x3, x4, x5, x6, x7, x8 = self.coord[:, 0]
        y1, y2, y3, y4, y5, y6, y7, y8 = self.coord[:, 1]
        z1, z2, z3, z4, z5, z6, z7, z8 = self.coord[:, 2]

        rx0 = (
            y2 * ((z6 - z3) - (z4 - z5))
            + y3 * (z2 - z4)
            + y4 * ((z3 - z8) - (z5 - z2))
            + y5 * ((z8 - z6) - (z2 - z4))
            + y6 * (z5 - z2)
            + y8 * (z4 - z5)
        )
        rx1 = (
            y3 * ((z7 - z4) - (z1 - z6))
            + y4 * (z3 - z1)
            + y1 * ((z4 - z5) - (z6 - z3))
            + y6 * ((z5 - z7) - (z3 - z1))
            + y7 * (z6 - z3)
            + y5 * (z1 - z6)
        )
        rx2 = (
            y4 * ((z8 - z1) - (z2 - z7))
            + y1 * (z4 - z2)
            + y2 * ((z1 - z6) - (z7 - z4))
            + y7 * ((z6 - z8) - (z4 - z2))
            + y8 * (z7 - z4)
            + y6 * (z2 - z7)
        )
        rx3 = (
            y1 * ((z5 - z2) - (z3 - z8))
            + y2 * (z1 - z3)
            + y3 * ((z2 - z7) - (z8 - z1))
            + y8 * ((z7 - z5) - (z1 - z3))
            + y5 * (z8 - z1)
            + y7 * (z3 - z8)
        )
        rx4 = (
            y8 * ((z4 - z7) - (z6 - z1))
            + y7 * (z8 - z6)
            + y6 * ((z7 - z2) - (z1 - z8))
            + y1 * ((z2 - z4) - (z8 - z6))
            + y4 * (z1 - z8)
            + y2 * (z6 - z1)
        )
        rx5 = (
            y5 * ((z1 - z8) - (z7 - z2))
            + y8 * (z5 - z7)
            + y7 * ((z8 - z3) - (z2 - z5))
            + y2 * ((z3 - z1) - (z5 - z7))
            + y1 * (z2 - z5)
            + y3 * (z7 - z2)
        )
        rx6 = (
            y6 * ((z2 - z5) - (z8 - z3))
            + y5 * (z6 - z8)
            + y8 * ((z5 - z4) - (z3 - z6))
            + y3 * ((z4 - z2) - (z6 - z8))
            + y2 * (z3 - z6)
            + y4 * (z8 - z3)
        )
        rx7 = (
            y7 * ((z3 - z6) - (z5 - z4))
            + y6 * (z7 - z5)
            + y5 * ((z6 - z1) - (z4 - z7))
            + y4 * ((z1 - z3) - (z7 - z5))
            + y3 * (z4 - z7)
            + y1 * (z5 - z4)
        )

        vol = (
            x1 * rx0
            + x2 * rx1
            + x3 * rx2
            + x4 * rx3
            + x5 * rx4
            + x6 * rx5
            + x7 * rx6
            + x8 * rx7
        ) / 12.0

        return vol

    @property
    def center(self):
        """Compute the coordinates of the center of a hex8 element.

        Note
        ----
        - Simple average in physical space.
        - The result is the same as for hex8_subdiv with intervals=1.

        """
        return np.average(self.coord, axis=0)

    def subdiv(self, intervals):
        """Compute an equispaced subdivision of a hex8 element.

        Quadrature points are equispaced, rather than Gaussian; improved
        order of accuracy of Gaussian quadrature is not achieved for
        discontinuous data.

        Parameters
        ----------
        intervals : int
            The element will be subdivided into nx*ny*nz equispaced subelements,
            where nx = ny = nz = intervals

        """
        coord = []
        x, y, z = self.coord[:, 0], self.coord[:, 1], self.coord[:, 2]
        for kk in range(intervals):
            k = (0.5 + kk) / intervals
            for jj in range(intervals):
                j = (0.5 + jj) / intervals
                for ii in range(intervals):
                    i = (0.5 + ii) / intervals
                    xp = (
                        (1 - k) * (1 - j) * (1 - i) * x[0]
                        + (1 - k) * (1 - j) * i * x[1]
                        + (1 - k) * j * i * x[2]
                        + (1 - k) * j * (1 - i) * x[3]
                        + k * (1 - j) * (1 - i) * x[4]
                        + k * (1 - j) * i * x[5]
                        + k * j * i * x[6]
                        + k * j * (1 - i) * x[7]
                    )
                    yp = (
                        (1 - k) * (1 - j) * (1 - i) * y[0]
                        + (1 - k) * (1 - j) * i * y[1]
                        + (1 - k) * j * i * y[2]
                        + (1 - k) * j * (1 - i) * y[3]
                        + k * (1 - j) * (1 - i) * y[4]
                        + k * (1 - j) * i * y[5]
                        + k * j * i * y[6]
                        + k * j * (1 - i) * y[7]
                    )
                    zp = (
                        (1 - k) * (1 - j) * (1 - i) * z[0]
                        + (1 - k) * (1 - j) * i * z[1]
                        + (1 - k) * j * i * z[2]
                        + (1 - k) * j * (1 - i) * z[3]
                        + k * (1 - j) * (1 - i) * z[4]
                        + k * (1 - j) * i * z[5]
                        + k * j * i * z[6]
                        + k * j * (1 - i) * z[7]
                    )
                    coord.append([xp, yp, zp])

        return coord

    def subcoord(self, intervals):
        """Compute an equispaced subdivision of a hex8 element. Return
        the nodal coordinates of the subelements.

        Parameters
        ----------
        intervals : int
            The element will be subdivided into nx*ny equispaced subelements, where
            nx = ny = intervals

        Returns
        -------
        coord : ndarray
            the nodal coordinates of the subelements.

        """
        if intervals <= 0:
            return np.array(self.coord)

        coord = []
        x, y, z = self.coord[:, 0], self.coord[:, 1], self.coord[:, 2]
        for kk in range(intervals + 1):
            k = float(kk) / intervals
            for jj in range(intervals + 1):
                j = float(jj) / intervals
                for ii in range(intervals + 1):
                    i = float(ii) / intervals
                    xp = (
                        (1 - k) * (1 - j) * (1 - i) * x[0]
                        + (1 - k) * (1 - j) * i * x[1]
                        + (1 - k) * j * i * x[2]
                        + (1 - k) * j * (1 - i) * x[3]
                        + k * (1 - j) * (1 - i) * x[4]
                        + k * (1 - j) * i * x[5]
                        + k * j * i * x[6]
                        + k * j * (1 - i) * x[7]
                    )
                    yp = (
                        (1 - k) * (1 - j) * (1 - i) * y[0]
                        + (1 - k) * (1 - j) * i * y[1]
                        + (1 - k) * j * i * y[2]
                        + (1 - k) * j * (1 - i) * y[3]
                        + k * (1 - j) * (1 - i) * y[4]
                        + k * (1 - j) * i * y[5]
                        + k * j * i * y[6]
                        + k * j * (1 - i) * y[7]
                    )
                    zp = (
                        (1 - k) * (1 - j) * (1 - i) * z[0]
                        + (1 - k) * (1 - j) * i * z[1]
                        + (1 - k) * j * i * z[2]
                        + (1 - k) * j * (1 - i) * z[3]
                        + k * (1 - j) * (1 - i) * z[4]
                        + k * (1 - j) * i * z[5]
                        + k * j * i * z[6]
                        + k * j * (1 - i) * z[7]
                    )
                    coord.append([xp, yp, zp])

        return np.array(coord)

    def subconn(self, intervals):
        """Compute the connectivity matrix relating the subelements to the
        nodes produced by subcoord.

        Parameters
        ----------
        intervals : int
            The element will be subdivided into nx*ny equispaced subelements, where
            nx = ny = intervals

        Returns
        -------
        conn : ndarray
            Element connectivity

        Note
        ----
        Node indices are 0-based.


        Input:
        intervals:    The element will be subdivided into nx*ny*nz equispaced
                        subelements, where nx = ny = nz = intervals
        """

        conn = []
        n = intervals + 1
        nn = n * n
        for k in range(intervals):
            knn = k * nn  # k *(intervals+1)^2
            k1nn = (k + 1) * nn  # (k+1)*(intervals+1)^2
            for j in range(intervals):
                jn_knn = j * n + knn  # j *(intervals+1) +    k *(intervals+1)^2
                j1n_knn = (
                    j + 1
                ) * n + knn  # (j+1)*(intervals+1) +    k *(intervals+1)^2
                jn_k1nn = j * n + k1nn  # j *(intervals+1) + (k+1)*(intervals+1)^2
                j1n_k1nn = (
                    j + 1
                ) * n + k1nn  # (j+1)*(intervals+1) + (k+1)*(intervals+1)^2
                for i in range(intervals):
                    i1 = i + 1
                    # 0-based element number is j*intervals + i
                    # local node numbers n1, n2, n3, n4, n5, n6, n7, n8
                    n1 = i + jn_knn
                    n2 = i1 + jn_knn
                    n3 = i1 + j1n_knn
                    n4 = i + j1n_knn
                    n5 = i + jn_k1nn
                    n6 = i1 + jn_k1nn
                    n7 = i1 + j1n_k1nn
                    n8 = i + j1n_k1nn
                    conn.append([n1, n2, n3, n4, n5, n6, n7, n8])

        return np.array(conn, dtype=int)

    def subvols(self, intervals):
        """Compute the subelement volumes of an equispaced subdivision of
        a hex8 element.

        Parameters
        ----------
        intervals : int
            The element will be subdivided into nx*ny equispaced subelements, where
            nx = ny = intervals

        Returns
        -------
        vols : ndarray
            Volumes of the subelements

        """

        coord = self.subcoord(intervals)
        conn = self.subconn(intervals)

        m = intervals ** 3
        vols = np.zeros(m)
        for subel in range(m):
            loc_coords = coord[conn[subel]]
            vols[subel] = Hex8(loc_coords).volume

        return vols


def _midpoint(a, b):
    return [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0]


def _distsquare(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


class Tri3:
    dim = 2
    name = "TRI3"
    nnode = 3

    def __init__(self, coord):
        self.coord = coord

    @property
    def volume(self):
        x, y = self.coord[:, 0], self.coord[:, 1]
        return 0.5 * abs(x[0] * y[1] - y[0] * x[1])

    @property
    def center(self):
        """Compute the coordinates of the center of a tri element.

        Note
        ----
        Simple average in physical space.

        """
        return np.average(self.coord, axis=0)

    def subcoord(self, intervals):
        """Divides the tri into 4^(intervals-1) new triangles, each with equal
        volume. The triangles are recursively divided along their longest edge to
        create two smaller triangles.

        Parameters
        ----------
        intervals : int
            The element will be subdivided into nx*ny*nz equispaced subelements,
            where nx = ny = nz = intervals
        """
        if intervals <= 1:
            return np.array(self.coord)
        triindices = [[0, 1, 2]]
        coord = np.array(self.coord)
        self._subcoord(coord, triindices, intervals * 2 - 2)
        return np.array(coord)

    def subdiv(self, intervals):
        """Compute node center list for a subdivided tri element"""
        triindices = [[0, 1, 2]]
        coord = np.array(self.coord)
        if intervals > 1:
            self._subcoord(coord, triindices, intervals * 2 - 2)
        ntri = len(triindices)
        subdiv = [Tri3(coord[triindices[i]]).center for i in range(ntri)]
        return np.array(subdiv)

    def subvols(self, intervals):
        """Compute volumes of the subdivided tris"""
        triindices = [[0, 1, 2]]
        coord = np.array(self.coord)
        self._subcoord(coord, triindices, intervals * 2 - 2)
        ntri = len(triindices)
        vol = [Tri3(coord[triindices[i]]).volume for i in range(ntri)]
        return np.array(vol)

    def subconn(self, intervals):
        """Compute connectivity map for subdivided tris. elem_coords must be
        supplied."""
        triindices = [[0, 1, 2]]
        if intervals > 1:
            coord = np.array(self.coord)
            self._subcoord(coord, triindices, intervals * 2 - 2)
        return np.array(triindices, dtype=int)

    def _subcoord(self, coord, tris, iterations):
        """Main subivision function. 'coord' contains a list of nodes, and 'tris' is
        a list containing objects like [2,5,7], where coord[2], coord[5], and
        coord[7] would give the nodes of a triangle. This function divides
        each triangle in half recursively, up to 'iterations' recursion levels.
        Each triangle is divided in half on its longest edge, so the final
        triangles should have similar dimensions.

        After running, the lists passed as 'coord' and 'tris' will contain the
        node locations and indexes respectively for the subdivided triangles.
        """

        if iterations <= 0:
            return

        newtris = []
        for tri in tris:
            # find longest edge
            longest = 0
            longi1 = 0
            longi2 = 0
            i3 = 0
            for idx in range(-1, len(tri) - 1):
                ds = _distsquare(coord[idx], coord[idx+1])
                if ds > longest:
                    longest = ds
                    longi1 = tri[idx]
                    longi2 = tri[idx+1]

            i3 = (set(tri) - set((longi1, longi2))).pop()

            # split it
            mp = _midpoint(coord[longi1], coord[longi2])
            coord = np.row_stack((coord, mp))
            mpnode = len(coord) - 1
            # replace it with the two new tets
            newtris.append([longi1, mpnode, i3])
            newtris.append([longi2, mpnode, i3])
        # recurse
        self._subcoord(coord, newtris, iterations - 1)
        tris[:] = newtris


class Tet4:
    dim = 3
    name = "TET4"
    nnode = 4

    def __init__(self, coord):
        self.coord = coord

    @property
    def volume(self):
        amd = [self.coord[0, i] - self.coord[3, i] for i in range(3)]
        bmd = [self.coord[1, i] - self.coord[3, i] for i in range(3)]
        cmd = [self.coord[2, i] - self.coord[3, i] for i in range(3)]
        return abs(np.dot(amd, np.cross(bmd, cmd)) / 6.0)

    @property
    def center(self):
        """Compute the coordinates of the center of a tet element.

        Note
        ----
        Simple average in physical space.

        """
        return np.average(self.coord, axis=0)

    def subcoord(self, intervals):
        """Divides the tet into 4^(intervals-1) new tetrahedrons, each with equal
        volume. The tetrahedrons are recursively divided along their longest edge to
        create two smaller tetrahedrons.

        Parameters
        ----------
        intervals : int
            The element will be subdivided into nx*ny*nz equispaced subelements,
            where nx = ny = nz = intervals
        """
        if intervals <= 1:
            return np.array(self.coord)
        tetindices = [[0, 1, 2, 3]]
        coord = np.array(self.coord)
        self._subcoord(coord, tetindices, intervals * 2 - 2)
        return np.array(coord)

    def subdiv(self, intervals):
        """Compute node center list for a subdivided tet element"""
        tetindices = [[0, 1, 2, 3]]
        coord = np.array(self.coord)
        if intervals > 1:
            self._subcoord(coord, tetindices, intervals * 2 - 2)
        ntet = len(tetindices)
        subdiv = [Tet4(coord[tetindices[i]]).center for i in range(ntet)]
        return np.array(subdiv)

    def subvols(self, intervals):
        """Compute volumes of the subdivided tets"""
        tetindices = [[0, 1, 2, 3]]
        coord = np.array(self.coord)
        self._subcoord(coord, tetindices, intervals * 2 - 2)
        ntet = len(tetindices)
        vol = [Tet4(coord[tetindices[i]]).volume for i in range(ntet)]
        return np.array(vol)

    def subconn(self, intervals):
        """Compute connectivity map for subdivided tets. elem_coords must be
        supplied."""
        tetindices = [[0, 1, 2, 3]]
        if intervals > 1:
            coord = np.array(self.coord)
            self._subcoord(coord, tetindices, intervals * 2 - 2)
        return np.array(tetindices, dtype=int)

    def _subcoord(self, coord, tets, iterations):
        """Main subivision function. 'coord' contains a list of nodes, and 'tets' is
        a list containing objects like [2,5,7,8], where coord[2],coord[5],coord[7]
        and coord[8] would give the nodes of a tetrahedron. This function divides
        each tetrahedron in half recursively, up to 'iterations' recursion levels.
        Each tetrahedron is divided in half on its longest edge, so the final
        tetrahedrons should have similar dimensions.

        After running, the lists passed as 'coord' and 'tets' will contain the
        node locations and indexes respectively for the subdivided tetrahedrons

        """

        if iterations == 0:
            return

        newtets = []
        for tet in tets:
            # find longest edge
            longest = 0
            longi1 = 0
            longi2 = 0
            i3 = 0
            i4 = 0
            for a in tet:
                for b in tet:
                    ds = _distsquare(coord[a], coord[b])
                    if ds > longest:
                        longest = ds
                        longi1 = a
                        longi2 = b

            for a in tet:
                if a != longi1 and a != longi2:
                    if i3 == 0:
                        i3 = a
                    else:
                        i4 = a
            # split it
            mp = _midpoint(coord[longi1], coord[longi2])
            coord = np.row_stack((coord, mp))
            mpnode = len(coord) - 1
            # replace it with the two new tets
            newtets.append([longi1, mpnode, i3, i4])
            newtets.append([longi2, mpnode, i3, i4])
        # recurse
        self._subcoord(coord, newtets, iterations - 1)
        tets[:] = newtets


class Wedge6:
    dim = 3
    name = "WEDGE6"
    nnode = 6

    def __init__(self, coord):
        self.coord = np.array(coord)

    @property
    def center(self):
        return np.average(self.coord, axis=0)

    @property
    def volume(self):
        """Computes volume of the wedge given nodes.

        Calculates volume by cutting into three tetrahedra
        """
        x, y, z = self.coord[:, 0], self.coord[:, 1], self.coord[:, 2]

        i, j, k = 2, 1, 4
        ux, uy, uz = x[i] - x[0], y[i] - y[0], z[i] - z[0]
        vx, vy, vz = x[j] - x[0], y[j] - y[0], z[j] - z[0]
        wx, wy, wz = x[k] - x[0], y[k] - y[0], z[k] - z[0]
        vol = (
            wx * (uy * vz - uz * vy)
            + wy * (vx * uz - ux * vz)
            + wz * (ux * vy - uy * vx)
        )

        i, j, k = 3, 5, 4
        ux, uy, uz = x[i] - x[0], y[i] - y[0], z[i] - z[0]
        vx, vy, vz = x[j] - x[0], y[j] - y[0], z[j] - z[0]
        wx, wy, wz = x[k] - x[0], y[k] - y[0], z[k] - z[0]
        vol += (
            wx * (uy * vz - uz * vy)
            + wy * (vx * uz - ux * vz)
            + wz * (ux * vy - uy * vx)
        )

        i, j, k = 5, 2, 4
        ux, uy, uz = x[i] - x[0], y[i] - y[0], z[i] - z[0]
        vx, vy, vz = x[j] - x[0], y[j] - y[0], z[j] - z[0]
        wx, wy, wz = x[k] - x[0], y[k] - y[0], z[k] - z[0]
        vol += (
            wx * (uy * vz - uz * vy)
            + wy * (vx * uz - ux * vz)
            + wz * (ux * vy - uy * vx)
        )

        return vol

    def subcoord(self, intervals):
        """Divides the wedge into 4^(intervals) new wedges. The wedges are
        recursively divided along their bottom triangular face's longest edge to
        create two and through the middle of vertical extrusion to produce 4 sub
        wedges

        Parameters
        ----------
        intervals : int
            The element will be subdivided into nx*ny*nz equispaced subelements,
            where nx = ny = nz = intervals
        """
        coord = np.array(self.coord)
        if intervals > 1:
            wedgeindices = [[0, 1, 2, 3, 4, 5]]
            self._subcoord(coord, wedgeindices, intervals * 2 - 2)
        return np.array(coord)

    def subdiv(self, intervals):
        """Compute node center list for a subdivided wedges element"""
        wedgeindices = [[0, 1, 2, 3, 4, 5]]
        coord = np.array(self.coord)
        if intervals > 1:
            self._subcoord(coord, wedgeindices, intervals * 2 - 2)
        nwedge = len(wedgeindices)
        subdiv = [Wedge6(coord[wedgeindices[i]]).center for i in range(nwedge)]
        return np.array(subdiv)

    def subvols(self, intervals):
        """Compute volumes of the subdivided wedges"""
        wedgeindices = [[0, 1, 2, 3, 4, 5]]
        coord = np.array_self.coord
        if intervals > 1:
            self._subcoord(coord, wedgeindices, intervals * 2 - 2)
        nwedge = len(wedgeindices)
        vols = [Wedge6(coord[wedgeindices[i]]).volume for i in range(nwedge)]
        return np.array(vols)

    def subconn(self, intervals):
        """Compute connectivity map for subdivided wedges. elem_coords must be
        supplied."""
        wedgeindices = [[0, 1, 2, 3, 4, 5]]
        if intervals > 1:
            coord = np.array(self.coord)
            self._subcoord(coord, wedgeindices, intervals * 2 - 2)
        return np.array(wedgeindices, dtype=int)

    def _subcoord(self, coord, wedges, iterations):

        if iterations == 0:
            return

        newwedges = []
        for wedge in wedges:
            longest = 0
            ni = 0
            nj = 0
            nk = 0

            for i in range(0, 2):
                for j in range(i + 1, 3):
                    ds = _distsquare(coord[wedge[i]], coord[wedge[j]])
                    if ds > longest:
                        longest = ds
                        ni = i
                        nj = j

            for k in range(0, 2):
                if k != ni and k != nj:
                    nk = k

            gi = wedge[ni]
            gj = wedge[nj]
            gk = wedge[nk]
            giu = wedge[ni + 3]
            gju = wedge[nj + 3]
            gku = wedge[nk + 3]

            # SPLIT
            mpbtri = _midpoint(coord[gi], coord[gj])
            mpttri = _midpoint(coord[giu], coord[gju])
            mpvi = _midpoint(coord[gi], coord[giu])
            mpvj = _midpoint(coord[gj], coord[gju])
            mpvk = _midpoint(coord[gk], coord[gku])
            mpvm = _midpoint(mpbtri, mpttri)

            old_max = len(coord)
            coord = np.row_stack(
                (
                    coord,
                    mpbtri,  # old_max+1
                    mpttri,  # old_max+2
                    mpvi,  # old_max+3
                    mpvj,  # old_max+4
                    mpvk,  # old_max+5
                    mpvm,  # old_max+6
                )
            )

            newwedges.extend(
                [
                    # wedge [ni, mpb, nk, vi, vm, vk]
                    [gi, old_max + 1, gk, old_max + 3, old_max + 6, old_max + 5],
                    # wedge [mpb, nj, nk, vm, vj, vk]
                    [old_max + 1, gj, gk, old_max + 6, old_max + 4, old_max + 5],
                    # wedge [vi, vm, vk, ni+3, mpt, nk+3]
                    [old_max + 3, old_max + 6, old_max + 5, giu, old_max + 2, gku],
                    # wedge [vm, vj, vk, mpt, nj+3, nk+3]
                    [old_max + 6, old_max + 4, old_max + 5, old_max + 2, gju, gku],
                ]
            )

        # RECURSE
        self._subcoord(coord, newwedges, iterations - 1)
        wedges[:] = newwedges
