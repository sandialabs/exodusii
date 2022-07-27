import numpy as np


class bounded_time_domain:
    """Represents a bounded time region [t0, tf]

    Parameters
    ----------
    t_min : float
        min of domain
        if None, t_min is set to 0.
    t_max : float
        max of domain
        if None, t_max is set to bounded_time_domain.tmax

    Notes
    -----
    provides a single public method `__call__` that determines whether a time
    (or times) is contained inside the time domain.

    """

    tmax = 1e20

    def __init__(self, t_min=None, t_max=None):
        self.t_min = t_min or 0.0
        self.t_max = t_max or bounded_time_domain.tmax

    def __call__(self, times):
        return np.array((times >= self.t_min) & (times <= self.t_max))


class cylinder:
    """Region defining a cylinder in 2 or 3d space

    Parameters
    ----------
    point1, point2 : array_like
        x, y[, z] coordinates.  If x, y, or z is None, it is replaced with -/+pmax
        in point1 and point 2, respectively.
    radius : float
        The radius of the cylinder

    Notes
    -----
    cylinder provides a single public method `contains` that determines if a point
    (or points) is contained inside the cylinder.

    """

    pmax = 1e20

    def __init__(self, point1, point2, radius):
        self.dimension = len(point1)
        if self.dimension not in (2, 3):
            raise ValueError("Expected cylinder point dimension to be 2 or 3")
        if len(point2) != self.dimension:
            raise ValueError("Inconsistent point dimensions")
        self.p1 = self.aspoint(point1, -cylinder.pmax)
        self.p2 = self.aspoint(point2, cylinder.pmax)
        self.radius = radius

    @staticmethod
    def aspoint(p, default):
        return np.asarray([x if x is not None else default for x in p])

    def contains(self, points):
        """Determine with points is contained in the cylinder

        Parameters
        ----------
        points : array_like
            points[i] are the x, y[, z] coordinates of the point to be queried

        Returns
        -------
        a : ndarray of bool
           a[i] is True if points[i] is in the cylinder

        """
        if self.dimension == 2:
            return self._contains2d(points)
        points = np.asarray(points)
        one_point = points.ndim == 1
        if one_point:
            points = points[np.newaxis, :]
        axis = self.p2 - self.p1
        # points lie between end points of the cylinder
        condition1 = np.einsum("ij,j->i", points - self.p1, axis) >= 0
        condition2 = np.einsum("ij,j->i", points - self.p2, axis) <= 0
        # points lie within curved surface of cylinder
        cp = np.cross(points - self.p1, axis)
        norm = np.abs(cp) if cp.ndim == 1 else np.linalg.norm(cp, axis=1)
        condition3 = norm <= self.radius * np.linalg.norm(axis)
        contains = np.array(condition1 & condition2 & condition3)
        return contains[0] if one_point else contains

    def _contains2d(self, points):
        """A cylinder in 2d is really just a rectangle"""
        axis = self.p2 - self.p1
        u = np.array([-axis[1], axis[0]])
        r = self.radius * u / np.sqrt(np.dot(u, u))
        a = self.p1 - r
        b = self.p2 - r
        c = self.p2 + r
        d = self.p1 + r
        return convex_polygon.contains([a, b, c, d], points)


class convex_polygon:
    dimension = 2

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("complex_polygon is an abstract base class")

    @staticmethod
    def contains(vertices, points):
        """Determine with points is contained in the polygon

        Parameters
        ----------
        points : array_like
            points[i] are the x, y coordinates of the point to be queried

        Returns
        -------
        a : ndarray of bool
           a[i] is True if points[i] is in the polygon

        """
        points = np.asarray(points)
        one_point = points.ndim == 1
        if one_point:
            points = points[np.newaxis, :]
        condition = None
        for (i, a) in enumerate(vertices, start=1):
            b = vertices[i] if i < len(vertices) else vertices[0]
            edge = b - a
            v = points - a
            c = edge[0] * v[:, 1] - v[:, 0] * edge[1] >= 0
            if condition is None:
                condition = c
            else:
                condition = condition & c
        contains = np.array(condition)
        return contains[0] if one_point else contains


class quad(convex_polygon):
    """Region defining a quadrilateral

    Parameters
    ----------
    a, b, c, d : array_like
        x, y coordinates of points in quad

    Notes
    -----
    quad provides a single public method `contains` that determines if a point
    (or points) is contained inside the quad.

    The points are ordered counter-clockwise:

        d-------------------------------c
        |                               |
        |                               |
        a-------------------------------b

    """

    def __init__(self, a, b, c, d):
        for p in (a, b, c, d):
            if len(p) != self.dimension:
                raise ValueError("Expected quad points to be 2D")
        self.a = np.asarray(a)
        self.b = np.asarray(b)
        self.c = np.asarray(c)
        self.d = np.asarray(d)

    def contains(self, points):
        vertices = [self.a, self.b, self.c, self.d]
        return convex_polygon.contains(vertices, points)


class rectangle(quad):
    """Region defining a rectangle in 2d space

    Parameters
    ----------
    origin : array_like
        x, y coordinates of *center* of left hand side
    width : float
        The width of the rectangle
    height : float
        The height of the rectangle

    Notes
    -----
    rectangle provides a single public method `contains` that determines if a point
    (or points) is contained inside the rectangle.

    The origin of the rectangle is as shown below
         _______________________________
        |                               |
        o                               h
        |_____________ w _______________|

    """

    pmax = 1e20

    def __init__(self, origin, width=None, height=None):
        if len(origin) != self.dimension:
            raise ValueError("Expected rectangle origin to be 2D")

        width = width or rectangle.pmax
        if width <= 0:
            raise ValueError("Expected rectangle width > 0")

        height = height or width
        if height <= 0:
            raise ValueError("Expected rectangle height > 0")

        origin = np.asarray(origin)
        self.a = origin + np.array([0, -height / 2.0])
        self.b = origin + np.array([width, -height / 2.0])
        self.c = origin + np.array([width, height / 2.0])
        self.d = origin + np.array([0, height / 2.0])


class sphere:
    """Region defining a sphere in 3d space

    Parameters
    ----------
    center : array_like
        x, y, z coordinates of center of sphere
    radius : float
        The radius of the sphere

    Notes
    -----
    sphere provides a single public method `contains` that determines if a point
    (or points) is contained inside the sphere.

    """

    pmax = 1e20

    def __init__(self, center, radius):
        self.dimension = 3
        if len(center) != self.dimension:
            raise ValueError("Expected sphere center to be 3D")
        self.center = np.asarray(center)

        if radius <= 0:
            raise ValueError("Expected sphere radius > 0")
        self.radius = radius

    def contains(self, points):
        """Determine with points is contained in the sphere

        Parameters
        ----------
        points : array_like
            points[i] are the x, y, z coordinates of the point to be queried

        Returns
        -------
        a : ndarray of bool
           a[i] is True if points[i] is in the sphere

        """
        p = np.asarray(points)
        one_point = p.ndim == 1
        if one_point:
            p = p[np.newaxis, :]

        cx, cy, cz = self.center
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        condition = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < self.radius ** 2
        contains = np.array(condition)

        return contains[0] if one_point else contains


class circle:
    """Region defining a circle in 2d space

    Parameters
    ----------
    center : array_like
        x, y coordinates of center of circle
    radius : float
        The radius of the circle

    Notes
    -----
    circle provides a single public method `contains` that determines if a point
    (or points) is contained inside the circle.

    """

    pmax = 1e20

    def __init__(self, center, radius):
        self.dimension = 2
        if len(center) != self.dimension:
            raise ValueError("Expected circle center to be 3D")
        self.center = np.asarray(center)

        if radius <= 0:
            raise ValueError("Expected circle radius > 0")
        self.radius = radius

    def contains(self, points):
        """Determine with points is contained in the circle

        Parameters
        ----------
        points : array_like
            points[i] are the x, y coordinates of the point to be queried

        Returns
        -------
        a : ndarray of bool
           a[i] is True if points[i] is in the circle

        """
        p = np.asarray(points)
        one_point = p.ndim == 1
        if one_point:
            p = p[np.newaxis, :]

        cx, cy = self.center
        x, y = p[:, 0], p[:, 1]
        condition = (x - cx) ** 2 + (y - cy) ** 2 < self.radius ** 2
        contains = np.array(condition)

        return contains[0] if one_point else contains
