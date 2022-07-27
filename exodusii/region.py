import numpy as np


class bounded_time_region:
    """Represents a bounded time region [t0, tf]

    Parameters
    ----------
    t_min, t_max : float
        min and max of region

    Notes
    -----
    provides a single public method `__call__` that determines whether a time
    (or times) is contained inside the time domain.

    """
    tmax = 1e20

    def __init__(self, t_min=None, t_max=None):
        self.t_min = t_min or 0.0
        self.t_max = t_max or bounded_time_region.tmax

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
