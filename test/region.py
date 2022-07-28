import numpy as np
import exodusii


def test_region_cylinder_2d_0():
    p1 = [0., 0.]
    p2 = [1., 0.]
    radius = .5
    region = exodusii.region.cylinder(p1, p2, radius)
    a = [0, -radius]
    b = [1, -radius]
    c = [1, radius]
    d = [0, radius]
    points = [a, b, c, d, [b[0], 1.005*b[1]], [c[0], 1.005*c[1]]]
    assert region.contains(points[0])
    assert region.contains(points[1])
    assert region.contains(points[2])
    assert region.contains(points[3])
    assert not region.contains(points[4])
    assert not region.contains(points[5])
    contains = region.contains(points)
    assert contains[0]
    assert contains[1]
    assert contains[2]
    assert contains[3]
    assert not contains[4]
    assert not contains[5]


def test_region_cylinder_2d_1():
    p1 = [0., 0.]
    p2 = [1., 1.]
    radius = .5
    region = exodusii.region.cylinder(p1, p2, radius)
    x = .5 * np.sqrt(2) / 2
    a = [.99 * x, -.99 * x]
    b = [1 + .99 * x, 1 - .99 * x]
    c = [1 - .99 * x, 1 + .99 * x]
    d = [-.99 * x, .99 * x]
    points = [a, b, c, d, [b[0], 1.005*b[1]], [c[0], 1.005*c[1]]]
    assert region.contains(points[0])
    assert region.contains(points[1])
    assert region.contains(points[2])
    assert region.contains(points[3])
    assert not region.contains(points[4])
    assert not region.contains(points[5])
    contains = region.contains(points)
    assert contains[0]
    assert contains[1]
    assert contains[2]
    assert contains[3]
    assert not contains[4]
    assert not contains[5]


def test_region_cylinder_3d():
    p1 = [0., 0., 0.]
    p2 = [1., 0., 0.]
    radius = .5
    region = exodusii.region.cylinder(p1, p2, radius)
    points = [[0, .5, 0], [0, 1.5, 0]]
    assert region.contains(points[0])
    assert not region.contains(points[1])
    contains = region.contains(points)
    assert contains[0]
    assert not contains[1]


def test_region_rectangle():
    origin = [0., -2.5]
    region = exodusii.region.rectangle(origin, 5., 5.)
    points = [[0, 2.5], [-3., 2.5]]
    assert region.contains(points[0])
    assert not region.contains(points[1])
    contains = region.contains(points)
    assert contains[0]
    assert not contains[1]


def test_region_quad():
    region = exodusii.region.quad([1., 1.], [5., 2.], [6., 5.], [0., 3.])
    points = [[2, 2.5], [4., 0.]]
    assert region.contains(points[0])
    assert not region.contains(points[1])
    contains = region.contains(points)
    assert contains[0]
    assert not contains[1]


def test_region_circle():
    region = exodusii.region.circle([0., 0.], 1.)
    points = [[0., 0.], [0., 1.], [1., 0.], [-1., 0], [0., -1.], [0, 1.1]]
    for point in points[:-1]:
        assert region.contains(point)
    assert not region.contains(points[-1])
    contains = region.contains(points)
    assert all(contains[:-1])
    assert not contains[-1]


def test_region_sphere():
    region = exodusii.region.sphere([0., 0., 0.], 1.)
    points = [[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0, -1.], [0., -1., 0], [0, 1.1, .8]]
    for point in points[:-1]:
        assert region.contains(point)
    assert not region.contains(points[-1])
    contains = region.contains(points)
    assert all(contains[:-1])
    assert not contains[-1]
