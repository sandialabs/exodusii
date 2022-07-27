import exodusii


def test_region_cylinder():
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
    origin = [0., 0.]
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
