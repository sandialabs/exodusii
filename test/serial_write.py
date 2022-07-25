import os
import numpy as np

import exodusii
import exodusii.util as util


def test_exodusii_write_1(tmpdir):
    # this test sets some initialization values then reads back those values
    with util.working_dir(tmpdir.strpath):
        f = "baz.exo"
        with exodusii.File(f, mode="w") as exof:
            exof.put_init(f"Test {f}", 2, 1, 2, 3, 4, 5)

        with exodusii.File(f, mode="r") as exof:
            assert exof.title() == f"Test {f}", exof.title()
            assert exof.num_dimensions() == 2
            assert exof.num_nodes() == 1
            assert exof.num_elems() == 2
            assert exof.num_blks() == 3
            assert exof.num_node_sets() == 4
            assert exof.num_side_sets() == 5


def test_exodusii_write_2(tmpdir):
    # this test sets some initialization values then reads back those values
    with util.working_dir(tmpdir.strpath):
        baz = np.linspace(0, 10, 25)
        foo = np.linspace(10, 20, 25)
        spam = np.linspace(20, 30, 25)
        data = {"baz": baz, "foo": foo, "spam": spam}
        times = np.linspace(0, 100, 25)
        f = exodusii.write_globals(data, times, "Test")
        with exodusii.File(f, mode="r") as exof:
            assert exof.title() == "Test"
            assert exof.num_dimensions() == 1
            assert exof.num_nodes() == 0
            assert exof.num_elems() == 0
            assert exof.num_blks() == 0
            assert exof.num_node_sets() == 0
            assert exof.num_side_sets() == 0
            assert sorted(exof.get_global_variable_names()) == ["baz", "foo", "spam"]

            values = exof.get_global_variable_values("baz")
            assert np.allclose(values, baz)

            values = exof.get_global_variable_values("foo")
            assert np.allclose(values, foo)

            values = exof.get_global_variable_values("spam")
            assert np.allclose(values, spam)

            values = exof.get_times()
            assert np.allclose(values, times)
