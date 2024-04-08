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


def test_exodusii_write_3(tmpdir):
    # this test sets more values then reads back those values
    ndim = 2
    type = 'QUAD'
    node_count = 4
    cell_count = 1
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    conn = np.array([[0, 1, 2, 3]])
    block_names = ['cell']
    sideset_names = ['sideset_1', 'sideset_2']
    sideset_cells = [[0], [0, 0]]
    sideset_sides = [[1], [2, 3]]
    with util.working_dir(tmpdir.strpath):
        with exodusii.File('write_3.exo', mode="w") as exof: 
            exof.put_init('Write_3', ndim, node_count, cell_count,
                          len(block_names), 0, len(sideset_names))
            exof.put_coord(x, y)
            exof.put_element_block(1, type, cell_count, node_count)
            exof.put_element_block_name(1, 'block_1')
            exof.put_element_conn(1, conn)
            # Side set 1 
            exof.put_side_set_param(1, len(sideset_cells[0]))
            exof.put_side_set_name(1, sideset_names[0])
            exof.put_side_set_sides(1, sideset_cells[0], sideset_sides[0])
            # Side set 2
            exof.put_side_set_param(2, len(sideset_cells[1]))
            exof.put_side_set_name(2, sideset_names[1])
            exof.put_side_set_sides(2, sideset_cells[1], sideset_sides[1])
        with exodusii.File('write_3.exo', mode="r") as exof:
            assert exof.title() == "Write_3"
            assert exof.num_dimensions() == 2
            assert exof.num_nodes() == 4
            assert exof.num_elems() == 1
            assert exof.num_blks() == 1
            assert exof.num_node_sets() == 0
            assert exof.num_side_sets() == 2
            coords = exof.get_coords()
            assert coords.shape == (4,2)
            xf = coords[:,0]
            assert np.allclose(x, xf)
            yf = coords[:,1]
            assert np.allclose(y, yf)
            blk = exof.get_element_block(1)
            assert blk.name == 'block_1'
            ss = exof.get_side_set(1)
            assert ss.name == 'sideset_1'
            ss = exof.get_side_set(2)
            assert ss.name == 'sideset_2'


def test_exodusii_write_4(tmpdir):
    # this test sets multiple block names at once then reads back those values
    ndim = 2
    type = 'QUAD'
    node_count = 4
    cell_count = 1
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    block_names = ['block1', 'block2', 'block3']
    with util.working_dir(tmpdir.strpath):
        with exodusii.File('write_4.exo', mode="w") as exof: 
            exof.put_init('Write_4', ndim, node_count, cell_count,
                          len(block_names), 0, 0)
            exof.put_coord(x, y)
            exof.put_element_block(1, type, cell_count, node_count)
            exof.put_element_block(2, type, cell_count, node_count)
            exof.put_element_block(3, type, cell_count, node_count)
            exof.put_element_block_names(block_names)
        with exodusii.File('write_4.exo', mode="r") as exof:
            assert exof.title() == "Write_4"
            assert exof.num_dimensions() == 2
            assert exof.num_nodes() == 4
            assert exof.num_elems() == 1
            assert exof.num_blks() == 3
            assert exof.num_node_sets() == 0
            assert exof.num_side_sets() == 0
            blk = exof.get_element_block(1)
            assert blk.name == block_names[0]
            blk = exof.get_element_block(2)
            assert blk.name == block_names[1]
            blk = exof.get_element_block(3)
            assert blk.name == block_names[2]

