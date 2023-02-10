import numpy as np


def similar(file1, file2, times=None):
    """Compare two Exodus files, except the solution."""
    from .file import ExodusIIFile

    if not isinstance(file1, ExodusIIFile):
        file1 = ExodusIIFile(file1)

    if not isinstance(file2, ExodusIIFile):
        file2 = ExodusIIFile(file2)

    if file1.num_dimensions() != file2.num_dimensions():
        raise ValueError("Files do not have the same dimension")

    if file1.num_element_blocks() != file2.num_element_blocks():
        raise ValueError("Files do not have same number of element blocks")

    if file1.num_nodes() != file2.num_nodes():
        raise ValueError("Files do not have same number of nodes")

    if file1.num_elems() != file2.num_elems():
        raise ValueError("Files do not have same number of elements")

    if not compare_varnames(
        file1.get_node_variable_names(), file2.get_node_variable_names()
    ):
        raise ValueError("Files do not define the same node variables")

    if not compare_varnames(
        file1.get_edge_variable_names(), file2.get_edge_variable_names()
    ):
        raise ValueError("Files do not define the same edge variables")

    if not compare_varnames(
        file1.get_face_variable_names(), file2.get_face_variable_names()
    ):
        raise ValueError("Files do not define the same face variables")

    if not compare_varnames(
        file1.get_element_variable_names(), file2.get_element_variable_names()
    ):
        raise ValueError("Files do not define the same element variables")

    if not np.allclose(file1.get_element_block_ids(), file1.get_element_block_ids()):
        raise ValueError("Files do not define the same element block IDs")

    for block_id in file1.get_element_block_ids():
        conn1 = file1.get_element_conn(block_id)
        conn2 = file2.get_element_conn(block_id)
        if not np.allclose(conn1, conn2):
            raise ValueError("Files do not have the same node connectivity")

    coords1 = file1.get_coords()
    coords2 = file2.get_coords()
    if not np.allclose(coords1, coords2):
        raise ValueError("Files do not have the same node coordinates")

    if times is not None:
        if not compare_times(file1.get_times(), file2.get_times(), times):
            raise ValueError("Files do not contain the same times")

    return True


def compare_varnames(arg1, arg2):
    if arg1 is None and arg2 is None:
        return True
    elif arg1 is None and arg2 is not None:
        return False
    elif arg2 is None and arg1 is not None:
        return False
    list1 = [x.lower() for x in arg1]
    list2 = [x.lower() for x in arg2]
    return all([x in list2 for x in list1])


def compare_times(times1, times2, times):
    found1, found2 = False, False
    for time in times:
        for t in times1:
            if abs(time - t) < 1e-12:
                found1 = True
                break
        for t in times2:
            if abs(time - t) < 1e-12:
                found2 = True
                break
    return found1 and found2
