from .file import ExodusIIFile
from . import exodus_h as ex

def put_nodal_solution(filename, nodmap, elemap, coord, elecon, elem_blocks, u):

    exo = ExodusIIFile(filename, mode="w")
    fh = exo.fh

    # initialize file with parameters
    num_nodes, num_dim = coord.shape
    num_elem, node_per_elem = elecon.shape
    num_elem_block = len(elem_blocks)
    num_node_sets = 0
    num_side_sets = 0
    exo.put_init(
        "FETK Nodal Solution",
        num_dim,
        num_nodes,
        num_elem,
        num_elem_bock,
        num_node_sets,
        num_side_sets,
    )
    exo.create_dimension(ex.DIM_NUM_GLO_VAR, 1)
    exo.create_variable(ex.VALS_GLO_VAR, float, (ex.DIM_TIME_STEP,))

    node_variable_names = ["displ%s" % _ for _ in "xyz"[:num_dim]]
    num_node_variables = len(node_variable_names)
    exo.create_dimension(ex.DIM_NUM_NOD_VAR, num_node_variables)
    exo.createVariable(VAR_NAME_NOD_VAR, CHAR, (DIM_NUM_NOD_VAR, DIM_LEN_STRING))
    for (k, node_variable) in enumerate(node_variable_names):
        key = adjstr(nodvar)
        fh.variables[VAR_NAME_NOD_VAR][k, :] = key
        fh.createVariable(VALS_NOD_VAR(k + 1), FLOAT, (DIM_TIME_STEP, DIM_NUM_NOD))

    u0 = zeros_like(u)
    fh.variables[VAR_TIME_WHOLE][0] = 0.0
    fh.variables[VALS_GLO_VAR][0] = 0.0
    for (k, label) in enumerate(nodvarnames):
        fh.variables[VALS_NOD_VAR(k + 1)][0] = u0[:, k]

    fh.variables[VAR_TIME_WHOLE][1] = 1.0
    fh.variables[VALS_GLO_VAR][1] = 1.0
    for (k, label) in enumerate(nodvarnames):
        fh.variables[VALS_NOD_VAR(k + 1)][1] = u[:, k]

    exo.update()
    exo.close()
