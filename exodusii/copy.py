from .exodus_h import types


copy_extra_set_info = False


def copy(source, target):
    """Copy the source ExodusII file to the target"""
    target.put_init(
        source.title(),
        source.num_dimensions(),
        source.num_nodes(),
        source.num_elems(),
        source.num_blks(),
        source.num_node_sets(),
        source.num_side_sets(),
        num_edge=source.num_edges(),
        num_edge_blk=source.num_edge_blk(),
        num_face=source.num_faces(),
        num_face_blk=source.num_face_blk(),
    )
    copy_mesh(source, target)
    copy_variable_params(source, target)
    copy_variable_histories(source, target)


def copy_mesh(source, target):
    """Copies ExodusII mesh information from source to target"""

    target.put_coord_names(source.get_coord_names())
    target.put_coords(source.get_coords())

    for block in source.elem_blocks():
        target.put_element_block(
            block.id,
            block.elem_type,
            block.num_block_elems,
            block.num_elem_nodes,
            num_faces_per_elem=block.num_elem_faces,
            num_edges_per_elem=block.num_elem_edges,
            num_attr=block.num_elem_attrs,
        )
        for type in (types.node, types.edge, types.face):
            conn = source.get_element_conn(block.id, type=type)
            if conn is not None:
                target.put_element_conn(block.id, conn, type=type)

    if target.num_faces():
        for block in source.face_blocks():
            target.put_face_block(
                block.id,
                block.elem_type,
                block.num_block_faces,
                block.num_face_nodes,
                block.num_face_attrs,
            )
            conn = source.get_face_block_conn(block.id)
            target.put_face_conn(block.id, conn)

    if target.num_edges():
        for block in source.edge_blocks():
            target.put_edge_block(
                block.id,
                block.elem_type,
                block.num_block_edges,
                block.num_edge_nodes,
                block.num_edge_attrs,
            )
            conn = source.get_edge_block_conn(block.id)
            target.put_edge_conn(block.id, conn)

    for ns in source.node_sets():
        target.put_node_set_param(ns.id, ns.num_nodes, ns.num_dist_facts)
        target.put_node_set_name(ns.id, ns.name)
        target.put_node_set_nodes(ns.id, ns.nodes)
        if ns.num_dist_facts:
            target.put_node_set_dist_fact(ns.id, ns.dist_facts)

    for es in source.edge_sets():
        target.put_edge_set_param(
            es.id, es.num_edges, es.num_nodes_per_edge, es.num_dist_facts
        )

    for fs in source.face_sets():
        target.put_face_set_param(
            fs.id, fs.num_faces, fs.num_nodes_per_face, fs.num_dist_facts
        )

    for es in source.elem_sets():
        target.put_element_set_param(es.id, es.num_elems, es.num_dist_facts)

    for ss in source.side_sets():
        target.put_side_set_param(ss.id, ss.num_sides, ss.num_dist_facts)
        target.put_side_set_name(ss.id, ss.name)
        target.put_side_set_sides(ss.id, ss.elems, ss.sides)
        if ss.num_dist_facts:
            target.put_side_set_dist_fact(ss.id, ss.dist_facts)

    node_id_map = source.get_node_id_map()
    target.put_node_id_map(node_id_map)

    elem_id_map = source.get_element_id_map()
    target.put_element_id_map(elem_id_map)

    if source.num_edges():
        edge_id_map = source.get_edge_id_map()
        target.put_edge_id_map(edge_id_map)

    if source.num_faces():
        face_id_map = source.get_face_id_map()
        target.put_face_id_map(face_id_map)


def copy_variable_params(source, target):

    target.put_global_variable_params(source.get_global_variable_number())
    target.put_global_variable_names(source.get_global_variable_names())

    target.put_node_variable_params(source.get_node_variable_number())
    target.put_node_variable_names(source.get_node_variable_names())

    if source.get_element_variable_number() is not None:
        target.put_element_variable_params(source.get_element_variable_number())
        target.put_element_variable_names(source.get_element_variable_names())
        table = source.get_element_variable_truth_table()
        if table is not None:
            target.put_element_variable_truth_table(table)

    if source.get_face_variable_number() is not None:
        target.put_face_variable_params(source.get_face_variable_number())
        target.put_face_variable_names(source.get_face_variable_names())
        table = source.get_face_variable_truth_table()
        if table is not None:
            target.put_face_variable_truth_table(table)

    if source.get_edge_variable_number() is not None:
        target.put_edge_variable_params(source.get_edge_variable_number())
        target.put_edge_variable_names(source.get_edge_variable_names())
        table = source.get_edge_variable_truth_table()
        if table is not None:
            target.put_edge_variable_truth_table(table)

    target.put_node_set_variable_params(source.get_node_set_variable_number())
    target.put_node_set_variable_names(source.get_node_set_variable_names())

    if copy_extra_set_info:

        target.put_edge_set_variable_params(source.get_edge_set_variable_number())
        target.put_edge_set_variable_names(source.get_edge_set_variable_names())

        target.put_face_set_variable_params(source.get_face_set_variable_number())
        target.put_face_set_variable_names(source.get_face_set_variable_names())

        target.put_element_set_variable_params(source.get_element_set_variable_number())
        target.put_element_set_variable_names(source.get_element_set_variable_names())

        target.put_side_set_variable_params(source.get_side_set_variable_number())
        target.put_side_set_variable_names(source.get_side_set_variable_names())


def copy_variable_histories(source, target):

    for (time_step, time) in enumerate(source.get_times(), start=1):
        target.put_time(time_step, time)

    values = source.get_all_global_variable_values()
    target.put_global_variable_values(None, values)

    for name in source.get_node_variable_names():
        values = source.get_node_variable_values(name)
        target.put_node_variable_values(None, name, values)

    for block_id in source.get_element_block_ids():
        for name in source.get_element_variable_names():
            values = source.get_element_variable_values(block_id, name)
            target.put_element_variable_values(None, block_id, name, values)

    for block_id in source.get_edge_block_ids():
        for name in source.get_edge_variable_names():
            values = source.get_edge_variable_values(block_id, name)
            target.put_edge_variable_values(None, block_id, name, values)

    for block_id in source.get_face_block_ids():
        for name in source.get_face_variable_names():
            values = source.get_face_variable_values(block_id, name)
            target.put_face_variable_values(None, block_id, name, values)

    if copy_extra_set_info:
        for set_id in source.get_node_set_ids():
            for name in source.get_node_set_variable_names():
                values = source.get_node_set_variable_values(block_id, name)
                target.put_node_set_variable_values(None, block_id, name, values)

        for set_id in source.get_side_set_ids():
            for name in source.get_side_set_variable_names():
                values = source.get_side_set_variable_values(block_id, name)
                target.put_side_set_variable_values(None, block_id, name, values)
