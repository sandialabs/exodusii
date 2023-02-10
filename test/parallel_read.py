#!/usr/bin/env python

import os
import glob
import numpy as np
import exodusii
from exodusii.exodus_h import maps


def test_exodusii_read_1(datadir):

    files = glob.glob(os.path.join(datadir, "noh.exo.?.?"))
    exof = exodusii.exo_file(*files)
    assert exof.num_dimensions() == 2
    assert exof.storage_type() == "d"

    assert exof.get_info_records() is None

    nums = [
        exof.num_elem_blk(),
        exof.num_node_sets(),
        exof.num_side_sets(),
        exof.num_elem_maps(),
        exof.num_node_maps(),
        exof.num_edge_blk(),
        exof.num_edge_sets(),
        exof.num_face_blk(),
        exof.num_face_sets(),
        exof.num_elem_sets(),
        exof.num_edge_maps(),
        exof.num_face_maps(),
    ]
    assert nums == [2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    assert exof.get_coord_names().tolist() == ["X", "Y"]

    coords = exof.get_coords()
    assert coords.shape[1] == 2
    xL = [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
    ]
    yL = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert np.allclose(coords[:, 0], xL)
    assert np.allclose(coords[:, 1], yL)

    assert exof.get_element_block_ids().tolist() == [1, 2]
    assert exof.get_node_set_ids().tolist() == [10, 20, 30, 40]
    assert exof.get_element_block_id(1) == 1
    assert exof.get_node_set_id(4) == 40

    info = exof.get_element_block(1)
    assert info.elem_type == "QUAD"
    assert info.num_block_elems == 5
    assert info.num_elem_nodes == 4
    assert info.num_elem_attrs == 0

    info = exof.get_element_block(2)
    assert info.elem_type == "QUAD"
    assert info.num_block_elems == 5
    assert info.num_elem_nodes == 4
    assert info.num_elem_attrs == 0

    assert exof.get_node_set_params(10).num_nodes == 2
    assert exof.get_node_set_params(20).num_nodes == 2
    assert exof.get_node_set_params(30).num_nodes == 11
    assert exof.get_node_set_params(40).num_nodes == 11

    ia = np.array(
        [0, 1, 12, 11, 1, 2, 13, 12, 2, 3, 14, 13, 3, 4, 15, 14, 4, 5, 16, 15]
    )
    conn = exof.get_element_conn(1)
    assert np.allclose(conn.flatten(), ia + 1)

    assert np.allclose([1, 12], exof.get_node_set_nodes(10))
    assert np.allclose([11, 22], exof.get_node_set_nodes(20))
    assert np.allclose([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], exof.get_node_set_nodes(30))
    assert np.allclose(
        [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], exof.get_node_set_nodes(40)
    )

    assert np.allclose([1, 2, 3, 4], exof.get_element_id_map(exof.files[0]))
    assert np.allclose(
        [1, 2, 3, 4, 5, 12, 13, 14, 15, 16],
        exof.get_node_id_map(exof.files[0]),
    )

    assert np.allclose(
        exof.get_times(),
        [
            0.0,
            0.0018,
            0.00396,
            0.006552,
            0.0096624,
            0.01339488,
            0.017873856,
            0.0232486272,
            0.02969835264,
            0.037438023168,
            0.0467256278016,
            0.05787075336192,
            0.071244904034304,
            0.0872938848411648,
            0.106552661809398,
            0.129663194171277,
            0.157395833005533,
            0.190674999606639,
            0.230609999527967,
            0.27853199943356,
            0.336038399320273,
            0.405046079184327,
            0.487855295021193,
            0.587226354025431,
            0.706471624830517,
            0.849565949796621,
            1.02127913975594,
            1.22733496770713,
            1.47460196124856,
            1.77132235349827,
            2.12738682419793,
        ],
    )

    assert exof.get_global_variable_names().tolist() == [
        "DT_HYDRO",
        "DT_HYDROCFL",
        "DT_SOUND",
        "DT_ELASTIC",
        "DT_MATVEL",
        "DT_ARTVIS",
        "DT_VOLUME",
        "ALE_REMESH_CNT",
        "NSTEPS",
        "CPU",
        "GRIND",
        "CPUNOIO",
        "GRINDNOIO",
        "MEMORY_PMAX",
        "MEMFRAGS_PMAX",
        "MEMORY_PMIN",
        "MEMFRAGS_PMIN",
        "TMSTP_CONT_EL",
        "TMSTP_CONT_PROC",
        "ETOT",
        "EINT",
        "ESOURCE",
        "EERROR",
        "PERROR",
        "PTOT",
        "PINT",
        "TM_STEP",
        "MASSTOT",
        "NODEMASS",
        "MASSERR",
        "MASSGAIN",
        "MASSLOSS",
        "EKIN",
        "EKINRZERR",
        "EKINRZTAL",
        "ETOTHYDRZERR",
        "PTOTHYDRZERR",
        "EINTRZFIXADD",
        "EINTRZLOSTUPDATING",
        "PINTRZLOSTUPDATING",
        "EKINLAGSTEP",
        "ETOTHYDLAGSTEP",
        "EINTGAIN",
        "EINTLOSS",
        "EKINGAIN",
        "EKINLOSS",
        "EVELBC",
        "PKIN",
        "EPDV",
        "ENONPDV",
        "PPDV",
        "PNONPDV",
        "PINTRZFIXTOT",
        "PVELBC",
        "XMOM",
        "YMOM",
        "ZMOM",
        "XMOMLAG",
        "YMOMLAG",
        "ZMOMLAG",
        "XMOMRZERR",
        "YMOMRZERR",
        "ZMOMRZERR",
        "NUM_SOLKIN_RESETS",
        "MAT_MASS_1",
        "MAT_MOM_X_1",
        "MAT_MOM_Y_1",
        "MAT_ETOT_1",
        "MAT_EK_1",
        "MAT_EINT_1",
        "MAT_MIN_TEMP_1",
        "MAT_MAX_TEMP_1",
        "MAT_MIN_DENS_1",
        "MAT_MAX_DENS_1",
    ]
    assert exof.get_node_variable_names().tolist() == [
        "DISPLX",
        "DISPLY",
        "VELOCITY_X",
        "VELOCITY_Y",
    ]
    assert exof.get_element_variable_names().tolist() == [
        "DENSITY",
        "ENERGY_1",
        "PROC_ID",
        "VOID_FRC",
        "VOLFRC_1",
    ]

    assert exof.get_displ_variable_names()[0] == "DISPLX"

    gvarL = [
        0.200582879393687,
        0.245792238995311,
        12.4840883973724,
        12.4840883973724,
        0.70980262403434,
        0.249127814687454,
        0.200582879393687,
        0.0,
        2.0,
        0.0921449661254883,
        0.00329089164733887,
        0.0,
        0.0,
        327.0234375,
        0.0,
        326.515625,
        0.0,
        6.0,
        1.0,
        2.75,
        0.00287593306393069,
        0.0,
        1.70002900645727e-16,
        1.01694148207696e-13,
        1.01634750022269e-13,
        0.91515882589384,
        0.00216,
        10.0,
        10.0,
        0.0,
        0.0,
        0.0,
        2.74712406693607,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.74712406693607,
        2.75,
        0.0,
        0.0,
        0.0,
        0.0,
        9.478660739328e-64,
        -0.915158825893738,
        1.2922561445999e-06,
        0.00395353472268889,
        0.000598266733611063,
        0.997756816059669,
        0.0,
        3.0169348418e-61,
        5.5,
        0.0,
        0.0,
        5.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        10.0,
        5.5,
        0.0,
        2.75,
        2.74712406693607,
        0.00287593306393069,
        1.21e-43,
        704885553397428.0,
        1.0,
        1.00395991914769,
    ]
    gvar = exof.get_all_global_variable_values(3)
    assert np.allclose(gvar, gvarL)

    velxL = [
        1.0,
        1.0,
        1.0,
        1.0,
        0.999999978628986,
        0.994734772493227,
        0.00526522750677266,
        2.13710140534428e-08,
        6.11150627070137e-21,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.999999978628986,
        0.994734772493227,
        0.00526522750677266,
        2.13710140534428e-08,
        6.11150627070137e-21,
        0.0,
        0.0,
    ]
    velx = exof.get_node_variable_values("VELOCITY_X", time_step=4)
    assert len(velx) == len(velxL)
    assert np.allclose(velx, velxL)

    engyL1 = [
        2.59816018018567e-35,
        8.29069561528397e-15,
        4.10477981379954e-06,
        0.0101737788835637,
        0.103766889750046,
    ]
    engy = exof.get_element_variable_values(1, "ENERGY_1", 31)
    assert len(engy) == len(engyL1)
    assert np.allclose(engy, engyL1)

    engyL2 = [
        0.28334004018958,
        0.103766889750046,
        0.0101737788835637,
        4.1047798137994e-06,
        8.29070241942468e-15,
    ]
    engy = exof.get_element_variable_values(2, "ENERGY_1", 31)
    assert len(engy) == len(engyL2)
    assert np.allclose(engy, engyL2)

    engyL = engyL1 + engyL2
    engy = exof.get_element_variable_values(None, "ENERGY_1", 31)
    assert len(engy) == len(engyL)
    assert np.allclose(engy, engyL)

    tsL = [
        1.8000000000000002e-03,
        1.8000000000000002e-03,
        2.1600000000000000e-03,
        2.5920000000000001e-03,
        3.1104000000000001e-03,
        3.7324799999999998e-03,
        4.4789760000000000e-03,
        5.3747711999999996e-03,
        6.4497254399999990e-03,
        7.7396705279999985e-03,
        9.2876046335999985e-03,
        1.1145125560319998e-02,
        1.3374150672383997e-02,
        1.6048980806860794e-02,
        1.9258776968232954e-02,
        2.3110532361879543e-02,
        2.7732638834255450e-02,
        3.3279166601106538e-02,
        3.9934999921327846e-02,
        4.7921999905593413e-02,
        5.7506399886712092e-02,
        6.9007679864054511e-02,
        8.2809215836865416e-02,
        9.9371059004238496e-02,
        1.1924527080508619e-01,
        1.4309432496610341e-01,
        1.7171318995932408e-01,
        2.0605582795118890e-01,
        2.4726699354142667e-01,
        2.9672039224971197e-01,
        3.5606447069965436e-01,
    ]
    ts = exof.get_global_variable_values("TM_STEP")
    assert len(ts) == len(tsL)
    assert np.allclose(ts, tsL)

    vL = [
        1.0,
        0.9991,
        0.99711574964,
        0.994734772493,
        0.991880991528,
        0.98846350487,
        0.984374734605,
        0.979488078236,
        0.973655459212,
        0.966705028309,
        0.95843938673,
        0.948634916753,
        0.937043122046,
        0.923395311499,
        0.907412498792,
        0.888822959682,
        0.867390275439,
        0.842954435118,
        0.815486876904,
        0.785156158094,
        0.752393464328,
        0.717937232906,
        0.682828124286,
        0.648327635188,
        0.615752544501,
        0.586246600528,
        0.560525941575,
        0.538624922599,
        0.519713479061,
        0.502336759139,
        0.485852780561,
    ]
    v = exof.get_node_variable_history("VELOCITY_X", 6)
    assert len(v) == len(vL)
    assert np.allclose(v, vL)

    eL = [
        4.9368e-61,
        4.93680133294e-61,
        3.03459399385e-12,
        8.35805930438e-11,
        6.12654077006e-10,
        2.82997861173e-09,
        1.02552350281e-08,
        3.20073624819e-08,
        9.03501695757e-08,
        2.37400453261e-07,
        5.91332274304e-07,
        1.41334733801e-06,
        3.26840718583e-06,
        7.35486975052e-06,
        1.61675206433e-05,
        3.48020804957e-05,
        7.34552970585e-05,
        0.000152063791808,
        0.000308565227843,
        0.000612845535871,
        0.00118865930759,
        0.0022447788773,
        0.00411283766336,
        0.00728102875914,
        0.0123991686804,
        0.0202141071934,
        0.031386178943,
        0.0461691796204,
        0.0640648397757,
        0.0837768855597,
        0.10376688975,
    ]
    e = exof.get_element_variable_history("ENERGY_1", 7)

    assert len(e) == len(eL)
    assert np.allclose(eL, e)

    exof.close()


def test_exodusii_read_mkmesh(tmpdir, datadir):
    from exodusii.util import working_dir

    with working_dir(tmpdir.strpath):
        exof = exodusii.exo_file(
            os.path.join(datadir, "mkmesh.par.2.0"),
            os.path.join(datadir, "mkmesh.par.2.1"),
        )
        assert exof.num_dimensions() == 2
        assert exof.storage_type() == "d"
        assert np.allclose([7, 8, 9, 10, 11, 12], exof.get_node_set_nodes(100))
        assert np.allclose([1, 2, 3, 4, 5, 6], exof.get_node_set_nodes(101))
        assert np.allclose(
            [1.1, 2.1, 3.1, 4.1, 5.1, 6.1], exof.get_node_set_dist_facts(101)
        )

        assert np.allclose([3, 2], exof.num_elems_in_all_blks())
        assert 3 == exof.num_elems_in_blk(10)
        assert 2 == exof.num_elems_in_blk(20)

        elem_map = {(0, 1): 1, (0, 2): 2, (1, 1): 3, (1, 2): 4, (1, 3): 5}
        assert exof.get_mapping(maps.elem_local_to_global) == elem_map

        node_map = {
            (0, 1): 1,
            (0, 2): 2,
            (0, 3): 7,
            (0, 4): 8,
            (0, 5): 3,
            (0, 6): 9,
            (1, 1): 4,
            (1, 2): 5,
            (1, 3): 6,
            (1, 4): 10,
            (1, 5): 11,
            (1, 6): 12,
            (1, 7): 3,
            (1, 8): 9,
        }
        assert exof.get_mapping(maps.node_local_to_global) == node_map

        ss = exof.get_side_set(200)
        assert np.allclose([4, 3, 3, 3, 3, 3], ss.sides)

        ss = exof.get_side_set(201)
        assert np.allclose([1, 2, 3, 4, 5, 5], ss.elems)
        assert np.allclose([1, 1, 1, 1, 1, 2], ss.sides)

        df = exof.get_side_set_dist_facts(201)
        dfL = [1.1, 1.2, 2.1, 2.2, 3.1, 3.2, 4.1, 4.2, 5.1, 5.2, 6.1, 6.2]
        assert np.allclose(df, dfL)
