from pathlib import Path

import numpy as np

import exodusii


def test_vcomp_required_exodus_api(tmp_path: Path) -> None:
    path = tmp_path / "vcomp.exo"

    with exodusii.File(path, mode="w") as exo:
        exo.put_init("vcomp", 2, 4, 1, 1, 0, 0)
        exo.put_coords(np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
        exo.put_element_block(10, "quad", 1, 4)
        exo.put_element_conn(10, [[1, 2, 3, 4]])

        exo.put_node_variable_params(1)
        exo.put_node_variable_names(["TEMP"])
        exo.put_element_variable_params(1)
        exo.put_element_variable_names(["ENERGY"])

        exo.put_time(1, 0.0)
        exo.put_node_variable_values(1, "TEMP", [1.0, 2.0, 3.0, 4.0])
        exo.put_element_variable_values(1, 10, "ENERGY", [5.0])

        exo.put_time(2, 1.0)
        exo.put_node_variable_values(2, "TEMP", [2.0, 3.0, 4.0, 5.0])
        exo.put_element_variable_values(2, 10, "ENERGY", [6.0])

    exo = exodusii.exo_file(path)

    assert np.allclose(exo.get_times(), [0.0, 1.0])
    assert exo.get_time_step(1.0) == 2

    varnames = exo.get_node_variable_names().tolist()
    varnames.extend(exo.get_element_variable_names())
    assert varnames == ["TEMP", "ENERGY"]

    block_ids = exo.get_element_block_ids()
    assert block_ids.tolist() == [10]

    assert np.allclose(exo.get_node_variable_values("TEMP", time_step=2), [2.0, 3.0, 4.0, 5.0])
    assert np.allclose(exo.get_element_variable_values(10, "ENERGY", time_step=2), [6.0])

    exo.close()


def test_similar_accepts_legacy_positional_times(tmp_path: Path) -> None:
    path = tmp_path / "vcomp.exo"

    with exodusii.File(path, mode="w") as exo:
        exo.put_init("vcomp", 2, 0, 0, 0, 0, 0)
        exo.put_time(1, 0.0)

    a = exodusii.exo_file(path)
    b = exodusii.exo_file(path)

    assert exodusii.similar(a, b, [0.0])

    a.close()
    b.close()
