# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pytest

from exodusii import util


def test_stringify_export() -> None:
    value = np.frombuffer(b"hello   ", dtype="S1")

    assert util.stringify(value) == "hello"


def test_decode() -> None:
    assert util.decode(b"hello   ") == "hello"


def test_index_list() -> None:
    assert util.index(["a", "b", "c"], "b") == 1


def test_index_array() -> None:
    assert util.index(np.asarray([10, 20, 30]), 20) == 1


def test_index_missing_raises() -> None:
    with pytest.raises(ValueError, match="not in array"):
        util.index(np.asarray([10, 20, 30]), 40)


def test_working_dir(tmp_path: Path) -> None:
    cwd = Path.cwd()

    with util.working_dir(tmp_path):
        assert Path.cwd() == tmp_path

    assert Path.cwd() == cwd


def test_compute_connected_average() -> None:
    conn = np.asarray([[0, 1], [1, 2]], dtype=int)
    values = np.asarray([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]])

    result = util.compute_connected_average(conn, values)

    assert np.allclose(result, [[1.0, 2.0], [3.0, 6.0]])


def test_streamify_none() -> None:
    stream, owned = util.streamify(None)

    assert stream is None
    assert not owned


def test_streamify_stream() -> None:
    class Stream:
        def __init__(self) -> None:
            self.text = ""

        def write(self, text: str) -> int:
            self.text += text
            return len(text)

    stream = Stream()
    result, owned = util.streamify(stream)

    assert result is stream
    assert not owned


def test_streamify_path(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"

    stream, owned = util.streamify(path)
    assert stream is not None
    assert owned

    stream.write("hello")
    stream.close()

    assert path.read_text() == "hello"


def test_fmt_join() -> None:
    assert util.fmt_join(fmt="%02d", items=[1, 2, 3], sep=",") == "01,02,03"


@pytest.mark.parametrize(
    ("left", "right"),
    [("node_set", "Node Set"), ("node-set", "node set"), ("node   set", "node_set")],
)
def test_fuzzy_compare_true(left: str, right: str) -> None:
    assert util.fuzzy_compare(left, right)


def test_fuzzy_compare_false() -> None:
    assert not util.fuzzy_compare("node", "element")


def test_find_index_strict() -> None:
    assert util.find_index(["A", "B", "C"], "B") == 1
    assert util.find_index(["A", "B", "C"], "b") is None


def test_find_index_fuzzy() -> None:
    assert util.find_index(["Node Set", "Side Set"], "node_set", strict=False) == 0


def test_find_nearest() -> None:
    index, value = util.find_nearest([0.0, 1.0, 2.0], 1.6)

    assert index == 2
    assert value == 2.0


def test_check_bounds() -> None:
    assert util.check_bounds([0.0, 1.0, 2.0], 1.0)
    assert not util.check_bounds([0.0, 1.0, 2.0], 0.0)
    assert not util.check_bounds([0.0, 1.0, 2.0], 2.0)
    assert not util.check_bounds([], 1.0)


def test_contains() -> None:
    assert util.contains([1, 2, 3], 2)
    assert not util.contains([1, 2, 3], 4)


def test_which_missing() -> None:
    with pytest.raises(ValueError, match="Required executable"):
        util.which("__definitely_not_an_executable__")


def test_epu_no_files() -> None:
    assert util.epu() is None


def test_epu_one_file() -> None:
    assert util.epu("mesh.exo") == "mesh.exo"
