# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from exodusii.core.strings import decode_text
from exodusii.core.strings import encode_fixed_width
from exodusii.core.strings import string_array
from exodusii.core.strings import stringify


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("hello", "hello"),
        ("hello   ", "hello"),
        (b"hello", "hello"),
        (b"hello   ", "hello"),
        (np.bytes_(b"hello"), "hello"),
        (np.str_("hello"), "hello"),
        ("hello\x00\x00", "hello"),
    ],
)
def test_decode_text(value: object, expected: str) -> None:
    assert decode_text(value) == expected


def test_decode_text_masked_value() -> None:
    assert decode_text(np.ma.masked) == ""


def test_decode_text_can_preserve_whitespace() -> None:
    assert decode_text("hello   ", strip=False) == "hello   "


def test_stringify_python_string() -> None:
    assert stringify("mesh title") == "mesh title"


def test_stringify_python_bytes() -> None:
    assert stringify(b"mesh title   ") == "mesh title"


def test_stringify_zero_dimensional_array() -> None:
    value = np.asarray(b"title")
    assert stringify(value) == "title"


def test_stringify_one_dimensional_byte_character_vector() -> None:
    value = np.frombuffer(b"title   ", dtype="S1")
    assert stringify(value) == "title"


def test_stringify_one_dimensional_unicode_character_vector() -> None:
    value = np.asarray(list("title   "), dtype="U1")
    assert stringify(value) == "title"


def test_stringify_one_dimensional_string_array() -> None:
    value = np.asarray(["alpha", "beta", "gamma"])
    result = stringify(value)

    assert isinstance(result, np.ndarray)
    assert result.tolist() == ["alpha", "beta", "gamma"]


def test_stringify_two_dimensional_byte_character_array() -> None:
    value = np.frombuffer(b"alpha   beta    gamma   ", dtype="S1").reshape(3, 8)

    result = stringify(value)

    assert isinstance(result, np.ndarray)
    assert result.tolist() == ["alpha", "beta", "gamma"]


def test_stringify_two_dimensional_unicode_character_array() -> None:
    value = np.asarray([list("alpha   "), list("beta    "), list("gamma   ")], dtype="U1")

    result = stringify(value)

    assert isinstance(result, np.ndarray)
    assert result.tolist() == ["alpha", "beta", "gamma"]


def test_stringify_three_dimensional_qa_array() -> None:
    value = np.asarray(
        [
            [list("code    "), list("version "), list("date    "), list("time    ")],
            [list("other   "), list("v2      "), list("date2   "), list("time2   ")],
        ],
        dtype="U1",
    )

    result = stringify(value)

    assert isinstance(result, np.ndarray)
    assert result.tolist() == ["code version date time", "other v2 date2 time2"]


def test_stringify_masked_character_array() -> None:
    value = np.ma.asarray(np.frombuffer(b"abc   ", dtype="S1").copy())
    value[3:] = np.ma.masked

    assert stringify(value) == "abc"


def test_string_array_from_scalar() -> None:
    result = string_array("alpha")

    assert result.shape == (1,)
    assert result.tolist() == ["alpha"]


def test_string_array_from_character_name_table() -> None:
    value = np.frombuffer(b"alpha   beta    ", dtype="S1").reshape(2, 8)

    result = string_array(value)

    assert result.shape == (2,)
    assert result.tolist() == ["alpha", "beta"]


def test_string_array_flattens_higher_dimensional_decoded_arrays() -> None:
    value = np.asarray([[list("a   "), list("b   ")], [list("c   "), list("d   ")]], dtype="U1")

    result = string_array(value)

    assert result.shape == (2,)
    assert result.tolist() == ["a b", "c d"]


def test_encode_fixed_width_single_string() -> None:
    encoded = encode_fixed_width("abc", width=5)

    assert encoded.shape == (5,)
    assert encoded.dtype.kind == "S"
    assert stringify(encoded, strip=False) == "abc  "


def test_encode_fixed_width_iterable_of_strings() -> None:
    encoded = encode_fixed_width(["abc", "def"], width=5)

    assert encoded.shape == (2, 5)
    assert stringify(encoded).tolist() == ["abc", "def"]  # ty: ignore[unresolved-attribute]


def test_encode_fixed_width_truncates() -> None:
    encoded = encode_fixed_width("abcdef", width=3)

    assert stringify(encoded) == "abc"


def test_encode_fixed_width_rejects_nonpositive_width() -> None:
    with pytest.raises(ValueError, match="width must be positive"):
        encode_fixed_width("abc", width=0)
