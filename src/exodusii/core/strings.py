# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""String decoding helpers for Exodus NetCDF character arrays."""

from collections.abc import Iterable
from typing import Any

import numpy as np


def decode_text(value: Any, *, strip: bool = True) -> str:
    """Decode a scalar text-like value to ``str``.

    Parameters
    ----------
    value
        A Python string, bytes object, NumPy byte/string scalar, masked value, or
        other scalar object.
    strip
        If true, strip trailing NULs and whitespace. Exodus string fields are
        fixed-width and commonly padded.

    Returns
    -------
    str
        The decoded text.

    Notes
    -----
    Masked character entries decode to ``""``. This is useful for NetCDF4 char
    arrays where unused trailing characters may be masked.
    """

    if np.ma.is_masked(value):
        return ""

    if isinstance(value, str):
        text = value
    elif isinstance(value, bytes):
        text = value.decode("utf-8")
    elif isinstance(value, np.bytes_):
        text = bytes(value).decode("utf-8")
    elif isinstance(value, np.str_):
        text = str(value)
    else:
        text = str(value)

    text = text.replace("\x00", "")
    return text.strip() if strip else text.rstrip("\x00")


def stringify(value: Any, *, strip: bool = True) -> str | np.ndarray:
    """Decode Exodus-style string data.

    Parameters
    ----------
    value
        A scalar string/bytes value, NumPy string scalar, or NumPy array. Common
        Exodus forms include:

        - ``S1`` or ``U1`` char vectors, e.g. ``["t", "i", "m", "e"]``
        - 2-D char arrays, e.g. ``(num_names, len_string)``
        - 3-D char arrays, e.g. QA records ``(num_qa, 4, len_string)``

    strip
        If true, strip trailing whitespace from decoded strings.

    Returns
    -------
    str or ndarray
        A scalar string for scalar or 1-D character data. A NumPy array of strings
        for 2-D or higher character data.
    """

    if isinstance(value, np.ndarray | np.ma.MaskedArray):
        return _stringify_array(value, strip=strip)

    return decode_text(value, strip=strip)


def string_array(value: Any, *, strip: bool = True) -> np.ndarray:
    """Decode input as a 1-D NumPy array of strings."""

    decoded = stringify(value, strip=strip)

    if isinstance(decoded, str):
        return np.asarray([decoded], dtype=str)

    array = np.asarray(decoded, dtype=str)
    if array.ndim == 0:
        return array.reshape(1)

    return array.reshape(-1)


def encode_fixed_width(
    values: str | Iterable[str], *, width: int = 32, dtype: str = "S1"
) -> np.ndarray:
    """Encode strings as a fixed-width NetCDF-style character array.

    Parameters
    ----------
    values
        A single string or iterable of strings.
    width
        Fixed output width.
    dtype
        Character dtype. Usually ``"S1"`` for bytes or ``"U1"`` for Unicode.

    Returns
    -------
    ndarray
        If ``values`` is a string, shape is ``(width,)``. Otherwise shape is
        ``(len(values), width)``.
    """

    if width < 1:
        raise ValueError("width must be positive")

    if isinstance(values, str):
        return _encode_one_fixed_width(values, width=width, dtype=dtype)

    encoded = [_encode_one_fixed_width(value, width=width, dtype=dtype) for value in values]
    return np.asarray(encoded, dtype=dtype)


def _stringify_array(value: np.ndarray | np.ma.MaskedArray, *, strip: bool) -> str | np.ndarray:
    array = np.ma.asarray(value)

    if array.ndim == 0:
        return decode_text(array.item(), strip=strip)

    if array.ndim == 1:
        if _is_character_vector(array):
            return _join_character_vector(array, strip=strip)

        return np.asarray([decode_text(item, strip=strip) for item in array], dtype=str)

    rows = [_stringify_array(row, strip=strip) for row in array]

    if array.ndim == 3:
        return np.asarray(
            [" ".join(np.asarray(row, dtype=str).tolist()) for row in rows], dtype=str
        )

    return np.asarray(rows, dtype=str)


def _is_character_vector(array: np.ndarray | np.ma.MaskedArray) -> bool:
    dtype = np.asarray(array).dtype

    if dtype.kind not in {"S", "U"}:
        return False

    return dtype.itemsize <= np.dtype(dtype.kind + "1").itemsize


def _join_character_vector(array: np.ndarray | np.ma.MaskedArray, *, strip: bool) -> str:
    text = "".join(decode_text(item, strip=False) for item in array)
    text = text.replace("\x00", "")
    return text.strip() if strip else text.rstrip("\x00")


def _encode_one_fixed_width(value: str, *, width: int, dtype: str) -> np.ndarray:
    text = f"{value:<{width}}"[:width]
    return np.asarray(list(text), dtype=dtype)


__all__ = ["decode_text", "encode_fixed_width", "string_array", "stringify"]
