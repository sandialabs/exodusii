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
        If true, strip trailing whitespace.  Exodus string fields are
        fixed-width and padded; leading whitespace is significant.

    Returns
    -------
    str
        The decoded text.

    Notes
    -----
    Masked character entries decode to ``""``. This is useful for NetCDF4 char
    arrays where unused trailing characters may be masked.

    Text ends at the first NUL (stale bytes may follow it).  Bytes that are
    not valid UTF-8 are decoded as Latin-1.
    """

    if np.ma.is_masked(value):
        return ""

    if isinstance(value, str):
        text = value
    elif isinstance(value, bytes | np.bytes_):
        text = _decode_bytes(bytes(value))
    else:
        text = str(value)

    text = text.split("\x00", 1)[0]
    return text.rstrip() if strip else text


def _decode_bytes(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


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
    values: str | Iterable[str], *, width: int = 32, dtype: str = "S1", total: int | None = None
) -> np.ndarray:
    """Encode strings as a fixed-width NetCDF-style character array.

    Parameters
    ----------
    values
        A single string or iterable of strings.
    width
        Maximum number of characters stored per string (longer strings are
        truncated).
    dtype
        Character dtype. Usually ``"S1"`` for bytes or ``"U1"`` for Unicode.
    total
        Width of the output field (default ``width``).  Use ``total = width +
        1`` to guarantee a terminating NUL, as Exodus C readers expect.

    Returns
    -------
    ndarray
        If ``values`` is a string, shape is ``(total,)``. Otherwise shape is
        ``(len(values), total)``.
    """

    if width < 1:
        raise ValueError("width must be positive")
    field = width if total is None else total
    if field < width:
        raise ValueError("total must be at least width")

    if isinstance(values, str):
        return _pad(_encode_one_fixed_width(values, width=width, dtype=dtype), field, dtype)

    encoded = [
        _pad(_encode_one_fixed_width(value, width=width, dtype=dtype), field, dtype)
        for value in values
    ]
    return np.asarray(encoded, dtype=dtype)


def _pad(chars: np.ndarray, total: int, dtype: str) -> np.ndarray:
    if len(chars) == total:
        return chars
    out = np.zeros(total, dtype=dtype)
    out[: len(chars)] = chars
    return out


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
    plain = np.asarray(array)
    if plain.dtype.kind == "S" and not np.ma.is_masked(array):
        # Decode the field as a whole so multi-byte UTF-8 survives.
        text = _decode_bytes(plain.tobytes())
    else:
        text = "".join(_character(item) for item in array)
    text = text.split("\x00", 1)[0]
    return text.rstrip() if strip else text


def _character(item: Any) -> str:
    if np.ma.is_masked(item):
        return "\x00"
    if isinstance(item, bytes | np.bytes_):
        return _decode_bytes(bytes(item)) or "\x00"
    return str(item) or "\x00"


def _encode_one_fixed_width(value: str, *, width: int, dtype: str) -> np.ndarray:
    text = str(value).split("\x00", 1)[0]

    np_dtype = np.dtype(dtype)
    out = np.zeros(width, dtype=dtype)

    if np_dtype.kind == "S":
        raw = text.encode("utf-8")[:width]
        # Do not cut a multi-byte UTF-8 sequence in half.
        raw = raw.decode("utf-8", errors="ignore").encode("utf-8")
        if raw:
            out[: len(raw)] = np.frombuffer(raw, dtype="S1")
        return out

    if np_dtype.kind == "U":
        text = text[:width]
        if text:
            out[: len(text)] = np.asarray(list(text), dtype=dtype)
        return out

    raise ValueError(f"unsupported fixed-width character dtype {dtype!r}")


__all__ = ["decode_text", "encode_fixed_width", "string_array", "stringify"]
