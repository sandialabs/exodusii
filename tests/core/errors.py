# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import pytest

from exodusii.core.errors import ExodusConsistencyError
from exodusii.core.errors import ExodusDimensionError
from exodusii.core.errors import ExodusError
from exodusii.core.errors import ExodusInvalidEntityError
from exodusii.core.errors import ExodusInvalidModeError
from exodusii.core.errors import ExodusInvalidTimeError
from exodusii.core.errors import ExodusIOError
from exodusii.core.errors import ExodusLookupError
from exodusii.core.errors import ExodusReadError
from exodusii.core.errors import ExodusUnsupportedFeatureError
from exodusii.core.errors import ExodusVariableError
from exodusii.core.errors import ExodusWriteError


def test_base_error_is_exception() -> None:
    assert issubclass(ExodusError, Exception)


@pytest.mark.parametrize(
    "error_type",
    [
        ExodusConsistencyError,
        ExodusDimensionError,
        ExodusIOError,
        ExodusInvalidEntityError,
        ExodusInvalidModeError,
        ExodusInvalidTimeError,
        ExodusLookupError,
        ExodusReadError,
        ExodusUnsupportedFeatureError,
        ExodusVariableError,
        ExodusWriteError,
    ],
)
def test_all_errors_are_exodus_errors(error_type: type[BaseException]) -> None:
    assert issubclass(error_type, ExodusError)


def test_io_error_hierarchy() -> None:
    assert issubclass(ExodusReadError, ExodusIOError)
    assert issubclass(ExodusWriteError, ExodusIOError)


def test_lookup_error_hierarchy() -> None:
    assert issubclass(ExodusLookupError, LookupError)
    assert issubclass(ExodusDimensionError, ExodusLookupError)
    assert issubclass(ExodusVariableError, ExodusLookupError)


def test_value_error_hierarchy() -> None:
    assert issubclass(ExodusInvalidEntityError, ValueError)
    assert issubclass(ExodusInvalidModeError, ValueError)
    assert issubclass(ExodusInvalidTimeError, ValueError)


def test_unsupported_feature_hierarchy() -> None:
    assert issubclass(ExodusUnsupportedFeatureError, NotImplementedError)


def test_error_message_round_trips() -> None:
    error = ExodusLookupError("node set 10 was not found")
    assert str(error) == "node set 10 was not found"


def test_can_catch_as_exodus_error() -> None:
    with pytest.raises(ExodusError):
        raise ExodusVariableError("missing vals_nod_var1")
