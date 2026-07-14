# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest


@pytest.fixture(scope="function")
def datadir() -> Path:
    return Path(__file__).resolve().parent / "data"
