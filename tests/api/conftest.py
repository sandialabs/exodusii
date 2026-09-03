# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""pytest configuration for the API test suite.

Suppress the parallel sequential-fallback warning for test fixtures that
deliberately write component files without id maps (non-overlapping
partitions where the fallback is correct).  The warning is still raised
in production code; this suppression only applies during testing.
"""

import warnings

import pytest


@pytest.fixture(autouse=True)
def _suppress_parallel_map_fallback_warning():
    """Suppress the sequential-fallback UserWarning in parallel tests."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*sequential fallback.*",
            category=UserWarning,
        )
        yield
