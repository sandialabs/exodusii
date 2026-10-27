# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""pytest configuration for the API test suite.

Many parallel fixtures deliberately write component files without
``node_num_map``/``elem_num_map`` (non-overlapping partitions, where the
sequential fallback is correct).  Ignore that one warning here so the suite
stays quiet; ``tests/api/test_parallel_id_map_warnings.py`` checks it is still
emitted.  Missing edge/face maps are not warned about at all.
"""

import warnings

import pytest


@pytest.fixture(autouse=True)
def _ignore_missing_node_element_map_warning(request: pytest.FixtureRequest):
    if request.node.get_closest_marker("id_map_warnings"):
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*'(node|elem)_num_map' missing.*", category=UserWarning
        )
        yield
