import os
import pytest


@pytest.fixture(scope="function")
def datadir():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    yield os.path.join(test_dir, "data")
