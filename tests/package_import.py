# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import exodusii


def test_package_imports() -> None:
    assert isinstance(exodusii.__version__, str)
    assert exodusii.__version__


def test_public_exports_include_version() -> None:
    assert "__version__" in exodusii.__all__
