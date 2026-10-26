# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Guard the installed console-script names.

exodusii < 2 installed the reader CLI as ``exoread.py``.  Downstream test
suites still shell out to that name, so it must remain an entry point that
targets the same callable as ``exoread``.
"""

import importlib
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _scripts() -> dict[str, str]:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]["scripts"]


@pytest.mark.parametrize("name", ["exoread", "exoread.py"])
def test_console_script_declared(name: str) -> None:
    scripts = _scripts()
    assert name in scripts
    assert scripts[name] == "exodusii.cli.exoread:main"


def test_legacy_console_script_matches_modern() -> None:
    scripts = _scripts()
    assert scripts["exoread.py"] == scripts["exoread"]


def test_console_script_target_is_callable() -> None:
    module_name, _, attr = _scripts()["exoread.py"].partition(":")
    target = getattr(importlib.import_module(module_name), attr)
    assert callable(target)
