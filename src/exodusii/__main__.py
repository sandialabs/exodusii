# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Agent-oriented command-line interface for exodusii.

Usage examples
--------------
python -m exodusii inspect mesh.exo
python -m exodusii variables mesh.exo
python -m exodusii query mesh.exo --select n/TEMP --time last
python -m exodusii stats mesh.exo --select e/ENERGY --time last
"""

from exodusii.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
