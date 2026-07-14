#!/usr/bin/env python
import glob
import os

import exodusii
from exodusii.util import working_dir


def test_exodusii_parallel_write(tmpdir, datadir):
    with working_dir(tmpdir):
        name = "edges"
        files = glob.glob(f"{datadir}/{name}.exo.*")
        file = exodusii.exo_file(*files)
        joined = file.write(f"{name}.exo")
        basefile = os.path.join(datadir, f"{name}.base.exo")
        assert os.path.exists(basefile)
        base = exodusii.exo_file(basefile)
        dimensions = "~four|len_line|len_string"
        result = exodusii.allclose(base, joined, dimensions=dimensions, variables=None, result=True)
        print(result.equal)
        print("\n".join(result.errors))
        assert result.equal
