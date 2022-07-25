import os
import re
import subprocess
import numpy as np
from contextlib import contextmanager


string_kinds = ("U", "S")
string_types = (str,)


def stringify(a):
    if isinstance(a, str):
        return a
    elif isinstance(a, bytes):
        return a.decode()
    elif isinstance(a, np.ndarray):
        if len(a.shape) == 1:
            return "".join(decode(x) for x in a if x).rstrip()
        elif len(a.shape) == 2:
            return np.array([stringify(row) for row in a])
        elif len(a.shape) == 3:
            x = [stringify(row) for row in a]
            return np.array([" ".join(_) for _ in x])
        else:
            raise TypeError(f"Cannot stringify arrays with shape {a.shape}")
    else:
        raise TypeError(f"Cannot stringify items of type {type(a).__name__}")


def decode(x):
    if isinstance(x, np.ma.core.MaskedConstant):
        return ""
    return x.decode()


def index(array, val):
    if isinstance(array, (list, tuple)):
        return array.index(val)
    (ix,) = np.where(array == val)
    if not len(ix):
        raise ValueError(f"{val} is not in array")
    return ix[0]


def is_exe(path):
    return os.path.isfile(path) and os.access(path, os.X_OK)


def which(*args):
    """Like ``which`` on the command line"""
    path = os.getenv("PATH").split(os.pathsep)
    for name in args:
        exe = os.path.abspath(name)
        if is_exe(exe):
            return exe
        for directory in path:
            exe = os.path.join(directory, name)
            if is_exe(exe):
                return exe
    raise ValueError(
        f"Required executable {args[0]} not found. Make sure it is in your path"
    )


@contextmanager
def working_dir(dirname):
    cwd = os.getcwd()
    os.chdir(dirname)
    yield
    os.chdir(cwd)


def epu(*files):
    """Concatenate exodus files"""
    if not files:
        return None
    if len(files) == 1:
        return files[0]

    epu = which("epu")
    workdir = os.path.dirname(files[0])
    if not os.path.isdir(workdir):
        raise ValueError(f"{workdir} is not a directory")
    with working_dir(workdir):
        files = [os.path.basename(f) for f in files]
        for file in files:
            if not os.path.exists(file):
                raise ValueError(f"{workdir} is not a file")
        # determine the basename
        parts = files[0].split(".")
        try:
            base, suff, p, n = parts
        except ValueError:
            raise ValueError("Expected files `base.suf.#p.#n`") from None
        cmd = [epu, "-auto", files[0]]
        f = ".epu.log"
        with open(f, "w") as fh:
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            p.wait()
        if p.returncode != 0:
            os.rename(f, f[1:])
            raise SystemExit(f"Exodus file concatenation failed, see {f[1:]}")
        else:
            os.remove(f)

    os.rename(os.path.join(workdir, f"{base}.{suff}"), f"{base}.{suff}")
    return f"{base}.{suff}"


def compute_connected_average(conn, values):
    """Computes the average of values

    Parameters
    ----------
    conn : ndarray of int
        Entity connectivity, 0 based
    values : ndarray of float
        Nodal values

    Returns
    -------
    averaged : ndarray of float

    """
    num_ent = len(conn)
    num_dimensions = values.shape[1]
    averaged = np.zeros((num_ent, num_dimensions))
    for (e, ix) in enumerate(conn):
        for dim in range(num_dimensions):
            x = values[ix, dim]
            averaged[e, dim] = x.sum() / conn.shape[1]
    return averaged


def streamify(file):
    if isinstance(file, string_types):
        return open(file, "w"), True
    else:
        return file, False


def fmt_join(*, fmt, items, sep=" "):
    return sep.join(fmt % item for item in items)


def fuzzy_compare(arg1, arg2):
    """Compares arg1 to arg2. The comparison is case insensitive and spaces, '-', and
    '_' are stripped from both ends. Treats '-', '_' as a space. Multiple spaces
    treated as one space.
    """
    regex = re.compile(r"[-_]")
    transform = lambda s: " ".join(regex.sub(" ", s).split()).lower()
    return transform(arg1) == transform(arg2)


def find_index(sequence, arg, strict=True):
    """Searches for 'aname' in the list of names 'nameL' using the name_compare()
    function.  Returns the index into the 'nameL' list if found, or None if not.
    """
    for (i, item) in enumerate(sequence):
        if strict:
            if arg == item:
                return i
        elif fuzzy_compare(arg, item):
            return i
    return None


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def check_bounds(array, value, tol=1e-12):
    if value + tol <= np.amin(array):
        return False
    elif value - tol >= np.amax(array):
        return False
    return True


def contains(array, value):
    return value in list(array)
