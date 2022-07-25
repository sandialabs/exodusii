import numpy as np
from functools import wraps
from .util import stringify, string_kinds
from .config import config


if not config.use_netcdf4_if_possible:
    _netcdf4 = False
    from .netcdf import netcdf_file as Dataset
else:
    try:
        _netcdf4 = True
        from netCDF4 import Dataset
    except (ImportError, ValueError):
        _netcdf4 = False
        from .netcdf import netcdf_file as Dataset

if _netcdf4:
    import warnings

    # netCDF4 uses numpy.tostring, which is deprecated
    warnings.filterwarnings("ignore")


def library():
    return "netcdf4" if _netcdf4 else "netcdf"


def open(filename, mode="r"):
    if _netcdf4:
        try:
            fh = Dataset(filename, mode=mode, format="NETCDF4_CLASSIC")
        except (OSError, TypeError):
            fh = Dataset(filename, mode=mode, format="NETCDF3_64BIT_OFFSET")
    else:
        fh = Dataset(filename, mode=mode)
    return fh


def close(*files):
    for file in files:
        try:
            file.close()
        except Exception:
            pass


def filename(fh):
    return fh.filepath() if _netcdf4 else fh.filename


def cache(fun):
    sentinel = object()
    global_cache = dict()
    cache_get = global_cache.get

    def make_key(fh, name, **kwds):
        s = ", ".join(f"{k}={v}" for (k, v) in kwds.items())
        return f"{filename(fh)}::{fun.__name__}::{name}({s})"

    @wraps(fun)
    def wrapper(fh, name, **kwds):
        key = make_key(fh, name, **kwds)
        val = cache_get(key, sentinel)
        if val is not sentinel:
            return val
        val = fun(fh, name, **kwds)
        global_cache[key] = val
        return val

    return wrapper


def get_variable(fh, name, default=None, raw=False):
    if name not in fh.variables:
        return default
    var = fh.variables[name]
    if raw:
        return var
    if _netcdf4:
        val = var[:].data
    else:
        val = var.data
    if isinstance(val, np.ndarray) and val.dtype.kind in string_kinds:
        val = stringify(val)
    elif isinstance(val, bytes):
        val = stringify(val)
    return val


def get_dimension(fh, name, default=None):
    if name not in fh.dimensions:
        return default
    x = fh.dimensions[name]
    dim = x if not _netcdf4 else x.size
    if dim is None and name != "time_step":
        return 0
    return int(dim)


def setncattr(fh, variable, name, value):
    if _netcdf4:
        fh.variables[variable].setncattr(name, value)
    else:
        setattr(fh.variables[variable], name, value)


def create_dimension(fh, name, value):
    fh.createDimension(name, value)


def create_variable(fh, id, type, shape):
    kind = {str: "c", int: "i", float: "f"}[type]
    fh.createVariable(id, kind, shape)


def fill_variable(fh, name, *args):
    value = args[-1]
    if len(args) == 1:
        fh.variables[name][:] = value
    elif len(args) == 2:
        i = args[0]
        fh.variables[name][i, :] = value
    elif len(args) == 3:
        i, j = args[0:2]
        fh.variables[name][i, j, :] = value
    elif len(args) == 4:
        i, j, k = args[0:3]
        fh.variables[name][i, j, k, :] = value
    else:
        raise ValueError("Unknown fill shape")
