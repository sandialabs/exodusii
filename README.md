# ExodusII

A pure python implementation of the [Exodus finite element database
model](https://gsjaardema.github.io/seacas-docs/sphinx/html/index.html).  The
API strives to be compatible with the API provided by
[`exodus.py`](https://gsjaardema.github.io/seacas-docs/exodus.html) built as
part of SEACAS.  The main advantage is that this implementation does not require
building SEACAS.

## Dependencies

Exodus files are written in the netCDF file format.  netCDF files are read in
directly using a netCDF reader copied from `scipy.io`.  Exodus files can
optionally be written using netCDF file format version 4.  These files require
the [`netcdf4`](https://unidata.github.io/netcdf4-python/) python module be
installed.

## Install

```
python -m pip install .
```

## Copyright

Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

SCR# 2748
