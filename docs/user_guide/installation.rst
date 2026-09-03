Installation
============

Requirements
------------

* Python **≥ 3.13**
* `numpy <https://numpy.org>`_ **≥ 2.0**
* `netCDF4 <https://unidata.github.io/netcdf4-python>`_ **≥ 1.7**

Stable release
--------------

Install from PyPI::

    pip install exodusii

Development install
-------------------

Clone the repository and install in editable mode::

    git clone https://github.com/sandialabs/exodusii.git
    cd exodusii
    pip install -e ".[dev]"

The ``dev`` extra pulls in ``pytest`` and ``ruff``.

Running the tests
-----------------

After installing in editable mode::

    pytest

All tests should pass.  Three geometry tests for the ``Tri3`` element are
known to fail on NumPy < 2.0 due to a ``np.cross`` API change; these are
fixed in the current source.

Optional: building the documentation
-------------------------------------

Install the documentation dependencies::

    pip install sphinx numpydoc furo sphinx-copybutton

Then from the ``docs/`` directory::

    make html

Open ``docs/_build/html/index.html`` in a browser.
