import sys
import numpy as np
from io import StringIO
from .util import string_kinds


def allclose(
    file1,
    file2,
    atol=1.0e-12,
    rtol=1.0e-12,
    dimensions=True,
    variables=True,
    verbose=False,
):
    """Returns True if two files are data-wise equal within a tolerance."""

    from .file import ExodusIIFile

    if not isinstance(file1, ExodusIIFile):
        file1 = ExodusIIFile(file1)
    if not isinstance(file2, ExodusIIFile):
        file2 = ExodusIIFile(file2)

    ach = allclose_helper(file1, file2, atol, rtol, verbose=verbose)

    ach.set_dimensions(dimensions)
    ach.set_variables(variables)

    ach.compare_dimensions()
    ach.compare_variables()

    return True if not ach.errors else False


def _allclose(a, b, atol=1.0e-08, rtol=1.0e-05):
    """Returns True if two arrays are element - wise equal within a tolerance."""

    def compatible_dtypes(x, y):
        if x.dtype.kind in string_kinds and y.dtype.kind not in string_kinds:
            return False
        if y.dtype.kind in string_kinds and x.dtype.kind not in string_kinds:
            return False
        return True

    def compatible_shape(x, y):
        return x.shape == y.shape

    if a is None or b is None:
        if a != b:
            raise ValueError(
                f"cannot compare type {type(a).__name__} to type {type(b).__name__}"
            )
        return True

    x = np.asanyarray(a)
    y = np.asanyarray(b)

    if not compatible_dtypes(x, y):
        raise TypeError(
            f"adiff not supported for input dtypes {x.dtype.kind} and {y.dtype.kind}"
        )
    elif not compatible_shape(x, y):
        raise ValueError("input arguments must have same shape")

    if x.dtype.kind in string_kinds:
        x = sorted(x.flatten())
        y = sorted(y.flatten())
        return all([x[i] == y[i] for i in range(len(x))])

    return np.allclose(x, y, atol=atol, rtol=rtol)


class allclose_helper:
    def __init__(self, file1, file2, atol, rtol, print_threshold=10, verbose=False):
        self.file1 = file1
        self.file2 = file2
        self.atol = atol
        self.rtol = rtol
        self.print_threshold = print_threshold
        self.verbose = verbose

        self._dims_to_compare = None
        self._vars_to_compare = None

        self.errors = 0

    def log_error(self, message, end="\n"):
        if self.verbose:
            sys.stderr.write(f"==> Error: {message}{end}")
        self.errors += 1

    def all_dimensions(self):
        dims1 = self.file1.dimension_names()
        dims2 = self.file2.dimension_names()
        return sorted(set(dims1 + dims2))

    def all_variables(self):
        vars1 = self.file1.variable_names()
        vars2 = self.file2.variable_names()
        return sorted(set(vars1 + vars2))

    def set_dimensions(self, dimensions):
        if dimensions is True:
            dimensions = self.all_dimensions()
        elif dimensions is None or dimensions is False:
            dimensions = []
        elif isinstance(dimensions, str):
            dimensions = [dimensions]
            if dimensions[0].startswith("~"):
                # Compare all - except those being negated
                skip = dimensions[0][1:].split("|")
                dimensions = [_ for _ in self.all_dimensions() if _ not in skip]
        if not isinstance(dimensions, (list, tuple)):
            raise ValueError("Expected list of dimensions to compare")
        self.validate_dimensions(dimensions)
        self._dims_to_compare = tuple(dimensions)

    def set_variables(self, variables):
        if variables is True:
            variables = self.all_variables()
        elif variables is None or variables is False:
            variables = []
        elif isinstance(variables, str):
            variables = [variables]
            if variables[0].startswith("~"):
                # Compare all - except those being negated
                skip = variables[0][1:].split("|")
                variables = [_ for _ in self.all_variables() if _ not in skip]
        if not isinstance(variables, (list, tuple)):
            raise ValueError("Expected list of variables to compare")
        self.validate_variables(variables)
        self._vars_to_compare = tuple(variables)

    def validate_dimensions(self, dimensions):
        invalid = 0
        valid_dimensions = self.all_dimensions()
        for dimension in dimensions:
            if dimension not in valid_dimensions:
                self.log_error(f"{dimension} is not a valid dimension")
                invalid += 1
        if invalid:
            raise ValueError("One or more invalid dimensions")

    def validate_variables(self, variables):
        invalid = 0
        valid_variables = self.all_variables()
        for variable in variables:
            if variable not in valid_variables:
                self.log_error(f"{variable} is not a valid variable")
                invalid += 1
        if invalid:
            raise ValueError("One or more invalid variables")

    def compare_dimensions(self):
        if self._dims_to_compare is None:
            raise ValueError("Dimensions to compare must first be set")
        for dimension in self._dims_to_compare:
            self.compare_dimension(dimension)
        self._dims_to_compare = None

    def compare_dimension(self, dim):
        if dim not in self.file1.fh.dimensions:
            self.log_error(f"dimension {dim} not found in {self.file1.filename}")
            return
        elif dim not in self.file2.fh.dimensions:
            self.log_error(f"dimension {dim} not found in {self.file2.filename}")
            return
        elif dim in ("time_step",):
            return

        dim1 = self.file1.get_dimension(dim)
        dim2 = self.file1.get_dimension(dim)
        if not _allclose(dim1, dim2, atol=self.atol, rtol=self.rtol):
            err = StringIO()
            err.write(
                f"{self.file1.filename}::{dim} != "
                f"{self.file2.filename}::{dim} ({dim1} != {dim2})"
            )
            self.log_error(err.getvalue())

    def compare_variables(self):
        if self._vars_to_compare is None:
            raise ValueError("Variables to compare must first be set")
        for variable in self._vars_to_compare:
            self.compare_variable(variable)
        self._vars_to_compare = None

    def compare_variable(self, var):
        if var not in self.file1.fh.variables:
            self.log_error(f"variable {var} not found in {self.file1.filename}")
            return
        elif var not in self.file2.fh.variables:
            self.log_error(f"variable {var} not found in {self.file2.filename}")
            return

        var1 = self.file1.get_variable(var)
        var2 = self.file2.get_variable(var)
        if var1.shape != var2.shape:
            err = StringIO()
            err.write(
                f"{self.file1.filename}::{var}.shape != "
                f"{self.file2.filename}::{var}.shape "
                f"({var1.shape} != {var2.shape})"
            )
            self.log_error(err.getvalue())
            return

        if not _allclose(var1, var2, atol=self.atol, rtol=self.rtol):
            err = StringIO()
            s1 = np.array2string(var1, threshold=self.print_threshold)
            s2 = np.array2string(var2, threshold=self.print_threshold)
            err.write(
                f"{self.file1.filename}::{var} != "
                f"{self.file2.filename}::{var} ({s1} != {s2})"
            )
            self.log_error(err.getvalue())
