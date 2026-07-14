# exodusii

A Python interface for reading, writing, comparing, querying, and joining
[Exodus II](https://sandialabs.github.io/seacas-docs/sphinx/html/index.html)
finite-element databases.

This package provides:

- a modern API built around `ExodusFile`, `ExodusWriter`, and `ParallelExodusFile`
- compatibility with the historical `exodusii.File` / `ExodusIIFile` interface
- serial Exodus read/write support
- decomposed/parallel Exodus aggregation and serial join support
- Exodus global-ID-map-aware parallel behavior
- mesh geometry helpers
- region predicates
- lineout/query utilities
- comparison utilities
- a required `netCDF4` backend

---

## Installation

Install in editable mode for development:

```bash
python -m pip install -e .
```

Run tests:

```bash
pytest
```

Run pre-commit checks, if available in your checkout:

```bash
./bin/pre-commit
```

---

## Dependencies

The refreshed implementation uses the `netCDF4` Python package as its NetCDF
backend.

The old bundled pure-Python NetCDF fallback is no longer the primary runtime
path.

---

## Quick start: legacy-compatible API

The historical API is still available:

```python
import exodusii

with exodusii.File("mesh.exo") as exo:
    print(exo.title())
    print(exo.num_dimensions())
    print(exo.num_nodes())
    print(exo.num_elems())

    print(exo.get_element_block_ids())
    print(exo.get_node_set_ids())
    print(exo.get_node_variable_names())

    times = exo.get_times()
    temp_last = exo.get_node_variable_values("TEMP", time_step=-1)
```

Write a simple mesh:

```python
import numpy as np
import exodusii

coords = np.asarray(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ]
)

with exodusii.File("square.exo", mode="w") as exo:
    exo.put_init("square", 2, 4, 1, 1, 0, 0)
    exo.put_coords(coords)
    exo.put_element_block(10, "quad", 1, 4)
    exo.put_element_conn(10, [[1, 2, 3, 4]])
```

Write globals only:

```python
import numpy as np
import exodusii

times = np.asarray([0.0, 1.0, 2.0])
data = {
    "energy": np.asarray([10.0, 11.0, 12.0]),
    "mass": np.asarray([20.0, 20.0, 20.0]),
}

exodusii.write_globals(data, times, title="global history", filename="globals.exo")
```

---

## Modern API

The modern API uses properties and explicit entity names:

```python
from exodusii import ExodusFile

with ExodusFile.open("mesh.exo") as exo:
    print(exo.title)
    print(exo.dimension)
    print(exo.node_count)
    print(exo.element_count)

    print(exo.element_block_ids())
    print(exo.node_set_ids())
    print(exo.variable_names("node"))

    coords = exo.coordinates()
    temp = exo.values("TEMP", on="node", time="last")
```

Write a file with the modern writer:

```python
from exodusii import ExodusWriter

with ExodusWriter.create("square.exo") as writer:
    writer.initialize("square", 2, 4, 1, element_blocks=1)
    writer.write_coordinates(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    writer.define_element_block(10, "quad", [[1, 2, 3, 4]], name="block_10")

    writer.define_node_variables(["TEMP"])
    writer.write_time(0.0)
    writer.write_node_values("TEMP", [10.0, 20.0, 30.0, 40.0])
```

---

## Parallel/decomposed Exodus files

Open multiple decomposed Exodus files:

```python
import glob
import exodusii

files = glob.glob("mesh.exo.*.*")

with exodusii.exo_file(*files) as exo:
    print(exo.num_nodes())
    print(exo.num_elems())
    print(exo.get_node_set_ids())

    coords = exo.get_coords()
```

Join decomposed files into one serial Exodus file:

```python
import glob
import exodusii

files = glob.glob("mesh.exo.*.*")
exo = exodusii.exo_file(*files)

joined = exo.write("mesh_joined.exo")
print(joined)
```

Parallel aggregation respects Exodus global ID maps where available:

- `node_num_map`
- `elem_num_map`
- `edge_num_map`
- `face_num_map`

User-facing set APIs return global labels. For example:

```python
nodes = exo.get_node_set_nodes(100)   # global node labels
elems = exo.get_side_set_elems(200)   # global element labels
```

Connectivity arrays are returned as 1-based logical indices suitable for
indexing the merged coordinate array. Where supported, label-oriented access may
be available through lower-level modern methods.

---

## Comparison utilities

Check two files for raw data-wise equality within tolerances:

```python
import exodusii

same = exodusii.allclose("one.exo", "two.exo")
```

Skip variables or selected dimensions:

```python
same_mesh_dimensions = exodusii.allclose(
    "one.exo",
    "two.exo",
    variables=None,
    dimensions="~four|len_line|len_string",
)
```

Get a detailed comparison result:

```python
result = exodusii.allclose("one.exo", "two.exo", result=True)
if not result:
    print("\n".join(result.errors))
```

Check whether two files have similar mesh/layout while ignoring solution values:

```python
exodusii.similar("baseline.exo", "candidate.exo")
```

---

## Query and printing

Query variables into a structured NumPy array:

```python
from exodusii import ExodusFile, query

with ExodusFile.open("mesh.exo") as exo:
    result = query(exo, "n/TEMP", time="last", object_index=True)

print(result.names)
print(result.data)
```

Print a table:

```python
from exodusii import ExodusFile, print_query

with ExodusFile.open("mesh.exo") as exo:
    print_query(exo, "g/TM_STEP", time="last")
```

Supported selector prefixes include:

- `g/NAME` for global variables
- `n/NAME` for nodal variables
- `e/NAME` for element variables
- `d/NAME` for edge variables
- `f/NAME` for face variables

Special nodal selectors include:

- `n/coordinates`
- `n/displacements`

---

## Lineout

Restrict tabular spatial output to a coordinate line:

```python
from exodusii import ExodusFile, Lineout, query

with ExodusFile.open("mesh.exo") as exo:
    result = query(
        exo,
        "n/coordinates",
        "n/TEMP",
        time="last",
        lineout=Lineout(x="x", y=0.0, tol=1.0e-12),
    )
```

CLI-style parsing is also available:

```python
line = Lineout.from_cli("x/0.0/T1e-12")
```

Uppercase coordinate selectors request displaced coordinates:

```python
Lineout.from_cli("X/0.0/T1e-12")
```

---

## Regions

Geometric region predicates are available through both modern and legacy imports:

```python
import exodusii

region = exodusii.region.circle([0.0, 0.0], 1.0)
assert region.contains([0.0, 0.0])
```

Available regions include:

- `circle`
- `sphere`
- `rectangle`
- `quad`
- `cylinder`

Time-domain predicates are also available:

- `unbounded_time_domain`
- `bounded_time_domain`
- `bound_time_domain`

---

## Mesh geometry helpers

Element geometry classes and mesh utilities live under `exodusii.mesh`:

```python
from exodusii.mesh import Quad4, element_volumes

quad = Quad4(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ]
)

print(quad.center)
print(quad.volume)
```

Supported element geometry includes:

- `Quad4`
- `Hex8`
- `Tri3`
- `Tet4`
- `Wedge6`

---

## Compatibility modules

The following historical modules are provided as compatibility shims:

- `exodusii.file`
- `exodusii.parallel_file`
- `exodusii.allclose`
- `exodusii.similar`
- `exodusii.lineout`
- `exodusii.region`
- `exodusii.element`
- `exodusii.extension`
- `exodusii.exodus_h`
- `exodusii.util`

`exodusii.exodus_h` is retained for downstream code that imports Exodus-style
constant names or enum values.

---

## CLI

If installed with console scripts enabled, `exoread` can describe or extract
values from an Exodus file:

```bash
exoread mesh.exo
exoread -g TM_STEP mesh.exo
exoread -n TEMP --index -1 mesh.exo
```

Use lineout:

```bash
exoread -n coordinates -L 'x/0.0/T1e-12' mesh.exo
```

---

## Development notes

This package intentionally separates:

- modern API implementation in `exodusii.api`
- compatibility wrappers in `exodusii.compat`
- core naming/entity/schema utilities in `exodusii.core`
- mesh geometry and regions in `exodusii.mesh`
- NetCDF backend code in `exodusii.io`

The compatibility layer is broad and is tested against historical serial and
parallel workflows, including:

- edge and face blocks
- all set families
- variables and truth tables
- properties
- block attributes
- QA and info records
- global ID maps
- parallel global metadata
- decomposed-file serial joining
