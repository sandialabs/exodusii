# exodusii

A Python interface for reading, writing, comparing, querying, and spatially
reducing [Exodus II](https://sandialabs.github.io/seacas-docs/sphinx/html/index.html)
finite-element databases.

Requires Python ≥ 3.13 and `netCDF4`.

---

## Installation

```bash
python -m pip install -e .
```

Run tests:

```bash
cd /path/to/exodusii
pytest tests/api/ tests/mesh/ tests/core/ tests/cli/ tests/io/ tests/compat/ tests/test_attributes.py
```

---

## Reading a database

```python
from exodusii import ExodusFile

with ExodusFile.open("results.exo") as exo:
    print(exo.title)
    print(exo.dimension)  # 1, 2, or 3
    print(exo.node_count)
    print(exo.element_count)
    print(exo.times())

    coords = exo.coordinates()  # (n_nodes, dim)
    temp = exo.values("TEMP", on="node", time="last")
    energy = exo.values("ENERGY", on="element", block_id=1, time="last")
    ke = exo.values("KE", on="global")  # full time history

    print(exo.element_block_ids())
    print(exo.variable_names("node"))
    print(exo.variable_names("element"))
```

Time selectors: `"first"`, `"last"`, an int (0-based index), or a float
(nearest physical time).

---

## Writing a database

```python
import numpy as np
from exodusii import ExodusWriter

coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

with ExodusWriter.create("out.exo") as w:
    # Declare all counts upfront — this cannot change after initialize()
    w.initialize("my mesh", dimension=2, node_count=4, element_count=1, element_blocks=1)
    w.write_coordinates(coords)
    w.define_element_block(10, "quad", [[1, 2, 3, 4]])

    w.define_global_variables(["TOTAL_ENERGY"])
    w.define_node_variables(["TEMP"])
    w.define_element_variables(["STRESS"])

    w.write_time(0.0)
    w.write_global_values([0.0])
    w.write_node_values("TEMP", [100.0, 200.0, 300.0, 400.0])
    w.write_element_values("STRESS", [42.0], block_id=10)
```

---

## Parallel / decomposed Exodus files

```python
from exodusii import ParallelExodusFile

with ParallelExodusFile.open("mesh.e.4.0", "mesh.e.4.1", "mesh.e.4.2", "mesh.e.4.3") as exo:
    print(exo.node_count)  # global node count
    coords = exo.coordinates()  # assembled from all pieces
    temp = exo.values("TEMP", on="node", time="last")

# Join all pieces into a single serial file
with ParallelExodusFile.open(*files) as exo:
    exo.write("joined.exo")
```

Aggregation uses the per-file `node_num_map` / `elem_num_map`.  If a map is
absent a `UserWarning` is emitted and a sequential fallback is used (only
correct for non-overlapping partitions).

### Fast global reads from decomposed files

Global scalar variables are fully present in every component file.  Use the
`--piece` CLI flag to read one small piece directly and avoid opening the full
joined (potentially multi-GB) database:

```bash
python -m exodusii stats  run.exo.96.00 --select g/MAT_MASS_1 --piece 0
python -m exodusii query  run.exo.96.00 --select g/MASSDELETED --time last --piece 0
```

---

## Region-masked statistics and mass

Compute statistics of a field variable restricted to a geometric region, with
an optional field-threshold predicate:

```python
from exodusii import ExodusFile
from exodusii.mesh import Cylinder

with ExodusFile.open("run.exo") as exo:
    cyl = Cylinder([0, 0, 0], [0.05, 0, 0], radius=0.013)

    # Field statistics inside a cylinder, filtered by a field predicate
    r = exo.region_stats(
        "YIELD_STRESS_2",
        block_id=3,
        region=cyl,
        where="EQPS_2 > 1.0",  # VARNAME OP VALUE only
        reduce=["mean", "max", "count"],
        time="last",
        symmetry_factor=4.0,  # applied to sum/count; NOT mean/max
    )
    print(r.stats["mean"], r.count_selected)

    # Material mass inside a region
    result = exo.region_mass(
        block_id=3,
        region=cyl,
        density_name="DENSITY",
        volfrac_name="VOLFRC_2",  # optional volume fraction
        time="last",
        symmetry_factor=4.0,
    )
    print(result.mass)  # = 4 * sum(|vol| * density * volfrac)
```

Available reducers: `mean`, `max`, `min`, `sum`, `count`, `std`.
Symmetry factor scales only the **extensive** reducers (`sum`, `count`,
`mass`).

CLI:

```bash
python -m exodusii region-stats run.exo \
  --select e/YIELD_STRESS_2 \
  --cylinder 0 0 0 0.05 0 0 0.013 \
  --where "EQPS_2 > 1.0" \
  --reduce mean,max,count \
  --time last --block 3 --symmetry 4
```

Region flags: `--cylinder`, `--sphere`, `--circle`, `--rectangle`.

---

## Exodiff-style comparison

```python
from exodusii import diff, DiffOptions, Tolerance, ToleranceMode

result = diff("gold.exo", "test.exo")
if not result:
    for vd in result.variable_diffs:
        print(vd.entity, vd.name, vd.max_delta)

# Custom tolerances
opts = DiffOptions(
    default_tolerance=Tolerance(ToleranceMode.ABSOLUTE, 1e-8),
    variable_tolerances={"PRESSURE": Tolerance(ToleranceMode.COMBINED, 1e-5)},
    exclude=frozenset({"TIME"}),
)
result = diff("gold.exo", "test.exo", opts)

# Coordinate-based mesh matching (different node/element ordering)
opts = DiffOptions(coordinate_matching=True, matching_tolerance=1e-8)
result = diff("gold.exo", "reordered.exo", opts)
```

Tolerance modes mirror SEACAS `exodiff`: `relative`, `absolute`, `combined`,
`ignore`, `eigenrel`, `eigenabs`, `eigencom`, `ulps_float`, `ulps_double`.

`TimeSelection` controls which steps are compared and whether file-2 values
are interpolated to file-1 time points.

```bash
exodiff gold.exo test.exo
exodiff --absolute -t 1e-8 gold.exo test.exo
exodiff --start LAST gold.exo test.exo
exodiff --match-coordinates gold.exo reordered.exo
exodiff --format json --terse gold.exo test.exo
```

Exit codes: `0` same, `1` error, `2` different.

---

## Query and stats CLI

```bash
# Inspect a database
python -m exodusii inspect  mesh.exo
python -m exodusii variables mesh.exo
python -m exodusii blocks   mesh.exo
python -m exodusii times    mesh.exo

# Query result variables (JSON output)
python -m exodusii query mesh.exo --select g/TOTAL_ENERGY
python -m exodusii query mesh.exo --select n/TEMP --time last --limit 10
python -m exodusii query mesh.exo --select e/ENERGY --time step:5

# Compact numeric summaries
python -m exodusii stats mesh.exo --select n/TEMP --time last
python -m exodusii stats mesh.exo --select e/ENERGY --time last --by-block
```

Variable selector format: `ENTITY/NAME` — e.g. `g/TOTAL_ENERGY`, `n/TEMP`,
`e/ENERGY`, `ns/FLUX`, `ss/PRESSURE`.

---

## Python query API

```python
from exodusii import ExodusFile, query, print_query

with ExodusFile.open("mesh.exo") as exo:
    result = query(exo, "n/TEMP", "e/ENERGY", time="last")
    print(result.names)
    print(result.data[:5])
    print_query(exo, "g/TM_STEP")
```

---

## Mesh geometry helpers

```python
from exodusii.mesh import entity_centers, element_volumes, bounding_box
from exodusii.mesh import Cylinder, Sphere, Circle, Rectangle

with ExodusFile.open("run.exo") as exo:
    coords = exo.coordinates()
    conn = exo.element_connectivity(block_id=1, zero_based=True)

# Element centroids — feed to region.contains() to select elements
centers = entity_centers(conn, coords)

# Per-element volumes/areas
vols = element_volumes("hex8", conn, coords)

# Bounding box
lo, hi = bounding_box(coords)

# Region predicates
cyl = Cylinder([0, 0, 0], [0.05, 0, 0], radius=0.013)
mask = cyl.contains(centers)  # boolean ndarray
```

Supported element types: `quad4`, `hex8`, `tri3`, `tet4`, `wedge6`.

---

## Agent CLI (self-learning)

```bash
# Static capability reference (no file required)
python -m exodusii learn capabilities overview
python -m exodusii learn capabilities region_reduce
python -m exodusii learn capabilities python_api.writer

# Skills (self-contained workflow guides)
python -m exodusii learn skills list
python -m exodusii learn skills exodusii-region-reduce
python -m exodusii learn skills exodusii-geometry
```

---

## Package layout

```
exodusii/
├── api/             modern API (ExodusFile, ExodusWriter, ParallelExodusFile,
│                    diff, query, region_reduce, lineout, copy)
├── core/            domain model (Entity, Block, Tolerance, errors, schema)
├── io/              NetCDF backend (Protocol + netCDF4 implementation)
├── mesh/            geometry (elements, regions, entity_centers, element_volumes)
├── compat/          legacy ExodusIIFile / put_*/get_* API
├── cli/             JSON-oriented CLI (agent.py, exodiff, exoread, learn)
└── data/            capabilities.json, skills.json
```

---

## Legacy API (compatibility)

The historical `put_*` / `get_*` API from the original exodusii library is
preserved for downstream code that has not yet migrated:

```python
import exodusii

# Read
with exodusii.File("mesh.exo") as exo:
    times = exo.get_times()
    temp = exo.get_node_variable_values("TEMP", time_step=-1)

# Write
with exodusii.File("out.exo", mode="w") as exo:
    exo.put_init("title", 2, 4, 1, 1, 0, 0)
    exo.put_coords(coords)
    exo.put_element_block(10, "quad", 1, 4)
    exo.put_element_conn(10, [[1, 2, 3, 4]])

# Globals only
exodusii.write_globals({"energy": values}, times, filename="globals.exo")
```

New code should use the modern API (`ExodusFile`, `ExodusWriter`,
`ParallelExodusFile`).  The compatibility layer is fully tested and maintained
but will not receive new features.
