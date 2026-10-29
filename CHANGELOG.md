## `CHANGELOG.md`

# Changelog

All notable changes to this project are documented here.

This project follows the spirit of [Keep a Changelog](https://keepachangelog.com/)
and uses semantic versioning once versioned releases are established.

---

## Unreleased

### Added

- Added modern API:
  - `ExodusFile`
  - `ExodusWriter`
  - `ParallelExodusFile`
  - `copy`
  - `copy_file`
  - `allclose`
  - `similar`
  - `query`
  - `print_query`
  - `Lineout`
- Added required `netCDF4` backend through `NetCDF4Backend`.
- Added core schema mapping for Exodus entities, blocks, sets, variables, maps,
  properties, and truth tables.
- Added support for serial reading and writing of:
  - coordinates
  - element blocks
  - edge blocks
  - face blocks
  - node sets
  - side sets
  - edge sets
  - face sets
  - element sets
  - global variables
  - node variables
  - element variables
  - edge variables
  - face variables
  - node-set variables
  - side-set variables
  - edge-set variables
  - face-set variables
  - element-set variables
  - variable truth tables
  - object ID maps
  - block and set status arrays
  - properties
  - block attributes and attribute names
  - QA records
  - info records
- Added support for decomposed/parallel Exodus aggregation.
- Added support for serial joining of decomposed Exodus files using
  `ParallelExodusFile.write(...)`.
- Added parallel aggregation that respects Exodus global ID maps:
  - `node_num_map`
  - `elem_num_map`
  - `edge_num_map`
  - `face_num_map`
- Added use and validation of parallel/Nemesis global metadata where present:
  - `num_nodes_global`
  - `num_elems_global`
  - `num_el_blk_global`
  - `num_ns_global`
  - `num_ss_global`
  - `el_blk_cnt_global`
  - `ns_node_cnt_global`
  - `ss_side_cnt_global`
  - global block/set ID arrays
- Added user-facing parallel set behavior using global labels for:
  - node sets
  - side sets
  - edge sets
  - face sets
  - element sets
- Added parallel active/inactive block and set handling.
- Added preservation of local distribution factors for parallel reads while
  respecting global distribution-factor counts during joined-file writes.
- Added block attribute support for element, edge, and face blocks.
- Added property support for block and set entities.
- Added semantic `similar(...)` comparison coverage for:
  - edge/face blocks
  - set families
  - ID maps
  - status arrays
  - truth tables
  - block attributes
- Added legacy compatibility shims for:
  - `exodusii.File`
  - `exodusii.exo_file`
  - `exodusii.ExodusIIFile`
  - `exodusii.exodusii_file`
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
- Added callable-module compatibility for legacy `exodusii.allclose`,
  `exodusii.similar`, and `exodusii.lineout` import patterns.
- Added CLI module and `exoread` command implementation.
- Added mesh geometry primitives:
  - `Quad4`
  - `Hex8`
  - `Tri3`
  - `Tet4`
  - `Wedge6`
- Added mesh geometry utilities:
  - `connected_average`
  - `entity_centers`
  - `element_volumes`
  - `nodal_volumes`
  - `characteristic_element_length`
  - `bounding_box`
- Added region predicates:
  - circle
  - sphere
  - rectangle
  - quad
  - flat-capped cylinder
  - bounded and unbounded time domains
- Added lineout support for structured-array and dense-array tabular data.

### Changed

- Reworked implementation around a required `netCDF4` backend.
- Reworked serial and parallel APIs to share centralized entity schema.
- Changed parallel aggregation to use global Exodus labels for user-facing set
  entries.
- Changed parallel joined-file writing to write variable histories in bulk where
  possible.
- Changed fixed-width character decoding to handle legacy ASCII-zero-padded
  Exodus strings while preserving meaningful trailing zeros such as
  `nodeset_100`.
- Changed cylinder region behavior to use flat end caps, matching legacy tests
  and behavior expectations.
- Changed `len_name` writer dimension default to `256` for compatibility with
  legacy Exodus fixtures.
- Improved comparison behavior:
  - `allclose(...)` can return detailed `ComparisonResult`
  - `similar(...)` now compares a broader semantic mesh/layout surface
- Improved compatibility with `pathlib.Path` inputs.

### Fixed

- Fixed legacy `exodusii.region` access after plain `import exodusii`.
- Fixed legacy parallel `storage_type()` method compatibility using callable
  string behavior.
- Fixed legacy `allclose(...)` accepting `ExodusIIFile` adapter objects.
- Fixed parallel file ordering to match historical sorted filename behavior.
- Fixed parallel side-set ordering with repeated element labels.
- Fixed side-set distribution-factor aggregation for parallel files.
- Fixed joined parallel writes creating unwanted node-set distribution-factor
  dimensions when global metadata indicated zero distribution factors.
- Fixed repeated string-table decoding causing parallel write timeouts.
- Fixed masked-array handling from `netCDF4` by disabling automatic masking and
  adding defensive conversion.
- Fixed copy behavior to avoid materializing default ID maps when source files
  did not explicitly contain them.
- Fixed copy behavior to preserve explicit ID maps.
- Fixed copy behavior to preserve:
  - edge/face blocks
  - extended set families
  - edge/face variables
  - set variables
  - truth tables
  - properties
  - block attributes
  - QA records
  - info records
  - status arrays
- Fixed legacy info-record truncation to match fixed-width Exodus behavior.
- Fixed legacy block attribute name writing so it no longer overwrites existing
  attribute values.
- Fixed writer error reporting for missing block/set locations.
- Fixed name tables losing trailing zeros: names of 16+ characters ending in
  `00` (e.g. `kinetic_energy_100`) were read back truncated, which could make
  a lookup return another variable's data.  Only NULs and trailing whitespace
  are padding (libexodus `exi_trim`).
- Fixed blank variable and attribute names shifting the on-disk index, so
  values after a blank slot were read from the wrong column.
- Fixed `ExodusWriter.write_values(..., on=GLOBAL)` overwriting every global
  variable in the step instead of only the named one.
- Fixed integer IDs/maps above `2**31 - 1` silently wrapping.  Out-of-range
  values now raise `ExodusWriteError`; `ExodusWriter.create(path, int64=True)`
  stores IDs, maps, connectivity and set entries as NC_INT64 and writes the
  `int64_status` attribute.  `ExodusFile.int64` reports the stored width, and
  `copy_file` and parallel joins preserve it.
- Fixed zero-entry blocks and sets crashing the writer (`NC_UNLIMITED size
  already in use`); they are now written as NULL entities (status 0, no
  dimensions), as SEACAS does.
- Fixed node/element/edge-set distribution factors to reuse the entry-count
  dimension (e.g. `dist_fact_ns1(num_nod_ns1)`) and to require one factor per
  entry, matching `ex_put_sets`.
- Fixed `ParallelExodusFile.write()` (join), to match SEACAS epu:
  - side sets and element sets referenced the wrong elements whenever a
    block's element IDs were not one contiguous range;
  - node, element, edge and face ID maps were dropped (IDs became 1..n);
  - element/edge/face sets, block attributes, QA/info records and user
    properties were dropped;
  - the truth table was written as all ones; it is now the OR of the component
    tables;
  - blocks and node sets empty on every processor made `block()`,
    `node_set()` and `write()` fail;
  - node-set distribution factors are always written (missing components
    contribute 1.0), even when Nemesis `ns_df_cnt_global` is zero.  This
    replaces the earlier rule that suppressed them, and matches both epu and
    the serial reference files.
- Fixed legacy `get_element_variable_history` on parallel files returning
  another element's history.
- Fixed `diff` / `exodiff` reporting "Files are the same" where SEACAS exodiff
  reports a difference:
  - time values differing beyond the time tolerance, and differing numbers of
    time steps, are differences (new `DiffResult.differences`; exit status 2);
  - element connectivity, element types (exodiff `close_compare`) and
    per-block element counts are compared (mesh error; exit status 1);
  - nodes and elements are matched by their ID maps by default (exodiff
    `-match_ids`); unmatched ID sets are an error.  `DiffOptions(match_ids=False)`
    / `--match-file-order` matches by position instead;
  - element blocks are matched by ID rather than by file position;
  - any NaN is a difference, even at the same location in both files;
    `DiffOptions(ignore_nans=True)`, `--ignore-nans` and the `IGNORE NANS`
    command-file directive skip them.
- Fixed `--match-coordinates` building inverted element and free-node maps, so
  files with permuted elements (or free nodes) were reported as different.
- Fixed exodiff command files: variable categories that are not listed are
  not compared, and per-variable tolerances and `!NAME` exclusions apply only
  to the category they are listed under (`DiffOptions.categories_specified`,
  `category_variable_tolerances`, `category_exclude`).
- Fixed `copy()` / `copy_file()` crashing on files with gaps in their truth
  tables, writing float32 files as float64, and dropping element-to-edge /
  element-to-face connectivity.
- Fixed QA records being split on whitespace: each of the four fields is
  decoded separately, so fields with spaces or empty fields read back intact.
- Fixed `displacements(time=None)` returning `(n_steps, n_nodes * dim)`; it
  now returns `(n_steps, n_nodes, dim)` (serial and parallel), and
  `coordinates(displaced=True)` without a time uses the last step instead of
  crashing.
- Fixed old-format (`file_size = 0`) files: nodal variables stored in the
  combined `vals_nod_var(time_step, num_nod_var, num_nodes)` array are read.
- Fixed `region_stats(where=...)` with several blocks (broadcast error), and
  `time="all"` with no eligible blocks now returns one empty result per step.
- Fixed `write_time(value, step=k)` on an earlier step: later writes without
  an explicit step now go to step `k`.
- Defining block or set variables before all declared blocks/sets now raises
  a clear `ExodusWriteError` instead of writing a truth table that claims
  variables exist for blocks defined later.
- The writer now records `maximum_name_length` (raised to the longest name
  written), stores `api_version`/`version` as NC_FLOAT, and NUL-terminates
  every name, QA field and info line (at most `len - 1` characters are kept).
- `diff`: node and side sets are compared with sorted entries (exodiff
  `-nsmap`/`-ssmap`), their member lists must agree when they carry
  variables, and side-set distribution factors are compared
  (`DiffOptions.sideset_df_tolerance` / `compare_sideset_df`).
- `diff`: when interpolating, every file-1 step is compared, times snap to
  file-2 steps within the time tolerance, and the final times must agree
  within `DiffOptions.final_time_tolerance` (`--final-time-tolerance`).  A
  step selection that compares nothing is a difference.  Automatic step
  offsets `STEP_OFFSET_AUTOMATIC` / `STEP_OFFSET_MATCH` (`-TA` / `-TM`,
  `STEP OFFSET AUTOMATIC` / `MATCH`) are implemented.
- `diff` exit codes follow exodiff: a variable in only one file, a variable
  requested in a command file but in neither file, and differing node or
  element counts are differences (exit 2; counts stop the comparison);
  a truth-table mismatch skips that block with a warning.
- exodiff command files: a bare per-variable value (`<tab> T 0.5`) sets the
  tolerance value; `FINAL TIME TOLERANCE` no longer overwrites the time-step
  tolerance; `EXCLUDE TIMES` (and `--exclude-steps`) accept ranges
  (`1-3,7`); `NODESET MATCH` / `SIDESET MATCH` no longer enable coordinate
  matching; `IGNORE SIDESET DISTRIBUTION FACTORS` and `SIDESET DISTRIBUTION`
  are honoured.
- Fixed `ParallelExodusFile.write()` dropping side-set distribution factors
  when Nemesis `ss_df_cnt_global` is 0 (it routinely is); factors stored by
  the component files are written, as SEACAS epu does.
- Fixed 3-D surface elements: SHELL4 and TRI3/TRISHELL areas used only x
  and y (a shell in the xz-plane had area 0).  Elements in an XY plane are
  unchanged.
- Element geometry supports more Exodus types: SHELL/TRISHELL, PYRAMID
  (`Pyramid5`), BAR/BEAM/TRUSS/ROD (`Line2`, length), and higher-order
  variants (TETRA10, HEX20/27, TRI6, QUAD8/9, WEDGE15, …) via their corner
  nodes.  New `element_class()` lookup.
- Values of block/set variables that the truth table marks as not stored
  are now NaN instead of zeros (which were indistinguishable from data);
  `values(..., fill=)` chooses the fill value.  The legacy API
  (`ExodusIIFile`, `ParallelExodusIIFile`) still returns zeros.
- Block and set IDs of 0 or below are accepted (libexodus reserves only
  -1).
- Fixed string decoding: text ends at the first NUL (stale bytes after it
  were kept, e.g. `abc\0old` read as `abcold`), bytes that are not valid
  UTF-8 decode as Latin-1 instead of raising, and leading whitespace is
  kept (only trailing padding is removed, as in libexodus).
- Fixed legacy `put_element_block(..., num_attr=, num_edges_per_elem=,
  num_faces_per_elem=)` ignoring those arguments; attributes and
  element-to-edge/face connectivity are allocated as in exodusii < 2.
- Fixed `Lineout(y=0)` (an integer coordinate) being treated as a free
  axis, and the automatic tolerance being cached on the lineout and reused
  for differently scaled data.
- Fixed WEDGE6 volumes for wedges whose quadrilateral faces are not planar
  (a fixed three-tetrahedron split was off by up to ~18%); the volume is
  now the exact isoparametric integral.  The point-on-boundary test for
  quadrilateral regions is now relative to the element size instead of a
  fixed 1e-8.
- `ExodusWriter` methods taking an entity (`write_block_attributes`,
  `define_block`, `define_set`, `define_variables`, `define_property`,
  `set_block_status`, `set_set_status`, `write_id_map`, `write_values`)
  accept string names such as `"element_block"`, not only `Entity` members.
- The extension helpers (`compute_element_volumes`, `compute_element_centers`,
  `compute_node_volumes`, …) accept `ExodusFile` / `ParallelExodusFile`.
- `block_connectivity("element", …)` (and `"edge"`, `"face"`) selects the
  block entity instead of raising a bare `KeyError`.
- `region_mass(block_id=None)` sums over every block that stores the
  density.
- Writer edge cases: node variables can be defined on a mesh without nodes
  (e.g. an empty rank); unwritten float values (skipped time steps) read
  back as NaN instead of 9.97e36; rewriting QA/info records with a
  different count raises instead of writing a partial table; block
  attributes always get an `attrib_name%d` variable, as in SEACAS.
- Parallel files: component paths are ordered by numeric rank (`.10` after
  `.2`); a different number of time steps raises `ExodusConsistencyError`
  (was a numpy broadcast error); a warning is issued when the number of
  files differs from `num_processors`.  `util.epu()` accepts base names
  containing dots.
- Fixed the legacy writer creating every block/set value variable in
  `put_*_variable_params`, leaving variables the truth table marks as
  absent.  Variables are created by `put_*_variable_truth_table`, or on
  first write when no truth table is given (the table then records what was
  written).
- The writer rejects duplicate block/set IDs and the reserved ID -1, and
  writes `EX_INVALID_ID` (-1) for slots that are allocated but never
  defined (was 0, which could collide with a real ID).
- NFACED (polyhedral) element blocks report their element type, read from
  the element-face (or element-edge) connectivity as libexodus does.
- `allclose` no longer compares string-padding dimensions (`len_string`,
  `len_name`, `len_line`, `four`), so a plain copy compares equal to its
  source; the strings themselves are still compared.
- `exodiff` CLI: time values are compared with the default tolerance
  (`-t`, mode flags, `--floor`), as SEACAS exodiff does; it was always
  relative 1e-6.
- `diff` tolerance boundaries follow exodiff: field variables use
  `Delta > value` (COMBINED values exactly at the tolerance compare equal),
  and global variables use the scalar test (skipped when both magnitudes
  are at or below the floor).
- `diff` reports the file time step of the largest difference
  (`VariableDiff.time_index`, 0-based) instead of its position in the
  selected steps, including for global variables; the CLI prints it
  1-based.
- exodiff command files: `ELEMENT ATTRIBUTES` lists select which
  attributes are compared (with per-attribute tolerances and `!` /
  `(all)`), and attributes are not compared when a command file omits the
  section, as in exodiff.  `IGNORE MAPS` matches entities by file position.
- `compat.exodus_h`: restored the names exodusii < 2 provided (all
  `EX_INQ_*` codes, `VAR_EDGE_SET_VAR` / `VAR_FACE_SET_VAR` /
  `VAR_ELEM_SET_VAR`, set-attribute and map-name variables, …) and fixed
  `EX_ELEM_UNK`, which was the tuple `(-1,)`.  Existing values are
  unchanged.
- `copy()` warns about source data it cannot carry over (assemblies,
  blobs, reduction variables, numbered maps, set attributes, polyhedral
  counts, Nemesis decomposition data), and no longer fails on decomposed
  component files whose truth table lists variables in NULL blocks.
- Coordinate matching warns when a point has several candidates within the
  tolerance (the closest is used), as its documentation stated.

### Compatibility

- Historical serial read/write tests are supported.
- Historical parallel read/write tests are supported.
- Legacy private-ish `exodus_h` constants and enums are available as a
  compatibility surface.
- Legacy user-facing APIs generally preserve method-style access such as:
  - `title()`
  - `storage_type()`
  - `num_nodes()`
  - `get_*`
  - `put_*`

### Notes

- The `netCDF4` backend disables automatic character-to-string conversion and
  automatic masking/scaling to preserve raw Exodus storage semantics.
- Parallel set APIs return global labels, not internal contiguous indices.
- Connectivity arrays remain 1-based logical indices suitable for indexing the
  merged coordinate array unless otherwise documented.
- The old pure-Python NetCDF fallback is not the primary runtime path in the
  refreshed implementation.
