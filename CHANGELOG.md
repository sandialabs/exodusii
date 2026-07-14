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
