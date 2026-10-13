# ADR 0036: The CBST trace format and the link-time recorder

- **Status:** Accepted under delegation (2026-10-07; GOVERNANCE.md, "Delegation of technical
  ADRs"). It carries out M8a step 1 (PLAN 6.3 steps 1-3, 6.4.5 "Parity", 8.3; the M8a acceptance
  criteria 1-7). It changes no rule the author approved: the export patches stay additive
  (ADR 0013), no golden comparison is weakened, no gate or measurement protocol changes.
- **Date:** 2026-10-07
- **Deciders:** the AI assistant, on the author's behalf (M8a, step refs-traces)
- **Number:** 0035 is taken by the 0.2.0 release branch (`release-0.2.0`, the mosp API review);
  renumber at the merge if another branch takes 0036 first.

## Context

PLAN 6.4.5 proves the ESCHER store port by **CBST trace replay**: every operation that
ESCHER-GPU@abcf9b2 (A) and MOSP_ESCHER@4b86159 (B) perform on a CBST is recorded with its inputs
and the state of the tree after it, and the dynG store (and, first, the host model of M8a step 2)
must reproduce it. ADR 0013 allows only **additive** export patches: no tracked file of a scratch
copy may change. Neither original has a hook for recording, and the CBST functions are called
from several translation units (A: `dynamic_hypergraph.cu`, `graph_maintenance.cu`; B:
`DynamicHypergraph.cpp`, `DynamicGraph.cpp`, the unit tests, through the static library
`libescher_core.a`). Adding a recorder to the sources would change tracked files.

## Decision

### 1. The recorder is linked in, not compiled in

`parity/export_patches/{ESCHER-GPU,MOSP_ESCHER}/build.sh` call
`parity/exporters/escher/relink_twins.sh`, which, for every executable of the original that links
the CBST (A: `build/main`, `type1-3`, `coarseTriangle`; B: the 14 executables of `bin/` that link
`libescher_core.a`):

1. takes the original's own link command from `make -n -B <target>` (with the make variables of
   the reference's build command);
2. runs it again with the recorder objects first and `-Xlinker --wrap=<symbol>` for every symbol of
   `wrap_symbols.txt` (and, for executables that link the batch entry points, `wrap_hooks.txt`):
   the original's UNCHANGED objects and archive are relinked; every call of a wrapped symbol from
   another object reaches `__wrap_<symbol>`, which records and calls `__real_<symbol>`, the
   original;
3. keeps the original executable as `parity_export/orig/<name>` and installs the twin in its
   place, with a fresh mtime, so the original's test scripts run the twins and `make -q <target>`
   still holds (make does not relink them away).

`verify_copy` of `build_reference.sh` proves after the patch that every tracked file of the copy
still equals the archive. The recorder sources (`cbst_recorder.{hpp,cpp}`,
`recorder_escher_gpu{,_hooks}.cpp`, `recorder_mosp_escher{,_hooks}.cpp`) are dyng-authored
Apache-2.0 code: they include the copy's headers, read the documented fields of `CBSTContext` /
`CBSTNode` (a mirror struct checked by `static_assert` against the original's layout) and call the
original's own functions (`context()`, `deviceBytes()`, `degreeBinLaunches()`, `checkTreeRows`,
`checkSubtreeAvail`). They copy no original code. A member function is wrapped as a free function
whose first parameter is `this` (the Itanium C++ ABI; a returned `InsertMapping` or
`BatchResult` comes through the hidden result pointer before it, as for a free function).

Wrapped: the `CBSTOperations` constructor, destructor, move constructor and move assignment (a
tree id survives moves), `construct`, `rebuildFromDevice` (A), `insert`, `fill`, `erase`, `unfill`
(A), `compact` (B), every free `*CBST` function (B's `unfillCBST` is called from
`DynamicHypergraph.cpp:53`, `DynamicGraph.cpp:431` and `test_cbst_ops.cu`), and the batch hooks
`DynamicHypergraph::build` / `update(Batch&, int)` (A) and
`escher_mosp::DynamicHypergraph::bulkLoad` / `beginBatch` / `finishBatch` (B).

**The census.** `--wrap` sees only calls between objects. `parity/exporters/escher/cbst_census.py`
runs `nm` on every object and archive member of every twin's link command, finds the object that
defines `CBSTOperations` and fails when another object references an API symbol that is neither
wrapped nor one of the read-only accessors (`context()`, `deviceBytes()`, `findAndPrint()`,
`degreeBinLaunches()`), or when a twin lacks a `__wrap_` function it needs. Calls inside the
defining object (members calling the free functions, B's compaction inside `fillCBST` and
`insertCBST`, A's payload growth in `ensurePayloadCapacity`) are not separate calls: the **state
digest after every wrapped call** shows their effect, and the reader infers compaction and growth
from it. The executables that are not twins (A's `build/cbst_test`, B's
`tests/local/local_tests`) are checked to have no CBST API symbol.

**Off means off.** Without `DYNG_CBST_TRACE_DIR` every wrapper calls `__real_` and returns (one
`getenv` at the first call); nothing is read back, synchronized, counted or written.
`parity/exporters/escher/check_twins.py` runs the original's suite in the patched copy and replays
every twin invocation it made with the unpatched executable and with the twin; their exit status
and output (timing lines left out) must be equal. B's `bin/main` draws its graph from
`std::random_device` (`mosp/src/generateGraph.cu:89-90`): two runs of the unpatched copy differ,
so it is compared on its exit status and verdict lines only.

**The invocation log.** With `DYNG_CBST_INVOCATION_LOG=<dir>` a twin logs its arguments and input
files at start-up and the number of CBST operations it called at exit. The export requires one
trace per invocation that called the CBST, with the same arguments and as many operations.

### 2. The trace format (version 1)

One trace per process, `<DYNG_CBST_TRACE_DIR>/NNNNNN.cbt` (numbered in the order of first use,
so sequential test scripts give deterministic names), and its side file `NNNNNN.blob`.

- **File:** `DYNGCBT1`, u32 format version (1), u32 0, then records. **Record:** u32 tag, u64
  payload length, payload. **Payload:** a field list: u32 count, then per field u8 type, u16 name
  length, name, value. Types: 1 int64; 2 string (u32 length + bytes); 3 array (u8 storage
  0 inline / 1 side file / 2 hash only / 3 null, u8 element size 4 or 8, u64 count, then the
  values, or u64 offset + u64 hash, or u64 hash); 4 string list. Little-endian. Hash: FNV-1a 64 of
  the little-endian bytes.
- **Side file:** `DYNGCBB1`, u32 1, u32 0, then raw arrays (the large construct, rebuild and
  batch inputs in digest mode).
- **Records:** `header` (format and recorder versions, repo, commit, mode, the argv with absolute
  paths shortened to `<abs>/<name>`, the input files named on the command line with hash and size
  and their copies in `inputs/` (not the value of an output option, `--out`, `--timing`, `-o`,
  `--output`: a CSV that an earlier run appended to is an output, not an input), an environment subset, the hash-only policy); `tree` (tree id,
  dataset name, payload capacity, alignment, origin); `event` (construct / destroy / move of a
  tree object); `op`; `batch` (a batch hook, phases `enter` and `exit`); `note`; `end` (exit
  `normal` or `exit-during-op`, the number of ops, the tree table: id, name, capacity, alignment).
  Every record has `seq`, the global order.
- **op:** `tree`, `op` (construct, rebuild, insert, fill, erase, unfill, compact); the full
  inputs `in.*` (keys, payload, prefix sizes, requested keys; construct: keys, start offsets,
  flat payload, flat size, capacity, alignment and B's row occupancy or `row_occupancy_null`;
  rebuild: the device start offsets and payload, copied before the call); `pre.matches` (1 if the
  state at entry equals the state recorded after the tree's previous op, 0 if something changed
  it unrecorded, 2 first op); `before.*` and `after.*` (fixed size, number of records, initial
  payload size, alignment, scratch capacities, A's `deviceBytes()`, D = the root's
  `subtreeAvail`); `status` and `exception.type` / `exception.what`; `out.mapping` (the
  `InsertMapping`), `out.compact_return`; `bins.fill|reuse|unfill` (A's `degreeBinLaunches()`
  delta, thread / warp / block); the **state digest**; `check.rows` / `check.subtree_avail`
  (the original's `checkTreeRows` on the rows as read, order-sensitive, and B's
  `checkSubtreeAvail`; `check.log`, their report, only when they found a violation); `rec.*` (the recorder's own checks: subtreeAvail consistency, unterminated
  rows, key order, unreachable nodes, duplicate keys, the longest chain).
- **State digest:** `state.keys` (BST in-order), `state.subtree_avail` (by heap position), and per
  key `key.key, avail, length, size, occupancy, tail_capacity, value, tail_base, segments,
  row_len, terminated, row_hash` and `key.rows` (the rows in read order: positive values from
  `value`, a negative value other than `INT_MIN` continues at its negation, 0 or `INT_MIN` ends
  it), plus `state.keys_hash`, `state.subtree_avail_hash`, `state.hash` (of the scalars, the two
  hashes and every key's record hash). **No pointer is recorded.**
- **Modes.** `full` digests write every key every op. A **delta** digest writes only the keys
  whose record changed or that the op names, `state.keys` / `state.subtree_avail` only when they
  changed, and `state.key_hashes` (the 64-bit hash of every key's record, in key order: "per-key
  hashes for the rest"). `DYNG_CBST_TRACE_MODE`: `full`; `digest` (delta digests, large inputs to
  the side file); `auto` (the default: delta digests for trees with more than
  `DYNG_CBST_TRACE_FULL_MAX` = 4096 keys or row values, full otherwise; a tree's first digest is
  always full); `batches` (the batch dumps of M9: no digests, and the ops' input arrays as hashes
  only, since the batch records carry the inputs). Both are lossless: the readers
  rebuild the full state after every op and check every hash.
- **Hash only.** `DYNG_CBST_TRACE_HASH_ONLY=<dataset>:<construct|rebuild|rows>,...` records those
  arrays as hashes only (storage 2): the readers report them by name. Used where the 10 GB budget
  requires it (parity/results/M8a.md lists which).

### 3. Readers and the golden sets

`cpp/tests/parity/escher/cbst_trace.{hpp,cpp}` (host-only: `reader`, `state_tracker`) and
`parity/escher_trace.py` (`summary`, `dump`, `verify`) read the format; both rebuild the states and
check every hash, so recorder and readers agree on every byte. `parity/cbst_trace_goldens.py`
(through `parity/export_goldens.py cbst_trace`) exports the sets of PLAN 6.4.5 into
`$DYNG_SCRATCH/goldens/cbst_trace/<set>` (zstd, `--long=27`), twice from fresh copies with
identical manifests, and records them in `parity/goldens.toml` as `[sets.cbst_trace_<set>]`
(`cbst_trace/<set>` in the milestone text; the section names avoid `/` because the goldens.toml
writers split sections by name). A subset of small traces is committed under
`cpp/tests/data/goldens/cbst_trace` and read by `dyng_cbst_trace_tests` (labels cpu, parity, cbst)
in hosted CI.

The recorder is written for reuse: M9's golden batches (synthetic 100K / 1M / 2M, coauth seeds
1-3, batch sizes 20 to 200K) are `batches`-mode runs of the same twins.

## Consequences

- The export patches stay additive: the twins are build products of the patched copy; the
  unpatched copy (performance baselines) is untouched.
- A call between objects that the census does not see cannot exist (the census reads the actual
  objects); a mutation of a tree outside the wrapped calls would show as `pre.matches = 0`.
- Traces with tracing on are slower (every op reads the tree back and runs the original's
  integrity check), never different: tracing changes timing only.
- MOSP_ESCHER's demo `bin/main` cannot be part of a reproducible set; it is recorded and listed
  as excluded with the reason and nothing measured from its run, so that the set's `info.json`
  and manifest stay reproducible (its CBST calls are those of `DynamicGraph`, which the seeded
  `stressTest`, `parallelStressTest` and `test_dynamicgraph_roundtrip` make too).

## Later changes

- M8a acceptance (2026-10-13): the `excluded` record of `bin/main` in the `info.json` of
  `cbst_trace_mosp_escher_test` carried the op counts of that run (108 or 111 ops, by the graph
  `std::random_device` drew), so the set's manifest was reproducible only by chance. The record
  now holds the invocation and the reason only (`cbst_trace_goldens.py`, `excluded_entry`,
  tested); the set was exported again twice and re-pinned. `cbst_trace_goldens.py once main`
  still records and replays one run of `bin/main`.
