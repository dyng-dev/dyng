# M8a parity record: the ESCHER CBST store

Date: 2026-10-07 onwards. M8a ports the CBST store of ESCHER-GPU@abcf9b2 (A) and merges the
behaviours of MOSP_ESCHER@4b86159 (B) into it, proven by CBST trace replay against both
originals (PLAN 6.4.5). Each step adds its section; step 1 (refs-traces) writes sections 1-9, step 2 (host-model) section 10, step 3 (verbatim-port) section 11, step 4 (merge-invariant) section 12.

| Item | Value |
|---|---|
| Branch | `m8a-escher-store`, from `main` at `8b9067ea437bdc35e08e7019f6adcef03fe670f7` (`8b9067e`), which contains M7 (`7d64c90`, pull request #8: `git merge-base --is-ancestor 7d64c90 origin/main`) and M6b (pull request #10: `origin/m6b-hardening` is an ancestor) |
| References | ESCHER-GPU `abcf9b2bd42f896ec3103a6835edf713ed5e416c` (A), MOSP_ESCHER `4b86159ec0b3bfe6e18c718ff4bef1494d7276a5` (B, the committed tree only) |
| Toolchain | GCC 12.2.0, CUDA 13.1 (`/usr/local/cuda-13.1/bin/nvcc`), sm_86 |
| Host | Xeon Gold 6258R (28 cores, 56 threads), 124 GB, 2x RTX A5000 (sm_86); Debian 12, Linux 6.1; GPU 1 for every run of this step (none is timed) |

## 1. Step refs-traces: the references and their builds

**The pins** (`parity/references.toml`):

| | ESCHER-GPU (A) | MOSP_ESCHER (B) |
|---|---|---|
| commit | `abcf9b2bd42f896ec3103a6835edf713ed5e416c` | `4b86159ec0b3bfe6e18c718ff4bef1494d7276a5` (the committed tree; the clone's uncommitted changes never reach a copy) |
| `baseline-2026-09` | `6a44c1cf5f1639721d35b418f99b420c5c45674e` | `80cee6284aae8836e5319eabf5e8bd127c0dca82` |
| build | `make -j{jobs} all-types build/cbst_test NVCC=/usr/local/cuda-13.1/bin/nvcc ARCH=sm_86` | `make -j{jobs} all NVCC=/usr/local/cuda-13.1/bin/nvcc CUDA_ARCH=sm_86` (builds `build/lib/libescher_core.a`) |
| tests | `make test ...` (`tests/run_tests.sh`) | `make test ...` (`tests/run_tests.sh`) |
| cxx / toolchain | `/usr/bin/g++` 12.2; nvcc 13.1 `-std=c++17 -O3 -arch=sm_86 -lineinfo -Xcompiler -fopenmp`; the Makefile's hard-coded `CXX = g++` | `/usr/bin/g++` 12.2; nvcc 13.1 `-std=c++17 -O3 -lineinfo --extended-lambda -arch=sm_86 -Xcompiler -fopenmp` for `.cu` and `.cpp` |
| gitlinks | `Irregular Optimizations` -> `2591f3e`, `Irregular-Graph-Optimizations-for-GPU` -> `a1b334f`: empty directories in the commit (no `.gitmodules`); `archive-check.txt`: "unused: ... absent in the copy, named by no dependency file (*.d) and not by the build log" | none |
| external inputs | coauth-DBLP-full.tar.gz (`get_coauth.sh`'s SHA-256) and the ESCHER variant (`datasets/registry.toml`); `bench/baseline` (archives the baseline tag from the original; not used); the awk inputs of `tests/run_tests.sh` | the same archive and the H-SOSP variant; the awk input of `tests/run_tests.sh`; `tests/local` (host g++) |

`parity/build_reference.sh --fresh ESCHER-GPU MOSP_ESCHER` builds an unpatched and a patched copy of
each from `git archive` into `$DYNG_SCRATCH/ref/<name>@<sha7>/{unpatched,patched}`; `verify_copy`
checks every tracked file of every copy against the archive (content, type, executable bit) after
the build and again after the export patch. The original's `git rev-parse HEAD` and
`git status --porcelain=v1 --untracked-files=all` are kept as `origin-state.before.txt` /
`origin-state.after.txt` and must be byte-identical (A: the commit, clean; B: the commit and its
pre-existing changes `M Makefile`, `M README.md`, `M tests/run_tests.sh` and five untracked
snapshot-experiment files, unchanged by every build, test and export of this step).

**The originals' own suites in the unpatched copies** (`build_reference.sh --test`, GPU 1): A
94 PASS, 0 FAIL ("All tests passed."); B 28 PASS, 0 FAIL, 0 XFAIL ("=== all tests passed ==="). Every store-related case passes, so
the GO of step 1 holds. Every result:

```text
ESCHER-GPU@abcf9b2 tests/run_tests.sh
PASS  cbst_test (1..2^24 keys)
PASS  vertex id 2^31-1 rejected
PASS  max_vertex_id 2^31-1 rejected
PASS  vertex id 2^31-2 beyond memory rejected
PASS  --insertions 2^31-1 beyond memory rejected
PASS  negative --batch-size rejected
PASS  --batches beyond int rejected
PASS  num_hyperedges beyond int rejected
PASS  unwritable --timing rejected at start
PASS  smoke main
PASS  smoke type1
PASS  smoke type2
PASS  smoke type3
PASS  smoke coarseTriangle
PASS  main reference, small file
PASS  main reference, synthetic 100K
PASS  type1 reference, small file
PASS  type1 reference, synthetic 100K
PASS  type2 reference, small file
PASS  type2 reference, synthetic 100K
PASS  type3 reference, small file
PASS  type3 reference, synthetic 100K
PASS  coarseTriangle reference, small file
PASS  coarseTriangle reference, synthetic 100K
PASS  main small file, batch 20
PASS  main small file, batch 200
PASS  main small file, batch 1000
PASS  main synthetic 100K, batch 5000
PASS  type1 small file, batch 20
PASS  type1 small file, batch 200
PASS  type1 small file, batch 1000
PASS  type1 synthetic 100K, batch 5000
PASS  type2 small file, batch 20
PASS  type2 small file, batch 200
PASS  type2 small file, batch 1000
PASS  type2 synthetic 100K, batch 5000
PASS  type3 small file, batch 20
PASS  type3 small file, batch 200
PASS  type3 small file, batch 1000
PASS  type3 synthetic 100K, batch 5000
PASS  coarseTriangle small file, batch 20
PASS  coarseTriangle small file, batch 200
PASS  coarseTriangle small file, batch 1000
PASS  coarseTriangle synthetic 100K, batch 5000
PASS  main deletions only
PASS  main insertions only
PASS  main more deletions than insertions
PASS  main random inserts on file input
PASS  main sampled inserts on synthetic
PASS  main batch larger than the graph
PASS  main skewed degrees (block paths)
PASS  main skewed degrees (block paths): block-level kernels ran
PASS  main medium degrees (warp paths)
PASS  main medium degrees (warp paths): warp-level kernels ran
PASS  type1 deletions only
PASS  type1 insertions only
PASS  type1 more deletions than insertions
PASS  type1 random inserts on file input
PASS  type1 sampled inserts on synthetic
PASS  type1 batch larger than the graph
PASS  type1 skewed degrees (block paths)
PASS  type1 skewed degrees (block paths): block-level kernels ran
PASS  type1 medium degrees (warp paths)
PASS  type1 medium degrees (warp paths): warp-level kernels ran
PASS  type2 deletions only
PASS  type2 insertions only
PASS  type2 more deletions than insertions
PASS  type2 random inserts on file input
PASS  type2 sampled inserts on synthetic
PASS  type2 batch larger than the graph
PASS  type2 skewed degrees (block paths)
PASS  type2 skewed degrees (block paths): block-level kernels ran
PASS  type2 medium degrees (warp paths)
PASS  type2 medium degrees (warp paths): warp-level kernels ran
PASS  type3 deletions only
PASS  type3 insertions only
PASS  type3 more deletions than insertions
PASS  type3 random inserts on file input
PASS  type3 sampled inserts on synthetic
PASS  type3 batch larger than the graph
PASS  type3 skewed degrees (block paths)
PASS  type3 skewed degrees (block paths): block-level kernels ran
PASS  type3 medium degrees (warp paths)
PASS  type3 medium degrees (warp paths): warp-level kernels ran
PASS  coarseTriangle deletions only
PASS  coarseTriangle insertions only
PASS  coarseTriangle more deletions than insertions
PASS  coarseTriangle random inserts on file input
PASS  coarseTriangle sampled inserts on synthetic
PASS  coarseTriangle batch larger than the graph
PASS  coarseTriangle skewed degrees (block paths)
PASS  coarseTriangle skewed degrees (block paths): block-level kernels ran
PASS  coarseTriangle medium degrees (warp paths)
PASS  coarseTriangle medium degrees (warp paths): warp-level kernels ran

MOSP_ESCHER@4b86159 tests/run_tests.sh
PASS  test_cbst_smoke
PASS  cbst_scale
PASS  cbst_reuse
PASS  cbst_terminator
PASS  cbst_erase
PASS  cbst_surplus
PASS  cbst_bestfit
PASS  cbst_unfill-chain
PASS  cbst_random
PASS  cbst_churn
PASS  test_dynamicgraph_roundtrip
PASS  test_snapshot_matches_updateCSR
PASS  test_h2h_construction
PASS  test_h2h_delta
PASS  test_hsosp_matches_dijkstra
PASS  test_hsosp_scale
PASS  local_tests
PASS  hsospStress
PASS  hsospStress_escher
PASS  hsospBench_smoke
PASS  hsospBench_real_hyperedge
PASS  hsospBench_real_vertex
PASS  hsospBench_cli
PASS  test_mosp_update
PASS  main
PASS  stressTest
PASS  parallelStressTest
PASS  parallelStressTest_seed6
```

## 2. The recorders (additive link-time twins; ADR 0036)

`parity/export_patches/{ESCHER-GPU,MOSP_ESCHER}/build.sh` relink, in the PATCHED copy, every
executable that links the CBST from the original's own link command (`make -n -B <target>`) with
the recorder objects first and `-Xlinker --wrap=<symbol>` for every symbol of `wrap_symbols.txt`
(and `wrap_hooks.txt` where the executable links the batch entry points), and install the twins
in place of the executables (the originals stay in `parity_export/orig/`; `make -q <target>`
holds, so `make test` runs the twins). A: `build/main`, `type1`, `type2`, `type3`,
`coarseTriangle` (`build/cbst_test` links only `build_tree.o` and `insert_reuse.o`, no CBST API:
not a twin). B: the 14 executables of `bin/` that link `libescher_core.a` (`tests/local/
local_tests` links no CBST: not a twin). `verify_copy` after the patch: every tracked file of
both patched copies equals the archive.

**The nm census** (`parity/exporters/escher/cbst_census.py`; records
`M8a-census-ESCHER-GPU.json`, `M8a-census-MOSP_ESCHER.json`): passed for every twin. The API is
defined in `operations.o` (A) / `libescher_core.a(operations.cu.o)` (B); the references from
other objects:

| original | symbol | referenced from | kind |
|---|---|---|---|
| A | `CBSTOperations::CBSTOperations(const char*, int, int)`, `~CBSTOperations()`, `construct`, `insert`, `fill`, `erase`, `unfill` | `dynamic_hypergraph.o` | wrapped |
| A | `CBSTOperations::rebuildFromDevice` | `graph_maintenance.o` | wrapped |
| A | `CBSTOperations::context()`, `deviceBytes()`; `degreeBinLaunches()` | `dynamic_hypergraph.o`, `graph_maintenance.o`; `driver.o` | read-only accessors |
| A | `DynamicHypergraph::build`, `update(Batch&, int)` | `driver.o` | batch hooks (wrapped) |
| B | `CBSTOperations::CBSTOperations`, `~CBSTOperations()`, `construct`, `insert`, `erase` | `DynamicGraph.cpp.o`, `DynamicHypergraph.cpp.o`, `test_cbst_ops.cu.o`, `test_cbst_smoke.cu.o` | wrapped |
| B | `CBSTOperations::fill`; `unfillCBST` (free) | `DynamicGraph.cpp.o`, `DynamicHypergraph.cpp.o`, `test_cbst_ops.cu.o` | wrapped |
| B | `CBSTOperations::context()` | `DynamicGraph.cpp.o`, `DynamicHypergraph.cpp.o`, `test_cbst_ops.cu.o` | read-only accessor |
| B | `escher_mosp::DynamicHypergraph::bulkLoad`; `beginBatch`; `finishBatch` | `hsospBench.cu.o`, `hsospStress.cu.o`, the `test_h2h_*` and `test_hsosp_*` objects; `hsospDelta.cu.o`, `test_hsosp_matches_dijkstra.cu.o`; `hsospDelta.cu.o` | batch hooks (wrapped) |

No other object calls `compact()`, `compactCBST` or the other free `*CBST` functions; they are
wrapped as well (the targeted scenarios call `compact()`). Calls inside the defining object are
seen in the digest after the wrapped call: B's compaction inside `fillCBST` / `insertCBST`
(inferred by `escher_trace.py summary` when rows the op does not name change their offsets,
tails or chains, or the bump pointer falls: 17 fills and 1 insert of `test_cbst_ops churn`, 30
fills of `test_hsosp_matches_dijkstra`), A's growth in `ensurePayloadCapacity` (`after.fixed_size`
above `before.fixed_size`).

**Off means off** (`parity/exporters/escher/check_twins.py`; records `M8a-twins-ESCHER-GPU.json`,
`M8a-twins-MOSP_ESCHER.json`): the suites pass in the patched copies with tracing off (A 94 PASS,
B 28 PASS), and every twin invocation they made was replayed with the unpatched executable and
with the twin: same exit status and the same output with the timing lines left out, A 83 of 83
(75 that call the CBST, 8 that reject their input first), B 32 of 32 (25 that call the CBST). B's
`bin/main` draws its graph from `std::random_device` (two runs of the unpatched copy differ), so it
is compared on its exit status and verdict lines ("Test case i: PASS", the summary).

## 3. The trace format and the golden sets (PLAN 6.4.5 "Parity"; ADR 0036)

**Format 1** (ADR 0036, section 2): one trace per process (`NNNNNN.cbt`, in the order of first
use) and its side file (`NNNNNN.blob`); per op the tree id and op, the full inputs (keys,
payload, prefix sizes, requested keys, the construct arrays with B's `rowOccupancy` or
`row_occupancy_null`, alignment, capacity), the exception type and message, the `InsertMapping`
and D for inserts, `initialPayloadSize` and `fixedSize` before and after, A's `deviceBytes()`
and the `degreeBinLaunches()` delta; after each op the state digest (`numRecords`, the keys in
order, per key avail, length, occupancy, tail capacity, segment count and the row in read order,
`subtreeAvail` by heap position, the original's integrity verdict). No pointer is recorded.
Delta digests (`auto` above 4,096 keys or row values, `digest`) keep the rows of touched keys in
full and a 64-bit hash of every other key's record; hash-only inputs are labelled (storage 2)
and listed per set. Readers: `cpp/tests/parity/escher/cbst_trace.{hpp,cpp}` (host-only, library
`dyng_cbst_trace`, test `dyng_cbst_trace_tests`, labels `cpu;parity;cbst`) and
`parity/escher_trace.py` (`summary`, `dump`, `verify`); both rebuild every state and check every
hash.

**The sets** (`parity/export_goldens.py cbst_trace <set> --twice`: every set exported from the
scratch reference and again from a fresh `git archive` copy, with byte-identical
`MANIFEST.sha256`; manifests in `parity/goldens.toml` as `[sets.cbst_trace_<set>]`, distinct
from the sssp group `escher`; data zstd-compressed in `$DYNG_SCRATCH/goldens/cbst_trace/<set>`):

| set | cases | what (the originals' own commands) | mode | hash-only | compressed | `MANIFEST.sha256` |
|---|---:|---|---|---|---:|---|
| `cbst_trace_escher_gpu_test` | 75 | every CBST-calling invocation of A's `tests/run_tests.sh`, the skewed and medium cases included; the awk inputs copied into each case (`/usr/bin/gawk`, GNU Awk 5.2.1, in `info.json`) | auto | none | 775.4 MB | `3b978a38514b...` |
| `cbst_trace_escher_gpu_coauth50k` | 1 | `build/main --input <abs>/coauth.hg --batches 3 --batch-size 50000 --seed 1 --check` | digest | `H2H:rebuild`, `H2H:rows` | 888.1 MB | `7ddec4438292...` |
| `cbst_trace_escher_gpu_syn1m` | 3 | `build/main 1000000 10 1 333333 --batches 3 --batch-size 50000 --seed s`, s = 1, 2, 3 (batch dumps) | batches | the ops' inputs (the batch records carry them) | 55.7 MB | `d73a368586dd...` |
| `cbst_trace_escher_gpu_targeted` | 2 | `cbst_targeted small|large` through A's unchanged `CBSTOperations` | full | none | 24 KB | `6f4c9428ac5e...` |
| `cbst_trace_mosp_escher_test` | 24 | every CBST-calling invocation of B's `make test` but `bin/main` (retrospective, deviation 1): `test_cbst_smoke`, the 9 `test_cbst_ops` scenarios, `test_dynamicgraph_roundtrip`, `test_snapshot_matches_updateCSR`, the `test_h2h_*` tests, `test_hsosp_matches_dijkstra` (300-batch churn), `test_hsosp_scale`, `hsospStress` (both), `hsospBench` (smoke, real hyperedge / vertex, cli), `test_mosp_update`, `stressTest`, `parallelStressTest` (both) | auto | none | 265.5 MB | `ed6400698e3d...` (re-pinned, section 15; was `fe2d29edce2a...`) |
| `cbst_trace_mosp_escher_dblp50k_hyperedge` | 1 | `hsospBench --hg <abs>/coauth.hg --kind hyperedge --batch 50000 --batches 3 --verify none --out hyperedge_50000_r1.csv` (B's `results/README.md`; seed 20260725 and `--maxcard 25` are the defaults of `scripts/run_experiments.sh` and `hsospBench`) | digest | none | 2,753.8 MB | `3b26a7697d18...` |
| `cbst_trace_mosp_escher_dblp50k_vertex` | 1 | the same with `--kind vertex` | digest | none | 2,008.3 MB | `1068e319acbc...` |
| `cbst_trace_mosp_escher_targeted` | 2 | `cbst_targeted small|large` through B's unchanged `CBSTOperations` | full | none | 25 KB | `280d185aaa4f...` |

**Total: 6.75 GB** (limit 10 GB). The full SHA-256 of every manifest and every case is in
`goldens.toml`. The batch dumps: A's (test, coauth50k, syn1m) record per run the loaded rows,
`maxVertexId`, and per batch `deletedKeys`, `deletedRows`, `insertedRows`, `insertedKeys`, the
seed and options (argv) and `DynamicHypergraph::deviceBytes()` after the build and after every
batch (syn1m seed 1: 703,051,032 bytes after the build, 703,183,432 / 703,130,968 / 703,185,928
after the batches); B's (hsosp runs) the bulk-load rows (virtual source and target included) and
weights, then per batch the `HgBatch`, `finalIds`, `deadHe` and `skippedOps`.

**The committed subset** (`cpp/tests/data/goldens/cbst_trace`, 307,170 bytes, 9 traces listed in
`SUBSET.txt` with their commands; the repository's test data stays at 470 KB): A's demo run and
targeted scenarios, B's `test_cbst_ops` terminator / erase / surplus / bestfit / unfill-chain and
targeted scenarios. Together they cover reuse, surplus with rebuild, fill overflow (with A's
growth), unfill on chained rows, and B's compaction inside fill and inside insert (section 5).

**Resume of 2026-10-07 (the two exports that failed in the first full chain).**
`cbst_trace_mosp_escher_test` differed between its two exports in `021-hsospBench`: the recorder
took the value of `--out hsospBench_real.csv`, which an earlier run in the same copy had
appended to, as an input file; the recorder now skips the values of output options (`--out`,
`--timing`, `-o`, `--output`; ADR 0036), and the set exports identically twice (58 files instead
of 60). `cbst_trace_escher_gpu_syn1m` failed in `escher_trace.py summary`, which read the
hash-only inputs of a `batches`-mode trace; the summary now skips what is recorded as a hash
(tested). The other sets were not affected (no other invocation names an existing output file)
and were not re-exported.

## 4. Dataset pins (PLAN 5.7; `datasets/registry.toml`)

`parity/cbst_trace_goldens.py datasets` checks the archive against the SHA-256 that
ESCHER-GPU@abcf9b2's `bench/datasets/get_coauth.sh` pins, extracts it into a temporary directory
and runs each original's own converter from its scratch copy:

| dataset | file | SHA-256 | produced by | counts |
|---|---|---|---|---|
| coauth-DBLP-full | `datasets/hyper/coauth-DBLP-full.tar.gz` | `feb4bed6d3c79a9ba16142e71405b7eff1515504741281b8edfe13dc0f960433` (= `get_coauth.sh`'s `archive_sha`) | Benson et al.'s data collection | 3,700,681 timestamped simplices |
| coauth-DBLP-escher | `datasets/hyper/coauth.hg` | `65fc83b22dfb3ecd14b15436f4aaee54631657f2f4afec8260150679387b7cb6` (= `get_coauth.sh`'s `hg_sha`) | A's `bench/datasets/convert_coauth.py` | 2,467,389 hyperedges, 1,930,378 vertices, 7,748,737 pins, max cardinality 280 |
| coauth-DBLP-hsosp | the same file | the same SHA-256 | B's `scripts/benson_to_hg.py` (writes the same bytes) | 2,467,389 lines; `hsospBench --hg` loads 2,466,792 hyperedges with `--maxcard 25` (its default): the bulk load of the DBLP batch dumps holds 2,466,794 rows (the virtual source, id 1, and target, id 2,466,794, added) and a line graph of 250,891,568 entries (125,445,784 pairs) |

Both converters keep the first occurrence of every distinct vertex set and renumber the vertices
1..V in id order, so they write the same file; the variants differ in how the originals load it
(B's loader drops the hyperedges of more than 25 vertices and adds the virtual source and target).
The existing `datasets/hyper/coauth.hg` of the work area has this SHA-256 and is used as is; no
dataset is in the repository.

## 5. Coverage per set (`parity/cbst_trace_goldens.py coverage`)

Counted by `escher_trace.py summary` from the traces (inserts and fills per tree; levels from the
item sizes of the inputs: thread < 32, warp < 1,024, block otherwise; compactions inferred from
the state digests).

| set | reuse + surplus + rebuild | reuse only | K = 0 | D = 0 | fill overflow + growth | unfill fallback on chained rows | fill T/W/B | reuse T/W/B | unfill T/W/B | compaction in fill | compaction in insert | compact() | capacity throw | reissued dead max key | max chain hops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| escher_gpu_test | 100 | 50 | 15 | 15 | 15 | 0 | 296685/15/10 | 108835/835/80 | 316915/15/10 | 0 | 0 | 0 | 0 | 0 | 3 |
| escher_gpu_coauth50k | 3 | 0 | 0 | 0 | 0 | 0 | 205961/0/0 | 74876/1/0 | 207312/0/0 | 0 | 0 | 0 | 0 | 0 | 3 |
| escher_gpu_syn1m | - | - | - | - | - | - | - | - | - | 0 | 0 | 0 | 0 | - | - |
| escher_gpu_targeted | 2 | 2 | 1 | 2 | 2 | 2 | 49/6/1 | 5/1/1 | 43/2/2 | 0 | 0 | 0 | 0 | 1 | 4 |
| mosp_escher_test | 406 | 707 | 0 | 183 | 0 | 18 | 1574209/2976/134 | 120562/6856/212 | 1525103/2548/5 | 47 | 1 | 0 | 1 | 3 | 98 |
| mosp_escher_targeted | 2 | 2 | 1 | 2 | 0 | 2 | 49/6/1 | 5/1/1 | 43/2/2 | 1 | 1 | 1 | 1 | 1 | 4 |
| mosp_escher_dblp50k_hyperedge | 3 | 2 | 0 | 1 | 0 | 1 | 3219477/6/0 | 82473/41516/0 | 3323223/28/0 | 0 | 0 | 0 | 0 | 0 | 3 |
| mosp_escher_dblp50k_vertex | 0 | 0 | 0 | 0 | 0 | 2 | 1202888/11453/0 | 0/0/0 | 1690239/17705/0 | 0 | 0 | 0 | 0 | 0 | 3 |

`escher_gpu_syn1m` is a `batches`-mode set (batch dumps for M9: no digests, the ops' inputs as
hashes), so its paths are not counted ("-"); its recorded launches are thread-level only. Every
path of the plan's list is covered: the paths A's suite does not reach (the serial unfill
fallback on chained rows, the reissued deleted maximal key) and B's compaction and capacity
throw in small form are in the targeted sets (`parity/exporters/escher/cbst_targeted.cpp`,
calling each original's unchanged `CBSTOperations` API). The longest chain of any trace is 98
hops (B's `test_cbst_ops` churn), far below B's literal 1,024-hop bound: the bound is not
observable in any of B's traces.

## 6. The cross-check of the two cores (PLAN 6.3 step 2)

`parity/escher_crosscheck.py run --set <set> --store <original>` replays every trace of a set
through the other original's UNCHANGED store (`parity_export/bin/cbst_crossreplay`, linked from
that original's objects like a twin, so the replay is itself recorded) and compares the two
traces op by op: the exception, the `InsertMapping`, the key list, `subtreeAvail`, per key avail,
length, occupancy, tail capacity, segment count and the row in read order, then the offsets
(value, tail base) and the bump pointer. With the trace's own store the replay must be identical,
which checks the driver and the comparator: **A through A: 240 of 240 trees identical, B through
B: 3,390 of 3,390 trees identical** (`M8a-crosscheck-a_on_a.json`, `M8a-crosscheck-b_on_b.json`).

| replay | trees | identical | first divergences and their cause |
|---|---:|---:|---|
| A's test traces through B's store (`--capacity-floor 67108864`: B's capacity is fixed, A grows) | 240 | 162 | H2V 75/75 and V2H 75/75 identical (offsets included); H2H 65 and V2V 13 at their first `rebuildFromDevice` (`rebuild_only_in_a`: B has no rebuild; the 10 H2H and 2 V2V without a rebuild are identical) |
| B's test traces through A's store | 3,390 | 3,372 | 11 `test_cbst_smoke` trees at their construct (`caller_counts`: the test constructs with `rowOccupancy = nullptr`, B's legacy occupancy 0, where A scans the payload, `initTailMetadata`); 7 trees at a fill or insert that B compacts (`grows_vs_compacts`: `test_cbst_ops churn` small / random / insert / growth at ops 197, 113, 31 and 61, `test_hsosp_matches_dijkstra` h2v / v2h / h2h at ops 100, 104 and 42; B's compaction rewrites the slot lengths, occupancies and offsets of the rows the op does not name, A keeps them and grows) |
| A's coauth-DBLP trace through B's store (H2V and V2H, `--capacity-floor 134217728`) | 2 | 2 | none: 7 of 7 ops of each identical (A's H2H is rebuilt from the device every batch, which B has no call for, and its coauth inputs are hash-only: not replayed) |
| B's DBLP hyperedge trace through A's store (h2v, v2h, h2h) | 3 | 3 | none: 7, 7 and 13 ops identical |
| B's DBLP vertex trace through A's store (h2v, v2h, h2h) | 3 | 3 | none: 7, 7 and 7 ops identical |
| A's targeted scenarios through B's store | 13 | 9 | fill_compact (op 4) and insert_compact (op 3): B compacts where A grows (`grows_vs_compacts`); capacity_throw (op 1): B throws `escher::EscherError` at its fixed capacity, A grows (`grows_vs_compacts`); small_capacity (op 0): B rejects a construct below the payload size (`b_validates`) |
| B's targeted scenarios through A's store | 13 | 8 | the same four, seen from B, and compact (op 3): A has no `compact()` (`compact_only_in_b`; the replay stops there) |

Classes: `rebuild_only_in_a`, `caller_counts`, `grows_vs_compacts`, `b_validates`,
`compact_only_in_b` (the comparator's documented behaviour differences, `escher_crosscheck.py`);
`unexplained` occurs **0 times** in every replay. Every first divergence of every tree, with its
case, op index, sequence number, field and class, is recorded in
`parity/results/M8a-crosscheck-<replay>.json` (`a_on_a`, `b_on_b`, `a_on_b`, `b_on_a`,
`a_on_b_coauth`, `b_on_a_dblp_hyperedge`, `b_on_a_dblp_vertex`, `a_on_b_targeted`,
`b_on_a_targeted`).

Every other tree of B's suite (the 9 scenarios but churn, `DynamicGraph`'s 1,809 trees, H-SOSP's
h2v / v2h / h2h of the stress, delta, scale and bench runs) is identical through A's store, offsets
included: before its first compaction B behaves exactly as A. No divergence is unexplained.

## 7. The timed-region map (PLAN 6.3 step 3)

`parity/timed_regions/hypergraph.toml` (committed in `81a8897`, before any port commit) maps
ESCHER-GPU's timers to the planned dynG stages; `perf_ab.py`'s strict loader reads it
(`load_regions("escher_gpu", HYPERGRAPH_REGION_MAP, ...)`, tested by
`test_hypergraph_region_map_loads` and `test_hypergraph_region_map_rejects_unknown_timers`):

| region | ESCHER-GPU timers | dynG stages |
|---|---|---|
| maintain | `update.maintain` (`DynamicHypergraph::update`, including the host preparation `maintain.prepare`) | `hypergraph.apply` |
| maintain.delete | `maintain.h2v_delete` + `maintain.v2h_unfill` | `hypergraph.apply.delete` |
| maintain.insert | `maintain.h2v_insert` + `maintain.v2h_fill` | `hypergraph.apply.insert` |
| maintain.line_graph | `maintain.h2h_merge` (+ `maintain.v2h_view` + `maintain.v2v_merge` with the vertex graph) | `hypergraph.apply.line_graph` |
| maintain.views | `maintain.row_views` | `hypergraph.apply.views` (optional) |
| maintain.prepare | `maintain.prepare` (host, no sync) | inside `hypergraph.apply` |
| build | `init.v2h_build` + `init.h2h_build` + `init.v2v_build` + `init.flatten` | `hypergraph.build` |
| upload | `init.construct` + `init.row_views` | `hypergraph.upload` |
| load | `init.load` | `io.read_hypergraph` |

Every gate of the map is "none": M8b reads the maintain region, M9 gates `update` (subtract +
maintain + add). H-SOSP's `t_escher_ms` (`[reference.mosp_escher]`, `csv:t_escher_ms`: the h2v
insert of `beginBatch` plus `finishBatch`) is noted for M11.

## 8. Open points for step 2 (the host model)

- **Read through the C++ reader.** `dyng::test::cbst_trace::reader` / `state_tracker`
  (`cpp/tests/parity/escher/cbst_trace.hpp`, library `dyng_cbst_trace`) parse every set and rebuild
  every state with its recorded hashes; the comparator compares a store's state with the
  tracker's. Large sets: decompress `trace.cbt.zst` and `trace.blob.zst` with
  `zstd -d --long=31` (the side file holds the construct, rebuild and batch inputs of digest
  mode).
- **What the model must reproduce** (the cross-check confirms it on both originals' suites): with
  enough capacity B's store gives A's traces exactly (offsets included); B's compaction starts at
  the ops listed in section 6, where A grows instead; `test_cbst_smoke` constructs with
  `rowOccupancy = nullptr` (B's occupancy 0), which the model must take from the trace's
  `in.row_occupancy_null`.
- **Hash-only inputs.** A's coauth H2H rebuild inputs and H2H rows are recorded as hashes
  (`goldens.toml`: `hash_only = ["H2H:rebuild", "H2H:rows"]`): A's coauth H2H is checked by its
  hashes; its construct payload is in the side file.
- **B's `bin/main`** is not in `mosp_escher_test` (deviation 1 of the retrospective); its
  `DynamicGraph` calls are covered by the seeded MOSP stress tests in the set.
- **The release certificate.** `certify.py` lists the `cbst_trace` sets with their kind and
  checks any replay record of one like any other set's, but a set without a replay record is
  shown as "listed" and kept out of the golden verdict (there is no store yet). The store's
  replay (steps 2-4) should write `parity-cbst_trace_<set>-*.json` records (`set`, the
  manifest, `configs` = the presets, `matrix`, `passed`) so that the 0.3.0 certificate gates
  them; then the `passed is None` exception of `golden_suites` can be removed.
- **The 1,024-hop chain bound of B** is not reached in any trace of B (the longest chain of every
  set is in section 5's table; PLAN 6.4.5's bounded `readFlat` is merged in step 4).

## 9. The checks of step 1

The runs of 2026-10-07 (scripts and logs in `$DYNG_SCRATCH/m8a/final` and `$DYNG_SCRATCH/m8a/resume`;
GPU 1, shared perf lock, niced):

| check | result |
|---|---|
| `build_reference.sh --fresh --test ESCHER-GPU MOSP_ESCHER` | both copies of both originals built from `git archive`, archive-equality check passed, both suites passed (section 1) |
| `build_reference.sh ESCHER-GPU MOSP_ESCHER` again (idempotent) and `--variant patched` after the recorder change | passed, archive-equality check passed |
| `cbst_census.py` (both) | passed: every CBST entry point referenced outside its defining object is wrapped |
| `check_twins.py` (both) | passed: A 83/83, B 32/32 invocations equal with tracing off |
| `cbst_trace_goldens.py datasets` | passed (section 4) |
| `export_goldens.py cbst_trace <set> --twice`, the 8 sets | all identical twice (section 3) |
| `cbst_trace_goldens.py subset`, `coverage` | written (sections 3 and 5) |
| `escher_crosscheck.py run`, 9 replays | 0 unexplained (section 6) |
| `git -C <original> rev-parse HEAD` and `status --porcelain=v1 --untracked-files=all` before and after every run, both originals | byte-identical (B keeps its pre-existing `M Makefile`, `M README.md`, `M tests/run_tests.sh` and untracked files) |
| `pytest parity/tests`, `ctest -L cbst`, `ci/check.sh` (`harness reuse provenance precommit docs regen`), the Clang 18 syntax pass and the `_FORTIFY_SOURCE=3` cpu-only build | passed |

## 10. Step host-model: the CBST host model, the go/no-go (PLAN 5.8; criterion 8; ADR 0037)

`dyng::testing::cbst_host_model` (`cpp/include/dyng/testing/cbst_host_model.hpp`,
`cpp/src/testing/cbst_host_model.cpp`) models one CBST of either original sequentially on the
host, with its own policy presets (`cbst_host_policy::escher_gpu()`, `mosp_escher()`, the
provisional `dyng_default()`); it keeps the flat payload, so every recorded field can be
predicted, and it predicts growth and compaction from the policy, never from the trace. The
comparator (`cpp/tests/parity/escher/cbst_compare.{hpp,cpp}`, tool `dyng_cbst_replay`) drives a
store through an adapter and compares after every op; `parity/compare.py cbst_trace --store
host_model` replays the sets. The field sets (ADR 0037, section 2): `semantic` (criterion 8: the
exception, every `InsertMapping`, the keys, `subtree_avail`, per key `avail`, length and the row,
in order for H2V / H2H / V2V and as a multiset otherwise, the bump pointer and capacity, the ops
of every growth and compaction), `offsets` (+ offset and tail base), `exact` (+ every other
recorded field: `size`, occupancy, tail capacity, segments, row length and termination, every
row in order, record count, scratch capacities, D, A's `deviceBytes()` and degree-bin launches,
B's `compact()` return, the state before every op).

**Result: GO.** Every replayable op of every set agrees in the `exact` field set (which contains
`semantic` and `offsets`), each trace under its original's preset
(`compare.py cbst_trace --store host_model --fields exact --record parity/results`, Release
build of `dyng_cbst_replay`, 8 jobs, 4 min 45 s with the manifest checks; records
`parity/results/M8a-host-model-<set>-exact.json`):

| set | traces | ops | compared | unreplayable | mappings | key states compared | hash-only rows | growths | compactions | exceptions | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `escher_gpu_test` | 75 | 1,176 | 1,176 | 0 | 180 | 10,440,560 | 0 | 30 | 0 | 0 | **0** |
| `escher_gpu_coauth50k` | 1 | 18 | 15 | 3 (H2H rebuilds: hash-only inputs) | 3 | 33,251,758 | 2,467,389 | 0 | 0 | 0 | **0** |
| `escher_gpu_syn1m` | 3 | 54 | 0 | 54 (batches mode: no digests) | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| `escher_gpu_targeted` | 2 | 52 | 52 | 0 | 7 | 443 | 0 | 3 | 0 | 0 | **0** |
| `mosp_escher_test` | 24 | 14,104 | 14,104 | 0 | 1,297 | 31,462,824 | 0 | 0 | 48 | 1 | **0** |
| `mosp_escher_targeted` | 2 | 53 | 53 | 0 | 7 | 445 | 0 | 0 | 3 | 2 | **0** |
| `mosp_escher_dblp50k_hyperedge` | 1 | 27 | 27 | 0 | 6 | 62,960,817 | 0 | 0 | 0 | 0 | **0** |
| `mosp_escher_dblp50k_vertex` | 1 | 21 | 21 | 0 | 0 | 48,010,053 | 0 | 0 | 0 | 0 | **0** |

- **The events** the model predicted are those the traces show: A's 33 growths (`test` 30,
  `targeted` 3); B's 51 compactions (the 18 of `test_cbst_ops churn`, 17 fills and 1 insert, and
  the 30 fills of `test_hsosp_matches_dijkstra`, as section 2 inferred them; the 3 of the
  targeted scenarios: in fill, in insert, `compact()`); B's 3 exceptions (`churn growth`'s capacity
  throw, the targeted capacity throw and construct validation).
- **Unreplayable, by the sets' design only** (ADR 0037): the 3 H2H rebuilds of
  `escher_gpu_coauth50k`, whose inputs (A's line-graph merge) are recorded as hashes only
  (`goldens.toml`: `hash_only = ["H2H:rebuild", "H2H:rows"]`; the H2H construct is replayed and its
  2,467,389 rows compared by their hashes), and `escher_gpu_syn1m`, the batch dumps of M9 (mode
  `batches`: no state digests). M8b can check the H2H rebuilds through the recorded hashes of their
  inputs once dynG computes the merge. No other op of any set is unreplayable;
  `parity/compare.py` and `dyng_cbst_model_large_tests` fail on any other.
- **Cross-policy replays reproduce the cross-check's classes** (section 6): B's traces under
  `escher_gpu` diverge first at the ops section 6 lists (`grows_vs_compacts`: churn small op 197,
  random op 113, ...; `compact_only_in_b`; `b_validates`), A's targeted trace under `mosp_escher`
  at its growth and capacity scenarios (`grows_vs_compacts`, `caller_counts`): the model's policies
  are the documented behaviour differences, nothing else.
- **The cbst_test layout** (A's `tests/cbst_test.cu`) on `check_cbst_layout`: n = 1..1024 and the
  17 sizes up to 2^24 (2,467,389 included) pass (permutation, in-order, every key found up to
  2,467,389 and a 65,536-key sample above, offsets, inverse formula).
- **Perturbations** are mismatches at the perturbed op: an insert item one value shorter
  (semantic and exact), two items in deleted slots of equal capacity with their recorded keys
  swapped (`out.mapping`), an op the policy cannot run (`compact()` under `escher_gpu`), a trace
  ending inside an op; a hash-only input makes its tree unreplayable with the reason.
- **Findings** (the facts given to the step against the code; ADR 0037, "Findings"): A's growth
  is `max(needed + needed/4, 3 * fixedSize / 2)` in integers; a surplus row of 0 values has length
  1; erase keeps the payload (a dead row reads its old values until reused or compacted; B's
  compaction empties dead rows); unfill depends on the degree bin of the removal count (the warp
  and block kernels compact the first `occupancy` slots of a one-segment row and drop non-positive
  values there; the thread kernel walks the whole chain); A's construct below the payload builds
  no tree but records the record count and the bump pointer; B's insert failing after its reuse
  leaves `subtree_avail` stale; on a 0-record tree, fill and unfill are no-ops while insert and
  erase hit a CUDA error (A exits, B throws). None of these was a divergence: the model followed
  the code from the start, and every set agreed at the first full run.
- **Isolation:** CTest `cbst_model_isolation` (cpu, parity, cbst): the model's source includes no
  project file but its header and its object references no `dyng::` symbol.

**The checks of step 2** (2026-10-07; shared perf lock, niced; no GPU):

| check | result |
|---|---|
| `cmake --preset cpu-only` build; `ctest -L 'parity\|cbst\|cpu' -LE slow` | built; 875 of 875 passed (`dyng_cbst_model_tests`, `dyng_cbst_trace_tests`, `cbst_model_isolation` included) |
| `ctest -R CbstLargeSets` (`dyng_cbst_model_large_tests`, every set in `$DYNG_SCRATCH`) | passed |
| `parity/compare.py cbst_trace --store host_model --fields exact` (all 8 sets) | 8 of 8 OK (table above) |
| `pytest parity/tests` | passed |
| `ci/docs.sh` (`--update-api`: 98 lines of `public_api.txt` for `dyng/testing/cbst_host_model.hpp`), `ci/check.sh`, the Clang 18 syntax pass, the `_FORTIFY_SOURCE=3` cpu-only build | passed |

## 11. Step verbatim-port: the port of ESCHER-GPU's store (criteria 9, 10(a), 14, 15, 16; ADR 0038)

`dyng::detail::escher::cbst_store` (`cpp/src/hypergraph/escher`, the OBJECT module
`dyng_hypergraph`) is ESCHER-GPU@abcf9b2's store, transplanted (c1) and changed by one class of
mechanical change per commit (c2 errors, c3 memory, c4 streams, c5 registration and counters),
then the dynG additions and the payload policy (ADR 0038; the store's README documents the
encoding). The transplant is reproducible: `python3 parity/escher_port/transplant.py --check
--rev 477d592` regenerates c1 from the original with `rename_map.toml` and
`manual_hunks.patch` ("17 files, identical"; pytest `test_escher_port.py`).

**The replay of every commit** (`parity/cbst_merge_steps.sh <commit>...`: a temporary worktree,
a Release CUDA build of the commit's replay targets, its own replay tests of the committed traces
and `compare.py cbst_trace --store device --fields exact` of the golden sets with the commit's
tool; GPU 1). ESCHER-GPU's sets must match in the `exact` field set (criterion 10(a) and more:
InsertMapping, rows in order, avail, subtree_avail, length, size, offsets, occupancy, tail
capacity, segments, the bump pointer, the capacity, the record count, the scratch capacities, D,
`device_bytes()` = A's `deviceBytes()`, the per-op degree-bin launches, the exception, the
integrity verdict, the state before every op). MOSP_ESCHER's sets run under `escher_gpu()` (the
store has no B preset in this step) and must diverge in every tree at exactly the op and in
exactly the field where the host model under `escher_gpu()` does (ADR 0037's behaviour classes),
and nowhere else. No B source line is merged in this step (B's behaviours are steps 4-5), so the
table has no "cited B lines" yet.

**The merge-step table.** Every row: the commit's own `dyng_cbst_device_tests` passed; A's sets
in `exact` (traces, ops compared, mismatches, documented divergences); B's sets under
`escher_gpu()` judged against the host model under the same preset (trees that diverge, each at
the model's op and field); "cited B lines": none (no B behaviour is merged in step 3). Small
sets at every commit; the large sets at the verbatim-port head `f2c48df` and at `7e397a2`.

| commit | change (class) | A: escher_gpu_test | A: escher_gpu_targeted | B: mosp_escher_test | B: mosp_escher_targeted | large sets |
|---|---|---|---|---|---|---|
| `477d592` | c1 transplant | 75 traces, 1,176 ops, 0 mismatches | 2 traces, 52 ops, 0 mismatches | 50 trees diverge, as the model | 7 trees diverge, as the model | |
| `3634128` | c2 errors | 1,176 ops, 0 | 52 ops, 0, 1 documented (`construct_below_payload`) | 50, as the model | 7, as the model | |
| `528b41a` | c3 memory | 1,176 ops, 0 | 52 ops, 0, 1 documented | 50, as the model | 7, as the model | |
| `9220f9c` | c4 streams | 1,176 ops, 0 | 52 ops, 0, 1 documented | 50, as the model | 7, as the model | |
| `f2c48df` | c5 registration, counters (the verbatim-port head) | 1,176 ops, 0 | 52 ops, 0, 1 documented | 50, as the model | 7, as the model | coauth50k: 15 ops, 0 mismatches, 3 unreplayable (hash-only H2H rebuilds); syn1m: 54 unreplayable (batches mode); DBLP hyperedge 27 ops and vertex 21 ops: 0 trees diverge |
| `69cac6d` | harness: `--store device` in compare.py | 1,176 ops, 0 | 52 ops, 0, 1 documented | 50, as the model | 7, as the model | |
| `c89a9b8` | addition: the empty tree | 1,176 ops, 0 | 52 ops, 0, 1 documented | 50, as the model | 7, as the model | |
| `e22cd71` | addition: clone | 1,176 ops, 0 | 52 ops, 0, 1 documented | 50, as the model | 7, as the model | |
| `9e54f3a` | addition: download | 1,176 ops, 0 | 52 ops, 0, 1 documented | 50, as the model | 7, as the model | |
| `cef82b5` | addition: `allocated_bytes()` | 1,176 ops, 0 | 52 ops, 0, 1 documented | 50, as the model | 7, as the model | |
| `d2be78a` | tests | 1,176 ops, 0 | 52 ops, 0, 1 documented | 50, as the model | 7, as the model | |
| `f67eee1` | `store_policy::escher_gpu()` | 1,176 ops, 0 | 52 ops, 0, 1 documented | 50, as the model | 7, as the model | |
| `7e397a2` | the step's last code commit | 1,176 ops, 0 | 52 ops, 0, 1 documented | 50, as the model | 7, as the model | as at `f2c48df` |

The rows are `$DYNG_SCRATCH/m8a/merge-steps/rows.md` (the first run of `477d592` failed in the
harness, a missing key of the totals fixed in `762522e`, and was run again); the records of the
last row are `parity/results/M8a-device-<set>-exact.json`. B's 10,163 compared ops of
`mosp_escher_test` are those up to each tree's first divergence (the rest of a diverged tree is
not compared).

**The documented divergence.** One op of the golden sets: the targeted scenario
`small_capacity` (a construct with a capacity of 8 below a payload of 32), where ESCHER-GPU
printed "Overflow" and kept a tree with 8 records and no nodes, MOSP_ESCHER threw its
`EscherError`, and dynG throws `invalid_argument_error` before it allocates anything (c2 on;
rule `construct_below_payload`, ADR 0038). The tree has no later op.

**MOSP_ESCHER's first divergences under `escher_gpu()`** (the same at every commit from c1; the
host model under the same preset agrees tree by tree): `mosp_escher_test` 50 of its trees
(`test_cbst_smoke`: 11 at their construct, `key.occupancy`: B's caller counts are null there,
occupancy 0, where A scans, `caller_counts`; 34 at the first operation that grows B's scratch,
`after.scratch_payload`: B grows its operation scratch geometrically, A to the exact size; 5 at a
fill or insert that B compacts and A grows, `after.fixed_size`, `grows_vs_compacts`; the
remaining trees, `DynamicGraph`'s and the H-SOSP trees that neither grow nor compact, are
identical in every field); `mosp_escher_targeted` 7 trees (4 at a
scratch growth, among them the tree of the explicit `compact()`; one each at a compaction where
A grows, `grows_vs_compacts`, at B's capacity throw, `status`, and at
`construct_below_payload`). B's DBLP runs (`dblp50k_hyperedge`, `dblp50k_vertex`) are identical
to A's behaviour in every field (0 divergences: no compaction, no scratch growth at that scale).

**Criterion 14, kernel-path coverage** (`dyng_cbst_coverage_tests`): replaying A's skewed (042-main)
and medium (043-main) traces runs every degree bin, every op with A's recorded launch counts
(`exact`): fill thread / warp / block 5 / 3 / 2 launches, reuse 5 / 3 / 2, unfill 5 / 3 / 2.

**Criterion 15, the early store memory** (the author's decision of 2026-10-07: store accounting;
`dyng_cbst_memory_tests`, `parity/perf_ab.py cbst memory --gpu 0` under the exclusive perf lock;
record `M8a-memory.json`). On `escher_gpu_coauth50k` every tree's `device_bytes()` after its
construct and after every operation of the 3 batches equals ESCHER-GPU's recorded
`deviceBytes()` of the same tree (ratio 1.0000, gate <= 1.05), and the store holds exactly that
(`allocated_bytes()`):

| tree | after construct | after batch 1 | after batch 2 | after batch 3 |
|---|---:|---:|---:|---:|
| H2V | 316,792,888 | 316,792,888 | 316,792,888 | 316,792,888 |
| V2H | 258,644,044 | 258,644,044 | 258,644,044 | 258,644,044 |
| H2H | 1,288,129,912 | 1,288,644,432 | 1,289,056,696 | 1,289,513,600 |

(bytes; dynG = ESCHER-GPU at every point and after every op of every batch, 18 readings). The
H2H rebuilds have hash-only inputs (A's line-graph merge): they are replayed with arrays of the
recorded sizes, all the accounting depends on. ESCHER-GPU's `DynamicHypergraph::deviceBytes()`
(1,922,784,204 after the build) also counts its row views (59.2 MB), which are the container's
(M8b) and compared there. **Nsight Systems** (recorded, not gated in M8): the peak of live device
allocations is 3,018.9 MiB for A's whole `build/main` (store, row views, motif counting) and
2,772.6 MiB for the store replay (the three trees, 1,777 MiB, plus the new row offsets and
payload of an H2H rebuild, about 985 MiB, which the store takes before it releases the old
ones, as A's rebuild does; the stream-ordered pool reached 2,912 MiB). Measured at `7e397a2`
(the docs of this section uncommitted), GPU 0.

**Criterion 16, robustness.** compute-sanitizer memcheck (with leak check), racecheck,
synccheck and initcheck: 0 errors and 0 hazards on `dyng_cbst_store_tests` (14 tests: the layout
sizes, the empty tree, clone, download, the accounting, the errors, two streams, the syncs, the
policy), `dyng_cbst_device_tests` (A's and B's committed traces, replayed twice) and
`dyng_cbst_coverage_tests`; `ci/gpu_local.sh` runs them (step `cbst`). Two stores on two
non-blocking streams equal a sequential run; two replays give identical digests; the host
synchronizations per op are A's read points and never more than A's blocking calls
(`CbstStoreSyncs`; ADR 0038, section 5).

**Criterion 9.** `cbst_source_rules` (cpu, cbst): no `exit(`, print, `verbose(`, `ESCHER_DEBUG`,
`cudaDeviceSynchronize`, raw `cudaMalloc` / `cudaFree`, synchronous copy, Thrust without the
resources' policy or default-stream launch in `cpp/src/hypergraph`, and no public header exposes
the store. `ci/provenance_check.py` covers the CBST identifiers; every file carries its
`// Derived from ESCHER-GPU@abcf9b2:<path>` line.

**The checks of step 3** (2026-10-07; shared perf lock, niced; GPU 1 unless stated):

| check | result |
|---|---|
| `ci/check.sh` (cpu-only and dev: build and `ctest -L cpu`, format, tidy, reuse, provenance, harness, docs, regen, python, api, scaffold, pre-commit) | passed (877 and 906 tests), after the REUSE fix `b0fd3b3` |
| `ctest -L cbst` in dev-cuda (cpu and gpu, the slow replays included) | 63 of 63 passed |
| the replays of the small sets at every commit, the large sets at `f2c48df` (the verbatim-port head) and `7e397a2` (the step's last code commit) | the merge-step table above |
| the sanitizers (memcheck, racecheck, synccheck, initcheck) | clean (criterion 16) |
| `transplant.py --check --rev 477d592` | identical |
| the Clang 18 syntax pass (dev: 222 files; dev-cuda: 304 files), the `_FORTIFY_SOURCE=3` cpu-only build | passed |
| compile with the ci-cuda12 preset (CUDA 12.9) and with CUDA 13.4 (ci-cuda13), release architectures, warnings as errors (the library and every CBST target) | passed |
| `ci/gpu_local.sh` (dev-cuda, GPU 1; at `7e397a2` with this section's docs uncommitted) | all 14 steps passed: build, `ctest -L gpu` (486), `ctest -L cpu` (906), the sssp and cycle_count CUDA goldens, memcheck on every GPU executable, synccheck (sssp, mosp, cycle_count), racecheck, the cbst sanitizers (racecheck, synccheck, initcheck), the cbst replays of the small sets, the plugin wheel, clang-tidy |
| both originals' HEAD and `git status --porcelain` | unchanged (MOSP_ESCHER keeps its pre-existing uncommitted files) |

## 12. Step merge-invariant: MOSP_ESCHER's behaviours, part 1 (criteria 10, 11, parts of 13 and 16; ADR 0039)

The behaviours of B that change no trace of A, merged one per commit (ADR 0039, section 2, cites
B's lines and lists every deviation from B's literal code). **How each commit is judged**
(`parity/cbst_merge_steps.sh <commit>`; Release CUDA build of the commit's replay targets, its own
`dyng_cbst_device_tests`, `compare.py cbst_trace --store device --fields exact` with the commit's
tool, GPU 1):

- **A's sets under `escher_gpu()`**: identical in the `exact` field set (criterion 10(a) and
  more: offsets, occupancy, tail capacity, `device_bytes()` = A's `deviceBytes()`, per-op
  degree-bin launches, the integrity verdicts), the one documented divergence of ADR 0038 apart.
- **B's sets under `escher_gpu()`** and, from `2bc0e24`, **under the staging preset
  `mosp_escher_staged()`** (B's behaviours merged so far, B's fixed capacity, no automatic
  compaction yet): every tree diverges at exactly the op and in exactly the field where the host
  model under the same preset does (criterion 10(b): every listed divergence is caused by a
  behaviour not merged yet). The cells list the first divergences by field.
- From `4ac07db` every construct is also built the other way (row counts / scan) and the node
  arrays must be equal byte for byte ("twin constructs"; criterion 11, caller counts).
- From `bd195c8` the replay compares, after every op, the store's three integrity checks with the
  originals': `check.rows` (A's and B's verdict), `check.subtree_avail` (B's; 0 for A's states)
  and `check.index` (dynG's; 0).

| commit | behaviour (B's lines) | expected effect on the traces | A: escher_gpu_test | A: escher_gpu_targeted | B: mosp_escher_test under escher_gpu / staging | B: mosp_escher_targeted under escher_gpu / staging |
|---|---|---|---|---|---|---|
| `2bc0e24` | harness: the staging preset `mosp_escher_staged()` (no store behaviour) | none; B's sets also run under the staging preset | 75 traces, 1,176 ops, 0 mismatches | 52 ops, 0 mismatches, 1 documented | 50 / 50 trees diverge as the model | 7 / 7 as the model |
| `1e01ad8` | harness: the rows list the first divergences by field | none | 1,176 ops, 0 | 52 ops, 0, 1 documented | as `e18912f` | as `e18912f` |
| `e18912f` | (a) construct validation (`operations.cu:196-211`) | none (every construct of the traces is valid; the construct below its payload was already rejected) | 1,176 ops, 0 | 52 ops, 0, 1 documented | escher_gpu: fixed_size 5, scratch_payload 34, key.occupancy 11; staging: scratch_payload 34, key.occupancy 11, status 5 | escher_gpu: fixed_size 1, scratch_payload 4, construct validation 1, status 1; staging: scratch_payload 4, construct validation 1, exception.type 1, status 1 |
| `a831fc4` | (b) the fill clamps (`payload.cu:123, 178, 241`) | none (no state of the traces has occupancy > tail capacity) | 1,176 ops, 0 | 52 ops, 0, 1 documented | as `e18912f` | as `e18912f` |
| `4ac07db` | (c) caller row counts (`operations.cu:123-132, 268-275`) | none; 240 + 12 (A) and 3,390 + 12 (B) twin constructs byte-identical | 1,176 ops, 0, 240 twins | 52 ops, 0, 1 documented, 12 twins | as `e18912f` (the 11 constructs without counts keep B's legacy occupancy 0: never merged) | as `e18912f` |
| `4402fcf` | ci: the provenance check knows B's identifiers | none | 1,176 ops, 0, 240 twins | 52 ops, 0, 1 documented, 12 twins | as `e18912f` | as `e18912f` |
| `d3af511` | (d) one bounded chain walk (`find.cu:9-20`) | none (the longest chain of any trace: 98 hops) | 1,176 ops, 0 | 52 ops, 0, 1 documented | as `e18912f` | as `e18912f` |
| `bd195c8` | (e) the integrity merge (`integrity.cu`, `:131-134`, `:218-243`) and dynG's index checks | none; every state passes all three checks as the originals' | 1,176 ops, 0 | 52 ops, 0, 1 documented | as `e18912f` | as `e18912f` |
| `6cc54ef` | (f) the `find.cu` and `delete_avail.cu` differences (`rows()`) | none | 1,176 ops, 0 | 52 ops, 0, 1 documented | as `e18912f` | as `e18912f` |
| `c981416` | (g) scratch growth (`operations.cu:139-165`) | A: none (exact); B under the staging preset: the 38 scratch divergences go | 1,176 ops, 0 | 52 ops, 0, 1 documented | escher_gpu: as before; **staging: 18 trees (key.occupancy 11, status 7), 10,620 ops compared** | escher_gpu: as before; **staging: 5 trees (construct validation 1, exception.type 1, status 3)** |
| `c05d7cc` | (h) an explicit `compact()` (`operations.cu:289-339`, `payload.cu:338-409`) | A: none (A has no `compact()`); B's traced `compact()` now replays under the staging preset, offsets included | 1,176 ops, 0 | 52 ops, 0, 1 documented | as `c981416` | **staging: 4 trees** (construct validation 1, exception.type 1, status 2) |
| `8bbb8eb` | (i) accounting and storage counters (`DynamicHypergraph.cpp:145-156`; not `operations.cu:57-69`) | none | 1,176 ops, 0 | 52 ops, 0, 1 documented | as `c981416` | as `c05d7cc` |
| `be449b6` | (j) dead or absent keys (dynG's rule) | none (no item of the traces names one) | 1,176 ops, 0 | 52 ops, 0, 1 documented | as `c981416` | as `c05d7cc` |

Every row is `ok`: the commit's own committed-trace replays passed, and every set passed its
judgement (rows in `$DYNG_SCRATCH/m8a/merge-steps/rows.md`, records in
`$DYNG_SCRATCH/m8a/merge-steps/<sha7>/`). "as `e18912f`" means the same trees diverge at the same
ops in the same fields. **The large sets at the step's last code commit `be449b6`**
(`parity/cbst_merge_steps.sh --large`): `escher_gpu_coauth50k` 15 ops, 0 mismatches, 3
unreplayable (the hash-only H2H rebuilds), 3 twin constructs; `escher_gpu_syn1m` 54 unreplayable
(batches mode); `mosp_escher_dblp50k_hyperedge` 27 ops and `mosp_escher_dblp50k_vertex` 21 ops:
0 trees diverge under `escher_gpu()` and under the staging preset, 3 twin constructs each. The
records of this run are `parity/results/M8a-device-<set>-exact[-mosp_escher_staged].json`
(replacing step 3's).

**B's divergences that remain under the staging preset** (step 5 removes the first two classes):
`test_cbst_ops` churn small / random / insert / growth and `test_hsosp_matches_dijkstra` h2v /
v2h / h2h at their first automatic compaction (`status`: the store throws `capacity_error` at B's
fixed capacity, B compacted); the targeted `fill_compact` and `insert_compact` (the same), the
targeted capacity throw (B compacts, then throws `escher::EscherError`; the store throws
`dyng::capacity_error` at the op: `exception.type`), the targeted construct below its payload
(the documented divergence of ADR 0038); and the 11 `test_cbst_smoke` constructs with
`rowOccupancy = nullptr` (B's legacy occupancy 0, `key.occupancy`), which dynG never takes over
(ADR 0039: step 5 must document them).

**The compact-after-every-op proof (behaviour (h)).** `dyng_cbst_compact_tests` (committed
traces, every GPU build) and `dyng_cbst_compact_large_tests` (the four small golden sets, slow):
the store under `escher_gpu()` and the host model (whose compaction is written independently of
the store, ADR 0037) replay every op of every trace in lockstep and both compact after every op
that succeeded. The four small sets, **measured at `c05d7cc`** (the test failed from `be449b6` on,
which step 4 did not re-run it after; fixed by `f03b3eb`, section 13): **15,385 ops, 15,383
compactions (4,881 reclaimed space),
41,315,718 live and 595,997 deleted rows checked after a compaction; 0 problems** (2 trees stop
early: the targeted constructs below their payload, the documented divergence of ADR 0038). After every op and every
compaction the two states are equal in every field (offsets included), and every compaction kept
the keys, avail, subtree_avail and every live row (in order) unchanged, left every live row one
segment of (count+4)/4*4 slots and every deleted row its length, emptied, with clean integrity
checks (a compaction padding changed to (count+8)/4*4 is caught on the committed traces).

**Unit tests** (`cpp/tests/hypergraph/escher/cbst_merge_test.cpp`, `dyng_cbst_store_tests`): the
validation; occupancy above the tail capacity on rows of every degree bin (clamped; memcheck
clean; without the clamps the neighbouring rows are corrupted); the row-count fast path (same
nodes, null and out-of-range counts rejected, wrong counts an `internal_error` in Debug builds);
2,000 overflowing appends on one key (2,000 hops) read back completely by the integrity checker,
the row locator, `rows()` and the unfill writer; a hand-made cyclic chain and a chain pointer past
the payload raise `internal_error` without a hang or a truncated row (remove_values with
`check_errors()`, the next fill's read point, the row locator, `rows()`, the host checker); B's
unfill-chain scenario; the integrity checks (a corrupted subtree_avail entry, key, row offset and
avail flag caught; unsorted expected rows no false failure); the scratch rules; churn-small (500
rounds in B's fixed 2,000 slots with explicit compactions); the compacted layout; compacted rows
that do not fit; the storage counters and B's accounting scope; no process-global mempool
setting; appends and removals on dead or absent keys at the thread, warp and block level.

**The checks of step 4** (2026-10-07; shared perf lock, niced; GPU 1):

| check | result |
|---|---|
| `ctest -L cbst -LE slow` in dev-cuda (Debug), before every commit | passed (82 tests at the end) |
| the merge-step replays of every commit (table above) | every row `ok` |
| `dyng_cbst_compact_large_tests` (Release) | passed at `c05d7cc` (above); failing from `be449b6` on, which step 4 did not re-run (found and fixed in step 5, `f03b3eb`; section 13) |
| compute-sanitizer memcheck on the touched tests before each commit; memcheck and racecheck on `dyng_cbst_store_tests` after (d) and (h); memcheck on `dyng_cbst_device_tests` and `dyng_cbst_compact_tests`; at the end synccheck and initcheck on `dyng_cbst_store_tests` and `dyng_cbst_compact_tests`, racecheck on `dyng_cbst_compact_tests` | 0 errors, 0 hazards |
| `ci/check.sh` with `DYNG_CHECK_ONLY="format reuse provenance precommit"` before every commit; the full `ci/check.sh` at the end | passed: every step of the full run (format, the cpu-only and dev builds with `ctest -L cpu`, 879 and 908 tests, tidy, reuse, provenance, regen, harness, python, api, scaffold, docs, parity) at the step's last code commit; the pre-commit end-of-file fixer corrected the then uncommitted `parity/results/M8a.md`, and its re-run with the final documents passed |
| the Clang 18 syntax pass (dev-cuda: 308 files), the `_FORTIFY_SOURCE=3` cpu-only build | passed |
| both originals' HEAD and `git status --porcelain` | unchanged (MOSP_ESCHER keeps its pre-existing uncommitted files) |

## 13. Step merge-policies: automatic compaction, the presets, scenarios, mutations (criteria 10-18; ADR 0040)

B's automatic compaction and its preset `mosp_escher()`, dynG's default `dyng_default()`, the
partial-failure rule, B's 9 scenarios and smoke test, the graph-on-CBST test, the 7 mutations and
the final readings (ADR 0040 cites B's lines and lists every deviation from B's literal code).
**How each commit is judged** (`parity/cbst_merge_steps.sh <commit>`, as in section 12):

- **A's sets under `escher_gpu()`**: identical in the `exact` field set (criterion 10(a)).
- **B's sets under `mosp_escher()`** (from `00319da`, the commit that completes it): identical in
  the `exact` field set (criterion 10(b) and more: InsertMapping, rows in order, avail,
  subtree_avail, length, offsets, occupancy, tail capacity, segments, the bump pointer, the
  capacity, scratch, D, the state before every op, the integrity verdicts; compactions inferred by
  the same rule on both sides; B's `escher::EscherError` mapped to the dynG exception of its throw
  site). The only documented divergences: `construct_without_row_counts` (the occupancy of the
  keys of B's 11 `test_cbst_smoke` constructs with `rowOccupancy = nullptr`, which criterion 11
  forbids porting, and B's own integrity verdict on exactly those keys; every other field and
  every later op compared; 31 ops) and `construct_below_payload` (B's targeted construct below its
  payload; B threw, as the store does; the tree has no other op).
- **B's sets under `escher_gpu()`**: judged against the host model under the same preset, tree by
  tree (39 trees of `mosp_escher_test` diverge, not 50: the smoke trees' occupancy rule applies to
  the model too, so they no longer diverge at their constructs).
- **Every set under `dyng_default()`** (from `9d599aa`; criterion 10(c)): the field set `policy`
  (ADR 0040, section 3): rows, avail and subtree_avail identical on all sets; InsertMapping
  identical up to the first policy difference; the policy differences listed below.

| commit | change (B's lines) | A: test / targeted (`escher_gpu`) | B: test / targeted (`mosp_escher`) | B under `escher_gpu` | every set under `dyng_default` |
|---|---|---|---|---|---|
| `00319da` | B's automatic compaction (`operations.cu:409-444`, `694-721`), `mosp_escher()` (`DynamicHypergraph.cpp:79-107`); the exception mapping; `construct_without_row_counts` | 1,176 ops, 0 / 52 ops, 0, 1 documented | **14,104 ops, 0 mismatches**, 31 documented; **53 ops, 0**, 1 documented; 48 + 3 compactions, 1 + 2 exceptions at B's ops | 39 / 7 trees diverge, as the model | (not yet) |
| `9d599aa` | `dyng_default()`, `max_payload`; the field set `policy`; the preset lockstep | as `00319da` | as `00319da` | as `00319da` | A test: 1,166 ops, 0 mismatches (65 `policy_difference`, 5 `policy_mapping`); A targeted: 52 ops, 0 (2 differences, 1 `construct_below_payload`); B test: 14,104 ops, 0 (1 `policy_exception`); B targeted: 53 ops, 0 (1 `policy_exception`, 1 `construct_below_payload`) |
| `78a8dd5` | partial-failure tests | as `00319da` | as `00319da` | as `00319da` | as `9d599aa` |
| `6e56612` | B's scenarios and smoke test | as `00319da` | as `00319da` | as `00319da` | as `9d599aa` |
| `33b0ac7` | the graph-on-CBST test | as `00319da` | as `00319da` | as `00319da` | as `9d599aa` |
| `c385ac8` | the 7 mutation hooks (off) | as `00319da` | as `00319da` | as `00319da` | as `9d599aa` |
| `043a6dd` | the early-warning tooling | as `00319da` | as `00319da` | as `00319da` | as `9d599aa` |
| `e7157af` | `ci/gpu_local.sh`, reduced sizes | as `00319da` | as `00319da` | as `00319da` | as `9d599aa` |
| `f03b3eb` | the compact lockstep's dead-key stop (test only; see below) | as `00319da` | as `00319da` | as `00319da` | as `9d599aa` |

"as `00319da`" means the same ops, mismatches (none) and documented divergences. **The large
sets** at `00319da` (the commit that completes `mosp_escher()`) and at the final code commit
`e7157af` (`f03b3eb` changes only `cbst_compact_test.cpp`, which no replay builds):
`escher_gpu_coauth50k` 15 ops, 0 mismatches, 3 unreplayable (the hash-only H2H rebuilds) under
`escher_gpu()` and under `dyng_default()`; `escher_gpu_syn1m` 54 unreplayable (batches mode);
`mosp_escher_dblp50k_hyperedge` 27 ops and `mosp_escher_dblp50k_vertex` 21 ops: 0 mismatches
under `mosp_escher()` and `dyng_default()`, 0 trees diverge under `escher_gpu()` (judged against
the model). The records of the final run are `parity/results/M8a-device-<set>-exact[-<preset>].json`
and `M8a-device-<set>-policy-dyng_default.json` (the step-4 records of the staging preset are
kept; their `head` is the main worktree's at the time, the `exe` the commit's build).

**dynG's default: where the policies differ** (criterion 10(c), from the records):

- `escher_gpu_test`: in 30 trees, at the op where A grows and the default compacts first (the
  bump pointer, the capacity and the compacted rows' lengths differ: `policy_difference`); in 5 of
  them (the H2V trees of A's medium graph `043-main`, `051-type1`, `059-type2`, `067-type3`,
  `075-coarseTriangle`) the insert at op 4, after the compaction at op 2, maps items to other
  slots (`policy_mapping`, the end of the comparison of those trees: 10 ops). Rows, avail and
  subtree_avail are identical everywhere else; no InsertMapping differs before a policy
  difference.
- `escher_gpu_targeted`: `fill_compact` (op 4) and `insert_compact` (op 3), where A grows.
- `mosp_escher_test`: `churn-growth` op 79, where B throws at its fixed capacity and the default
  grows (`policy_exception`); `mosp_escher_targeted`: the capacity throw (op 1), the same.
- The large sets: none (no growth or compaction happens on coauth50k or the DBLP runs).

**Scenarios and unit tests (criterion 13)** (`dyng_cbst_scenario_tests`, `dyng_cbst_policy_tests`,
`dyng_cbst_graph_tests`, `dyng_cbst_store_tests`; every op mirrored on the host model and compared
in every field): B's scale (n = 1..1100 and 65,535 .. 2^20), reuse, terminator, erase, surplus,
bestfit, unfill-chain, random (rows of 1..3,000 values: every fill, reuse and unfill kernel level
ran), churn (small, random, insert, growth) and `test_cbst_smoke` (with and without counts),
under `escher_gpu()`, `mosp_escher()` and `dyng_default()`: 39 tests, all passing; churn-growth
throws `capacity_error` only under `mosp_escher()` (with more than 245 live values in 256 slots)
and grows under the other two; the churn scenarios compact under the B presets and grow under
`escher_gpu()`. 40 random sequences per preset (2,400 / 601 / 2,400 ops; growths, compactions,
compactions then growths, compactions then exceptions) agree with the host model; permuted,
garbage and live requested keys give the same InsertMapping and state; batches of many ties match
the model. The graph-on-CBST agrees with `dyng::graph::apply` on sorted rows after every batch
under every preset, and B's `test_dynamicgraph_roundtrip` trace replays under `mosp_escher()`
with no mismatch. Earlier steps' tests cover the 0-record tree, the 2,000 overflowing appends, the
cyclic chain, the clamps, dead or absent keys and the clone.

**Presets and errors (criterion 12)** (`CbstStorePresets`, `CbstScenario.ChurnGrowth`): the default
compacts first and grows only when the compacted rows do not fit; `escher_gpu()` grows;
`mosp_escher()` compacts, then throws; `dyng_default(64)` raises `capacity_error` beyond 64 slots
and poisons the store; an injected allocation failure at every allocation of an insert (reuse and
surplus) and of an overflowing fill raises `out_of_memory_error` under every preset and poisons
the store, after which every operation throws until a construct.

**The compact-after-every-op lockstep** (`dyng_cbst_compact_large_tests`, the four small sets) had
failed since `be449b6`; with the dead-key stop of `f03b3eb`: 14,007 ops, 13,964 compactions
(4,105 reclaimed space), 34,933,621 live and 400,925 deleted rows checked after a compaction, 0
problems, 43 trees stopped at a failing op (41 at a fill or unfill naming a key that the lockstep's
state holds deleted or absent, confirmed in the model's state; 2 at the targeted constructs below
their payload).

**Mutations (criterion 17)** (`ctest -L mutation -R cbst`: a copy of libdyng with the mutated
hypergraph module against the committed-trace replays, the preset lockstep and the graph test):

| mutation | killed by |
|---|---|
| control (no mutation) | passes |
| `unfill_drops_last` (B's #4) | 10 failed tests (the replays of both originals, the lockstep, the graph test, B's roundtrip trace) |
| `graph_own_ids` (B's #5) | 6 (the graph-on-CBST test under every preset) |
| `reversed_ties` | 13 (the replays, the lockstep incl. the ties test, the roundtrip trace) |
| `surplus_rule1` | 12 (the replays, the lockstep) |
| `keep_dead_slots` | 9 (the replays, the lockstep) |
| `compaction_padding` | 4 (B's replays under `mosp_escher()`, the lockstep under the B presets) |
| `requested_keys` | 12 (the replays, the requested-keys test, the lockstep) |

**Criterion 15, the store memory** (GPU 0, exclusive perf lock, at `043a6dd`; records
`M8a-memory.json` and `M8a-memory-dyng_default.json`): every tree's `device_bytes()` after its
construct and after every op of the 3 batches of `escher_gpu_coauth50k` equals ESCHER-GPU's
recorded `deviceBytes()` under `escher_gpu()` (18 readings, ratio 1.0000, gate <= 1.05) and under
`dyng_default()` (the same 18 values: no compaction on this run, and the scratch never grows
past its size after the construct), and the store holds what it accounts (`allocated_bytes()`). The nsys peak of live device allocations (recorded,
not gated in M8): 3,018.9 MiB for A's whole `build/main`, 2,772.6 MiB for the store replay under
either preset (the pool reached 2,912 MiB).

**The early-warning table** (reported, not gated; `parity/perf_ab.py cbst early`; GPU 0, exclusive
lock; A = unpatched `build/main` of coauth50k, dynG = the store replay under `escher_gpu()`; per
op, the 3 ops of each class). **Superseded by section 14** (the M8a review): this first table was
one run per side at the default clocks, and its allocation column missed every allocation of the
dynG store (`cudaMallocFromPoolAsync`): dynG makes as many allocation calls as A (25 / 2 / 1 / 3
per op), all stream-ordered. The record `M8a-early-warning.json` is now section 14's.

| op | device time per op, A / dynG | ratio | blocking runtime calls per op | allocation calls per op |
|---|---:|---:|---:|---:|
| h2v erase | 0.100 / 0.100 ms | 1.00 | 0 / 0 | 2 / 0 |
| v2h unfill | 0.711 / 0.790 ms | **1.11** | 0 / 0 | 1 / 0 |
| h2v insert | 3.183 / 3.069 ms | 0.96 | 17 / 7 | 25 / 0 |
| v2h fill | 0.105 / 0.100 ms | 0.95 | 2 / 1 | 3 / 0 |

Registers per thread (CUPTI), A / dynG: `insert_node_thread` 32 / 32, `apply_reuse` 30 / 30,
`apply_reuse_warp` 28 / 28, `unfill_thread` 23 / 24, `locate_delete_targets`, `apply_deletes`,
`reduce_avail_level`, `fixup_overflow_metadata` 16 / 16, `allocate_space` 26 / 26 (the warp and
block fill and unfill kernels do not run on coauth50k). dynG allocates from the stream-ordered
pool (as many calls as A, none synchronous; the "0" above was a counting error, section 14); its
blocking calls are A's read points (ADR 0038). The 1.11x of v2h unfill (the bounded chain walk and the dead-key lookup of ADR 0039)
is the item to watch in M9, which gates `update`.

**Criterion 16, robustness.** compute-sanitizer memcheck (leak check), racecheck, synccheck and
initcheck: 0 errors, 0 hazards on `dyng_cbst_scenario_tests`, `dyng_cbst_policy_tests`,
`dyng_cbst_graph_tests` (reduced sizes, `DYNG_CBST_REDUCED=1`), `dyng_cbst_store_tests` and
`dyng_cbst_device_tests` (one replay of each original's committed traces); `ci/gpu_local.sh`
runs them (step `cbst`).

**The checks of step 5** (2026-10-08; shared perf lock, niced; GPU 1 unless stated):

| check | result |
|---|---|
| `ctest -L cbst -LE slow` (dev-cuda) and the fast `ci/check.sh` steps before every commit | passed |
| `ctest -L cbst` with the slow tests (dev-cuda) | 155 of 155 after `f03b3eb` (the large lockstep had failed before it, above) |
| the merge-step replays of every commit, the large sets at `00319da` and `e7157af` | every row `ok` (table above) |
| at the end: `parity/cbst_merge_steps.sh` over the whole merge-step table (the 34 commits of sections 11-13 from `477d592` to `e7157af`, small sets) with the final comparator, then `--large f03b3eb` (the head's code) | every row `ok`; every commit reproduces its recorded result (A's sets 1,176 + 52 ops with 0 mismatches at every commit; B's sets diverging tree by tree where the host model does up to `be449b6`; from `00319da` 14,104 + 53 ops with 0 mismatches under `mosp_escher()`); at `f03b3eb` every small and large set as in the table above |
| `ctest -L mutation -R cbst` | the control passes, all 7 mutations killed |
| the sanitizers | clean (criterion 16) |
| the full `ci/check.sh` (format, cpu-only and dev builds with `ctest -L cpu`: 881 and 910 tests, tidy, reuse, provenance, regen, harness, python, api, scaffold, docs, pre-commit) | passed at `e7157af`, and again at `96143fb` (the step's last commit before this record; 881 and 910 tests) |
| the Clang 18 syntax pass (dev: 222 files; dev-cuda: 455 files), the `_FORTIFY_SOURCE=3` cpu-only build | passed (after the `struct cbst_store` forward declaration of `e7157af`), and again at `96143fb` (222 and 455 files, 0 failures; the fortified build clean) |
| fresh clone of the branch at `f03b3eb` and again at `96143fb`: the cpu-only (with `ctest -L cbst`: 48 tests), dev-cuda and parity-cuda presets | built and passed |
| `ctest -L cbst -LE slow` (dev-cuda, GPU 1) and `ctest -L cbst` (cpu-only) at `96143fb` | 151 of 151 and 48 of 48 |
| `ci/gpu_local.sh` (dev-cuda, GPU 1; at `f03b3eb` with only the step's documents and records uncommitted; about 4 hours) | all 16 steps passed, among them the `cbst` sanitizers, the device replays of the small sets under the originals' presets and under `dyng_default()`, and the compact-after-every-op lockstep |
| both originals' HEAD and `git status --porcelain` | unchanged (MOSP_ESCHER keeps its pre-existing uncommitted files) |

## 14. The review fixes (2026-10-08)

Independent reviewers confirmed 26 problems at `2ee8bcc` (store correctness, parity honesty,
provenance, measurements); every one is fixed on the branch (`docs/developer/retrospectives/M8a.md`,
"The review fixes", lists each with its commit). This section records the evidence.

**The store** (`aa26772`, `9175297`, `e4ceaff`, `3961dc0`, `0feda27`; tests in
`cpp/tests/hypergraph/escher/cbst_contract_test.cpp`, 13 tests): an empty reconstruct drops the
earlier tree (find, rows and an append to a former key: the documented error, nothing written,
`allocated_bytes() == device_bytes()`); `rebuild_from_device` with 0 records under every preset,
then insert and append; the row locator on `max_key` 0 and `row_locator::build` (contiguous rows
with `out[]`, a chained row `internal_error`); six malformed batches rejected by insert, append and
remove_values (and the free functions) with the state unchanged and the store not poisoned; six
malformed construct layouts rejected before any allocation; every read of a poisoned store but
`download()` throws; a pending device error is copied by `clone()`; `clone()` onto the other GPU
and back under every preset (both GPUs of the machine); the counted Debug check of the row
counts; two stores on two non-blocking streams through churn that grows and compacts, under
`dyng_default()` and `escher_gpu()`, equal a sequential run; the hysteresis: on 2,000 rows with 5
values of net growth per round, the default compacts at rounds 22, 43, 136, 212, 286 and 354
(one growth), the same churn without the hysteresis at 22, 43, 60, 74, 85, 92, 97, 100, 102, 104,
106, 108, 109, 110, 111, 188, ..., and the host model predicts every growth and compaction.
compute-sanitizer memcheck (with leak check) of `dyng_cbst_store_tests`: 0 errors.

**The default preset's policy differences** (`8214aab`). A planted compaction-padding defect
(`(count + 3) / 4 * 4 + 1` slots, the review's) in `compact.cu`, rebuilt into the replay tool:

| replay under `dyng_default()`, field set `policy` | before the fix (review, at `2ee8bcc`) | after |
|---|---|---|
| `escher_gpu_test` | OK, 70 documented | OK, 70 documented (counts pinned: a defect that changes them fails) |
| `escher_gpu_targeted` | OK | 4 documented, pinned 3: **fails** |
| `mosp_escher_test` | OK, 2,401 documented (unmutated: 1) | **10,519 mismatches**, the first at op 197 of `test_cbst_ops` (both sides compacted) |
| `mosp_escher_targeted` | 5 mismatches | 8 mismatches |
| the policy lockstep on the committed traces | (did not exist) | **fails** (e.g. key 1 length: store 5, model 4) |

Without the defect: the four small sets under `dyng_default()` 1,166 / 52 / 14,104 / 53 ops
compared, 0 mismatches, documented `policy_difference` 65 + `policy_mapping` 5 / 2 +
`construct_below_payload` 1 / `policy_difference` 4 + `policy_exception` 1 /
`construct_below_payload` 1 + `policy_exception` 1 (the hysteresis adds the 4 differences of
`mosp_escher_test`: B compacted, the default compacted and grew); the large sets 0 documented.
Every count is pinned in `parity/cbst_documented.toml`. The policy lockstep with the host model
(`CbstPolicyLockstepLarge`, the four small sets, Release, GPU 1): `escher_gpu()` 14,319 ops, 133
growths; `mosp_escher()` 15,237 ops (78 trees stop where the model cannot predict the original:
A's rebuilds, which B lacks); `dyng_default()` 15,385 ops, 110 growths; 0 problems in each.

**B's sets judged against the model** (`301dffc`): at the head, `mosp_escher_test` and
`mosp_escher_targeted` under `escher_gpu()` agree with the host model in the whole first
divergence of every tree (10,185 and 43 ops compared); the DBLP sets 0 trees diverge.

**The merge-step table, replayed again** with the whole-first-divergence comparison
(`parity/cbst_merge_steps.sh --out <dir>`; records and rows under
`$DYNG_SCRATCH/m8a/merge-steps/run-review` for `477d592` .. `7e397a2`, `run-review2` for
`2bc0e24` .. `be449b6`, `00319da` .. `f03b3eb`, `2ee8bcc` and the code commits of the fixes
`aa26772` .. `8214aab`, and `run-review2-head` for `--large` at `9acb3b5`). The first run
(`run-review`) passed every commit up to `7e397a2` and failed every commit from `2bc0e24` on in B's
sets under the staging preset; it was interrupted at `d3af511`. Diagnosed with the commits' own
replay tools: in every failing tree the store and the host model under `mosp_escher_staged()`
diverge at the same op with every compared field, key and value equal, and differ only in the
wording of the exception message (`capacity_error` at B's fixed capacity, where B compacted), in the
InsertMapping of a thrown insert (which neither returns) and, at `c981416` only, in the wording of
"this preset has no `compact()`" (merged one commit later, `c05d7cc`). The table's divergences were
therefore described correctly; `32cf67d` and `4b1c519` compare exactly those non-values as one
status (retrospective deviation 63), every other value as before. The second run: **every row
`ok`**, 30 commits (`2bc0e24` .. `be449b6`, 13; `00319da` .. `f03b3eb`, 9; `2ee8bcc`; the 6 code
commits of the fixes; and `9acb3b5`; `c981416` ran twice, its first row `FAILED` as above, its
second, after `4b1c519`, `ok`), each cell as the table of its section says (the same trees, ops, fields and documented
divergences; `mosp_escher_test` under `dyng_default()` documents 5 divergences from `3961dc0` on,
the hysteresis's 4 `policy_difference` and B's 1 `policy_exception`, 1 before). **At the head
`9acb3b5`, every set** (`--large`, the pinned counts of `parity/cbst_documented.toml` checked):
`escher_gpu_test` 1,176 ops, 0 mismatches (`dyng_default()`: 1,166 ops, 0, 65 `policy_difference`
+ 5 `policy_mapping`); `escher_gpu_targeted` 52 ops, 0, 1 `construct_below_payload`
(`dyng_default()`: + 2 `policy_difference`); `mosp_escher_test` under `mosp_escher()` 14,104 ops, 0,
31 `construct_without_row_counts` (`dyng_default()`: 14,104 ops, 0, 4 `policy_difference` + 1
`policy_exception`; `escher_gpu()`: 39 trees diverge as the model, 10,185 ops compared);
`mosp_escher_targeted` 53 ops, 0, 1 `construct_below_payload` (`dyng_default()`: + 1
`policy_exception`; `escher_gpu()`: 7 trees as the model); `escher_gpu_coauth50k` 15 ops, 0, 3
unreplayable (hash-only H2H rebuilds); `escher_gpu_syn1m` 54 unreplayable (batches mode);
`mosp_escher_dblp50k_hyperedge` 27 ops and `_vertex` 21 ops, 0 under `mosp_escher()` and
`dyng_default()`, 0 trees diverge under `escher_gpu()`; 3,390 + 12 + 240 + 12 + 3 + 3 + 3 twin
constructs byte-identical.

**The certificate and `bin/main`** (`16a7251`): `ci/gpu_local.sh` writes
`parity-cbst_trace_<set>-device-<preset>.json` for the small sets under both presets, and
`certify.py` fails a certificate without them; `cbst_trace_goldens.py once main` recorded
MOSP_ESCHER's `bin/main` (111 CBST ops of a fresh random graph) and replayed it under
`mosp_escher()`: 111 ops compared, 0 mismatches, nothing documented.

**The records** (`b912ff5`): the replay records `M8a-device-*.json` of the current presets are
those of the `--large` run at `9acb3b5` (the tool named relative to `$DYNG_SCRATCH`); the staging
preset's records of the small sets are `be449b6`'s from `run-review2` (the whole-first-divergence
comparison), those of the DBLP sets keep `be449b6`'s results (0 trees diverge, which the stricter
comparison cannot change) with the tool path made portable. A test checks that no record names a
personal path.

**Criterion 15, again** (GPU 0, exclusive perf lock, the store at its final code, `a6d3a03`; the
records' "+dirty" is only this section's uncommitted documents and records): every tree's
`device_bytes()` after its construct and after every op of the 3 batches equals ESCHER-GPU's
`deviceBytes()`, under `escher_gpu()` and `dyng_default()` (18 readings each, ratio 1.0000; no
compaction happens on coauth50k, so the hysteresis never triggers); nsys peaks 3,018.9 MiB (A) and
2,772.6 MiB (the store replay, pool 2,912 MiB), as before.

**The early-warning table, re-measured** (`parity/perf_ab.py cbst early --runs 20`, record
`M8a-early-warning.json`, schema 2; GPU 0, exclusive lock, clocks locked by `perf_ab.ClockLock`
at SM 1,695 MHz / memory 7,601 MHz; 20 runs per side, alternating; device time per op, median
[min, max] of the 20 runs; calls per op, A / dynG):

| op | A | dynG | ratio | blocking | allocations (synchronous) | frees |
|---|---:|---:|---:|---:|---:|---:|
| h2v insert | 3.249 [3.226, 3.281] ms | 3.212 [3.206, 3.219] ms | 0.99 | 17 / 7 | 25 (10) / 25 (0) | 25 / 25 |
| h2v erase | 0.102 [0.101, 0.103] ms | 0.103 [0.102, 0.103] ms | 1.00 | 0 / 0 | 2 (0) / 2 (0) | 2 / 2 |
| v2h unfill | 0.744 [0.735, 0.753] ms | 0.847 [0.846, 0.849] ms | **1.14** | 0 / 0 | 1 (0) / 1 (0) | 1 / 1 |
| v2h fill | 0.110 [0.109, 0.111] ms | 0.103 [0.103, 0.104] ms | 0.94 | 2 / 1 | 3 (1) / 3 (0) | 3 / 3 |

Every hot kernel, from `cuobjdump -res-usage` and `-sass` of the sm_86 code of both binaries
(registers, SASS instructions; theoretical occupancy at 256 threads per block is 1.0 for all of
them, A and dynG): `insert_node` thread 32, 376 / 32, 416; warp 28, 160 / 28, 192; block 16, 112
/ 16, 136; `apply_reuse` thread, warp and block equal (30, 144; 28, 152; 22, 72);
**`unfill` thread 23, 192 / 24, 376; warp 23, 352 / 25, 552; block 36, 568 / 36, 752**; the
delete kernels, `reduce_avail_level`, `allocate_space` and `fixup_overflow_metadata` equal. The
unfill kernels grew by the bounded chain walk and the dead-key rule (ADR 0039 (d) and section 5):
v2h unfill is 1.14x of A's device time at locked clocks (the first single run at default clocks
showed 1.11x); about 0.1 ms per op, the item to watch in M9, which gates `update`. Every other
class is at or below A. The allocation counts are corrected: dynG makes as many allocation calls
as A per op, all stream-ordered (`cudaMallocFromPoolAsync`), none of A's 10 synchronous
`cudaMalloc` per insert.

**The checks of the fixes** (2026-10-08; shared perf lock, niced; GPU 1 unless stated):

| check | result |
|---|---|
| `ctest -L cbst -LE slow` (dev-cuda) on the final tree | 167 of 167 passed (`CbstContractClone.OntoAnotherDevice` skips with one visible GPU; with both GPUs visible: passed) |
| every store commit of the fixes built in a temporary worktree (Release; `dyng_cbst_store_tests`, `_host_tests`, `_policy_tests`, `_compact_tests`, `_device_tests`) | all passed at each: `aa26772` 38 / 6 / 9 / 1 / 5 tests, `9175297` 40 / 6 / 9 / 1 / 5, `e4ceaff` 43 / 6 / 9 / 1 / 5, `3961dc0` 45 / 6 / 9 / 1 / 5, `0feda27` 47 / 7 / 9 / 1 / 5, `8214aab` 47 / 7 / 9 / 2 / 5 (one GPU visible: the two-GPU clone test skips) |
| compute-sanitizer memcheck (leak check), racecheck, synccheck, initcheck on `dyng_cbst_store_tests` and `dyng_cbst_compact_tests` (the final store code) | 0 errors, 0 hazards, 0 leaks; 47 + 2 tests passed under each tool |
| the Clang 18 syntax pass (dev: 239 files; dev-cuda: 456 files), the `_FORTIFY_SOURCE=3` cpu-only build | 0 failures; the fortified build clean |
| `ci/check.sh` (every step: cpu-only and dev builds with `ctest -L cpu`, format, tidy, reuse, provenance, harness, docs, regen, python, scaffold, pre-commit) | the full run failed `harness` (the replay records not yet regenerated, `b912ff5`) and `docs` (the API baseline of `compaction_headroom`, `f0eb0ca`); both passed again after those commits, every other step passed |
| `ci/gpu_local.sh` with `DYNG_GPU_SKIP="gpu cpu parity memcheck synccheck racecheck plugin tidy"` (the build and the changed `cbst` step; dev-cuda, GPU 1, at `299aa97`; about 1.5 hours) | all steps passed: the cbst sanitizers, the device replays of the small sets under the originals' presets and under `dyng_default()` with their certificate records written, MOSP_ESCHER's `bin/main` recorded and replayed once (108 ops this time, 0 mismatches), the compact-after-every-op lockstep (14,007 ops) and the policy lockstep (14,319 / 15,237 / 15,385 ops) |
| the six originals' HEAD and `git status --porcelain` before and after the fixes' runs | unchanged; MOSP_ESCHER keeps exactly its pre-existing uncommitted files of step 1 |

## 15. The acceptance fix (2026-10-13)

The acceptance verifier failed criterion 4 at `953fdbf`: the manifest of
`cbst_trace_mosp_escher_test` was reproducible only by chance. Its traces are deterministic, but
the set's `info.json` (hashed into `MANIFEST.sha256`) recorded the op counts of the excluded
`bin/main`, whose graph comes from `std::random_device` (108 or 111 ops; the verifier saw 111 in 2
of 6 runs, and a third export with 111 gave manifest `b5a4bfa2...` instead of the pinned
`fe2d29ed...`).

**The fix** (`c210ed3`): the `excluded` record of `info.json` holds the invocation and the reason
only (`cbst_trace_goldens.py`, `excluded_entry`); a test shows that two runs with different op
counts give the same record (`test_excluded_invocations_record_nothing_run_dependent`). One run of
`bin/main` is still recorded and replayed by `cbst_trace_goldens.py once main` (`ci/gpu_local.sh`).
ADR 0036 "Later changes" and retrospective deviation 1 say so.

**The re-pin** (`5e8555b`; shared perf lock, niced, GPU 1; logs under
`$DYNG_SCRATCH/runs/m8a-fix-mosp-test-export.log` and `$DYNG_SCRATCH/m8a/accept-fix`):

| check | result |
|---|---|
| `parity/export_goldens.py cbst_trace cbst_trace_mosp_escher_test --twice` (the scratch reference, then a fresh `git archive` copy) | identical manifests, `ed6400698e3d2ea9430139706d96f4e464966785e5d2aee2557434859a78ff60`; 24 cases, 58 files, 265,455,682 bytes; all 24 case digests equal to the earlier pins (only `info.json` changed) |
| three more exports from a third scratch (`build_reference.sh --fresh`, then `export_goldens.py ... --no-toml`), each followed by one traced `bin/main` (`cbst_trace_goldens.py once main`) | all three manifests `ed6400698e3d...`, while `bin/main` recorded 108, 111 and 111 ops (each replayed under `mosp_escher()`, 0 mismatches): the manifest no longer depends on the random graph |
| the set's replay records, re-run against the new manifest | host model 14,104 ops, 0 mismatches; device under `mosp_escher()` 14,104 ops, 0 mismatches, 31 `construct_without_row_counts`; under `escher_gpu()` 39 trees diverge as the model; under `dyng_default()` 0 mismatches, 5 documented (`96193bf`; every result equal to the earlier record) |
| `parity/cbst_merge_steps.sh be449b6` (the staging preset's record) | `ok`, the row equal to the table's: `mosp_escher_test` (`mosp_escher_staged`) 10,620 ops compared, 18 trees diverge where the host model does |
| `pytest parity/tests` | 114 passed |
| the originals' HEAD and `git status --porcelain=v1 --untracked-files=all` before and after | unchanged (MOSP_ESCHER keeps its pre-existing uncommitted files) |
