# M8a parity record: the ESCHER CBST store

Date: 2026-10-07 onwards. M8a ports the CBST store of ESCHER-GPU@abcf9b2 (A) and merges the
behaviours of MOSP_ESCHER@4b86159 (B) into it, proven by CBST trace replay against both
originals (PLAN 6.4.5). Each step adds its section; step 1 (refs-traces) writes sections 1-8.

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
| `cbst_trace_mosp_escher_test` | 24 | every CBST-calling invocation of B's `make test` but `bin/main` (retrospective, deviation 1): `test_cbst_smoke`, the 9 `test_cbst_ops` scenarios, `test_dynamicgraph_roundtrip`, `test_snapshot_matches_updateCSR`, the `test_h2h_*` tests, `test_hsosp_matches_dijkstra` (300-batch churn), `test_hsosp_scale`, `hsospStress` (both), `hsospBench` (smoke, real hyperedge / vertex, cli), `test_mosp_update`, `stressTest`, `parallelStressTest` (both) | auto | none | 265.5 MB | `fe2d29edce2a...` |
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
