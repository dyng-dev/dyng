# Retrospective: M8a (the ESCHER CBST store)

Status: **in progress** on the branch `m8a-escher-store` (from `main` at `8b9067e`, which
contains M7 and M6b). M8a is the first half of PLAN 11.3's M8 row: the internal ESCHER store,
proven against both originals by CBST trace replay. M8b (the hypergraph container, its I/O and
Python, the header freeze) follows on the accepted store; both are in 0.3.0 with M9. Each step
adds its section; the milestone's acceptance table and the open items close the record.

**Planning change (not an author decision; reported to the author on 2026-10-07).** M8 runs as
M8a then M8b, both in 0.3.0 (0.3.0 = M8a + M8b + M9; PLAN Appendix F unchanged in content), as
M1a/M1b, M2a/M2b and M6a/M6b did: the store port and the staged merge are internal and fully
checkable by trace replay, and the public hypergraph headers are frozen only after the
container, the line graph, the update path, the I/O and Python exist. `docs/developer/plan.md`
and `docs/roadmap.md` show the split; together M8a and M8b carry every item of PLAN 11.3's M8
row (the old single row omitted the row views, the `sorted_merge` line graph, the hypergraph
generators, the CBST host model and the header freeze).

**The author's decisions for M8 (2026-10-07; GOVERNANCE.md approvals log).** (1) The memory gate
is read as store accounting: dynG's store device bytes against ESCHER-GPU's own
`DynamicHypergraph::deviceBytes()` on coauth-DBLP after the build and after each of 3 x 50K
batches (seed 1); the Nsight Systems peak of live allocations of both processes is recorded,
not gated, in M8 (it is gated in M9). (2) O16: the hypergraph `id_reuse` default is
`erase_first`; the H-SOSP parity runs set `insert_first`.

## Step 1: references, recorders, traces (refs-traces)

### What was built

| Commit | What |
|---|---|
| `28de27b` | `parity/references.toml` pins ESCHER-GPU@abcf9b2 and MOSP_ESCHER@4b86159; `build_reference.sh` keeps the original's HEAD and `git status` before and after every build as files and requires them unchanged, checks them around `--test` too, and reports whether the build needs a gitlink |
| `71ca715` | the CBST trace recorders as link-time twins (`parity/exporters/escher`, the export patches of both originals): the originals' unchanged objects relinked with `-Xlinker --wrap`; the nm census; the twin output check |
| `f3089fa` | the readers of trace format 1: `cpp/tests/parity/escher/cbst_trace.{hpp,cpp}` (C++, host-only) and `parity/escher_trace.py` |
| `81a8897` | `parity/timed_regions/hypergraph.toml` and its strict-loader tests, before any port commit |
| `1a0f4cc` | smaller traces: integrity reports only on failure, `batches` mode hashes the ops' inputs |
| `cdd2ed7` | the cross-check drivers of the two cores (`cbst_crossreplay`), the targeted scenarios (`cbst_targeted`), the fast verifier |
| `f0d8920` | `datasets/registry.toml`: the coauth-DBLP archive and both originals' variants |
| `c418064`, `a040c20` | the fixes of the resume (below): output-option files are not inputs; the summary skips hash-only inputs; the comparator sees compactions |
| `40cc358` | the golden sets (`export_goldens.py cbst_trace`), their manifests in `goldens.toml`, the committed subset and the reader, census, dispatch and comparator tests |
| `f320cd6` | ADR 0036, `parity/results/M8a.md` with the census, twin and cross-check records, this retrospective, the M8a / M8b split in `docs/developer/plan.md` and `docs/roadmap.md`, GOVERNANCE.md, CHANGELOG |
| (the certify commit) | `certify.py` lists the `cbst_trace` sets without gating them before the store's replay records exist (deviation 7) |

### Deviations from the plan and the step's text (each with the same intent)

1. **MOSP_ESCHER's `bin/main` is not in `cbst_trace/mosp_escher_test`.** It draws its graph from
   `std::random_device` (`mosp/src/generateGraph.cu:89-90`), so two runs of the unpatched copy
   print different numbers and record different traces; no set that contains it can be exported
   twice with identical manifests. It is recorded in every export and listed in the set's
   `info.json` under `excluded` with its op counts; its CBST calls are `DynamicGraph`'s, which
   the seeded `stressTest`, `parallelStressTest` and `test_dynamicgraph_roundtrip` make too (all
   in the set). The twin output check compares it on its exit status and verdict lines.
2. **Section names.** The sets are named `cbst_trace/<set>` in the milestone text and live in
   `$DYNG_SCRATCH/goldens/cbst_trace/<set>`; their `goldens.toml` sections are
   `[sets.cbst_trace_<set>]`, because the goldens.toml writers split sections by a name without
   `/`. The writers now accept digits in set names (`coauth50k`).
3. **Digest modes.** "Full mode for small runs" is applied per tree: in the default `auto` mode a
   tree with more than 4,096 keys or row values gets delta digests (the changed and named keys in
   full, the 64-bit hash of every other key's record). Delta digests are lossless (both readers
   rebuild every state and check every hash); the test suites' 100K-hyperedge runs would
   otherwise need several GB per trace.
4. **Targeted sets.** `cbst_trace/escher_gpu_targeted` and `cbst_trace/mosp_escher_targeted`
   (`parity/exporters/escher/cbst_targeted.cpp`) drive each original's unchanged
   `CBSTOperations` API through the paths the suites miss or that are too large to commit (the
   coverage table of `parity/results/M8a.md`).
5. **ADR number.** The trace format and recorder ADR is 0036: the 0.2.0 release branch took 0035.
6. **The cross-check of A's line graph.** A rebuilds its H2H (and V2V) from the device every
   batch (`rebuildFromDevice`), which B's store has no call for: A's H2H and V2V traces are
   compared through B's store up to their first rebuild (`rebuild_only_in_a`); the step's
   minimum (H2V and V2H of A, and B's h2h through A) is fully replayed. A's coauth H2H inputs are
   hash-only (the 10 GB budget), so they could not be replayed anyway.
7. **The trace sets in the release certificate.** `certify.py` lists the `cbst_trace` sets
   (kind `cbst_trace`) but keeps a set without a replay record out of the golden verdict: the
   store that replays them does not exist before step 2. A replay record of one is checked as
   usual; the store's replay writes them later (`parity/results/M8a.md`, section 8).
8. **Subset size.** The committed subset is 307,170 bytes ("at most about 300 KB"); the
   repository's test data stays at 470 KB, under 1 MB.

### The interruption and the resume (2026-10-07)

The step was interrupted by a usage limit after its first full chain of checks. Two exports of that
chain had failed and were fixed on resume, each by a recorder- or reader-side change, never by
relaxing a comparison: `cbst_trace_mosp_escher_test` differed between its two exports because the
recorder took the value of `hsospBench --out` (a CSV an earlier run had appended to) as an input
file; `cbst_trace_escher_gpu_syn1m` failed in the summary, which read hash-only inputs. Both sets
were exported twice again with identical manifests; the sets that passed were not re-exported
(no other invocation names an existing output file, so their traces are unchanged by the
recorder fix). The cross-checks that read the changed set, and A through A, were re-run with the
final comparator: 0 unexplained divergences.

### Measurements

The goldens take 6.75 GB of zstd data (limit 10 GB; largest: B's DBLP hyperedge run, 2.75 GB);
the full chain (fresh builds and suites, census, twin checks, datasets, 8 double exports,
cross-checks) runs in about 80 minutes on GPU 1. See `parity/results/M8a.md` (the originals' suites, the census, the twin check, the golden sets
and their sizes, the dataset pins, the cross-check, the coverage table).

### Notes for step 2 (the host model)

- Read the traces through `dyng::test::cbst_trace::reader` and `state_tracker`
  (`cpp/tests/parity/escher/cbst_trace.hpp`): the tracker already rebuilds every state and
  checks the recorded hashes; the comparator compares a store's state with the tracker's.
- Large sets are zstd files: decompress to a temporary file (`zstd -dc --long=31`) or pipe into
  the reader's `std::istream` constructor; the side file (`trace.blob.zst`) is only needed for
  construct and rebuild inputs.
- Hash-only arrays (`DYNG_CBST_TRACE_HASH_ONLY`; listed per set in `goldens.toml`) cannot be
  replayed: a tree whose rebuild or construct input is hash-only is checked by its hashes only.
