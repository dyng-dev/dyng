# ADR 0081: The H-SOSP exporter and the hyper_sssp golden kind

- **Status:** Accepted under delegation (2026-10-28; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. It carries out M11a step 1 (PLAN 6.3 step 1, 6.4.8, 8.3; the M11a
  acceptance criteria 2 and 3). It changes no rule the author approved: the export patch stays
  additive (ADR 0013), the goldens come from the patched copy and the timings from the unpatched
  one (PLAN 8.3), the parity rules and the coverage are the author's (HD2, HD5, HD7 and HD18,
  decided by the author on 2026-10-28; ADR 0080).
- **Date:** 2026-10-28
- **Deciders:** the AI assistant, on the author's behalf (M11a, step
  regions-exporter-goldens-oracle)
- **Builds on:** ADR 0013 (parity and goldens), ADR 0036 (the CBST trace format and the
  link-time recorder, whose twins this extends), ADR 0080 (the parity modes and the canonical
  encodings).

## Context

PLAN 6.4.8 asks for `hyper_sssp`'s parity by id against MOSP_ESCHER@4b86159 (B), from exported
`HgBatch` files (B's generator reads the current state, so the batches must be exported, not
regenerated; HD5), and for the line-graph delta pairs, `newHe` / `deadHe`, `skippedOps` and
`h2h_pairs`. The CBST recording twins of M8a (ADR 0036) already dump B's batches
(`bulkLoad`, `beginBatch`, `finishBatch`), but none of the H-SOSP side: the device delta, the
counters of `hsosp::applyBatch`, the update's and the recompute's distances and parents, the
capacities. B has no hook for them, and ADR 0013 allows no change to a tracked file.

## Decision

### 1. The recorder: more wrapped symbols in the same twins

`parity/export_patches/MOSP_ESCHER/build.sh` passes `parity/exporters/escher/relink_twins.sh` a
new `--extra` group: the dump writer and its adapter (`parity/exporters/hsosp/hsosp_dump.{hpp,cpp}`,
`recorder_mosp_escher_hsosp.cpp`) and the wrap list `wrap_hsosp.txt`, linked into every twin whose
link command has `hsosp/src/hsospDelta.cu.o` (hsospBench, hsospStress, test_h2h_construction,
test_h2h_delta, test_hsosp_matches_dijkstra, test_hsosp_scale). The six symbols, each defined in
one object and referenced by others (`parity/hsosp_goldens.py census`; record
`parity/results/M11a-hsosp-census.json`):

| Symbol | Records |
|---|---|
| `DynamicHypergraph(int, const Caps&)` (C1) | the session: numVertices and Caps (private after construction) |
| `hsosp::buildDeviceH2H` | maxNodes, the initial CSR entry headroom, the device capacities |
| `hsosp::applyBatch` | at its exit (outside B's stage timers): the headroom argument; the deleted and inserted pairs decoded from `dev.delta`'s pinned sorted keys (row < col, bit 32 = insert); numDel / numIns / numNew / numDead; the `BatchTimes` counters (candidates, insPairs, delPairs, newHe, deadHe, skippedOps, csrRebuilt, mirrorReuploads); `host().h2hPairCount`; the post-batch rows (the batch's touched hyperedges always; every hyperedge with `DYNG_HSOSP_DUMP_ROWS=all`) |
| `hsosp::hsospUpdate`, `hsosp::hsospRecompute` | at their exit: `UpdateConfig` (maxIterations, workBudget, blockSize), the source id, the state (by first use), `UpdateStats` (the fallback and the seed count; iterations, work, frontier and jump rounds logged only), the distances and the 0-based parents (B's own download functions) |
| `hsosp::compareDistances` | the mismatch count |

The batch hooks of `recorder_mosp_escher_hooks.cpp` (`bulkLoad`, `beginBatch`) feed the dump
too, through weak functions that only the twins with the dump define: the loaded rows as given,
the `HgBatch` in B's op order, `final_ids`, `newW`, `dead_he`, `skipped_ops`, `max_id`, and the
incidence changes (`incKey` / `incSign`, the touched hyperedges and vertices with the vertices'
post lengths: the reference for the applied record of M11a step 2).

**Not wrapped**, recorded here: the update's own fallback recompute (`hsospUpdate` calls
`hsospRecompute` inside `hsospDevice.cu`, one translation unit): the fallback shows in the
update's stats and in its result; applyBatch's CSR rebuild after a tail overflow
(`detail::buildCsr` from `hsospDelta.cu`): it shows in `csrRebuilt`. The census also lists the
lower-level entries the S tests call directly: `DynamicHypergraph::beginBatch`
(test_hsosp_matches_dijkstra's refusal cases; recorded by the beginBatch hook) and the read-only
checks `downloadRows`, `incidenceMirrorMismatches`, `checkEscher` and the `HsospState` downloads.

**Off means off.** The dump records only when `DYNG_HSOSP_DUMP_DIR` is set (one `getenv` at the
first call), into its own files; the CBST trace stays off unless `DYNG_CBST_TRACE_DIR` is also
set, and its records are unchanged by the dump (either, both or neither can be on). With both
unset the twins equal the originals: `parity/exporters/escher/check_twins.py` now compares stdout
and stderr apart (timing lines left out) and the CSV files an invocation writes, hsospBench's on
its deterministic columns (`parity/hsosp_dump.py CSV_RUN_VARYING` lists the others: the
timestamp, the times, the speedup, the device memory in use, the logged-only counters), and drops
`DYNG_HSOSP_*` from the replay environment. **Additivity:** `cbst_trace_mosp_escher_test`
re-exported through the new twins with `DYNG_CBST_TRACE_DIR` set has a `MANIFEST.sha256`
identical to its golden's (`parity/hsosp_goldens.py additivity`). `build_reference.sh` checks the
original's HEAD and `git status --porcelain` before and after every build.

### 2. The dump format (version 1)

One file per process, `<DYNG_HSOSP_DUMP_DIR>/NNNNNN.hsd` (numbered in the order of first use):
`DYNGHSD1`, u32 version 1, u32 0, then records (u32 tag, u64 payload length, payload). The payload
is ADR 0036's field list (int64, string, inline int32 / int64 arrays, string lists). Records:
`header` (writer version, repo, commit, the rows mode, argv with absolute paths as `<abs>/<name>`,
the input files with FNV-1a and size, files up to 16 MiB copied to `inputs/`, an environment
subset), `session`, `load` (enter: the rows as `rows.offsets` / `rows.values`, the weights, the
source and target rows; exit: the host counts), `device_build`, `begin_batch` (enter / exit),
`apply_batch`, `sssp`, `compare`, `end`. Every record has `seq` (the global order) and the
session it belongs to. Readers: `parity/hsosp_dump.py` (Python) and
`cpp/tests/parity/hsosp/hsosp_dump.{hpp,cpp}` (C++, host only).

### 3. The golden kind `hyper_sssp` and the set `hyper_sssp_mosp_escher`

`parity/hsosp_goldens.py export|verify|manifest` (also `parity/export_goldens.py hyper_sssp` and
`parity/compare.py hyper_sssp`, whose `--part delta|results` names what a replay compares once
the port exists) writes `$DYNG_SCRATCH/goldens/hyper_sssp/mosp_escher`:

- **T** (`test/<NN>-<name>`): the nine H-SOSP invocations of B's `make test`, every session
  whole, every post-batch row and every array; `random.hg` (run_tests.sh's awk srand(7) file) is
  kept under `inputs/` with its SHA-256.
- **P** (`paper/<dblp|geology>_<kind>_<size>`): `hsospBench --hg <file> --kind <kind> --batch
  <size> --batches 3 --del 50 --verify none` at B's defaults; P50 (the four 50K configurations)
  with every post-batch row and every array; the other twelve with the touched rows, the batches
  in full, and the pairs, distances and parents replaced by the SHA-256 of their canonical
  encodings plus their counts (digest mode). The loaded rows of a dataset (identical in its eight
  configurations: B's loader is seeded) are stored once, `loads/<dataset>.hsd.zst`, and every
  paper case holds their digests.
- **Nothing run-varying is hashed:** the timings are not recorded; `UpdateStats` iterations,
  work, max_frontier, fallback iterations and jump rounds go to `<set>.logged/`, outside the set;
  the CSV keeps its deterministic columns; the stderr keeps its `[load]` / `[warn]` lines with the
  times replaced. The device bytes and payload offsets the dump holds are capacities that B plans
  deterministically; the double export shows it.
- **case.json** per case: argv, inputs, and per session the capacities, the load (rows, pins, the
  digests, the source and target rows, the h2h pairs), per batch the op counts, the batch digest,
  `final_ids`, `dead_he`, `skipped_ops`, the counters, the pairs digest and counts, `h2hPairCount`,
  and every update and recompute with its configuration, fallback decision (tabulated, not a
  parity target: HD7), seed count and the digests of its canonical distances and parents.
- **Export twice:** `export --twice <scratch2>` builds a FRESH patched copy (the first export
  uses another fresh copy) and requires byte-identical `MANIFEST.sha256` files.
- **verify** checks the manifest against `parity/goldens.toml`, every file against the manifest
  (and every file listed), every case digest, the case list (T complete, all sixteen P); `--full`
  recomputes every case summary from its dump (the digests of every array a dump holds, the
  fallback decisions, the pair counts), P50 stored in full, and the loads.
- **Cross-check:** the `bulk_load` and `begin_batch` records equal those of the CBST trace sets
  `cbst_trace_mosp_escher_{test, dblp50k_hyperedge, dblp50k_vertex}` where the cases overlap
  (`parity/hsosp_goldens.py cbst`).
- **Coverage** (HD2) and the parity modes per backend are recorded with the set in
  `parity/goldens.toml` (`[sets.hyper_sssp_mosp_escher.coverage]` and `.modes`).
- **The committed subset** (HD18, at most 100 KB of M11's 1.25 MB budget share): the first
  sessions of test_hsosp_matches_dijkstra, up to the first with a vertex-kind batch and a forced
  fallback (workBudget 0), `cpp/tests/data/goldens/hyper_sssp/test_hsosp_matches_dijkstra.hsd`
  (`parity/hsosp_goldens.py subset`), replayed by `dyng_hsosp_dump_tests` in hosted CI.
- **Disk:** at most 1.5 GB new under the goldens root (HD18).

## Consequences

- M11a steps 2-4 replay the set by id (ADR 0080's modes): the batches, ids, skipped ops and pairs
  from the dumps, the distances and parents from the full arrays or the digests.
- The twins carry three recorders now (the CBST trace, the batch hooks, the H-SOSP dump); each is
  off without its variable, and the CBST traces are byte-identical to M8a's.

## Alternatives considered

- **A source patch of hsospBench** (print the arrays). Rejected: ADR 0013 forbids changing a
  tracked file; the link-time wrap reads the same values from outside.
- **Wrapping `finishBatch` for the pairs** (its argument is the sorted keys). Equivalent, but the
  pairs belong with the `applyBatch` counters and the post-batch rows of the same record; the
  pinned keys are still on the host when applyBatch returns.
- **Digests computed in the recorder.** Not needed: the raw dumps of the digest cases are at most
  a few hundred MB and are converted and deleted one at a time.
