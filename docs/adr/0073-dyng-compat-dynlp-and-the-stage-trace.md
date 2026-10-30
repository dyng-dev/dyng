# ADR 0073: dyng-compat-dynlp, the parity driver of label_propagation, and the library's stage trace

- **Status:** Accepted under delegation (2026-10-29; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. M10a step 3 (PLAN 5.6, 6.4.7, 8.3). It carries out the comparator
  rules of sheet C8 (a negative `--tau` keeps the original's automatic mode; `--tau 0`, a NaN
  `--tau` and `--maxIter <= 0` are refused as documented differences) and the scope rules of AD2,
  **decided by the author on 2026-10-28** (GOVERNANCE.md, approvals log, the rows "The parity rules
  of 0.4 (sheet C1-C8)" and "The measurement protocol and gates of 0.4 (sheet B1-B6)"); the tool's
  name follows E1 (named after its `tools/compat` directory).
- **Amended by:** ADR 0075 (2026-10-29, M10a step 4): on cuda the timed region is the tool's own
  pair of CUDA events on the resources' stream around each `update()` call (the profiler's CUDA
  events synchronize at every stage end, so no profiler is attached to the timed calls unless
  `--profile` or `--sync-stages` asks for one); the batch vectors are page-locked in place, and
  the evaluation keeps the vertex states itself (no download of the rows).
- **Date:** 2026-10-29
- **Deciders:** the AI assistant, on the author's behalf (M10a, step
  host-container-sequential-driver)

## Decision

### 1. The tool (`tools/compat/dynlp/dynlp_compat.cpp`, `dyng-compat-dynlp`)

- **Public headers only:** `dyng/*.hpp`, `dyng/generators/*`, `dyng/testing/*` (and the standard
  library). The CTest `compat_dynlp.includes` checks the includes; `ci/cmake_consumer.sh` builds it
  against an install tree.
- **The original's argv** (`utils.cu` 25-137): `--totalNodes`, `--numBatches`, `--avgDegree`,
  `--tau`, `--delta`, `--maxIter`, `--update inplace|damped`, `--gtFraction`, `--seed`, `--output`,
  `--validate` (the default) / `--noValidate`, `--noGraphFiles`, `--verbose` (the default) /
  `--quiet`, `--profile`, `--check`, `--help`, with `atoi` / `atof` semantics, the original's
  defaults and its checks. C8: a negative `--tau` is the automatic threshold (the published
  `stream.similarity_threshold`, which is the original's `computeAvgEdgeWeight` of the full
  graph); `--tau 0` and a NaN `--tau`, and `--maxIter <= 0`, are refused (exit status 1).
- **Setup (untimed):** the legacy stream (on the host threads of an OpenMP handle, as O's
  generator); the graph files (`graph/csr{RowPtr,ColInd,Values,TrueLabels}.txt`) unless
  `--noGraphFiles`; `graph::with_capacity(stream.row_capacity)` with the slotted `net_effect`
  properties; tau; the batch arrays (pinned host memory on cuda, M10a step 4); `compute()` on the
  empty graph; `result::reserve(stream.max_batch_vertices, stream.max_batch_edges)`.
- **Timed, per batch:** one `label_propagation::update()`. On cuda the time is the CUDA-event time
  of the profiler's stage `label_propagation.update` on the resources' stream (the events bracket
  the public call's work; M10a step 4 reads it); on sequential the host wall time; the host wall
  time of the call is recorded beside it in `dyng_times.csv`.
- **After the timer:** the labels to the host and the accuracy over the active vertices without a
  seed label (O's `computeAccuracy`: `float(correct) / float(total)`, 1 without vertices); with
  `--validate` IrLP on the snapshot (`compute()` with the active seeds; its iterations from the
  profiler counter `label_propagation.irlp_iterations`), its accuracy and the class agreement.
- **Outputs, in the original's bytes and precisions:** `results.csv` (`writeResultsCSV`: the
  columns and `std::fixed` precisions of the original), `dynlp_labels.txt` and `irlp_labels.txt`
  (`i %.6f` through `std::to_chars` of the float widened to double, O's TextWriter), the graph
  files.
- **dynG's flags:** `--backend sequential|cuda` (default cuda when built), `--cuda-engine`,
  `--label-dump DIR` (`labels.f32`: n float32 per batch; with `--validate` also `irlp.f32`),
  `--trace DIR` (below), `--sync-stages` (the profiler's synchronized stages, the diagnostic
  reading), and `--check` reinterpreted as dyng::testing's oracles after every batch, untimed:
  `harmonic_solution` (mean |F - F*| <= 2000 δ, no PCG skip; the original checker's line format),
  the seeds' labels exact, every label in [0, 1]; exit status 1 if one fails.

### 2. The stage trace (`cpp/src/algorithms/label_propagation/trace.{hpp,cpp}`)

When the environment variable `DYNG_DYNLP_TRACE_DIR` is set, label_propagation writes the records
of O's recording twin (ADR 0065) for every update, in the twin's format (the file
`<DIR>/<program>.dlt`, never overwritten; `DYNG_DYNLP_TRACE_DIGEST=1` stores every array as its
SHA-256): `apply_vertex_changes` (Ins \ Del, the statuses in O's codes by name, the seed labels as
O's gtClass), `append_edges` (the row ends, every slot's column and value bits, b0's offsets, O's
error flags 0 / 0), and when vertices were inserted `sparsify` (tau, G' as sorted pairs),
`connected_components` (the component of every inserted vertex), `label_init` (W0, W1, the initial
labels' bits), then `propagation_begin` (the initial frontier) and `propagation_end`; `compute()`
writes `irlp_snapshot`. The batch number is the result's update count. `dyng-compat-dynlp --trace
DIR` sets the variable (a public-headers-only tool cannot reach the stages otherwise); the gate
harness removes every `DYNG_DYNLP_*` variable from the timed runs. Without the variable nothing is
recorded (one pointer test per stage).

### 3. The replays (`parity/label_propagation_port.py`)

`compare.py label_propagation container` (the container on O's batches, ADR 0069), `trace`
(every deterministic field of every batch of P+ against the trace golden), `oracle-port` (the
harmonic oracle and the class margin on C, update and compute) and `determinism` (run-to-run
bits).

## Differences from the original (documented)

- C8's refusals above; `--check` runs dyng's oracles, not O's checker (whose results the step-1
  records hold); `--profile` writes dynG's profiler stages to `stages.csv`.
- No `exit()`; an error prints `Error: ...` and exits with status 1, as O's `main` does for its
  own errors.
- The configuration banner and the summary are dynG's text (not compared); the files are O's.

## Alternatives considered

- **A public trace API** (an option or an observer): rejected; the trace is a parity diagnostic of
  the original's recorder format, not a library feature.
- **The trace written by the tool from public data:** the statuses and rows are public, the
  stages (V_aff, G', the components, W0 / W1, the initial labels) are not.
