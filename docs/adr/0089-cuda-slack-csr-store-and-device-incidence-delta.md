# ADR 0089: The CUDA slack_csr store and the as-is device incidence_delta

- **Status:** Accepted. The surface and its behaviour are the author's: HD9 (`slack_csr` storage
  and line graph on every backend, no public capacity field: MOSP_ESCHER's slack rules are
  internal knobs selected by the parity preset; the h2h view sorted on request; `overlaps` empty),
  HD3 and HD4 (the maintain row and its protocol), HD12 (`dyng-compat-hsosp` and its subcommands),
  **decided by the author on 2026-10-28** (GOVERNANCE.md, approvals log, the rows of 2026-10-28).
  The technical parts (the as-is design, the transplant and its mechanical classes, the views, the
  checks, the limits, the harness) are accepted under delegation (GOVERNANCE.md, "Delegation of
  technical ADRs"). It carries out M11a step 3 (PLAN 4.2, 4.5.3, 4.7.1-4.7.3, 6.3 step 5, 6.4.8,
  8.6; the M11a acceptance criteria 8-10 (cuda rows), 12, 15 (container, cuda) and 17 (the store's
  parts)).
- **Date:** 2026-10-29
- **Deciders:** S M Shovan (HD3, HD4, HD9, HD12, 2026-10-28); the AI assistant, on the author's
  behalf, for the technical parts (M11a, step cuda-slack-csr-translate-early-read)
- **Builds on:** ADR 0038 and 0042 (the transplant method), ADR 0041 (the container), ADR 0053
  (`reserve_with_headroom`), ADR 0080 (the by_id mode, the region map), ADR 0081 (the dumps), ADR
  0085 (the incidence edits and the allocator), ADR 0086 (the host slack_csr store, the knobs,
  incidence_delta inside the commit).

## Context

ADR 0086 gave slack_csr storage to the host backends; the cuda backend refused it. MOSP_ESCHER@4b86159
(B) keeps, besides its host model (`HostHypergraph`), a device graph (`hsosp/include/hsosp.cuh`):
the incidence mirror `DeviceIncidence` (hyperedge rows and vertex rows as slack rows, each with
its tail), the slack CSR line graph `DeviceH2H` (rowStart, deg, cap, colInd, per-node weights,
the tail cursor and the overflow flag) and the last batch's pairs `DeviceDelta`. Per batch its
`applyBatch` (`hsospDelta.cu:882-942`) runs the host incidence update, `deriveDelta` on the GPU
(the order-free pair delta from the incidence changes, `hsospDelta.cu:600-878`), the CSR apply
`applyDeltaPairs` / `applyRowsWarp` (`hsospDevice.cu:82-149, 704-771`) and, when the CSR's tail
overflows, a rebuild from the host line graph (`buildCsr`, `hsospDevice.cu:613-694`).

## Decision

### 1. The as-is design: the host model stays authoritative

`slack_device_store` (`cpp/src/hypergraph/slack_csr/device_store.{hpp,cu}`) holds a slack_csr
host store (ADR 0086) without its host line graph as the host model: the rows by id, v2h, the id
allocator (its fills and unfills), the weights by id. Every host phase of the commit (ids, H2V
erase / insert / unfill / fill, v2h) runs there, as B runs them on its `HostHypergraph`. The device
holds B's graph, ported as is (section 2), kept from every batch by the steps of B's `applyBatch`
inside the container's commit:

1. `line_graph_delta_of()` (stage `hypergraph.apply.translate`) ships B's `IncidenceBatch`: the
   (vertex, id + 1) keys with their signs, the touched hyperedges with their rows before and
   after the batch, the touched vertices with their lengths after it, the inserted ids with their
   weights and the dead ids (the applied record's fields: the record's rest is computed before the
   translate stage now), and runs `derive_delta`: CUB's radix sort and reduce-by-key to a net sign
   per incidence, the candidate counts and pass 0 on the pre-batch vertex mirror, the vertex rows
   that outgrow their slots relocated (planned on the host) or the whole vertex mirror re-uploaded
   at x1.5 when its tail is full, the vertex rows patched (removals by ballot search and
   swap-remove, then appends), pass 1, the candidates sorted, deduplicated and classified by
   their pre / post overlap (deleted pairs first), the post-batch hyperedge rows written into the
   mirror (or the hyperedge mirror re-uploaded at x1.5), the ids new then dead. More than
   `knobs.max_candidates` (INT_MAX, B's limit) candidates is a `capacity_error`.
2. `finish_line_graph()` (stage `hypergraph.apply.line_graph`) runs `apply_delta_pairs`: the
   directed keys `row << 33 | ins << 32 | col` sorted, run-length encoded by row and applied one
   warp per row (`apply_rows_warp_kernel`: the deletions by ballot search and swap-remove, a row
   that outgrows its slot relocated to the tail with one `atomicAdd` on the tail cursor, the
   insertions appended), the weights of the new ids scattered. The overflow flag (a relocation past
   the entries) makes the container rebuild the CSR from the host line graph with the call's entry
   headroom inside `hypergraph.apply.line_graph` (B's S5); the delta stays on the device. Then the
   degree of every dead id is set to 0 (B zeroed it in `hsospUpdate` through a const reference: it
   is the container's job here, so the engine never writes container state), and the pair count
   changes by inserted - deleted (`h2hPairCount`).

The pairs stay on the device (`slack_access::last_delta()`: the pairs, the new and dead ids and the
new weights, what hyper_sssp's CUDA engine reads); the applied record copies them to the host on
request (`hypergraph_access::pairs()`): a batch of DBLP's 50K configurations has about 4.9 million
pairs, whose copy would sit inside the maintain row (B copies its sorted keys in its escher timer).
The host stores fill the record's pairs as before.

The host model's v2h rows are unsorted, as B's are (appended, and the first occurrence
swap-removed: its `removeValue`); the public v2h view and the checks sort them (section 8).

The later alternative, recorded: plan the mirror's layout on the device (B's host loops over the
touched vertices and hyperedges, and its host-built `IncidenceBatch`, become kernels). It is not
done in 0.4: the as-is port is the parity baseline, and the early reading (section 7) shows that
the host side, made one pass per list (section 8), is within the maintain row's bar without it.

### 2. The transplant and its mechanical classes (PLAN 6.3 step 5; ADR 0038's method)

`parity/escher_port/transplant.py --map parity/escher_port/rename_map_hsosp.toml --check --rev
<c1>` regenerates the transplant from B's committed tree (`git show 4b86159:<path>`; the map's
`original_dir`, a new key of the script): `slack_csr/slack_csr_store.{hpp,cu}` (DeviceDelta,
DeviceIncidence and its scratch, DeviceH2H, their lifecycle, the mirror's slot rules, uploads,
build and check), `line_graph/slack_csr_h2h.cu` (buildDeviceH2H, buildCsr, applyDeltaPairs,
expandDeltaKeysKernel, applyRowsWarp, scatterWeightsKernel, downloadRows) and
`line_graph/incidence_delta.cu` (deriveDelta and its kernels), each headed `// Derived from
MOSP_ESCHER@4b86159:<path>`. B's stale paragraph of `hsosp.cuh:20-25` (a per-thread row apply that
`applyRowsWarp` replaced) is not copied. The manual hunks (`manual_hunks_hsosp.patch`): the views
of the host model the transplanted code reads (`host_hypergraph`, `line_graph_csr`,
`incidence_batch`: B's HostHypergraph, LineGraphCSR and IncidenceBatch over dynG's 0-based host
store), the scratch's definition and the two uploads moved to the header (B had the mirror and the
delta in one translation unit), `derive_delta` with its candidate limit. Then one mechanical class
per commit, the CUDA delta replay of make test green after each:

| Commit | Class | What changed |
|---|---|---|
| `e297773` | c1, the transplant | `transplant.py --check --rev e297773` regenerates its four files; the store over the host model; the CUDA refusal of slack_csr lifted |
| `a1217ef` | c2, errors | `HSOSP_CUDA_CHECK` -> `DYNG_CUDA_TRY`; `runtime_error` / `invalid_argument` -> `capacity_error` / `invalid_argument_error`; a broken invariant raises the sticky device error word |
| `005ba94` | c3, streams | every kernel, CUB call, copy and memset on the store's stream; synchronous copies made asynchronous, the host waiting only at its read points |
| `1f8a1ae` | c4, memory | every device and pinned array a `dyng::buffer` of the store's resources |
| `7356978` | c5, scratch | the per-batch arrays are scratch buffers sized with `reserve_with_headroom` (ADR 0053) |
| `cb8c542` | c6, stages | the profiler stages (`.translate.reupload`, `.line_graph.rebuild`, `.normalize`) and the kernel registration for `warm_up()` |

The CUDA delta replay of make test (every session, by id) had 0 mismatches after each commit.
`f62bdf5` then added what is not mechanical: node-capacity growth (B throws), the internal int32
limits, the forced-overflow and over-limit tests and the CUDA mutation twins (#3, #7, #8).

### 3. Capacities (HD9: internal knobs only)

The node arrays (the CSR's, the mirror's hyperedge rows, the touched index) have `node_capacity`
nodes: the parity knobs give B's `maxNodes` (rows + planned + 65536 + 1024, hsospBench's
planning; `dyng-compat-hsosp bench` and the replays set it); without them the bound plus
max(1024, bound / 8). An id past the capacity grows the node arrays by half (dynG; B throws), the
mirror's host offsets with them. The entry headroom (1.6, B's) sizes the CSR and the mirror at the
build, the call's headroom every rebuild; the mirror re-uploads at x1.5. The limits of the int32
counts are internal knobs too (`max_candidates`, `max_ids`, `max_entries`, INT_MAX by default, lower
in the over-limit tests): each is a `capacity_error`, never undefined behaviour or a silent wrap.
**Amended by the M11a review** (class c2's spirit, an error class only): the device delta checks
`max_candidates` right after the candidate totals are read, before it reserves the two candidate
arrays (8 bytes per candidate each) and before the vertex mirror changes; MOSP_ESCHER reserves
first and throws after (hsospDelta.cu:695-696, :772), so on a 24 GB device more than INT_MAX real
candidates was an `out_of_memory_error`, never the documented `capacity_error`
(`SlackOverflow.MoreCandidatesThanTheRealLimit` uses the real limit). A planned node capacity
counts only when the knob is set: an empty hypergraph gets the minimum of 1024 nodes too (it got
0, and a result's device state of 0 nodes left a launch error of zero blocks pending, which the
next commit reported as its own and poisoned the hypergraph; `HyperSssp.AnEmptyHypergraph...`).
The engine skips its fills for 0 nodes and checks their launches. The device error word of
class c2 has its own test (`SlackOverflow.ACorruptDeviceRowIsAnInternalError`: a planted
corruption of a device CSR row is an `internal_error` that poisons the hypergraph). The frozen
`hypergraph::from_hyperedges` no longer documents a `not_supported_error` for slack_csr storage
on the cuda backend, which this ADR made supported (a documentation-only api-change).

### 4. Views, checks, stats

- `h2h()` decodes the device CSR to sorted, contiguous rows (with `out`), on request, counted in
  `view_bytes`; `h2v()`, `v2h()` and the internal row views are the host model's, uploaded on
  request; `weights()` is the container's.
- `check_integrity()`: the host model (rows, v2h, allocator); the incidence mirror against it
  (B's `incidenceMirrorMismatches`); the device CSR against the line graph of the incidence lists
  as multisets; the maintained pair count; the capacity invariants (every degree within its slot,
  the slots below the tail cursor, the tail within the entries, no row above the nodes, the
  mirror's tails within its capacities); the device weights of the live ids.
- `store_stats().device_bytes`: the CSR, the mirror (`DeviceH2H::deviceBytes()`, B's graph_mb
  scope) and the delta's buffers; `slack_access::graph_scope_bytes()` is B's scope alone (the
  memory gate's store accounting, HD6).
- `clone()` and `to_backend()` across the three backends keep the ids, rows, weights, the
  allocator state and the line graph.

### 5. The parity on cuda (criteria 8-10, GO / NO-GO 2)

`dyng_hsosp_replay --backend cuda` (`parity/compare.py hyper_sssp --part delta --backend cuda`)
replays every session of make test and every paper configuration by id (a digest case's pairs by
their SHA-256): per batch the ids, the skips, the dead ids, the rows, the pairs and the pair count,
and B's candidates, csrRebuilt and mirrorReuploads, and B's device layout (node capacity, nodes,
entries, the tail cursor, the live entries, the mirror's capacities and tails, graph_mb) equal B's.
The numbers are in `parity/results/M11a.md`.

### 6. The driver and the harness

`dyng-compat-hsosp bench` (HD12) runs hsospBench's `--hg` options on the cuda backend: B's loader
(`read_hypergraph(hsosp())`, B's virtual rows at B's positions), B's capacities, the golden's
exported batches (HD5) converted before each timer, `--sync-stages`, the profiler's stages and the
driver timers in `--timing`, B's CSV columns it fills in `--out`, and with `--digests` the round
guard's pairs digest, pair count and rebuilds. `--maintain-only` is this step's mode;
hyper_sssp's rows arrive with it. `replay` gains `--backend cuda` and `--line-graph` (the pairs
against `testing::line_graph_delta`). `parity/hsosp_perf.py run` (reached through `perf_ab.py
hyper_sssp`) alternates B and the port with one layout, EAGER, no OMP_*, the clocks locked, the
contamination monitor, the audit of step 1's thresholds, and guards every round.

### 7. The early maintain reading (criterion 12)

`perf_ab.py hyper_sssp run --maintain-only`, 11 rounds per configuration at step 1's probe clocks
(base, GPU 0, two exclusive-lock windows of 2026-10-29, 22:50-22:58Z and 22:59-23:02Z), every
round guarded: every maintain[b1..b3] row of dblp_hyperedge_50000, dblp_vertex_50000,
dblp_vertex_200000, geology_hyperedge_200000 and geology_vertex_25000 is between 0.61 and 0.81,
**GO** (at most 1.03) on every row (`parity/results/M11a.md` section 9.3,
`M11a-early-maintain-{dblp,geology}.json`). No fix plan is needed; the binding gates of the whole
update follow in step 5.

### 8. The host costs of a batch (criterion 12's profile)

The first early profile of the maintain row (`dyng-compat-hsosp bench --maintain-only`, GPU 1,
development clocks) gave the port 80 ms against B's 12.9 ms on geology_vertex_25000 and 54
against 25.7 on dblp_vertex_50000 (B: `t_delta_ms + t_csr_apply_ms`): the device stages were B's
(the transplant), the host stages were not. B's `HostHypergraph::applyBatch` is one pass over the
batch in its order; the port's Step 0 and commit evaluated the edits on copies, grouped them,
sorted them, read the rows back and scanned the allocator. The host side is now:

- **Step 0 of the incidence edits** groups the edits by hyperedge (a stable radix sort of
  `id << 32 | edit`: a group keeps its removals, then its additions, in batch order) and evaluates
  the groups in parallel on a copy of each row (an edit's outcome depends only on its hyperedge's
  row on G_t and the earlier edits of that hyperedge, so the outcome is the batch-order
  evaluation's for every thread count); the groups come out by increasing id: the edited ids,
  their rows before and after, and the applied edits grouped by id for the allocator's H2V unfill
  and fill (B's `flushGrouped`). A live id is a non-empty row (`check_integrity()` checks it).
- **The commit** builds the (vertex, id, sign) list once (B's order), groups it by vertex with one
  radix sort (the touched vertices, sorted and unique, and the v2h groups), sets the edited rows to
  their post rows (`incidence_store::h2v_edit`: the allocator's unfill and fill, then the rows, in
  parallel), applies every v2h change grouped by vertex in one parallel pass at the end
  (`incidence_store::v2h_apply`: per vertex B's order of swap-removes and appends; nothing of the
  H2V side reads v2h) with the touched vertices' lengths, and merges the sorted deletions,
  insertions and edits into the record's touched ids and rows (no read-back of the rows).
- **The allocator** keeps its deleted keys as a bitset and a per-key flag for the rows whose tail
  is an overflow segment: an insert no longer scans every key twice (its dead keys, its rebuild),
  and an unfill or fill looks a tail up only where there is one (the same keys in the same order;
  the M8b and M11a lockstep replays are unchanged). It is shared with escher storage's host model,
  which gets the same ids faster.
- **A deleted row is cleared**, its capacity kept (B's `heVerts[id - 1].clear()`), not freed.
- **The host team** (`cpp/src/hypergraph/host_team.hpp`): every host region of slack_csr storage
  (the builds, Step 0, the H2V and v2h regions, the host store's pair delta and CSR apply) runs on
  one fixed team of min(threads, 16) threads, or on one thread below 4096 items. On the shared
  28-core host a team of all 56 hardware threads spent more in its fork and join than in a batch's
  work (Step 0 of a 25K-edit batch: 12 ms on 56 threads, under 1 ms on 16), and the line-graph
  build of geology.hg took 0.92 s on 56 against 0.75 s on 16; the size is fixed because libgomp
  ends the surplus threads of its pool when a smaller team starts (mixing 56 and 16 cost 20-30 ms
  in the first batch after the build). Every region carries its exceptions out with
  `region_exceptions` (ADR 0061) and has an allocation-failure test on 4 threads; a test applies
  batches of more than 4096 groups on 1 and 16 threads and on sequential and compares everything.

## Consequences

- hyper_sssp's CUDA engine (M11a step 4) reads the device CSR, the node weights and the last
  delta through `slack_access` and never writes them.
- The CSR rebuild needs the host line graph (the host model's rows): its host cost is in the
  maintain row of the batches that overflow, as in B.
