# ADR 0037: The CBST host model and the rules of the trace replay

- **Status:** Accepted under delegation (2026-10-07; GOVERNANCE.md, "Delegation of technical
  ADRs"). It carries out M8a step 2 (PLAN 5.8, 6.4.5 "Parity", 8.3; the M8a acceptance criterion
  8, the go/no-go before the store port). It changes no rule the author approved and weakens no
  comparison: the replay compares more than PLAN 6.4.5 asks for (offsets and every recorded field
  in the `exact` field set), never less.
- **Date:** 2026-10-07
- **Deciders:** the AI assistant, on the author's behalf (M8a, step host-model)
- **Number:** 0035 is the 0.2.0 release branch's, 0036 the trace format's.

## Context

PLAN 6.4.5 proves the port of the ESCHER CBST store by trace replay against ESCHER-GPU@abcf9b2
(A) and MOSP_ESCHER@4b86159 (B): the traces of ADR 0036 record every CBST operation of both
originals with its inputs and the state after it. Before any store code is ported, M8a requires
an independent oracle that replays every trace set with 100% agreement (criterion 8), so that the
port of step 3 is compared with something that has itself been validated against both originals,
and so that a divergence of the port can be told apart from a misreading of the originals. PLAN
5.8 requires that the oracles share no code with what they check, and PLAN 5.8 lists
`cbst_host_model` among the public helpers of `dyng::testing`.

## Decision

### 1. The model (`dyng::testing::cbst_host_model`)

`cpp/include/dyng/testing/cbst_host_model.hpp` and `cpp/src/testing/cbst_host_model.cpp`: a
sequential host model of one CBST, in the installed target `dyng::testing`.

- **What it keeps per tree:** the payload capacity, the bump pointer (`initialPayloadSize`), the
  record count, the **flat payload itself** (the used prefix; the rest of the capacity reads as
  0), the nodes by heap position (key, first-segment offset and length, the `size` field, the tail
  segment's base, capacity and occupancy), `avail` and `subtree_avail` by heap position, the
  sorted key array (the originals' `d_keys`), the layout (heap position <-> in-order rank by
  `cbst_rank_of_position`, computed in 64 bits), the scratch capacities, the predicted growth and
  compaction events, and whether the original's process would have ended.
- **Why the flat payload and not only rows and segment sizes.** The rows the originals expose
  depend on payload bytes that a row-level model loses: erase keeps a dead row's values (its row
  still reads them, until a reuse or B's compaction rewrites them); a reused slot is cleared only
  up to its first segment; the unfill kernels differ by degree bin (the thread kernel walks the
  whole row with its chain, the warp and block kernels compact only the first `occupancy` slots of
  a one-segment row and drop the non-positive values among them); a fill item of 0 values writes a
  terminator at its tail position. Keeping the payload makes the model exact for every recorded
  field (offsets, occupancy, segment counts) at no cost in clarity: every operation is a short
  sequential loop over it.
- **The policy** is a testing-side struct, `cbst_host_policy`, not a store header (M8b designs the
  public hypergraph headers): growth (none, or A's `max(needed + needed/4, 3*capacity/2)` capped
  at `INT_MAX`), compaction (none, or B's trigger points: a fill whose overflow segments do not
  fit, compact and refill the rest once; an insert whose surplus rows do not fit, compact before
  appending; `compact()`), failure (A's process exit, B's `escher::EscherError`), the construct's
  occupancy (A's scan, B's caller counts or 0, or counts-else-scan), scratch growth (A exact, B
  geometric), construct validation and the fill clamps (B), the presence of `rebuild` (A) and
  `compact` (B), and A's degree-bin launch counters. The presets `escher_gpu()` and
  `mosp_escher()` reproduce each original; `dyng_default()` composes them into PLAN 6.4.5's target
  ("compact first, then grow, fail only at the int32 offset limit"); it is provisional until the
  store's presets are merged (step 4) and has no trace of its own.
- **Predicted, never read.** Growth and compaction happen when the policy says so on the model's
  own bump pointer and capacity; the trace is only compared with. The model never consults a
  trace.
- **Undefined behaviour of the originals** (two items naming one row in one fill or unfill, which
  the originals' threads race on; a read or write outside the capacity; an endless chain; an
  operation on a tree that was never built although it has records) returns
  `cbst_op_status::undefined` with the reason: the model refuses to guess. No trace contains one.
- **Isolation.** The model is its own object library (`dyng_testing_cbst_model`, linked into
  `dyng_testing`) that uses only the C++ standard library. CTest `cbst_model_isolation` (labels
  cpu, parity, cbst; `cpp/tests/parity/escher/check_model_isolation.cmake`) checks with the
  compiler's `-MM` that its source includes no project file but its own header (so nothing of
  `cpp/src/hypergraph` or any other dynG header) and with `nm` that its object references no
  `dyng::` symbol. Its sources carry `// Derived from` lines because they restate the originals'
  documented algorithms (not their code: the kernels become sequential loops, the best-fit
  pipeline becomes its greedy form, which assigns the same slots).
- **The public-header exception.** The M8a rule that no public header exposes CBST keys, offsets,
  `+1` shifts or sentinels applies to the hypergraph API; `dyng/testing/cbst_host_model.hpp`,
  whose subject is the CBST layout itself, is the documented exception (the store's public-header
  check of step 3 is to cover the hypergraph headers, not `dyng/testing/`).
- **The layout checks** of A's `tests/cbst_test.cu` run on the model's layout function
  (`check_cbst_layout`): n = 1..1024 and the 17 sizes up to 2^24 (2,467,389 included): the ranks
  are a permutation, the in-order traversal of the heap visits them in order, a search from the
  root finds every key (every key up to 2,467,389; above that a deterministic sample of 65,536
  keys, the in-order check covering the rest), the found node holds the key's row, and the
  inverse formula recovers every key.

### 2. The comparator and its rules

`cpp/tests/parity/escher/cbst_compare.{hpp,cpp}` (library `dyng_cbst_compare`) drives any store
through `store_adapter` (create and destroy a tree object; run construct, rebuild, insert, fill,
erase, unfill, compact; the scalars of a tree; its digest: keys, `subtree_avail`, every key's
record and row), reads the trace with ADR 0036's reader and state tracker, and compares after
every operation. `host_model_adapter` drives the model; the store's adapter follows in step 3.

The field sets:

| field set | compared |
|---|---|
| `semantic` (criterion 8) | the exception of every op (the original's type; a process exit is never expected), every `InsertMapping`, the key list, `subtree_avail` by heap position, per key `avail`, `length` and the row (in order for H2V, H2H, V2V in any case of the name, as a multiset otherwise), the bump pointer and the capacity after every op, and the ops at which the payload grew or was compacted |
| `offsets` | also the first-segment offset and the tail base of every key (step 3 asks for this for A under `escher_gpu`) |
| `exact` | also `size`, occupancy, tail capacity, segment count, row length and termination of every key, every row in order, the record count, the scratch capacities, D after the op, A's `deviceBytes()` and degree-bin launch counts where recorded, B's `compact()` return, and the state before every op |

- **Why the per-key length is compared** even where offsets are not: a slot's length decides
  which items it can take in every later best-fit insert (its capacity is `length - 1`), and B's
  compaction rewrites it (a live row gets `(count + 4) / 4 * 4` slots). Two stores whose rows and
  `avail` agree but whose lengths differ would assign different keys at a later insert; comparing
  the length catches that at the op that caused it, not at the insert that shows it.
- **Events** are inferred by the same rule on both sides: growth when the capacity after the op
  exceeds the capacity before it; compaction for an explicit `compact()`, or when a row the op
  does not name (fill, unfill, erase: their keys; insert: its mapping) changed its offset, tail
  base, tail capacity or segment count, or the bump pointer fell (`escher_trace.py`'s rule). The
  model also reports the events it predicted (`cbst_host_model::events()`).
- **Hash-only rows** (goldens.toml `hash_only`, e.g. A's coauth H2H) are compared by the row's
  FNV-1a 64 hash, in order; they are counted, never skipped.
- **Inputs recorded as hashes only** cannot be replayed by any store: the tree is reported as
  unreplayable from that op on, with the reason, and the remaining ops are counted. The sets
  allow it only where documented: the H2H rebuilds of `escher_gpu_coauth50k` (3 ops; their
  inputs are the result of A's line-graph merge, whose recorded hashes M8b can check) and the
  batches-mode set `escher_gpu_syn1m` (no state digests: batch dumps for M9).
- **An op missing on either side fails:** an op the store cannot run (`unsupported`,
  `undefined`, e.g. `compact()` under `escher_gpu`), an op after which the model's process would
  have ended, and a trace that does not end normally (an op the trace could not record).
- **First divergence.** All differing fields of the first differing op of a tree are reported
  (op index in the trace, the tree's op index, tree, dataset, key, the trace's and the store's
  values); the tree is not compared further (the states have diverged), the other trees are. A
  JSON summary (`replay_report::to_json`) lists the counters, the mismatches and every tree.
- **The trace itself** is checked on the way: every recorded hash must match the state the
  tracker rebuilds (a `trace` mismatch otherwise), and `pre.matches = 0` (a change between
  recorded ops) is a mismatch.

### 3. Running it

`dyng_cbst_replay` (`cpp/tests/parity/escher/cbst_replay_main.cpp`) replays `.cbt` or
`.cbt.zst` traces (through `zstd -dc --long=31`; the side file is decompressed to a temporary
file) under a named policy (default: the trace's original) and field set, and writes the JSON
reports. `parity/compare.py cbst_trace --store host_model [--set ...] [--policy ...] [--fields
...] [--record DIR]` verifies a set's manifest, replays every case in parallel and writes the
record `M8a-host-model-<set>-<fields>.json`; `--store device` is reserved for step 3. CTest:
`dyng_cbst_model_tests` (cpu, parity, cbst: the layout, the model's rules, the committed subset
under all three field sets, the perturbed traces) in hosted CI; `dyng_cbst_model_large_tests`
(parity, cbst, slow) replays every set in `$DYNG_SCRATCH/goldens/cbst_trace` and skips with a
message when they are absent.

**Perturbations** (tests): one insert item one value shorter (its neighbour one longer) and one
tie reordered (two items that took deleted slots of equal capacity with their recorded keys
swapped) are reported as mismatches at that op; a policy without `compact()`, a trace that ends
inside an op and a hash-only input are reported as such.

## Findings (the facts of the step, checked against the code)

The facts the step was given agree with the code, with these precisions (the code wins):

1. A's growth is `max(needed + needed / 4, 3 * fixedSize / 2)` in integers, capped at `INT_MAX`
   (`needed * 1.25` and `1.5 * capacity` up to rounding), and happens only when the operation's
   appends need more than the capacity; the used prefix is copied, the rest zeroed.
2. A surplus row of 0 values has length 1 (`nextMultiple(0) = 0`), capacity 0.
3. Erase keeps the payload; a dead row reads its old values until a reuse or B's compaction
   (which empties dead rows and keeps their first-segment length).
4. Unfill depends on the degree bin of the removal count (finding of section 1); both kernels
   remove every occurrence.
5. A's construct with a capacity below the payload builds no tree but records the record count
   and the bump pointer; B's validation throws and leaves the tree empty.
6. B's insert that fails after its reuse (compaction before the surplus, then the per-row
   capacity throw) leaves the reused slots written and `subtree_avail` stale (it is recomputed
   only at the end of a successful insert).
7. On a tree of 0 records, fill and unfill change nothing, while insert (it reads D from no
   subtree counts) and erase (a launch of 0 blocks) hit a CUDA error: A exits, B throws.

## Consequences

- The model replays every replayable operation of every golden set with 100% agreement in the
  `exact` field set, under each original's preset (parity/results/M8a.md, section 10): the go of
  the go/no-go. The port of step 3 is compared with the traces directly and, in the scenarios of
  step 4, op by op with this model.
- `dyng_default()` is provisional; step 4 aligns it with the store's preset and records any
  change.
- The model is a public helper: it is in the public-API listing, documented, and follows the
  stability rules of `dyng::testing`.
