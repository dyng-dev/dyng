# ADR 0040: The CBST presets and automatic compaction (part 2 of the staged merge)

- **Status:** Accepted under delegation (2026-10-08; GOVERNANCE.md, "Delegation of technical
  ADRs"). It carries out M8a step 5 (PLAN 6.4.5 "target policy", 4.7.3, 8.4, 6.1 D8; the M8a
  acceptance criteria 10-17) and completes ADR 0039 (part 1). It changes no rule the author
  approved and weakens no comparison: ESCHER-GPU's traces are still compared in the `exact` field
  set at every commit, and from the commit that adds `mosp_escher()` MOSP_ESCHER's traces are too,
  with two documented divergences, each mandated by an accepted rule (section 3).
- **Date:** 2026-10-08
- **Deciders:** the AI assistant, on the author's behalf (M8a, step merge-policies)
- **Number:** 0035 is the 0.2.0 release branch's, 0036-0039 this milestone's.

## Context

ADR 0039 merged every behaviour of MOSP_ESCHER@4b86159 (B) that changes no trace of
ESCHER-GPU@abcf9b2 (A). Left were B's automatic compaction (the only store behaviour that can
change keys: a compaction rewrites the lengths of live rows, which best fit reads later), its
preset, dynG's default preset (PLAN 6.4.5: "compact first, then grow (A), throw only at the
offset limit or on OOM, with exceptions everywhere"), the partial-failure rule (PLAN 4.7.3), B's
9 CBST scenarios, the graph-on-CBST test (PLAN 6.1 D8) and the mutations (PLAN 8.4).

## Decision

### 1. The presets (`store_policy.hpp`, internal)

| | `escher_gpu()` | `mosp_escher(extra = 2^20)` | `dyng_default(max_payload = INT_MAX)` |
|---|---|---|---|
| initial capacity (`initial_capacity()`, the container's choice) | in-place trees `max(requested, size + size/8 + 1024)`, merged trees exact | `floor(flat * 1.3) + extra`, at most 2.0e9 (`hypergraph/src/DynamicHypergraph.cpp:79-107`; above the cap `capacity_error`) | as `escher_gpu()` |
| a payload that does not fit | grows to `max(needed + needed/4, 3 * capacity / 2)`, at most INT_MAX | compacts at B's trigger points, then `capacity_error` | compacts at B's trigger points, then grows by A's rule, at most `max_payload` |
| compaction (`compaction_rule`) | `never` | `mosp_escher` | `mosp_escher` |
| scratch | exact (A) | geometric (B) | geometric (ADR 0039 (g)) |
| failure | `capacity_error` beyond INT_MAX | `capacity_error` where B threw `escher::EscherError` | `capacity_error` only beyond `max_payload` (the int32 offsets), `out_of_memory_error` on an allocation failure |

**The hysteresis of the default** (`store_policy::compaction_headroom`, M8a review): after an
operation that compacted at a trigger point, `dyng_default()` also grows the payload by A's rule
when fewer than `capacity / 8` slots are free, to the capacity that leaves that many free (at
most `max_payload`; 8 is the slack of A's initial sizing). Compact-then-grow alone has none: after
a compaction the rows are tight, so while live data grows slowly toward the capacity the free
space after each compaction shrinks and the O(payload) compactions come more and more often (the
review measured gaps of 20, 21, ..., 4 rounds; the control run of `CbstContractHysteresis` 21,
17, 14, ..., 1) before the first growth. With it, the next compaction comes only after
`capacity / 8` slots were used again, so a compaction is amortized over that many slots. The rule applies at the
two automatic trigger points only (not to an explicit `compact()`), never under `mosp_escher()`
(B's literal policy, `compaction_headroom` 0) and is mirrored by the host model's
`cbst_host_policy::dyng_default()` (`compaction_headroom` 8), which predicts the extra growths:
the default preset's replays against the model (section 3) include them.

`max_payload` (dynG) bounds every growth and initial capacity of a policy; it is INT_MAX for every
preset and lowered only by tests, which reach the limit with it (`CbstStorePresets`). The trace
replay takes each tree's capacity from the trace (B's callers pass their own), so the initial
sizing of `mosp_escher()` is used by the container (M8b) only and unit-tested here.

### 2. Automatic compaction (`compaction_rule::mosp_escher`, `cbst_store.cu`)

- **fill** (B's `fillImpl`, `escher/structure/operations.cu:409-444`): the insert-node kernels
  write what fits each row's tail; when the overflow segments do not fit behind the bump pointer,
  the store reads the relocation plan (one more read point, B's copy), compacts, and appends the
  values that did not fit their tails once more, without compaction; what still does not fit
  grows by the policy or is a `capacity_error` (B: `EscherError`, `:436-444`).
- **insert** (B's `insertCBST`, `:694-721`): after the best-fit reuse, the surplus rows are
  sized; when they do not fit, the store compacts first; when they still do not fit and the
  policy does not grow, `capacity_error` names the first row that does not fit, before any surplus
  row is written (B threw per row, `:713-721`, also before writing).
- **Deviations from B's literal code:** the exception type (PLAN 4.7.3; the comparator maps B's
  `EscherError` per throw site, `equivalent_exception()`); the compaction is `compact_cbst` of
  ADR 0039 (h) (bounded walk, pooled temporary); under dynG's default the second pass grows
  instead of throwing.

### 3. The comparisons and the documented divergences

- **B's sets under `mosp_escher()`** are compared with the `exact` field set (criterion 10(b) and
  more: offsets, occupancy, tail capacity, scratch, D, the state before every op). The only
  documented divergences are:
  - `construct_without_row_counts`, mandated by M8a acceptance criterion 11 ("never B's `nullptr`
    path") and ADR 0039 (c): a construct of B without row counts gave every row the tail occupancy
    0 (B's legacy behaviour), where the store scans. Exempted are **only** the occupancy of the keys
    built that way, while the trace still has its 0 and the store its scanned count, and B's own
    integrity verdict `check.rows` when it is exactly one metadata violation per live such key
    (`escher/structure/integrity.cu@4b86159:160-186`) against the store's clean verdict. Every other
    field of the op, and every later op of the tree, is compared (in `test_cbst_smoke` the erase and
    the insert after each construct); a key leaves the exemption when its occupancy is rewritten.
  - `construct_below_payload` (ADR 0038): the targeted construct whose capacity is below its
    payload (B threw, as the store does; the tree has no other op).
- **B's sets under `escher_gpu()`** stay judged against the host model under the same preset
  (ADR 0037, 0038).
- **Every set under dynG's default** (criterion 10(c)) is compared with the field set `policy`:
  the semantic fields up to a tree's first policy difference (a growth or compaction on one side
  only, a different capacity or bump pointer: `policy_difference`; B's capacity exception where
  the default grows: `policy_exception`, which ends the tree because B's state after its exception
  is not a state of the default); from there on the keys, avail, subtree_avail, the rows of the
  live keys, every InsertMapping and every exception, while lengths, the layout and the rows of
  deleted keys (which a compaction empties, ADR 0039 (h)) are the policy's. An InsertMapping that
  differs after a policy difference changed a length is documented (`policy_mapping`) and ends the
  tree (its later rows sit under other keys); a mapping difference anywhere else is a mismatch.
  The store's own integrity checks must be clean throughout.
- **M8a review: "policy" never means "unchecked".** A reviewer planted a compaction-padding defect
  and the default-preset replay passed: every layout difference opened a policy point, so the
  defect's lengths and rows were filed as documented. Three changes close it:
  1. A tree's first policy point is an op where the trace's and the store's growth or compaction
     events differ (or the original threw, `policy_exception`); a capacity or bump-pointer
     difference where both sides grew and compacted alike is a mismatch.
  2. The store's own side of the policy differences is checked in lockstep with the host model
     under the same preset (`CbstPolicyLockstep` in `cbst_compact_test.cpp`: every op of the
     committed traces, and `CbstPolicyLockstepLarge` over the four small sets, under
     `escher_gpu()`, `mosp_escher()` and `dyng_default()`): InsertMapping, offsets, lengths,
     occupancy, the bump pointer and the capacity after every op, so every growth and compaction
     and the hysteresis of section 1. The model of the store's `mosp_escher()` scans where B's
     construct had no row counts (ADR 0039 (c)); where the model cannot predict the original (an
     op its policy lacks) the tree stops and is counted.
  3. The documented divergences of every set under every preset are pinned per rule in
     `parity/cbst_documented.toml`; `compare.py cbst_trace --store device` fails when a replay's
     counts differ (`--documented-counts none` for the store of another commit:
     `parity/cbst_merge_steps.sh` checks them at the head only).

### 4. Partial failure (PLAN 4.7.3)

Every operation that changes the tree (insert, append, erase, remove_values through `guarded()`;
construct after its validation, rebuild_from_device and compact, except as below, with their own
handlers): **an exception poisons the store**, which then throws `invalid_argument_error` for
every operation but construct and rebuild_from_device, and (M8a review) for the reads `rows()`,
`find()`, `clone()` and `check_errors()`, whose state would be undefined; `download()` is the one
forensic read a poisoned store allows. The guarantees:

| operation | guarantee |
|---|---|
| the precondition checks (a poisoned or unconstructed store) | strong (nothing changes) |
| construct: the validation (`validate_construct`; negative sizes, a capacity below the payload, null arrays, counts beyond their rows, `capacity_error` for the scratch indexing; M8a review: keys not strictly increasing, rows of no slot, unordered or outside the payload) | strong |
| insert, append, remove_values: the batch shape (`validate_batch`, M8a review: one prefix size per key, non-decreasing from 0, the last one the payload size; `capacity_error` beyond `INT_MAX / 3` items) | strong |
| `compact()`: `capacity_error` of compacted rows that do not fit a payload that cannot grow | strong (the store is not poisoned) |
| `initial_capacity()`, `grown_capacity()` (host) | strong |
| every other failure of an updating operation (a `capacity_error` of a fill or insert after its in-place writes, `out_of_memory_error` at any allocation, `internal_error` from the device error word, a CUDA error) | basic: the store is poisoned; nothing is leaked |

A failed fill leaves the values it wrote into the tails, a failed insert its reused slots (the
host model reproduces both: `AFailedInsertLeavesItsInPlaceWrites`). The store does not tell a
failure before its first write from a later one: poisoning is conservative, and the tests check
it at every allocation of an insert and a fill under every preset (`CbstStorePresets`).

### 5. The scenarios, the lockstep and the graph-on-CBST test

- **B's 9 `test_cbst_ops` scenarios and `test_cbst_smoke`** run on the device store under every
  preset (`cbst_scenarios_test.cpp`, `// Derived from MOSP_ESCHER@4b86159:tests/unit/...`), every op
  in lockstep with the host model (`cbst_lockstep.hpp`: outcome, InsertMapping, launches and every
  state field) and read back as B's harness did. Where a scenario depends on the policy the
  expectation is the preset's: churn-small, churn-random and churn-insert compact under the B
  presets and grow under `escher_gpu()`; churn-growth throws only under `mosp_escher()`.
- **The model's `dyng_default()`** (written in step 2 from the plan, independently of the store)
  gets `max_payload`; store and model agree on 40 random operation sequences per preset, on
  permuted and garbage requested keys and on batches of many ties (`cbst_policy_test.cpp`).
- **Graph-on-CBST** (`cbst_graph_test.cpp`, PLAN 6.1 D8: a test, not a product): a graph in an
  edge store (2-vertex records) and an adjacency store as B's `DynamicGraph`, ids from the
  InsertMapping, checked against `dyng::graph::apply` on sorted rows; B's
  `test_dynamicgraph_roundtrip` trace replays under `mosp_escher()`.
- **Reduced sizes** (`DYNG_CBST_REDUCED=1`): the compute-sanitizer runs of `ci/gpu_local.sh` use
  smaller scale, random and churn sizes and 6 instead of 40 random sequences; the full sizes run
  in every `ctest`.

### 6. The mutations (PLAN 8.4, criterion 17)

Seven hooks compiled in by `DYNG_MUTATION_CBST_*` (off in every normal build): the unfill drops
the last value of each removal list (B's #4), the graph keeps its own ids (B's #5, in its test),
the tie order reversed, surplus rows padded by rule 1, the surplus rebuild keeps unmatched dead
slots, the compaction pads 4 more slots, a surplus row takes its requested key. The CTests
`cbst.mutation.<name>` (labels gpu;mutation;cbst) build a copy of libdyng with the mutated
hypergraph module against the replays, the lockstep and the graph test, and pass when the suite
fails (a control copy must pass); `parity/mutate.py run --suites cbst` does the same from
`git archive HEAD`.

## Consequences

- `mosp_escher()` completes B's behaviours: B's traces replay with no mismatch in the `exact`
  field set from its commit on, and dynG's default is checked on every set by the `policy` field
  set, its divergence points listed (`parity/results/M8a.md`, section 13).
- The staging preset `mosp_escher_staged()` is gone; `parity/cbst_merge_steps.sh` still runs it
  for the commits of step 4.
- M8b sizes its trees with `initial_capacity()` of the chosen preset and maps the store's
  exceptions unchanged; the container's own partial-failure rule builds on section 4.
