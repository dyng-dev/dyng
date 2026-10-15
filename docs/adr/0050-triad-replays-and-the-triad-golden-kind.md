# ADR 0050: The triad replays (by key and by content) and the triad golden kind

- **Status:** Accepted under delegation (2026-10-15; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. It carries out the replay harness of M9a step 2 (PLAN 6.4.6, 8.3,
  D.1 T6; the harness parts of the M9a acceptance criteria 4 and 8). No approved rule changes:
  PLAN 6.4.6 fixes the replay rule (deletions by content; by key once the M8 id parity holds; both
  must agree) and PLAN 8.3 the exactness (none of the counts may differ).
- **Date:** 2026-10-15
- **Deciders:** the AI assistant, on the author's behalf (M9a, step
  oracle-harness-driver-prereads)
- **Builds on:** ADR 0041 (the container, `erase_first`, the container-level id parity), ADR 0045
  (`.hg`, `.dgb` with `deleted_members` and `expected_ids`), ADR 0048 (the golden set), ADR 0049
  (the oracle).

## Context

The golden set `triad_count_escher_gpu` (ADR 0048) holds, per case, A's full count and per batch
its delta and the counts after it, and per group A's loaded rows, A's vertex count, its H2V
payload capacity and one `.dgb` per batch with the deleted member sets and the ids A's CBST gave
the insertions. Step 2 must replay it through dynG before any engine exists, with an oracle in
place of the counter (the go/no-go), so that steps 3-4 only swap the counter.

## Decision

### 1. The triad golden kind

`parity/goldens.toml` marks the set `kind = "triad_count"`: a case is `<group>/<executable>`
(`main` = h_motif, `type1..3`), its golden is `cases/<group>/<exe>.json` (`full_count`; per batch
`deletions`, `insertions`, `delta`, `counts_after`, `batch_sha256`), and its inputs are the
group's (`group.json`: `input`, `vertices`, `payload_capacity`, the batch files). A replay of a
case compares **counts and batch facts only**: the full count, every batch's net delta and counts
after, the deletions and insertions, and the SHA-256 of each batch file it applied; ids are
compared in the by-key mode (below). The four cases of a group share one replay (one tool run
counts every requested pattern).

### 2. The replay tool `dyng_triad_replay` (`cpp/tests/parity/escher/triad_replay*.{hpp,cpp}`)

- The container is built from the group's `.hg` with A's vertex count, under {`id_reuse::
  erase_first`, `capacity_policy::escher_gpu(payload_capacity)`, `line_graph_policy::
  sorted_merge`} (A's run: its H2H and row views exist from the build).
- **by_key** applies each batch as exported (deletions by id = A's key - 1) and requires
  `hyper_apply_summary::inserted_ids` == the exported `expected_ids`, batch by batch.
- **by_content** resolves each exported deleted member set to the smallest live id with that set
  not taken earlier in the batch (`escher_dump::by_content_resolver`, on the host; ADR 0041
  section 11) and inserts A's rows in A's order; its counts depend on no id. `--mode both` runs
  both on two containers.
- `check_integrity()` after every batch, in each mode.
- **The counter** after the build and after every batch: `oracle` = `testing::triad_reference_all`
  on `to_hyperedge_list()` (the live hyperedges copied to the host) for every requested pattern;
  an evaluation whose list equals the previous one (the other mode in the same state) reuses its
  counts, because the oracle is a function of the list. `algorithm` is the hook steps 3-4 fill in
  (until then the tool exits 2 with "not available"). The report also carries the launch
  counters a counter records (empty for the oracle; the hook of criterion 9), the store's kernel
  paths, and `store_stats().device_bytes` after the build and every batch.
- Exit 0 when the replay ran; the comparison is the caller's.

### 3. The comparison (`parity/compare.py triad_count`, `parity/triad_replay.py`)

`--backend sequential|openmp|cuda [--threads N] --mode by_content|by_key|both --counter
oracle|algorithm [--cases ...]` selects the backend's cases from the coverage table (D6, pending
the author; `--no-coverage` selects among all 300, as the go/no-go does for the syn1m seed-1 grid
on the host backends), decompresses each input once, runs the tool per group and requires, per
case and mode: the full count; as many batches as the golden's, each file with the golden's
SHA-256, deletions and insertions; every delta and counts-after; integrity after every batch; by
key the expected ids; by content no unresolved deletion; and (both) the same counts in both
modes. `--record` writes the per-case verdicts as JSON.

### 4. The harness must fail on planted defects

`--plant bin|row|drop|swap|id` plants one defect after loading a group (a changed bin of the
golden full count; a perturbed member of a deleted row of batch 1, rewritten into a copy of the
`.dgb` with its CRC; batch 2 dropped; batches 1 and 2 swapped; the first expected id of batch 1
changed). `parity/tests/test_triad_replay.py` requires each to fail a replay of
`test/04-small_b20` with the specific finding (the full count; the by-content counts / an
unresolved row; the number of batches; the batch files' SHA-256 and counts; the inserted ids),
and the unmodified replay to pass; the C++ test `dyng_triad_replay_tests` checks the tool's own
reports (oracle counts follow the state in both modes, a wrong expected id is reported by key, a
perturbed row is unresolved by content, the algorithm counter is not available yet) on a case it
writes itself, so it runs on hosted CI without the goldens.

### 5. The go/no-go of step 2

Before any engine code, `--counter oracle --mode both` must reproduce A exactly on make-test, the
syn100k grid and the syn1m seed-1 grid (all four patterns) on the sequential, openmp (8 threads)
and cuda containers, and on `coauth_b50k_s1/type1`. Any miss stops the milestone. Outcome below.

## Outcome (M9a step 2)

**GO.** The oracle reproduces A exactly on all 423 case replays of the go/no-go (make-test and
the syn100k grid: 120 cases, the syn1m seed-1 grid: 20, coauth_b50k_s1/type1: 1; on each of
sequential, openmp with 8 threads and cuda), by key and by content: every full count, delta and
count after; the expected ids by key in every batch; no unresolved deletion by content;
integrity after every batch. Each planted defect fails the replay with its own finding.
Recorded in `parity/results/M9a.md` section 2.2 and `parity/results/M9a-gonogo-escher_gpu.json`.

## Consequences

- Steps 3-4 swap the counter (`--counter algorithm`) and keep the containers, the modes, the
  checks and the comparison; criterion 8's matrix is this harness with the coverage table.
- The oracle's cost bounds where it runs: coauth-DBLP is replayed with the oracle for type 1 only
  (the h-motif reference there costs minutes per evaluation).

## Alternatives considered

- **Comparing in C++ against the golden JSON.** Rejected: the goldens are JSON and the comparison
  rules (coverage, planted defects, records) live in the parity scripts; the tool reports and the
  script judges, as `dyng_hypergraph_ids` and `hypergraph_ids.py` do.
- **Replaying by content only.** PLAN 6.4.6 asks for both once the id parity holds (it does since
  M8b); by key also checks the ids a counter sees as its count(+) frontier.
