# ADR 0060: The triad_count API review and freeze (0.3)

- **Status:** Accepted under delegation (2026-10-21; GOVERNANCE.md, "Delegation of technical
  ADRs") for the review, its amendments and the freeze mechanism. It carries out the author's
  default D4 at the end of M9b (`triad_count`, `dyng.triad_count` and `dyng triad_count` frozen
  and `stable`; `experimental::count_local_patterns` experimental), by the method of ADRs 0023,
  0035 and 0047 and the checklist `docs/developer/api_review_checklist.md`, on the precedent of
  mosp (GOVERNANCE.md, 2026-10-07: the maturity after an API review, with the full gate record in
  the release certificate). **Pending the author:** the maturity `stable` from 0.3.0 (D4) and every
  new public name of M9 (D9, section "The new public names" below) await the author's
  confirmation at the 0.3.0 release; until then a change is still possible as an api-change with
  a CHANGELOG entry. It changes no rule the author approved: the stability policy of PLAN 5.9,
  the gates and their protocol, the parity rules and the frozen headers (`core/*`, `graph/*`,
  `hypergraph/*`, `update.hpp`) are untouched.
- **Date:** 2026-10-21
- **Deciders:** the AI assistant, on the author's behalf (M9b, step docs-freeze-release); the
  maturity and the names: S M Shovan (pending)
- **Builds on:** ADR 0023 (the method, the snapshot), ADR 0035 (the freeze enforced against the
  base branch, the maturity decides the freeze), ADR 0047 (the freeze's Doxygen rules), ADRs
  0048-0059 (triad_count, its engine, its gates, the experimental extension point, Python and the
  command line).

## Context

`triad_count` arrived in M9a (ADRs 0052-0057) at maturity `experimental`, its header tracked by
the API baseline; M9b added `experimental::count_local_patterns` over its driver (ADR 0058),
`dyng.triad_count`, `dyng.update()` over hypergraphs and `dyng triad_count` (ADR 0059). The
plan's default D4 freezes the C++ header, the Python module and the command line at the end of
M9b and makes the algorithm `stable` in 0.3.0, as mosp became stable in 0.2.0. From then on a
change of them is a SemVer matter, so the API is reviewed first, the way ADRs 0023, 0035 and 0047
reviewed the earlier APIs: against the checklist (naming and the verb vocabulary, the uniform
contract of ADR 0006, Doxygen, exception guarantees and `dyng::error`, host-compilability,
ownership and ABI, the bindings), against its aggregate-delta sibling `cycle_count`, and against
what the coming work could force to break (0.4: the coarse triangles, the OpenMP tuning, int64
hypergraph offsets, `hyper_sssp` on the same container).

Reviewed: every public entity of `cpp/include/dyng/triad_count.hpp` (`pattern`, `to_string`,
`max_bins`, `options`, `stats`, `result`, `compute()`, `update()`, and the `detail` contract the
baseline lists: `triad_count_supported_v`, `update_traits<triad_count::result>`, the two
`DYNG_TRIAD_COUNT_*` messages); the Python module `dyng.triad_count` (`python/dyng/triad_count.py`,
`python/bindings/triad_count.cpp`) with `dyng.update()` over hypergraphs; the command line
`dyng triad_count compute|update` (`python/dyng/cli/_triad_count.py`, `docs/api/cli.md`); and
`docs/algorithms/triad_count.md`. Out of scope, as tracked or experimental: `<dyng/testing/
triad_reference.hpp>`, the triad bins of `<dyng/io/result_io.hpp>`, `generators::legacy::
escher_host_model` and the experimental headers `dyng/experimental/*`.

## Findings and what was done

| # | Area | Finding | Resolution |
|---|---|---|---|
| 1 | Naming | The names follow ADR 0004 (clang-tidy's naming rules pass): `pattern` with snake_case enumerators (`h_motif`, `stathyper_type1..3`), `options::kind` (PLAN 5.5's `kind=`), `max_bins`, `stats` with `deleted_frontier`, `inserted_frontier`, `triads_removed`, `triads_added`, `delta`; the algorithm is named for what it computes, the papers (ESCHER, ESCHER+) are in `@paper`. | none needed |
| 2 | The verb vocabulary | `compute`, `update`, `clone`, `set_options`, `get_options` (ADR 0023's accepted exception), the accessors are nouns (`kind`, `num_bins`, `counts`, `count`, `total`, `graph_version`, `space`), `to_string` for the enumeration. | none needed |
| 3 | Contract uniformity with `cycle_count` | The same skeleton: an aggregate `options` (fields appended only), `stats : update_stats` with `batch` (here a `hyper_apply_summary`) first, an opaque move-only pimpl `result` that is not a template (a host array of counts, as `cycle_count::result`), `compute(res, h, opt)` never mutates `h` (the automatic line graph aside, documented), `update(res, h&, batch, r&)` applies the batch and throws `stale_result_error`, `update_traits` for `dyng::update()`. Differences: the per-bin signed `delta` is a fixed-size `std::array<std::int64_t, 30>` (the sketch's open question 1, ADR 0052: no allocation per update), where cycle_count's stats have only the phase sums. | Accepted |
| 4 | Fixed inputs | `kind` is fixed at `compute()` but lives in `options` (as `mosp::options::preferences`); `set_options()` refuses a change and names the fix. | Accepted (ADR 0035 finding 4) |
| 5 | `stats` | The inherited counters are documented in the header (`affected`, `iterations`, `frontier_visits`, `fallback_used`, `converged`, `engine_used`). `triads_removed` / `triads_added` said "for the weighted types 1-3: the sum of the destroyed triads' weights", which reads as hyperedge weights (ignored by every pattern). | Reworded: for the types 1-3, whose one bin is a sum of contributions (C(\|e\|, 3), C(a, 2) b c, x y z), the contributions the batch removed / added (`046a6cf`, documentation only; the Python `Stats` docstring the same) |
| 6 | Doxygen | `compute()`, `update()` carry `@backends`, `@determinism`, `@paper` (keys `escher2026`, `escherplus2026`) and `@guarantee`; `compute()`'s description described the h_motif shape only (out-wedges closed into triangles) for every pattern. The freeze's rules of ADR 0047 (every entity in the group, `@throws` or `noexcept` on every function, `@sync` / `@async` with a `cuda_error`) were not applied to the header, and `triad_count::result` was not among the classes whose mutating members must state `@guarantee`. | `compute()` describes each pattern's shape and the CUDA launch (`046a6cf`); `ci/doxygen_coverage.py` holds `dyng/triad_count.hpp` to the freeze's rules and `triad_count::result` to `@guarantee` (`7b8234f`); the header passes |
| 7 | Exceptions leaving the library | The header documents `out_of_memory_error` for `compute()`, `update()` and `clone()`, but `triad_count_compute`, `triad_count_update` and `result::clone` had no `DYNG_TRANSLATE_ALLOCATION_FAILURE` (cycle_count's and mosp's entry points have it): a host `std::bad_alloc` of `compute()` or `clone()` (the host engine's arrays, the state) left the library as `std::bad_alloc`. The same held for `count_local_patterns`' compute, update and clone. | **Fixed** (`b732aa9`, the only library change of the review): the six entry points end in the handlers; `dyng_triad_count_allocation_failure_tests` fails every host allocation of `compute()` (four patterns, sequential and openmp) and `clone()` in turn and requires `out_of_memory_error` (it reported `std::bad_alloc` before the fix). No signature changed; the counting kernels' machine code is unchanged (`triad_sass.py compare` against M9a's merge build: identical), and the gates were read on this library (`parity/results/M9b.md` section 3). |
| 8 | A container defect found by the same test | Failing a host allocation inside `update()` showed that the hypergraph container's OpenMP regions on the host backends (`incidence_model::v2h_append` in the commit; the line graph's row build under `line_graph_policy::automatic`) let a `std::bad_alloc` escape the parallel region, which ends the process (`std::terminate`) instead of an `out_of_memory_error` that poisons the hypergraph. | **Not changed here**: the code is `cpp/src/hypergraph` (M8b; the frozen container's implementation), outside M9b's scope, and a change there needs the ADR 0024 bar of the container's maintain reading. Reported to the author (the retrospective's decision list); the allocation-failure test sweeps `compute()` with a prebuilt line graph and leaves `update()` out, with the reason in its source. The device path is unaffected (M9a's OOM sweep of every device allocation of an update passes). |
| 9 | Exception guarantees | `compute()` strong; `update()`: strong before the commit, basic inside it (the hypergraph poisoned, the result unchanged and not poisoned: the M9a review's third outcome), basic after it (the result poisoned); `set_options()` strong (a trivially copyable assignment after the checks). | none needed |
| 10 | Host-compilability | `<dyng/triad_count.hpp>` includes no CUDA, CUB, Thrust or libcu++ header; the four patterns are built-in Ops pre-instantiated in libdyng (ADR 0058), so the header stays host-only; its self-containment unit builds in every preset; a CXX-only consumer of the installed package counts with it (`ci/cmake_consumer.sh`). | none needed |
| 11 | Ownership and ABI | `result` is a unique pimpl; `options` and `stats` are aggregates whose size changes when a field is appended (PLAN 5.9 allows it before 1.0). `stats::delta` is sized by `max_bins`: growing `max_bins` would change `stats` and the baseline. | Documented on `max_bins` (`046a6cf`): a pattern appended later has at most 30 bins; a count of more bins is an Op of `count_local_patterns` (whose bins are its own). The 0.4 coarse triangles (four values) fit. |
| 12 | Limits | Exact 64-bit counts on every backend, `capacity_error` past 2^64 - 1 or for a net change of a bin outside `stats::delta`'s range, `h_motif` and type 3 refuse 2,147,483,647 vertices (ADR 0057, ADR 0052). In the baseline (`max_bins = 30`). | Accepted: a later release may relax a limit (an api-change with a CHANGELOG entry), never tighten it |
| 13 | Composition | `dyng::update(res, h, b, r1, r2, ...)` composes triad_count results with each other and with experimental `count_local_patterns` results through one commit (conformance C10; `examples/cpp/triad_count_update.cpp`); a graph result with a hypergraph stops in the frozen `update_traits` (ADR 0047 finding 14, unchanged). The detail contract in the baseline is `triad_count_supported_v` and `update_traits<triad_count::result>`; `make_triad_count_participant` is the compiled hand-off behind the frozen traits (internal-stable, as the graph's `update_participant`, ADR 0047). | Accepted |
| 14 | 0.4 | Coarse triangles: new enumerators appended to `pattern` (with a property requirement such as a vertex graph checked per pattern by `compute()`, as today's), additive; the OpenMP tuning (roadmap) is behaviour only; int64 hypergraph offsets add instantiations (`triad_count_supported_v` widens, additive); `hyper_sssp` and `slack_csr` do not touch this header. | No API change needed |
| 15 | Python | `dyng.triad_count` mirrors the C++ API (ADR 0059): `Options` with the C++ fields and defaults, `Pattern` (a `StrEnum`) and the plain names accepted everywhere, `Result` (`counts` a read-only uint64 NumPy copy, `count`, `total`, `num_bins`, `kind`, `options`, `set_options`, `graph_version`, `space`, `clone`), the frozen dataclass `Stats` (`delta` int64 of `num_bins` values, `batch.inserted_ids` int64), `compute(hypergraph, *, kind, options, resources, **kwargs)`, `update(hypergraph, batch, result, *, resources)`. The docstrings of `compute()` / `update()` listed fewer exceptions than the C++ `@throws`. | `compute()` and `update()` list `CapacityError`, `OutOfMemoryError`, `InternalError` and the poisoned result (documentation only); `Stats.triads_removed` as finding 5 |
| 16 | Command line | `dyng triad_count compute|update`: one flag per option field (`--kind`, `--cuda-engine`), `dyng.io.read_hypergraph`'s keywords as read flags plus `--read-options` and `--min-initial-payload`, `--input` / `--hypergraph`, `--batches` (`.dgt` / `.dgb`, the expected ids checked), `--format json|escher`, `--output`, `--compare-recompute`; `--format escher` is ESCHER-GPU's output byte for byte; `--format json` the triad-bins JSON at format version 1. Under PLAN 5.9 these flags and outputs are now stable. | none needed; whether the triad-bins format version 1 is stable from 0.3.0 belongs with the `.hg` / `.dgb` open decision |
| 17 | Tracked and experimental neighbours | `testing::triad_reference`, the triad bins of `io/result_io.hpp` and `generators::legacy::escher_host_model` stay *tracked* (PLAN 5.9: io, generators and testing are frozen as a whole later; ADR 0035 finding 19); `dyng/experimental/*` stays in the API snapshot's experimental tier (reported, never failing; ADR 0058). | Unchanged; not frozen by this ADR |

## Decision

1. **The triad_count API is frozen** as reviewed, with the documentation amendments of findings
   5, 6, 11 and 15 and the fix of finding 7: every declaration of `<dyng/triad_count.hpp>` (its
   baseline section, the `detail` contract included), the Python names of `dyng.triad_count` (and
   `dyng.update()` over a `dyng.Hypergraph`), and the flags and output formats of `dyng
   triad_count`. No signature changed in the review, so the freeze needs no migration note.
2. **Maturity.** The manifest says `maturity = "stable"` (with its `origin` of ESCHER-GPU@abcf9b2
   unchanged), so `ci/api_snapshot.py` lists `dyng/triad_count.hpp` as frozen (ADR 0035 decision
   2) and `scripts/regen.py` writes `stable` into the algorithm tables and the registry. This is
   the plan's default D4 and **awaits the author's confirmation at the 0.3.0 release**
   (GOVERNANCE.md, open decisions, next to M8b's hypergraph item); the 0.3.0 certificate carries
   triad_count's full gate record (both ESCHER suites, read again on M9b's library), as mosp's did
   for 0.2.0.
3. **The baselines.** The C++ baseline `cpp/tests/api/api_snapshot/public_api.txt` lists the
   section `[dyng/triad_count.hpp] frozen` (`ci/docs.sh --update-api`); from the merge of M9b, a
   removed or changed declaration of it fails `ci/docs.sh` (against the baseline) and
   `ci/api_check.sh` (against the base branch) without the `api-change` label and a CHANGELOG
   entry. The Python half has no file of its own: `ci/api_check.sh` runs griffe against the base
   branch, so `dyng.triad_count` is checked from the commit that merges it into `main`; the tests
   below pin the same rule now.
4. **Both checks fail on a deliberate change** (shown on 2026-10-21 in a throwaway clone of this
   branch, reverted; logs in `$DYNG_SCRATCH/runs/m9b-step3/demo-*.log`), and the tests keep it:
   - `ci/tests/test_api_snapshot.py::test_the_triad_count_header_is_frozen_and_a_change_is_breaking`:
     the section is frozen, the oracle and `result_io.hpp` tracked, `local_patterns.cuh`
     experimental; a changed default of `options::kind`, a removed `stats::triads_added` and a
     changed `max_bins` are breaking changes of `[dyng/triad_count.hpp]`;
   - `ci/tests/test_api_check.py`: a renamed parameter of `dyng.triad_count.update`, a changed
     default of `Options.kind` and a removed `Result.total` are breaking changes for griffe;
   - the demonstration itself is recorded under "Outcome" below.
5. **The experimental tier stays unfrozen**: a change of `dyng/experimental/local_patterns.cuh`
   is reported by `ci/api_snapshot.py`, never failed (ADR 0058, `ci/tests/test_api_snapshot.py::
   test_an_experimental_change_is_reported_not_failed`).

## The new public names (for the author; D9)

Every public name M9 introduces, for the author's confirmation at the 0.3.0 release (naming is
not delegated). Frozen by this ADR (pending D4):

- **C++ `dyng::triad_count`** (ADR 0052): `pattern` and `h_motif`, `stathyper_type1`,
  `stathyper_type2`, `stathyper_type3`; `to_string(pattern)`; `max_bins`; `options` (`kind`,
  `cuda_engine`); `stats` (`batch`, `deleted_frontier`, `inserted_frontier`, `triads_removed`,
  `triads_added`, `delta`); `result` (`kind`, `num_bins`, `counts`, `count`, `total`,
  `get_options`, `set_options`, `graph_version`, `space`, `clone`); `compute`, `update`; the
  macros `DYNG_TRIAD_COUNT_TYPES_MESSAGE`, `DYNG_TRIAD_COUNT_CONTAINER_MESSAGE`; the detail
  contract `triad_count_supported_v`, `update_traits<triad_count::result>`; the profiler stages
  `triad_count.compute`, `.count`, `.reset`, `.update`, `.normalize`, `.count_minus`, `.commit`,
  `.identify_affected`, `.count_plus`, `.finalize`, `.enact_fused` (the benchmark suites read
  `triad_count.update` and `triad_count.compute`).
- **Python `dyng.triad_count`** (ADR 0059): `Pattern` and its four members, `PatternName`,
  `MAX_BINS`, `Options` (`kind`, `cuda_engine`), `Stats` (`affected`, `iterations`,
  `frontier_visits`, `fallback_used`, `converged`, `engine_used`, `batch`, `deleted_frontier`,
  `inserted_frontier`, `triads_removed`, `triads_added`, `delta`), `Result` (`counts`, `count`,
  `total`, `num_bins`, `kind`, `options`, `set_options`, `graph_version`, `space`, `clone`),
  `compute(hypergraph, *, kind, options, resources, **kwargs)`, `update(hypergraph, batch, result,
  *, resources)`; `dyng.update()` over a `dyng.Hypergraph`; `dyng.citation("triad_count")`.
- **The command line** (ADR 0059): `dyng triad_count compute|update` with `--kind`,
  `--cuda-engine`, `--input` / `--hypergraph`, the read flags (`--index-base`,
  `--dedupe-hyperedges`, `--max-cardinality`, `--relabel`, `--num-vertices`, `--weighted`,
  `--storage`, `--line-graph`, `--id-reuse`, `--capacity`, `--store-v2h`), `--read-options`,
  `--min-initial-payload`, `--batches`, `--format json|escher`, `--output`,
  `--compare-recompute`.

Tracked or experimental (named for the author, not frozen):

- **The experimental extension point** (ADR 0058 section 6): `dyng::experimental`;
  `count_local_patterns<op_t>` (`valid`, `result`, `stats`, `compute`, `update`);
  `local_patterns_result<op_t>` (`op_type`, `num_bins`, `counts`, `count`, `total`, `op`,
  `graph_version`, `space`, `clone`); `local_patterns_stats` (`batch`, `deleted_frontier`,
  `inserted_frontier`, `removed`, `added`, `delta`); `shape` (`triangles`, `pairs`, `keys`) and
  `to_string(shape)`; `max_pattern_bins`; the **Op member names** `num_bins`, `shape`,
  `begin_wedge`, `triangle`, `pair`, `key`, `flush` and the bins handle's `add`, `overflow`; the
  rows `member_code`, `member_span` (`begin`, `end`, `size`, `empty`, `operator[]`),
  `hyperedge_rows` (`row`, `size`, `decode`), `venn_counts` (`ab`, `bc`, `ca`, `abc`),
  `intersect_size`, `venn3`, `intersects3`; the **accessor** `row_access` (`hyperedges`,
  `id_bound`, `has_line_graph`, `space`, `device`) and `access_rows`; the macros `DYNG_HD`,
  `DYNG_HD_INLINE`, `DYNG_LOCAL_PATTERNS_TYPES_MESSAGE`; the stages `count_local_patterns.*`.
- **The oracle** (ADR 0049, tracked): `dyng::testing::triad_pattern` and its enumerators,
  `triad_num_classes`, `triad_num_venn_ids`, `triad_pattern_name`, `triad_num_bins`,
  `triad_class_representatives`, `triad_venn_permuted`, `triad_venn_canonical`,
  `triad_motif_class`, `triad_motif_class_map`, `triad_venn_id`, `triad_reference_options`
  (`threads`), `triad_reference_counts` (`h_motif`, `type1`-`type3`, `of`), `triad_reference_all`,
  `triad_reference`, `triad_reference_removed`, `triad_reference_added`; Python
  `dyng.testing.triad_reference`.
- **The triad bins** (ADR 0059, tracked): `io::triad_bins` (`kind`, `graph_version`, `counts`,
  `deltas`), `io::triad_bins_size`, `io::write_triad_bins_json`, `io::read_triad_bins_json`, the
  format `"dyng.triad_bins"` version 1; Python `dyng.io.TriadBins` (`kind`, `counts`,
  `graph_version`, `deltas`, `of`), `write_triad_bins`, `read_triad_bins`, `triad_bins_json`.
- **The generators' host-model addition** (ADR 0051, tracked):
  `dyng::generators::legacy::escher_host_model` (its constructor, `apply`, `num_vertices`,
  `num_hyperedges`, `num_rows`, `row`) and `escher_batch_generator::next(model)`.
- **The drivers** (tools, for parity): `dyng-compat-escher run` (ESCHER-GPU's argv; ADR 0051).

## Outcome

**The deliberate changes** (2026-10-21, a throwaway clone of this branch at `155f4ef`, the commit
after the freeze, reverted and deleted; logs `$DYNG_SCRATCH/runs/m9b-step3/demo-*.log`):

- C++: the default of `triad_count::compute(..., const options& opt = {})` removed:
  `ci/docs.sh --doxygen-only` exits 1 with

  ```text
  api-snapshot: the public C++ API differs from the baseline (above: - baseline, + headers).
  -  template <typename vertex_t, typename weight_t> [[nodiscard]] result dyng::triad_count::compute(const resources &res, const hypergraph< vertex_t, weight_t > &h, const options &opt={})
  +  template <typename vertex_t, typename weight_t> [[nodiscard]] result dyng::triad_count::compute(const resources &res, const hypergraph< vertex_t, weight_t > &h, const options &opt)
  ```

  and, with the baseline regenerated (`ci/docs.sh --update-api`, as a pull request would),
  `ci/api_check.sh 155f4ef` exits 1: `breaking change of the frozen API: [dyng/triad_count.hpp]
  - template <...> [[nodiscard]] result dyng::triad_count::compute(...)`.
- Python: `dyng.triad_count.update(hypergraph, ...)` with `hypergraph` renamed `hg`:
  `ci/api_check.sh 155f4ef` exits 1 (`Parameter was removed`, `Parameter was added as
  required`).
- Experimental: `experimental::max_pattern_bins` changed from 1024 to 2048: `ci/docs.sh
  --doxygen-only` passes and reports the changed line of the experimental section, and
  `ci/api_check.sh 155f4ef` (baseline regenerated) passes: "no breaking change of the frozen C++
  API".
- Reverted: `ci/api_check.sh 155f4ef` passes for both halves, and `ci/docs.sh --doxygen-only`
  reports `1244 declarations match public_api.txt`.

**The gates on the reviewed library.** Finding 7 changed the library (`b732aa9`), so M9a's gate
records no longer had the release's library sources (`certify.py`'s library scope). Every M9a
timing and memory gate was read again on `b732aa9` with M9a step 5's protocol, both suites
passed (45 / 45 and 11 / 11), and the dry 0.3 certificate passes over them
(`parity/results/M9b.md` section 3).

## Consequences

- From 0.3.0 (on the author's confirmation), `triad_count` is covered like `sssp`, `cycle_count`
  and `mosp`: the algorithm tables, the registry (`dyng::algorithms()`, `dyng.algorithms()`) and
  `docs/algorithms/triad_count.md` say `stable`.
- The pull request of M9b needs the `api-change` label (a header changes from tracked to frozen)
  and has its CHANGELOG entry.
- Open for the author: the container's OpenMP allocation-failure defect of finding 8, a change of
  `cpp/src/hypergraph` with its ADR 0024 reading.
