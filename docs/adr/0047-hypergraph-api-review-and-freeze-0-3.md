# ADR 0047: The hypergraph API review and freeze (0.3)

- **Status:** Accepted under delegation (2026-10-14; GOVERNANCE.md, "Delegation of technical
  ADRs") for the review, its additions and the freeze mechanism. It carries out PLAN 11.3's "the
  hypergraph headers frozen" (the M8 row) at the end of M8b, by the method of ADR 0023 and the
  checklist `docs/developer/api_review_checklist.md`. It changes no rule the author approved: the
  stability policy of PLAN 5.9, the gates and their protocol, the parity rules and the names PLAN
  fixes (`hypergraph`, `hypergraph_storage::escher`, `line_graph_policy::sorted_merge`,
  `id_reuse`, `capacity_policy::escher_gpu()` / `mosp_escher()`, `hyper_apply_summary`) are
  untouched. **Pending the author:** that the frozen hypergraph headers are *stable* from 0.3.0
  (and `.hg` and `.dgb` stable, `.dgb` at format version 1, `.hg` unversioned: ADR 0045) is the
  plan's default and awaits the author's confirmation at the 0.3.0 release; O16 (the `id_reuse` default `erase_first`) is the
  author's decision of 2026-10-07, which is also the plan's default.
- **Date:** 2026-10-14
- **Deciders:** the AI assistant, on the author's behalf (M8b, step python-freeze-records)
- **Builds on:** ADR 0023 (the method, the snapshot), ADR 0035 (the freeze enforced against the
  base branch), ADRs 0041-0046 (the container, the line graph, the update path, the scaffold, the
  files and generators, the Python layer).

## Context

M8b built the hypergraph container on the ESCHER store and froze nothing until its last step:
its headers were *tracked* in the API baseline from their first commit (ADR 0041). PLAN 11.3
freezes them at M8b, so that M9 (`triad_count`, 0.3) and M11 (`hyper_sssp` and the `slack_csr`
storage, 0.4) build on a fixed container API. Before the freeze the API is reviewed as the 0.1
and mosp APIs were (ADRs 0023, 0035), and its fitness for the two later milestones is proven by
code that compiles against it.

Reviewed: every public entity of `cpp/include/dyng/hypergraph/` (`hypergraph_properties.hpp`,
`hyperedge_list.hpp`, `hypergraph_view.hpp`, `hyperedge_batch.hpp`, `hyper_apply_summary.hpp`,
`hypergraph.hpp` with its `detail` contract: `participant_of<hypergraph<V, W>>`,
`hyper_update_participant`, `run_update`, `is_hypergraph_v`), the tracked hypergraph parts of
`io/hypergraph_io.hpp`, the `.dgb` parts of `io/batch_io.hpp`, `generators/hypergraphs.hpp`, the
ESCHER part of `generators/legacy.hpp`, `testing/cbst_host_model.hpp` and
`testing/line_graph_oracle.hpp`, the include list of `dyng/dyng.hpp`, and the Python layer of ADR
0046 (`dyng.Hypergraph` and its companions, `dyng.io`, `dyng.generators`, `dyng.testing`).

## Findings and what was done

| # | Area | Finding | Resolution |
|---|---|---|---|
| 1 | Naming | The names follow ADR 0004 (clang-tidy's naming rules pass): `hypergraph`, `hyperedge_batch(_view)`, `hyperedge_list(_view)`, `hyper_apply_summary`, `incidence_view`, `line_graph_view`, `storage_stats`, `storage_path_counters`, `capacity_policy`, the enumerations `hypergraph_storage`, `line_graph_policy`, `id_reuse` with snake_case values. The PLAN-fixed names are kept. | none needed |
| 2 | The verb vocabulary | `from_hyperedges` (`from_*`), `to_hyperedge_list`, `to_backend` (`to_*`), `apply`, `clone`, `check_integrity` (`check_*`); the views `h2v()`, `v2h()`, `h2h()` and `weights()`, `store_stats()` are nouns. Builders `delete_hyperedge` / `insert_hyperedge` / `delete_incidence` / `insert_incidence` (the plan's builder names, as `edge_batch`'s), `hyperedge_list::add_hyperedge` (as `edge_list::add_edge`), `reserve` / `clear`. | Accepted (ADR 0023 finding 2) |
| 3 | Ids and sentinels (PLAN 4.4.2) | Ids are 0-based everywhere, -1 means none, hyperedge ids and every offset are `int64`; no CBST key, `+1` shift, payload offset or sentinel appears in a declaration. One doc comment (`capacity_policy::row_alignment`) described the row padding with the store's "terminator". | Reworded ("its members plus one slot, rounded up to a multiple of this"): documentation only |
| 4 | Printable enumerations | `to_string` for `hypergraph_storage`, `line_graph_policy`, `id_reuse` (and, tracked, `vertex_relabel`, `batch_dtype`, `hyperedge_batch_mix`, `escher_insert_mode`). | none needed |
| 5 | Planned features | `hypergraph_storage::slack_csr`, `line_graph_policy::slack_csr` and the incidence edits exist in the API and raise `not_supported_error` naming 0.4 (tested from C++ and Python), so M11 adds behaviour, not declarations. | none needed |
| 6 | The four-op batch | `hyperedge_batch_view` holds the four operations; the owning `hyperedge_batch` could add incidence edits but not report them (only through `view()`), unlike its other arrays. | Added `delete_incidence_hyperedge()`, `delete_incidence_vertex()`, `insert_incidence_hyperedge()`, `insert_incidence_vertex()` (additive; tested). |
| 7 | Exception specifications | `hyperedge_batch`'s default constructor was the one function of the headers with neither `@throws` nor `noexcept`. | Declared `noexcept` (it allocates nothing). |
| 8 | Doxygen | Doxygen with `WARN_AS_ERROR` and `ci/doxygen_coverage.py` passed; `@guarantee` was already required on the mutating members of `hypergraph`, `hyperedge_batch`, `hyperedge_list` (ADR 0041). The freeze's rules were not checked: the group, `@throws` on every function, `@sync` / `@async` on functions doing device work without a `resources` argument (the views). | `ci/doxygen_coverage.py` now holds `dyng/hypergraph/` to them (every entity in `@ingroup hypergraph`; `@throws` or `noexcept` on every function; `@sync` / `@async` wherever a `cuda_error` is documented). The headers pass. |
| 9 | Exception guarantees | `apply()`: strong for every error found before the store changes, basic after (the hypergraph poisoned); `dyng::update()` over hypergraphs: strong before the commit, basic after (ADR 0043); the builders strong; `store_stats()` is `noexcept` and returns the counters of the last completed operation (zeros for a moved-from or failing store). | none needed |
| 10 | Host-compilability | Every hypergraph header compiles alone without CUDA, CUB, Thrust or libcu++ (the self-containment targets, ADR 0023 finding 9). | none needed |
| 11 | Ownership and ABI | `hypergraph` is a unique pimpl; the aggregates (`hypergraph_properties`, `capacity_policy`, `hyper_apply_summary`, `storage_stats`, the views) change size when a field is appended, which PLAN 5.9 allows before 1.0. `capacity_policy`'s positional field order is not PLAN 5.2's sketch (ADR 0041 added fields between them). | Accepted for 0.x; the order is frozen as it is (positional initialization of `capacity_policy` is not documented; the presets are the documented way). |
| 12 | Views and versions | Every view carries the `version` it describes; `h2v()` / `v2h()` / `h2h()` are built at most once per version and never on the apply path; `line_graph_view::weights` is host memory while the other arrays are in the container's memory space (each `array_view` carries its space). | Documented in the header (step 2); accepted |
| 13 | `hyperedge_id_bound()` | Can go down after a surplus rebuild (ADR 0041, deviation 10). | Documented in the header; accepted |
| 14 | M9's diagnostic (ADR 0035 finding 16) | A graph result passed to `dyng::update()` with a hypergraph stops in the frozen `update_traits` of sssp, cycle_count and mosp (`no type named 'edge_type'`); the frozen `update.hpp`'s `static_assert` for a wrong batch names `edge_batch_view` for a hypergraph too (ADR 0043 findings 3, 4). | Not changed: `update.hpp` and the graph headers stay byte-identical (M8b criterion 10); both are `detail` text, no SemVer surface; compile-fail tests pin the current diagnostics. A plain-English `static_assert` can be added inside the algorithms' `make_participant` later without an API change. |
| 15 | `testing::rebuild_line_graph` | Throws `std::invalid_argument` / `std::bad_alloc`, not `dyng::error`: the oracle includes no project header so that it shares no code with the container (its isolation test). | Accepted for the testing tier (tracked): documented in its header; the Python binding raises `ValueError` / `MemoryError`. |
| 16 | io, generators, testing | `read_hypergraph` / `write_hypergraph`, the option sets, the `.dgt` and `.dgb` functions, `batch_reader` / `batch_writer`, the generators and the oracles are consistent with the 0.2 io and generators headers. `batch_reader::edge_batch()` / `hyperedge_batch()` share their names with the batch types (inside `batch_reader` the types must be qualified). The by-content resolver of the ESCHER dumps stays on the parity side (ADR 0045): M9's parity replays by content through `.dgb` annotations and `h2v()` (the M9 skeleton), and no user needs it. | Accepted; these headers stay *tracked* (PLAN 5.9: io, generators and testing are frozen with the CLI and the Python layer as a whole, not with the container). |
| 17 | Python | ADR 0046: the C++ names and defaults (tested equal to the native defaults), enumerations as strings, `inserted_ids` int64, int64 members refused without `vertex_dtype="int32"`, the GIL released; every unbound symbol listed in `python_gaps.md`. | none needed |

### Fitness for M9 and M11 (compile-only skeletons)

`cpp/tests/api/fitness/` (the OBJECT library `dyng_api_fitness`, compiled in every preset, never
run) holds what the next milestones need, written against the frozen headers and the internal
access M9 and M11 are allowed (`cpp/src`, internal-stable):

- **M9, `m9_triad_count_fitness.cpp`:** a triad participant of `dyng::update()` through
  `update_traits` and `participant_of<hypergraph<V, W>>` that counts (-) in `before_apply()` on
  the old `h2h()` / `h2v()` rows (0-based, `int64` offsets, no sentinels, in the hypergraph's
  memory space) with the frontier = the live deleted ids of Step 0 (`use_normalized`), reads the
  internal row views of ESCHER-GPU's layout (`hypergraph_access::views()`, after
  `require_line_graph()`), and counts (+) in `after_apply()` with `inserted_ids` on the host and
  in device memory (`hyper_applied::inserted_ids_device`), the deleted member sets and the touched
  sets; pairs are taken from `out[r]` on (each triad by its smallest member); the replay of a
  `.dgb` file by key (`expected_ids` against `inserted_ids`) and by content (the deleted member
  sets resolved against `h2v()`), on sequential, openmp and cuda resources.
- **M11, `m11_hyper_sssp_fitness.cpp`:** the four-op batch (`delete_hyperedge`,
  `insert_hyperedge`, `delete_incidence`, `insert_incidence`, and the view's arrays);
  `id_reuse::insert_first` with `capacity_policy::mosp_escher(extra)`; `(int32, int32)` with a
  weight of 2^28; `line_graph_view::weights` (node weights) and `overlaps`; the `slack_csr`
  enumerators and a `slack_csr_store` implementing every member of `detail::hyper::incidence_store`;
  the incidence-delta hook (`hypergraph_access::apply(..., true)`, `hyper_applied::
  incidence_delta`).

## Decision

1. **The hypergraph headers are frozen** as reviewed, with the additions of findings 6 and 7:
   every declaration of `dyng/hypergraph/*.hpp` (six sections of the baseline, the `detail`
   contract of `hypergraph.hpp` included). `ci/api_snapshot.py` lists `dyng/hypergraph/*` with
   the frozen core; `io/hypergraph_io.hpp`, `io/batch_io.hpp`, `generators/hypergraphs.hpp`,
   `generators/legacy.hpp` and `testing/*` stay tracked. From the freeze a change of a frozen
   hypergraph declaration fails `ci/docs.sh` (against the baseline) and `ci/api_check.sh`
   (against the base branch) unless the pull request has the `api-change` label and a CHANGELOG
   entry. The Python names of ADR 0046 are checked by `griffe` in the same script.
2. **Stability.** By the plan's default the frozen headers are stable from 0.3.0 (PLAN 11.3, 5.9),
   with `.dgb` at format version 1 and `.hg` without a version field (amended, ADR 0045 section 1);
   this awaits the author's confirmation at the 0.3.0
   release (GOVERNANCE.md, open decisions). Until 0.3.0 is released a change is still possible as
   an api-change with a CHANGELOG entry.
3. **O16.** The `id_reuse` default stays `erase_first`: the author's decision of 2026-10-07
   (GOVERNANCE.md), which is the plan's default (PLAN 12.2). Changing it before 0.3.0 would be an
   api-change (a changed default of a frozen declaration; the baseline line
   `id_reuse reuse = id_reuse::erase_first` and `ci/tests/test_api_snapshot.py` pin it).
4. **Both API checks fail on a deliberate hypergraph change**, shown on 2026-10-14 in a throwaway
   clone of this branch at `753449e` (the freeze commit) and reverted (the logs are in
   `$DYNG_SCRATCH/runs/m8b-step5/demo-*.log`):
   - C++: the default of `hypergraph::from_hyperedges(..., const hypergraph_properties& props =
     {})` removed: `ci/docs.sh --doxygen-only` exits 1 with

     ```text
     api-snapshot: the public C++ API differs from the baseline (above: - baseline, + headers).
     -      [[nodiscard]] static hypergraph from_hyperedges(const resources &res, hyperedge_list_view< vertex_t, weight_t > hyperedges, const hypergraph_properties &props={})
     +      [[nodiscard]] static hypergraph from_hyperedges(const resources &res, hyperedge_list_view< vertex_t, weight_t > hyperedges, const hypergraph_properties &props)
     ```

     and, with the baseline regenerated (`ci/docs.sh --update-api`, as a pull request would),
     `ci/api_check.sh 753449e` exits 1: `breaking change of the frozen API:
     [dyng/hypergraph/hypergraph.hpp] - [[nodiscard]] static hypergraph from_hyperedges(...)`.
   - Python: `dyng.Hypergraph.from_lists(..., id_reuse=...)` renamed `reuse=`: the same
     `ci/api_check.sh 753449e` exits 1 with `Hypergraph.from_lists(id_reuse): Parameter was
     removed`.
   - Reverted, `ci/api_check.sh 753449e` passes ("no breaking change" for both halves) and
     `ci/docs.sh --doxygen-only` reports `1095 declarations match public_api.txt`.
   `ci/tests/test_api_snapshot.py::test_the_hypergraph_headers_are_frozen_and_a_change_is_breaking`
   pins the frozen labels and a breaking change of the `id_reuse` default.

## Consequences

- M9 and M11 build on a fixed container API; their headers (`triad_count.hpp`,
  `hyper_sssp.hpp`) are frozen by their own milestones (their manifests' maturity). Additions to
  the hypergraph headers (an appended field, a new accessor) are reported by the API check, not
  failed; a removal or change needs the label.
- The pull request of M8b needs the `api-change` label (the hypergraph headers change from
  tracked to frozen and gain four accessors); its CHANGELOG entry says so.

## Amendments of the M8b review (2026-10-14)

Independent reviewers checked the frozen headers' semantics against the code before the pull
request; each amendment below keeps the decisions above and corrects what the freeze records.

- **Finding 3, `capacity_policy::row_alignment`.** The reworded comment ("its members plus one
  slot, rounded up to a multiple of this", `ceil((n + 1) / a) * a`) contradicted the allocator
  (`aligned(n, a) + 1`, ADR 0041 rule 2: `ceil(n / a) * a + 1`; 5 slots, not 8, for n = a = 4),
  and the Python docstring still said "terminator". Both now say: a surplus row inserted by a
  batch reserves its members rounded up to a multiple of this, plus one slot; the concepts page
  says that its worked example is the construction layout, which `row_alignment` does not set.
- **Finding 12, `line_graph_view::weights`.** The finding said the host memory of `weights` was
  documented in the header, but only `hypergraph::h2h()` and `weights()` said so; the frozen
  struct's own comment said "the arrays live in the container's memory space". The struct and
  the field now say that `weights` is host memory on every backend (the decision stands; M11's
  device weights for `hyper_sssp` are a separate device copy, ADR 0041 section 3).
- **Finding 5, `skipped_ops`.** The field was documented as `ignored_deletions +
  dropped_members`, which left no room for the incidence-edit skips that M11 must count (B's
  `skippedOps` parity, check C9), so "M11 adds behaviour, not declarations" did not hold. The
  field is now documented as *every skipped operation* (in 0.3 that sum), and ADR 0041 section 8
  records the 0.4 rule: incidence edits that B skips are skipped and counted, an out-of-range
  vertex in an incidence edit included.
- **Decision 1, the `detail` contract.** The decision freezes the `detail` contract of
  `hypergraph.hpp` (`participant_of`, `hyper_update_participant`, `run_update`, `is_hypergraph_v`),
  but `ci/api_snapshot.py` read only `participant_of` / `update_traits` / `stats_of` and the
  `*_supported_v` traits from `detail`, and Doxygen excludes `dyng::detail`: a changed virtual
  signature of `hyper_update_participant` passed both checks. The snapshot now also lists the
  `is_hypergraph` trait, `is_hypergraph_v`, the aliases and virtual members of
  `hyper_update_participant` and the declaration of `run_update()` (five `detail:` lines in the
  frozen `hypergraph.hpp` section; `ci/tests/test_api_snapshot.py` checks the extraction). The
  graph's `update_participant` (`update.hpp`) stays out of the listing: it is internal-stable
  (ADR 0023), and `update.hpp` is unchanged by M8b. The types the interface names
  (`hyper_staged_batch`, `hyper_applied`) are internal-stable records, not part of the freeze.
  The deliberate change was shown again (2026-10-14, this branch, reverted; logs in
  `$DYNG_SCRATCH/runs/m8b-review/demo-detail-*.log`): `after_apply()` without its `summary`
  parameter fails `ci/docs.sh --doxygen-only` (the listing differs from the baseline), and with
  the baseline regenerated `ci/api_check.sh` against the commit that added the lines reports a
  breaking change of `[dyng/hypergraph/hypergraph.hpp]`.
