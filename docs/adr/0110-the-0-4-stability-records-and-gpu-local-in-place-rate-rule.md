# ADR 0110: The 0.4 stability records, and `ci/gpu_local.sh`'s in_place cloud under the rate rule

- **Status:** Accepted (2026-11-20; R040, step stability-and-carryovers). Both parts carry out
  decisions the author took on 2026-11-18 (relayed by the orchestrator's session to R040;
  GOVERNANCE.md, approvals log, the rows "AD15 and HD17: the stability of 0.4's surfaces" and
  "ADR 0111's rate rule at the release and in `ci/gpu_local.sh` (R040)"); their mechanism is a
  technical choice, accepted under delegation (GOVERNANCE.md, "Delegation of technical ADRs"),
  technical scope only. Neither part changes a rule the author approved: PLAN 5.9's stability
  policy, ADR 0111's rule (its approved wording is not rewritten; ADR 0111 gets only the dated
  note "Carried out in R040: ADR 0110"), the frozen envelope, the thresholds, the calibration,
  the gates and their protocol, and the parity rules are untouched.
- **Date:** 2026-11-20
- **Deciders:** S M Shovan (the decisions, 2026-11-18); the AI assistant, on the author's behalf
  (the mechanism)
- **Builds on:** ADR 0035 (an algorithm's header is frozen by its manifest), ADR 0060 (the last
  algorithm made stable at a release), ADR 0096 (hyper_sssp's freeze), ADR 0109 (label_propagation's
  freeze), ADR 0111 (the rate rule of the in_place certificate).
- **Number:** the first of the release line's block 0110-0119 (ADR 0063); ADR 0111 is the rate
  rule (it took a number of this block, M10b's review fixes), ADR 0120 the stream-ordering fix.

## Part A: the stability records of 0.4

### Context

AD15 (M10) and HD17 (M11), decided by the author on 2026-10-28, froze everything new of 0.4 at the
end of M10b and M11b and made it `stable` at 0.4.0 only on the author's confirmation at the
release. Until then label_propagation's manifest said `experimental` while
`ci/api_snapshot.py`'s `FROZEN_BEFORE_STABLE` froze its header and the headers only it added (the
vertex-arrival generators and three testing oracles); hyper_sssp's manifest already said `stable`
(ADR 0096 set it at the freeze, as ADR 0060 did for triad_count). The author decided on 2026-11-18
which of these surfaces are stable from 0.4.0, and that the three testing oracles stay frozen but
**outside** the stable tier, as the testing oracles of 0.1-0.3 are.

### Decision

1. **The tiers** (the author's decision, item by item):

   | Item | Tier from 0.4.0 |
   |---|---|
   | `<dyng/label_propagation.hpp>`, `dyng.label_propagation`, `dyng label_propagation` | stable |
   | the graph additions of M10 (in the frozen core headers `graph/*`): `graph::with_capacity`, slot and slack rows, `vertex_status`, `graph_view::status` / `row_end`, `num_active_vertices`, `net_effect()`, the vertex operations of `edge_batch`, the `apply_summary` fields, the float instantiations; their Python counterparts | stable |
   | the vertex-arrival generators: `<dyng/generators/vertex_arrival.hpp>`, Python, `dyng generate vertex_arrival_stream` / `dynlp_vertex_arrival_stream` | stable |
   | `io::write_labels` / `read_labels` (frozen by declaration inside the tracked io headers) | stable |
   | AD13's format extensions: `.dgt` `+v` / `-v` and float weight tokens, `.dgb` weight code 3 and flag bit 4 `vertex_records`, the label files | stable tier of the formats |
   | `<dyng/hyper_sssp.hpp>`, `dyng.hyper_sssp`, `dyng hyper_sssp` | stable |
   | the `slack_csr` storage and the incidence edits (in the frozen `hypergraph/*` headers and `dyng.Hypergraph`), HD10's format extensions (`.dgt` `+i` / `-i`, the `.dgb` flag `incidence_edits`) | stable; stable tier of the formats |
   | `dyng::testing::harmonic_solution`, `union_find`, `lp_host_model` (and `dyng.testing`'s counterparts) | frozen, outside the stable tier |

2. **The manifests.** label_propagation's maturity is `stable`; `scripts/regen.py` writes it into
   the README, landing-page and algorithm-index tables, the C++ registry and `dyng.algorithms()`.
   hyper_sssp's manifest keeps `stable`, with its comment naming the decision.
3. **The API baseline** (`ci/api_snapshot.py`). `FROZEN_BEFORE_STABLE` is gone; it is replaced by
   two lists whose names say what they are:
   - `STABLE_HEADERS`: stable headers besides the core and the algorithms' own headers, today
     `dyng/generators/vertex_arrival.hpp` (the other `generators/*` headers stay tracked);
   - `FROZEN_OUTSIDE_STABLE`: the three oracle headers, frozen by the API check (a changed
     declaration needs the `api-change` label and a CHANGELOG entry) but promised to no user.
   `dyng/label_propagation.hpp` is frozen through its manifest (`stable_algorithm_headers()`), so
   the maturity and the freeze cannot disagree (ADR 0035). Every section of
   `cpp/tests/api/api_snapshot/public_api.txt` keeps its status (all of them were `frozen`); only
   the baseline's comment, which the script writes, names the two groups. The io declarations of
   `FROZEN_DECLARATIONS` are unchanged; their comment says they are stable from 0.4.0.
4. **The documents.** The API pages (`label_propagation`, `hyper_sssp`, `graph`, `hypergraph`,
   `generators`, `io`, `testing`), the Python page, the format page (its top and the three
   sections), the algorithm pages and index, the landing page's sentence on 0.4, README's status
   rows, the design sketch, the roadmap and the CHANGELOG say what is stable from 0.4.0; the
   testing page says that the oracles are outside the stable tier.
5. **Tests.** `ci/tests/test_api_snapshot.py`: every algorithm header is frozen exactly when its
   manifest says stable (label_propagation, hyper_sssp and triad_count among them, no algorithm
   header in `STABLE_HEADERS`); the two lists hold exactly the headers above; a changed default of
   label_propagation's options, a removed stats field, a changed generator default and a changed
   oracle signature (`harmonic_options::relative_tolerance`, `union_find::unite`) are breaking
   changes of their sections.

### Outcome

**The deliberate change** (a throwaway clone of `release-0.4.0` at `744b768`, deleted after; the
log `$DYNG_SCRATCH/runs/r040-stability/demo.log`; R040's retrospective): the committed tree passes
`ci/api_check.sh d3375da` (the merge of main the branch starts from). With
`label_propagation::update()` taking its batch view by value instead of by const reference,
`ci/docs.sh --doxygen-only` fails against the baseline (the one changed declaration), and with the
baseline regenerated (`ci/api_snapshot.py --update`) `ci/api_check.sh d3375da` fails: "breaking
change of the frozen API: [dyng/label_propagation.hpp] ...". Reverted, both pass again
("api-snapshot: OK (1442 declarations match public_api.txt)").

## Part B: `ci/gpu_local.sh`'s in_place cloud under the rate rule

### Context

ADR 0111 (the author's rule of 2026-11-18, its wording approved as written) judges the
`label_propagation_inplace` certificate by a rate rule: per case 30 port runs and 30 runs of the
unpatched original on one GPU in one session, the case failing when the port's count of runs
outside the frozen iteration envelope is significantly higher (a one-sided Fisher exact test at
5 %), with every port run's labels, components, iteration sum and the deterministic trace still
checked run by run. `ci/gpu_local.sh`'s label_propagation step kept the every-run cloud (5k5 and
50k, 3 port runs each, every run inside the envelope), which the original itself fails about one
time in five at 50k (ADR 0111, Context); the author decided that the local check moves to the
rate rule in R040, before the release readings.

### Decision

1. **The step's in_place part** runs `parity/compare.py label_propagation inplace --rate --exe
   <dyng-compat-dynlp> --original <dynLP> --cases 5k5,50k --json
   <records>/inplace-cuda.json`: the cases it checked before, the rule's 30 port and 30 original
   runs per case alternated on the step's GPU (`CUDA_VISIBLE_DEVICES`, GPU 1 by default) in one
   session, the rule's run-by-run checks on every port run (`checked_every_run`) and the
   deterministic trace once per case, as the certificate's comparator does (ADR 0111 section 3).
   The comparator refuses another rule, another run count and a recording twin as the original;
   nothing of it changes here.
2. **The original** is the unpatched build of `parity/build_reference.sh --variant unpatched
   LabelPropagation-CUDA` (the copy the comparator would use by default; the step passes it
   explicitly so that a missing build is seen before the runs). Without it the in_place part is
   skipped and the step is **"partial"**, as it is without the goldens; `parity/certify.py`
   accepts only "passed", so a certificate cannot pass on a step that skipped the rule.
3. **The summary row keeps its name** ("label_propagation (CUDA tests, kit, replays of P,
   in_place cloud, oracle; sanitizers)"): the in_place cloud is still read, by the certificate's
   rule. `parity/certify.py`'s `GPU_LOCAL_STEPS[(0, 4)]` and its tests are therefore unchanged.
4. **A failure is a STOP** (the author's decision of 2026-11-18): a case that fails the rate rule
   in this step is not read again; if the step runs again for another reason, its first result
   stays recorded and a failure in either run is a STOP.
5. **The test** `parity/tests/test_release_cert.py::test_gpu_local_judges_the_in_place_cloud_by_the_rate_rule`
   reads the script: the label_propagation step has exactly one in_place call, with `--rate`, no
   `--runs`, the cases 5k5 and 50k, the original of the comparator's reference, the record path
   unchanged, and a missing original makes the step "partial" (never "passed").

### Consequences

- The local check is now exactly as strict as the certificate on these two cases, no longer
  stricter: a port whose tail is like the original's passes it; a heavier tail, any systematic
  change of the iteration counts, and any run-by-run failure fail it.
- The step takes 114 runs more than before (60 per case instead of 3); at 5k5 and 50k that is
  about two minutes on an RTX A5000 (the record below).
- The certificate's own in_place reading (all four cases, `compare.py label_propagation
  certificate --kind inplace`) is unchanged; the step's record is a local check, not a
  certificate part.

### Outcome: the step on GPU 1

`ci/gpu_local.sh --preset parity-cuda` with every other step skipped, GPU 1 after `quiet-ok
R040`, on a clean tree at `d0cbf07` (2026-11-20 01:58-02:40 UTC, the build included): build
passed, label_propagation passed, no sanitizer run retried (ADR 0112). The rate rule's record
(`parity/results/R040/gpulocal-lp-inplace-rate.json`; the summary
`parity/results/R040/gpulocal-lp-summary.md`): **5k5** the port 0 / 30 and the original 0 / 30
runs outside the envelope (p = 1); **50k** the port 0 / 30 and the original 4 / 30 (at batches 8,
6, 6 and 6; p = 1); every port run's labels within T_mean / T_max / T_flips of every reference
run (largest mean 1.552e-04 of 4.280e-04 and 2.185e-04 of 5.503e-04), its components exact and
its iteration sum inside the frozen bounds; the deterministic traces equal (141 and 281 fields, 0
mismatched).

## Alternatives considered

- **Keep `FROZEN_BEFORE_STABLE` and add label_propagation's header to it**: the name would say
  "before stable" of headers that are stable, and the oracles would not be told apart from the
  stable headers.
- **A new section status `frozen-outside-stable` in the baseline**: every reader of the baseline
  (the check against the base, the tests, the docs) would need to learn it, for a difference
  that matters to users, not to the check; the lists in the script and the testing page say it.
- **Rename the summary row** (for example "in_place rate rule"): it would need the same change in
  `parity/certify.py` and its tests in the same commit, for no gain in what the row means.
- **All four cases in `ci/gpu_local.sh`**: 500k and 5m take most of the reading's time, and the
  certificate reads them; the local check keeps its two cases.
