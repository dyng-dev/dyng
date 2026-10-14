# ADR 0044: The hypergraph scaffold and the requirements of a hypergraph manifest

- **Status:** Accepted under delegation (2026-10-14; GOVERNANCE.md, "Delegation of technical
  ADRs"). It carries out the scaffold part of M8b step 3 (PLAN 4.8, 9.4; M8b acceptance criterion
  11; ADR 0022 decision 8, which deferred `--container hypergraph` to the hypergraph container)
  and changes no rule the author approved.
- **Date:** 2026-10-14
- **Deciders:** the AI assistant, on the author's behalf (M8b, step update-kit-scaffold)

## Context

`scripts/new_algorithm.py --container hypergraph` refused with "the hypergraph container arrives
in 0.3". With the hypergraph's update path and the kit's hypergraph axis (ADR 0043) a hypergraph
algorithm can be scaffolded green. Every manifest has the required key `graph_requirements`
(PLAN 4.8); for graph algorithms it names `graph_properties` fields (`multi_edges`, `row_order`,
...) and is documentation: `regen.py` only checks that it is present, and the traits' `require()`
is what the kit applies. A hypergraph has other properties (`line_graph`, `store_v2h`, `reuse`,
`storage`), so the key's meaning for a hypergraph manifest had to be defined.

## Decision

1. **A second template folder.** `cpp/src/algorithms/_template_hypergraph/` has the same files,
   placeholders (`algorithm_template`, `{{title}}`, ...), markers (`//@@ fixed_point`,
   `//@@ aggregate_delta`, `//@@ openmp`) and destinations as `_template/`; `new_algorithm.py`
   picks the folder by `--container` (`TEMPLATE_DIRS`). Two folders instead of `//@@ graph` /
   `//@@ hypergraph` blocks in one: the graph and the hypergraph versions differ in almost every
   line (container, batch view, summary, views, instantiation macro), and a reader of either
   template sees one container. `--remove` is the same for both (the same destinations).
2. **The placeholders.** fixed_point: the component label of every vertex (the smallest vertex
   id it reaches through hyperedges; labels propagated over the hyperedges until nothing changes);
   aggregate_delta: the number of incidences (the sum of |e|). Both read the public `h2v()` view
   (built at most once per version, the container's own work, ADR 0043 finding 1), recompute in
   the family's hooks (Tier A) and set `stats.fallback_used`, as the graph template. The
   template's problem is `algorithm_template_problem<vertex_t, weight_t>` over
   `hypergraph<V,W>`, with the framework's `hyper_requested_batch` / `hyper_applied_batch`; its
   `update_traits` static_asserts `detail::is_hypergraph_v` and the supported types
   (`DYNG_FOR_EACH_HYPERGRAPH_TYPE`: (int32, unweighted) and (int32, int32)); its traits use
   `container_types`, `hypergraph_shape` and the kit's hypergraph presets. Host backends only, as
   for graphs; a CUDA backend is added by hand (the kit's fakes show the device row views).
3. **`graph_requirements` of a hypergraph manifest** keeps its name (the key is required for every
   manifest; renaming it would change the manifests of the graph algorithms and the registry
   tooling for no reader's benefit) and means the requirements on `hypergraph_properties`:
   `line_graph` ("any" | "required"), `store_v2h` ("any" | "required"), `reuse` ("any" |
   "erase_first" | "insert_first"), `storage` ("any" | "escher"). `regen.py` rejects another key
   or value in a hypergraph manifest (`HYPERGRAPH_REQUIREMENTS`); a graph manifest's table is
   unchanged (any key; documentation). As for graphs, the traits' `require()` is what the kit
   applies, and the algorithm's `compute()` checks what it needs.
4. **The scaffold check** (`ci/scaffold_check.sh`) scaffolds two hypergraph probes
   (`scaffold_probe_hg_ad`: aggregate delta on sequential and OpenMP; `scaffold_probe_hg_fp`:
   fixed point on sequential) next to the graph probes and the tutorial's `my_bfs`, builds the
   subset, runs their kits (13 checks x 2 hypergraph types each, C10 pairing the two probes) and
   hand cases, puts their headers through Doxygen, the coverage check, the API baseline and
   Sphinx, and removes them leaving nothing behind.

## Consequences

- `new_algorithm.py my_motifs --container hypergraph --family aggregate_delta` gives a hypergraph
  algorithm that builds and passes the kit on the first build; `triad_count` (M9) can start from
  it or be written by hand against the same traits.
- The stale refusal message of `new_algorithm.py` (M8b criterion 21) is gone.
- `ci/check.sh`'s clang-tidy step skips both template folders (they are not compiled).
