# Command-line interface

The `dyng` command comes with the Python package (`pip install dyng` installs the console
script; `python -m dyng` is the same program). It runs the algorithms on files in the formats
of the original tools and writes their output formats, so its results can be compared byte for
byte with the originals' (ADR 0025).

```text
dyng [--version] [--log-level LEVEL] COMMAND ...

  sssp compute | update         dynamic single-source shortest paths
  mosp compute | update         dynamic multi-objective shortest paths (0.2)
  cycle_count compute | update  directed simple-cycle histograms
  triad_count compute | update  hypergraph triad counts (0.3)
  prep SUBCOMMAND               MOSP's input preparation (the mospPrep subcommands)
  convert INPUT OUTPUT          convert a graph between formats
  generate GENERATOR            the seeded batch generators of the original tools
  config                        the build and runtime configuration (dyng.show_config())
```

`dyng COMMAND --help` lists every flag of a command. Exit status: 0 on success, 1 when the work
fails (the message is printed as `dyng: error: ...` on standard error), 2 for a usage error.

## Conventions

**Option flags are the option fields.** Every field of an algorithm's `Options` is a flag with
the field's name in kebab case, generated from the dataclass (`dyng.cycle_count.Options.max_length`
is `--max-length`); an enumeration takes its names (`--cuda-engine fused`), a boolean has a
`--no-` form (`--no-validate-inputs`). The flags of `dyng generate` are the keyword arguments of
the generator functions in the same way (`dyng.generators.legacy.cycle_enum_batch(num_deletions=)`
is `--num-deletions`). A flag that is not given keeps the field's default.

**Backends.** `--backend sequential|openmp|cuda` (default: `openmp` when the build has it),
`--threads T` (OpenMP threads; 0 is the OpenMP default, which honours `OMP_NUM_THREADS`) and
`--device D`. Every backend gives the same bytes. The CPU wheel has no `cuda` backend; asking for
it fails with a message that names the CUDA plugin wheels `dyng-cu12` and `dyng-cu13` (from
0.2.0: `pip install "dyng[cu13]"` or `"dyng[cu12]"`).

**Graph input** (`--graph PATH`, also spelled `--input`) with `--format`:

| `--format` | Reader | Notes |
|---|---|---|
| `csr` | `dyng.io.read_csr_triplet` | MOSP's text CSR: `PATH` is the prefix of `PATHRowPtr.txt`, `PATHColInd.txt`, `PATHValues.txt`; `--num-weights K` gives the objectives of a graph without edges (mospPrep's `-k`) |
| `mtx` | `dyng.io.read_matrix_market` | `--random-weights MIN,MAX,SEED` with `--num-weights K`: seeded weights, bit-exact with `mospPrep mtx2csr` |
| `edges` | `dyng.io.read_edge_list` | `src dst [w1 .. wK] [ts]` lines, TUDataset `*_A.txt`, SNAP; a `%%MatrixMarket` file is read as an edge list (CycleEnumeration-GPU's parser); `--num-weights`, `--ids compact\|as_is`, `--index-base`, `--symmetrize`, `--keep-self-loops`, `--duplicates merge\|keep` |
| `auto` (default) | | `csr` if `PATHRowPtr.txt` exists, else `mtx` for `*.mtx` (sssp, convert, mosp_changes), else `edges` |

`--vertex-type int32|int64` and `--edge-type int32|int64` choose the id types (sssp, mosp,
convert, mosp_changes; cycle_count supports int32 ids). `--properties default|mosp_compatible|cycle_enum_compatible`
chooses the graph properties; the default is the preset of the original tool (`mosp_compatible`
for sssp, mosp and MOSP's generator, `cycle_enum_compatible` for cycle_count and its generator).

**Batches.** MOSP's files: `--changes DIR` (`DIR/insert.txt` with `u v w1 .. wK` per line and
`DIR/delete.txt` with `u v`), or `--insert FILE --delete FILE`; they are read with MOSP's
accept and reject rules. The text batch of CycleEnumeration-GPU's generator: `--batch FILE`, one
change per line, `- u v` for a deletion and `+ u v [w1 .. wK]` for an insertion, in graph ids
(the output of `dyng generate cycle_enum_batch`; see {doc}`file_formats`). A `--batch` file
whose name ends in `.dgt` is read as the library's batch text format (`+e u v [w..]` / `-e u v`,
{doc}`file_formats`) and must hold one batch.

## `dyng sssp`

```text
dyng sssp compute --graph G [--source S] [options] --out DIR
dyng sssp update  --graph G (--changes DIR | --insert F --delete F) [--init DIR [--canonicalize]]
                  [--source S] [options] --out DIR [--write-graph PREFIX]
```

Options: `--delta`, `--objective`, `--cuda-engine`, and on `update` `--validate-inputs` /
`--no-validate-inputs` (the fields of `dyng.sssp.Options`; `validate_inputs` checks the trees
read with `--init`, so `compute` has no such flag). Without `--objective` every weight column of
the graph gets its own tree, as MOSP computes one tree per objective (`--help` says so). The
graph needs integer weights: a Matrix Market file without them (a `pattern` or `real` matrix)
needs `--random-weights MIN,MAX,SEED` (and `--num-weights K` for K objectives), and the error
message says so; `--source` must name a vertex of the graph.

`compute` writes `DIR/obj<k>/distancesOriginal.txt` and `DIR/obj<k>/SSSPTreeOriginal.txt`, the
files of `mospPrep init`. `update` reads the initial trees from `--init` (the same layout; with
`--canonicalize` the lowest-id tie rule is applied to them, MOSP's `canonicalizeTree`) or computes
them, applies the batch once and updates every tree (`dyng.update`), then writes
`DIR/obj<k>/distancesUpdated.txt` and `DIR/obj<k>/SSSPTreeUpdated.txt`, the files of MOSP's
`mosp` driver. `--write-graph PREFIX` also writes the updated graph as MOSP's text CSR. One
report line per tree goes to standard output (`--quiet` suppresses them).

```console
$ dyng prep mtx2csr roadNet-CA.mtx roadNet-CA_ 1 1 100 12345
$ dyng prep changes roadNet-CA_ batch --changes 50000 --ins 50 --seed 1
$ dyng sssp compute --graph roadNet-CA_ --out init
$ dyng sssp update --graph roadNet-CA_ --changes batch --init init --out updated
```

## `dyng mosp`

```text
dyng mosp compute --graph G [--source S] [options] --out DIR
dyng mosp update  --graph G (--changes DIR | --insert F --delete F) [--init DIR [--canonicalize]]
                  [--source S] [options] --out DIR [--write-graph PREFIX]
```

Options (the fields of `dyng.mosp.Options`): `--preferences P1,..,PK` (the preference vector,
one integer >= 1 per objective, comma-separated as MOSP's `--pref`; default all 1, lcm at most
2^20), `--num-objectives K` (the first K weight columns; default 0, every column; MOSP's `-k`),
`--delta`, `--cuda-engine automatic|fused|operators`, `--compute-path-costs` /
`--no-compute-path-costs`, and on `update` `--validate-inputs` / `--no-validate-inputs` (it checks
the trees read with `--init`). The graph and batch flags are those of `dyng sssp`.

`compute` writes `DIR/obj<k>/distancesOriginal.txt` and `SSSPTreeOriginal.txt` (the files of
`mospPrep init`) and the combined graph's outputs `DIR/combinedGraph/distancesCsr.txt` (the
combined distances, in units of 1/L with L = lcm of the preferences), `SSSPTreeCsr.txt` (the MOSP
tree) and `mospCosts.txt` (`v c1 .. cK` per vertex, `INF` when unreachable: the path costs).
`update` reads the initial trees from `--init` or computes them, applies the batch once, updates
the K trees and rebuilds the combined graph, the MOSP tree and the costs (`dyng.mosp.update`),
then writes `DIR/obj<k>/distancesUpdated.txt`, `SSSPTreeUpdated.txt` and the three
`combinedGraph/` files: every byte as MOSP's `mosp` driver writes them (with
`--num-objectives` below the graph's column count, `mospCosts.txt` covers every column, as the
original's does; with `--no-compute-path-costs` it is not written). The tests compare the output
with the committed fixtures of both originals, with `dyng-compat-mosp --mosp` and, where it is
built, with MOSP-OpenMP's `mosp`.

| `mosp` (MOSP-OpenMP, MOSP-CUDA) | `dyng mosp update` |
|---|---|
| `--graph`, `--changes`, `--init`, `--out`, `--source`, `--delta`, `--canonicalize` | the same flags |
| `--pref p1,..,pK` | `--preferences p1,..,pK` |
| `-k K` | `--num-objectives K` |
| `--no-output`, `--validate`, `--timing`, `--cache` | not in `dyng`; `dyng-compat-mosp --mosp` (tools/compat) keeps them for the parity harness |

```console
$ dyng prep mtx2csr roadNet-CA.mtx roadNet-CA_ 3 1 100 12345
$ dyng prep changes roadNet-CA_ batch --changes 50000 --ins 50 --seed 777
$ dyng prep init roadNet-CA_ init
$ dyng mosp update --graph roadNet-CA_ --changes batch --init init --preferences 4,1,4 --out out
obj0: invalidated ..., affected ..., iterations ..., engine fused
...
combined: L=4, reachable ... of 1971281, ... edges, affected ...
```

## `dyng cycle_count`

```text
dyng cycle_count compute --graph G [options] [--output FILE]
dyng cycle_count update  --graph G [options] [--output FILE]
                         (--batch FILE | --changes DIR | --insert F --delete F |
                          --num-deletions D --num-insertions I --seed S [--locality-window W])
                         [--write-batch FILE] [--compare-recompute]
```

Options: `--max-length`, `--method`, `--mode`, `--cuda-engine`, `--scheduler`, `--work-items`
(the fields of `dyng.cycle_count.Options`). Set `--max-length` for large graphs: without a bound
`compute` can take exponential time. `update` requires `--max-length` (a usage error, exit
status 2, without it), as the original's `--task update` requires `--max-cycle-length`: the
update enumerates the simple paths through the changed edges up to that length.

The histogram is printed in CycleEnumeration-GPU's CSV (`# cycle_size, num_of_cycles`, one
`len, count` line per non-zero length, `Total, N`), byte for byte as its `cycle-enum` prints it.
`update` counts the graph, applies the batch (read from a file, or generated with the arguments
of `dyng generate cycle_enum_batch`), updates the histogram and prints the updated histogram;
`update_seconds=` goes to standard error, and with `--compare-recompute` also
`recompute_seconds=` and `match=yes|no` (a mismatch exits with status 1).

The original's flags and dynG's:

| `cycle-enum` | `dyng cycle_count` |
|---|---|
| `--input` | `--graph` (`--input` is accepted) |
| `--task count` / `--task update` | `compute` / `update` |
| `--max-cycle-length`, `--max-length` | `--max-length` |
| `--openmp-threads`, `--threads` | `--threads` |
| `--algorithm`, `--mode` | `--method`, `--mode` |
| `--deletes`, `--inserts`, `--batch-seed`, `--batch-locality` | `--num-deletions`, `--num-insertions`, `--seed`, `--locality-window` |
| `--compare-recompute` | `--compare-recompute` |
| `--backend openmp\|omp\|cpu`, `--backend cuda\|gpu`, `--backend seq` | `--backend openmp`, `--backend cuda`, `--backend sequential` (the names only) |
| `--cuda-scheduler` (alias `--scheduler`) `naive\|work-queue` (also `work_queue`, `workqueue`, `queue`) | `--scheduler naive\|work_queue` |
| `--cuda-work-items auto\|roots\|edges\|two-hop` (also `root`, `edge`, `two_hop`, `twohop`) | `--work-items automatic\|roots\|edges\|two_hop` |
| `--cuda-device` (alias `--device`) | `--device` |

The value names are the enumerators of `dyng.cycle_count.Options` (`work_queue`, `automatic`,
`two_hop`); the original's other spellings are not accepted. On standard error `dyng
cycle_count update` prints `update_seconds=` (and `recompute_seconds=`, `match=`), not the
original's `deletions=N insertions=M` line; the batch's size is in the histogram's input (or
`--write-batch`). The drop-in clone with the original's flags, exit statuses and timing lines is the C++ tool
`dyng-compat-cycle-enum` (`tools/compat`, for the parity harness).

## `dyng triad_count`

```text
dyng triad_count compute --input H.hg [read flags] [options] [--format json|escher] [--output FILE]
dyng triad_count update  --input H.hg [read flags] [options] [--format json|escher] [--output FILE]
                         --batches FILE [FILE ...] [--compare-recompute]
```

Options: `--kind h_motif|stathyper_type1|stathyper_type2|stathyper_type3` and `--cuda-engine`
(the fields of `dyng.triad_count.Options`; default `--kind h_motif`, the 30 h-motif bins).

**Hypergraph input** (`--input PATH`, also spelled `--hypergraph`): a `.hg` file
({doc}`file_formats`), read by `dyng.io.read_hypergraph` with ESCHER-GPU's loader (the option
set `--read-options escher_gpu`, the default; `--read-options hsosp` is H-SOSP's). The read
flags are the keyword arguments of `dyng.io.read_hypergraph`: `--index-base`,
`--dedupe-hyperedges`, `--max-cardinality`, `--relabel`, `--num-vertices`, `--weighted` (a file
with `; w` on every line; the weights are ignored by triad_count) override the option set's
fields, and `--storage`, `--line-graph`, `--id-reuse`, `--capacity` and `--store-v2h` the
hypergraph's properties. `--min-initial-payload N` with `--capacity escher_gpu` is
`CapacityPolicy.escher_gpu(N)`, ESCHER-GPU's `--payload-capacity`. The h-motifs and types 2 and
3 need the line graph and the vertex-to-hyperedge rows (`--line-graph none` or
`--no-store-v2h` is an error for them; type 1 needs neither).

**Batches** (`update`): `--batches FILE ...`, applied in order; each file is a `.dgt` batch file
(`+h v1 v2 ..` / `-h id`, any number of batches) or a `.dgb` binary batch file (by its magic). A
`.dgb` file that carries the expected ids of every insertion (as the parity harness's recordings
do) is checked: a batch whose inserted ids differ fails with exit status 1 (the hypergraph was not
built as the recording's run built it, so later deletions would name other hyperedges: give the
same `--num-vertices`, `--id-reuse`, `--capacity` and `--min-initial-payload`).

**Output.** `--format json` (the default) writes the triad-bins JSON of
`dyng.io.write_triad_bins` ({doc}`file_formats`): the kind, the hypergraph version, the counts
(after the last batch for `update`) and, for `update`, one row of deltas per batch.
`--format escher` prints the lines of ESCHER-GPU@abcf9b2's `build/main` (`--kind h_motif`),
`type1`, `type2` and `type3`, byte for byte: `Hypergraph: N hyperedges, max vertex id V, P pins`,
the baseline (`Motif counts (30 bins): c0 .. c29`, `[GPU] Type1 baseline motif count (sum_e
C(|e|, 3)): N`, `[GPU] Type2 baseline motif count: N`, `[GPU] Type3 baseline motif count: N`),
and per batch `Batch b (D deletions, I insertions)` (when there is more than one batch, as the
original prints it) and the delta (`Motif delta counts (30 bins): d0 .. d29`, `[GPU] TypeN motif
delta count: N`). The `[GPU]` prefix is the original's on every backend; `max vertex id` is the
hypergraph's vertex count (ESCHER-GPU's 1-based largest id). Bin `k` is ESCHER-GPU's h-motif
`k + 1`; its open triads 21-26 (bins 20-25) are always 0. `update_seconds=` goes to standard
error, and with `--compare-recompute` also `recompute_seconds=` and `match=yes|no` (a mismatch
exits with status 1). The original generates its hypergraph and batches itself; the parity
harness records them as `.hg` and `.dgb` files, and `dyng triad_count update --format escher`
on those files prints the original's standard output without its generator lines (the test
`python/tests/test_triad_count_cli.py` diffs every make-test case).

The original's flags and dynG's:

| ESCHER-GPU | `dyng triad_count` |
|---|---|
| `build/main`, `type1`, `type2`, `type3` | `--kind h_motif`, `stathyper_type1`, `stathyper_type2`, `stathyper_type3` |
| `--input FILE` | `--input FILE` (ESCHER-GPU's loader: `--read-options escher_gpu`) |
| `n m r1 r2` (a generated hypergraph), `--batches`, `--batch-size`, `--seed`, `--insert-mode` | the hypergraph and batch files the generator writes (`dyng.generators.legacy.escher_hypergraph`, `EscherBatchGenerator`; `dyng-compat-escher run` is the drop-in driver) |
| `--payload-capacity N` | `--capacity escher_gpu --min-initial-payload N` |
| `--check`, `--reference` | `--compare-recompute`; the oracle is `dyng.testing.triad_reference` |

## `dyng prep`

The six subcommands of MOSP's `mospPrep` (MOSP-OpenMP c352151, MOSP-CUDA e220ee2), with its
positional arguments and flags, its output files byte for byte and its report lines (the times
differ); each ends with `<command> done in <ms> ms (rc=<status>)`.

```text
dyng prep mtx2csr  <in.mtx> <outPrefix> <K> <wmin> <wmax> <seed>
dyng prep widen    <inPrefix> <outPrefix> <K> <wmin> <wmax> <seed>
dyng prep cache    <csrPrefix> <binaryPath>
dyng prep changes  <csrPrefix> <outDir> [--changes N] [--ins PCT]
                   [--mode uniform|targeted|reweight|increase] [--local HOPS] [--safe]
                   [--seed S] [--source s] [--wmin a] [--wmax b] [-k K]
dyng prep init     <csrPrefix> <outDir> [--source s] [-k K] [--backend B] [--threads T]
dyng prep expected <csrPrefix> <changesDir> <outDir> [--source s] [-k K] [--backend B] [--threads T]
```

| Subcommand | What it writes |
|---|---|
| `mtx2csr` | Matrix Market to MOSP's text CSR: symmetric matrices get both directions, self-loops and duplicates are dropped, every edge gets K seeded weights in [wmin, wmax] (1 <= K <= 32) |
| `widen` | a copy of the graph with random objectives appended until it has K (the existing ones unchanged) |
| `cache` | MOSP's binary cache (`MOSPCSR2`, with the identity of the three text files) read by `mosp --cache` |
| `changes` | `outDir/insert.txt`, `outDir/delete.txt`: MOSP's change generator (`dyng.generators.legacy.mosp_changes`) |
| `init` | `outDir/obj<k>/distancesOriginal.txt`, `SSSPTreeOriginal.txt`: one tree per objective |
| `expected` | `outDir/obj<k>/distancesUpdated.txt`, `SSSPTreeUpdated.txt`: the trees of the graph after the batch |

## `dyng convert`

```text
dyng convert INPUT OUTPUT [--format auto|csr|mtx|edges] [reader flags] [--to auto|csr|mtx|edges]
             [--weight-column K]
```

Reads a graph with the reader flags above and writes it as MOSP's text CSR (`--to csr`, OUTPUT a
prefix; the graph needs weights), a `general` coordinate Matrix Market file (`--to mtx`, the
default for `*.mtx`; `--weight-column` chooses the value column) or an edge list (`src dst [w..]`
per line, the default otherwise). `dyng convert g.mtx g_ --to csr --num-weights K
--random-weights 1,100,12345` writes the same bytes as `mospPrep mtx2csr g.mtx g_ K 1 100 12345`.

## `dyng generate`

```text
dyng generate mosp_changes     --graph G --out DIR [--num-changes N] [--insertion-percentage P]
                               [--mode M] [--weight-min A] [--weight-max B] [--seed S]
                               [--local-hops H] [--safe-deletions] [--source S]
dyng generate cycle_enum_batch --graph G [--num-deletions D] [--num-insertions I] [--seed S]
                               [--locality-window W] [--out FILE]
```

`mosp_changes` is MOSP's `generateChangeBatch` (the same batch as `dyng prep changes` and
`mospPrep changes`, with the generator's argument names); it writes `DIR/insert.txt` and
`DIR/delete.txt` and prints the report line. `cycle_enum_batch` is CycleEnumeration-GPU's
`generate_batch`: the batch in graph ids, deletions first, as text on standard output or in FILE.
Both are bit-exact with the originals for a fixed seed.
