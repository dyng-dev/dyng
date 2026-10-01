# Command-line interface

The `dyng` command comes with the Python package (`pip install dyng` installs the console
script; `python -m dyng` is the same program). It runs the algorithms on files in the formats
of the original tools and writes their output formats, so its results can be compared byte for
byte with the originals' (ADR 0025; PLAN Section 5.6).

```text
dyng [--version] [--log-level LEVEL] COMMAND ...

  sssp compute | update         dynamic single-source shortest paths
  cycle_count compute | update  directed simple-cycle histograms
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
it fails with a message that names the CUDA plugins of 0.1.x.

**Graph input** (`--graph PATH`, also spelled `--input`) with `--format`:

| `--format` | Reader | Notes |
|---|---|---|
| `csr` | `dyng.io.read_csr_triplet` | MOSP's text CSR: `PATH` is the prefix of `PATHRowPtr.txt`, `PATHColInd.txt`, `PATHValues.txt`; `--num-weights K` gives the objectives of a graph without edges (mospPrep's `-k`) |
| `mtx` | `dyng.io.read_matrix_market` | `--random-weights MIN,MAX,SEED` with `--num-weights K`: seeded weights, bit-exact with `mospPrep mtx2csr` |
| `edges` | `dyng.io.read_edge_list` | `src dst [w1 .. wK] [ts]` lines, TUDataset `*_A.txt`, SNAP; a `%%MatrixMarket` file is read as an edge list (CycleEnumeration-GPU's parser); `--num-weights`, `--ids compact\|as_is`, `--index-base`, `--symmetrize`, `--keep-self-loops`, `--duplicates merge\|keep` |
| `auto` (default) | | `csr` if `PATHRowPtr.txt` exists, else `mtx` for `*.mtx` (sssp, convert, mosp_changes), else `edges` |

`--vertex-type int32|int64` and `--edge-type int32|int64` choose the id types (sssp, convert,
mosp_changes; cycle_count supports int32 ids). `--properties default|mosp_compatible|cycle_enum_compatible`
chooses the graph properties; the default is the preset of the original tool (`mosp_compatible`
for sssp and MOSP's generator, `cycle_enum_compatible` for cycle_count and its generator).

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
