# Hypergraph triads

In this tutorial you count the **triads** of a hypergraph (three hyperedges that pairwise share a
vertex) by their h-motif, keep the counts exact while hyperedges are deleted and inserted, and
read what changed. You do it twice, in C++ and in Python, with the same input and the same
output. It takes about 15 minutes.

You need dynG 0.3 or newer installed ({doc}`../getting_started/install`): for C++ a compiler and
CMake, for Python `pip install "dyng>=0.3"` (the hypergraph and `triad_count` are new in 0.3;
until 0.3.0 is on PyPI, take the release candidate's lines of the install page's "Python"
section, since a plain `pip install dyng` gives 0.2.0, which has neither). You should know what a
hypergraph is ({doc}`../concepts/hypergraphs`).
The algorithm is {doc}`../algorithms/triad_count`; this page shows how to use it.

The code is two example programs, `examples/cpp/triad_count_update.cpp` and
`examples/python/triad_count.py`, and their input `examples/data/triad_count.hg` and
`examples/data/triad_count.dgt`. CI builds the C++ program against the installed package and runs
both programs on the CPU backends (the local GPU gate, `ci/gpu_local.sh`, also on cuda), and
compares what they print with `examples/data/triad_count_update.expected.txt`; a test rebuilds
that file by a brute force over every triple of hyperedges after every batch
(`python/tests/test_examples.py`), and another runs the command of section 6 and compares its
output with this page (`python/tests/test_triad_count_cli.py`). The snippets below are quoted from
those files, and the documentation check fails when a quoted line changes without this page
(`ci/tutorial_quotes.py`).

The programs and their input are in the repository, not in the installed package: clone it and
run every command of this page from its root (the tag of your version keeps the files as this
page quotes them):

```bash
git clone --branch v0.3.0rc1 https://github.com/dyng-dev/dyng.git
cd dyng
```

## 1. The input: a `.hg` file and a batch file

A `.hg` file has one hyperedge per line, its vertex ids separated by spaces; `#` starts a
comment. The example's file has 12 lines over the vertices 1-10, and line 10 repeats line 1:

% quote: examples/data/triad_count.hg#L6-L17 08ab2dc1845b
```{literalinclude} ../../examples/data/triad_count.hg
:language: text
:lines: 6-17
```

The batches are a `.dgt` file ({doc}`../api/file_formats`): `%batch <n>` starts a batch, `-h id`
deletes the hyperedge with that id and `+h v1 v2 ...` inserts one. The ids and the members are
dynG's 0-based ones (the vertex of file id 1 is vertex 0):

% quote: examples/data/triad_count.dgt#L6-L13 802d6888d043
```{literalinclude} ../../examples/data/triad_count.dgt
:language: text
:lines: 6-13
```

## 2. Read the hypergraph (C++)

% quote: examples/cpp/triad_count_update.cpp#read 5e0cbac382d1
```{literalinclude} ../../examples/cpp/triad_count_update.cpp
:language: cpp
:start-after: "// [tutorial: read]"
:end-before: "// [tutorial: read end]"
```

- `read_hypergraph` reads the file with ESCHER-GPU's conventions: ids 1..n become vertices
  0..n-1, and `dedupe_hyperedges` keeps the first of every repeated member set (ESCHER-GPU's
  preprocessing of coauth-DBLP). The repeated line is dropped, so the hypergraph has 11
  hyperedges, ids 0-10 in file order.
- `line_graph_policy::sorted_merge` keeps the line graph (which hyperedges share a vertex) inside
  the hypergraph and maintains it under batches. The h-motifs and StatHyper's types 2 and 3 read
  it; `triad_count::compute()` names the property if it is missing. StatHyper's type 1 needs no
  line graph.

## 3. Count (C++)

% quote: examples/cpp/triad_count_update.cpp#compute 97cb45c76164
```{literalinclude} ../../examples/cpp/triad_count_update.cpp
:language: cpp
:start-after: "// [tutorial: compute]"
:end-before: "// [tutorial: compute end]"
```

Each pattern is its own result: `motifs` holds 30 bins (`options::kind` defaults to
`pattern::h_motif`), `type1` one bin. A result's counts are a host array on every backend:
`counts()`, `count(bin)` and `total()`. The program prints the total and the non-zero bins as
`bin:count`:

```text
11 hyperedges: 45 triads, type1 20
  bins 3:2 9:9 14:19 15:6 28:1 29:8
```

**Reading the bins.** Bin k is ESCHER-GPU's h-motif k + 1, the triads whose Venn diagram has a
given set of non-empty regions (the table is in section 1 of {doc}`../algorithms/triad_count`).
For example the 19 triads of bin 14 have a vertex in all three hyperedges (region ABC), a vertex in
two of the pairs only (AB, BC) and a vertex only in two of the hyperedges (A, C); the 8 of bin 29
share no vertex among all three, but every pair overlaps outside the third, and each hyperedge has
a vertex of its own. Bins 20-25 are always 0: they are the *open* h-motifs, where two of the
hyperedges are disjoint, and a triad is by definition closed. `type1` is 20: the sum of C(|e|, 3)
over the 11 hyperedges (eight of 3 vertices, three of 4).

## 4. Apply the batches to both results at once (C++)

% quote: examples/cpp/triad_count_update.cpp#update 5362ac77b05c
```{literalinclude} ../../examples/cpp/triad_count_update.cpp
:language: cpp
:start-after: "// [tutorial: update]"
:end-before: "// [tutorial: update end]"
```

- `dyng::update(res, hg, batch, motifs, type1)` applies the batch to the hypergraph **once** and
  brings both results up to date: each subtracts the triads through the deleted hyperedges on the
  hypergraph before the batch, then adds those through the inserted hyperedges on the hypergraph
  after it. It returns one `triad_count::stats` per result. (One result alone is
  `triad_count::update(res, hg, batch, motifs)`; updating `motifs` that way and then `type1` would
  fail, because `type1` would no longer match the hypergraph: `stale_result_error`.)
- `st.batch.inserted_ids` are the ids the inserted hyperedges received. The ESCHER store reuses the
  slots of deleted hyperedges by best fit, so the ids are its choice: read them here, never
  predict them. Batch 0 deletes 1 and 6 and its two insertions get 1 and 6 back.
- `st.triads_removed` and `st.triads_added` are the two phases summed over the bins, and
  `st.delta[k]` the change of bin k (after minus before; `std::array<std::int64_t, 30>`).

The whole output of the program:

% quote: examples/data/triad_count_update.expected.txt#L1-L8 3bdcf0874505
```{literalinclude} ../../examples/data/triad_count_update.expected.txt
:language: text
:lines: 1-8
```

Batch 0 destroys 27 triads and creates 35, so the total goes from 45 to 53; its type-1 sum stays
20, because the deleted and the inserted hyperedges have the same sizes. After every update the
counts equal a fresh `compute()` on the new hypergraph exactly, on every backend.

Build and run it against your installed dynG (the CMake block of {doc}`../getting_started/install`,
with `add_executable(triad_count_update triad_count_update.cpp)`):

```console
$ ./triad_count_update examples/data/triad_count.hg examples/data/triad_count.dgt openmp
```

The last argument is the backend: `sequential` (the default), `openmp` or `cuda` (the GPU of
`CUDA_VISIBLE_DEVICES`; the program exits with 77 if there is none). The output is the same on
all three. Another name is refused with the list; a file that cannot be read ends the program with
the library's message (the file, line and column) and exit status 1.

## 5. The same in Python

% quote: examples/python/triad_count.py#read a4d216e0fedd
```{literalinclude} ../../examples/python/triad_count.py
:language: python
:start-after: "# [tutorial: read]"
:end-before: "# [tutorial: read end]"
```

% quote: examples/python/triad_count.py#compute fe24daad91ba
```{literalinclude} ../../examples/python/triad_count.py
:language: python
:start-after: "# [tutorial: compute]"
:end-before: "# [tutorial: compute end]"
```

% quote: examples/python/triad_count.py#update e6a3ebe13b8d
```{literalinclude} ../../examples/python/triad_count.py
:language: python
:start-after: "# [tutorial: update]"
:end-before: "# [tutorial: update end]"
```

The Python names are the C++ ones: `kind` is a string (or a `dyng.triad_count.Pattern`),
`Result.counts` is a read-only uint64 NumPy array, `Stats.delta` an int64 array of the result's
`num_bins` values (30 here) and `st.batch.inserted_ids` an int64 array. `dyng.update()` takes
several results of the same hypergraph and returns their stats as a tuple. Run it:

```console
$ python examples/python/triad_count.py examples/data/triad_count.hg examples/data/triad_count.dgt
```

It prints the eight lines above, byte for byte (`python/tests/test_examples.py`).

## 6. From the command line

`dyng triad_count` does the same without a program; `--format escher` prints ESCHER-GPU's
output lines (all 30 bins, then the deltas of each batch):

```console
$ dyng triad_count update --input examples/data/triad_count.hg --dedupe-hyperedges \
      --batches examples/data/triad_count.dgt --format escher
Hypergraph: 11 hyperedges, max vertex id 10, 36 pins
Motif counts (30 bins): 0 0 0 2 0 0 0 0 0 9 0 0 0 0 19 6 0 0 0 0 0 0 0 0 0 0 0 0 1 8
Batch 1 (2 deletions, 2 insertions)
Motif delta counts (30 bins): 0 0 0 1 0 0 0 0 0 7 0 0 0 0 -6 -1 0 0 0 0 0 0 0 0 0 0 0 2 0 5
Batch 2 (1 deletions, 1 insertions)
Motif delta counts (30 bins): 0 0 0 2 0 0 7 0 0 -5 0 0 0 0 -3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4
```

and `--format json` (the default) the triad-bins JSON that `dyng.io.read_triad_bins` reads back
({doc}`../api/cli`).

## Next

- {doc}`count_a_new_hypergraph_pattern`: count a pattern that is not one of the four, as an Op of
  `experimental::count_local_patterns`, with the same engine and the same exact updates.
- {doc}`../algorithms/triad_count`: the 30-bin table, the requirements of each pattern, the
  performance gates against ESCHER-GPU and what differs from the papers.
