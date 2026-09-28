#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Export the sssp golden corpus from the pinned original MOSP-OpenMP@c352151 (or MOSP-CUDA).

PLAN Sections 6.3 (steps 1-2) and 8.3. The goldens are written OUTSIDE the repository, to
$DYNG_SCRATCH/goldens/sssp (default ~/Projects/dyng-work); the repository keeps only
parity/goldens.toml, the manifest with one SHA-256 per case (and the tiny cases, which are already
committed test fixtures under cpp/tests/data).

    parity/export_goldens.py [--jobs N] [--twice] [--groups g1,g2] [--out DIR] [--no-toml]
    parity/export_goldens.py --reference MOSP-CUDA [--compare-to DIR] [--jobs N]

Every case is produced by the original tools of the PATCHED scratch copy that
parity/build_reference.sh builds (the originals are never touched):

  group          cases  inputs produced by (c352151)
  testcases         10  tests/testCase0..9 (generateTestCases, tracked in the commit)
  regressions        3  bin/mospTest --only regressions (count-to-infinity: n = 6, seeds
                        621705 / 250813, d(1) = 90, and two more stress-test seeds)
  thesis             1  bin/mospTest --only thesis-example (thesis Ch. 4 worked example)
  escher             4  the MOSP_ESCHER unit cases (4-vertex disconnect d = 100/101, delete-all,
                        ties) and ties_k2 (committed inputs, cpp/tests/data/mosp_sssp)
  fixtures          17  the graph/io fixture inputs r00..r11, h0..h4 (duplicates, self-loops,
                        parallel edges, unsorted rows, delete-all, empty batch, K = 1..5, 32)
  sosp             148  bin/mospTest --only sosp --seed 1 (5 graphs x 3 seeds x 10 change sets)
  large_weights      2  bin/mospTest --only large-weights (320 x 320 grid, weights up to 2^31 - 1:
                        the distance-only fallback)
  packing            3  the packing boundaries of mospTest (MOSP-OpenMP: path n = 2^16 + 1,
                        weights 2^31 - 1; MOSP-CUDA: path n = 2^17 - 1, pull and push), written
                        here with the same construction
  stress           200  bin/stressTest 1 and bin/parallelStressTest 2 (100 cases each; the two
                        programs draw the same cases for the same seed, so the second uses seed 2)
  noncanonical          the inputs of the testcases, regressions, escher, fixtures and sosp
                        groups with NON-CANONICAL initial trees: bin/mospPrep init, then every
                        vertex with several tight in-neighbours takes a random one of them as its
                        parent (seeded by the case name; a valid shortest-path tree whose tie
                        parents are not the lowest ids, like MOSP's dataset trees). Cases where
                        no vertex has a tie are skipped. `mosp` (no --canonicalize) keeps the
                        parents of the vertices the batch does not reach; the goldens record it.

For every case (K objectives, source 0) the goldens are:

  input/graphCsr{RowPtr,ColInd,Values}.txt, input/insert.txt, input/delete.txt
  init/obj<k>/distancesOriginal.txt, SSSPTreeOriginal.txt    bin/mospPrep init (Dijkstra); for
                                                             noncanonical: the perturbed trees
  init_canonical/obj<k>/...                                  noncanonical only: mospPrep init
  updated/obj<k>/distancesUpdated.txt, SSSPTreeUpdated.txt   bin/mosp --validate (sospUpdateCpu)
  combined/{distancesCsr,SSSPTreeCsr,mospCosts}.txt          bin/mosp (for mosp, 0.2)
  applied/graphCsr{RowPtr,ColInd,Values}.txt                 applyChangeBatch + writeCsrGraph
  applied/graphCsrWeightIncrease.txt                         weightIncreaseMask bits
  case.json                                                  K, source, origin, invalidated per
                                                             objective (from the bin/mosp report)

Before a case is written, the originals are cross-checked (PLAN Section 6.3 step 2): the updated
trees of bin/mosp must equal byte for byte those of parallelSOSPUpdate and sequentialSOSPUpdate
(the file-based updates) and of `mospPrep expected` (Dijkstra on the updated graph); the initial
trees written by mospTest (where it writes them) must equal `mospPrep init`; for the testcases the
tracked expected Dijkstra files of the commit must equal the goldens. For the noncanonical group
only parallelSOSPUpdate (sospUpdateCpu, the rule of every dynG backend) must equal bin/mosp, and
`mospPrep expected` the distances: sequentialSOSPUpdate and Dijkstra pick lowest-id tie parents
where sospUpdateCpu keeps the input's (ADR 0006, "Tie rule of sssp").

--twice exports the corpus a second time from a FRESH scratch copy (a new git archive) into a
temporary work area and requires an identical manifest (PLAN Section 8.3, "the harness is honest").

--reference MOSP-CUDA (M1b, PLAN 6.4.2: parity against MOSP-CUDA e220ee2) runs the SAME export with
every tool of the patched MOSP-CUDA@e220ee2 copy instead: its mospTest (and its own
packing-boundary group as a precondition of the packing cases), its stressTest 1 and
parallelStressTest 2 (whose 100 cases validate the CUDA update against the sequential one), its
mospPrep init / expected, its bin/mosp --validate (sospUpdateGpu, the reference update) and its
file-based parallelSOSPUpdate (sospUpdateGpu) and sequentialSOSPUpdate through export_sssp. It runs
on CUDA_VISIBLE_DEVICES (default GPU 1, the development GPU), writes into
<scratch>/goldens/sssp-mosp-cuda by default and never writes parity/goldens.toml: the committed
corpus stays MOSP-OpenMP's. --compare-to DIR then requires the MOSP-CUDA corpus to be identical to
the corpus in DIR (default <scratch>/goldens/sssp): the same files with the same bytes, except
that case.json names the other reference (every other field, incl. the invalidated counters, must
be equal). That makes the golden corpus a MOSP-CUDA corpus as well, inputs included.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import filecmp
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# The originals the corpus can be exported from (--reference); the committed one is MOSP-OpenMP's.
COMMITS = {
    "MOSP-OpenMP": "c35215135341d5b5d1553458afe4b2226edc38fb",
    "MOSP-CUDA": "e220ee20d1b0948ece3df135a02d1b898264c22f",
}
REFERENCE = "MOSP-OpenMP"  # set by main() from --reference
COMMIT = COMMITS[REFERENCE]
INT_MAX = 2**31 - 1
GROUPS = [
    "testcases",
    "regressions",
    "thesis",
    "escher",
    "fixtures",
    "sosp",
    "large_weights",
    "packing",
    "stress",
    "noncanonical",
]
# The groups whose inputs the noncanonical group reuses.
NONCANONICAL_BASES = ["testcases", "regressions", "escher", "fixtures", "sosp"]
# generateTestCases: the objective each tracked testCase<i>/expected file belongs to.
TESTCASE_OBJECTIVE = [0, 1, 2, 0, 1, 0, 1, 2, 0, 0]
CSR_FILES = ["graphCsrRowPtr.txt", "graphCsrColInd.txt", "graphCsrValues.txt"]
INVALIDATED = re.compile(r"^obj(\d+)\s+SOSP update .*\(invalidated (\d+),", re.M)


class ExportError(RuntimeError):
    pass


@dataclass
class Case:
    group: str
    name: str
    graph: Path  # CSR prefix (…/graphCsr)
    insert: Path
    delete: Path
    origin: str
    k: int | None = None
    mosptest_init: Path | None = None  # init/obj<k>/{distances,tree}.txt written by mospTest
    tracked_expected: tuple[int, Path] | None = None  # (objective, testCase expected/ dir)
    perturb: bool = False  # noncanonical: perturb the tie parents of the initial trees
    notes: list[str] = field(default_factory=list)

    @property
    def rel(self) -> str:
        return f"{self.group}/{self.name}"


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> str:
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise ExportError(
            f"{' '.join(map(str, cmd))} failed ({proc.returncode}):\n{proc.stdout[-3000:]}"
        )
    return proc.stdout


def same(a: Path, b: Path, what: str) -> None:
    if not filecmp.cmp(a, b, shallow=False):
        raise ExportError(f"cross-check failed: {what}: {a} != {b}")


def k_args(k: int) -> list[str]:
    """`-k K` for mospPrep (MOSP-OpenMP only: MOSP-CUDA's mospPrep takes K from the Values file,
    and every graph of the corpus has edges)."""
    return [] if REFERENCE == "MOSP-CUDA" else ["-k", str(k)]


def num_weights(values: Path) -> int:
    with open(values) as f:
        for line in f:
            if line.strip():
                return len(line.split())
    return 0


# --- Case collection --------------------------------------------------------------------------


def collect(ref: Path, raw: Path, groups: list[str], env: dict) -> list[Case]:
    wanted = set(groups)
    if "noncanonical" in wanted:
        groups = [g for g in GROUPS if g in wanted or g in NONCANONICAL_BASES]
    cases = collect_base(ref, raw, groups, env)
    derived = []
    if "noncanonical" in wanted:
        for c in cases:
            if c.group in NONCANONICAL_BASES:
                derived.append(
                    Case(
                        "noncanonical",
                        f"{c.group}__{c.name}",
                        c.graph,
                        c.insert,
                        c.delete,
                        f"{c.origin}; initial trees with perturbed tie parents",
                        k=c.k,
                        perturb=True,
                    )
                )
    return [c for c in cases if c.group in wanted] + derived


def collect_base(ref: Path, raw: Path, groups: list[str], env: dict) -> list[Case]:
    cases: list[Case] = []
    mosptest = ref / "bin" / "mospTest"

    def mosp_test(group: str) -> Path:
        work = raw / f"mospTest-{group}"
        out = run([mosptest, "--seed", "1", "--work", work, "--only", group], env=env)
        if "all checks passed" not in out:
            raise ExportError(f"mospTest --only {group} did not pass:\n{out}")
        return work

    def mosptest_case(group: str, name: str, d: Path, origin: str) -> Case:
        return Case(
            group,
            name,
            d / "graph" / "graphCsr",
            d / "changes" / "insert.txt",
            d / "changes" / "delete.txt",
            origin,
            mosptest_init=d / "init",
        )

    if "testcases" in groups:
        for i in range(10):
            d = ref / "tests" / f"testCase{i}"
            cases.append(
                Case(
                    "testcases",
                    f"testCase{i}",
                    d / "originalGraph" / "graphCsr",
                    d / "changedEdges" / "insert.txt",
                    d / "changedEdges" / "delete.txt",
                    f"tests/testCase{i} (generateTestCases, tracked)",
                    tracked_expected=(TESTCASE_OBJECTIVE[i], d / "expected"),
                )
            )
    if "regressions" in groups:
        work = mosp_test("regressions")
        for i in range(3):
            cases.append(
                mosptest_case(
                    "regressions",
                    f"c2i_{i}",
                    work / f"regression_{i}" / "case",
                    f"mospTest --seed 1 --only regressions: regression_{i}",
                )
            )
    if "thesis" in groups:
        work = mosp_test("thesis-example")
        cases.append(
            mosptest_case(
                "thesis",
                "thesis_example",
                work / "thesis-example",
                "mospTest --only thesis-example",
            )
        )
    if "escher" in groups:
        data = REPO / "cpp" / "tests" / "data" / "mosp_sssp"
        for name in ["escher_disconnect", "escher_delete_all", "escher_ties", "ties_k2"]:
            d = data / name / "input"
            cases.append(
                Case(
                    "escher",
                    name,
                    d / "graphCsr",
                    d / "insert.txt",
                    d / "delete.txt",
                    f"cpp/tests/data/mosp_sssp/{name}/input (hand-written)",
                    k=1 if name == "escher_delete_all" else None,
                )
            )
    if "fixtures" in groups:
        data = REPO / "cpp" / "tests" / "data" / "mosp_graph_io"
        names = [
            n for n in (data / "cases.txt").read_text().split() if not n.startswith("testCase")
        ]
        for name in names:
            d = data / name
            cases.append(
                Case(
                    "fixtures",
                    name,
                    d / "graphCsr",
                    d / "insert.txt",
                    d / "delete.txt",
                    f"cpp/tests/data/mosp_graph_io/{name}",
                )
            )
    if "sosp" in groups:
        work = mosp_test("sosp")
        found = sorted(p.parent.parent for p in work.glob("*/*/graph/graphCsrRowPtr.txt"))
        for d in found:
            name = f"{d.parent.name}__{d.name}"
            cases.append(
                mosptest_case(
                    "sosp", name, d, f"mospTest --seed 1 --only sosp: {d.parent.name}/{d.name}"
                )
            )
    if "large_weights" in groups:
        work = mosp_test("large-weights")
        for kind in ["safe", "unsafe"]:
            d = work / f"large-weights-{kind}"
            cases.append(
                mosptest_case(
                    "large_weights", kind, d, f"mospTest --seed 1 --only large-weights: {kind}"
                )
            )
    if "packing" in groups:
        if REFERENCE == "MOSP-CUDA":
            mosp_test("packing-boundary")  # the original's own oracle of these constructions
        cases.extend(packing_cases(raw / "packing"))
    if "stress" in groups:
        for program, seed in [("stressTest", 1), ("parallelStressTest", 2)]:
            cwd = raw / f"{program}-{seed}"
            cwd.mkdir(parents=True)
            out = run([ref / "bin" / program, str(seed)], cwd=cwd, env=env)
            if "100/100 passed" not in out:
                raise ExportError(f"{program} {seed} did not pass:\n{out[-2000:]}")
            for i in range(100):
                d = cwd / program / f"run{i}"
                cases.append(
                    Case(
                        "stress",
                        f"{program}_{seed}_run{i:02d}",
                        d / "originalGraph" / "graphCsr",
                        d / "changedEdges" / "insert.txt",
                        d / "changedEdges" / "delete.txt",
                        f"bin/{program} {seed}: run{i}",
                    )
                )
    return cases


def write_csr(prefix: Path, n: int, rows: list[list[tuple[int, int]]]) -> None:
    """writeCsrGraph's format: n + 1 row offsets, one column and one weight line per edge."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    offsets, cols, weights = [0], [], []
    for row in rows:
        for v, w in row:
            cols.append(v)
            weights.append(w)
        offsets.append(len(cols))
    assert len(offsets) == n + 1
    for suffix, values in [("RowPtr", offsets), ("ColInd", cols), ("Values", weights)]:
        Path(f"{prefix}{suffix}.txt").write_text("".join(f"{x}\n" for x in values))


def packing_cases(root: Path) -> list[Case]:
    """The packing-boundary constructions of mospTest (checkPipeline only there, no files)."""
    cases = []

    def path_case(
        name: str, n: int, w: int, back_edge: bool, insert: tuple[int, int, int], origin: str
    ) -> Case:
        d = root / name
        rows = [[(u + 1, w)] if u + 1 < n else ([(1, w)] if back_edge else []) for u in range(n)]
        write_csr(d / "graphCsr", n, rows)
        (d / "insert.txt").write_text("{} {} {}\n".format(*insert))
        (d / "delete.txt").write_text("")
        return Case(
            "packing", name, d / "graphCsr", d / "insert.txt", d / "delete.txt", origin, k=1
        )

    n = (1 << 16) + 1
    cases.append(
        path_case(
            "openmp_path",
            n,
            INT_MAX,
            False,
            (n - 1, 1, INT_MAX),
            "MOSP-OpenMP mospTest runPackingBoundary: path n = 2^16 + 1, "
            "weights 2^31 - 1, insert n-1 -> 1",
        )
    )
    n, w = (1 << 17) - 1, (1 << 30) + (1 << 14)
    cases.append(
        path_case(
            "cuda_pull",
            n,
            w,
            False,
            (n - 1, 1, w),
            "MOSP-CUDA mospTest runPackingBoundary 'pull': path n = 2^17 - 1, "
            "W = 2^30 + 2^14, insert n-1 -> 1",
        )
    )
    cases.append(
        path_case(
            "cuda_push",
            n,
            w,
            True,
            (n - 2, n - 1, w - 1),
            "MOSP-CUDA mospTest runPackingBoundary 'push': path with back edge "
            "n-1 -> 1, insert n-2 -> n-1 (W - 1)",
        )
    )
    return cases


# --- Non-canonical initial trees -------------------------------------------------------------


def read_pairs(path: Path) -> list[str]:
    """The second column of a 'vertex value' file (distances or parents), as text."""
    return [line.split()[1] for line in path.read_text().splitlines() if line.strip()]


def perturb_initial_trees(prefix: Path, canonical: Path, out: Path, k: int, name: str) -> list[int]:
    """Copy the canonical trees to `out`, replacing every parent by a random tight in-neighbour.

    Weights are >= 1, so a tight in-neighbour u (dist[u] + w(u,v) == dist[v]) is strictly closer
    to the source than v and any choice gives an acyclic shortest-path tree. The random stream is
    seeded by the case name, so the export is reproducible. Returns the number of changed parents
    per objective.
    """
    rows = [int(x) for x in Path(f"{prefix}RowPtr.txt").read_text().split()]
    cols = [int(x) for x in Path(f"{prefix}ColInd.txt").read_text().split()]
    values = [
        line.split()
        for line in Path(f"{prefix}Values.txt").read_text().splitlines()
        if line.strip()
    ]
    n = len(rows) - 1
    rng = random.Random(int(hashlib.sha256(name.encode()).hexdigest()[:16], 16))
    changed = []
    for obj in range(k):
        src, dst = canonical / f"obj{obj}", out / f"obj{obj}"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / "distancesOriginal.txt", dst / "distancesOriginal.txt")
        dist = [None if d == "INF" else int(d) for d in read_pairs(src / "distancesOriginal.txt")]
        parents = [int(p) for p in read_pairs(src / "SSSPTreeOriginal.txt")]
        tight: list[list[int]] = [[] for _ in range(n)]
        for u in range(n):
            if dist[u] is None:
                continue
            for e in range(rows[u], rows[u + 1]):
                v = cols[e]
                if v != 0 and dist[v] is not None and dist[u] + int(values[e][obj]) == dist[v]:
                    tight[v].append(u)
        count = 0
        for v in range(n):
            options = sorted(set(tight[v]))
            if len(options) > 1:
                p = options[rng.randrange(len(options))]
                count += p != parents[v]
                parents[v] = p
        # SSSPTreeOriginal.txt format: "<vertex> <parent>" per line (writeParents).
        (dst / "SSSPTreeOriginal.txt").write_text(
            "".join(f"{v} {p}\n" for v, p in enumerate(parents))
        )
        changed.append(count)
    return changed


# --- Export of one case ---------------------------------------------------------------------


def export_case(case: Case, ref: Path, out: Path, tmp_root: Path, env: dict) -> dict | None:
    dst = out / case.group / case.name
    tmp = tmp_root / case.group / case.name
    shutil.rmtree(dst, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    (dst / "input").mkdir(parents=True)
    tmp.mkdir(parents=True)
    for f in CSR_FILES:
        shutil.copyfile(f"{case.graph}{f[len('graphCsr') :]}", dst / "input" / f)
    shutil.copyfile(case.insert, dst / "input" / "insert.txt")
    shutil.copyfile(case.delete, dst / "input" / "delete.txt")
    prefix = dst / "input" / "graphCsr"
    k = case.k if case.k is not None else num_weights(dst / "input" / "graphCsrValues.txt")
    if k <= 0:
        raise ExportError(f"{case.rel}: cannot determine K")
    bin_dir = ref / "bin"
    tools = ref / "parity_export" / "bin"

    # Initial trees (Dijkstra, lowest-id ties).
    perturbed: list[int] = []
    if case.perturb:
        run([bin_dir / "mospPrep", "init", prefix, dst / "init_canonical", *k_args(k)], env=env)
        perturbed = perturb_initial_trees(prefix, dst / "init_canonical", dst / "init", k, case.rel)
        if sum(perturbed) == 0:
            shutil.rmtree(dst, ignore_errors=True)
            shutil.rmtree(tmp, ignore_errors=True)
            return None  # no tie anywhere: nothing non-canonical to test
    else:
        run([bin_dir / "mospPrep", "init", prefix, dst / "init", *k_args(k)], env=env)
    if case.mosptest_init is not None:
        for obj in range(k):
            a, b = case.mosptest_init / f"obj{obj}", dst / "init" / f"obj{obj}"
            same(a / "distances.txt", b / "distancesOriginal.txt", f"{case.rel} mospTest init")
            same(a / "tree.txt", b / "SSSPTreeOriginal.txt", f"{case.rel} mospTest init")

    # The reference update: the in-memory driver with the original's own validation.
    log = run(
        [
            bin_dir / "mosp",
            "--graph",
            prefix,
            "--changes",
            dst / "input",
            "--init",
            dst / "init",
            "-k",
            str(k),
            "--out",
            tmp / "mosp",
            "--validate",
        ],
        env=env,
    )
    if "FAIL" in log or "VALIDATE" not in log:
        raise ExportError(f"{case.rel}: mosp --validate did not pass:\n{log}")
    invalidated = {int(o): int(v) for o, v in INVALIDATED.findall(log)}
    if sorted(invalidated) != list(range(k)):
        raise ExportError(f"{case.rel}: no invalidated counters in the mosp report:\n{log}")

    # Cross-check the originals: file-based OpenMP and sequential updates, and Dijkstra.
    run(
        [bin_dir / "mospPrep", "expected", prefix, dst / "input", tmp / "expected", *k_args(k)],
        env=env,
    )
    for obj in range(k):
        upd = tmp / "mosp" / f"obj{obj}"
        init = dst / "init" / f"obj{obj}"
        # sequentialSOSPUpdate re-scans in-neighbours and picks lowest-id ties: it agrees with
        # sospUpdateCpu only from canonical trees (ADR 0006).
        for impl in ["parallel"] if case.perturb else ["sequential", "parallel"]:
            o = tmp / impl / f"obj{obj}"
            run(
                [
                    tools / "export_sssp",
                    impl,
                    prefix,
                    init / "distancesOriginal.txt",
                    init / "SSSPTreeOriginal.txt",
                    dst / "input" / "insert.txt",
                    dst / "input" / "delete.txt",
                    str(obj),
                    "0",
                    o / "distances.txt",
                    o / "tree.txt",
                ],
                env=env,
            )
            same(upd / "distancesUpdated.txt", o / "distances.txt", f"{case.rel} obj{obj} {impl}")
            same(upd / "SSSPTreeUpdated.txt", o / "tree.txt", f"{case.rel} obj{obj} {impl}")
        exp = tmp / "expected" / f"obj{obj}"
        # Dijkstra's tree is canonical; from a perturbed tree only the distances must agree.
        for f in ["distancesUpdated.txt"] + ([] if case.perturb else ["SSSPTreeUpdated.txt"]):
            same(upd / f, exp / f, f"{case.rel} obj{obj} mospPrep expected")
        (dst / "updated" / f"obj{obj}").mkdir(parents=True)
        for f in ["distancesUpdated.txt", "SSSPTreeUpdated.txt"]:
            shutil.copyfile(upd / f, dst / "updated" / f"obj{obj}" / f)
    if case.tracked_expected is not None:
        obj, exp = case.tracked_expected
        pairs = [
            ("distancesOriginal.txt", dst / "init" / f"obj{obj}" / "distancesOriginal.txt"),
            ("SSSPTreeOriginal.txt", dst / "init" / f"obj{obj}" / "SSSPTreeOriginal.txt"),
            ("distancesUpdated.txt", dst / "updated" / f"obj{obj}" / "distancesUpdated.txt"),
            ("SSSPTreeUpdated.txt", dst / "updated" / f"obj{obj}" / "SSSPTreeUpdated.txt"),
            ("distancesSospUpdate.txt", dst / "updated" / f"obj{obj}" / "distancesUpdated.txt"),
            ("SSSPTreeSospUpdate.txt", dst / "updated" / f"obj{obj}" / "SSSPTreeUpdated.txt"),
        ]
        for tracked, golden in pairs:
            same(exp / tracked, golden, f"{case.rel} tracked expected/{tracked}")
    (dst / "combined").mkdir()
    for f in ["distancesCsr.txt", "SSSPTreeCsr.txt", "mospCosts.txt"]:
        shutil.copyfile(tmp / "mosp" / "combinedGraph" / f, dst / "combined" / f)

    # The updated CSR (applyChangeBatch + writeCsrGraph) and the weight-increase bits.
    (dst / "applied").mkdir()
    run(
        [
            tools / "export_graph_io",
            "apply",
            prefix,
            dst / "input" / "insert.txt",
            dst / "input" / "delete.txt",
            dst / "applied" / "graphCsr",
        ],
        env=env,
    )
    for f in (dst / "applied").glob("graphCsrTransposed*"):
        f.unlink()

    meta = {
        "case": case.rel,
        "reference": f"{REFERENCE}@{COMMIT}",
        "origin": case.origin,
        "num_objectives": k,
        "source": 0,
        "invalidated": [invalidated[o] for o in range(k)],
    }
    if case.perturb:
        meta["init"] = "perturbed tie parents (init_canonical/ holds mospPrep init)"
        meta["perturbed_parents"] = perturbed
    (dst / "case.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    shutil.rmtree(tmp, ignore_errors=True)
    return meta


# --- Manifest -------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def case_digest(case_dir: Path) -> tuple[str, int, int]:
    """SHA-256 over the sorted '<sha256>  <relative path>' lines of every file of a case."""
    lines, size = [], 0
    for p in sorted(case_dir.rglob("*")):
        if p.is_file():
            lines.append(f"{sha256_file(p)}  {p.relative_to(case_dir).as_posix()}\n")
            size += p.stat().st_size
    return hashlib.sha256("".join(lines).encode()).hexdigest(), len(lines), size


def write_manifest(out: Path, metas: list[dict]) -> tuple[str, list[str]]:
    lines = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.sha256":
            lines.append(f"{sha256_file(p)}  {p.relative_to(out).as_posix()}\n")
    text = "".join(lines)
    (out / "MANIFEST.sha256").write_text(text)
    return hashlib.sha256(text.encode()).hexdigest(), lines


def write_toml(out: Path, metas: list[dict], manifest_sha: str, n_files: int) -> None:
    by_group: dict[str, int] = {}
    total = 0
    rows = []
    for meta in sorted(metas, key=lambda m: m["case"]):
        group = meta["case"].split("/")[0]
        by_group[group] = by_group.get(group, 0) + 1
        digest, files, size = case_digest(out / meta["case"])
        total += size
        rows.append(
            f'"{meta["case"]}" = {{ k = {meta["num_objectives"]}, files = {files}, '
            f'sha256 = "{digest}" }}\n'
        )
    head = f"""# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Golden manifest of the parity harness (PLAN Section 8.3). GENERATED by parity/export_goldens.py;
# do not edit. The payload is not in the repository: it lives in the persistent work area
# ($DYNG_SCRATCH/goldens/<set>) and is regenerated from the pinned original by
# parity/export_goldens.py. Each case digest is the SHA-256 of the sorted "<sha256>  <path>" lines
# of the case's files (see case_digest() in export_goldens.py); manifest_sha256 is the SHA-256 of
# <set>/MANIFEST.sha256, which lists every file. parity/compare.py verifies both before it compares.
schema = 1

[sets.sssp]
reference = "{REFERENCE}"
commit = "{COMMIT}"
location = "$DYNG_SCRATCH/goldens/sssp"
generated_by = "parity/export_goldens.py"
num_cases = {len(metas)}
num_files = {n_files}
num_bytes = {total}
manifest_sha256 = "{manifest_sha}"

[sets.sssp.groups]
"""
    head += "".join(f"{g} = {by_group[g]}\n" for g in GROUPS if g in by_group)
    head += "\n[sets.sssp.cases]\n"
    (REPO / "parity" / "goldens.toml").write_text(head + "".join(rows))


# --- Driver ----------------------------------------------------------------------------------


def reference_dir(scratch: Path, fresh: bool) -> Path:
    cmd = [REPO / "parity" / "build_reference.sh", "--variant", "patched", "--scratch", scratch]
    if fresh:
        cmd.append("--fresh")
    run(cmd + [REFERENCE])
    out = run(
        [
            REPO / "parity" / "build_reference.sh",
            "--variant",
            "patched",
            "--scratch",
            scratch,
            "--print-dir",
            REFERENCE,
        ]
    )
    return Path(out.strip().splitlines()[-1])


def export(ref: Path, out: Path, groups: list[str], jobs: int) -> tuple[list[dict], str, int]:
    env = dict(os.environ, OMP_NUM_THREADS="4")
    if REFERENCE == "MOSP-CUDA":
        env.setdefault("CUDA_VISIBLE_DEVICES", "1")
        env.setdefault("CUDA_MODULE_LOADING", "EAGER")
    env.pop("OMP_PROC_BIND", None)
    env.pop("OMP_PLACES", None)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    raw = out.parent / f"{out.name}.raw"
    shutil.rmtree(raw, ignore_errors=True)
    raw.mkdir(parents=True)
    try:
        cases = collect(ref, raw, groups, env)
        names = [c.rel for c in cases]
        if len(set(names)) != len(names):
            raise ExportError("duplicate case names")
        print(f"exporting {len(cases)} cases from {ref} into {out} ({jobs} jobs)", flush=True)
        metas = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(export_case, c, ref, out, raw / "tmp", env): c for c in cases}
            for fut in concurrent.futures.as_completed(futures):
                meta = fut.result()
                if meta is not None:
                    metas.append(meta)
    finally:
        shutil.rmtree(raw, ignore_errors=True)
    manifest_sha, lines = write_manifest(out, metas)
    return metas, manifest_sha, len(lines)


def compare_corpora(ours: Path, theirs: Path) -> list[str]:
    """Differences between two exported corpora (--compare-to): file lists, bytes, and case.json
    with its "reference" field ignored."""

    def listing(root: Path) -> dict[str, str]:
        rows = {}
        for line in (root / "MANIFEST.sha256").read_text().splitlines():
            digest, rel = line.split("  ", 1)
            rows[rel] = digest
        return rows

    a, b = listing(ours), listing(theirs)
    problems = [f"only in {ours}: {r}" for r in sorted(set(a) - set(b))]
    problems += [f"only in {theirs}: {r}" for r in sorted(set(b) - set(a))]
    for rel in sorted(set(a) & set(b)):
        if a[rel] == b[rel]:
            continue
        if rel.endswith("case.json"):
            ja = json.loads((ours / rel).read_text())
            jb = json.loads((theirs / rel).read_text())
            ja.pop("reference", None)
            jb.pop("reference", None)
            if ja == jb:
                continue
        problems.append(f"differs: {rel}")
    for root, rows in [(ours, a), (theirs, b)]:  # the manifests must describe the files
        for rel, digest in rows.items():
            if sha256_file(root / rel) != digest:
                problems.append(f"{root / rel}: SHA-256 differs from its MANIFEST.sha256")
    return problems


def main() -> int:
    global REFERENCE, COMMIT
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--scratch",
        type=Path,
        default=Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work")),
    )
    parser.add_argument("--out", type=Path, help="default: <scratch>/goldens/sssp")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--no-toml", action="store_true", help="do not write parity/goldens.toml")
    parser.add_argument(
        "--reference",
        choices=sorted(COMMITS),
        default="MOSP-OpenMP",
        help="the original whose tools export the corpus (MOSP-CUDA: never writes goldens.toml)",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        nargs="?",
        const=Path(),
        help="after the export, require the corpus in DIR (default <scratch>/goldens/sssp) to be "
        "identical except for the reference named in case.json",
    )
    parser.add_argument(
        "--twice",
        action="store_true",
        help="export again from a fresh archive copy and require equal manifests",
    )
    args = parser.parse_args()
    groups = [g for g in args.groups.split(",") if g]
    unknown = set(groups) - set(GROUPS)
    if unknown:
        parser.error(f"unknown groups: {', '.join(sorted(unknown))}")
    REFERENCE, COMMIT = args.reference, COMMITS[args.reference]
    default_name = "sssp" if REFERENCE == "MOSP-OpenMP" else "sssp-mosp-cuda"
    out = args.out or args.scratch / "goldens" / default_name

    ref = reference_dir(args.scratch, fresh=False)
    metas, manifest_sha, n_files = export(ref, out, groups, args.jobs)
    counts: dict[str, int] = {}
    for m in metas:
        counts[m["case"].split("/")[0]] = counts.get(m["case"].split("/")[0], 0) + 1
    print("cases per group: " + ", ".join(f"{g} {counts[g]}" for g in GROUPS if g in counts))
    print(f"{len(metas)} cases, {n_files} files, MANIFEST.sha256 {manifest_sha}")
    if not args.no_toml and groups == GROUPS and REFERENCE == "MOSP-OpenMP":
        write_toml(out, metas, manifest_sha, n_files)
        print(f"wrote {REPO / 'parity' / 'goldens.toml'}")

    if args.compare_to is not None:
        other = args.compare_to if args.compare_to != Path() else args.scratch / "goldens" / "sssp"
        problems = compare_corpora(out, other)
        for line in problems[:50]:
            print(line, file=sys.stderr)
        if problems:
            print(f"CORPORA DIFFER: {len(problems)} problems ({out} vs {other})", file=sys.stderr)
            return 1
        print(
            f"{REFERENCE} corpus identical to {other} ({n_files} files; case.json equal except "
            "for the reference it names)"
        )

    if args.twice:
        with tempfile.TemporaryDirectory(prefix="dyng-goldens-twice-", dir=args.scratch) as t:
            second_scratch = Path(t)
            ref2 = reference_dir(second_scratch, fresh=True)
            _, sha2, _ = export(ref2, second_scratch / "goldens" / "sssp", groups, args.jobs)
            if sha2 != manifest_sha:
                subprocess.run(
                    [
                        "diff",
                        out / "MANIFEST.sha256",
                        second_scratch / "goldens" / "sssp" / "MANIFEST.sha256",
                    ]
                )
                print("SECOND EXPORT DIFFERS", file=sys.stderr)
                return 1
            print(f"second export from a fresh archive copy: identical manifest ({sha2})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ExportError as e:
        print(f"export_goldens: {e}", file=sys.stderr)
        sys.exit(1)
