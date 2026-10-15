#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The triad_count golden set of ESCHER-GPU@abcf9b2 (M9a step 1; PLAN 6.3, 6.4.6, 8.3; ADR 0048).

    parity/triad_goldens.py export [--twice] [--copy DIR] [--out DIR] [--jobs N] [--groups LIST]
                                   [--no-toml] [--keep-copies]
    parity/triad_goldens.py verify [--set-dir DIR]
    parity/triad_goldens.py crosscheck [--groups LIST] [--record FILE] [--jobs N]
    parity/triad_goldens.py list [--backend sequential|openmp|cuda]
    parity/export_goldens.py triad_count ...      (= export)
    parity/compare.py triad_count --verify|--list
    parity/compare.py triad_count --backend B [--mode M] [--counter C] ...  (parity/triad_replay.py)

The set `triad_count_escher_gpu` ($DYNG_SCRATCH/goldens/triad_count/escher_gpu; parity/
goldens.toml) holds, for every counting invocation of A's `make test` (tests/run_tests.sh; 15
groups x build/{main,type1,type2,type3}) and for PLAN 6.4.6's grid (syn100k, syn1m, syn2m, coauth
x seeds 1-3 x batch sizes 20, 5K, 50K, 100K, 200K x 3 batches x the four executables; 240 cases
named <dataset>_b<size>_s<seed>/<executable>):

  cases/<group>/<exe>.json   the case: argv, input, seed, batch composition, insert mode, the full
                             count (30 bins or 1) and per batch {index, deletions, insertions,
                             delta, counts_after (derived), the batch file's SHA-256}, and the
                             verdict of the --check run
  cases/<group>/<exe>.stdout A's unpatched stdout (make-test groups only; M9b's CLI uses it)
  cases/<group>/group.json   the recording run behind the batches: its trace's SHA-256 and recipe,
                             A's vertex count, H2V payload capacity, numHyperedgeKeys and
                             deviceBytes() after the build and after every batch
  cases/<group>/batch<N>.dgb.zst   batch N as a .dgb file of one record (flags deleted_members and
                             expected_ids; ids = A's keys - 1), dyng-compat-escher to-dgb's
                             output split per batch
  inputs/<name>.json, inputs/<name>.hg.zst   each distinct input once: A's loaded rows as a .hg
                             file (index base 1) whose SHA-256 is the loaded-rows hash; coauth-DBLP
                             is named by its SHA-256 only (its loaded rows are the dataset file);
                             synthetic inputs also carry their legacy-generator parameters, and
                             generators::legacy (dyng.generators.legacy) must write the same file
  info.json, MANIFEST.sha256

Counts come from A's UNPATCHED executables (the Makefile-built executables the export patch keeps
as parity_export/orig/<exe> in the patched copy, ADR 0036), run without --check / --reference
and parsed from stdout; every case runs again with --check (exit 0, no FAILED, byte-identical
count lines). Batches come from one run of the patched recording twin of build/main per group in
the recorder's batches mode, converted by `dyng-compat-escher to-dgb`; the four executables draw
the same batches (their batch lines must agree on every case, and the twins of type1-3 record
byte-identical batch files on one group per dataset). Nothing that varies between runs is in
the set (no timings, no [memory] lines, no paths): --twice exports from two FRESH patched copies
and requires byte-identical MANIFEST.sha256 files.

verify: the manifest and every case digest against parity/goldens.toml; bins 20-25 (A's open
triads 21-26) zero in every count and delta; the coauth-DBLP h_motif full count sums to
8,354,454,505; every type1 full count equals sum C(|e|, 3) over the input's rows; counts after
every batch non-negative and as recorded; the batch lines of the four executables equal; every
batch file's counts equal the case's.

crosscheck (PLAN 6.3 step 2): A's unpatched executables of the persistent copy with --check
--reference (OMP_NUM_THREADS=16) on make-test, syn100k s1-s3 at 5000 and 50000, syn1m s1 at
50000 (all four), coauth type1 b50k_s1 and coauth main --batches 1 --batch-size 20 (2 h at most);
every check must be OK and the counts must equal the goldens'. Run the heavy commands under the
shared lock: flock -s $DYNG_SCRATCH/perf.lock nice -n 10 parity/triad_goldens.py ...
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import tomllib
import zlib
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "parity"))

import escher_trace  # noqa: E402

SET = "triad_count_escher_gpu"
REFERENCE = "ESCHER-GPU"
COMMIT = "abcf9b2bd42f896ec3103a6835edf713ed5e416c"
EXES = ("main", "type1", "type2", "type3")
PATTERN = {
    "main": "h_motif",
    "type1": "stathyper_type1",
    "type2": "stathyper_type2",
    "type3": "stathyper_type3",
}
NUM_BINS = {"main": 30, "type1": 1, "type2": 1, "type3": 1}
OPEN_BINS = range(20, 26)  # A's open triads 21-26, 0-based: always 0
COAUTH = "datasets/hyper/coauth.hg"
COAUTH_SHA256 = "65fc83b22dfb3ecd14b15436f4aaee54631657f2f4afec8260150679387b7cb6"
COAUTH_H_MOTIF_TOTAL = 8_354_454_505
SEEDS = (1, 2, 3)
SIZES = (20, 5000, 50000, 100000, 200000)
SIZE_LABEL = {20: "20", 5000: "5k", 50000: "50k", 100000: "100k", 200000: "200k"}
# A's synthetic hypergraphs (n m r1 r2) by dataset name.
SYNTHETIC = {
    "syn100k": (100000, 10, 1, 33333),
    "syn1m": (1000000, 10, 1, 333333),
    "syn2m": (2000000, 10, 1, 666666),
    "syn8": (8, 5, 1, 100),  # make-test's smoke runs (the usage line's example)
    "syn3k": (3000, 8, 1, 2000),  # make-test's sampled inserts on a synthetic input
}
GRID_DATASETS = ("syn100k", "syn1m", "syn2m", "coauth")
# make-test's file inputs (tests/run_tests.sh writes them with gawk); the M8a set
# cbst_trace_escher_gpu_test keeps them, as recorded by its traces.
TEST_FILES = {
    "small": "016-main/inputs/small.hg",
    "skewed": "042-main/inputs/skewed.hg",
    "medium": "043-main/inputs/medium.hg",
}
M8A_TEST_SET = "goldens/cbst_trace/escher_gpu_test"
# The counting invocations of tests/run_tests.sh (every executable but coarseTriangle runs each),
# in the suite's order, with the suite's own arguments ({small} etc.: its gawk-written files).
TEST_GROUPS: list[tuple[str, list[str]]] = [
    ("01-smoke", ["8", "5", "1", "100", "4096"]),
    ("02-reference_small", ["--input", "{small}", "--batches", "0", "--reference"]),
    (
        "03-reference_syn100k",
        ["100000", "10", "1", "33333", "--batches", "0", "--reference", "--seed", "5"],
    ),
    *[
        (
            f"0{4 + k}-small_b{b}",
            ["--input", "{small}", "--batches", "3", "--batch-size", str(b)]
            + ["--check", "--reference", "--seed", "3"],
        )
        for k, b in enumerate((20, 200, 1000))
    ],
    (
        "07-syn100k_b5000",
        ["100000", "10", "1", "33333", "--batches", "3", "--batch-size", "5000"]
        + ["--check", "--reference", "--seed", "5"],
    ),
    (
        "08-deletions_only",
        ["--input", "{small}", "--batches", "3", "--deletions", "300", "--insertions", "0"]
        + ["--check", "--reference"],
    ),
    (
        "09-insertions_only",
        ["--input", "{small}", "--batches", "3", "--deletions", "0", "--insertions", "300"]
        + ["--check", "--reference"],
    ),
    (
        "10-more_deletions",
        ["--input", "{small}", "--batches", "3", "--deletions", "500", "--insertions", "100"]
        + ["--check", "--reference"],
    ),
    (
        "11-random_inserts_file",
        ["--input", "{small}", "--batches", "3", "--batch-size", "400", "--insert-mode", "random"]
        + ["--check", "--reference"],
    ),
    (
        "12-sampled_inserts_synthetic",
        ["3000", "8", "1", "2000", "--batches", "3", "--batch-size", "400", "--insert-mode"]
        + ["sample", "--check", "--reference"],
    ),
    (
        "13-batch_larger_than_graph",
        ["--input", "{small}", "--batches", "3", "--batch-size", "5000", "--check", "--reference"],
    ),
    (
        "14-skewed",
        ["--input", "{skewed}", "--batches", "2", "--batch-size", "2800", "--check", "--reference"],
    ),
    (
        "15-medium",
        ["--input", "{medium}", "--batches", "3", "--batch-size", "640", "--check", "--reference"],
    ),
]
# The groups whose batches the twins of type1, type2 and type3 record as well (one per dataset):
# their batch files must be byte-identical to main's (the same RNG stream).
TWIN_GROUPS = (
    "test/04-small_b20",
    "syn100k_b5k_s1",
    "syn1m_b5k_s1",
    "syn2m_b5k_s1",
    "coauth_b5k_s1",
)
# The coverage table (PLAN 6.4.6, 8.3; the author's decision D6 is pending: the plan's default).
COVERAGE = {
    "cuda": ["make-test", "grid"],
    "sequential": ["make-test", "syn100k"],
    "openmp": ["make-test", "syn100k", "syn1m_s1", "coauth_b50k_s1/type1", "coauth_b50k_s1/type2"],
}
ZSTD = ["zstd", "-q", "-12", "-T1"]
RUN_ENV = {"CUDA_MODULE_LOADING": "EAGER"}  # references.toml run_env of ESCHER-GPU
# A's stdout lines that carry counts (ADR 0048 section 2.2).
FULL_MAIN = re.compile(r"^Motif counts \(30 bins\): (.*)$")
DELTA_MAIN = re.compile(r"^Motif delta counts \(30 bins\): (.*)$")
FULL_TYPE1 = re.compile(r"^\[GPU\] Type1 baseline motif count \(sum_e C\(\|e\|, 3\)\): (-?\d+)$")
FULL_TYPEN = re.compile(r"^\[GPU\] Type([23]) baseline motif count: (-?\d+)$")
DELTA_TYPEN = re.compile(r"^\[GPU\] Type([123]) motif delta count: (-?\d+)$")
BATCH_LINE = re.compile(r"^Batch (\d+) \((\d+) deletions, (\d+) insertions\)$")
HYPERGRAPH_LINE = re.compile(r"^Hypergraph: (\d+) hyperedges, max vertex id (\d+), (\d+) pins$")
COUNT_LINES = (FULL_MAIN, DELTA_MAIN, FULL_TYPE1, FULL_TYPEN, DELTA_TYPEN, BATCH_LINE)
LAUNCHES = re.compile(
    r"^\[check\] (fill|reuse|unfill) launches: thread (\d+), warp (\d+), block (\d+)$"
)
DGB_MAGIC = b"DYNGBAT1"


class GoldenError(Exception):
    pass


def scratch_default() -> Path:
    return Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))


def set_dir(scratch: Path) -> Path:
    return scratch / "goldens" / "triad_count" / "escher_gpu"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def zstd_compress(src: Path, dst: Path) -> None:
    subprocess.run([*ZSTD, "-f", str(src), "-o", str(dst)], check=True)


def zstd_read(path: Path) -> bytes:
    return subprocess.run(["zstd", "-q", "-dc", str(path)], check=True, capture_output=True).stdout


# --- The cases ---------------------------------------------------------------------------------


@dataclass
class Options:
    """What A's parseOptions makes of an argument list (utils/options.cpp)."""

    positional: list[int] = field(default_factory=list)
    input: str | None = None
    batches: int = 1
    deletions: int | None = None
    insertions: int | None = None
    batch_size: int | None = None
    seed: int = 1
    insert_mode: str | None = None
    check: bool = False
    reference: bool = False

    @staticmethod
    def parse(args: list[str]) -> Options:
        o = Options()
        it = iter(range(len(args)))
        for i in it:
            a = args[i]
            if not a.startswith("--"):
                o.positional.append(int(a))
                continue
            if a in ("--check", "--reference"):
                setattr(o, a[2:], True)
                continue
            v = args[i + 1]
            next(it)
            if a == "--input":
                o.input = v
            elif a == "--batches":
                o.batches = int(v)
            elif a == "--batch-size":
                o.batch_size = int(v)
                o.deletions, o.insertions = int(v) // 2, int(v) - int(v) // 2
            elif a == "--deletions":
                o.deletions = int(v)
            elif a == "--insertions":
                o.insertions = int(v)
            elif a == "--seed":
                o.seed = int(v)
            elif a == "--insert-mode":
                o.insert_mode = v
            else:
                raise GoldenError(f"unknown option {a}")
        return o

    def mode(self) -> str:
        return self.insert_mode or ("sample" if self.input else "random")

    def composition(self) -> dict:
        if self.deletions is None and self.insertions is None:
            return {"default": "min(2, live) deletions and 3 insertions (A's default batch)"}
        out = {"deletions": self.deletions or 0, "insertions": self.insertions or 0}
        if self.batch_size is not None:
            out["batch_size"] = self.batch_size
        return out


@dataclass
class Group:
    """One recording run and the four executables that share its batches."""

    name: str  # "test/04-small_b20", "coauth_b50k_s1"
    kind: str  # "test" or "grid"
    args: list[str]  # A's arguments after the executable ({small}, {coauth}: input files)
    dataset: str

    @property
    def options(self) -> Options:
        return Options.parse(self.args)

    @property
    def count_args(self) -> list[str]:
        return [a for a in self.args if a not in ("--check", "--reference")]

    @property
    def input_file(self) -> str | None:
        m = re.fullmatch(r"\{(\w+)\}", self.options.input or "")
        return m.group(1) if m else None

    @property
    def input_name(self) -> str:
        if self.input_file:
            return self.input_file
        n, m, r1, r2 = self.options.positional[:4]
        base = next(k for k, v in SYNTHETIC.items() if v == (n, m, r1, r2))
        return f"{base}_s{self.options.seed}"

    def argv(self, exe: str, extra: list[str] | None = None, label: bool = False) -> list[str]:
        """The command line in the run directory (input files as <name>.hg there), or with
        label=True as recorded (<input:name>)."""
        out = []
        for a in self.count_args + (extra or []):
            m = re.fullmatch(r"\{(\w+)\}", a)
            out.append((f"<input:{m.group(1)}>" if label else f"{m.group(1)}.hg") if m else a)
        return [exe, *out]


def grid_groups() -> list[Group]:
    groups = []
    for ds in GRID_DATASETS:
        for seed in SEEDS:
            for size in SIZES:
                head = ["--input", "{coauth}"] if ds == "coauth" else list(map(str, SYNTHETIC[ds]))
                args = [*head, "--batches", "3", "--batch-size", str(size), "--seed", str(seed)]
                groups.append(Group(f"{ds}_b{SIZE_LABEL[size]}_s{seed}", "grid", args, ds))
    return groups


def test_groups() -> list[Group]:
    out = []
    for name, args in TEST_GROUPS:
        g = Group(f"test/{name}", "test", args, "make-test")
        out.append(g)
    return out


def all_groups() -> list[Group]:
    return test_groups() + grid_groups()


def case_names() -> list[str]:
    return [f"{g.name}/{exe}" for g in all_groups() for exe in EXES]


def coverage_cases(backend: str) -> list[str]:
    """The cases a backend replays (COVERAGE; D6 pending the author)."""
    out = []
    for item in COVERAGE[backend]:
        for g in all_groups():
            for exe in EXES:
                case = f"{g.name}/{exe}"
                if (
                    (item == "make-test" and g.kind == "test")
                    or (item == "grid" and g.kind == "grid")
                    or (item == g.dataset and g.kind == "grid")
                    or (item == f"{g.dataset}_s{g.options.seed}" and g.kind == "grid")
                    or item == case
                ):
                    out.append(case)
    return sorted(set(out))


def suite_invocations() -> list[str]:
    """The 60 invocations as M8a's trace set records them (build/<exe> ..., <abs>/<file>)."""
    out = []
    for g in test_groups():
        for exe in EXES:
            args = [re.sub(r"^\{(\w+)\}$", r"<abs>/\1.hg", a) for a in g.args]
            out.append(" ".join([f"build/{exe}", *args]))
    return out


# --- Parsing A's output ------------------------------------------------------------------------


def count_lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if any(p.match(line) for p in COUNT_LINES)]


def parse_counts(stdout: str, exe: str) -> dict:
    """{full, batches: [{index, printed: (d, i) | None, delta}], hypergraph} from A's stdout."""
    full = None
    batches: list[dict] = []
    pending = None
    hypergraph = None
    for line in stdout.splitlines():
        if m := HYPERGRAPH_LINE.match(line):
            hypergraph = tuple(int(x) for x in m.groups())
        elif m := BATCH_LINE.match(line):
            pending = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        elif (m := FULL_MAIN.match(line)) and exe == "main":
            full = [int(x) for x in m.group(1).split()]
        elif (m := DELTA_MAIN.match(line)) and exe == "main":
            batches.append({"printed": pending, "delta": [int(x) for x in m.group(1).split()]})
            pending = None
        elif (m := FULL_TYPE1.match(line)) and exe == "type1":
            full = [int(m.group(1))]
        elif (m := FULL_TYPEN.match(line)) and exe == f"type{m.group(1)}":
            full = [int(m.group(2))]
        elif (m := DELTA_TYPEN.match(line)) and exe == f"type{m.group(1)}":
            batches.append({"printed": pending, "delta": [int(m.group(2))]})
            pending = None
    if full is None or len(full) != NUM_BINS[exe]:
        raise GoldenError(f"{exe}: no full count of {NUM_BINS[exe]} values in its stdout")
    for k, b in enumerate(batches, start=1):
        if len(b["delta"]) != NUM_BINS[exe]:
            raise GoldenError(f"{exe}: batch {k}: {len(b['delta'])} delta values")
        if b["printed"] is not None and b["printed"][0] != k:
            raise GoldenError(f"{exe}: batch line {b['printed'][0]} where batch {k} was expected")
    return {"full": full, "batches": batches, "hypergraph": hypergraph}


# --- .dgb files (docs/api/file_formats.md, format version 1) -----------------------------------


def dgb_records(data: bytes) -> tuple[bytes, list[bytes]]:
    """The 64-byte header and the records (length, CRC32, padding, payload) of a .dgb file."""
    if len(data) < 64 or data[:8] != DGB_MAGIC:
        raise GoldenError("not a .dgb file")
    (count,) = struct.unpack_from("<Q", data, 32)
    pos, records = 64, []
    for _ in range(count):
        (length,) = struct.unpack_from("<Q", data, pos)
        records.append(data[pos : pos + 16 + length])
        pos += 16 + length
    if pos != len(data):
        raise GoldenError(".dgb: bytes after the last record")
    return data[:64], records


def dgb_counts(record: bytes) -> dict:
    batch_id, d, i, m, q = struct.unpack_from("<5Q", record, 16)
    return {
        "batch_id": batch_id,
        "deletions": d,
        "insertions": i,
        "members": m,
        "deleted_members": q,
    }


def dgb_single(header: bytes, record: bytes) -> bytes:
    """A valid .dgb file holding one record: the header with a batch count of 1 (and its CRC32)."""
    head = bytearray(header)
    struct.pack_into("<Q", head, 32, 1)
    struct.pack_into("<I", head, 60, zlib.crc32(bytes(head[:60])) & 0xFFFFFFFF)
    return bytes(head) + record


# --- Running A ---------------------------------------------------------------------------------


def run_env(gpu: str, extra: dict | None = None) -> dict:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("DYNG_CBST_") and not k.startswith("OMP_")
    }
    env.update(RUN_ENV)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.update(extra or {})
    return env


def run(cmd: list[str], cwd: Path, env: dict, timeout: float | None = None) -> dict:
    start = time.monotonic()
    try:
        p = subprocess.run(
            cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, check=False
        )
        rc, out, err, timed_out = p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired as e:
        rc, timed_out = None, True
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
    return {
        "exit": rc,
        "stdout": out,
        "stderr": err,
        "seconds": time.monotonic() - start,
        "timed_out": timed_out,
    }


def find_compat() -> Path:
    env = os.environ.get("DYNG_COMPAT_ESCHER")
    if env:
        return Path(env)
    for preset in ("parity", "release", "cpu-only", "dev", "parity-cuda", "dev-cuda"):
        exe = REPO / "build" / preset / "tools" / "compat" / "dyng-compat-escher"
        if exe.is_file():
            return exe
    raise GoldenError(
        "dyng-compat-escher not built (cmake --build build/parity --target "
        "dyng-compat-escher), or set DYNG_COMPAT_ESCHER"
    )


def stage_inputs(work: Path, group: Group, sources: dict[str, Path]) -> None:
    work.mkdir(parents=True, exist_ok=True)
    name = group.input_file
    if name:
        link = work / f"{name}.hg"
        if not link.exists():
            os.symlink(sources[name], link)


def record_batches(copy: Path, group: Group, exe: str, work: Path, gpu: str, compat: Path) -> dict:
    """One run of the recording twin build/<exe> in batches mode, converted and split."""
    raw = work / f"rec-{exe}"
    shutil.rmtree(raw, ignore_errors=True)
    for d in ("traces", "inv", "tmp"):
        (raw / d).mkdir(parents=True)
    env = run_env(
        gpu,
        {
            "DYNG_CBST_TRACE_DIR": str(raw / "traces"),
            "DYNG_CBST_INVOCATION_LOG": str(raw / "inv"),
            "DYNG_CBST_TRACE_MODE": "batches",
            "TMPDIR": str(raw / "tmp"),
        },
    )
    r = run([str(copy / "build" / exe), *group.argv(exe)[1:]], work, env)
    if r["exit"] != 0:
        raise GoldenError(f"{group.name}: recording twin {exe} failed: {r['stderr'][-1500:]}")
    invs = sorted((raw / "inv").glob("*.inv"))
    traces = sorted((raw / "traces").glob("*.cbt"))
    if len(invs) != 1 or len(traces) != 1:
        raise GoldenError(f"{group.name}: {len(invs)} invocation logs, {len(traces)} traces")
    inv = invs[0].read_text()
    if "exit=normal" not in inv:
        raise GoldenError(f"{group.name}: the recording run did not exit normally")
    trace = traces[0]
    facts = {"device_bytes": [], "num_hyperedge_keys": [], "inserted_keys": []}
    head = end = None
    for kind, rec in escher_trace.Trace(trace).records():
        if kind == "header":
            head = rec
        elif kind == "end":
            end = rec
        elif kind == "batch" and rec.get("phase") == "exit":
            if "exception" in rec or "exception.type" in rec:
                raise GoldenError(f"{group.name}: A threw in {rec.get('hook')}")
            facts["device_bytes"].append(rec["device_bytes"])
            facts["num_hyperedge_keys"].append(rec["num_hyperedge_keys"])
            if rec["hook"] == "escher_gpu.build":
                facts["max_vertex_id"] = rec["max_vertex_id"]
            else:
                facts["inserted_keys"].append(rec["inserted_keys"].count)
    if head is None or end is None or end.get("exit") != "normal":
        raise GoldenError(f"{group.name}: the trace did not end normally")
    if head.get("mode") != "batches":
        raise GoldenError(f"{group.name}: trace mode {head.get('mode')}")
    blob = trace.with_suffix(".blob")
    conv = raw / "conv"
    p = subprocess.run(
        [str(compat), "to-dgb", str(trace), str(conv), "--tmp", str(raw / "tmp")],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise GoldenError(f"{group.name}: to-dgb failed: {p.stderr}")
    txt = {}
    for line in Path(f"{conv}.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        key, _, value = line.partition(" ")
        txt[key] = value
    header, records = dgb_records(Path(f"{conv}.dgb").read_bytes())
    batches = []
    for n, record in enumerate(records, start=1):
        data = dgb_single(header, record)
        path = raw / f"batch{n}.dgb"
        path.write_bytes(data)
        counts = dgb_counts(record)
        if counts["batch_id"] != n - 1:
            raise GoldenError(f"{group.name}: record {n} has batch id {counts['batch_id']}")
        batches.append(
            {
                "index": n,
                "path": path,
                "sha256": sha256_bytes(data),
                "bytes": len(data),
                **{k: counts[k] for k in ("deletions", "insertions", "members", "deleted_members")},
            }
        )
    if len(batches) != group.options.batches:
        raise GoldenError(f"{group.name}: {len(batches)} batches, {group.options.batches} asked")
    for b, keys in zip(batches, facts["inserted_keys"], strict=True):
        if b["insertions"] != keys:
            raise GoldenError(f"{group.name}: batch {b['index']}: inserted keys != insertions")
    return {
        "trace_sha256": sha256_file(trace),
        "trace_bytes": trace.stat().st_size,
        "blob_sha256": sha256_file(blob) if blob.is_file() else None,
        "blob_bytes": blob.stat().st_size if blob.is_file() else 0,
        "argv": head.get("argv"),
        "hg": Path(f"{conv}.hg"),
        "txt": txt,
        "batches": batches,
        "facts": facts,
        "raw": raw,
    }


def run_case(copy: Path, group: Group, exe: str, work: Path, gpu: str) -> dict:
    """The unpatched count run and its --check twin run of one executable."""
    orig = copy / "parity_export" / "orig" / exe
    env = run_env(gpu)
    count = run([str(orig), *group.argv(exe)[1:]], work, env)
    if count["exit"] != 0:
        raise GoldenError(f"{group.name}/{exe}: exit {count['exit']}: {count['stderr'][-1500:]}")
    check = run([str(orig), *group.argv(exe, ["--check"])[1:]], work, env)
    lines = count_lines(count["stdout"])
    verdict = {
        "exit": check["exit"],
        "failed_lines": sum("FAILED" in x for x in check["stderr"].splitlines()),
        "integrity_ok": sum(
            "data-structure integrity OK" in x for x in check["stderr"].splitlines()
        ),
        "recount_ok": sum(
            "baseline + deltas == full GPU recount: OK" in x for x in check["stderr"].splitlines()
        ),
        "count_lines_identical": count_lines(check["stdout"]) == lines,
        "stdout_identical": check["stdout"] == count["stdout"],
        "launches": {
            m.group(1): [int(m.group(k)) for k in (2, 3, 4)]
            for x in check["stderr"].splitlines()
            if (m := LAUNCHES.match(x))
        },
    }
    batches = group.options.batches
    if (
        check["exit"] != 0
        or verdict["failed_lines"]
        or not verdict["count_lines_identical"]
        or verdict["integrity_ok"] != batches + 1
        or verdict["recount_ok"] != batches
    ):
        raise GoldenError(f"{group.name}/{exe}: the --check run disagrees: {verdict}")
    return {
        "stdout": count["stdout"],
        "parsed": parse_counts(count["stdout"], exe),
        "check": verdict,
        "seconds": count["seconds"] + check["seconds"],
        "stderr_memory": [x for x in count["stderr"].splitlines() if x.startswith("[memory]")],
    }


def type1_sum(hg: bytes) -> tuple[int, int, int, int]:
    """(sum C(|e|, 3), hyperedges, pins, max cardinality) of a .hg file of loaded rows."""
    total = edges = pins = maxcard = 0
    for line in hg.splitlines():
        if not line or line[:1] in (b"#", b"%"):
            continue
        k = len(line.split())
        total += math.comb(k, 3)
        edges += 1
        pins += k
        maxcard = max(maxcard, k)
    return total, edges, pins, maxcard


LEGACY_SCRIPT = r"""
import hashlib, sys, tempfile, os
from dyng import io
from dyng.generators import legacy
n, m, r1, r2, seed = map(int, sys.argv[1:6])
lst = legacy.escher_hypergraph_arrays(legacy.EscherEngine(seed), num_hyperedges=n,
                                      max_cardinality=m, min_vertex=r1 - 1, max_vertex=r2 - 1)
fd, path = tempfile.mkstemp(suffix=".hg", dir=sys.argv[6]); os.close(fd)
io.write_hypergraph(path, lst, index_base=1)
h = hashlib.sha256(open(path, "rb").read()).hexdigest(); os.remove(path)
print(h, lst.num_vertices)
"""


LEGACY_LOCK = threading.Lock()  # one import of dyng at a time (its editable install may rebuild)


def legacy_hash(params: tuple[int, int, int, int], seed: int, tmp: Path) -> tuple[str, int]:
    """SHA-256 of the .hg file that dyng's generators::legacy::escher_hypergraph draws (through
    the Python binding), and its vertex count."""
    with LEGACY_LOCK:
        return _legacy_hash(params, seed, tmp)


def _legacy_hash(params: tuple[int, int, int, int], seed: int, tmp: Path) -> tuple[str, int]:
    p = subprocess.run(
        [sys.executable, "-c", LEGACY_SCRIPT, *map(str, params), str(seed), str(tmp)],
        capture_output=True,
        text=True,
        cwd=tmp,
    )
    if p.returncode != 0:
        raise GoldenError(f"generators::legacy (Python) failed: {p.stderr[-1500:]}")
    digest, vertices = p.stdout.split()
    return digest, int(vertices)


# --- Export ------------------------------------------------------------------------------------


def tool_versions() -> dict:
    return {
        "zstd": subprocess.run(
            ["zstd", "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "zstd_flags": " ".join(ZSTD[1:]),
        "converter": "dyng-compat-escher to-dgb (tools/compat/escher), split one record per file",
    }


def test_sources(scratch: Path) -> dict[str, Path]:
    base = scratch / M8A_TEST_SET
    out = {}
    for name, rel in TEST_FILES.items():
        path = base / rel
        case = json.loads((path.parent.parent / "case.json").read_text())
        want = next(i["sha256"] for i in case["inputs"] if i["name"] == f"{name}.hg")
        if not path.is_file() or sha256_file(path) != want:
            raise GoldenError(f"{path}: missing or not the SHA-256 M8a's set records")
        out[name] = path
    return out


def check_suite_invocations(scratch: Path) -> int:
    """make-test's 60 counting invocations are exactly those M8a's trace set recorded."""
    recorded = []
    for f in sorted((scratch / M8A_TEST_SET).glob("*/case.json")):
        argv = json.loads(f.read_text())["argv"]
        if argv[0].rsplit("/", 1)[-1] in EXES:
            recorded.append(" ".join(argv))
    mine = suite_invocations()
    if sorted(recorded) != sorted(mine):
        raise GoldenError(
            "make-test's invocations differ from M8a's set: "
            f"{sorted(set(recorded) ^ set(mine))[:5]}"
        )
    return len(mine)


def export_group(
    copy: Path, group: Group, raw: Path, out: Path, gpu: str, compat: Path, log
) -> dict:
    work = raw / group.name.replace("/", "__")
    shutil.rmtree(work, ignore_errors=True)
    stage_inputs(work, group, log["sources"])
    t0 = time.monotonic()
    rec = record_batches(copy, group, "main", work, gpu, compat)
    twins = {}
    if group.name in TWIN_GROUPS:
        for exe in EXES[1:]:
            other = record_batches(copy, group, exe, work, gpu, compat)
            same = [b["sha256"] for b in other["batches"]] == [b["sha256"] for b in rec["batches"]]
            if not same:
                raise GoldenError(f"{group.name}: the {exe} twin recorded other batches than main")
            twins[exe] = "byte-identical batch files"
            shutil.rmtree(other["raw"], ignore_errors=True)
    runs = {exe: run_case(copy, group, exe, work, gpu) for exe in EXES}
    gdir = out / "cases" / group.name
    gdir.mkdir(parents=True, exist_ok=True)
    opts = group.options
    batch_entries = []
    for b in rec["batches"]:
        name = f"batch{b['index']}.dgb.zst"
        zstd_compress(b["path"], gdir / name)
        batch_entries.append({k: v for k, v in b.items() if k != "path"} | {"file": name})
    facts = rec["facts"]
    vertices = int(rec["txt"]["vertices"])
    hyperedges = int(rec["txt"]["hyperedges"])
    group_doc = {
        "group": group.name,
        "kind": group.kind,
        "dataset": group.dataset,
        "input": group.input_name,
        "seed": opts.seed,
        "argv": group.argv("main", label=True),
        "suite_argv": group.args if group.kind == "test" else None,
        "recording": {
            "recipe": "the patched copy's recording twin build/main (ADR 0036), "
            "DYNG_CBST_TRACE_MODE=batches, CUDA_VISIBLE_DEVICES=" + gpu + ", run in a directory "
            "holding the input as <name>.hg; then dyng-compat-escher to-dgb and one .dgb per "
            "record (the raw trace is not kept)",
            "trace_sha256": rec["trace_sha256"],
            "trace_bytes": rec["trace_bytes"],
            "blob_sha256": rec["blob_sha256"],
            "blob_bytes": rec["blob_bytes"],
            "trace_argv": rec["argv"],
        },
        "vertices": vertices,
        "hyperedges": hyperedges,
        "payload_capacity": int(rec["txt"]["payload_capacity"]),
        "max_vertex_id": facts["max_vertex_id"],
        "num_hyperedge_keys": facts["num_hyperedge_keys"],
        "device_bytes": facts["device_bytes"],
        "batches": batch_entries,
        "twins": twins,
        "loaded_rows_sha256": sha256_file(rec["hg"]),
    }
    write_json(gdir / "group.json", group_doc)
    for exe in EXES:
        r = runs[exe]
        parsed = r["parsed"]
        if parsed["hypergraph"] and (parsed["hypergraph"][0], parsed["hypergraph"][1]) != (
            hyperedges,
            vertices,
        ):
            raise GoldenError(f"{group.name}/{exe}: Hypergraph line {parsed['hypergraph']}")
        if len(parsed["batches"]) != len(rec["batches"]):
            raise GoldenError(f"{group.name}/{exe}: {len(parsed['batches'])} deltas printed")
        counts = list(parsed["full"])
        batches = []
        for b, p in zip(rec["batches"], parsed["batches"], strict=True):
            if p["printed"] is not None and (p["printed"][1], p["printed"][2]) != (
                b["deletions"],
                b["insertions"],
            ):
                raise GoldenError(f"{group.name}/{exe}: batch {b['index']}: {p['printed']}")
            counts = [c + d for c, d in zip(counts, p["delta"], strict=True)]
            batches.append(
                {
                    "index": b["index"],
                    "deletions": b["deletions"],
                    "insertions": b["insertions"],
                    "batch_line_printed": p["printed"] is not None,
                    "delta": p["delta"],
                    "counts_after": counts,
                    "batch_file": f"batch{b['index']}.dgb.zst",
                    "batch_sha256": b["sha256"],
                }
            )
        case = {
            "case": f"{group.name}/{exe}",
            "group": group.name,
            "reference": REFERENCE,
            "commit": COMMIT,
            "executable": exe,
            "pattern": PATTERN[exe],
            "num_bins": NUM_BINS[exe],
            "argv": group.argv(exe, label=True),
            "check_argv": group.argv(exe, ["--check"], label=True),
            "suite_argv": [exe, *group.args] if group.kind == "test" else None,
            "input": group.input_name,
            "loaded_rows_sha256": group_doc["loaded_rows_sha256"],
            "seed": opts.seed,
            "batches_requested": opts.batches,
            "composition": opts.composition(),
            "insert_mode": opts.mode(),
            "full_count": parsed["full"],
            "batches": batches,
            "check": r["check"],
            "stdout_sha256": sha256_bytes(r["stdout"].encode()),
        }
        write_json(gdir / f"{exe}.json", case)
        if group.kind == "test":
            (gdir / f"{exe}.stdout").write_text(r["stdout"])
    # The input, once (the caller deduplicates by name and checks the loaded-rows hash).
    result = {
        "group": group.name,
        "input": group.input_name,
        "hg": rec["hg"],
        "loaded_rows_sha256": group_doc["loaded_rows_sha256"],
        "vertices": vertices,
        "seconds": time.monotonic() - t0,
        "memory": {exe: runs[exe]["stderr_memory"] for exe in EXES},
        "work": work,
    }
    return result


def export_set(
    copy: Path,
    out: Path,
    scratch: Path,
    gpu: str,
    jobs: int,
    only: list[str] | None,
    log_path: Path,
) -> dict:
    """One export from the patched copy `copy` into `out`."""
    compat = find_compat()
    raw = out.parent / (out.name + ".raw")
    shutil.rmtree(raw, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    raw.mkdir(parents=True)
    (out / "inputs").mkdir(parents=True)
    sources = test_sources(scratch)
    coauth = scratch / COAUTH
    if sha256_file(coauth) != COAUTH_SHA256:
        raise GoldenError(f"{coauth}: SHA-256 is not {COAUTH_SHA256}")
    sources["coauth"] = coauth
    n_suite = check_suite_invocations(scratch)
    groups = [g for g in all_groups() if not only or g.name in only or g.dataset in only]
    log = {"sources": sources}
    lock = threading.Lock()
    inputs: dict[str, dict] = {}
    timings = {}
    logf = open(log_path, "a")  # noqa: SIM115 - closed below

    def one(g: Group) -> None:
        res = export_group(copy, g, raw, out, gpu, compat, log)
        with lock:
            name = res["input"]
            if name in inputs:
                if inputs[name]["loaded_rows_sha256"] != res["loaded_rows_sha256"]:
                    raise GoldenError(f"{g.name}: input {name} loaded other rows than before")
                res["hg"].unlink()
            else:
                inputs[name] = res
            timings[g.name] = res["seconds"]
            print(f"  {g.name}: {res['seconds']:.1f} s", flush=True)
            logf.write(
                json.dumps({"group": g.name, "seconds": res["seconds"], "memory": res["memory"]})
                + "\n"
            )
            logf.flush()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            for f in [pool.submit(one, g) for g in groups]:
                f.result()
        legacy_tmp = raw / "legacy"
        legacy_tmp.mkdir()
        for name, res in sorted(inputs.items()):
            hg = res["hg"].read_bytes()
            t1, edges, pins, maxcard = type1_sum(hg)
            doc = {
                "name": name,
                "loaded_rows_sha256": res["loaded_rows_sha256"],
                "vertices": res["vertices"],
                "hyperedges": edges,
                "pins": pins,
                "max_cardinality": maxcard,
                "type1_sum_choose3": t1,
            }
            if name == "coauth":
                if res["loaded_rows_sha256"] != COAUTH_SHA256:
                    raise GoldenError("A's loaded coauth-DBLP rows differ from the dataset file")
                doc["file"] = {"path": COAUTH, "sha256": COAUTH_SHA256}
                doc["stored"] = None
            else:
                zstd_compress(res["hg"], out / "inputs" / f"{name}.hg.zst")
                doc["stored"] = f"inputs/{name}.hg.zst"
                if name in TEST_FILES:
                    src = sources[name]
                    zstd_compress(src, out / "inputs" / f"{name}.source.hg.zst")
                    doc["source"] = {
                        "stored": f"inputs/{name}.source.hg.zst",
                        "sha256": sha256_file(src),
                        "origin": f"tests/run_tests.sh (gawk), as M8a's set {M8A_TEST_SET} "
                        "recorded it",
                    }
                else:
                    base, seed = name.rsplit("_s", 1)
                    params = SYNTHETIC[base]
                    digest, nv = legacy_hash(params, int(seed), legacy_tmp)
                    doc["generator"] = {
                        "original": "A's positional arguments n m r1 r2 and --seed",
                        "n_m_r1_r2": list(params),
                        "seed": int(seed),
                        "dyng": "generators::legacy::escher_hypergraph(std::mt19937_64(seed), "
                        "{n, m, r1 - 1, r2 - 1}), written with index_base 1",
                        "dyng_sha256": digest,
                        "dyng_num_vertices": nv,
                        "reproduces_loaded_rows": digest == res["loaded_rows_sha256"],
                    }
                    if digest != res["loaded_rows_sha256"]:
                        raise GoldenError(f"generators::legacy does not reproduce {name}")
            write_json(out / "inputs" / f"{name}.json", doc)
        info = {
            "set": SET,
            "reference": REFERENCE,
            "commit": COMMIT,
            "what": "ESCHER-GPU@abcf9b2's triad counts (build/main, type1, type2, type3) on "
            "make-test and PLAN 6.4.6's grid; batches from the recording twin",
            "counts_from": "parity_export/orig/<exe> of a fresh patched copy: A's executables "
            "as its Makefile built them (unpatched), without --check / --reference; each case "
            "again with --check (exit 0, byte-identical count lines)",
            "grid": {
                "datasets": {
                    k: (" ".join(map(str, SYNTHETIC[k])) if k in SYNTHETIC else "--input coauth.hg")
                    for k in GRID_DATASETS
                },
                "seeds": list(SEEDS),
                "batch_sizes": list(SIZES),
                "batches": 3,
                "size_labels": {str(k): v for k, v in SIZE_LABEL.items()},
            },
            "make_test": {
                "groups": [g.name for g in test_groups()],
                "invocations": n_suite,
                "matches_m8a_set": M8A_TEST_SET,
                "not_cases": "the 7 input-rejection invocations of build/main (no counts)",
            },
            "executables": list(EXES),
            "twin_groups": list(TWIN_GROUPS),
            "coverage": COVERAGE,
            "tools": tool_versions(),
            "groups": len(groups),
            "cases": len(groups) * len(EXES),
            "inputs": sorted(inputs),
        }
        write_json(out / "info.json", info)
    finally:
        logf.close()
        shutil.rmtree(raw, ignore_errors=True)
    sha, n_files, size = write_manifest(out)
    return {
        "manifest_sha256": sha,
        "files": n_files,
        "bytes": size,
        "groups": groups,
        "timings": timings,
    }


def write_manifest(out: Path) -> tuple[str, int, int]:
    lines, size = [], 0
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.sha256":
            lines.append(f"{sha256_file(p)}  {p.relative_to(out).as_posix()}\n")
            size += p.stat().st_size
    text = "".join(lines)
    (out / "MANIFEST.sha256").write_text(text)
    return sha256_bytes(text.encode()), len(lines), size


def case_files(out: Path, case: str) -> list[Path]:
    group, exe = case.rsplit("/", 1)
    gdir = out / "cases" / group
    files = [gdir / f"{exe}.json", gdir / "group.json", *sorted(gdir.glob("batch*.dgb.zst"))]
    if (gdir / f"{exe}.stdout").is_file():
        files.append(gdir / f"{exe}.stdout")
    return files


def case_digest(out: Path, case: str) -> str:
    lines = [f"{sha256_file(p)}  {p.relative_to(out).as_posix()}\n" for p in case_files(out, case)]
    return sha256_bytes("".join(sorted(lines)).encode())


def toml_section(out: Path, result: dict) -> str:
    rows = []
    for g in result["groups"]:
        for exe in EXES:
            case = f"{g.name}/{exe}"
            doc = json.loads((out / "cases" / g.name / f"{exe}.json").read_text())
            rows.append(
                f'"{case}" = {{ batches = {len(doc["batches"])}, '
                f'sha256 = "{case_digest(out, case)}" }}\n'
            )
    inputs = []
    for f in sorted((out / "inputs").glob("*.json")):
        doc = json.loads(f.read_text())
        extra = ""
        if "generator" in doc:
            extra = (
                f', generator = "{" ".join(map(str, doc["generator"]["n_m_r1_r2"]))}", '
                f"seed = {doc['generator']['seed']}"
            )
        elif "file" in doc:
            extra = f', file = "{doc["file"]["path"]}"'
        inputs.append(
            f'"{doc["name"]}" = {{ loaded_rows_sha256 = "{doc["loaded_rows_sha256"]}", '
            f"hyperedges = {doc['hyperedges']}{extra} }}\n"
        )
    cov = "".join(f"{k} = {json.dumps(v)}\n" for k, v in COVERAGE.items())
    return (
        f"[sets.{SET}]\n"
        f'reference = "{REFERENCE}"\n'
        f'commit = "{COMMIT}"\n'
        'kind = "triad_count"\n'
        'what = "triad counts of build/main, type1, type2, type3 on make-test (15 groups) and '
        "PLAN 6.4.6's grid (syn100k, syn1m, syn2m, coauth x seeds 1-3 x 5 batch sizes x 3 "
        'batches); batches as one .dgb per batch"\n'
        'location = "$DYNG_SCRATCH/goldens/triad_count/escher_gpu"\n'
        'generated_by = "parity/triad_goldens.py export --twice (parity/export_goldens.py '
        'triad_count)"\n'
        f"num_groups = {len(result['groups'])}\n"
        f"num_cases = {len(result['groups']) * len(EXES)}\n"
        f"num_files = {result['files']}\n"
        f"num_bytes = {result['bytes']}\n"
        f'manifest_sha256 = "{result["manifest_sha256"]}"\n'
        f"\n[sets.{SET}.coverage]\n"
        "# Which backend replays which cases (PLAN 6.4.6, 8.3): the plan's default, pending the "
        "author's decision D6 (2026-10-15).\n"
        'status = "pending the author (D6): the plan\'s default"\n'
        + cov
        + f"\n[sets.{SET}.inputs]\n"
        + "".join(inputs)
        + f"\n[sets.{SET}.cases]\n"
        + "".join(rows)
    )


def build_copy(scratch: Path) -> Path:
    tool = REPO / "parity" / "build_reference.sh"
    base = [str(tool), "--variant", "patched", "--scratch", str(scratch)]
    subprocess.run([*base, "--fresh", REFERENCE], check=True)
    out = subprocess.check_output([*base, "--print-dir", REFERENCE], text=True)
    return Path(out.strip().splitlines()[-1])


def origin_status() -> str:
    origin = Path(os.environ.get("DYNG_ORIGINALS_DIR", Path.home() / "Projects")) / REFERENCE
    head = subprocess.run(
        ["git", "-C", str(origin), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout
    status = subprocess.run(
        ["git", "-C", str(origin), "status", "--porcelain"], capture_output=True, text=True
    ).stdout
    return head + status


def export_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="triad_goldens.py export",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--scratch", type=Path, default=scratch_default())
    p.add_argument(
        "--twice",
        action="store_true",
        help="export from two FRESH patched copies (built under $DYNG_SCRATCH/ref/"
        "m9a-export-<k>) and require identical manifests; the copies are removed",
    )
    p.add_argument("--copy", type=Path, help="export once from this patched copy (development)")
    p.add_argument("--out", type=Path, help="output directory (default: the set's location)")
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1").split(",")[0])
    p.add_argument("--jobs", type=int, default=3, help="groups recorded at once per export")
    p.add_argument("--groups", default="", help="only these groups or datasets (comma list)")
    p.add_argument("--no-toml", action="store_true")
    p.add_argument("--keep-copies", action="store_true")
    args = p.parse_args(argv)
    only = [x for x in args.groups.split(",") if x] or None
    out = args.out or set_dir(args.scratch)
    log_dir = args.scratch / "runs" / "m9a" / "goldens"
    log_dir.mkdir(parents=True, exist_ok=True)
    before = origin_status()
    stamp = time.strftime("%Y%m%dT%H%M%S")
    try:
        if args.copy:
            result = export_set(
                args.copy,
                out,
                args.scratch,
                args.gpu,
                args.jobs,
                only,
                log_dir / f"export-{stamp}.jsonl",
            )
        elif args.twice:
            copies = [args.scratch / "ref" / f"m9a-export-{k}" for k in (1, 2)]
            dirs = [build_copy(c) for c in copies]
            second = out.parent / (out.name + ".second")
            first = out.parent / (out.name + ".first")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futs = [
                    pool.submit(
                        export_set,
                        d,
                        o,
                        args.scratch,
                        args.gpu,
                        args.jobs,
                        only,
                        log_dir / f"export-{stamp}-{k}.jsonl",
                    )
                    for k, (d, o) in enumerate(zip(dirs, (first, second), strict=True), start=1)
                ]
                results = [f.result() for f in futs]
            m1 = (first / "MANIFEST.sha256").read_bytes()
            m2 = (second / "MANIFEST.sha256").read_bytes()
            if m1 != m2:
                subprocess.run(
                    ["diff", str(first / "MANIFEST.sha256"), str(second / "MANIFEST.sha256")]
                )
                print(
                    "SECOND EXPORT DIFFERS (the copies and both exports are kept)", file=sys.stderr
                )
                return 1
            print(
                f"second export from a fresh copy: identical MANIFEST.sha256 "
                f"({results[1]['manifest_sha256']})",
                flush=True,
            )
            shutil.rmtree(second)
            shutil.rmtree(out, ignore_errors=True)
            first.rename(out)
            result = results[0]
            if not args.keep_copies:
                for c in copies:
                    shutil.rmtree(c)
                print(f"removed the copies {', '.join(map(str, copies))}")
        else:
            p.error("give --twice (the golden export) or --copy DIR (one development export)")
    except GoldenError as e:
        print(f"triad_goldens export: {e}", file=sys.stderr)
        return 1
    after = origin_status()
    if before != after:
        print(f"{REFERENCE}'s git state changed during the export", file=sys.stderr)
        return 1
    print(
        f"{SET}: {len(result['groups'])} groups, {len(result['groups']) * 4} cases, "
        f"{result['files']} files, {result['bytes'] / 1e6:.1f} MB, MANIFEST.sha256 "
        f"{result['manifest_sha256']}"
    )
    if not args.no_toml and not only and (args.twice or args.out is None):
        import cycle_count_goldens  # the goldens.toml section writer

        cycle_count_goldens.write_toml(toml_section(out, result), SET)
        print(f"wrote the [sets.{SET}] section of parity/goldens.toml")
    return 0


# --- verify ------------------------------------------------------------------------------------


def golden_spec() -> dict | None:
    return tomllib.loads((REPO / "parity" / "goldens.toml").read_text()).get("sets", {}).get(SET)


def verify_set(out: Path, scratch: Path, spec: dict | None = None) -> list[str]:
    spec = spec if spec is not None else golden_spec()
    if spec is None:
        return [f"parity/goldens.toml has no [sets.{SET}]"]
    if not (out / "MANIFEST.sha256").is_file():
        return [f"{out}: no MANIFEST.sha256 (parity/triad_goldens.py export --twice)"]
    problems: list[str] = []
    manifest = (out / "MANIFEST.sha256").read_bytes()
    if sha256_bytes(manifest) != spec["manifest_sha256"]:
        problems.append("MANIFEST.sha256 differs from goldens.toml")
    listed = set()
    for line in manifest.decode().splitlines():
        digest, rel = line.split("  ", 1)
        listed.add(rel)
        if not (out / rel).is_file() or sha256_file(out / rel) != digest:
            problems.append(f"{rel} differs from MANIFEST.sha256")
    on_disk = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    if on_disk - listed - {"MANIFEST.sha256"}:
        problems.append(f"files not in the manifest: {sorted(on_disk - listed)[:5]}")
    cases = spec.get("cases", {})
    if sorted(cases) != sorted(case_names()):
        problems.append("the cases of goldens.toml are not make-test plus the grid")
    for case, c in cases.items():
        if case_digest(out, case) != c["sha256"]:
            problems.append(f"case {case} differs from goldens.toml")
    problems += check_contents(out, scratch)
    return problems


def check_contents(out: Path, scratch: Path) -> list[str]:
    """The assertions of ADR 0048 section 2.8 on the set's contents."""
    problems: list[str] = []
    inputs = {}
    for f in sorted((out / "inputs").glob("*.json")):
        doc = json.loads(f.read_text())
        if doc["stored"]:
            data = zstd_read(out / doc["stored"])
        else:
            data = (scratch / doc["file"]["path"]).read_bytes()
        if sha256_bytes(data) != doc["loaded_rows_sha256"]:
            problems.append(f"input {doc['name']}: rows differ from the loaded-rows hash")
        t1, edges, pins, _ = type1_sum(data)
        if (t1, edges, pins) != (doc["type1_sum_choose3"], doc["hyperedges"], doc["pins"]):
            problems.append(f"input {doc['name']}: sum C(|e|,3), hyperedges or pins differ")
        if "generator" in doc and not doc["generator"]["reproduces_loaded_rows"]:
            problems.append(f"input {doc['name']}: generators::legacy does not reproduce it")
        inputs[doc["name"]] = doc | {"type1_from_rows": t1}
    present = [g for g in all_groups() if (out / "cases" / g.name / "group.json").is_file()]
    for g in present:
        gdir = out / "cases" / g.name
        group = json.loads((gdir / "group.json").read_text())
        files = {}
        for b in group["batches"]:
            data = zstd_read(gdir / b["file"])
            if sha256_bytes(data) != b["sha256"]:
                problems.append(f"{g.name}: {b['file']}: SHA-256 differs from group.json")
            _, records = dgb_records(data)
            counts = dgb_counts(records[0]) if len(records) == 1 else None
            if counts is None or (counts["deletions"], counts["insertions"]) != (
                b["deletions"],
                b["insertions"],
            ):
                problems.append(f"{g.name}: {b['file']}: record counts differ from group.json")
            files[b["index"]] = b
        if group["loaded_rows_sha256"] != inputs[group["input"]]["loaded_rows_sha256"]:
            problems.append(f"{g.name}: loaded rows differ from input {group['input']}")
        shapes = {}
        for exe in EXES:
            case = json.loads((gdir / f"{exe}.json").read_text())
            name = case["case"]
            counts = list(case["full_count"])
            for b in case["batches"]:
                if exe == "main" and any(b["delta"][i] for i in OPEN_BINS):
                    problems.append(f"{name}: batch {b['index']}: open-triad bins in the delta")
                counts = [c + d for c, d in zip(counts, b["delta"], strict=True)]
                if counts != b["counts_after"] or min(counts) < 0:
                    problems.append(f"{name}: batch {b['index']}: counts after differ or < 0")
                f = files.get(b["index"])
                if f is None or (f["sha256"], f["deletions"], f["insertions"]) != (
                    b["batch_sha256"],
                    b["deletions"],
                    b["insertions"],
                ):
                    problems.append(f"{name}: batch {b['index']}: the batch file differs")
            if exe == "main":
                if any(case["full_count"][i] for i in OPEN_BINS):
                    problems.append(f"{name}: open-triad bins in the full count")
                if g.dataset == "coauth" and sum(case["full_count"]) != COAUTH_H_MOTIF_TOTAL:
                    problems.append(f"{name}: the full count sums to {sum(case['full_count'])}")
            if exe == "type1" and case["full_count"] != [inputs[group["input"]]["type1_from_rows"]]:
                problems.append(f"{name}: type1 full count != sum C(|e|, 3) of the rows")
            check = case["check"]
            nb = case["batches_requested"]
            if (
                check["exit"] != 0
                or check["failed_lines"]
                or not check["count_lines_identical"]
                or check["integrity_ok"] != nb + 1
                or check["recount_ok"] != nb
            ):
                problems.append(f"{name}: the --check run did not agree")
            shapes[exe] = [(b["deletions"], b["insertions"]) for b in case["batches"]]
        if len({tuple(v) for v in shapes.values()}) != 1:
            problems.append(f"{g.name}: the executables' batch lines differ")
    # The full counts of one input and executable agree across its groups.
    seen: dict[tuple[str, str], list[int]] = {}
    for g in present:
        for exe in EXES:
            doc = json.loads((out / "cases" / g.name / f"{exe}.json").read_text())
            key = (doc["input"], exe)
            if seen.setdefault(key, doc["full_count"]) != doc["full_count"]:
                problems.append(f"{doc['case']}: the full count differs from another group's")
    return problems


def verify_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="triad_goldens.py verify")
    p.add_argument("--scratch", type=Path, default=scratch_default())
    p.add_argument("--set-dir", type=Path)
    args = p.parse_args(argv)
    out = args.set_dir or set_dir(args.scratch)
    if not out.is_dir():
        print(f"{out}: no golden set (parity/triad_goldens.py export --twice)", file=sys.stderr)
        return 77
    problems = verify_set(out, args.scratch)
    for x in problems[:50]:
        print(x, file=sys.stderr)
    spec = golden_spec() or {}
    print(
        f"{SET}: {'OK' if not problems else f'{len(problems)} PROBLEMS'} "
        f"({len(spec.get('cases', {}))} cases, MANIFEST.sha256 {spec.get('manifest_sha256')})"
    )
    return 0 if not problems else 1


# --- crosscheck (PLAN 6.3 step 2) --------------------------------------------------------------

CROSSCHECK_TIMEOUT = 7200.0  # 2 h for coauth main's CPU reference


def crosscheck_cases() -> list[tuple[str, Group, str]]:
    """(name, group, exe): make-test, syn100k s1-s3 at 5000 and 50000, syn1m s1 at 50000 (all
    four executables), coauth type1 b50k_s1, coauth main --batches 1 --batch-size 20."""
    out = [(f"{g.name}/{exe}", g, exe) for g in test_groups() for exe in EXES]
    grid = {g.name: g for g in grid_groups()}
    for name in [f"syn100k_b{s}_s{seed}" for seed in SEEDS for s in ("5k", "50k")] + [
        "syn1m_b50k_s1"
    ]:
        out += [(f"{name}/{exe}", grid[name], exe) for exe in EXES]
    out.append(("coauth_b50k_s1/type1", grid["coauth_b50k_s1"], "type1"))
    g = Group(
        "coauth_b20_s1_batches1",
        "grid",
        ["--input", "{coauth}", "--batches", "1", "--batch-size", "20", "--seed", "1"],
        "coauth",
    )
    out.append(("coauth_b20_s1_batches1/main", g, "main"))
    return out


def crosscheck_one(
    name: str,
    g: Group,
    exe: str,
    unpatched: Path,
    work: Path,
    gpu: str,
    sources: dict,
    golden: Path | None,
) -> dict:
    wdir = work / name.replace("/", "__")
    shutil.rmtree(wdir, ignore_errors=True)
    stage_inputs(wdir, g, sources)
    env = run_env(gpu, {"OMP_NUM_THREADS": "16"})
    argv = g.argv(exe, ["--check", "--reference"])
    timeout = CROSSCHECK_TIMEOUT if name.startswith("coauth_b20") else None
    r = run(["nice", "-n", "10", str(unpatched / "build" / exe), *argv[1:]], wdir, env, timeout)
    err = r["stderr"].splitlines()
    ref_ok = sum(": counts == CPU reference: OK" in x for x in err)
    recount_ok = sum("baseline + deltas == full GPU recount: OK" in x for x in err)
    integrity_ok = sum("data-structure integrity OK" in x for x in err)
    failed = sum("FAILED" in x for x in err)
    batches = g.options.batches
    entry = {
        "case": name,
        "argv": g.argv(exe, ["--check", "--reference"], label=True),
        "exit": r["exit"],
        "timed_out": r["timed_out"],
        "seconds": round(r["seconds"], 1),
        "reference_ok": ref_ok,
        "recount_ok": recount_ok,
        "integrity_ok": integrity_ok,
        "failed_lines": failed,
        "expected_reference_checks": batches + 1,
    }
    entry["agree"] = (
        r["exit"] == 0
        and not failed
        and ref_ok == batches + 1
        and integrity_ok == batches + 1
        and recount_ok == batches
    )
    # The counts this run printed against the goldens (where the set has the case).
    if not r["timed_out"] and r["exit"] == 0:
        parsed = parse_counts(r["stdout"], exe)
        gold = golden / "cases" / name.rsplit("/", 1)[0] / f"{exe}.json" if golden else None
        if gold is not None and gold.is_file():
            doc = json.loads(gold.read_text())
            entry["counts_equal_goldens"] = parsed["full"] == doc["full_count"] and [
                b["delta"] for b in parsed["batches"]
            ] == [b["delta"] for b in doc["batches"]]
        elif golden is not None and name.startswith("coauth_b20"):
            doc = json.loads((golden / "cases" / "coauth_b50k_s1" / "main.json").read_text())
            entry["counts_equal_goldens"] = parsed["full"] == doc["full_count"]
            entry["compared"] = "the full count against coauth_b50k_s1/main's (same input)"
    entry["stderr_tail"] = (
        [x for x in err if not x.startswith("[timing]")][-3:] if not entry["agree"] else []
    )
    shutil.rmtree(wdir, ignore_errors=True)
    return entry


def crosscheck_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="triad_goldens.py crosscheck")
    p.add_argument("--scratch", type=Path, default=scratch_default())
    p.add_argument("--gpu", default="1")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--groups", default="", help="only cases whose name starts with these")
    p.add_argument("--record", type=Path, help="write the JSON record here")
    args = p.parse_args(argv)
    unpatched = args.scratch / "ref" / "ESCHER-GPU@abcf9b2" / "unpatched"
    sources = test_sources(args.scratch)
    sources["coauth"] = args.scratch / COAUTH
    golden = set_dir(args.scratch)
    golden = golden if golden.is_dir() else None
    work = args.scratch / "runs" / "m9a" / "crosscheck-work"
    work.mkdir(parents=True, exist_ok=True)
    prefixes = [x for x in args.groups.split(",") if x]
    cases = [c for c in crosscheck_cases() if not prefixes or c[0].startswith(tuple(prefixes))]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {
            pool.submit(crosscheck_one, n, g, e, unpatched, work, args.gpu, sources, golden): n
            for n, g, e in cases
        }
        for f in concurrent.futures.as_completed(futs):
            e = f.result()
            results.append(e)
            differ = "" if e.get("counts_equal_goldens", True) else " COUNTS DIFFER"
            verdict = "agree" if e["agree"] else "DISAGREE"
            print(f"  {e['case']}: {verdict} ({e['seconds']} s){differ}", flush=True)
    results.sort(key=lambda e: [c[0] for c in cases].index(e["case"]))
    ok = all(e["agree"] and e.get("counts_equal_goldens", True) for e in results)
    record = {
        "what": "ESCHER-GPU@abcf9b2 against its own CPU reference (PLAN 6.3 step 2): the "
        "unpatched executables with --check --reference, OMP_NUM_THREADS=16, nice -n 10, GPU "
        + args.gpu,
        "reference": REFERENCE,
        "commit": COMMIT,
        "binaries": "$DYNG_SCRATCH/ref/ESCHER-GPU@abcf9b2/unpatched/build",
        "date": time.strftime("%Y-%m-%d"),
        "cases": results,
        "all_agree": ok,
        "golden_manifest_sha256": (golden_spec() or {}).get("manifest_sha256"),
    }
    if args.record:
        write_json(args.record, record)
    print(f"crosscheck: {len(results)} runs, {'ALL AGREE' if ok else 'DISAGREEMENT'}")
    return 0 if ok else 1


# --- list / compare ----------------------------------------------------------------------------


def list_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="triad_goldens.py list")
    p.add_argument("--backend", choices=sorted(COVERAGE))
    args = p.parse_args(argv)
    for case in coverage_cases(args.backend) if args.backend else case_names():
        print(case)
    return 0


def compare_main(argv: list[str]) -> int:
    """parity/compare.py triad_count: the set's verification (--verify), the coverage table
    (--list --backend B), and the replays through dynG (parity/triad_replay.py, M9a step 2)."""
    p = argparse.ArgumentParser(prog="compare.py triad_count", add_help=False)
    p.add_argument("--verify", action="store_true", help="verify the golden set")
    p.add_argument("--list", action="store_true", help="the cases a backend replays")
    p.add_argument("--backend", choices=sorted(COVERAGE))
    p.add_argument("--scratch", type=Path, default=scratch_default())
    args, rest = p.parse_known_args(argv)
    if args.list:
        return list_main(["--backend", args.backend] if args.backend else [])
    if args.verify:
        return verify_main(["--scratch", str(args.scratch), *rest])
    import triad_replay

    return triad_replay.compare_main(argv)


def main() -> int:
    commands = {
        "export": export_main,
        "verify": verify_main,
        "crosscheck": crosscheck_main,
        "list": list_main,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(__doc__, file=sys.stderr)
        return 2
    return commands[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
