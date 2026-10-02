#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The cycle_count golden corpus from CycleEnumeration-GPU@0a976ad: export and compare.

    parity/export_goldens.py cycle_count [--twice] [--cases ...] [--no-sequential-check]
    parity/compare.py cycle_count --exe build/parity/tools/compat/dyng-compat-cycle-enum
                                  [--configs sequential,openmp:4,openmp:56] [--full] [--json ...]

    parity/export_goldens.py cycle_count --set cycle_count_cuda [--cases ...]
    parity/compare.py cycle_count --exe ... --configs cuda,cuda:resident,cuda:int64

(or `parity/cycle_count_goldens.py export|compare ...` directly; PLAN Sections 6.3, 6.4.3, 8.3).

Two sets: `cycle_count` (the original's OpenMP backend, M2a) and `cycle_count_cuda` (its CUDA
backend, M2b: `cycle-enum --backend cuda` of the patched copy on the GPU of $DYNG_TEST_GPU, default
1; the cases of the M2b acceptance: DD k = 3..7, GitHub and Twitch k = 3, 4, COLLAB k = 3, and the
seed-1 updates 1K+1K, 25K+25K, 50K+50K, 100K+100K on DD and GitHub, 1K+1K, 25K+25K, 50K+50K on
Twitch and 25K+25K on COLLAB, k = 4). Every CUDA histogram is checked equal to the OpenMP golden of
the same case where the `cycle_count` set has it (the originals cross-checked once, PLAN 6.3 step
2). The COLLAB update (its prior is the k = 4 count of COLLAB, minutes on the CPU) is replayed on
the cuda configurations only.

export  runs the original `cycle-enum` driver of the PATCHED scratch copy that
        parity/build_reference.sh builds (the sources are unchanged; the patch only adds the
        exporter parity_export/bin/export_cycle_enum), on the TUDataset files of
        $DYNG_SCRATCH/datasets/cycle, and writes $DYNG_SCRATCH/goldens/cycle_count:

          count/<graph>_k<k>/histogram.csv    `cycle-enum --backend openmp --openmp-threads 56
                                              --max-cycle-length k` (standard output)
          update/<graph>_k<k>_<d>_<i>_s<seed>[_w<window>]/
              histogram.csv                   `cycle-enum --task update ... --compare-recompute`
                                              (the updated histogram, standard output)
              prior.csv                       the static histogram of the same graph and k
              delta.csv                       histogram.csv - prior.csv per length and in total
              batch.txt                       the generated batch (export_cycle_enum generate:
                                              "- u v" per deletion, then "+ u v")
          <case>/case.json                    the command, the input's SHA-256, the batch sizes
          MANIFEST.sha256                     one SHA-256 per file

        Before a case is written the original is cross-checked against itself: the OpenMP
        histogram equals the original's sequential backend (every case but COLLAB, whose
        sequential count takes tens of minutes; --no-sequential-check skips it), the histogram
        equals the committed cpp/tests/data/cycle_enum/datasets_counts.txt (made by the exporter
        from the original's functions, a second path through the same code), the totals are the
        plan's numbers (PLAN 6.4.3), the update's own --compare-recompute says match=yes, and the
        batch file has the reported number of deletions and insertions. The [sets.cycle_count]
        section of parity/goldens.toml records the digests. --twice exports again from a FRESH
        scratch copy (a new git archive, built from scratch) and requires an identical manifest.

compare verifies the corpus against parity/goldens.toml, then replays every case with
        dyng-compat-cycle-enum (dynG, tools/compat/cycle_enum) in every configuration and compares
        byte for byte: the standard output (the histogram CSV) with histogram.csv, the batch that
        dynG's generators::legacy::cycle_enum_batch() produced (--write-batch) with batch.txt, and
        the reported batch sizes with case.json. The updated histogram and prior.csv (the static
        count of the same graph, compared in count/) fix the delta. Cases marked heavy (COLLAB) are
        not replayed sequentially unless --full. Exit status: 0 all equal, 1 a mismatch or an
        error, 77 the goldens are missing (CTest skips the test).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "parity"))

import compare as sssp_compare  # noqa: E402  (verify_goldens, host_info, build_info, git_head)

SET = "cycle_count"
SET_CUDA = "cycle_count_cuda"
SETS = (SET, SET_CUDA)
REFERENCE = "CycleEnumeration-GPU"
COMMIT = "0a976adfa801a712135bf1adb51a228f353a0751"
BASELINE_COMMIT = "da2067d62c2ce8f9234908089fa15bc53b5979c0"
SKIP = 77
THREADS = 56  # the original's docs/RESULTS.md measures OpenMP with 56 threads
GROUPS = ["count", "update"]
GRAPHS = {
    "DD": "DD/DD_A.txt",
    "github": "github_stargazers/github_stargazers_A.txt",
    "twitch": "twitch_egos/twitch_egos_A.txt",
    "collab": "COLLAB/COLLAB_A.txt",
}
# The totals PLAN 6.4.3 names (and the original's docs/RESULTS.md).
PLAN_TOTALS = {
    ("DD", 3): 2_020_240,
    ("DD", 4): 4_396_674,
    ("DD", 5): 9_476_048,
    ("DD", 6): 21_485_606,
    ("DD", 7): 54_966_172,
    ("github", 3): 7_112_157,
    ("github", 4): 75_872_845,
    ("twitch", 3): 48_891_585,
    ("twitch", 4): 389_362_369,
    ("collab", 3): 1_257_799_573,
}
BATCH_LINE = re.compile(r"^deletions=(\d+) insertions=(\d+)$", re.M)


@dataclass(frozen=True)
class Case:
    graph: str
    k: int
    deletions: int = 0
    insertions: int = 0
    seed: int = 1
    window: int | None = None

    @property
    def update(self) -> bool:
        return self.deletions + self.insertions > 0

    @property
    def group(self) -> str:
        return "update" if self.update else "count"

    @property
    def name(self) -> str:
        if not self.update:
            return f"{self.graph}_k{self.k}"
        w = f"_w{self.window}" if self.window is not None else ""
        return f"{self.graph}_k{self.k}_{self.deletions}_{self.insertions}_s{self.seed}{w}"

    @property
    def rel(self) -> str:
        return f"{self.group}/{self.name}"

    @property
    def heavy(self) -> bool:
        """The sequential count takes tens of minutes (COLLAB)."""
        return self.graph == "collab"

    def cli_args(self, datasets: Path) -> list[str]:
        """The arguments of `cycle-enum` / dyng-compat-cycle-enum (without the backend)."""
        args = ["--input", str(datasets / GRAPHS[self.graph]), "--max-cycle-length", str(self.k)]
        if self.update:
            args += [
                "--task",
                "update",
                "--deletes",
                str(self.deletions),
                "--inserts",
                str(self.insertions),
                "--batch-seed",
                str(self.seed),
            ]
            if self.window is not None:
                args += ["--batch-locality", str(self.window)]
        return args


def all_cases() -> list[Case]:
    cases = [Case("DD", k) for k in range(3, 8)]
    cases += [Case(g, k) for g in ("github", "twitch") for k in (3, 4)]
    cases += [Case("collab", 3)]
    for g in ("DD", "github", "twitch"):
        for size in (1000, 25000, 50000):
            cases.append(Case(g, 4, size, size, 1))
    # A small locality sweep on DD (PLAN 6.4.3): the generator's window (a window of 1000 has too
    # few edges for 25K deletions: the original rejects it), and other bounds.
    cases += [Case("DD", 4, 1000, 1000, 1, 1000)]
    cases += [Case("DD", 4, 25000, 25000, 1, w) for w in (10000, 100000)]
    cases += [Case("DD", k, 25000, 25000, 1) for k in (3, 5)]
    return cases


def cuda_cases() -> list[Case]:
    """The cases of the CUDA set (M2b acceptance criterion 3)."""
    cases = [Case("DD", k) for k in range(3, 8)]
    cases += [Case(g, k) for g in ("github", "twitch") for k in (3, 4)]
    cases += [Case("collab", 3)]
    for g in ("DD", "github"):
        for size in (1000, 25000, 50000, 100000):
            cases.append(Case(g, 4, size, size, 1))
    for size in (1000, 25000, 50000):
        cases.append(Case("twitch", 4, size, size, 1))
    cases += [Case("collab", 4, 25000, 25000, 1)]
    cases += [Case("DD", k, 25000, 25000, 1) for k in (3, 5)]
    return cases


def set_cases(name: str) -> list[Case]:
    return cuda_cases() if name == SET_CUDA else all_cases()


def cuda_only(case: Case) -> bool:
    """Replayed on the cuda configurations only: the COLLAB update's prior is the k = 4 count of
    COLLAB (minutes on the CPU backends)."""
    return case.graph == "collab" and case.update


def select(cases: list[Case], spec: str) -> list[Case]:
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    if not wanted:
        return cases
    out = [c for c in cases if c.rel in wanted or c.group in wanted or c.name in wanted]
    missing = [w for w in wanted if not any(w in (c.rel, c.group, c.name) for c in cases)]
    if missing:
        raise SystemExit(f"unknown case(s) or group(s): {', '.join(missing)}")
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_histogram(text: str) -> dict[int, int]:
    """The original's histogram CSV ("# cycle_size, num_of_cycles", "len, count", "Total, N")."""
    hist: dict[int, int] = {}
    total = None
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(",")
        if key.strip() == "Total":
            total = int(value)
        else:
            hist[int(key)] = int(value)
    if total is None or total != sum(hist.values()):
        raise ValueError(f"not a histogram CSV (total {total}):\n{text[:400]}")
    return hist


def delta_csv(prior: dict[int, int], updated: dict[int, int], k: int) -> str:
    lines = ["# cycle_size, delta"]
    for length in range(2, k + 1):
        lines.append(f"{length}, {updated.get(length, 0) - prior.get(length, 0)}")
    lines.append(f"Total, {sum(updated.values()) - sum(prior.values())}")
    return "\n".join(lines) + "\n"


def committed_counts() -> dict[tuple, dict[int, int]]:
    """cpp/tests/data/cycle_enum/datasets_counts.txt: (file, k[, d, i, seed]) -> histogram."""
    out: dict[tuple, dict[int, int]] = {}
    path = REPO / "cpp" / "tests" / "data" / "cycle_enum" / "datasets_counts.txt"
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if f[0] == "count":
            key: tuple = (f[1], int(f[2]))
            pairs = f[3:]
        else:
            key = (f[1], int(f[2]), int(f[3]), int(f[4]), int(f[5]))
            pairs = f[6:]
        out[key] = {int(a): int(b) for a, b in (p.split(":") for p in pairs)}
    return out


# --- export --------------------------------------------------------------------------------------


class ExportError(RuntimeError):
    pass


def run_checked(cmd: list, env: dict) -> tuple[str, str]:
    proc = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ExportError(f"{' '.join(map(str, cmd))} failed ({proc.returncode}):\n{proc.stderr}")
    return proc.stdout, proc.stderr


def clean_env() -> dict:
    env = dict(os.environ)
    for var in ["OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY"]:
        env.pop(var, None)
    return env


def backend_args(set_name: str) -> list[str]:
    """The original's backend of a set: OpenMP with 56 threads, or CUDA device 0 (of the GPUs made
    visible by CUDA_VISIBLE_DEVICES, see cuda_env)."""
    if set_name == SET_CUDA:
        return ["--backend", "cuda", "--cuda-device", "0"]
    return ["--backend", "openmp", "--openmp-threads", str(THREADS)]


def cuda_env(env: dict) -> dict:
    env = dict(env)
    env.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DYNG_TEST_GPU", "1"))
    env["CUDA_MODULE_LOADING"] = "EAGER"
    return env


def export_case(
    case: Case,
    ref: Path,
    datasets: Path,
    out: Path,
    known: dict,
    set_name: str = SET,
    cpu_goldens: Path | None = None,
) -> dict:
    """Run the original on one case and write its files (case.json is written by export())."""
    exe = ref / "build" / "cycle-enum"
    exporter = ref / "parity_export" / "bin" / "export_cycle_enum"
    env = clean_env() if set_name == SET else cuda_env(clean_env())
    omp = backend_args(set_name)
    args = case.cli_args(datasets)
    started = time.monotonic()
    extra = ["--compare-recompute"] if case.update else []
    stdout, stderr = run_checked([exe, *args, *omp, *extra], env)
    hist = parse_histogram(stdout)
    checks = ["histogram CSV well formed"]
    d = out / case.rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "histogram.csv").write_text(stdout)
    meta: dict = {
        "case": case.rel,
        "reference": REFERENCE,
        "commit": COMMIT,
        "dataset": GRAPHS[case.graph],
        "dataset_sha256": sha256_file(datasets / GRAPHS[case.graph]),
        "max_cycle_length": case.k,
        "task": "update" if case.update else "count",
        "command": "build/cycle-enum "
        + " ".join(
            a.replace(str(datasets), "$DYNG_SCRATCH/datasets/cycle") for a in [*args, *omp, *extra]
        ),
        "heavy": case.heavy,
        "backend": "cuda" if set_name == SET_CUDA else "openmp",
        "cuda_only": set_name == SET_CUDA and cuda_only(case),
    }
    if case.update:
        if "match=yes" not in stderr.splitlines():
            raise ExportError(f"{case.rel}: the original's --compare-recompute: {stderr}")
        checks.append("the original's --compare-recompute: match=yes")
        m = BATCH_LINE.search(stderr)
        if m is None:
            raise ExportError(f"{case.rel}: no deletions=/insertions= line:\n{stderr}")
        meta.update(
            deletions_requested=case.deletions,
            insertions_requested=case.insertions,
            seed=case.seed,
            locality_window=case.window,
            deletions=int(m.group(1)),
            insertions=int(m.group(2)),
        )
        prior_out, _ = run_checked(
            [exe, "--input", datasets / GRAPHS[case.graph], "--max-cycle-length", case.k, *omp],
            env,
        )
        prior = parse_histogram(prior_out)
        (d / "prior.csv").write_text(prior_out)
        (d / "delta.csv").write_text(delta_csv(prior, hist, case.k))
        gen = [exporter, "generate", datasets / GRAPHS[case.graph], case.deletions, case.insertions]
        gen += [case.seed] + ([case.window] if case.window is not None else [])
        batch, _ = run_checked(gen, env)
        (d / "batch.txt").write_text(batch)
        lines = batch.splitlines()
        dels = sum(1 for x in lines if x.startswith("- "))
        ins = sum(1 for x in lines if x.startswith("+ "))
        if (dels, ins) != (meta["deletions"], meta["insertions"]) or dels + ins != len(lines):
            raise ExportError(f"{case.rel}: batch.txt has {dels} - / {ins} + lines, CLI reported")
        checks.append("batch.txt (export_cycle_enum generate) has the CLI's batch sizes")
        key: tuple = (GRAPHS[case.graph], case.k, case.deletions, case.insertions, case.seed)
    else:
        key = (GRAPHS[case.graph], case.k)
    if case.window is None and key in known:
        if known[key] != hist:
            raise ExportError(f"{case.rel}: differs from datasets_counts.txt (the exporter's)")
        checks.append("equal to cpp/tests/data/cycle_enum/datasets_counts.txt (exporter path)")
    if cpu_goldens is not None and (cpu_goldens / case.rel / "histogram.csv").is_file():
        # PLAN 6.3 step 2: the original's CUDA and OpenMP backends agree on the case.
        for name in ["histogram.csv", "prior.csv", "batch.txt"]:
            cpu = cpu_goldens / case.rel / name
            if cpu.is_file() and cpu.read_bytes() != (d / name).read_bytes():
                raise ExportError(
                    f"{case.rel}: {name} of the CUDA backend differs from the OpenMP golden {cpu}"
                )
        checks.append(
            f"histogram.csv, prior.csv and batch.txt equal to the OpenMP golden ({SET} set)"
        )
    want = PLAN_TOTALS.get((case.graph, case.k))
    if not case.update and want is not None:
        if sum(hist.values()) != want:
            raise ExportError(f"{case.rel}: total {sum(hist.values())} != PLAN 6.4.3 {want}")
        checks.append(f"total {want} as PLAN 6.4.3")
    meta["total"] = sum(hist.values())
    meta["cross_checks"] = checks
    print(f"{case.rel}: total {meta['total']} ({time.monotonic() - started:.1f} s)", flush=True)
    return meta


def sequential_check(case: Case, ref: Path, datasets: Path, out: Path) -> str:
    """The original's sequential backend must print the same bytes as its OpenMP backend."""
    started = time.monotonic()
    args = [*case.cli_args(datasets), "--backend", "sequential"]
    seq, _ = run_checked([ref / "build" / "cycle-enum", *args], clean_env())
    if seq != (out / case.rel / "histogram.csv").read_text():
        raise ExportError(f"{case.rel}: the original's sequential and OpenMP backends differ")
    print(f"{case.rel}: sequential equal ({time.monotonic() - started:.1f} s)", flush=True)
    return "the original's sequential backend prints the same bytes"


def case_digest(case_dir: Path) -> tuple[str, int, int]:
    """SHA-256 over the sorted '<sha256>  <relative path>' lines (as export_goldens.py)."""
    lines, size = [], 0
    for p in sorted(case_dir.rglob("*")):
        if p.is_file():
            lines.append(f"{sha256_file(p)}  {p.relative_to(case_dir).as_posix()}\n")
            size += p.stat().st_size
    return hashlib.sha256("".join(lines).encode()).hexdigest(), len(lines), size


def write_manifest(out: Path) -> tuple[str, int]:
    lines = [
        f"{sha256_file(p)}  {p.relative_to(out).as_posix()}\n"
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "MANIFEST.sha256"
    ]
    text = "".join(lines)
    (out / "MANIFEST.sha256").write_text(text)
    return hashlib.sha256(text.encode()).hexdigest(), len(lines)


def toml_section(
    out: Path, metas: list[dict], manifest_sha: str, n_files: int, set_name: str = SET
) -> str:
    rows, total, groups = [], 0, {g: 0 for g in GROUPS}
    for meta in sorted(metas, key=lambda m: m["case"]):
        digest, files, size = case_digest(out / meta["case"])
        total += size
        groups[meta["case"].split("/")[0]] += 1
        rows.append(
            f'"{meta["case"]}" = {{ k = {meta["max_cycle_length"]}, total = {meta["total"]}, '
            f'files = {files}, sha256 = "{digest}" }}\n'
        )
    backend = 'backend = "cuda (cycle-enum --backend cuda)"\n' if set_name == SET_CUDA else ""
    extra = "--set cycle_count_cuda " if set_name == SET_CUDA else ""
    head = f"""[sets.{set_name}]
reference = "{REFERENCE}"
commit = "{COMMIT}"
baseline_commit = "{BASELINE_COMMIT}"
{backend}location = "$DYNG_SCRATCH/goldens/{set_name}"
generated_by = "parity/export_goldens.py cycle_count {extra}(parity/cycle_count_goldens.py)"
num_cases = {len(metas)}
num_files = {n_files}
num_bytes = {total}
manifest_sha256 = "{manifest_sha}"

[sets.{set_name}.groups]
""" + "".join(f"{g} = {n}\n" for g, n in groups.items() if n)
    return head + f"\n[sets.{set_name}.cases]\n" + "".join(rows)


def write_toml(
    section: str, set_name: str = SET, path: Path = REPO / "parity" / "goldens.toml"
) -> None:
    """Replace the [sets.<set_name>] section (and its subtables) of parity/goldens.toml; the
    cycle_count sets are kept after the sssp set, in the order of SETS, and the later sets (e.g.
    mosp_scale, parity/mosp_scale_goldens.py) after them in their order."""
    text = path.read_text()
    first = re.search(r"^\[sets\.(?!sssp[\].])[a-z_]+[\].]", text, re.M)
    head, tail = (text[: first.start()], text[first.start() :]) if first else (text, "")
    blocks: dict[str, list[str]] = {}
    current = None
    for line in tail.splitlines(keepends=True):
        m = re.match(r"^\[sets\.([a-z_]+)[\].]", line)
        if m:
            current = m.group(1)
        if current is not None:
            blocks.setdefault(current, []).append(line)
    blocks[set_name] = [section]
    order = [n for n in SETS if n in blocks] + [n for n in blocks if n not in SETS]
    body = "\n".join("".join(blocks[n]).strip("\n") + "\n" for n in order)
    path.write_text(head.rstrip("\n") + "\n\n" + body)


def reference_dir(scratch: Path, fresh: bool) -> Path:
    tool = REPO / "parity" / "build_reference.sh"
    base = [tool, "--variant", "patched", "--scratch", scratch]
    subprocess.run(
        [str(c) for c in [*base, *(["--fresh"] if fresh else []), REFERENCE]], check=True
    )
    out = subprocess.check_output([str(c) for c in [*base, "--print-dir", REFERENCE]], text=True)
    return Path(out.strip().splitlines()[-1])


def export(
    ref: Path,
    datasets: Path,
    out: Path,
    cases: list[Case],
    sequential: bool,
    jobs: int,
    set_name: str = SET,
    cpu_goldens: Path | None = None,
) -> tuple[list[dict], str, int]:
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    known = committed_counts()
    print(f"exporting {len(cases)} {set_name} cases from {ref} into {out}", flush=True)
    # The OpenMP runs use every core, the CUDA ones the GPU: one case at a time.
    metas = [export_case(c, ref, datasets, out, known, set_name, cpu_goldens) for c in cases]
    if sequential and set_name == SET:
        # The sequential cross-checks use one core each: --jobs of them at a time.
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(sequential_check, c, ref, datasets, out): i
                for i, c in enumerate(cases)
                if not c.heavy
            }
            for fut in concurrent.futures.as_completed(futures):
                metas[futures[fut]]["cross_checks"].append(fut.result())
    for meta in metas:
        path = out / meta["case"] / "case.json"
        path.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    manifest_sha, n_files = write_manifest(out)
    return metas, manifest_sha, n_files


def export_main(argv: list[str]) -> int:
    scratch = Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))
    p = argparse.ArgumentParser(
        prog="export_goldens.py cycle_count",
        description="Export the cycle_count golden corpus from CycleEnumeration-GPU@0a976ad.",
    )
    p.add_argument("--scratch", type=Path, default=scratch)
    p.add_argument("--datasets", type=Path, default=scratch / "datasets" / "cycle")
    p.add_argument("--set", choices=SETS, default=SET, help="the golden set (default cycle_count)")
    p.add_argument("--out", type=Path, help="default: <scratch>/goldens/<set>")
    p.add_argument("--cases", default="", help="restrict to these cases or groups (comma list)")
    p.add_argument("--jobs", type=int, default=8, help="parallel sequential cross-checks")
    p.add_argument(
        "--no-sequential-check",
        action="store_true",
        help="skip the cross-check against the original's sequential backend",
    )
    p.add_argument("--no-toml", action="store_true", help="do not write parity/goldens.toml")
    p.add_argument(
        "--twice",
        action="store_true",
        help="export again from a fresh archive copy and require an identical manifest",
    )
    args = p.parse_args(argv)
    cases = select(set_cases(args.set), args.cases)
    out = args.out or args.scratch / "goldens" / args.set
    cpu_goldens = args.scratch / "goldens" / SET if args.set == SET_CUDA else None
    ref = reference_dir(args.scratch, fresh=False)
    try:
        metas, sha, n_files = export(
            ref,
            args.datasets,
            out,
            cases,
            not args.no_sequential_check,
            args.jobs,
            args.set,
            cpu_goldens,
        )
    except ExportError as e:
        print(f"export_goldens cycle_count: {e}", file=sys.stderr)
        return 1
    print(f"{len(metas)} cases, {n_files} files, MANIFEST.sha256 {sha}")
    if not args.no_toml and len(cases) == len(set_cases(args.set)):
        write_toml(toml_section(out, metas, sha, n_files, args.set), args.set)
        print(f"wrote the [sets.{args.set}] section of {REPO / 'parity' / 'goldens.toml'}")
    if args.twice:
        with tempfile.TemporaryDirectory(prefix="dyng-goldens-twice-", dir=args.scratch) as t:
            second = Path(t)
            ref2 = reference_dir(second, fresh=True)
            try:
                _, sha2, _ = export(
                    ref2,
                    args.datasets,
                    second / "goldens" / args.set,
                    cases,
                    not args.no_sequential_check,
                    args.jobs,
                    args.set,
                    cpu_goldens,
                )
            except ExportError as e:
                print(f"export_goldens cycle_count (second export): {e}", file=sys.stderr)
                return 1
            if sha2 != sha:
                subprocess.run(
                    [
                        "diff",
                        out / "MANIFEST.sha256",
                        second / "goldens" / args.set / "MANIFEST.sha256",
                    ]
                )
                print("SECOND EXPORT DIFFERS", file=sys.stderr)
                return 1
            print(f"second export from a fresh archive copy: identical manifest ({sha2})")
    return 0


# --- compare -------------------------------------------------------------------------------------


def replay(exe: Path, golden: Path, meta: dict, config: str, datasets: Path, tmp: Path) -> list:
    """Replay one case with dyng-compat-cycle-enum; the list of problems (empty: equal)."""
    backend, _, option = config.partition(":")
    args = ["--input", datasets / meta["dataset"], "--max-cycle-length", meta["max_cycle_length"]]
    args += ["--backend", backend]
    if backend == "openmp":
        args += ["--openmp-threads", option or str(THREADS)]
    elif backend == "cuda" and option == "resident":
        args += ["--scope", "resident"]
    elif backend == "cuda" and option == "int64":
        args += ["--edge-type", "int64"]
    bad = []
    tmp.mkdir(parents=True, exist_ok=True)
    if meta["task"] == "update":
        args += [
            "--task",
            "update",
            "--deletes",
            meta["deletions_requested"],
            "--inserts",
            meta["insertions_requested"],
            "--batch-seed",
            meta["seed"],
        ]
        if meta.get("locality_window") is not None:
            args += ["--batch-locality", meta["locality_window"]]
        args += ["--write-batch", tmp / "batch.txt"]
    env = cuda_env(dict(os.environ)) if backend == "cuda" else None
    proc = subprocess.run([str(a) for a in [exe, *args]], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        return [f"exit status {proc.returncode}: {proc.stderr[-400:]}"]
    if proc.stdout != (golden / "histogram.csv").read_text():
        bad.append("histogram.csv differs")
    if meta["task"] == "update":
        m = BATCH_LINE.search(proc.stderr)
        got = (int(m.group(1)), int(m.group(2))) if m else None
        if got != (meta["deletions"], meta["insertions"]):
            bad.append(f"batch sizes {got} != {(meta['deletions'], meta['insertions'])}")
        batch = tmp / "batch.txt"
        if not batch.is_file() or batch.read_bytes() != (golden / "batch.txt").read_bytes():
            bad.append("batch.txt differs")
    return bad


def compare_main(argv: list[str]) -> int:
    scratch = Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))
    p = argparse.ArgumentParser(
        prog="compare.py cycle_count",
        description="Replay the cycle_count golden corpus through dyng-compat-cycle-enum.",
    )
    p.add_argument(
        "--set",
        choices=SETS,
        help="the golden set (default: cycle_count_cuda when every configuration is cuda, "
        "else cycle_count)",
    )
    p.add_argument("--goldens", type=Path, help="default: $DYNG_SCRATCH/goldens/<set>")
    p.add_argument("--datasets", type=Path, default=scratch / "datasets" / "cycle")
    p.add_argument("--exe", type=Path, help="dyng-compat-cycle-enum")
    p.add_argument(
        "--configs",
        default=f"sequential,openmp:4,openmp:{THREADS}",
        help="list of sequential, openmp[:threads], cuda[:resident|:int64]",
    )
    p.add_argument("--cases", default="", help="restrict to these cases or groups (comma list)")
    p.add_argument("--full", action="store_true", help="also replay heavy cases sequentially")
    p.add_argument("--jobs", type=int, default=8, help="parallel sequential replays")
    p.add_argument("--json", type=Path, help="write the result matrix here")
    p.add_argument("--label", default="", help="free text stored in the JSON")
    p.add_argument("--skip-verify", action="store_true", help="do not re-hash the goldens")
    args = p.parse_args(argv)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    set_name = args.set or (
        SET_CUDA if configs and all(c.startswith("cuda") for c in configs) else SET
    )
    if args.goldens is None:
        args.goldens = scratch / "goldens" / set_name

    toml = tomllib.loads((REPO / "parity" / "goldens.toml").read_text())
    golden_set = toml.get("sets", {}).get(set_name)
    if golden_set is None:
        print(f"parity/goldens.toml has no [sets.{set_name}] (skipping)", file=sys.stderr)
        return SKIP
    if not (args.goldens / "MANIFEST.sha256").is_file():
        print(
            f"goldens not found under {args.goldens}; run parity/export_goldens.py cycle_count "
            "(skipping)",
            file=sys.stderr,
        )
        return SKIP
    if args.exe is None or not args.exe.is_file():
        p.error("--exe <dyng-compat-cycle-enum> is required")
    for c in configs:
        backend, _, option = c.partition(":")
        ok = (
            (backend == "sequential" and not option)
            or (backend == "openmp" and (not option or option.isdigit()))
            or (backend == "cuda" and option in ("", "resident", "int64"))
        )
        if not ok:
            p.error(
                f"--configs: '{c}' is not sequential, openmp[:<threads>] or cuda[:resident|:int64]"
            )
    if not configs:
        p.error("--configs selects no configuration")
    if not args.skip_verify:
        sssp_compare.verify_goldens(args.goldens, golden_set)
        print(
            f"goldens verified: {golden_set['num_cases']} cases, {golden_set['num_files']} files, "
            f"manifest {golden_set['manifest_sha256'][:16]}"
        )
    wanted = [s.strip() for s in args.cases.split(",") if s.strip()]
    cases = sorted(
        c
        for c in golden_set["cases"]
        if not wanted or c in wanted or c.split("/")[0] in wanted or c.split("/")[1] in wanted
    )
    if not cases:
        print("compare.py cycle_count: no case selected", file=sys.stderr)
        return 1
    metas = {c: json.loads((args.goldens / c / "case.json").read_text()) for c in cases}
    work = Path(tempfile.mkdtemp(prefix="dyng-compare-cc-", dir=scratch / "runs"))
    exe = args.exe.resolve()
    results: dict[tuple[str, str], list | None] = {}

    def job(case: str, config: str) -> tuple[str, str, list, float]:
        t0 = time.monotonic()
        tmp = work / config.replace(":", "_") / case
        try:
            bad = replay(exe, args.goldens / case, metas[case], config, args.datasets, tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return case, config, bad, time.monotonic() - t0

    def record(case: str, config: str, bad: list, seconds: float) -> None:
        results[(case, config)] = bad
        verdict = "equal" if not bad else "MISMATCH " + "; ".join(bad[:3])
        print(f"{case} [{config}]: {verdict} ({seconds:.1f} s)", flush=True)

    try:
        todo = []
        for c in cases:
            for cfg in configs:
                if cfg == "sequential" and metas[c]["heavy"] and not args.full:
                    results[(c, cfg)] = None  # skipped
                elif metas[c].get("cuda_only") and not cfg.startswith("cuda"):
                    results[(c, cfg)] = None  # its prior takes minutes on the CPU
                else:
                    todo.append((c, cfg))
        # Sequential replays in parallel; the OpenMP ones one at a time (they use the cores).
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(job, c, cfg) for c, cfg in todo if cfg == "sequential"]
            for fut in concurrent.futures.as_completed(futures):
                record(*fut.result())
        for c, cfg in todo:
            if cfg != "sequential":
                record(*job(c, cfg))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    failures = [(c, cfg, bad) for (c, cfg), bad in sorted(results.items()) if bad]
    print("\n| case | total | " + " | ".join(configs) + " |")
    print("|---|---:|" + "---|" * len(configs))
    for c in cases:
        cells = []
        for cfg in configs:
            bad = results[(c, cfg)]
            cells.append("skipped" if bad is None else ("equal" if not bad else "MISMATCH"))
        print(f"| {c} | {metas[c]['total']:,} | " + " | ".join(cells) + " |")
    compared = sum(1 for v in results.values() if v is not None)
    if args.json:
        doc = {
            "schema": 1,
            "algorithm": SET,
            "set": set_name,
            "label": args.label,
            "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "reference": {
                "name": REFERENCE,
                "commit": COMMIT,
                "baseline_commit": BASELINE_COMMIT,
            },
            "port": {"repository": "dyng", "commit": sssp_compare.git_head()},
            "executable": sssp_compare.portable_path(args.exe),
            "build": sssp_compare.build_info(args.exe),
            "host": sssp_compare.host_info(),
            "goldens": {
                "manifest_sha256": golden_set["manifest_sha256"],
                "cases": golden_set["num_cases"],
                "files": golden_set["num_files"],
            },
            "tolerance": "none: byte equality of the histogram CSV and the generated batch",
            "configs": configs,
            "matrix": {
                c: {
                    cfg: (
                        "skipped"
                        if results[(c, cfg)] is None
                        else ("equal" if not results[(c, cfg)] else results[(c, cfg)])
                    )
                    for cfg in configs
                }
                for c in cases
            },
            "passed": not failures,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
    for c, cfg, bad in failures:
        print(f"MISMATCH {c} [{cfg}]: {'; '.join(bad[:4])}")
    print(
        f"\n{'ALL EQUAL' if not failures else f'{len(failures)} MISMATCHES'}: {compared} replays "
        f"({len(cases)} cases x {len(configs)} configurations, "
        f"{len(results) - compared} skipped)"
    )
    return 0 if not failures else 1


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("export", "compare"):
        print("usage: parity/cycle_count_goldens.py export|compare [options]\n")
        print(__doc__)
        return 0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2
    if sys.argv[1] == "export":
        return export_main(sys.argv[2:])
    return compare_main(sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
