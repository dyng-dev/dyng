#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The golden sets of label_propagation from LabelPropagation-CUDA@a276a3a (O), its in_place
calibration and its own --check margins (M10a step 1; PLAN 6.3 step 1, 6.4.7, 8.3; ADRs 0013,
0036, 0065; the author's decisions AD5, AD7 and AD10 of 2026-10-28).

    label_propagation_goldens.py export --twice [--sets ...]   (export_goldens.py label_propagation)
    label_propagation_goldens.py export --copy-dir DIR --gpu G --out OUT [--sets ...]
    label_propagation_goldens.py compare-exports OUT1 OUT2 [--sets ...]
    label_propagation_goldens.py install OUT [--sets ...]
    label_propagation_goldens.py verify [--sets ...]   (= compare.py label_propagation --verify)
    label_propagation_goldens.py calibrate --gpu G --runs N --tag T [--lock-clocks none|base]
                                           [--cases 5k5,50k,500k,5m] [--delta 0.0001]
    label_propagation_goldens.py calibrate-analyze [--json FILE]
    label_propagation_goldens.py margins --cases 5k5,50k|500k,5m --delta D --gpu 1 --json FILE
                                         [--time-box SECONDS]

The sets ($DYNG_SCRATCH/goldens/<set>; parity/goldens.toml lists each with its MANIFEST.sha256
digest and per-case digests; no release asset in 0.4, AD10):

  label_propagation_generator  G: the generator-only twin runs (DYNG_DYNLP_GENERATOR_ONLY=1).
      bytes cases (5k5, 50k, 500k at seed 42; 5k5 and 50k at seeds 1-3; 50k at --avgDegree 3
      and 7): graph/csr{RowPtr,ColInd,Values,TrueLabels}.txt.zst (O's text files, also checked
      against the unpatched O's own files of the same argv) and generator.dlt.zst (the CSR with
      its value bits, the true-label bits, every batch); digest cases (5m, 5m1, 50m, with
      --noGraphFiles): generator.dlt.zst in the recorder's digest mode (the SHA-256 of every
      array; those files are never written).
  label_propagation_damped     P+: per case the unpatched O's dynlp_labels.txt.zst, results.json
      (the deterministic results.csv columns: iterations, frontier_visits, components, accuracy;
      and tau) and the twin's per-batch label bits (label_bits.bin.zst; 5m: label_bits.json, the
      per-batch SHA-256).
  label_propagation_trace      P+: per case the twin's canonical stage trace trace.dlt.zst (every
      record of parity/exporters/dynlp/recorder_dynlp.cpp with the timings left out; 5m in the
      digest mode) and info.json (the per-batch stages and counts).
  label_propagation_hub        propagation_test's hub graph: the twin's canonical record of the
      three calls (the inputs of the in-place call, the damped calls in full), info.json; the
      in-place call's outputs of HUB_INPLACE_RUNS runs under inplace/ are a distribution, outside
      the manifest (not reproducible).
  label_propagation_irlp       5k5 and 50k damped with --validate: the IrLP label bits and
      iterations per snapshot (irlp_bits.bin.zst, irlp.json) and O's irlp_labels.txt.zst.
  label_propagation_inplace    item 6 of M10a step 1 (AD5): N = 10 unpatched --update inplace runs
      per I case on GPU 1 at default clocks and 3 on GPU 0 at the base lock (labels and
      results.csv per run), the N = 3 runs at delta 1e-6 on 5k5 and 50k (tight/), and the
      calibration record (calibration.json) whose thresholds parity/goldens.toml freezes. A
      distribution: exported once, exempt from the double export (PLAN 8.3).
  label_propagation_check      O's --check runs (C and L; margins): the logs, outside the manifest.

Hashing (M9a finding 4): MANIFEST.sha256 lists '<sha256>  <path>' of every hashed file, the
SHA-256 of its content (a .zst file decompressed, so the zstd version does not enter it);
PROVENANCE.json (the GPU, the copy, the run times, the zstd version) is hashed by nothing. A case
digest is the SHA-256 of the sorted lines of the case's files. Run-varying values (time_ms,
eval_ms, irlp time) are never hashed.

`export --twice` exports from two FRESH patched copies (parity/build_reference.sh --variant
patched --fresh --scratch $DYNG_SCRATCH/ref/m10a-export-<1|2>): the first on GPU 1 under the shared
lock (after `locklog.py quiet-ok M10a`), the second on GPU 0 inside `locklog.py run --reset-gpu 0
EXCL` (no timing), then requires identical MANIFEST.sha256 files, installs the first and writes
the sets' sections of parity/goldens.toml. It refuses to start an export with less than 20 GB
free. Every comparison of an export can fail: the twin's labels and results.csv columns against
the unpatched O of the same copy, the twin's final label bits printed in O's format against O's
file, the twin's generator text files against the unpatched O's, the IrLP iterations and the
last snapshot's labels against O's results.csv and irlp_labels.txt, the hub's damped calls
across runs.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "parity"))

import dynlp_trace as dlt  # noqa: E402

REFERENCE = "LabelPropagation-CUDA"
COMMIT = "a276a3a4e00db1f4b6d24d2bc7ba4d9bc71b6d87"
SETS = {
    "generator": "label_propagation_generator",
    "damped": "label_propagation_damped",
    "trace": "label_propagation_trace",
    "hub": "label_propagation_hub",
    "irlp": "label_propagation_irlp",
    "inplace": "label_propagation_inplace",
}
EXPORTED_TWICE = ("generator", "damped", "trace", "hub", "irlp")
CHECK_SET = "label_propagation_check"
ZSTD = ["zstd", "-q", "-12", "--long=27", "-T1"]
MIN_FREE_BYTES = 20 * 10**9
HUB_INPLACE_RUNS = 5
# The run-varying fields of a recording, left out of the canonical traces.
DROP = {
    ("run_end", "time_ms"),
    ("run_end", "eval_ms"),
    ("run_end", "irlp_time_ms"),
    ("irlp_snapshot", "time_ms"),
}
# The in-place call of propagation_test (call 0) is not reproducible: its outputs are left out of
# the hub's canonical record and kept as a distribution.
HUB_INPLACE_OUTPUTS = (
    "iterations",
    "frontier_visits",
    "persistent_iterations",
    "next_tag",
    "labels",
)
CSR_FILES = ("csrRowPtr.txt", "csrColInd.txt", "csrValues.txt", "csrTrueLabels.txt")
SIZES = {
    "5k5": (5000, 5),
    "50k": (50000, 10),
    "500k": (500000, 10),
    "5m": (5000000, 10),
    "5m1": (5000000, 2),
    "50m": (50000000, 11),
}
DELTAS = {"1e-4": "0.0001", "1e-6": "0.000001"}
# AD5 (decided by the author on 2026-10-28): k = 3, floors 1e-6, N = 10 on GPU 1 + 3 on GPU 0.
CALIBRATION = {
    "k": 3,
    "floor_mean": 1e-6,
    "floor_max": 1e-6,
    "n_gpu1": 10,
    "n_gpu0": 3,
    "iterations_rel": 0.10,
    "iterations_floor": 2,
    "cases": ["5k5", "50k", "500k", "5m"],
    "tight_cases": ["5k5", "50k"],
    "tight_runs": 3,
}
VERTEX_ACTIVE, VERTEX_DELETED, VERTEX_GROUND_TRUTH, VERTEX_INACTIVE = 0, 1, 2, 3


class GoldenError(Exception):
    pass


@dataclass(frozen=True)
class Case:
    size: str
    avg_degree: int = 5
    delta: str = "1e-4"
    seed: int = 42
    schedule: str = "damped"
    digest: bool = False
    validate: bool = False

    @property
    def n(self) -> int:
        return SIZES[self.size][0]

    @property
    def batches(self) -> int:
        return SIZES[self.size][1]

    @property
    def name(self) -> str:
        deg = "" if self.avg_degree == 5 else f"_deg{self.avg_degree}"
        rule = "" if self.schedule == "damped" else f"_{self.schedule}"
        return f"{self.size}{deg}_d{self.delta}_s{self.seed}{rule}"

    def argv(self, out: Path, graph_files: bool = False, quiet: bool = True) -> list[str]:
        a = [
            "--totalNodes",
            str(self.n),
            "--numBatches",
            str(self.batches),
            "--avgDegree",
            str(self.avg_degree),
            "--delta",
            DELTAS[self.delta],
            "--seed",
            str(self.seed),
            "--update",
            self.schedule,
        ]
        a += ["--quiet"] if quiet else []
        a += ["--validate" if self.validate else "--noValidate"]
        a += [] if graph_files else ["--noGraphFiles"]
        return a + ["--output", str(out)]


GEN_CASES = [
    Case("5k5"),
    Case("50k"),
    Case("500k"),
    Case("5m", digest=True),
    Case("5m1", digest=True),
    Case("50m", digest=True),
    *[Case(s, seed=k) for s in ("5k5", "50k") for k in (1, 2, 3)],
    Case("50k", avg_degree=3),
    Case("50k", avg_degree=7),
]
P_CASES = [
    Case("5k5"),
    Case("50k"),
    Case("500k"),
    Case("5k5", delta="1e-6"),
    Case("50k", delta="1e-6"),
    *[Case(s, seed=k) for s in ("5k5", "50k") for k in (1, 2)],
    Case("5m", digest=True),
]
IRLP_CASES = [Case("5k5", validate=True), Case("50k", validate=True)]


# --- Helpers -------------------------------------------------------------------------------------


def scratch() -> Path:
    return Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))


def set_dir(root: Path, short: str) -> Path:
    return root / SETS[short]


def now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_sha256(path: Path) -> str:
    """The SHA-256 a set hashes a file by: of its content (a .zst file decompressed)."""
    if path.suffix != ".zst":
        return sha256_file(path)
    h = hashlib.sha256()
    with subprocess.Popen(
        ["zstd", "-q", "-dc", "--long=27", str(path)], stdout=subprocess.PIPE
    ) as z:
        assert z.stdout is not None
        for chunk in iter(lambda: z.stdout.read(1 << 20), b""):
            h.update(chunk)
    if z.returncode != 0:
        raise GoldenError(f"{path}: zstd -d failed")
    return h.hexdigest()


def zstd_write(data: bytes, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([*ZSTD, "-f", "-o", str(dst)], input=data, check=True)


def zstd_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([*ZSTD, "-f", str(src), "-o", str(dst)], check=True)


def zstd_read(path: Path) -> bytes:
    return subprocess.run(
        ["zstd", "-q", "-dc", "--long=27", str(path)], check=True, capture_output=True
    ).stdout


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def zstd_version() -> str:
    return subprocess.run(["zstd", "--version"], capture_output=True, text=True).stdout.strip()


def label_text(labels) -> bytes:
    """O's dynlp_labels.txt / irlp_labels.txt of label floats: 'i label' with the label as
    std::to_chars(double, fixed, 6) (utils.cu writeLabelsToFile); Python's '%.6f' of the double
    is correctly rounded, as to_chars is."""
    return "".join(f"{i} {float(x):.6f}\n" for i, x in enumerate(labels)).encode()


def csr_text(trace: dlt.Trace) -> dict[str, bytes]:
    """O's four graph files (generateSparseGraph.cu writeGraphFiles) from a raw generator
    record: RowPtr and ColInd as integers, Values as '%.6f', TrueLabels 'i 0|1'."""
    g = trace.one("generator")
    rp, ci = g["row_ptr"].values(), g["col_ind"].values()
    vals, labels = g["values"].values(), g["true_labels"].values()
    return {
        "csrRowPtr.txt": "".join(f"{x}\n" for x in rp).encode(),
        "csrColInd.txt": "".join(f"{x}\n" for x in ci).encode(),
        "csrValues.txt": "".join(f"{float(x):.6f}\n" for x in vals).encode(),
        "csrTrueLabels.txt": "".join(
            f"{i} {1 if x >= 0.5 else 0}\n" for i, x in enumerate(labels)
        ).encode(),
    }


def results_columns(text: str) -> list[dict]:
    import csv
    import io

    rows = list(csv.DictReader(io.StringIO(text)))
    keep = (
        "iterations",
        "frontier_visits",
        "components",
        "accuracy",
        "irlp_iterations",
        "irlp_accuracy",
        "accuracy_vs_irlp",
    )
    return [{k: r[k] for k in keep if k in r} for r in rows]


def clean_env(gpu: int) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMP_", "DYNG_DYNLP_"))}
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_MODULE_LOADING="EAGER")
    return env


def run(cmd: list, env: dict, timeout: float | None = None) -> subprocess.CompletedProcess:
    p = subprocess.run(
        [str(c) for c in cmd], env=env, capture_output=True, text=True, timeout=timeout
    )
    if p.returncode != 0:
        raise GoldenError(
            f"{' '.join(map(str, cmd))} failed ({p.returncode}):\n{(p.stdout + p.stderr)[-2000:]}"
        )
    return p


def gpu_record(gpu: int) -> dict:
    out = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    ).stdout
    name, uuid, driver = (x.strip() for x in (out.strip().split(",") + ["", "", ""])[:3])
    return {"index": gpu, "name": name, "uuid": uuid, "driver": driver}


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def require_free(path: Path) -> None:
    free = free_bytes(path)
    if free < MIN_FREE_BYTES:
        raise GoldenError(f"STOP: {free / 1e9:.1f} GB free below 20 GB: the export is not started")


def origin_status() -> str:
    origin = Path(os.environ.get("DYNG_ORIGINALS_DIR", Path.home() / "Projects")) / REFERENCE
    return (
        subprocess.run(
            ["git", "-C", str(origin), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True,
            text=True,
        ).stdout
        + subprocess.run(
            ["git", "-C", str(origin), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout
    )


# --- Status of the vertices after the batches (O's host model, dynLP.cu:453-473) ---------------


def final_status(n: int, batches: list[dlt.Record]):
    """The vertex status after the last batch, as runDynLP's host pass sets it (the host and the
    device status agree, which verify checks against the device status of the raw traces):
    every batch marks Del, makes Ins \\ Del ACTIVE, the ground truth GROUND_TRUTH and Del
    DELETED, in this order."""
    import numpy as np

    status = np.full(n, VERTEX_INACTIVE, dtype=np.int32)
    for b in sorted(batches, key=lambda r: r.batch):
        dele = b["deleted"].numpy()
        ins = b["inserted"].numpy()
        gt = b["gt_ids"].numpy()
        mark = np.zeros(n, dtype=bool)
        mark[dele] = True
        status[ins[~mark[ins]]] = VERTEX_ACTIVE
        status[gt] = VERTEX_GROUND_TRUTH
        status[dele] = VERTEX_DELETED
    return status


# --- One export ----------------------------------------------------------------------------------


class Exporter:
    def __init__(self, copy: Path, gpu: int, out: Path, work: Path) -> None:
        self.copy, self.gpu, self.out, self.work = copy, gpu, out, work
        self.orig = copy / "parity_export" / "orig" / "dynLP"
        self.twin = copy / "bin" / "dynLP"
        self.hub_orig = copy / "parity_export" / "orig" / "propagation_test"
        self.hub_twin = copy / "bin" / "propagation_test"
        marker = b"dyng-dynlp-recorder-twin format 1"
        for exe, twin in [
            (self.orig, False),
            (self.twin, True),
            (self.hub_orig, False),
            (self.hub_twin, True),
        ]:
            if not exe.is_file() or (marker in exe.read_bytes()) != twin:
                raise GoldenError(f"{exe}: expected the {'twin' if twin else 'original'}")
        self.env = clean_env(gpu)
        self.timings: list[dict] = []

    def tmp(self, name: str) -> Path:
        d = self.work / name
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        return d

    def timed(self, what: str, cmd: list, env: dict) -> subprocess.CompletedProcess:
        t0 = datetime.datetime.now(datetime.UTC)
        p = run(cmd, env)
        self.timings.append(
            {"run": what, "seconds": (datetime.datetime.now(datetime.UTC) - t0).total_seconds()}
        )
        return p

    def trace_env(self, d: Path, digest: bool, generator: bool = False) -> dict:
        e = dict(self.env, DYNG_DYNLP_TRACE_DIR=str(d))
        if digest:
            e["DYNG_DYNLP_TRACE_DIGEST"] = "1"
        if generator:
            e["DYNG_DYNLP_GENERATOR_ONLY"] = "1"
        return e

    # G
    def generator(self, case: Case) -> None:
        d = set_dir(self.out, "generator") / "cases" / case.name
        w = self.tmp(f"gen-{case.name}")
        self.timed(
            f"generator twin {case.name}",
            [self.twin, *case.argv(w / "out", graph_files=not case.digest)],
            self.trace_env(w / "trace", case.digest, generator=True),
        )
        t = dlt.read(w / "trace" / "dynLP.dlt")
        if t.digest_mode != case.digest or len(t.records("batch_data")) != case.batches:
            raise GoldenError(f"{case.name}: the generator trace is not the expected one")
        canonical = dlt.encode(t.all, t.digest_mode)
        zstd_write(canonical, d / "generator.dlt.zst")
        info = {
            "case": case.name,
            "argv": case.argv(Path("<out>"), not case.digest),
            "mode": "digest" if case.digest else "bytes",
            "num_nodes": t.one("generator").scalar("num_nodes"),
            "num_edges": t.one("generator").scalar("num_edges"),
            "generator_dlt_sha256": sha256_bytes(canonical),
            "arrays_sha256": {
                f"{r.tag}[{r.batch}].{k}": f.sha256
                for r in t.all
                for k, f in r.fields.items()
                if f.count > 1
            },
        }
        if not case.digest:
            text = csr_text(t)
            o = self.tmp(f"gen-orig-{case.name}")
            self.timed(
                f"generator orig {case.name}",
                [self.orig, *case.argv(o / "out", graph_files=True)],
                self.env,
            )
            for f in CSR_FILES:
                twin_bytes = (w / "out" / "graph" / f).read_bytes()
                if twin_bytes != (o / "out" / "graph" / f).read_bytes():
                    raise GoldenError(f"{case.name}: {f}: the twin's differs from the original's")
                if twin_bytes != text[f]:
                    raise GoldenError(f"{case.name}: {f}: the text is not the recorded arrays")
                zstd_file(w / "out" / "graph" / f, d / "graph" / f"{f}.zst")
            info["text_sha256"] = {f: sha256_bytes(text[f]) for f in CSR_FILES}
            shutil.rmtree(o, ignore_errors=True)
        write_json(d / "info.json", info)
        shutil.rmtree(w, ignore_errors=True)

    # P+ (damped and trace)
    def damped(self, case: Case) -> dict:
        dd = set_dir(self.out, "damped") / "cases" / case.name
        dt = set_dir(self.out, "trace") / "cases" / case.name
        o = self.tmp(f"orig-{case.name}")
        w = self.tmp(f"twin-{case.name}")
        po = self.timed(f"orig {case.name}", [self.orig, *case.argv(o / "out")], self.env)
        self.timed(
            f"twin {case.name}",
            [self.twin, *case.argv(w / "out")],
            self.trace_env(w / "trace", case.digest),
        )
        labels = (o / "out" / "dynlp_labels.txt").read_bytes()
        if (w / "out" / "dynlp_labels.txt").read_bytes() != labels:
            raise GoldenError(f"{case.name}: the twin's dynlp_labels.txt differs from O's")
        cols = results_columns((o / "out" / "results.csv").read_text())
        if results_columns((w / "out" / "results.csv").read_text()) != cols:
            raise GoldenError(f"{case.name}: the twin's results.csv columns differ from O's")
        t = dlt.read(w / "trace" / "dynLP.dlt")
        batches = t.batches()
        if sorted(batches) != list(range(case.batches)):
            raise GoldenError(f"{case.name}: batches {sorted(batches)} in the trace")
        per = []
        for b in range(case.batches):
            st = batches[b]
            end = st["propagation_end"]
            if end.scalar("iterations") != int(cols[b]["iterations"]):
                raise GoldenError(f"{case.name}: batch {b}: iterations differ from results.csv")
            if end.scalar("frontier_visits") != int(cols[b]["frontier_visits"]):
                raise GoldenError(f"{case.name}: batch {b}: frontier_visits differ")
            comps = (
                st["connected_components"].scalar("num_components")
                if "connected_components" in st
                else 0
            )
            if comps != int(cols[b]["components"]):
                raise GoldenError(f"{case.name}: batch {b}: components differ")
            per.append(
                {
                    "batch": b,
                    "stages": [s for s in dlt.BATCH_STAGES if s in st],
                    "iterations": end.scalar("iterations"),
                    "frontier_visits": end.scalar("frontier_visits"),
                    "persistent_iterations": end.scalar("persistent_iterations"),
                    "next_tag": end.scalar("next_tag"),
                    "components": comps,
                    "num_sub_edges": st["sparsify"].scalar("num_sub_edges")
                    if "sparsify" in st
                    else None,
                    "hook_rounds": st["connected_components"].scalar("hook_rounds")
                    if "connected_components" in st
                    else None,
                    "frontier_size": st["propagation_begin"].scalar("frontier_size"),
                    "labels_sha256": end["labels"].sha256,
                }
            )
        final = t.one("run_end")["final_labels"]
        if final.sha256 != batches[case.batches - 1]["propagation_end"]["labels"].sha256:
            raise GoldenError(f"{case.name}: the final labels are not the last batch's")
        if not case.digest and label_text(final.values()) != labels:
            raise GoldenError(f"{case.name}: the twin's final label bits do not print O's file")
        tau = next(
            (
                x.split(":", 1)[1].strip()
                for x in po.stdout.splitlines()
                if x.startswith("Auto tau")
            ),
            None,
        )
        zstd_write(labels, dd / "dynlp_labels.txt.zst")
        write_json(
            dd / "results.json",
            {
                "case": case.name,
                "argv": case.argv(Path("<out>")),
                "tau_printed": tau,
                "batches": cols,
            },
        )
        if case.digest:
            write_json(dd / "label_bits.json", [x["labels_sha256"] for x in per])
        else:
            zstd_write(
                b"".join(
                    batches[b]["propagation_end"]["labels"]._raw() for b in range(case.batches)
                ),
                dd / "label_bits.bin.zst",
            )
        canonical = dlt.encode(dlt.without(t.all, DROP), t.digest_mode)
        zstd_write(canonical, dt / "trace.dlt.zst")
        write_json(
            dt / "info.json",
            {
                "case": case.name,
                "argv": case.argv(Path("<out>")),
                "mode": "digest" if case.digest else "raw",
                "trace_sha256": sha256_bytes(canonical),
                "batches": per,
            },
        )
        shutil.rmtree(o, ignore_errors=True)
        shutil.rmtree(w, ignore_errors=True)
        return {"per_batch": per}

    # IrLP
    def irlp(self, case: Case, damped_per: list[dict] | None) -> None:
        d = set_dir(self.out, "irlp") / "cases" / case.name
        o = self.tmp(f"irlp-orig-{case.name}")
        w = self.tmp(f"irlp-twin-{case.name}")
        self.timed(f"irlp orig {case.name}", [self.orig, *case.argv(o / "out")], self.env)
        self.timed(
            f"irlp twin {case.name}",
            [self.twin, *case.argv(w / "out")],
            self.trace_env(w / "trace", False),
        )
        t = dlt.read(w / "trace" / "dynLP.dlt")
        cols = results_columns((o / "out" / "results.csv").read_text())
        if results_columns((w / "out" / "results.csv").read_text()) != cols:
            raise GoldenError(f"{case.name}: the twin's results.csv columns differ from O's")
        snaps = sorted(t.records("irlp_snapshot"), key=lambda r: r.batch)
        if [r.batch for r in snaps] != list(range(case.batches)):
            raise GoldenError(f"{case.name}: IrLP snapshots {[r.batch for r in snaps]}")
        per = []
        for r in snaps:
            if r.scalar("iterations") != int(cols[r.batch]["irlp_iterations"]):
                raise GoldenError(f"{case.name}: snapshot {r.batch}: IrLP iterations differ")
            per.append(
                {
                    "snapshot": r.batch,
                    "iterations": r.scalar("iterations"),
                    "labels_sha256": r["labels"].sha256,
                }
            )
        irlp_labels = (o / "out" / "irlp_labels.txt").read_bytes()
        if label_text(snaps[-1]["labels"].values()) != irlp_labels:
            raise GoldenError(f"{case.name}: the last snapshot's bits do not print irlp_labels.txt")
        if (w / "out" / "irlp_labels.txt").read_bytes() != irlp_labels:
            raise GoldenError(f"{case.name}: the twin's irlp_labels.txt differs from O's")
        if damped_per is not None:  # the --validate run's DynLP part is the plain run's
            mine = [t.batches()[b]["propagation_end"]["labels"].sha256 for b in range(case.batches)]
            if mine != [x["labels_sha256"] for x in damped_per]:
                raise GoldenError(f"{case.name}: --validate changed the DynLP labels")
        zstd_write(b"".join(r["labels"]._raw() for r in snaps), d / "irlp_bits.bin.zst")
        zstd_write(irlp_labels, d / "irlp_labels.txt.zst")
        write_json(
            d / "irlp.json",
            {
                "case": case.name,
                "argv": case.argv(Path("<out>")),
                "snapshots": per,
                "results": cols,
            },
        )
        shutil.rmtree(o, ignore_errors=True)
        shutil.rmtree(w, ignore_errors=True)

    # Hub
    def hub(self) -> None:
        d = set_dir(self.out, "hub")
        p = self.timed("hub orig", [self.hub_orig], self.env)
        orig_damped = [x for x in p.stdout.splitlines() if not x.startswith("inplace")]
        canonical = None
        for i in range(HUB_INPLACE_RUNS):
            w = self.tmp(f"hub-{i}")
            q = self.timed(f"hub twin {i}", [self.hub_twin], self.trace_env(w / "trace", False))
            if [x for x in q.stdout.splitlines() if not x.startswith("inplace")] != orig_damped:
                raise GoldenError(f"hub run {i}: the twin's damped lines differ from O's")
            t = dlt.read(w / "trace" / "propagation_test.dlt")
            calls = sorted(t.records("hub_propagation"), key=lambda r: r.batch)
            rules = [c.scalar("update_rule") for c in calls]
            if rules != [0, 1, 1]:
                raise GoldenError(f"hub run {i}: rules {rules}, expected in place, damped, damped")
            if calls[1]["labels"].sha256 != calls[2]["labels"].sha256:
                raise GoldenError(f"hub run {i}: the two damped calls differ")
            drop = {("hub_propagation", k) for k in HUB_INPLACE_OUTPUTS}
            mine = dlt.encode(dlt.without(calls[:1], drop) + calls[1:])
            if canonical is None:
                canonical = mine
            elif mine != canonical:
                raise GoldenError(f"hub run {i}: the canonical record differs from run 0's")
            c0 = calls[0]
            write_json(
                d / "inplace" / f"run-{i}.json",
                {k: c0.scalar(k) for k in HUB_INPLACE_OUTPUTS[:-1]}
                | {"labels_sha256": c0["labels"].sha256},
            )
            zstd_write(c0["labels"]._raw(), d / "inplace" / f"run-{i}.labels.bin.zst")
            shutil.rmtree(w, ignore_errors=True)
        assert canonical is not None
        zstd_write(canonical, d / "hub.dlt.zst")
        tt = dlt.read_bytes(canonical, "hub.dlt")
        calls = sorted(tt.records("hub_propagation"), key=lambda r: r.batch)
        write_json(
            d / "info.json",
            {
                "program": "bin/propagation_test (default delta 1e-7)",
                "original_damped_lines": orig_damped,
                "calls": [
                    {
                        "call": c.batch,
                        "update_rule": c.scalar("update_rule"),
                        "delta": c.scalar("delta"),
                        "num_nodes": c.scalar("num_nodes"),
                        **(
                            {k: c.scalar(k) for k in HUB_INPLACE_OUTPUTS[:-1]}
                            | {"labels_sha256": c["labels"].sha256}
                            if c.batch > 0
                            else {}
                        ),
                    }
                    for c in calls
                ],
                "hub_dlt_sha256": sha256_bytes(canonical),
                "inplace_runs": HUB_INPLACE_RUNS,
            },
        )


def unhashed(short: str, rel: str) -> bool:
    """Files of a set outside its manifest: provenance, and the hub's in-place distribution."""
    if rel in ("PROVENANCE.json", "MANIFEST.sha256"):
        return True
    return short == "hub" and rel.startswith("inplace/")


def write_manifest(set_path: Path, short: str) -> tuple[str, int, int]:
    lines, size = [], 0
    for p in sorted(set_path.rglob("*")):
        rel = p.relative_to(set_path).as_posix()
        if p.is_file() and not unhashed(short, rel):
            lines.append(f"{content_sha256(p)}  {rel}\n")
            size += p.stat().st_size
    text = "".join(lines)
    (set_path / "MANIFEST.sha256").write_text(text)
    return sha256_bytes(text.encode()), len(lines), size


def case_digests(set_path: Path) -> dict[str, dict]:
    """Per case: the SHA-256 of the sorted '<sha256>  <path in the case>' lines of its files."""
    manifest = (set_path / "MANIFEST.sha256").read_text().splitlines()
    cases: dict[str, list[str]] = {}
    for line in manifest:
        digest, rel = line.split("  ", 1)
        parts = rel.split("/")
        if parts[0] == "cases" and len(parts) > 2:
            cases.setdefault(parts[1], []).append(f"{digest}  {'/'.join(parts[2:])}\n")
    return {
        c: {"sha256": sha256_bytes("".join(sorted(v)).encode()), "files": len(v)}
        for c, v in sorted(cases.items())
    }


def export_copy(copy: Path, gpu: int, out: Path, sets: list[str]) -> dict:
    require_free(out.parent if out.parent.exists() else scratch())
    for s in sets:
        shutil.rmtree(set_dir(out, s), ignore_errors=True)
        set_dir(out, s).mkdir(parents=True)
    work = Path(tempfile.mkdtemp(prefix="lp-export-", dir=scratch() / "runs"))
    ex = Exporter(copy, gpu, out, work)
    started = now()
    try:
        if "generator" in sets:
            for c in GEN_CASES:
                print(f"generator {c.name}", flush=True)
                ex.generator(c)
        per: dict[str, list] = {}
        if "damped" in sets or "trace" in sets:
            for c in P_CASES:
                print(f"damped/trace {c.name}", flush=True)
                per[c.name] = ex.damped(c)["per_batch"]
        if "irlp" in sets:
            for c in IRLP_CASES:
                print(f"irlp {c.name}", flush=True)
                ex.irlp(c, per.get(Case(c.size).name))
        if "hub" in sets:
            print("hub", flush=True)
            ex.hub()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    result = {}
    # damped and trace are written together; a set not asked for is removed again
    for s in ("damped", "trace"):
        if s not in sets:
            shutil.rmtree(set_dir(out, s), ignore_errors=True)
    for s in sets:
        p = set_dir(out, s)
        manifest, files, size = write_manifest(p, s)
        write_json(
            p / "PROVENANCE.json",
            {
                "set": SETS[s],
                "reference": REFERENCE,
                "commit": COMMIT,
                "copy": str(copy),
                "copy_reference": (copy / ".dyng-reference").read_text(),
                "gpu": gpu_record(gpu),
                "date": started,
                "date_end": now(),
                "zstd": zstd_version(),
                "runs": ex.timings,
                "hashed_by_nothing": "this file; the hub's inplace/ distribution",
            },
        )
        result[s] = {"manifest_sha256": manifest, "files": files, "bytes": size}
        print(f"{SETS[s]}: {files} files, {size / 1e6:.1f} MB, MANIFEST.sha256 {manifest}")
    return result


def compare_exports(a: Path, b: Path, sets: list[str]) -> list[str]:
    problems = []
    for s in sets:
        ma, mb = set_dir(a, s) / "MANIFEST.sha256", set_dir(b, s) / "MANIFEST.sha256"
        if not ma.is_file() or not mb.is_file():
            problems.append(f"{SETS[s]}: a MANIFEST.sha256 is missing")
        elif ma.read_bytes() != mb.read_bytes():
            la, lb = set(ma.read_text().splitlines()), set(mb.read_text().splitlines())
            problems.append(f"{SETS[s]}: MANIFEST.sha256 differs: {sorted(la ^ lb)[:6]}")
    return problems


SET_WHAT = {
    "generator": "G: the generator-only twin runs: O's graph text files and the CSR value, "
    "label and batch bits (bytes at 5k5, 50k, 500k, seeds 1-3 at 5k5 and 50k, avgDegree 3 and 7 "
    "at 50k); the SHA-256 of every array at 5m, 5m1 and 50m",
    "damped": "P+: the unpatched O's dynlp_labels.txt, the deterministic results.csv columns and "
    "the twin's per-batch label bits (damped; 5m by digest)",
    "trace": "P+: the twin's canonical stage traces (status, rows and slots, G', components, "
    "W0/W1, initial labels and frontier, iterations, frontier_visits, persistentIterations, "
    "nextTag; 5m by digest)",
    "hub": "propagation_test's hub graph: the inputs and the damped calls; the in-place call's "
    "outputs as a distribution (not hashed)",
    "irlp": "IrLP per snapshot (5k5 and 50k damped, --validate): the label bits and iterations, "
    "O's irlp_labels.txt",
    "inplace": "the in_place calibration (AD5): N = 10 unpatched runs per I case on GPU 1 and 3 "
    "on GPU 0 at the base lock, the delta 1e-6 runs, the calibration record; a distribution",
}


def toml_section(short: str, set_path: Path) -> str:
    manifest = set_path / "MANIFEST.sha256"
    lines = manifest.read_text().splitlines()
    size = sum((set_path / ln.split("  ", 1)[1]).stat().st_size for ln in lines)
    out = [
        f"[sets.{SETS[short]}]",
        f'reference = "{REFERENCE}"',
        f'commit = "{COMMIT}"',
        f'kind = "{SETS[short]}"',
        f'what = "{SET_WHAT[short]}"',
        f'location = "$DYNG_SCRATCH/goldens/{SETS[short]}"',
        'generated_by = "parity/label_propagation_goldens.py '
        + (
            "calibrate + calibrate-analyze (exported once: a distribution)"
            if short == "inplace"
            else "export --twice (parity/export_goldens.py label_propagation)"
        )
        + '"',
        f"num_files = {len(lines)}",
        f"num_bytes = {size}",
        f'manifest_sha256 = "{sha256_bytes(manifest.read_bytes())}"',
    ]
    if short == "inplace":
        out += inplace_toml(set_path)
    cases = case_digests(set_path)
    if cases:
        out += ["", f"[sets.{SETS[short]}.cases]"]
        for name, c in cases.items():
            out.append(f'"{name}" = {{ sha256 = "{c["sha256"]}", files = {c["files"]} }}')
    return "\n".join(out) + "\n"


def write_toml(short: str, set_path: Path) -> None:
    import cycle_count_goldens  # the goldens.toml section writer (keeps the other sets)

    cycle_count_goldens.write_toml(toml_section(short, set_path), SETS[short])


def install(src: Path, sets: list[str], root: Path) -> None:
    for s in sets:
        dst = set_dir(root, s)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(set_dir(src, s)), str(dst))
        write_toml(s, dst)
        print(f"installed {dst} and wrote [sets.{SETS[s]}] of parity/goldens.toml")


# --- verify --------------------------------------------------------------------------------------


def golden_spec(name: str) -> dict | None:
    doc = tomllib.loads((REPO / "parity" / "goldens.toml").read_text())
    return doc.get("sets", {}).get(name)


def verify_set(short: str, root: Path) -> list[str]:
    name = SETS[short]
    spec = golden_spec(name)
    p = set_dir(root, short)
    if spec is None:
        return [f"parity/goldens.toml has no [sets.{name}]"]
    if not (p / "MANIFEST.sha256").is_file():
        return [f"{p}: no MANIFEST.sha256"]
    problems = []
    if sha256_file(p / "MANIFEST.sha256") != spec["manifest_sha256"]:
        problems.append(f"{name}: MANIFEST.sha256 differs from goldens.toml")
    for line in (p / "MANIFEST.sha256").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        if not (p / rel).is_file():
            problems.append(f"{name}: {rel} missing")
        elif content_sha256(p / rel) != digest:
            problems.append(f"{name}: {rel} differs from MANIFEST.sha256")
    if case_digests(p) != {k: dict(v) for k, v in spec.get("cases", {}).items()}:
        problems.append(f"{name}: the case digests differ from goldens.toml")
    problems += SEMANTIC[short](p) if short in SEMANTIC else []
    return problems


def semantic_damped(p: Path) -> list[str]:
    problems = []
    for c in P_CASES:
        d = p / "cases" / c.name
        res = json.loads((d / "results.json").read_text())
        if len(res["batches"]) != c.batches:
            problems.append(f"{c.name}: {len(res['batches'])} batches")
        labels = zstd_read(d / "dynlp_labels.txt.zst")
        lines = labels.count(b"\n")
        if lines != c.n:
            problems.append(f"{c.name}: dynlp_labels.txt has {lines} lines")
        if not c.digest:
            import array

            bits = zstd_read(d / "label_bits.bin.zst")
            if len(bits) != 4 * c.n * c.batches:
                problems.append(f"{c.name}: label_bits.bin has {len(bits)} bytes")
            else:
                last = array.array("f")
                last.frombytes(bits[4 * c.n * (c.batches - 1) :])
                if label_text(last) != labels:
                    problems.append(f"{c.name}: the last batch's bits do not print the labels")
    return problems


def semantic_trace(p: Path) -> list[str]:
    """Every batch has its marker; the host model of the batches (final_status) is the device
    status of the last batch; the per-batch counts of info.json are the trace's."""
    problems = []
    for c in P_CASES:
        d = p / "cases" / c.name
        t = dlt.read_bytes(zstd_read(d / "trace.dlt.zst"), f"{c.name}/trace.dlt")
        info = json.loads((d / "info.json").read_text())
        b = t.batches()
        if sorted(b) != list(range(c.batches)):
            problems.append(f"{c.name}: batches {sorted(b)}")
            continue
        for e in info["batches"]:
            end = b[e["batch"]]["propagation_end"]
            if end.scalar("iterations") != e["iterations"]:
                problems.append(f"{c.name}: batch {e['batch']}: iterations differ from info.json")
        if c.digest:
            continue
        status = final_status(c.n, t.records("batch_data"))
        device = b[c.batches - 1]["apply_vertex_changes"]["status"].numpy()
        if not (status == device).all():
            problems.append(f"{c.name}: the host status of the batches is not the device status")
    return problems


def semantic_generator(p: Path) -> list[str]:
    problems = []
    for c in GEN_CASES:
        d = p / "cases" / c.name
        t = dlt.read_bytes(zstd_read(d / "generator.dlt.zst"), f"{c.name}/generator.dlt")
        if len(t.records("batch_data")) != c.batches:
            problems.append(f"{c.name}: {len(t.records('batch_data'))} batches")
        if t.digest_mode != c.digest:
            problems.append(f"{c.name}: digest mode {t.digest_mode}")
        if not c.digest:
            text = csr_text(t)
            for f in CSR_FILES:
                if zstd_read(d / "graph" / f"{f}.zst") != text[f]:
                    problems.append(f"{c.name}: {f} is not the recorded arrays' text")
    return problems


def semantic_hub(p: Path) -> list[str]:
    t = dlt.read_bytes(zstd_read(p / "hub.dlt.zst"), "hub.dlt")
    calls = sorted(t.records("hub_propagation"), key=lambda r: r.batch)
    problems = []
    if [c.scalar("update_rule") for c in calls] != [0, 1, 1]:
        problems.append("hub: the calls are not in place, damped, damped")
    elif calls[1]["labels"].sha256 != calls[2]["labels"].sha256:
        problems.append("hub: the damped calls differ")
    if len(list((p / "inplace").glob("run-*.json"))) != HUB_INPLACE_RUNS:
        problems.append("hub: the in-place distribution is incomplete")
    return problems


def semantic_irlp(p: Path) -> list[str]:
    import array

    problems = []
    for c in IRLP_CASES:
        d = p / "cases" / c.name
        info = json.loads((d / "irlp.json").read_text())
        bits = zstd_read(d / "irlp_bits.bin.zst")
        if len(bits) != 4 * c.n * c.batches or len(info["snapshots"]) != c.batches:
            problems.append(f"{c.name}: {len(bits)} bytes of IrLP bits")
            continue
        last = array.array("f")
        last.frombytes(bits[4 * c.n * (c.batches - 1) :])
        if label_text(last) != zstd_read(d / "irlp_labels.txt.zst"):
            problems.append(f"{c.name}: the last snapshot does not print irlp_labels.txt")
    return problems


SEMANTIC = {
    "damped": semantic_damped,
    "trace": semantic_trace,
    "generator": semantic_generator,
    "hub": semantic_hub,
    "irlp": semantic_irlp,
}


# --- The in_place calibration (AD5) ---------------------------------------------------------------


def parse_labels(data: bytes):
    import numpy as np

    tokens = data.split()
    ids = np.array(tokens[0::2], dtype=np.int64)
    if not (ids == np.arange(len(ids))).all():
        raise GoldenError("a label file's vertex ids are not 0..n-1")
    return np.array(tokens[1::2], dtype=np.float64)


def calibrate(args: argparse.Namespace) -> int:
    """`runs` unpatched --update inplace runs per case on GPU `gpu` (default clocks, or the
    base lock inside the caller's EXCL window), each process's labels and results.csv kept."""
    sys.path.insert(0, str(REPO / "parity"))
    import label_propagation_perf as lpp
    import perf_ab

    exe = lpp.original_exe()
    root = set_dir(Path(args.root), "inplace")
    env = clean_env(args.gpu)
    delta = DELTAS[args.delta]
    clocks = perf_ab.ClockLock(args.gpu, args.lock_clocks)
    require_free(root.parent)
    work = Path(tempfile.mkdtemp(prefix="lp-calibrate-", dir=scratch() / "runs"))
    try:
        with clocks:
            for size in args.cases:
                case = Case(size, delta=args.delta, schedule="inplace")
                sub = "tight" if args.delta != "1e-4" else "runs"
                d = root / sub / case.name
                for i in range(args.runs):
                    out = work / "out"
                    shutil.rmtree(out, ignore_errors=True)
                    t0 = datetime.datetime.now(datetime.UTC)
                    run([exe, *case.argv(out)], env)
                    sec = (datetime.datetime.now(datetime.UTC) - t0).total_seconds()
                    stem = f"{args.tag}-run-{i:02d}"
                    labels = (out / "dynlp_labels.txt").read_bytes()
                    results = (out / "results.csv").read_bytes()
                    zstd_write(labels, d / f"{stem}.labels.txt.zst")
                    (d / f"{stem}.results.csv").write_bytes(results)
                    write_json(
                        d / f"{stem}.json",
                        {
                            "case": case.name,
                            "tag": args.tag,
                            "run": i,
                            "gpu": gpu_record(args.gpu),
                            "clocks": clocks.record,
                            "argv": case.argv(Path("<out>")),
                            "labels_sha256": sha256_bytes(labels),
                            "results_sha256": sha256_bytes(results),
                            "seconds": sec,
                            "date": now(),
                            "delta": delta,
                        },
                    )
                    print(f"{case.name} {stem}: {sec:.1f} s", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


def run_record(d: Path, stem: str) -> dict:
    meta = json.loads((d / f"{stem}.json").read_text())
    labels = zstd_read(d / f"{stem}.labels.txt.zst")
    if sha256_bytes(labels) != meta["labels_sha256"]:
        raise GoldenError(f"{d / stem}: the labels are not the recorded ones")
    cols = results_columns((d / f"{stem}.results.csv").read_text())
    return {"meta": meta, "labels": labels, "cols": cols}


def pair_stats(a, b, mask) -> dict:
    import numpy as np

    diff = np.abs(a[mask] - b[mask])
    flips = int(np.count_nonzero((a[mask] >= 0.5) != (b[mask] >= 0.5)))
    return {
        "mean": float(diff.mean()) if diff.size else 0.0,
        "max": float(diff.max()) if diff.size else 0.0,
        "flips": flips,
    }


def iteration_window(median: float) -> tuple[float, float]:
    """AD5: per batch within +/- 10 % of the reference median, at least +/- 2."""
    half = max(CALIBRATION["iterations_rel"] * median, CALIBRATION["iterations_floor"])
    return median - half, median + half


def status_of(case: Case, copy: Path, gpu: int):
    """The final status of a case's batches, from a raw generator-only run of the twin (the
    batches do not depend on the schedule)."""
    work = Path(tempfile.mkdtemp(prefix="lp-status-", dir=scratch() / "runs"))
    try:
        env = dict(
            clean_env(gpu), DYNG_DYNLP_TRACE_DIR=str(work / "trace"), DYNG_DYNLP_GENERATOR_ONLY="1"
        )
        run([copy / "bin" / "dynLP", *case.argv(work / "out")], env)
        t = dlt.read(work / "trace" / "dynLP.dlt")
        return final_status(case.n, t.records("batch_data"))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def calibrate_analyze(args: argparse.Namespace) -> int:
    import numpy as np

    sys.path.insert(0, str(REPO / "parity"))
    import perf_ab

    root = set_dir(Path(args.root), "inplace")
    copy = Path(
        subprocess.check_output(
            [
                str(REPO / "parity" / "build_reference.sh"),
                "--variant",
                "patched",
                "--print-dir",
                REFERENCE,
            ],
            text=True,
        )
        .strip()
        .splitlines()[-1]
    )
    k = CALIBRATION["k"]
    record: dict = {
        "schema": 1,
        "what": "the in_place calibration of LabelPropagation-CUDA@a276a3a (AD5, "
        "decided by the author on 2026-10-28): every pair of the N reference runs per case",
        "date": now(),
        "harness_commit": perf_ab.port_commit(),
        "reference": REFERENCE,
        "commit": COMMIT,
        "toolchain": "the unpatched copy built by its Makefile (CUDA 13.1 nvcc, "
        "sm_86, -O3 --use_fast_math, g++ 12.2; parity/references.toml)",
        "rule": CALIBRATION,
        "vertices": "ACTIVE and not ground truth after the last batch (O's host status of the "
        "batches; the classes of the 6-decimal label files, >= 0.5)",
        "cases": {},
        "gpu0_outside": [],
    }
    for size in CALIBRATION["cases"]:
        case = Case(size, schedule="inplace")
        d = root / "runs" / case.name
        ref = [run_record(d, f"gpu1-run-{i:02d}") for i in range(CALIBRATION["n_gpu1"])]
        g0 = [run_record(d, f"gpu0-run-{i:02d}") for i in range(CALIBRATION["n_gpu0"])]
        comps = [[int(x["components"]) for x in r["cols"]] for r in ref + g0]
        if any(c != comps[0] for c in comps):
            raise GoldenError(f"STOP: {case.name}: the components differ between the runs")
        status = status_of(case, copy, 1)
        mask = status == VERTEX_ACTIVE
        labels = [parse_labels(r["labels"]) for r in ref]
        pairs = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                pairs.append({"a": i, "b": j, **pair_stats(labels[i], labels[j], mask)})
        t_mean = max(k * max(p["mean"] for p in pairs), CALIBRATION["floor_mean"])
        t_max = max(k * max(p["max"] for p in pairs), CALIBRATION["floor_max"])
        t_flips = math.ceil(k * max(p["flips"] for p in pairs))
        its = np.array([[int(x["iterations"]) for x in r["cols"]] for r in ref])
        per_batch = [
            {
                "batch": b,
                "median": float(np.median(its[:, b])),
                "min": int(its[:, b].min()),
                "max": int(its[:, b].max()),
            }
            for b in range(case.batches)
        ]
        sums = its.sum(axis=1)
        cross = []
        for gi, g in enumerate(g0):
            lg = parse_labels(g["labels"])
            git = [int(x["iterations"]) for x in g["cols"]]
            for ri, lr in enumerate(labels):
                s = pair_stats(lg, lr, mask)
                ok = s["mean"] <= t_mean and s["max"] <= t_max and s["flips"] <= t_flips
                cross.append({"gpu0": gi, "gpu1": ri, **s, "within": ok})
            its_ok = all(
                iteration_window(e["median"])[0]
                <= git[e["batch"]]
                <= iteration_window(e["median"])[1]
                for e in per_batch
            )
            sum_ok = abs(sum(git) - float(np.median(sums))) <= 0.10 * float(np.median(sums))
            if not all(c["within"] for c in cross if c["gpu0"] == gi) or not its_ok or not sum_ok:
                record["gpu0_outside"].append(
                    {
                        "case": case.name,
                        "gpu0_run": gi,
                        "labels_within": all(c["within"] for c in cross if c["gpu0"] == gi),
                        "iterations_within": its_ok,
                        "iteration_sum_within": sum_ok,
                    }
                )
        # The iteration rule applied to the reference runs themselves (each against the median
        # of all N): reported, so the rule's reach is known before the port is read by it.
        self_check = []
        for ri, r in enumerate(ref):
            rit = [int(x["iterations"]) for x in r["cols"]]
            outside = [e["batch"] for e in per_batch
                       if not iteration_window(e["median"])[0] <= rit[e["batch"]]
                       <= iteration_window(e["median"])[1]]  # fmt: skip
            s_ok = abs(sum(rit) - float(np.median(sums))) <= 0.10 * float(np.median(sums))
            self_check.append({"gpu1_run": ri, "batches_outside": outside, "sum_within": s_ok})
        record["cases"][case.name] = {
            "n": case.n,
            "batches": case.batches,
            "vertices_compared": int(mask.sum()),
            "runs": {"gpu1": [r["meta"] for r in ref], "gpu0": [r["meta"] for r in g0]},
            "components": comps[0],
            "pairs": pairs,
            "largest": {
                "mean": max(p["mean"] for p in pairs),
                "max": max(p["max"] for p in pairs),
                "flips": max(p["flips"] for p in pairs),
            },
            "thresholds": {"t_mean": t_mean, "t_max": t_max, "t_flips": t_flips},
            "iterations": {
                "per_batch": per_batch,
                "sum_median": float(np.median(sums)),
                "sum_min": int(sums.min()),
                "sum_max": int(sums.max()),
            },
            "gpu0_cross_check": cross,
            "reference_iteration_self_check": self_check,
        }
        print(
            f"{case.name}: T_mean {t_mean:.3e} T_max {t_max:.3e} T_flips {t_flips}; "
            f"GPU 0 inside: {all(c['within'] for c in cross)}",
            flush=True,
        )
    # the tight runs (delta 1e-6), kept for M10a step 2's class margin
    tight = {}
    for size in CALIBRATION["tight_cases"]:
        case = Case(size, delta="1e-6", schedule="inplace")
        d = root / "tight" / case.name
        rs = [run_record(d, f"gpu1-run-{i:02d}") for i in range(CALIBRATION["tight_runs"])]
        tight[case.name] = {
            "runs": [r["meta"] for r in rs],
            "components": [int(x["components"]) for x in rs[0]["cols"]],
        }
        if any(
            [int(x["components"]) for x in r["cols"]] != tight[case.name]["components"] for r in rs
        ):
            raise GoldenError(f"STOP: {case.name}: the components differ between the runs")
    record["tight"] = tight
    write_json(root / "calibration.json", record)
    manifest, files, size = write_manifest(root, "inplace")
    write_json(
        root / "PROVENANCE.json", {"set": SETS["inplace"], "date": now(), "zstd": zstd_version()}
    )
    if args.json:
        write_json(args.json, record)
    print(f"{SETS['inplace']}: {files} files, {size / 1e6:.1f} MB, MANIFEST {manifest}")
    if not args.no_toml:
        write_toml("inplace", root)
        print("wrote [sets.label_propagation_inplace] of parity/goldens.toml")
    return 0


def inplace_toml(set_path: Path) -> list[str]:
    rec = json.loads((set_path / "calibration.json").read_text())
    rule = rec["rule"]
    out = [
        f'calibration_sha256 = "{sha256_file(set_path / "calibration.json")}"',
        "",
        f"[sets.{SETS['inplace']}.rule]",
        "# AD5 (decided by the author on 2026-10-28): frozen before the port; recalibration",
        "# only by ADR with the author's approval. T = k x the largest pairwise value of the N",
        "# reference runs (floors for the mean and the max), over the vertices ACTIVE and not",
        "# ground truth",
        "# after the last batch; iterations per batch within +/- 10 % of the reference median (at",
        "# least +/- 2) and their sum within +/- 10 %; components and the initial frontier exact.",
        f"k = {rule['k']}",
        f"floor_mean = {rule['floor_mean']}",
        f"floor_max = {rule['floor_max']}",
        f"n_gpu1 = {rule['n_gpu1']}",
        f"n_gpu0 = {rule['n_gpu0']}",
        f"iterations_rel = {rule['iterations_rel']}",
        f"iterations_floor = {rule['iterations_floor']}",
    ]
    for name, c in rec["cases"].items():
        t = c["thresholds"]
        meds = ", ".join(f"{e['median']:g}" for e in c["iterations"]["per_batch"])
        out += [
            "",
            f'[sets.{SETS["inplace"]}.thresholds."{name}"]',
            f"t_mean = {t['t_mean']!r}",
            f"t_max = {t['t_max']!r}",
            f"t_flips = {t['t_flips']}",
            f"components = {c['components']}",
            f"iterations_median = [{meds}]",
            f"iterations_sum_median = {c['iterations']['sum_median']!r}",
            f"vertices_compared = {c['vertices_compared']}",
        ]
    return out


def recompute_thresholds(rec: dict) -> dict[str, dict]:
    """The thresholds of a calibration record recomputed from its pairs (certify.py's
    frozen-tolerance check calls this; M10a step 5)."""
    k, rule = rec["rule"]["k"], rec["rule"]
    out = {}
    for name, c in rec["cases"].items():
        pairs = c["pairs"]
        out[name] = {
            "t_mean": max(k * max(p["mean"] for p in pairs), rule["floor_mean"]),
            "t_max": max(k * max(p["max"] for p in pairs), rule["floor_max"]),
            "t_flips": math.ceil(k * max(p["flips"] for p in pairs)),
        }
    return out


def semantic_inplace(p: Path) -> list[str]:
    rec = json.loads((p / "calibration.json").read_text())
    spec = golden_spec(SETS["inplace"]) or {}
    problems = []
    for name, t in recompute_thresholds(rec).items():
        frozen = spec.get("thresholds", {}).get(name)
        if frozen is None or any(frozen[key] != t[key] for key in t):
            problems.append(f"inplace {name}: the frozen thresholds are not the record's")
        if rec["cases"][name]["thresholds"] != t:
            problems.append(f"inplace {name}: the record's thresholds are not its pairs'")
    return problems


SEMANTIC["inplace"] = semantic_inplace


# --- O's --check margins (AD7) --------------------------------------------------------------------


def margins(args: argparse.Namespace) -> int:
    """The unpatched O with --check (verbose: the per-batch label line), OMP_NUM_THREADS=8,
    every output line timestamped as it arrives (stdbuf -oL), per batch: mean |F - F*| against
    2000 x delta (the margin = 2000 delta / mean), the PCG iterations and residual, skips, and the
    host time of the oracle's PCG (from the line before the label line to it)."""
    import re
    import threading
    import time

    sys.path.insert(0, str(REPO / "parity"))
    import label_propagation_perf as lpp
    import perf_ab

    exe = lpp.original_exe()
    env = dict(clean_env(args.gpu), OMP_NUM_THREADS="8")
    line_re = re.compile(
        r"\[check\] batch (\d+) ok: labels vs harmonic optimum \(PCG, (\d+) it, residual "
        r"([0-9.e+-]+)\): (\d+) vertices, mean \|F-F\*\| ([0-9.e+-]+) \(tol ([0-9.e+-]+)\), max "
        r"([0-9.e+-]+), class flips (\d+) \(([0-9.]+)%\); (\d+) in GT-free components excluded"
    )
    bad_re = re.compile(r"\[check\] batch (\d+) (FAILED|SKIPPED): (.*)")
    logdir = scratch() / "goldens" / CHECK_SET
    old = json.loads(args.json.read_text()) if args.json and args.json.is_file() else {}
    results = dict(old.get("results", {}))
    deadline = time.monotonic() + args.time_box
    for size in args.cases:
        for rule in args.rules:
            case = Case(
                size,
                delta=args.delta,
                schedule=rule,
                validate=(size == "5k5" and args.delta == "1e-6"),
            )
            key = case.name
            if args.skip_existing and results.get(key, {}).get("complete"):
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                results[key] = {"case": key, "complete": False, "reason": "time box spent"}
                continue
            work = Path(tempfile.mkdtemp(prefix="lp-check-", dir=scratch() / "runs"))
            cmd = [
                "stdbuf",
                "-oL",
                str(exe),
                *case.argv(work / "out", quiet=False)[:-2],
                "--check",
                "--output",
                str(work / "out"),
            ]
            lines: list[tuple[float, str]] = []
            t0 = time.monotonic()
            proc = subprocess.Popen(
                cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )

            def reader(p=proc, out=lines) -> None:
                assert p.stdout is not None
                for ln in p.stdout:
                    out.append((time.monotonic(), ln.rstrip("\n")))

            th = threading.Thread(target=reader, daemon=True)
            th.start()
            try:
                rc = proc.wait(timeout=remaining)
                timed_out = False
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()
                timed_out = True
            th.join()
            seconds = time.monotonic() - t0
            per = []
            prev_t = t0
            for t, ln in lines:
                m = line_re.search(ln)
                if m:
                    mean, tol = float(m.group(5)), float(m.group(6))
                    per.append(
                        {
                            "batch": int(m.group(1)),
                            "pcg_iterations": int(m.group(2)),
                            "residual": float(m.group(3)),
                            "vertices": int(m.group(4)),
                            "mean_abs_err": mean,
                            "tol": tol,
                            "max_abs_err": float(m.group(7)),
                            "class_flips": int(m.group(8)),
                            "excluded": int(m.group(10)),
                            "margin": tol / mean if mean > 0 else math.inf,
                            "pcg_host_s": t - prev_t,
                        }
                    )
                if "[check]" in ln:
                    prev_t = t
            skips = [ln for _, ln in lines if (m := bad_re.search(ln)) and m.group(2) == "SKIPPED"]
            fails = [ln for _, ln in lines if (m := bad_re.search(ln)) and m.group(2) == "FAILED"]
            summary = next(
                (ln for _, ln in lines if ln.startswith("[check]") and "checks," in ln), None
            )
            logdir.mkdir(parents=True, exist_ok=True)
            zstd_write(
                "\n".join(f"{t - t0:10.3f} {ln}" for t, ln in lines).encode() + b"\n",
                logdir / f"{key}.log.zst",
            )
            shutil.rmtree(work, ignore_errors=True)
            results[key] = {
                "case": key,
                "argv": cmd[2:],
                "exit": rc,
                "timed_out": timed_out,
                "seconds": seconds,
                "complete": not timed_out and rc == 0 and len(per) == case.batches,
                "batches": per,
                "skipped": skips,
                "failed": fails,
                "summary": summary,
                "two_thousand_delta": 2000 * float(DELTAS[args.delta]),
                "min_margin": min((b["margin"] for b in per), default=None),
                "log": f"$DYNG_SCRATCH/goldens/{CHECK_SET}/{key}.log.zst",
            }
            print(
                f"{key}: exit {rc}{' TIMEOUT' if timed_out else ''}, {len(per)} batches, "
                f"min margin {results[key]['min_margin']}, {len(skips)} skipped, "
                f"{seconds:.0f} s",
                flush=True,
            )
            if args.json:
                write_json(
                    args.json,
                    {
                        "schema": 1,
                        "what": "O's --check per batch (AD7)",
                        "reference": REFERENCE,
                        "commit": COMMIT,
                        "harness_commit": perf_ab.port_commit(),
                        "gpu": gpu_record(args.gpu),
                        "omp_num_threads": 8,
                        "date": old.get("date", now()),
                        "date_end": now(),
                        "results": dict(sorted(results.items())),
                    },
                )
    return 0


# --- The command line ---------------------------------------------------------------------------


def fresh_copy(n: int) -> Path:
    base = scratch() / "ref" / f"m10a-export-{n}"
    tool = REPO / "parity" / "build_reference.sh"
    subprocess.run(
        [str(tool), "--variant", "patched", "--fresh", "--scratch", str(base), REFERENCE],
        check=True,
    )
    out = subprocess.check_output(
        [str(tool), "--variant", "patched", "--scratch", str(base), "--print-dir", REFERENCE],
        text=True,
    )
    return Path(out.strip().splitlines()[-1])


def export_main(argv: list[str] | None = None) -> int:
    return main(["export", *(argv or [])])


def compare_main(argv: list[str] | None = None) -> int:
    """compare.py label_propagation --verify [--sets ...]: the goldens' own checks (the replays
    through dynG arrive with the port, M10a steps 2-4)."""
    argv = list(argv or [])
    if "--verify" not in argv:
        print("compare.py label_propagation: only --verify exists before the port", file=sys.stderr)
        return 2
    argv.remove("--verify")
    return main(["verify", *argv])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="label_propagation_goldens.py", description=__doc__.split("\n\n")[0]
    )
    sub = ap.add_subparsers(dest="command", required=True)
    sets_arg = {"default": list(EXPORTED_TWICE), "type": lambda s: s.split(",")}
    e = sub.add_parser("export")
    e.add_argument("--twice", action="store_true")
    e.add_argument("--copy-dir", type=Path, help="one export from this patched copy")
    e.add_argument("--fresh-copy", type=int, help="build the fresh copy m10a-export-<N> first")
    e.add_argument("--gpu", type=int, default=1)
    e.add_argument("--out", type=Path)
    e.add_argument("--sets", **sets_arg)
    c = sub.add_parser("compare-exports")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("--sets", **sets_arg)
    i = sub.add_parser("install")
    i.add_argument("src", type=Path)
    i.add_argument("--sets", **sets_arg)
    v = sub.add_parser("verify")
    v.add_argument("--sets", default=list(SETS), type=lambda s: s.split(","))
    k = sub.add_parser("calibrate")
    k.add_argument("--gpu", type=int, required=True)
    k.add_argument("--runs", type=int, required=True)
    k.add_argument("--tag", required=True, choices=["gpu1", "gpu0"])
    k.add_argument("--lock-clocks", choices=["none", "base"], default="none")
    k.add_argument("--cases", default=CALIBRATION["cases"], type=lambda s: s.split(","))
    k.add_argument("--delta", default="1e-4", choices=sorted(DELTAS))
    k.add_argument("--root", default=str(scratch() / "goldens"))
    a = sub.add_parser("calibrate-analyze")
    a.add_argument("--root", default=str(scratch() / "goldens"))
    a.add_argument("--json", type=Path)
    a.add_argument("--no-toml", action="store_true")
    m = sub.add_parser("margins")
    m.add_argument("--cases", required=True, type=lambda s: s.split(","))
    m.add_argument("--rules", default=["inplace", "damped"], type=lambda s: s.split(","))
    m.add_argument("--delta", required=True, choices=sorted(DELTAS))
    m.add_argument("--gpu", type=int, default=1)
    m.add_argument("--json", type=Path)
    m.add_argument("--time-box", type=float, default=7200.0)
    m.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args(argv)
    for s in getattr(args, "sets", []):
        if s not in SETS:
            ap.error(f"unknown set '{s}' (known: {', '.join(SETS)})")
    try:
        if args.command == "export":
            before = origin_status()
            if args.twice:
                return export_twice(args.sets)
            copy = fresh_copy(args.fresh_copy) if args.fresh_copy else args.copy_dir
            if copy is None or args.out is None:
                ap.error("give --twice, or --copy-dir / --fresh-copy and --out")
            export_copy(copy, args.gpu, args.out, args.sets)
            if origin_status() != before:
                raise GoldenError(f"{REFERENCE}'s git state changed during the export")
            return 0
        if args.command == "compare-exports":
            problems = compare_exports(args.a, args.b, args.sets)
            for p in problems:
                print(p)
            print("the exports are identical" if not problems else "the exports DIFFER")
            return 1 if problems else 0
        if args.command == "install":
            install(args.src, args.sets, scratch() / "goldens")
            return 0
        if args.command == "verify":
            problems = []
            for s in args.sets:
                found = verify_set(s, scratch() / "goldens")
                print(f"{SETS[s]}: {'ok' if not found else 'FAILED'}")
                problems += found
            for p in problems:
                print(f"  {p}")
            return 1 if problems else 0
        if args.command == "calibrate":
            return calibrate(args)
        if args.command == "calibrate-analyze":
            return calibrate_analyze(args)
        if args.command == "margins":
            return margins(args)
    except GoldenError as err:
        print(f"label_propagation_goldens {args.command}: {err}", file=sys.stderr)
        return 1
    return 2


def export_twice(sets: list[str]) -> int:
    """The double export (PLAN 8.3; M9a finding 4): copy 1 on GPU 1 under the shared lock, copy 2
    on GPU 0 in an exclusive-lock window of locklog.py (no timing), identical manifests, the
    first installed. Called holding no lock."""
    locklog = scratch() / "tools" / "locklog.py"
    lock = scratch() / "perf.lock"
    me = [sys.executable, str(Path(__file__).resolve())]
    outs = [scratch() / "runs" / f"lp-goldens-export-{n}" for n in (1, 2)]
    if subprocess.run([sys.executable, str(locklog), "quiet-ok", "M10a"]).returncode != 0:
        raise GoldenError("a GPU1-QUIET window is open: no GPU-1 job is started")
    require_free(scratch())
    subprocess.run(
        [
            "flock",
            "-s",
            str(lock),
            "nice",
            "-n",
            "10",
            *me,
            "export",
            "--fresh-copy",
            "1",
            "--gpu",
            "1",
            "--out",
            str(outs[0]),
            "--sets",
            ",".join(sets),
        ],
        check=True,
    )
    require_free(scratch())
    subprocess.run(
        [
            sys.executable,
            str(locklog),
            "run",
            "--reset-gpu",
            "0",
            "EXCL",
            "M10a",
            "reference-side-baseline-calibration",
            "goldens-export-2",
            "120",
            "--",
            *me,
            "export",
            "--fresh-copy",
            "2",
            "--gpu",
            "0",
            "--out",
            str(outs[1]),
            "--sets",
            ",".join(sets),
        ],
        check=True,
    )
    problems = compare_exports(outs[0], outs[1], sets)
    for p in problems:
        print(p)
    if problems:
        raise GoldenError(
            f"STOP: the two exports differ (kept for inspection: {outs[0]}, {outs[1]})"
        )
    print("the two exports are identical")
    install(outs[0], sets, scratch() / "goldens")
    shutil.rmtree(outs[1], ignore_errors=True)
    for n in (1, 2):
        shutil.rmtree(scratch() / "ref" / f"m10a-export-{n}", ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
