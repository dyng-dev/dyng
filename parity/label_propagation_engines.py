#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The refactored engines of label_propagation against the fused engine's damped bits (M10b step 1;
AD6, decided by the author on 2026-10-28; ADR 0101).

    compare.py label_propagation calibrate [--variants control,no_fma,permuted,both]
                                           [--cases ...] [--root DIR] [--keep] [--json F]
    compare.py label_propagation operators --exe COMPAT [--cases ...] [--json F]

AD6: a refactored engine (the operators engine) is judged on P per batch by max |dF| <= c x eps,
eps = 2^-23, with identical per-batch iterations, components and frontier_visits; c = 3 x the
largest max |dF| / eps that perturbing ONLY the association and the contraction of dynG's fused
damped engine produces over every batch of P, rounded up to a power of two, frozen in
parity/goldens.toml ([sets.label_propagation_damped.refactored]) before the refactored engine
exists. A looser bound is never allowed; if identical counts are impossible the engine stays
non-default and reported (ADR 0026's pattern).

calibrate   Exports HEAD (`git archive`) into ROOT/src and builds dyng-compat-dynlp in ROOT/build
            (the parity-cuda preset with DYNG_ALGORITHMS = the compatibility tools' algorithms),
            one variant after the other: each reconfigures DYNG_LP_CALIBRATION_CUDA_FLAGS, which
            only label_propagation's CUDA sources take, so a variant recompiles those alone:
              control   no extra flag (parity-cuda's -O3 -lineinfo -fmad=true); it must
                        reproduce the goldens' bits exactly (else the reduced build is not the
                        engine and the calibration stops)
              no_fma    -fmad=false (the contraction: every a * b + c a separate multiply and add;
                        an explicit -fmad=false wins over --use_fast_math's -fmad=true)
              permuted  DYNG_CALIBRATION_LP_PERMUTED_REDUCTION (propagation.cu: the lane and warp
                        trees in the other association order, the block's warp sums from the last)
              both      the two together
            Then every case of P (damped, the goldens' argv, --backend cuda --cuda-engine fused
            --label-dump) on GPU $DYNG_TEST_GPU (default 1), after `locklog.py quiet-ok
            $DYNG_LOCKLOG_MILESTONE` when that is set: per batch max |dF| against the twin's
            label bits and the per-batch iterations, components and frontier_visits against the
            golden results. The record holds c = 3 x max / eps rounded up to a power of two and
            whether a perturbation changed a count (which predicts AD6's fallback).
operators   The operators engine (--cuda-engine operators) on P+: per batch max |dF| <= c x eps
            against the twin's bits (5m: the per-batch SHA-256 of the bits against the golden's;
            on a difference the fused engine's own dump of the same run is the reference, the
            fused engine being bitwise the golden there by the damped replay), the per-batch
            iterations, components and frontier_visits identical to the golden's, and the
            dynlp_labels.txt bytes (reported: the file prints 6 decimals).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "parity"))

import label_propagation_goldens as g  # noqa: E402

EPSILON = 2.0**-23
# The counts AD6 requires identical (results.csv's columns, golden results.json's keys).
COUNTS = ("iterations", "components", "frontier_visits")
STEP = "m10b-operators-engine"
# The perturbations of the calibration (AD6: the association and the contraction only).
VARIANTS = {
    "control": {"fmad": True, "defines": []},
    "no_fma": {"fmad": False, "defines": []},
    "permuted": {"fmad": True, "defines": ["DYNG_CALIBRATION_LP_PERMUTED_REDUCTION"]},
    "both": {"fmad": False, "defines": ["DYNG_CALIBRATION_LP_PERMUTED_REDUCTION"]},
}
# P (AD6's calibration corpus): P+ without the digest case.
P_CASES = [c for c in g.P_CASES if not c.digest]
REFACTORED_TABLE = "refactored"  # [sets.label_propagation_damped.refactored]


class EngineError(Exception):
    pass


# --- The pure parts (tested in parity/tests/test_label_propagation_engines.py) -------------------


def constant_c(max_df: float, epsilon: float = EPSILON) -> int:
    """AD6's c: 3 x max |dF| / eps, rounded up to a power of two (at least 1)."""
    if not (max_df >= 0.0) or math.isinf(max_df):
        raise EngineError(f"max |dF| = {max_df} is not a finite non-negative number")
    m = 3.0 * max_df / epsilon
    if m <= 1.0:
        return 1
    c = 1 << math.ceil(math.log2(m))
    return c if c >= m else c * 2


def per_batch_max_abs(mine: bytes, gold: bytes, n: int) -> list[float]:
    """Per batch the largest |F_mine - F_gold| over every vertex (float32 dumps of n labels per
    batch); a NaN anywhere, or dumps of different lengths, raises EngineError."""
    import numpy as np

    if len(mine) != len(gold) or len(mine) % (4 * n):
        raise EngineError(
            f"the dumps differ in length ({len(mine)} and {len(gold)} bytes, n = {n})"
        )
    a = np.frombuffer(mine, dtype="<f4").reshape(-1, n).astype(np.float64)
    b = np.frombuffer(gold, dtype="<f4").reshape(-1, n).astype(np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise EngineError("a label is not finite")
    return [float(x) for x in np.abs(a - b).max(axis=1)]


def per_batch_digests(bits: bytes, n: int) -> list[str]:
    """The SHA-256 of every batch's label bits (the golden digest form of 5m)."""
    return [g.sha256_bytes(bits[i * 4 * n : (i + 1) * 4 * n]) for i in range(len(bits) // (4 * n))]


def count_mismatches(mine: list[dict], gold: list[dict]) -> list[dict]:
    """The per-batch counts (COUNTS) that differ: [{batch, field, mine, golden}]."""
    if len(mine) != len(gold):
        return [{"batch": -1, "field": "batches", "mine": len(mine), "golden": len(gold)}]
    out = []
    for b, (x, y) in enumerate(zip(mine, gold, strict=True)):
        for k in COUNTS:
            if str(x.get(k)) != str(y.get(k)):
                out.append({"batch": b, "field": k, "mine": x.get(k), "golden": y.get(k)})
    return out


def first_divergence(mine: bytes, gold: bytes, n: int) -> dict | None:
    """The first batch and vertex whose label bits differ (None when identical)."""
    import numpy as np

    a = np.frombuffer(mine, dtype="<u4").reshape(-1, n)
    b = np.frombuffer(gold, dtype="<u4").reshape(-1, n)
    for batch in range(a.shape[0]):
        diff = np.nonzero(a[batch] != b[batch])[0]
        if diff.size:
            v = int(diff[0])
            fa = np.frombuffer(a[batch, v].tobytes(), dtype="<f4")[0]
            fb = np.frombuffer(b[batch, v].tobytes(), dtype="<f4")[0]
            return {"batch": batch, "vertex": v, "mine": float(fa), "golden": float(fb)}
    return None


def frozen_bound(path: Path = REPO / "parity" / "goldens.toml") -> dict:
    """The frozen c and eps of goldens.toml's [sets.label_propagation_damped.refactored]."""
    doc = tomllib.loads(path.read_text())
    spec = doc["sets"]["label_propagation_damped"].get(REFACTORED_TABLE)
    if not spec:
        raise EngineError(
            "goldens.toml has no [sets.label_propagation_damped.refactored] (c is calibrated and "
            "frozen first: compare.py label_propagation calibrate)"
        )
    c, eps = int(spec["c"]), float(spec["epsilon"])
    if eps != EPSILON or c < 1 or c & (c - 1):
        raise EngineError(f"the frozen bound is malformed: c = {c}, epsilon = {eps}")
    record = spec.get("calibration_record")
    if record:  # the frozen c is the calibration's (its record unchanged since)
        data = (REPO / record).read_bytes()
        if g.sha256_bytes(data) != spec.get("calibration_sha256"):
            raise EngineError(f"{record} is not the calibration goldens.toml froze (SHA-256)")
        doc = json.loads(data)
        if not doc.get("control_exact") or constant_c(doc["max_df"]) != c:
            raise EngineError(f"the frozen c = {c} is not the calibration's ({record})")
    return {"c": c, "epsilon": eps, "bound": c * eps, "record": record}


def judge_batches(max_df: list[float], counts: list[dict], bound: float) -> dict:
    """AD6's verdict of one case: every batch's max |dF| <= bound and no count different."""
    over = [{"batch": b, "max_df": d} for b, d in enumerate(max_df) if d > bound]
    return {"within": not over, "over": over, "counts_identical": not counts, "equal": not over
            and not counts}  # fmt: skip


# --- Running ------------------------------------------------------------------------------------


def runs_dir() -> Path:
    d = g.scratch() / "runs" / STEP
    d.mkdir(parents=True, exist_ok=True)
    return d


def work_dir(prefix: str) -> Path:
    """A work directory of this invocation under runs_dir(): concurrent runs of the comparator
    (ci/gpu_local.sh, reviewers, verifiers) share $DYNG_SCRATCH, and a fixed directory would let
    one run delete or read another's dumps."""
    return Path(tempfile.mkdtemp(prefix=prefix, dir=runs_dir()))


def heavy(argv: list[str]) -> list[str]:
    """A build under the shared perf lock, niced (no lock inside a window that holds it)."""
    lock = os.environ.get("DYNG_PERF_LOCK", str(g.scratch() / "perf.lock"))
    return (["flock", "-s", lock] if lock else []) + ["nice", "-n", "10", *argv]


def wait_gpu_quiet() -> None:
    """Before a GPU-1 job: wait while another GPU1-QUIET window is open (the shared calendar's
    `locklog.py quiet-ok $DYNG_LOCKLOG_MILESTONE`), when that variable is set."""
    milestone = os.environ.get("DYNG_LOCKLOG_MILESTONE")
    tool = g.scratch() / "tools" / "locklog.py"
    if not milestone or not tool.is_file():
        return
    while subprocess.run([sys.executable, str(tool), "quiet-ok", milestone],
                         capture_output=True).returncode == 1:  # fmt: skip
        print("[engines] waiting: a GPU1-QUIET window is open (locklog.py quiet-ok)", flush=True)
        time.sleep(60)


def gpu_env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DYNG_DYNLP_", "OMP_"))}
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("DYNG_TEST_GPU", "1")
    return env


def run_case(exe: Path, case: g.Case, engine: str, work: Path) -> dict:
    """One damped run of `case` with `engine`: its per-batch label bits and results.csv counts."""
    import label_propagation_port as port

    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    argv = [*case.argv(work / "out"), "--backend", "cuda", "--cuda-engine", engine,
            "--label-dump", str(work / "d")]  # fmt: skip
    wait_gpu_quiet()
    p = subprocess.run([str(exe), *argv], capture_output=True, text=True, env=gpu_env())
    if p.returncode != 0:
        raise EngineError(f"{exe} {' '.join(argv)} failed ({p.returncode}): "
                          f"{(p.stdout + p.stderr)[-2000:]}")  # fmt: skip
    return {
        "bits": (work / "d" / "labels.f32").read_bytes(),
        "labels": (work / "out" / "dynlp_labels.txt").read_bytes(),
        "counts": port.results_columns((work / "out" / "results.csv").read_text()),
    }


def golden_of(case: g.Case) -> dict:
    root = g.set_dir(g.scratch() / "goldens", "damped") / "cases" / case.name
    res = json.loads((root / "results.json").read_text())
    out = {"counts": res["batches"], "labels": g.zstd_read(root / "dynlp_labels.txt.zst")}
    if case.digest:
        out["digests"] = json.loads((root / "label_bits.json").read_text())
    else:
        out["bits"] = g.zstd_read(root / "label_bits.bin.zst")
    return out


# The algorithms the compatibility tools need (CMakeLists.txt: tools/compat is built with them).
CALIBRATION_ALGORITHMS = "label_propagation;sssp;cycle_count;mosp;triad_count"


def variant_flags(variant: str) -> str:
    """DYNG_LP_CALIBRATION_CUDA_FLAGS of a variant (label_propagation's CUDA sources only)."""
    spec = VARIANTS[variant]
    flags = [] if spec["fmad"] else ["-fmad=false"]
    return " ".join(flags + [f"-D{d}" for d in spec["defines"]])


def build_variant(src: Path, tree: Path, variant: str, log: Path) -> Path:
    """dyng-compat-dynlp of `variant` in `tree` (one tree for every variant: a reconfigure
    recompiles label_propagation's CUDA sources only)."""
    configure = ["cmake", "-S", str(src), "-B", str(tree), "--preset", "parity-cuda",
                 f"-DDYNG_ALGORITHMS={CALIBRATION_ALGORITHMS}", "-DDYNG_BUILD_TESTS=OFF",
                 "-DDYNG_BUILD_PARITY_TESTS=OFF", "-DDYNG_BUILD_EXAMPLES=OFF",
                 "-DDYNG_BUILD_TUTORIALS=OFF", "-DDYNG_BUILD_BENCHMARKS=OFF",
                 f"-DDYNG_LP_CALIBRATION_CUDA_FLAGS={variant_flags(variant)}"]  # fmt: skip
    exe = tree / "tools" / "compat" / "dyng-compat-dynlp"
    with open(log, "a") as f:
        for argv in (configure, ["cmake", "--build", str(tree), "--target", "dyng-compat-dynlp"]):
            f.write(f"== {' '.join(argv)}\n")
            f.flush()
            if subprocess.run(heavy(argv), cwd=src, stdout=f, stderr=subprocess.STDOUT).returncode:
                raise EngineError(f"the {variant} build failed (log: {log})")
    if not exe.is_file():
        raise EngineError(f"the {variant} build has no {exe}")
    return exe


def head_commit() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()  # fmt: skip


def cmd_calibrate(args: argparse.Namespace) -> int:
    root = args.root or (g.scratch() / "m10-builds" / "calibration")
    commit = head_commit()
    src = root / "src"
    shutil.rmtree(src, ignore_errors=True)
    src.mkdir(parents=True)
    archive = subprocess.run(["git", "-C", str(REPO), "archive", "HEAD"], check=True,
                             capture_output=True).stdout  # fmt: skip
    subprocess.run(["tar", "-x", "-C", str(src)], input=archive, check=True)
    log = runs_dir() / "calibrate-build.log"
    tree = root / "build"
    cases = [c for c in P_CASES if not args.cases or c.name in args.cases]
    variants: dict = {}
    overall = 0.0
    counts_changed = []
    control_ok = True
    work = work_dir("calibrate-")
    # One variant at a time: build (a reconfigure recompiles label_propagation's CUDA sources),
    # then every case on it (the tree's shared library is the variant's until the next build).
    for v in args.variants:
        exe = build_variant(src, tree, v, log)
        per_case = {}
        for case in cases:
            t0 = time.monotonic()
            mine = run_case(exe, case, "fused", work / v)
            gold = golden_of(case)
            dfs = per_batch_max_abs(mine["bits"], gold["bits"], case.n)
            counts = count_mismatches(mine["counts"], gold["counts"])
            per_case[case.name] = {
                "max_df": dfs,
                "max_df_over_eps": [d / EPSILON for d in dfs],
                "count_mismatches": counts,
                "first_divergence": first_divergence(mine["bits"], gold["bits"], case.n),
                "labels_txt_equal": mine["labels"] == gold["labels"],
                "seconds": round(time.monotonic() - t0, 1),
            }
            differ = f"; counts differ: {counts[:3]}" if counts else ""
            print(f"{v} {case.name}: max |dF| = {max(dfs):.3e} ({max(dfs) / EPSILON:.1f} eps)"
                  f"{differ}", flush=True)  # fmt: skip
            if v == "control":
                control_ok = control_ok and max(dfs) == 0.0 and not counts
            else:
                overall = max(overall, max(dfs))
                if counts:
                    counts_changed.append({"variant": v, "case": case.name, "first": counts[0]})
        variants[v] = per_case
    shutil.rmtree(work, ignore_errors=True)
    perturbed = [v for v in args.variants if v != "control"]
    c = constant_c(overall) if perturbed else None
    doc = {
        "schema": 1,
        "what": "AD6's calibration of c (M10b step 1; ADR 0101): dynG's fused damped engine with "
        "only its association (the lane trees) and contraction (-fmad) perturbed, over every "
        "batch of P, against the twin's per-batch label bits; c = 3 x max |dF| / eps rounded up "
        "to a power of two, eps = 2^-23",
        "reference": g.REFERENCE,
        "commit": g.COMMIT,
        "port_commit": commit,
        "date": g.now(),
        "epsilon": EPSILON,
        "variants": {v: {**VARIANTS[v], "flags": variant_flags(v)} for v in args.variants},
        "build": "git archive of port_commit; the parity-cuda preset with DYNG_ALGORITHMS="
        + CALIBRATION_ALGORITHMS
        + " (the compatibility tools' algorithms) and "
        "DYNG_LP_CALIBRATION_CUDA_FLAGS per variant",
        "cases": [c_.name for c_ in cases],
        "control_exact": control_ok,
        "max_df": overall,
        "max_df_over_eps": overall / EPSILON,
        "c": c,
        "bound": None if c is None else c * EPSILON,
        "counts_changed": counts_changed,
        "fallback_predicted": bool(counts_changed),
        "per_variant": variants,
    }
    if args.json:
        g.write_json(args.json, doc)
    if not args.keep:
        shutil.rmtree(tree, ignore_errors=True)
        shutil.rmtree(src, ignore_errors=True)
    print(f"calibrate: control {'exact' if control_ok else 'NOT EXACT'}; max |dF| = {overall:.3e} "
          f"= {overall / EPSILON:.2f} eps; c = {c}; counts changed by a perturbation: "
          f"{len(counts_changed)}")  # fmt: skip
    return 0 if control_ok else 1


def cmd_operators(args: argparse.Namespace) -> int:
    bound = frozen_bound()
    cases = [c for c in g.P_CASES if not args.cases or c.name in args.cases]
    results, ok = [], True
    work = work_dir("operators-")
    for case in cases:
        t0 = time.monotonic()
        mine = run_case(args.exe, case, "operators", work / "operators")
        gold = golden_of(case)
        counts = count_mismatches(mine["counts"], gold["counts"])
        row = {"case": case.name}
        if case.digest:
            identical = per_batch_digests(mine["bits"], case.n) == gold["digests"]
            row["bits_identical_by_digest"] = identical
            if identical:
                dfs = [0.0] * case.batches
            else:  # the fused engine's dump of the same case is the reference
                ref = run_case(args.exe, case, "fused", work / "operators-fused")
                if per_batch_digests(ref["bits"], case.n) != gold["digests"]:
                    raise EngineError(f"{case.name}: the fused engine is not the golden's bits")
                dfs = per_batch_max_abs(mine["bits"], ref["bits"], case.n)
                row["first_divergence"] = first_divergence(mine["bits"], ref["bits"], case.n)
                shutil.rmtree(work / "operators-fused", ignore_errors=True)
        else:
            dfs = per_batch_max_abs(mine["bits"], gold["bits"], case.n)
            row["first_divergence"] = first_divergence(mine["bits"], gold["bits"], case.n)
        verdict = judge_batches(dfs, counts, bound["bound"])
        row.update(
            {
                "max_df": dfs,
                "max_df_over_eps": [d / EPSILON for d in dfs],
                "count_mismatches": counts,
                "labels_txt_equal": mine["labels"] == gold["labels"],
                "seconds": round(time.monotonic() - t0, 1),
                **verdict,
            }
        )
        ok = ok and verdict["equal"]
        results.append(row)
        print(f"{case.name}: {'within' if verdict['equal'] else 'OUTSIDE'} (max |dF| = "
              f"{max(dfs):.3e}, bound {bound['bound']:.3e}; counts "
              f"{'identical' if not counts else counts[:3]})", flush=True)  # fmt: skip
    shutil.rmtree(work, ignore_errors=True)
    if args.json:
        g.write_json(
            args.json,
            {
                "schema": 1,
                "what": "the operators engine against the fused engine's damped bits on P+ (AD6; "
                "M10b criterion 2): per batch max |dF| <= c x eps with identical iterations, "
                "components and frontier_visits",
                "reference": g.REFERENCE,
                "commit": g.COMMIT,
                "bound": bound,
                "date": g.now(),
                "equal": ok,
                "cases": results,
            },
        )
    print("operators:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


COMMANDS = {"calibrate": cmd_calibrate, "operators": cmd_operators}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="compare.py label_propagation", description=__doc__.split("\n\n")[0]
    )
    sub = ap.add_subparsers(dest="command", required=True)
    names = lambda s: [x for x in s.split(",") if x]  # noqa: E731
    cp = sub.add_parser("calibrate")
    cp.add_argument("--variants", type=names, default=list(VARIANTS))
    cp.add_argument("--cases", type=names)
    cp.add_argument("--root", type=Path, help="the builds (default $DYNG_SCRATCH/m10-builds/"
                    "calibration)")  # fmt: skip
    cp.add_argument("--keep", action="store_true", help="keep the source export and the builds")
    cp.add_argument("--json", type=Path)
    op = sub.add_parser("operators")
    op.add_argument("--exe", type=Path, required=True, help="dyng-compat-dynlp (parity-cuda)")
    op.add_argument("--cases", type=names)
    op.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    if args.command == "calibrate" and "control" not in args.variants:
        raise SystemExit("calibrate: the control variant is required (it proves the build)")
    try:
        rc = COMMANDS[args.command](args)
    except (EngineError, g.GoldenError) as err:
        print(f"compare.py label_propagation {args.command}: {err}", file=sys.stderr)
        return 1
    if args.command == "operators":
        import label_propagation_cert

        label_propagation_cert.stamp_port(args.json, args.exe)
    return rc


if __name__ == "__main__":
    sys.exit(main())
