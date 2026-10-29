#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The replays of dynG's label_propagation (the port) against LabelPropagation-CUDA@a276a3a's
(O's) goldens (M10a step 3 on; ADRs 0069-0073; the author's decisions AD5-AD7 and C8 of
2026-10-28).

    compare.py label_propagation container --exe REPLAY [--backend sequential|cuda] [--cases ...]
                                           [--json F]
    compare.py label_propagation trace --exe COMPAT [--backend sequential] [--cases ...] [--json F]
    compare.py label_propagation oracle-port --exe COMPAT [--backend sequential] [--json F]
    compare.py label_propagation determinism --exe COMPAT [--backend sequential] [--json F]

REPLAY is dyng_lp_replay (cpp/tests/parity/label_propagation), COMPAT dyng-compat-dynlp
(tools/compat/dynlp) of a build.

container   For every case of P+ (label_propagation_goldens.P_CASES): O's own recorded batches
            (5m: the generator's, whose digests are checked first) applied to graph::with_capacity()
            (slotted rows, batch_semantics::net_effect(); the host apply), and after every batch the
            graph's row ends, slot columns and value bits and its states (O's codes by name, the
            seed labels: ADR 0069) against the golden trace, check_integrity() and lp_host_model.
trace       For every case of P+: dyng-compat-dynlp with the case's argv, --backend, and --trace
            (the library's stage trace in the recorder's format; DYNG_DYNLP_TRACE_DIGEST=1 for the
            digest cases), then every deterministic field of every batch against the golden trace:
            the statuses, Ins \\ Del, the rows (row ends, slot columns and value bits, b0's
            offsets), the error flags, tau, G' (count and sorted pairs), the components (count and
            the id of every inserted vertex), the supernode weights W0 / W1, the initial labels'
            bits, and the initial frontier (size and vertices); the stages O does not run in a
            batch must be absent. The labels after propagation are not compared here (on
            sequential the oracle judges them).
oracle-port On C = {5k5, 50k} at delta 1e-6, both rules: dyng-compat-dynlp --check (the harmonic
            oracle per batch, AD7: mean |F - F*| <= 2000 delta, no PCG skip, seeds exact, labels
            in [0, 1]) and the class margin rule (no flip where |F* - 0.5| > 3 m, m from
            parity/results/M10a-class-margin.json) on its label dump, and IrLP (--validate) on
            the snapshots at 5k5 delta 1e-6 (agreement with DynLP reported).
determinism Two runs of every C case on the backend: damped label bits identical run to run, and
            in_place too on sequential (it updates each half-sweep in vertex order).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "parity"))

import dynlp_trace as dlt  # noqa: E402
import label_propagation_goldens as g  # noqa: E402

STEP = "m10a-host-container-sequential-driver"
# The deterministic fields of each stage of a batch (O's recorder's names; ADR 0065).
TRACE_FIELDS = {
    "apply_vertex_changes": ["num_inserted", "inserted", "status", "gt_class"],
    "append_edges": ["row_end", "slot_col_ind", "slot_values", "error_flags"],
    "sparsify": ["tau", "num_sub_edges", "sub_edges_sorted"],
    "connected_components": ["num_components", "component_of_inserted"],
    "label_init": ["num_components", "w0", "w1", "initial_labels_of_inserted"],
    "propagation_begin": ["frontier_size", "initial_frontier"],
}
C_CASES = [
    g.Case(s, delta="1e-6", schedule=r) for s in ("5k5", "50k") for r in ("damped", "inplace")
]
MARGIN_RECORD = REPO / "parity" / "results" / "M10a-class-margin.json"


def runs_dir() -> Path:
    d = g.scratch() / "runs" / STEP
    d.mkdir(parents=True, exist_ok=True)
    return d


def select(cases: list, names: list | None) -> list:
    if not names:
        return cases
    known = {c.name: c for c in cases}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise g.GoldenError(f"unknown cases {unknown} (known: {sorted(known)})")
    return [known[n] for n in names]


def clean_env(**extra: str) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DYNG_DYNLP_", "OMP_"))}
    env.update(extra)
    return env


def decompressed(src: Path, tmp: Path) -> Path:
    out = tmp / (src.name[:-4] if src.name.endswith(".zst") else src.name)
    out.write_bytes(g.zstd_read(src))
    return out


def run_compat(exe: Path, argv: list[str], env: dict) -> subprocess.CompletedProcess:
    p = subprocess.run([str(exe), *argv], capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise g.GoldenError(f"{exe.name} {' '.join(argv)} failed ({p.returncode}): "
                            f"{(p.stdout + p.stderr)[-2000:]}")  # fmt: skip
    return p


def compare_traces(ours: dlt.Trace, golden: dlt.Trace) -> tuple[int, list[dict]]:
    """Every deterministic field of every batch (TRACE_FIELDS; b0's row offsets too)."""
    # The batch loop only: the port's compute() on the empty graph records an irlp_snapshot
    # outside it (batch -1).
    mine = {b: r for b, r in ours.batches().items() if b >= 0}
    theirs = {b: r for b, r in golden.batches().items() if b >= 0}
    compared, mismatches = 0, []
    if sorted(mine) != sorted(theirs):
        mismatches.append(
            {"field": "batches", "detail": f"port {sorted(mine)} vs golden {sorted(theirs)}"}
        )
    for b in sorted(theirs):
        for tag, names in TRACE_FIELDS.items():
            gold = theirs[b].get(tag)
            port = mine.get(b, {}).get(tag)
            if gold is None or port is None:
                if (gold is None) != (port is None):
                    mismatches.append(
                        {
                            "batch": b,
                            "field": tag,
                            "detail": "recorded by the "
                            + ("golden" if port is None else "port")
                            + " only",
                        }
                    )
                continue
            fields = list(names) + (["row_ptr"] if tag == "append_edges" and b == 0 else [])
            for name in fields:
                compared += 1
                f, h = port.fields.get(name), gold.fields.get(name)
                if f is None or h is None:
                    mismatches.append({"batch": b, "field": f"{tag}.{name}", "detail": "missing"})
                elif (f.dtype, f.count, f.sha256) != (h.dtype, h.count, h.sha256):
                    mismatches.append(
                        {
                            "batch": b,
                            "field": f"{tag}.{name}",
                            "detail": f"port {f.type_name}[{f.count}] {f.sha256[:12]} vs golden "
                            f"{h.type_name}[{h.count}] {h.sha256[:12]}",
                        }
                    )
    return compared, mismatches


# --- container ------------------------------------------------------------------------------------


def cmd_container(args: argparse.Namespace) -> int:
    root = g.set_dir(g.scratch() / "goldens", "trace")
    results, ok = [], True
    for case in select(g.P_CASES, args.cases):
        with tempfile.TemporaryDirectory(prefix="lp-container-", dir=runs_dir()) as t:
            trace = decompressed(root / "cases" / case.name / "trace.dlt.zst", Path(t))
            p = subprocess.run(
                [
                    str(args.exe),
                    "container",
                    "--trace",
                    str(trace),
                    "--threads",
                    str(args.threads),
                    "--backend",
                    args.backend,
                ],
                capture_output=True,
                text=True,
            )
        if p.returncode not in (0, 1):
            raise g.GoldenError(f"dyng_lp_replay container failed: {p.stderr[-2000:]}")
        out = json.loads(p.stdout)
        good = p.returncode == 0 and out["equal"]
        ok = ok and good
        results.append({"case": case.name, "backend": args.backend, **out})
        print(
            f"{case.name}: {'exact' if good else 'DIFFERENT'} ({out['compared']} fields, "
            f"{out['mismatched']} mismatched, inputs {out['inputs']}; {out['seconds']:.1f} s)"
            + (f": {out['problem']} {out['mismatches'][:5]}" if not good else ""),
            flush=True,
        )
    if args.json:
        g.write_json(
            args.json,
            {
                "schema": 1,
                "what": "the container replay (criterion 9): O's batches on graph::with_capacity, "
                "the host apply, against the trace golden and lp_host_model",
                "reference": g.REFERENCE,
                "commit": g.COMMIT,
                "date": g.now(),
                "equal": ok,
                "cases": results,
            },
        )
    print("container replay:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# --- trace ----------------------------------------------------------------------------------------


def cmd_trace(args: argparse.Namespace) -> int:
    root = g.set_dir(g.scratch() / "goldens", "trace")
    results, ok = [], True
    for case in select(g.P_CASES, args.cases):
        with tempfile.TemporaryDirectory(prefix="lp-trace-", dir=runs_dir()) as t:
            tmp = Path(t)
            golden = dlt.read(decompressed(root / "cases" / case.name / "trace.dlt.zst", tmp))
            env = clean_env(**({"DYNG_DYNLP_TRACE_DIGEST": "1"} if case.digest else {}))
            argv = [*case.argv(tmp / "out"), "--backend", args.backend, "--trace", str(tmp / "t")]
            import time

            t0 = time.monotonic()
            run_compat(args.exe, argv, env)
            seconds = time.monotonic() - t0
            ours = dlt.read(tmp / "t" / f"{args.exe.name}.dlt")
        compared, mismatches = compare_traces(ours, golden)
        good = not mismatches and ours.digest_mode == golden.digest_mode
        ok = ok and good
        results.append(
            {
                "case": case.name,
                "backend": args.backend,
                "batches": len(golden.batches()),
                "compared": compared,
                "mismatches": mismatches,
                "seconds": round(seconds, 1),
                "equal": good,
            }
        )
        print(
            f"{case.name}: {'exact' if good else 'DIFFERENT'} ({compared} fields of "
            f"{len(golden.batches())} batches; {seconds:.1f} s)"
            + (f": {mismatches[:4]}" if mismatches else ""),
            flush=True,
        )
    if args.json:
        g.write_json(
            args.json,
            {
                "schema": 1,
                "what": "the deterministic stages of dynG's label_propagation against O's trace "
                "golden (criterion 11): status, rows, V_aff, G', components, W0/W1, the initial "
                "labels, tau",
                "reference": g.REFERENCE,
                "commit": g.COMMIT,
                "date": g.now(),
                "fields": TRACE_FIELDS,
                "equal": ok,
                "cases": results,
            },
        )
    print(f"trace replay ({args.backend}):", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# --- oracle-port ----------------------------------------------------------------------------------


def class_margin() -> float:
    doc = json.loads(MARGIN_RECORD.read_text())
    return float(doc["m"])


def labels_of_dump(path: Path, n: int) -> list:
    import array

    data = path.read_bytes()
    a = array.array("f")
    a.frombytes(data)
    return [a[i * n : (i + 1) * n] for i in range(len(a) // n)]


def cmd_oracle_port(args: argparse.Namespace) -> int:
    margin = class_margin()
    results, ok = [], True
    for case in C_CASES:
        validate = case.size == "5k5"
        c = g.Case(case.size, delta=case.delta, schedule=case.schedule, validate=validate)
        with tempfile.TemporaryDirectory(prefix="lp-oracle-", dir=runs_dir()) as t:
            tmp = Path(t)
            argv = c.argv(tmp / "out", quiet=False)
            argv += ["--backend", args.backend, "--check", "--label-dump", str(tmp / "dump")]
            p = subprocess.run(
                [str(args.exe), *argv], capture_output=True, text=True, env=clean_env()
            )
            lines = [x for x in p.stdout.splitlines() if x.startswith("[check]")]
            dump = tmp / "dump" / "labels.f32"
            margin_out = subprocess.run(
                [
                    str(args.replay),
                    "oracle",
                    *margin_args(c),
                    "--labels-bits",
                    str(dump),
                    "--threads",
                    str(args.threads),
                ],
                capture_output=True,
                text=True,
            )
            results_csv = (tmp / "out" / "results.csv").read_text() if validate else ""
            irlp_out = None
            if validate:
                irlp_out = subprocess.run(
                    [
                        str(args.replay),
                        "oracle",
                        *margin_args(c),
                        "--labels-bits",
                        str(tmp / "dump" / "irlp.f32"),
                        "--threads",
                        str(args.threads),
                    ],
                    capture_output=True,
                    text=True,
                )
        if margin_out.returncode != 0 or (irlp_out is not None and irlp_out.returncode != 0):
            raise g.GoldenError(f"dyng_lp_replay oracle failed: {margin_out.stderr[-2000:]}")
        oracle = json.loads(margin_out.stdout)
        flips_beyond = []
        batches = []
        sources = [("update", b) for b in oracle["batches"]]
        if irlp_out is not None:
            sources += [("compute (IrLP)", b) for b in json.loads(irlp_out.stdout)["batches"]]
        for source, b in sources:
            limit = 2000.0 * float(g.DELTAS[c.delta])
            good_b = (
                b["converged"] and b["mean"] <= limit and b["largest_flip_distance"] <= 3 * margin
            )
            if b["largest_flip_distance"] > 3 * margin:
                flips_beyond.append(f"{source} {b['batch']}")
            batches.append(
                {
                    "labels": source,
                    "batch": b["batch"],
                    "mean": b["mean"],
                    "bound": limit,
                    "margin": (limit / b["mean"]) if b["mean"] > 0 else None,
                    "max": b["max"],
                    "flips": b["flips"],
                    "largest_flip_distance": b["largest_flip_distance"],
                    "pcg_converged": b["converged"],
                    "within": good_b,
                }
            )
        good = p.returncode == 0 and all(x["within"] for x in batches)
        ok = ok and good
        results.append(
            {
                "case": c.name,
                "backend": args.backend,
                "exit_status": p.returncode,
                "check_lines": lines,
                "batches": batches,
                "flips_beyond_3m": flips_beyond,
                "irlp_results_csv": results_csv,
                "within": good,
            }
        )
        worst = min((x["margin"] for x in batches if x["margin"] is not None), default=None)
        print(
            f"{c.name}: {'within' if good else 'OUTSIDE'} (exit {p.returncode}; smallest margin "
            f"2000 delta / mean = {worst if worst is None else round(worst, 2)}; flips beyond 3m "
            f"in batches {flips_beyond})",
            flush=True,
        )
    if args.json:
        g.write_json(
            args.json,
            {
                "schema": 1,
                "what": "the harmonic oracle on dynG's labels over C (AD7; criterion 11)",
                "reference": g.REFERENCE,
                "date": g.now(),
                "m": margin,
                "within": ok,
                "cases": results,
            },
        )
    print(f"oracle ({args.backend}):", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def margin_args(case: g.Case) -> list[str]:
    return [
        "--nodes",
        str(case.n),
        "--batches",
        str(case.batches),
        "--degree",
        str(case.avg_degree),
        "--seed",
        str(case.seed),
    ]


# --- determinism ----------------------------------------------------------------------------------


def cmd_determinism(args: argparse.Namespace) -> int:
    results, ok = [], True
    for case in C_CASES:
        digests = []
        for run in range(2):
            with tempfile.TemporaryDirectory(prefix="lp-det-", dir=runs_dir()) as t:
                tmp = Path(t)
                argv = [*case.argv(tmp / "out"), "--backend", args.backend]
                argv += ["--label-dump", str(tmp / "dump")]
                run_compat(args.exe, argv, clean_env())
                digests.append(g.sha256_bytes((tmp / "dump" / "labels.f32").read_bytes()))
            del run
        required = case.schedule == "damped" or args.backend == "sequential"
        same = digests[0] == digests[1]
        good = same or not required
        ok = ok and good
        results.append(
            {"case": case.name, "digests": digests, "identical": same, "required": required}
        )
        print(f"{case.name}: {'identical' if same else 'different'} label bits over two runs "
              f"({'required' if required else 'reported'})", flush=True)  # fmt: skip
    if args.json:
        g.write_json(
            args.json,
            {
                "schema": 1,
                "what": "run-to-run label bits (criterion 11)",
                "date": g.now(),
                "backend": args.backend,
                "equal": ok,
                "cases": results,
            },
        )
    print("determinism:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# --- damped (AD6: the fused engine's byte parity) -------------------------------------------------

# Every field the golden trace holds for a batch, including the propagation's (the labels' bits,
# the counts, the tags): the fused engine reproduces all of them.
FUSED_TRACE_FIELDS = {
    **TRACE_FIELDS,
    "connected_components": [*TRACE_FIELDS["connected_components"], "hook_rounds"],
    "propagation_begin": [*TRACE_FIELDS["propagation_begin"], "labels", "next_tag"],
    "propagation_end": [
        "iterations",
        "frontier_visits",
        "persistent_iterations",
        "next_tag",
        "labels",
    ],
}
RESULT_COLUMNS = ("iterations", "frontier_visits", "components", "accuracy")


def build_preset(exe: Path) -> dict:
    """The preset and the CUDA flags of the build that holds `exe` (its CMakeCache.txt)."""
    for d in [exe.parent, *exe.parents]:
        cache = d / "CMakeCache.txt"
        if cache.exists():
            out = {"build": str(d)}
            for line in cache.read_text().splitlines():
                for key in (
                    "CMAKE_BUILD_TYPE",
                    "CMAKE_CUDA_FLAGS_RELEASE",
                    "DYNG_CUDA_ARCHITECTURES",
                ):
                    if line.startswith(key + ":"):
                        out[key] = line.split("=", 1)[1]
            return out
    return {}


def parity_cuda_preset(preset: dict) -> bool:
    """Whether the build is the parity-cuda preset (Release, -O3 -fmad=true, sm_86)."""
    flags = preset.get("CMAKE_CUDA_FLAGS_RELEASE", "")
    return (
        preset.get("CMAKE_BUILD_TYPE") == "Release"
        and "-O3" in flags
        and "-fmad=true" in flags
        and preset.get("DYNG_CUDA_ARCHITECTURES") == "86"
    )


def results_columns(csv_text: str) -> list[dict]:
    lines = [x for x in csv_text.splitlines() if x]
    head = lines[0].split(",")
    return [
        {k: v for k, v in zip(head, row.split(","), strict=False) if k in RESULT_COLUMNS}
        for row in lines[1:]
    ]


def cmd_damped(args: argparse.Namespace) -> int:
    """compare.py label_propagation damped --config cuda-fused: P+ (AD6)."""
    preset = build_preset(args.exe)
    if args.config == "cuda-fused" and not parity_cuda_preset(preset):
        raise g.GoldenError(f"the cuda-fused replay needs a parity-cuda build, got {preset}")
    damped_root = g.set_dir(g.scratch() / "goldens", "damped")
    trace_root = g.set_dir(g.scratch() / "goldens", "trace")
    results, ok = [], True
    for case in select(g.P_CASES, args.cases):
        import time

        gold = damped_root / "cases" / case.name
        expected = json.loads((gold / "results.json").read_text())
        with tempfile.TemporaryDirectory(prefix="lp-damped-", dir=runs_dir()) as t:
            tmp = Path(t)
            t0 = time.monotonic()
            # The timed-like run: the labels' bytes, the per-batch bits, results.csv.
            argv = [*case.argv(tmp / "out"), "--backend", "cuda", "--label-dump", str(tmp / "d")]
            run_compat(args.exe, argv, clean_env())
            labels = (tmp / "out" / "dynlp_labels.txt").read_bytes()
            bits = (tmp / "d" / "labels.f32").read_bytes()
            mine = results_columns((tmp / "out" / "results.csv").read_text())
            checks = {}
            gold_labels = g.zstd_read(gold / "dynlp_labels.txt.zst")
            checks["dynlp_labels.txt"] = labels == gold_labels
            n = case.n
            if case.digest:
                want = json.loads((gold / "label_bits.json").read_text())
                got = [
                    g.sha256_bytes(bits[i * 4 * n : (i + 1) * 4 * n])
                    for i in range(len(bits) // (4 * n))
                ]
                checks["label_bits"] = got == want
            else:
                checks["label_bits"] = bits == g.zstd_read(gold / "label_bits.bin.zst")
            theirs = [{k: b[k] for k in RESULT_COLUMNS} for b in expected["batches"]]
            checks["results.csv"] = mine == theirs
            # The traced run: every field of every batch (the trace synchronizes per stage).
            env = clean_env(**({"DYNG_DYNLP_TRACE_DIGEST": "1"} if case.digest else {}))
            argv = [*case.argv(tmp / "out2"), "--backend", "cuda", "--trace", str(tmp / "t")]
            run_compat(args.exe, argv, env)
            ours = dlt.read(tmp / "t" / f"{args.exe.name}.dlt")
            golden = dlt.read(decompressed(trace_root / "cases" / case.name / "trace.dlt.zst", tmp))
            saved = dict(TRACE_FIELDS)
            TRACE_FIELDS.clear()
            TRACE_FIELDS.update(FUSED_TRACE_FIELDS)
            try:
                compared, mismatches = compare_traces(ours, golden)
            finally:
                TRACE_FIELDS.clear()
                TRACE_FIELDS.update(saved)
            checks["trace"] = not mismatches and ours.digest_mode == golden.digest_mode
            seconds = time.monotonic() - t0
        good = all(checks.values())
        ok = ok and good
        results.append(
            {
                "case": case.name,
                "checks": checks,
                "trace_fields": compared,
                "trace_mismatches": mismatches[:20],
                "results_mine": mine if not checks["results.csv"] else "equal",
                "seconds": round(seconds, 1),
                "equal": good,
            }
        )
        print(
            f"{case.name}: {'identical' if good else 'DIFFERENT'} {checks} ({compared} trace "
            f"fields; {seconds:.1f} s)" + (f": {mismatches[:3]}" if mismatches else ""),
            flush=True,
        )
    if args.json:
        g.write_json(
            args.json,
            {
                "schema": 1,
                "what": "the fused CUDA engine's damped byte parity on P+ (AD6; criterion 12): "
                "dynlp_labels.txt bytes against unpatched O, the per-batch label bits against the "
                "twin's, the results.csv columns iterations, frontier_visits, components and "
                "accuracy, and every field of the stage trace",
                "reference": g.REFERENCE,
                "commit": g.COMMIT,
                "config": args.config,
                "build": preset,
                "date": g.now(),
                "equal": ok,
                "cases": results,
            },
        )
    print(f"damped ({args.config}):", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# --- irlp (compute() on CUDA: IrLP's bits and iterations per snapshot) ----------------------------


def cmd_irlp(args: argparse.Namespace) -> int:
    root = g.set_dir(g.scratch() / "goldens", "irlp")
    results, ok = [], True
    for case in g.IRLP_CASES:
        gold = root / "cases" / case.name
        expected = json.loads((gold / "irlp.json").read_text())
        with tempfile.TemporaryDirectory(prefix="lp-irlp-", dir=runs_dir()) as t:
            tmp = Path(t)
            argv = [
                *case.argv(tmp / "out"),
                "--backend",
                args.backend,
                "--label-dump",
                str(tmp / "d"),
            ]
            run_compat(args.exe, argv, clean_env())
            bits = (tmp / "d" / "irlp.f32").read_bytes()
            csv = (tmp / "out" / "results.csv").read_text().splitlines()
            irlp_labels = (tmp / "out" / "irlp_labels.txt").read_bytes()
        head = csv[0].split(",")
        rows = [dict(zip(head, r.split(","), strict=False)) for r in csv[1:] if r]
        n = case.n
        snaps = [bits[i * 4 * n : (i + 1) * 4 * n] for i in range(len(bits) // (4 * n))]
        per = []
        for s in expected["snapshots"]:
            i = s["snapshot"]
            per.append(
                {
                    "snapshot": i,
                    "bits_equal": i < len(snaps) and g.sha256_bytes(snaps[i]) == s["labels_sha256"],
                    "iterations_equal": i < len(rows)
                    and int(rows[i]["irlp_iterations"]) == int(s["iterations"]),
                    "agreement": rows[i]["accuracy_vs_irlp"] if i < len(rows) else None,
                }
            )
        labels_equal = irlp_labels == g.zstd_read(gold / "irlp_labels.txt.zst")
        good = all(x["bits_equal"] and x["iterations_equal"] for x in per) and labels_equal
        ok = ok and good
        results.append(
            {"case": case.name, "snapshots": per, "irlp_labels.txt": labels_equal, "equal": good}
        )
        print(
            f"{case.name}: {'identical' if good else 'DIFFERENT'} "
            f"({sum(x['bits_equal'] for x in per)}/{len(per)} snapshots' bits, "
            f"{sum(x['iterations_equal'] for x in per)} iterations, irlp_labels.txt "
            f"{'equal' if labels_equal else 'different'})",
            flush=True,
        )
    if args.json:
        g.write_json(
            args.json,
            {
                "schema": 1,
                "what": "compute() (IrLP) on CUDA against the twin's per-snapshot bits and "
                "iterations (criterion 12)",
                "reference": g.REFERENCE,
                "commit": g.COMMIT,
                "backend": args.backend,
                "build": build_preset(args.exe),
                "date": g.now(),
                "equal": ok,
                "cases": results,
            },
        )
    print(f"irlp ({args.backend}):", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# --- inplace (AD5: the calibrated cloud, frozen in parity/goldens.toml) --------------------------


def frozen_inplace() -> dict:
    """The thresholds and the iteration envelope frozen in parity/goldens.toml (AD5)."""
    import tomllib

    doc = tomllib.loads((REPO / "parity" / "goldens.toml").read_text())
    return doc["sets"]["label_propagation_inplace"]


def patched_copy() -> Path:
    out = subprocess.check_output(
        [
            str(REPO / "parity" / "build_reference.sh"),
            "--variant",
            "patched",
            "--print-dir",
            g.REFERENCE,
        ],
        text=True,
    )
    return Path(out.strip().splitlines()[-1])


def cmd_inplace(args: argparse.Namespace) -> int:
    frozen = frozen_inplace()
    rule = frozen["rule"]
    root = g.set_dir(g.scratch() / "goldens", "inplace")
    trace_root = g.set_dir(g.scratch() / "goldens", "trace")
    copy = patched_copy()
    results, ok = [], True
    sizes = args.cases or list(g.CALIBRATION["cases"])
    for size in sizes:
        case = g.Case(size, schedule="inplace")
        th = frozen["thresholds"][case.name]
        d = root / "runs" / case.name
        refs = [
            g.parse_labels(g.run_record(d, f"gpu1-run-{i:02d}")["labels"])
            for i in range(int(rule["n_gpu1"]))
        ]
        status = g.status_of(case, copy, 1)
        mask = status == g.VERTEX_ACTIVE
        runs = []
        for k in range(args.runs):
            with tempfile.TemporaryDirectory(prefix="lp-inplace-", dir=runs_dir()) as t:
                tmp = Path(t)
                run_compat(args.exe, [*case.argv(tmp / "out"), "--backend", "cuda"], clean_env())
                labels = g.parse_labels((tmp / "out" / "dynlp_labels.txt").read_bytes())
                cols = g.results_columns((tmp / "out" / "results.csv").read_text())
            pairs = [g.pair_stats(labels, r, mask) for r in refs]
            within = [
                p["mean"] <= th["t_mean"]
                and p["max"] <= th["t_max"]
                and p["flips"] <= th["t_flips"]
                for p in pairs
            ]
            iterations = [int(x["iterations"]) for x in cols]
            components = [int(x["components"]) for x in cols]
            its = g.iterations_inside(th, iterations)
            good_run = all(within) and its["within"] and components == list(th["components"])
            runs.append(
                {
                    "run": k,
                    "largest": {
                        "mean": max(p["mean"] for p in pairs),
                        "max": max(p["max"] for p in pairs),
                        "flips": max(p["flips"] for p in pairs),
                    },
                    "labels_within_every_reference": all(within),
                    "iterations": iterations,
                    "iteration_check": its,
                    "components_exact": components == list(th["components"]),
                    "within": good_run,
                }
            )
        # The deterministic stages (components, G', the initial labels and the initial frontier)
        # do not depend on the schedule: one traced run against the damped trace golden.
        dcase = g.Case(size, digest=size == "5m")
        with tempfile.TemporaryDirectory(prefix="lp-inplace-trace-", dir=runs_dir()) as t:
            tmp = Path(t)
            env = clean_env(**({"DYNG_DYNLP_TRACE_DIGEST": "1"} if dcase.digest else {}))
            argv = [*case.argv(tmp / "out"), "--backend", "cuda", "--trace", str(tmp / "t")]
            run_compat(args.exe, argv, env)
            ours = dlt.read(tmp / "t" / f"{args.exe.name}.dlt")
            golden = dlt.read(
                decompressed(trace_root / "cases" / dcase.name / "trace.dlt.zst", tmp)
            )
            compared, mismatches = compare_traces(ours, golden)
        good = all(r["within"] for r in runs) and not mismatches and len(runs) >= 3
        ok = ok and good
        results.append(
            {
                "case": case.name,
                "thresholds": {k: th[k] for k in ("t_mean", "t_max", "t_flips")},
                "envelope": {k: th[k] for k in ("iterations_lo", "iterations_hi",
                                                "iterations_sum_lo", "iterations_sum_hi")},
                "vertices_compared": int(mask.sum()),
                "runs": runs,
                "deterministic_trace": {"compared": compared, "mismatches": mismatches[:20]},
                "within": good,
            }
        )  # fmt: skip
        print(
            f"{case.name}: {'within' if good else 'OUTSIDE'} ({len(runs)} runs; largest mean "
            f"{max(r['largest']['mean'] for r in runs):.3e} of {th['t_mean']:.3e}, flips "
            f"{max(r['largest']['flips'] for r in runs)} of {th['t_flips']}; iterations "
            f"{[r['iteration_check']['sum'] for r in runs]} in [{th['iterations_sum_lo']}, "
            f"{th['iterations_sum_hi']}]; trace {compared} fields, {len(mismatches)} mismatched)",
            flush=True,
        )
    if args.json:
        g.write_json(
            args.json,
            {
                "schema": 1,
                "what": "CUDA in_place against the frozen calibration (AD5 as decided on "
                "2026-10-28 "
                "and 2026-10-29; criterion 13): every port run against every one of the N GPU-1 "
                "reference runs within T_mean, T_max and T_flips over the vertices ACTIVE and not "
                "ground truth after the last batch, the per-batch iterations and their sum inside "
                "the frozen envelope, the components exact; the deterministic stages (components, "
                "G', the initial labels, the initial frontier) against the trace golden",
                "reference": g.REFERENCE,
                "commit": g.COMMIT,
                "build": build_preset(args.exe),
                "gpu": "GPU 1 (default clocks)",
                "date": g.now(),
                "within": ok,
                "cases": results,
            },
        )
    print("inplace (cuda):", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


COMMANDS = {
    "container": cmd_container,
    "trace": cmd_trace,
    "oracle-port": cmd_oracle_port,
    "determinism": cmd_determinism,
    "damped": cmd_damped,
    "irlp": cmd_irlp,
    "inplace": cmd_inplace,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="compare.py label_propagation", description=__doc__.split("\n\n")[0]
    )
    sub = ap.add_subparsers(dest="command", required=True)
    names = lambda s: [x for x in s.split(",") if x]  # noqa: E731
    cp = sub.add_parser("container")
    cp.add_argument("--exe", type=Path, required=True, help="dyng_lp_replay")
    cp.add_argument("--threads", type=int, default=8)
    cp.add_argument("--backend", choices=["sequential", "cuda"], default="sequential")
    cp.add_argument("--cases", type=names)
    cp.add_argument("--json", type=Path)
    tp = sub.add_parser("trace")
    tp.add_argument("--exe", type=Path, required=True, help="dyng-compat-dynlp")
    tp.add_argument("--backend", choices=["sequential", "cuda"], default="sequential")
    tp.add_argument("--cases", type=names)
    tp.add_argument("--json", type=Path)
    op = sub.add_parser("oracle-port")
    op.add_argument("--exe", type=Path, required=True, help="dyng-compat-dynlp")
    op.add_argument("--replay", type=Path, required=True, help="dyng_lp_replay (the oracle)")
    op.add_argument("--backend", choices=["sequential", "cuda"], default="sequential")
    op.add_argument("--threads", type=int, default=8)
    op.add_argument("--json", type=Path)
    dp = sub.add_parser("determinism")
    dp.add_argument("--exe", type=Path, required=True, help="dyng-compat-dynlp")
    dp.add_argument("--backend", choices=["sequential", "cuda"], default="sequential")
    dp.add_argument("--json", type=Path)
    fp = sub.add_parser("damped")
    fp.add_argument("--exe", type=Path, required=True, help="dyng-compat-dynlp (parity-cuda)")
    fp.add_argument("--config", choices=["cuda-fused"], default="cuda-fused")
    fp.add_argument("--cases", type=names)
    fp.add_argument("--json", type=Path)
    ip = sub.add_parser("irlp")
    ip.add_argument("--exe", type=Path, required=True, help="dyng-compat-dynlp")
    ip.add_argument("--backend", choices=["cuda"], default="cuda")
    ip.add_argument("--json", type=Path)
    np_ = sub.add_parser("inplace")
    np_.add_argument("--exe", type=Path, required=True, help="dyng-compat-dynlp")
    np_.add_argument("--runs", type=int, default=3)
    np_.add_argument("--cases", type=names, help="sizes of I (default: all four)")
    np_.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except g.GoldenError as err:
        print(f"compare.py label_propagation {args.command}: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
