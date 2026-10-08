#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The CBST trace golden sets of ESCHER-GPU@abcf9b2 and MOSP_ESCHER@4b86159 (M8a; ADRs 0013, 0036).

    parity/export_goldens.py cbst_trace <set>|all [--twice] [--no-toml]
    parity/compare.py cbst_trace [--set <set>] [--list] [--store host_model|device]
                       [--policy <preset>] [--fields semantic|offsets|exact] [--record DIR]
    parity/cbst_trace_goldens.py datasets          check the coauth-DBLP variants (registry.toml)
    parity/cbst_trace_goldens.py subset            write the committed subset (cpp/tests/data)

Every set is recorded by the CBST recording twins of the PATCHED scratch copy (parity/exporters/
escher; the originals' unchanged objects relinked with -Xlinker --wrap), with
DYNG_CBST_TRACE_DIR set, from the originals' own test suites and command lines:

  set (goldens.toml section; directory $DYNG_SCRATCH/goldens/cbst_trace/<name>)
  cbst_trace_escher_gpu_test        ESCHER-GPU's `make test` (tests/run_tests.sh): every invocation
                                    that calls the CBST, with its awk-generated inputs
  cbst_trace_escher_gpu_coauth50k   build/main --input <coauth-DBLP, ESCHER variant> --batches 3
                                    --batch-size 50000 --seed 1 --check (digest mode)
  cbst_trace_escher_gpu_syn1m       build/main 1000000 10 1 333333 --batches 3 --batch-size 50000
                                    --seed s, s = 1, 2, 3 (batch dumps only: mode "batches")
  cbst_trace_escher_gpu_targeted
  cbst_trace_mosp_escher_targeted   parity_export/bin/cbst_targeted: the scenarios of
                                    parity/exporters/escher/cbst_targeted.cpp through each
                                    original's unchanged CBSTOperations API (paths the suites
                                    miss: K = 0, D = 0, the reissued maximal key, the three
                                    kernel levels, the serial unfill fallback, growth and
                                    compaction, construct validation)
  cbst_trace_mosp_escher_test       MOSP_ESCHER's `make test` (tests/run_tests.sh), every
                                    CBST-calling invocation except bin/main (see EXCLUDED)
  cbst_trace_mosp_escher_dblp50k_hyperedge
  cbst_trace_mosp_escher_dblp50k_vertex
                                    bin/hsospBench --hg <coauth-DBLP, H-SOSP variant> --kind
                                    hyperedge|vertex --batch 50000 --batches 3 --verify none --out
                                    <csv>, the form of MOSP_ESCHER's results/README.md (seed
                                    20260725 and --maxcard 25 are hsospBench's defaults)

A set directory holds one case per CBST-calling invocation: <case>/trace.cbt.zst (and
trace.blob.zst, the side file), <case>/inputs/ (the input files the invocation named, as the
recorder copied them) and <case>/case.json (arguments, inputs, CBST calls, the trace summary of
parity/escher_trace.py); info.json (the reference, the commands, the suite's verdict lines, the
tools); MANIFEST.sha256. The export fails when the suite or command fails, when an invocation
that called the CBST (the invocation log, DYNG_CBST_INVOCATION_LOG) has no trace, when a trace
does not end normally, or when a trace does not verify (escher_trace.py verify). --twice exports
again from a FRESH archive copy and requires an identical MANIFEST.sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "parity"))

import escher_trace  # noqa: E402

REFS = {
    "ESCHER-GPU": "abcf9b2bd42f896ec3103a6835edf713ed5e416c",
    "MOSP_ESCHER": "4b86159ec0b3bfe6e18c718ff4bef1494d7276a5",
}
# The input of the coauth-DBLP sets: both originals' converters write the same bytes
# (datasets/registry.toml); hsospBench --maxcard 25 then loads 2,466,792 of its 2,467,389 lines.
COAUTH = "datasets/hyper/coauth.hg"
COAUTH_SHA256 = "65fc83b22dfb3ecd14b15436f4aaee54631657f2f4afec8260150679387b7cb6"
COAUTH_ARCHIVE = "datasets/hyper/coauth-DBLP-full.tar.gz"
COAUTH_ARCHIVE_SHA256 = "feb4bed6d3c79a9ba16142e71405b7eff1515504741281b8edfe13dc0f960433"

SETS: dict[str, dict] = {
    "cbst_trace_escher_gpu_test": {
        "reference": "ESCHER-GPU",
        "kind": "suite",
        "mode": "auto",
        "what": "every CBST-calling invocation of tests/run_tests.sh (make test)",
    },
    "cbst_trace_escher_gpu_coauth50k": {
        "reference": "ESCHER-GPU",
        "kind": "commands",
        "mode": "digest",
        "hash_only": ["H2H:rebuild", "H2H:rows"],
        "commands": {
            "coauth_b50k_s1": [
                "build/main",
                "--input",
                "{coauth}",
                "--batches",
                "3",
                "--batch-size",
                "50000",
                "--seed",
                "1",
                "--check",
            ],
        },
        "what": "coauth-DBLP (ESCHER variant), 3 x 50K batches, seed 1, --check",
    },
    "cbst_trace_escher_gpu_syn1m": {
        "reference": "ESCHER-GPU",
        "kind": "commands",
        "mode": "batches",
        "commands": {
            f"syn1m_b50k_s{s}": [
                "build/main",
                "1000000",
                "10",
                "1",
                "333333",
                "--batches",
                "3",
                "--batch-size",
                "50000",
                "--seed",
                str(s),
            ]
            for s in (1, 2, 3)
        },
        "what": "batch dumps of synthetic 1M, 3 x 50K batches, seeds 1-3 (results/README.md)",
    },
    "cbst_trace_escher_gpu_targeted": {
        "reference": "ESCHER-GPU",
        "kind": "commands",
        "mode": "full",
        "commands": {
            "targeted_small": ["parity_export/bin/cbst_targeted", "small"],
            "targeted_large": ["parity_export/bin/cbst_targeted", "large"],
        },
        "what": "the targeted scenarios of parity/exporters/escher/cbst_targeted.cpp",
    },
    "cbst_trace_mosp_escher_test": {
        "reference": "MOSP_ESCHER",
        "kind": "suite",
        "mode": "auto",
        "what": "every CBST-calling invocation of tests/run_tests.sh (make test) but bin/main",
    },
    "cbst_trace_mosp_escher_targeted": {
        "reference": "MOSP_ESCHER",
        "kind": "commands",
        "mode": "full",
        "commands": {
            "targeted_small": ["parity_export/bin/cbst_targeted", "small"],
            "targeted_large": ["parity_export/bin/cbst_targeted", "large"],
        },
        "what": "the targeted scenarios of parity/exporters/escher/cbst_targeted.cpp",
    },
    "cbst_trace_mosp_escher_dblp50k_hyperedge": {
        "reference": "MOSP_ESCHER",
        "kind": "commands",
        "mode": "digest",
        "commands": {
            "dblp_hyperedge_50000": [
                "bin/hsospBench",
                "--hg",
                "{coauth}",
                "--kind",
                "hyperedge",
                "--batch",
                "50000",
                "--batches",
                "3",
                "--verify",
                "none",
                "--out",
                "hyperedge_50000_r1.csv",
            ],
        },
        "what": "hsospBench --hg coauth-DBLP (H-SOSP variant) --kind hyperedge --batch 50000",
    },
    "cbst_trace_mosp_escher_dblp50k_vertex": {
        "reference": "MOSP_ESCHER",
        "kind": "commands",
        "mode": "digest",
        "commands": {
            "dblp_vertex_50000": [
                "bin/hsospBench",
                "--hg",
                "{coauth}",
                "--kind",
                "vertex",
                "--batch",
                "50000",
                "--batches",
                "3",
                "--verify",
                "none",
                "--out",
                "vertex_50000_r1.csv",
            ],
        },
        "what": "hsospBench --hg coauth-DBLP (H-SOSP variant) --kind vertex --batch 50000",
    },
}
# Invocations of a suite that are recorded but not kept in the set: their output differs from run
# to run in the original itself, so no two exports can be identical.
EXCLUDED = {
    ("MOSP_ESCHER", "main"): "MOSP's demo pipeline draws its graph from std::random_device "
    "(mosp/src/generateGraph.cu:89-90); its CBST calls are DynamicGraph's, which stressTest, "
    "parallelStressTest and test_dynamicgraph_roundtrip also make with fixed seeds",
}
ZSTD = ["zstd", "-q", "-12", "--long=27", "-T8"]


class ExportError(Exception):
    pass


def scratch_default() -> Path:
    return Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))


def portable(path: Path | str) -> str:
    """A path for the committed records (parity/compare.py's portable_path; PLAN 3.4: no personal
    paths in the repository): relative to the repository, or under $DYNG_SCRATCH."""
    p = Path(path).resolve()
    for base, label in [(REPO, None), (scratch_default().resolve(), "$DYNG_SCRATCH")]:
        try:
            rel = p.relative_to(base).as_posix()
        except ValueError:
            continue
        return rel if label is None else f"{label}/{rel}"
    if p.is_relative_to(Path.home()):
        return f"<outside the repository>/{p.name}"
    return p.as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reference_dir(name: str, scratch: Path, fresh: bool) -> Path:
    tool = REPO / "parity" / "build_reference.sh"
    base = [str(tool), "--variant", "patched", "--scratch", str(scratch)]
    subprocess.run([*base, *(["--fresh"] if fresh else []), name], check=True)
    out = subprocess.check_output([*base, "--print-dir", name], text=True)
    return Path(out.strip().splitlines()[-1])


def references() -> dict:
    return {
        r["name"]: r
        for r in tomllib.loads((REPO / "parity" / "references.toml").read_text())["reference"]
    }


def check_coauth(scratch: Path) -> Path:
    path = scratch / COAUTH
    if not path.is_file():
        raise ExportError(f"{path} is missing (parity/cbst_trace_goldens.py datasets)")
    if sha256_file(path) != COAUTH_SHA256:
        raise ExportError(f"{path}: SHA-256 is not the registry's {COAUTH_SHA256}")
    return path


def normalize(arg: str) -> str:
    return "<abs>/" + arg.rsplit("/", 1)[-1] if arg.startswith("/") else arg


def tool_versions() -> dict:
    awk = shutil.which("awk") or "awk"
    out = {
        "awk": os.path.realpath(awk),
        "awk_version": subprocess.run(
            [awk, "--version"], capture_output=True, text=True
        ).stdout.splitlines()[0],
        "zstd": subprocess.run(
            ["zstd", "--version"], capture_output=True, text=True
        ).stdout.strip(),
    }
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
            "-i",
            os.environ.get("CUDA_VISIBLE_DEVICES", "1").split(",")[0],
        ],
        capture_output=True,
        text=True,
    )
    out["gpu"] = gpu.stdout.strip()
    return out


def run_env(ref: dict, raw: Path, spec: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("DYNG_CBST_")}
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")
    for item in ref.get("run_env", []):
        k, _, v = item.partition("=")
        env[k] = v
    env["DYNG_CBST_TRACE_DIR"] = str(raw / "traces")
    env["DYNG_CBST_INVOCATION_LOG"] = str(raw / "invocations")
    env["DYNG_CBST_TRACE_MODE"] = spec["mode"]
    if spec.get("hash_only"):
        env["DYNG_CBST_TRACE_HASH_ONLY"] = ",".join(spec["hash_only"])
    env["TMPDIR"] = str(raw / "tmp")
    (raw / "tmp").mkdir(parents=True, exist_ok=True)
    return env


def read_invocations(d: Path) -> list[dict]:
    out = []
    for f in sorted(d.glob("*.inv")):
        inv = {"file": f.name, "args": [], "calls": None, "exit": None}
        for line in f.read_text(errors="replace").splitlines():
            key, _, value = line.partition("=")
            if key == "arg":
                inv["args"].append(value)
            elif key == "calls":
                inv["calls"] = int(value)
            elif key == "exit":
                inv["exit"] = value
        out.append(inv)
    return out


def trace_header_end(path: Path) -> tuple[dict, dict]:
    head, end = {}, {}
    for kind, rec in escher_trace.Trace(path).records():
        if kind == "header":
            head = rec
        elif kind == "end":
            end = rec
    return head, end


def collect(raw: Path, ref_name: str, case_names: list[str] | None) -> list[dict]:
    """Pairs every CBST-calling invocation with its trace (in order: each twin opens its trace at
    its first CBST call and its invocation log at start-up)."""
    invs = [i for i in read_invocations(raw / "invocations") if (i["calls"] or 0) > 0]
    for i in read_invocations(raw / "invocations"):
        if i["exit"] != "normal":
            raise ExportError(f"invocation {i['file']} {' '.join(i['args'])} did not exit normally")
    traces = sorted((raw / "traces").glob("*.cbt"))
    if len(traces) != len(invs):
        raise ExportError(
            f"{len(invs)} invocations called the CBST but {len(traces)} traces were written"
        )
    cases = []
    for n, (inv, trace) in enumerate(zip(invs, traces, strict=True), start=1):
        head, end = trace_header_end(trace)
        want = [normalize(a) for a in inv["args"]]
        if head.get("argv") != want:
            raise ExportError(f"{trace.name}: argv {head.get('argv')} is not {want}")
        if end.get("exit") != "normal":
            raise ExportError(f"{trace.name}: the trace did not end normally ({end.get('exit')})")
        if end.get("num_ops") != inv["calls"]:
            raise ExportError(
                f"{trace.name}: {end.get('num_ops')} ops recorded, {inv['calls']} CBST calls made"
            )
        exe = inv["args"][0].rsplit("/", 1)[-1]
        name = case_names[n - 1] if case_names else f"{n:03d}-{exe}"
        cases.append(
            {
                "case": name,
                "trace": trace,
                "header": head,
                "end": end,
                "inv": inv,
                "exe": exe,
                "excluded": EXCLUDED.get((ref_name, exe)),
            }
        )
    return cases


def write_case(case: dict, out: Path, raw: Path) -> dict:
    d = out / case["case"]
    d.mkdir(parents=True)
    trace = case["trace"]
    problems = verify(trace)
    if problems:
        raise ExportError(f"{trace.name}: {problems}")
    summary = escher_trace.summarize(trace)
    summary["trace"] = "trace.cbt"
    subprocess.run([*ZSTD, str(trace), "-o", str(d / "trace.cbt.zst")], check=True)
    blob = trace.with_suffix(".blob")
    if blob.is_file():
        subprocess.run([*ZSTD, str(blob), "-o", str(d / "trace.blob.zst")], check=True)
    inputs = []
    for item in case["header"].get("inputs", []):
        idx, base, fnv, size, stored = item.split(" ", 4)
        entry = {"arg": int(idx), "name": base, "fnv1a64": fnv, "bytes": int(size)}
        if stored != "-":
            src = raw / "traces" / stored
            (d / "inputs").mkdir(exist_ok=True)
            shutil.copyfile(src, d / "inputs" / base.replace("/", "_"))
            entry["file"] = "inputs/" + base.replace("/", "_")
            entry["sha256"] = sha256_file(src)
        inputs.append(entry)
    meta = {
        "case": case["case"],
        "argv": case["header"].get("argv"),
        "env": case["header"].get("env"),
        "mode": case["header"].get("mode"),
        "hash_only": case["header"].get("hash_only"),
        "inputs": inputs,
        "cbst_calls": case["inv"]["calls"],
        "trace_bytes": trace.stat().st_size,
        "blob_bytes": blob.stat().st_size if blob.is_file() else 0,
        "trace_sha256": sha256_file(trace),
        "summary": summary,
    }
    (d / "case.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return meta


VERIFIER: Path | None = None  # <patched copy>/parity_export/bin/cbst_trace_tool


def verify(trace: Path) -> list[str]:
    """Rebuilds every state of the trace and checks every hash: with the C++ reader of the
    patched copy's cbst_trace_tool when it is there, else with parity/escher_trace.py."""
    if VERIFIER is not None and VERIFIER.is_file():
        cmd = [str(VERIFIER), "verify", str(trace)]
    else:
        cmd = [sys.executable, str(REPO / "parity" / "escher_trace.py"), "verify", str(trace)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return [] if p.returncode == 0 else (p.stdout + p.stderr).strip().splitlines()[-3:]


def write_manifest(out: Path) -> tuple[str, int, int]:
    lines, size = [], 0
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.sha256":
            lines.append(f"{sha256_file(p)}  {p.relative_to(out).as_posix()}\n")
            size += p.stat().st_size
    text = "".join(lines)
    (out / "MANIFEST.sha256").write_text(text)
    return hashlib.sha256(text.encode()).hexdigest(), len(lines), size


def case_digest(case_dir: Path) -> tuple[str, int]:
    lines = [
        f"{sha256_file(p)}  {p.relative_to(case_dir).as_posix()}\n"
        for p in sorted(case_dir.rglob("*"))
        if p.is_file()
    ]
    return hashlib.sha256("".join(lines).encode()).hexdigest(), len(lines)


def export_set(set_name: str, ref_dir: Path, out: Path, scratch: Path) -> dict:
    global VERIFIER
    VERIFIER = ref_dir / "parity_export" / "bin" / "cbst_trace_tool"
    spec = SETS[set_name]
    ref = references()[spec["reference"]]
    raw = out.parent / (out.name + ".raw")
    shutil.rmtree(raw, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    raw.mkdir(parents=True)
    out.mkdir(parents=True)
    env = run_env(ref, raw, spec)
    info: dict = {
        "set": set_name,
        "reference": spec["reference"],
        "commit": ref["commit"],
        "what": spec["what"],
        "mode": spec["mode"],
        "hash_only": spec.get("hash_only", []),
        "tools": tool_versions(),
    }
    try:
        if spec["kind"] == "suite":
            cmd = ref["smoke_test"]
            p = subprocess.run(
                ["bash", "-c", cmd], cwd=ref_dir, env=env, capture_output=True, text=True
            )
            verdicts = [
                line
                for line in (p.stdout + p.stderr).splitlines()
                if re.match(r"^(PASS|FAIL|XFAIL|XPASS|==|All tests|\d+ test)", line)
            ]
            info["command"] = cmd
            info["verdicts"] = verdicts
            if p.returncode != 0:
                raise ExportError(
                    f"{cmd} failed in {ref_dir} (exit {p.returncode}):\n"
                    + "\n".join((p.stdout + p.stderr).splitlines()[-30:])
                )
            case_names = None
        else:
            coauth = (
                check_coauth(scratch)
                if any("{coauth}" in a for c in spec["commands"].values() for a in c)
                else None
            )
            info["commands"] = {}
            work = raw / "work"
            for name, argv in spec["commands"].items():
                cmd = [a.replace("{coauth}", str(coauth)) if coauth else a for a in argv]
                work.mkdir(exist_ok=True)
                exe = str(ref_dir / cmd[0])
                p = subprocess.run(
                    [exe, *cmd[1:]], cwd=work, env=env, capture_output=True, text=True
                )
                info["commands"][name] = {
                    "argv": [normalize(a) for a in argv],
                    "status": p.returncode,
                    "verdicts": [
                        line
                        for line in (p.stdout + p.stderr).splitlines()
                        if re.search(
                            r"\[check\]|OK|FAIL|Hypergraph:|\[memory\]|"
                            r"^Batch|launches",
                            line,
                        )
                        and "ms" not in line
                    ],
                }
                if p.returncode != 0:
                    raise ExportError(
                        f"{' '.join(cmd)} failed (exit {p.returncode}):\n"
                        + "\n".join((p.stdout + p.stderr).splitlines()[-30:])
                    )
            if coauth:
                info["input"] = {"file": COAUTH, "sha256": COAUTH_SHA256}
            case_names = list(spec["commands"])
        cases = collect(raw, spec["reference"], case_names)
        metas, excluded = [], []
        for c in cases:
            if c["excluded"]:
                s = escher_trace.summarize(c["trace"])
                excluded.append(
                    {
                        "invocation": " ".join(c["header"].get("argv", [])),
                        "reason": c["excluded"],
                        "ops": c["end"].get("num_ops"),
                        "summary_ops": s["ops"],
                    }
                )
                continue
            metas.append(write_case(c, out, raw))
        info["excluded"] = excluded
        info["cases"] = len(metas)
        (out / "info.json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n")
    finally:
        shutil.rmtree(raw, ignore_errors=True)
    sha, n_files, size = write_manifest(out)
    return {"metas": metas, "manifest_sha256": sha, "files": n_files, "bytes": size, "info": info}


def toml_section(set_name: str, out: Path, result: dict) -> str:
    spec = SETS[set_name]
    rows = []
    for meta in sorted(result["metas"], key=lambda m: m["case"]):
        digest, files = case_digest(out / meta["case"])
        ops = sum(meta["summary"]["ops"].values())
        rows.append(
            f'"{meta["case"]}" = {{ ops = {ops}, files = {files}, '
            f'trace_bytes = {meta["trace_bytes"]}, sha256 = "{digest}" }}\n'
        )
    hash_only = ", ".join(f'"{h}"' for h in spec.get("hash_only", []))
    return (
        f"[sets.{set_name}]\n"
        f'reference = "{spec["reference"]}"\n'
        f'commit = "{REFS[spec["reference"]]}"\n'
        f'kind = "cbst_trace"\n'
        f'what = "{spec["what"]}"\n'
        f'location = "$DYNG_SCRATCH/goldens/cbst_trace/{set_name[len("cbst_trace_") :]}"\n'
        f'generated_by = "parity/export_goldens.py cbst_trace (parity/cbst_trace_goldens.py)"\n'
        f'mode = "{spec["mode"]}"\n'
        f"hash_only = [{hash_only}]\n"
        f"num_cases = {len(result['metas'])}\n"
        f"num_files = {result['files']}\n"
        f"num_bytes = {result['bytes']}\n"
        f'manifest_sha256 = "{result["manifest_sha256"]}"\n'
        f"\n[sets.{set_name}.cases]\n" + "".join(rows)
    )


def set_dir(scratch: Path, set_name: str) -> Path:
    return scratch / "goldens" / "cbst_trace" / set_name[len("cbst_trace_") :]


def export_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="export_goldens.py cbst_trace",
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("set", choices=[*SETS, "all"])
    p.add_argument("--scratch", type=Path, default=scratch_default())
    p.add_argument(
        "--twice",
        action="store_true",
        help="export again from a fresh archive copy and require an identical manifest",
    )
    p.add_argument("--no-toml", action="store_true", help="do not write parity/goldens.toml")
    args = p.parse_args(argv)
    names = list(SETS) if args.set == "all" else [args.set]
    import cycle_count_goldens  # the goldens.toml section writer

    for set_name in names:
        spec = SETS[set_name]
        out = set_dir(args.scratch, set_name)
        try:
            ref_dir = reference_dir(spec["reference"], args.scratch, fresh=False)
            result = export_set(set_name, ref_dir, out, args.scratch)
        except ExportError as e:
            print(f"export_goldens cbst_trace {set_name}: {e}", file=sys.stderr)
            return 1
        print(
            f"{set_name}: {len(result['metas'])} cases, {result['files']} files, "
            f"{result['bytes'] / 1e6:.1f} MB, MANIFEST.sha256 {result['manifest_sha256']}",
            flush=True,
        )
        if args.twice:
            with tempfile.TemporaryDirectory(prefix="dyng-goldens-twice-", dir=args.scratch) as t:
                second = Path(t)
                (second / "datasets" / "hyper").mkdir(parents=True)
                if (args.scratch / COAUTH).is_file():
                    os.symlink(args.scratch / COAUTH, second / COAUTH)
                try:
                    ref2 = reference_dir(spec["reference"], second, fresh=True)
                    result2 = export_set(set_name, ref2, set_dir(second, set_name), second)
                except ExportError as e:
                    print(
                        f"export_goldens cbst_trace {set_name} (second export): {e}",
                        file=sys.stderr,
                    )
                    return 1
                if result2["manifest_sha256"] != result["manifest_sha256"]:
                    subprocess.run(
                        [
                            "diff",
                            str(out / "MANIFEST.sha256"),
                            str(set_dir(second, set_name) / "MANIFEST.sha256"),
                        ]
                    )
                    print(f"{set_name}: SECOND EXPORT DIFFERS", file=sys.stderr)
                    return 1
                print(
                    f"{set_name}: second export from a fresh archive copy: identical manifest "
                    f"({result2['manifest_sha256']})",
                    flush=True,
                )
        if not args.no_toml:
            cycle_count_goldens.write_toml(toml_section(set_name, out, result), set_name)
            print(f"wrote the [sets.{set_name}] section of parity/goldens.toml")
    total = sum(
        f.stat().st_size
        for f in (args.scratch / "goldens" / "cbst_trace").rglob("*")
        if f.is_file()
    )
    print(
        f"cbst_trace goldens in total: {total / 1e9:.2f} GB "
        f"({args.scratch / 'goldens' / 'cbst_trace'})"
    )
    return 0


# --- compare: the goldens verify, and the replays through a store (M8a step 2 on) ---------------


def verify_set(set_name: str, scratch: Path) -> list[str]:
    toml = tomllib.loads((REPO / "parity" / "goldens.toml").read_text())
    spec = toml.get("sets", {}).get(set_name)
    out = set_dir(scratch, set_name)
    if spec is None:
        return [f"parity/goldens.toml has no [sets.{set_name}]"]
    if not (out / "MANIFEST.sha256").is_file():
        return [
            f"{out}: no MANIFEST.sha256 (export it: parity/export_goldens.py cbst_trace {set_name})"
        ]
    problems = []
    if (
        hashlib.sha256((out / "MANIFEST.sha256").read_bytes()).hexdigest()
        != spec["manifest_sha256"]
    ):
        problems.append(f"{set_name}: MANIFEST.sha256 differs from goldens.toml")
    for line in (out / "MANIFEST.sha256").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        if sha256_file(out / rel) != digest:
            problems.append(f"{set_name}: {rel} differs from MANIFEST.sha256")
    for case, c in spec.get("cases", {}).items():
        if case_digest(out / case)[0] != c["sha256"]:
            problems.append(f"{set_name}: case {case} differs from goldens.toml")
    return problems


def find_replay_exe(cuda: bool = False) -> Path | None:
    """The replay tool (cpp/tests/parity/escher/cbst_replay_main.cpp): $DYNG_CBST_REPLAY, else the
    first build tree that has it (a Release build replays the large sets fastest); with `cuda`,
    only the CUDA builds, whose tool has the device store (--store device)."""
    env = os.environ.get("DYNG_CBST_REPLAY")
    if env:
        return Path(env)
    presets = (
        ("release-cuda", "parity-cuda", "dev-cuda")
        if cuda
        else ("cpu-only", "release", "dev", "parity", "dev-cuda")
    )
    for preset in presets:
        exe = REPO / "build" / preset / "cpp" / "tests" / "dyng_cbst_replay"
        if exe.is_file():
            return exe
    return None


def allowed_unreplayable(set_name: str, tree: dict) -> bool:
    """Operations a set is documented not to replay: a batches-mode set has no state digests; a
    hash-only rebuild input (goldens.toml hash_only, e.g. H2H:rebuild) has no values."""
    spec = SETS[set_name]
    if spec.get("mode") == "batches":
        return True
    datasets = {h.split(":")[0] for h in spec.get("hash_only", []) if h.endswith(":rebuild")}
    return tree["dataset"] in datasets and "hash-only inputs" in tree["unreplayable_reason"]


def replay_case(
    exe: Path, trace: Path, policy: str, fields: str, tmp: Path, store: str = "host_model"
) -> dict:
    out = tmp / (trace.parent.name + "." + store + ".json")
    cmd = [str(exe), "--store", store, "--fields", fields, "--tmp", str(tmp), "--json", str(out)]
    cmd += ["--max-mismatches", "100000", str(trace)]
    if policy:
        cmd[1:1] = ["--policy", policy]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 1) or not out.is_file():
        return {"case": trace.parent.name, "error": (proc.stderr or proc.stdout).strip()[-2000:]}
    report = json.loads(out.read_text())[0]
    out.unlink()
    report["case"] = trace.parent.name
    return report


def certificate_record(set_name: str, record: dict, args) -> dict:
    """The replay record of a set in the release certificate's format (parity/certify.py
    golden_suites: set, goldens manifest, port commit, configs, matrix, passed): one config,
    device:<preset>/<fields>, and per case "equal" (ESCHER-GPU's sets: no mismatch; MOSP_ESCHER's
    under escher_gpu(): every tree diverges where the host model under the preset does) or
    "DIFFERENT"."""
    head = record["head"]
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--", "cpp"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    config = f"device:{args.policy or original_preset(set_name)}/{args.fields}"
    matrix = {
        c["case"]: {config: "equal" if c.get("ok") and "error" not in c else "DIFFERENT"}
        for c in record["cases"]
    }
    return {
        "set": set_name,
        "date": record["date"],
        "goldens": {"manifest_sha256": record["manifest_sha256"]},
        "port": {"commit": head + ("+dirty" if dirty else "")},
        "executable": record["exe"],
        "reference": SETS[set_name]["reference"],
        "configs": [config],
        "judged_against_host_model": record.get("against_model", False),
        "matrix": matrix,
        "passed": record["passed"],
    }


# The presets of the dynG device store (store_policy::from_name): ESCHER-GPU's, MOSP_ESCHER's and
# dynG's default (M8a step 5, ADR 0039). The staging preset mosp_escher_staged of M8a step 4 is
# accepted for the replay tools of the commits that have it (parity/cbst_merge_steps.sh over the
# merge-step table).
DEVICE_PRESETS = ("escher_gpu", "mosp_escher", "dyng_default", "mosp_escher_staged")


def against_model(set_name: str, args) -> bool:
    """Whether a device replay of a set is judged against the host model under the same preset:
    the traces of an original replayed under another original's preset (MOSP_ESCHER's under
    escher_gpu()), and, for the replay tools of M8a step 4, under the staging preset
    mosp_escher_staged(), which had no automatic compaction yet. A set replayed under its own
    original's preset (MOSP_ESCHER's under mosp_escher() from M8a step 5) is judged against the
    trace itself, and so is a set under dynG's default (dyng_default; field set "policy",
    acceptance criterion 10(c))."""
    if args.policy == "dyng_default":
        return False
    return (args.policy or original_preset(set_name)) != original_preset(set_name)


def original_preset(set_name: str) -> str:
    """The store preset of a set's original: escher_gpu for ESCHER-GPU's, mosp_escher for
    MOSP_ESCHER's (the device store's preset without --policy, from M8a step 5)."""
    return {"ESCHER-GPU": "escher_gpu", "MOSP_ESCHER": "mosp_escher"}[SETS[set_name]["reference"]]


def first_divergences(report: dict) -> dict[int, tuple[int, str] | None]:
    """The first divergence of every tree of a replay report: (op of the tree, field), None for a
    tree without one. The field is the first one the comparator reported at that op other than
    the integrity verdict check.rows, which only a store with its own checker reports (the device
    store; earlier comparators compared it before the state). A construct below its payload is
    one class whether a store reports it as a mismatch of the status (the host model under
    escher_gpu(): ESCHER-GPU's behaviour) or as the documented divergence construct_below_payload
    (the dynG store throws)."""
    first_op: dict[int, int] = {}
    fields: dict[int, list[str]] = {}
    for m in report["mismatches"]:
        t = m["tree"]
        if t not in first_op:
            first_op[t] = m["tree_op"]
        if m["tree_op"] != first_op[t]:
            continue
        field = m["field"]
        if m["op"] == "construct" and field in ("status", "exception.type"):
            field = "construct validation"
        fields.setdefault(t, []).append(field)
    out: dict[int, tuple[int, str] | None] = {t["tree"]: None for t in report["trees"]}
    for t, fs in fields.items():
        named = [f for f in fs if f != "check.rows"]
        out[t] = (first_op[t], (named or fs)[0])
    for d in report.get("documented", []):
        if d.get("rule") != "construct_below_payload":
            continue  # construct_without_row_counts exempts fields, it does not end a tree
        cur = out.get(d["tree"])
        if cur is None or d["tree_op"] < cur[0]:
            out[d["tree"]] = (d["tree_op"], "construct validation")
    return out


# The actual InsertMapping of an op that the store threw (see first_divergence_sets).
THREW_NO_MAPPING = "(none: the op threw)"


def exception_status(actual: str) -> str:
    """The status a replay reports for an op that threw, "exception <type>: <message>", reduced to
    "exception <type>": the exception type is the recorded field (criterion 3, PLAN 6.4.5); the
    message is each store's own wording of the same failure (the device store's
    "dyng: CBST h2v: 27664 payload entries do not fit its capacity of 27655 ...", the host
    model's "fillCBST: payload overflow (27664 > 27655)")."""
    return actual.split(": ", 1)[0]


def first_divergence_sets(report: dict) -> dict[int, tuple[int, frozenset] | None]:
    """The whole first divergence of every tree of a replay report (M8a review): the op of the
    tree and the set of (field, key, expected, actual) of every mismatch the comparator reported
    at that op (the integrity verdict check.rows aside, as in first_divergences); a construct
    below its payload is the one class "construct validation" (see first_divergences). Two stores
    diverge alike only when these are equal: the same op, the same fields, keys and values.

    Two outputs of an op that the store threw are not values of the parity: the exception's
    message (the status is compared as "exception <type>", exception_status) and the
    InsertMapping, which a thrown insert does not return (the device store reports none, the host
    model the zeros of its unfilled output): an out.mapping mismatch of such an op is compared as
    the trace's mapping against THREW_NO_MAPPING. Every other field, key and value at the op is
    compared as reported."""
    first_op: dict[int, int] = {}
    found: dict[int, set] = {}
    validation: set[int] = set()
    threw: set[int] = set()
    for m in report["mismatches"]:
        t = m["tree"]
        if t not in first_op:
            first_op[t] = m["tree_op"]
        if m["tree_op"] != first_op[t] or m["field"] == "check.rows":
            continue
        if m["op"] == "construct" and m["field"] in ("status", "exception.type"):
            validation.add(t)
        actual = str(m.get("actual"))
        if m["field"] == "status" and actual.startswith("exception "):
            threw.add(t)
            actual = exception_status(actual)
        found.setdefault(t, set()).add((m["field"], m.get("key"), str(m.get("expected")), actual))
    out: dict[int, tuple[int, frozenset] | None] = {t["tree"]: None for t in report["trees"]}
    for t, op in first_op.items():
        if t in validation:
            out[t] = (op, frozenset({("construct validation",)}))
            continue
        items = found.get(t, set())
        if t in threw:
            # An op the store threw has no InsertMapping: the trace's is compared with none.
            items = {
                (f, k, e, THREW_NO_MAPPING) if f == "out.mapping" else (f, k, e, a)
                for f, k, e, a in items
            }
        out[t] = (op, frozenset(items))
    for d in report.get("documented", []):
        if d.get("rule") != "construct_below_payload":
            continue
        cur = out.get(d["tree"])
        if cur is None or d["tree_op"] <= cur[0]:
            out[d["tree"]] = (d["tree_op"], frozenset({("construct validation",)}))
    return out


# The documented divergences a record keeps per case (all are counted, by rule).
DOCUMENTED_KEPT = 50


def rule_counts(report: dict) -> dict[str, int]:
    """The documented divergences of a replay report, counted by rule."""
    out: dict[str, int] = {}
    for d in report.get("documented", []):
        out[d["rule"]] = out.get(d["rule"], 0) + 1
    return dict(sorted(out.items()))


def replay_set(set_name: str, args, exe: Path) -> dict:
    """Replays every case of a set through the store; the record of parity/results."""
    import concurrent.futures
    import datetime

    toml = tomllib.loads((REPO / "parity" / "goldens.toml").read_text())
    traces = sorted(set_dir(args.scratch, set_name).glob("*/trace.cbt.zst"))
    (args.scratch / "runs").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cbst-replay-", dir=args.scratch / "runs") as t:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            reports = list(
                pool.map(
                    lambda tr: replay_case(exe, tr, args.policy, args.fields, Path(t), args.store),
                    traces,
                )
            )
            # A trace of the other original replayed under another preset (MOSP_ESCHER's traces
            # under escher_gpu() in M8a step 3) is expected to diverge where the behaviours
            # differ: it must diverge exactly where the host model under the same preset does.
            models = None
            if args.store == "device" and against_model(set_name, args):
                models = list(
                    pool.map(
                        lambda tr: replay_case(
                            exe, tr, args.policy or "escher_gpu", args.fields, Path(t), "host_model"
                        ),
                        traces,
                    )
                )
    cases, bad = [], []
    totals = {
        k: 0
        for k in (
            "ops",
            "compared",
            "unreplayable",
            "mappings",
            "keys_compared",
            "hash_only_rows",
            "growths",
            "compactions",
            "exceptions",
            "mismatch_count",
        )
    }
    totals["documented"] = 0
    # The device store builds every construct both from row counts and by the scan and compares
    # the node arrays byte for byte (M8a step 4); the tool notes how many it built.
    totals["twin_constructs"] = 0
    for i, r in enumerate(reports):
        if "error" in r:
            bad.append(f"{r['case']}: {r['error']}")
            cases.append({"case": r["case"], "error": r["error"]})
            continue
        r["twin_constructs"] = sum(
            int(n.split(": ", 1)[1])
            for n in r.get("notes", [])
            if n.startswith("twin constructs: ")
        )
        for k in totals:
            totals[k] += len(r.get("documented", [])) if k == "documented" else r[k]
        if models is not None:
            m = models[i]
            if "error" in m:
                bad.append(f"{r['case']}: host model: {m['error']}")
                continue
            dev_first, model_first = first_divergences(r), first_divergences(m)
            # The whole first divergence (every field, key and value at that op), not only its
            # op and first field name (M8a review).
            dev_sets, model_sets = first_divergence_sets(r), first_divergence_sets(m)
            same = dev_first == model_first and dev_sets == model_sets
            if not same:
                diff = sorted(
                    t
                    for t in set(dev_sets) | set(model_sets)
                    if (dev_sets.get(t), dev_first.get(t))
                    != (model_sets.get(t), model_first.get(t))
                )

                def shown(sets, t):
                    v = sets.get(t)
                    return None if v is None else (v[0], sorted(map(str, v[1]))[:6])

                bad.append(
                    f"{r['case']}: the first divergences differ from the host model's under "
                    f"{args.policy or 'escher_gpu'} in trees {diff[:8]}: store "
                    f"{[shown(dev_sets, t) for t in diff[:3]]}, model "
                    f"{[shown(model_sets, t) for t in diff[:3]]}"
                )
            # Unreplayable trees are checked here too (M8a review).
            disallowed = [
                t for t in r["trees"] if t["unreplayable"] and not allowed_unreplayable(set_name, t)
            ]
            for tr in disallowed:
                bad.append(
                    f"{r['case']}: tree {tr['tree']} ({tr['dataset']}) unreplayable: "
                    f"{tr['unreplayable_reason']}"
                )
            cases.append(
                {
                    "case": r["case"],
                    "ok": same and not disallowed,
                    "store": r["store"],
                    "exit": r["exit"],
                    **{
                        k: (len(r.get("documented", [])) if k == "documented" else r[k])
                        for k in totals
                    },
                    "first_divergences": {
                        str(t): v for t, v in sorted(dev_first.items()) if v is not None
                    },
                    "documented": r.get("documented", [])[:DOCUMENTED_KEPT],
                    "documented_rules": rule_counts(r),
                }
            )
            continue
        disallowed = [
            t for t in r["trees"] if t["unreplayable"] and not allowed_unreplayable(set_name, t)
        ]
        if not r["ok"]:
            m = r["mismatches"][0] if r["mismatches"] else {}
            bad.append(
                f"{r['case']}: {r['mismatch_count']} mismatches, first at op {m.get('op_index')} "
                f"(tree {m.get('tree')} {m.get('dataset')}, {m.get('op')}) {m.get('field')} key "
                f"{m.get('key')}: trace {m.get('expected')}, store {m.get('actual')}"
            )
        for tr in disallowed:
            bad.append(
                f"{r['case']}: tree {tr['tree']} ({tr['dataset']}) unreplayable: "
                f"{tr['unreplayable_reason']}"
            )
        cases.append(
            {
                "case": r["case"],
                "ok": r["ok"] and not disallowed,
                "store": r["store"],
                "exit": r["exit"],
                **{
                    k: (len(r.get("documented", [])) if k == "documented" else r[k]) for k in totals
                },
                "unreplayable_trees": [
                    {k: tr[k] for k in ("tree", "dataset", "unreplayable", "unreplayable_reason")}
                    for tr in r["trees"]
                    if tr["unreplayable"]
                ],
                "first_mismatches": r["mismatches"][:5],
                "documented": r.get("documented", [])[:DOCUMENTED_KEPT],
                "documented_rules": rule_counts(r),
            }
        )
    spec = toml.get("sets", {}).get(set_name, {})
    documented_rules: dict[str, int] = {}
    for r in reports:
        for rule, n in rule_counts(r).items() if "error" not in r else []:
            documented_rules[rule] = documented_rules.get(rule, 0) + n
    return {
        "set": set_name,
        "documented_rules": dict(sorted(documented_rules.items())),
        "reference": SETS[set_name]["reference"],
        "manifest_sha256": spec.get("manifest_sha256"),
        "store": args.store,
        "policy": args.policy or "the original's preset",
        "fields": args.fields,
        "exe": portable(exe),
        "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "head": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "cases": cases,
        "totals": totals,
        "problems": bad,
        "against_model": models is not None,
        "passed": not bad and bool(traces),
    }


DOCUMENTED_COUNTS = REPO / "parity" / "cbst_documented.toml"


def documented_count_problems(set_name: str, preset: str, got: dict, path: Path) -> list[str]:
    """The documented divergences of a replay against the counts pinned for its set and preset
    (M8a review: a defect that the comparator labels a documented divergence changes the counts,
    and a replay with other counts fails instead of passing)."""
    table = tomllib.loads(path.read_text())
    want = table.get(set_name, {}).get(preset)
    if want is None:
        return [
            f"no documented-divergence counts pinned for {set_name} under {preset} in "
            f"{portable(path)} (the replay found {got or 'none'})"
        ]
    if dict(sorted(want.items())) != dict(sorted(got.items())):
        return [
            f"documented divergences under {preset}: {got or 'none'}, pinned "
            f"{dict(sorted(want.items())) or 'none'} ({portable(path)})"
        ]
    return []


def compare_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="compare.py cbst_trace",
        description="Verify the CBST trace goldens and, with --store, replay them through a store "
        "(M8a; ADR 0037): every case of every set, under the original's preset (or --policy), "
        "comparing --fields; a JSON record per set with --record.",
    )
    p.add_argument("--set", choices=list(SETS), action="append")
    p.add_argument("--scratch", type=Path, default=scratch_default())
    p.add_argument("--list", action="store_true", help="list the sets and their cases")
    p.add_argument(
        "--verify-traces",
        action="store_true",
        help="also rebuild every trace (escher_trace.py verify)",
    )
    p.add_argument(
        "--store",
        choices=["host_model", "device"],
        help="replay through this store (host_model: dyng::testing::cbst_host_model; device: the "
        "dynG store, from M8a step 3)",
    )
    p.add_argument(
        "--policy",
        choices=["escher_gpu", "mosp_escher", "dyng_default", "mosp_escher_staged"],
        default="",
        help="the store's policy (default: the preset of each trace's original)",
    )
    p.add_argument(
        "--fields",
        choices=["semantic", "offsets", "exact", "policy"],
        default=None,
        help="the compared fields (default: exact; policy under --policy dyng_default, M8a "
        "criterion 10(c))",
    )
    p.add_argument("--replay-exe", type=Path, help="dyng_cbst_replay (default: find it in build/)")
    p.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 4)))
    p.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="the GPU of --store device (CUDA_VISIBLE_DEVICES, unless already set; default 1, the "
        "development GPU)",
    )
    p.add_argument("--record", type=Path, help="write M8a-<store>-<set>-<fields>.json here")
    p.add_argument(
        "--certificate",
        type=Path,
        help="with --store device: also write the release certificate's replay record "
        "parity-<set>-device-<preset>.json here (parity/certify.py, which requires the original's "
        "preset and dyng_default for every set)",
    )
    p.add_argument(
        "--documented-counts",
        default=str(DOCUMENTED_COUNTS),
        help="with --store device: the pinned documented-divergence counts per set, preset and "
        "rule (default parity/cbst_documented.toml; 'none' for the store of another commit, as "
        "parity/cbst_merge_steps.sh passes for every commit but the head)",
    )
    p.add_argument("--skip-verify", action="store_true", help="do not re-hash the goldens")
    args = p.parse_args(argv)
    if args.fields is None:
        args.fields = "policy" if args.policy == "dyng_default" else "exact"
    names = args.set or list(SETS)
    if args.list:
        toml = tomllib.loads((REPO / "parity" / "goldens.toml").read_text())
        for n in names:
            s = toml.get("sets", {}).get(n)
            print(
                f"{n}: {SETS[n]['reference']} {SETS[n]['what']}: "
                + (
                    f"{s['num_cases']} cases, {s['num_bytes'] / 1e6:.1f} MB"
                    if s
                    else "not exported"
                )
            )
        return 0
    exe = None
    if args.store == "device":
        if args.policy not in ("",) + DEVICE_PRESETS:
            print(
                "compare.py cbst_trace --store device: the store has the presets "
                + ", ".join(DEVICE_PRESETS)
                + " (M8a step 5)",
                file=sys.stderr,
            )
            return 2
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    if args.store is not None:
        exe = args.replay_exe or find_replay_exe(cuda=args.store == "device")
        if exe is None or not exe.is_file():
            print(
                "dyng_cbst_replay not found: build it (cmake --build <build> --target "
                "dyng_cbst_replay) or pass --replay-exe",
                file=sys.stderr,
            )
            return 2
    missing = [n for n in names if not (set_dir(args.scratch, n) / "MANIFEST.sha256").is_file()]
    if missing and len(missing) == len(names):
        print(
            f"no CBST trace goldens under {args.scratch / 'goldens' / 'cbst_trace'} "
            "(parity/export_goldens.py cbst_trace all); skipping",
            file=sys.stderr,
        )
        return 77
    bad = 0
    for n in names:
        problems = [] if args.skip_verify else verify_set(n, args.scratch)
        if args.verify_traces and not problems:
            with tempfile.TemporaryDirectory(dir=args.scratch / "runs") as t:
                for zst in sorted(set_dir(args.scratch, n).glob("*/trace.cbt.zst")):
                    plain = Path(t) / "trace.cbt"
                    subprocess.run(
                        ["zstd", "-q", "-d", "--long=31", "-f", str(zst), "-o", str(plain)],
                        check=True,
                    )
                    blob = zst.with_name("trace.blob.zst")
                    if blob.is_file():
                        subprocess.run(
                            [
                                "zstd",
                                "-q",
                                "-d",
                                "--long=31",
                                "-f",
                                str(blob),
                                "-o",
                                str(Path(t) / "trace.blob"),
                            ],
                            check=True,
                        )
                    problems += [f"{zst.parent.name}: {x}" for x in verify(plain)]
                    for f in Path(t).iterdir():
                        f.unlink()
        record = None
        if exe is not None and not problems:
            record = replay_set(n, args, exe)
            problems += record["problems"]
            if args.store == "device" and args.documented_counts != "none":
                pinned = documented_count_problems(
                    n,
                    args.policy or original_preset(n),
                    record["documented_rules"],
                    Path(args.documented_counts),
                )
                problems += pinned
                record["problems"] += pinned
                record["passed"] = record["passed"] and not pinned
            t = record["totals"]
            print(
                f"{n}: replay through {args.store} ({record['policy']}, {args.fields}): "
                f"{len(record['cases'])} traces, {t['ops']} ops, {t['compared']} compared, "
                f"{t['unreplayable']} unreplayable, {t['mappings']} mappings, "
                f"{t['keys_compared']} key states, {t['hash_only_rows']} hash-only rows, "
                f"{t['growths']} growths, {t['compactions']} compactions, {t['exceptions']} "
                f"exceptions, {t['mismatch_count']} mismatches, {t['documented']} documented "
                f"divergences, {t['twin_constructs']} twin constructs"
                + (
                    " (judged against the host model under the same preset)"
                    if record.get("against_model")
                    else ""
                )
            )
            if args.record:
                args.record.mkdir(parents=True, exist_ok=True)
                name = f"M8a-{args.store.replace('_', '-')}-{n[len('cbst_trace_') :]}-{args.fields}"
                if args.policy:
                    name += f"-{args.policy}"
                (args.record / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n")
            if args.certificate and args.store == "device":
                args.certificate.mkdir(parents=True, exist_ok=True)
                doc = certificate_record(n, record, args)
                preset = args.policy or original_preset(n)
                path = args.certificate / f"parity-{n}-device-{preset}.json"
                path.write_text(json.dumps(doc, indent=1) + "\n")
                print(f"{n}: wrote {path}")
        print(f"{n}: " + ("OK" if not problems else "FAILED"))
        for pr in problems:
            print("  " + pr)
        bad += bool(problems)
    return 1 if bad else 0


# --- datasets and the committed subset ----------------------------------------------------------


def datasets_main(argv: list[str]) -> int:
    """Re-derives both coauth-DBLP variants with the originals' own converters in the scratch
    copies and checks them against datasets/registry.toml."""
    p = argparse.ArgumentParser(prog="cbst_trace_goldens.py datasets")
    p.add_argument("--scratch", type=Path, default=scratch_default())
    args = p.parse_args(argv)
    reg = tomllib.loads((REPO / "datasets" / "registry.toml").read_text())
    by = {d["name"]: d for d in reg["dataset"]}
    archive = args.scratch / COAUTH_ARCHIVE
    if sha256_file(archive) != by["coauth-DBLP-full"]["sha256"]:
        print(f"{archive}: SHA-256 is not the registry's", file=sys.stderr)
        return 1
    ok = True
    with tempfile.TemporaryDirectory(dir=args.scratch / "runs") as t:
        subprocess.run(["tar", "-xzf", str(archive), "-C", t], check=True)
        prefix = Path(t) / "coauth-DBLP-full" / "coauth-DBLP-full"
        for name in ("coauth-DBLP-escher", "coauth-DBLP-hsosp"):
            d = by[name]
            ref = reference_dir(d["reference"], args.scratch, fresh=False)
            script = ref / d["converter"]
            out = Path(t) / f"{name}.hg"
            subprocess.run([sys.executable, str(script), str(prefix), str(out)], check=True)
            got = sha256_file(out)
            print(
                f"{name}: {d['converter']} of {d['reference']} -> {got}"
                f" ({'OK' if got == d['sha256'] else 'DIFFERS from ' + d['sha256']})"
            )
            ok &= got == d["sha256"]
            out.unlink()
    installed = args.scratch / COAUTH
    got = sha256_file(installed)
    print(f"{installed}: {got} ({'OK' if got == COAUTH_SHA256 else 'DIFFERS'})")
    return 0 if ok and got == COAUTH_SHA256 else 1


# The committed subset (cpp/tests/data/goldens/cbst_trace): small traces that let the CPU replay
# of hosted CI cover reuse, surplus with rebuild, fill overflow, unfill on chained rows and
# MOSP_ESCHER's compaction inside fill and inside insert (set, case, why).
SUBSET = [
    ("cbst_trace_escher_gpu_test", "001-main", "A: the demo run (reuse, surplus with rebuild)"),
    (
        "cbst_trace_escher_gpu_targeted",
        "targeted_small",
        "A: K = 0, D = 0, reuse only, reuse "
        "with surplus and rebuild, the reissued maximal key, fill overflow with growth, construct "
        "below the payload size",
    ),
    ("cbst_trace_mosp_escher_test", "004-test_cbst_ops", "B: test_cbst_ops terminator"),
    ("cbst_trace_mosp_escher_test", "005-test_cbst_ops", "B: test_cbst_ops erase"),
    (
        "cbst_trace_mosp_escher_test",
        "006-test_cbst_ops",
        "B: test_cbst_ops surplus (fill overflow, surplus with rebuild)",
    ),
    ("cbst_trace_mosp_escher_test", "007-test_cbst_ops", "B: test_cbst_ops bestfit (reuse)"),
    (
        "cbst_trace_mosp_escher_test",
        "008-test_cbst_ops",
        "B: test_cbst_ops unfill-chain (unfill on chained rows)",
    ),
    (
        "cbst_trace_mosp_escher_targeted",
        "targeted_small",
        "B: compaction inside fill and inside "
        "insert, the capacity throw, an explicit compact(), construct validation",
    ),
]
SUBSET_DIR = REPO / "cpp" / "tests" / "data" / "goldens" / "cbst_trace"
SUBSET_LIMIT = 320_000


def subset_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="cbst_trace_goldens.py subset")
    p.add_argument("--scratch", type=Path, default=scratch_default())
    p.add_argument(
        "--extra",
        action="append",
        default=[],
        help="set:case[:file name] to add (the compaction traces, see SUBSET_EXTRA)",
    )
    args = p.parse_args(argv)
    shutil.rmtree(SUBSET_DIR, ignore_errors=True)
    SUBSET_DIR.mkdir(parents=True)
    rows = []
    items = list(SUBSET) + [
        tuple(e.split(":", 2)) + ("",) * (3 - len(e.split(":", 2))) for e in args.extra
    ]
    for set_name, case, why in items:
        src = set_dir(args.scratch, set_name) / case
        dst = SUBSET_DIR / set_name[len("cbst_trace_") :] / f"{case}.cbt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["zstd", "-q", "-d", "--long=31", "-f", str(src / "trace.cbt.zst"), "-o", str(dst)],
            check=True,
        )
        if (src / "trace.blob.zst").exists():
            print(
                f"{set_name}/{case}: has a side file; not suitable for the subset", file=sys.stderr
            )
            return 1
        meta = json.loads((src / "case.json").read_text())
        if sha256_file(dst) != meta["trace_sha256"]:
            print(f"{dst}: not the exported trace", file=sys.stderr)
            return 1
        rows.append(
            f"{sha256_file(dst)}  {dst.relative_to(SUBSET_DIR).as_posix()}  "
            f"{' '.join(meta['argv'])}  # {why}\n"
        )
    total = sum(f.stat().st_size for f in SUBSET_DIR.rglob("*.cbt"))
    (SUBSET_DIR / "SUBSET.txt").write_text(
        "# The committed subset of the CBST trace goldens (parity/cbst_trace_goldens.py subset):\n"
        "# <sha256>  <file>  <command of the original>  # what it covers. Each file is the\n"
        "# decompressed trace.cbt.zst of the case in $DYNG_SCRATCH/goldens/cbst_trace.\n"
        + "".join(rows)
    )
    print(f"wrote {len(rows)} traces, {total} bytes, into {SUBSET_DIR}")
    if total > SUBSET_LIMIT:
        print(f"the subset is larger than {SUBSET_LIMIT} bytes", file=sys.stderr)
        return 1
    return 0


def coverage_main(argv: list[str]) -> int:
    """The coverage table of parity/results/M8a.md from the case summaries of every exported set
    (the paths of PLAN 6.4.5 and the M8a step 1 list; counts are of ops, or of items for the
    kernel levels)."""
    p = argparse.ArgumentParser(prog="cbst_trace_goldens.py coverage")
    p.add_argument("--scratch", type=Path, default=scratch_default())
    args = p.parse_args(argv)
    cols = [
        "reuse + surplus + rebuild",
        "reuse only",
        "K = 0",
        "D = 0",
        "fill overflow + growth",
        "unfill fallback on chained rows",
        "fill T/W/B",
        "reuse T/W/B",
        "unfill T/W/B",
        "compaction in fill",
        "compaction in insert",
        "compact()",
        "capacity throw",
        "reissued dead max key",
        "max chain hops",
    ]
    print("| set | " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * len(cols))
    for set_name in SETS:
        d = set_dir(args.scratch, set_name)
        metas = [json.loads(f.read_text()) for f in sorted(d.glob("*/case.json"))]
        if not metas:
            continue

        def tot(path, metas=metas):
            out = 0
            for m in metas:
                v = m["summary"]
                for k in path:
                    v = v[k]
                out += v
            return out

        def lv(which, metas=metas):
            return "/".join(
                str(sum(m["summary"]["levels"][which][i] for m in metas)) for i in range(3)
            )

        row = [
            tot(["insert", "reuse_surplus_rebuild"]),
            tot(["insert", "reuse_only"]),
            tot(["insert", "k0"]),
            tot(["insert", "d0"]),
            tot(["fill", "overflow_growth"]),
            tot(["unfill", "chained_fallback"]),
            lv("fill"),
            lv("reuse"),
            lv("unfill"),
            tot(["fill", "compaction"]),
            tot(["insert_compaction"]),
            tot(["compact_calls"]),
            tot(["capacity_throws"]),
            tot(["insert", "reissued_dead_key"]),
            max(m["summary"]["max_chain_hops"] for m in metas),
        ]
        print(f"| {set_name[len('cbst_trace_') :]} | " + " | ".join(map(str, row)) + " |")
    return 0


# Invocations of a suite that are recorded and replayed once instead of kept as a golden set
# (EXCLUDED: their inputs differ from run to run): `cbst_trace_goldens.py once`.
ONCE = {"main": ("MOSP_ESCHER", ["bin/main"], "mosp_escher")}


def once_main(argv: list[str]) -> int:
    """Records one run of an excluded invocation (EXCLUDED) of an original's patched copy and
    replays its trace right away through the dynG store under the original's preset, every field
    compared (M8a review: MOSP_ESCHER's bin/main, whose graph comes from std::random_device, is
    not a golden set, but its CBST operations are replayed; the trace carries the original's own
    states, so one run checks itself). Not a golden: nothing is kept but the summary."""
    p = argparse.ArgumentParser(
        prog="cbst_trace_goldens.py once",
        description=once_main.__doc__,
    )
    p.add_argument("invocation", choices=sorted(ONCE))
    p.add_argument("--scratch", type=Path, default=scratch_default())
    p.add_argument("--replay-exe", type=Path, help="dyng_cbst_replay (default: find it in build/)")
    p.add_argument("--gpu", type=int, default=1, help="CUDA_VISIBLE_DEVICES (default 1)")
    p.add_argument("--json", type=Path, help="write the replay report here")
    args = p.parse_args(argv)
    repo, argv_rel, preset = ONCE[args.invocation]
    ref_dir = reference_dir(repo, args.scratch, False)
    exe = args.replay_exe or find_replay_exe(cuda=True)
    if exe is None or not exe.is_file():
        print("dyng_cbst_replay (a CUDA build) not found: pass --replay-exe", file=sys.stderr)
        return 2
    global VERIFIER
    VERIFIER = ref_dir / "parity_export" / "bin" / "cbst_trace_tool"
    ref = references()[repo]
    (args.scratch / "runs").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cbst-once-", dir=args.scratch / "runs") as t:
        raw = Path(t)
        env = run_env(ref, raw, {"mode": "auto"})
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        work = raw / "work"
        work.mkdir()
        run = subprocess.run(
            [str(ref_dir / argv_rel[0]), *argv_rel[1:]],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
        )
        if run.returncode != 0:
            print(f"{argv_rel[0]} failed (exit {run.returncode}):", file=sys.stderr)
            print("\n".join((run.stdout + run.stderr).splitlines()[-20:]), file=sys.stderr)
            return 1
        cases = collect(raw, repo, None)
        if not cases:
            print(f"{argv_rel[0]}: no CBST call was recorded", file=sys.stderr)
            return 1
        bad = 0
        reports = []
        for c in cases:
            problems = verify(c["trace"])
            report = replay_case(exe, c["trace"], preset, "exact", raw, "device")
            reports.append(report)
            if "error" in report:
                problems.append(report["error"])
            elif not report["ok"]:
                m = report["mismatches"][0] if report["mismatches"] else {}
                problems.append(
                    f"{report['mismatch_count']} mismatches, first at op {m.get('op_index')} "
                    f"(tree {m.get('tree')} {m.get('dataset')}, {m.get('op')}) {m.get('field')} "
                    f"key {m.get('key')}: trace {m.get('expected')}, store {m.get('actual')}"
                )
            unreplayable = [tr for tr in report.get("trees", []) if tr["unreplayable"]]
            problems += [
                f"tree {tr['tree']} unreplayable: {tr['unreplayable_reason']}"
                for tr in unreplayable
            ]
            ops = c["end"].get("num_ops")
            rules = rule_counts(report) if "error" not in report else {}
            print(
                f"{repo} {' '.join(argv_rel)} (recorded once, replayed under {preset}, exact): "
                f"{ops} ops, {report.get('compared', 0)} compared, "
                f"{report.get('mismatch_count', '?')} mismatches, documented {rules or 'none'}: "
                + ("OK" if not problems else "FAILED")
            )
            for pr in problems:
                print("  " + pr)
            bad += bool(problems)
        if args.json:
            args.json.write_text(json.dumps(reports, indent=1) + "\n")
    return 1 if bad else 0


def main() -> int:
    if sys.argv[1:2] == ["once"]:
        return once_main(sys.argv[2:])
    if sys.argv[1:2] == ["coverage"]:
        return coverage_main(sys.argv[2:])
    if sys.argv[1:2] == ["datasets"]:
        return datasets_main(sys.argv[2:])
    if sys.argv[1:2] == ["subset"]:
        return subset_main(sys.argv[2:])
    if sys.argv[1:2] == ["compare"]:
        return compare_main(sys.argv[2:])
    return export_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
