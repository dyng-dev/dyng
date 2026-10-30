#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The certificate records of label_propagation's golden replays (M10a step 5; ADR 0078).

    compare.py label_propagation certificate --kind KIND --from RECORD [--from RECORD ...]
                                             --out DIR

converts the --json records of `compare.py label_propagation generator|trace|damped|hub|irlp|
inplace|oracle-port` (each stamped with the port's commit, executable and build when it was
written: stamp_port()) into parity/certify.py's replay format: one record per set and backend,
`DIR/parity-label_propagation_<kind>-<backend>.json`, with the set, its golden manifest's SHA-256
(the goldens root's MANIFEST.sha256; the oracle has no manifest, its coverage table names its
cases), the port's commit and build, one configuration `<backend>:lp/<kind>` (damped and hub:
`cuda-fused:lp/<kind>`), the case x configuration matrix ("equal" or "DIFFERENT"), the checks and,
for the calibrated kinds (inplace, oracle), the tolerance the replay applied, which certify.py
recomputes from the committed calibration record (frozen_tolerance_problems).

KIND is one of KINDS. A record without its port stamp, or of another kind, is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "parity"))

KINDS = ("generator", "trace", "damped", "hub", "irlp", "inplace", "oracle")
WHAT = {
    "generator": "generators::legacy::dynlp_vertex_arrival_stream against the golden set (the "
    "arrays and O's text files byte for byte, 5m / 5m1 / 50m by SHA-256)",
    "trace": "the deterministic stages of label_propagation (statuses, rows, G', components, "
    "W0 / W1, the initial labels, V_aff) against O's trace golden, every batch",
    "damped": "the fused engine's damped byte parity: dynlp_labels.txt, every batch's label bits, "
    "the results.csv counts and every trace field (a parity-cuda build)",
    "hub": "propagation_test's hub graph on the fused engine: the damped calls' label bits and "
    "counts, the in_place call within O's bounds (a parity-cuda build)",
    "irlp": "compute() (IrLP) on CUDA: every snapshot's label bits and iterations",
    "inplace": "CUDA in_place against the frozen calibration (AD5): every port run against every "
    "reference run within T_mean, T_max, T_flips, the iteration envelope, components and V_aff",
    "oracle": "the harmonic oracle (AD7): mean |F - F*| <= 2000 delta, the PCG converged, no "
    "class flip where |F* - 0.5| > 3m, on every batch of C (and L), both schedules, IrLP",
}


class CertError(Exception):
    pass


def goldens_root() -> Path:
    return Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work")) / "goldens"


def build_preset(exe: Path) -> dict:
    import label_propagation_port as port

    return port.build_preset(Path(exe))


def stamp_port(json_path: Path | None, exe: Path | None) -> None:
    """Adds the port's commit, executable and build to a replay's --json record (called by the
    replays' main after the command wrote it)."""
    if json_path is None or exe is None or not Path(json_path).is_file():
        return
    import perf_ab

    doc = json.loads(Path(json_path).read_text())
    preset = build_preset(Path(exe))
    doc["port"] = {
        "commit": perf_ab.port_commit(),
        "executable": perf_ab.portable_path(Path(exe).resolve()),
        "build": preset,
        "parity_cuda": parity_cuda(preset),
    }
    Path(json_path).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def parity_cuda(preset: dict) -> bool:
    import label_propagation_port as port

    return port.parity_cuda_preset(preset)


def manifest_sha256(kind: str) -> str | None:
    path = goldens_root() / f"label_propagation_{kind}" / "MANIFEST.sha256"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def backend_of(kind: str, doc: dict) -> str:
    if kind in ("damped", "hub"):
        return "cuda-fused"
    if kind == "generator":
        return "sequential"
    if kind in ("irlp", "inplace"):
        return doc.get("backend", "cuda")
    backends = {c.get("backend") for c in doc.get("cases", [])}
    if len(backends) != 1:
        raise CertError(f"a {kind} record must be of one backend, not {sorted(map(str, backends))}")
    return backends.pop()


def verdicts(kind: str, doc: dict) -> dict[str, bool]:
    """{case: passed} of a replay record."""
    if kind == "hub":
        return {"hub": str(doc.get("equal")) == "True"}
    out = {}
    for c in doc.get("cases", []):
        if kind == "generator":
            good = all(
                r["arrays_equal"] and r.get("text_files_equal") is not False for r in c["runs"]
            ) and c.get("same_stream_for_every_run")
        elif kind in ("inplace", "oracle"):
            good = c.get("within") is True
        else:
            good = c.get("equal") is True
        out[c["case"]] = bool(good)
    return out


def tolerance_of(kind: str, doc: dict) -> dict | None:
    if kind == "inplace":
        return {
            c["case"]: {**c["thresholds"], **c["envelope"]}
            for c in doc.get("cases", [])
            if "thresholds" in c and "envelope" in c
        }
    if kind == "oracle":
        return {"bound_factor": 2000, "class_margin_m": doc.get("m"), "class_margin_k": 3}
    return None


def certificate_record(kind: str, doc: dict) -> dict:
    if kind not in KINDS:
        raise CertError(f"unknown kind {kind!r} (known: {', '.join(KINDS)})")
    port = doc.get("port")
    if not port or not port.get("commit"):
        raise CertError("the record has no port stamp (its commit and build): replay it again")
    backend = backend_of(kind, doc)
    config = f"{backend}:lp/{kind}"
    cases = verdicts(kind, doc)
    rec = {
        "schema": 1,
        "set": f"label_propagation_{kind}",
        "kind": f"label_propagation_{kind}",
        "what": f"parity/compare.py label_propagation: {WHAT[kind]}",
        "date": doc.get("date"),
        "reference": {"name": "LabelPropagation-CUDA", "commit": doc.get("commit")},
        "port": {"commit": port["commit"]},
        "executable": port.get("executable"),
        "build": {
            **(port.get("build") or {}),
            "preset": "parity-cuda" if port.get("parity_cuda") else "other",
        },  # fmt: skip
        "goldens": {"manifest_sha256": manifest_sha256(kind)} if kind != "oracle" else {},
        "backend": backend,
        "configs": [config],
        "cases": sorted(cases),
        "matrix": {c: {config: "equal" if ok else "DIFFERENT"} for c, ok in sorted(cases.items())},
        "checks": {"replay": kind},
        "passed": bool(cases) and all(cases.values()),
    }
    tol = tolerance_of(kind, doc)
    if tol is not None:
        rec["tolerance"] = tol
    return rec


def write_certificate(kind: str, docs: list[dict], out: Path) -> Path:
    """One record of the set and backend from one or more replay records (merged by case)."""
    recs = [certificate_record(kind, d) for d in docs]
    first = recs[0]
    for r in recs[1:]:
        if r["backend"] != first["backend"] or r["port"] != first["port"]:
            raise CertError("records of one certificate record share the backend and the commit")
        first["cases"] = sorted(set(first["cases"]) | set(r["cases"]))
        first["matrix"].update(r["matrix"])
        first["passed"] = first["passed"] and r["passed"]
        if "tolerance" in r and kind == "inplace":
            first["tolerance"].update(r["tolerance"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"parity-{first['set']}-{first['backend']}.json"
    path.write_text(json.dumps(first, indent=1) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="compare.py label_propagation certificate")
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--from", dest="records", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        path = write_certificate(
            args.kind, [json.loads(p.read_text()) for p in args.records], args.out
        )
    except CertError as e:
        print(f"certificate: {e}", file=sys.stderr)
        return 1
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
