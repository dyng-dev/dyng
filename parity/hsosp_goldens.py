#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The hyper_sssp golden set of MOSP_ESCHER@4b86159 (M11a; ADRs 0080, 0081, 0082).

    hsosp_goldens.py export [--twice DIR] [--cases SEL]   the set (export_goldens.py hyper_sssp)
    hsosp_goldens.py verify [--full]        the set against goldens.toml (compare.py hyper_sssp)
    hsosp_goldens.py manifest               MANIFEST.sha256 against goldens.toml
    hsosp_goldens.py datasets               the Geology and DBLP inputs against the registry
    hsosp_goldens.py census                 nm: every wrapped symbol is called across objects
    hsosp_goldens.py cbst                   the batch records against the CBST traces
    hsosp_goldens.py additivity             cbst_trace_mosp_escher_test through the new twins
    hsosp_goldens.py crosscheck             B's own cross-checks (make test, --verify all, CSVs)
    hsosp_goldens.py oracle                 the oracle go/no-go (dyng_hsosp_oracle)
    hsosp_goldens.py subset                 the committed subset for hosted CI
    hsosp_goldens.py convert [--cases SEL]  dyng-compat-hsosp to-dgb twice, replayed (M11a step 2)

The set `hyper_sssp_mosp_escher` ($DYNG_SCRATCH/goldens/hyper_sssp/mosp_escher) is recorded by the
H-SOSP dump of the recording twins of the PATCHED scratch copy (parity/exporters/hsosp; the
original's unchanged objects relinked with -Xlinker --wrap; DYNG_HSOSP_DUMP_DIR):

  T (test/<NN>-<name>)   MOSP_ESCHER's `make test` (tests/run_tests.sh): its nine H-SOSP invocations
                         (test_h2h_construction, test_h2h_delta, test_hsosp_matches_dijkstra,
                         test_hsosp_scale, hsospStress twice, hsospBench smoke and on random.hg
                         twice), each session replayed whole; every post-batch row, every array
  P (paper/<config>)     hsospBench --hg <coauth.hg | geology.hg> --kind <hyperedge | vertex>
                         --batch <25000 | 50000 | 100000 | 200000> --batches 3 --del 50
                         --verify none
                         with B's defaults; P50 (the four 50K configurations) with every post-batch
                         row and every array; the others with the touched rows, the batches in full
                         and the pairs, distances and parents as SHA-256 digests plus counts

A case holds dump.hsd.zst (the dump with the run-varying fields left out: LOGGED_FIELDS go to the
sibling directory <set>.logged, outside the set; in a digest case the arrays named above replaced by
"<name>.sha256" and "<name>.count"; in a paper case the loaded rows replaced by their digest, the
rows themselves in loads/<dataset>.hsd.zst once per dataset) and case.json (the summary: per
session the capacities, the load, per batch the ops, ids, counters, pairs digest, and every
update / recompute with its fallback decision and the digests of its canonical distances and
parents). The canonical encodings are those of ADR 0080 section 2 (parity/hsosp_dump.py).
--twice exports again from a FRESH patched copy and requires an identical MANIFEST.sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "parity"))

import escher_trace  # noqa: E402
import hsosp_dump as hd  # noqa: E402

REFERENCE = "MOSP_ESCHER"
COMMIT = "4b86159ec0b3bfe6e18c718ff4bef1494d7276a5"
SET = "hyper_sssp_mosp_escher"
KIND = "hyper_sssp"
ZSTD = ["zstd", "-q", "-12", "--long=27", "-T8"]
LOCKLOG = Path("/home/sskg8/Projects/dyng-work/tools/locklog.py")
MILESTONE = "M11a"
# The inputs of P (datasets/registry.toml: coauth-DBLP-hsosp, coauth-MAG-Geology-hsosp).
DATASETS = {
    "dblp": (
        "datasets/hyper/coauth.hg",
        "65fc83b22dfb3ecd14b15436f4aaee54631657f2f4afec8260150679387b7cb6",
    ),
    "geology": (
        "datasets/hyper/geology.hg",
        "884e04ffef5d0caf0528230808dec742a41f59348402fe6b584b6c3323ce9de1",
    ),
}
KINDS = ("hyperedge", "vertex")
SIZES = (25000, 50000, 100000, 200000)
P50 = tuple(f"{d}_{k}_50000" for d in DATASETS for k in KINDS)
# The nine H-SOSP invocations of tests/run_tests.sh, in order (argv as the dump records it).
T_CASES = [
    ("01-test_h2h_construction", ["<abs>/test_h2h_construction"]),
    ("02-test_h2h_delta", ["<abs>/test_h2h_delta"]),
    ("03-test_hsosp_matches_dijkstra", ["<abs>/test_hsosp_matches_dijkstra"]),
    ("04-test_hsosp_scale", ["<abs>/test_hsosp_scale"]),
    ("05-hsospStress", ["<abs>/hsospStress", "--configs", "50"]),
    (
        "06-hsospStress_escher",
        ["<abs>/hsospStress", "--configs", "20", "--seed", "11", "--check-escher"],
    ),
    (
        "07-hsospBench_smoke",
        [
            "<abs>/hsospBench",
            "--suite",
            "smoke",
            "--reps",
            "1",
            "--check-escher",
            "--out",
            "hsospBench_smoke.csv",
        ],
    ),
] + [
    (
        f"{8 + i:02d}-hsospBench_real_{kind}",
        [
            "<abs>/hsospBench",
            "--hg",
            "<abs>/random.hg",
            "--kind",
            kind,
            "--batch",
            "2000",
            "--batches",
            "3",
            "--verify",
            "all",
            "--check-escher",
            "--out",
            "hsospBench_real.csv",
        ],
    )
    for i, kind in enumerate(KINDS)
]
# The arrays a digest case stores as SHA-256 + count (the canonical encodings of ADR 0080).
DIGEST_ARRAYS = {
    "sssp": ("distances", "parents"),
    "apply_batch": ("pairs.deleted.a", "pairs.deleted.b", "pairs.inserted.a", "pairs.inserted.b"),
}
# The backends that replay each case (HD2, decided by the author on 2026-10-28; M11a criterion 14).
COVERAGE = {
    "cuda": ["test", "paper"],
    "sequential": [
        "test",
        "paper/dblp_hyperedge_50000",
        "paper/dblp_vertex_50000",
        "paper/geology_hyperedge_50000",
    ],
    "openmp": ["container level only (HD14: no hyper_sssp OpenMP backend in 0.4)"],
}
MODES = {
    "cuda": {
        "by_id": ["test", "paper"],
        "by_id_public": ["test", *[f"paper/{c}" for c in P50]],
        "by_key": ["test", "paper/dblp_vertex_50000"],
    },
    "sequential": {
        "by_id": [
            "test",
            "paper/dblp_hyperedge_50000",
            "paper/dblp_vertex_50000",
            "paper/geology_hyperedge_50000",
        ],
        "by_id_public": ["test"],
        "by_key": ["test", "paper/dblp_vertex_50000"],
    },
}


class ExportError(Exception):
    pass


# --- Helpers --------------------------------------------------------------------------------------


def scratch_default() -> Path:
    return Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))


def set_dir(scratch: Path) -> Path:
    return scratch / "goldens" / "hyper_sssp" / "mosp_escher"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def portable(path: Path | str) -> str:
    p = Path(path).resolve()
    for base, label in [(REPO, None), (scratch_default().resolve(), "$DYNG_SCRATCH")]:
        try:
            rel = p.relative_to(base).as_posix()
        except ValueError:
            continue
        return rel if label is None else f"{label}/{rel}"
    return f"<outside the repository>/{p.name}"


def references() -> dict:
    return {
        r["name"]: r
        for r in tomllib.loads((REPO / "parity" / "references.toml").read_text())["reference"]
    }


def reference_dir(scratch: Path, variant: str, fresh: bool = False) -> Path:
    tool = REPO / "parity" / "build_reference.sh"
    base = [str(tool), "--variant", variant, "--scratch", str(scratch)]
    if fresh or variant == "patched":
        subprocess.run([*base, *(["--fresh"] if fresh else []), REFERENCE], check=True)
    out = subprocess.check_output([*base, "--print-dir", REFERENCE], text=True)
    return Path(out.strip().splitlines()[-1])


def config_name(dataset: str, kind: str, size: int) -> str:
    return f"{dataset}_{kind}_{size}"


def p_configs() -> list[tuple[str, str, str, int]]:
    return [(config_name(d, k, s), d, k, s) for d in DATASETS for k in KINDS for s in SIZES]


def p_argv(path: Path, kind: str, size: int, out: str, verify: str = "none") -> list[str]:
    return [
        "--hg",
        str(path),
        "--kind",
        kind,
        "--batch",
        str(size),
        "--batches",
        "3",
        "--del",
        "50",
        "--verify",
        verify,
        "--out",
        out,
    ]


def check_dataset(scratch: Path, name: str) -> Path:
    rel, sha = DATASETS[name]
    path = scratch / rel
    if not path.is_file():
        raise ExportError(f"{path} is missing (datasets/registry.toml)")
    if sha256_file(path) != sha:
        raise ExportError(f"{path}: SHA-256 is not the registry's {sha}")
    return path


def wait_gpu1_quiet() -> None:
    """Every GPU-1 job first checks the shared calendar: no GPU1-QUIET window of another agent may
    be open (tools/locklog.py quiet-ok); wait while one is."""
    if not LOCKLOG.is_file():
        return
    while subprocess.run(
        [sys.executable, str(LOCKLOG), "quiet-ok", MILESTONE], capture_output=True
    ).returncode:
        print("waiting: a GPU1-QUIET window is open (tools/locklog.py quiet-ok)", flush=True)
        time.sleep(60)


def run_env(ref: dict, extra: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DYNG_CBST_", "DYNG_HSOSP_"))}
    env["CUDA_VISIBLE_DEVICES"] = env.get("DYNG_GPU", "1")
    for item in ref.get("run_env", []):
        k, _, v = item.partition("=")
        env[k] = v
    env.update(extra)
    return env


def tool_versions() -> dict:
    out = {
        "zstd": subprocess.run(
            ["zstd", "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "awk": subprocess.run(
            ["awk", "--version"], capture_output=True, text=True
        ).stdout.splitlines()[0],
    }
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "1"],
        capture_output=True,
        text=True,
    )
    out["gpu"] = gpu.stdout.strip()
    return out


def deterministic_log(text: str) -> list[str]:
    """B's stderr lines that do not vary between runs: the [load] and [warn] lines with their times
    replaced by <ms> and the initial recompute's iterations by <n>; the rest dropped (the [row]
    lines are all times and counters)."""
    keep = []
    for line in text.splitlines():
        if line.startswith(("[load]", "[warn]", "[FAIL]", "[error]")):
            # the times, and the initial recompute's iteration count (logged only, HD7: it
            # differed between the two exports of the first try)
            line = re.sub(r"[0-9.]+ ms", "<ms>", line)
            keep.append(re.sub(r"\(\d+ iterations\)", "(<n> iterations)", line))
    return keep


# --- Rewriting a dump into a case ----------------------------------------------------------------


def rewrite_dump(
    raw: Path,
    out_dump: Path,
    digest: bool,
    paper: bool,
    loads_dir: Path | None,
    dataset: str | None,
) -> tuple[dict, list[dict]]:
    """Writes the case's dump (run-varying fields left out; digest arrays replaced; a paper case's
    loaded rows stored once in loads/) and returns (case summary, logged records)."""
    d = hd.Dump(raw)
    logged: list[dict] = []
    out = bytearray(hd.file_head())
    summary: dict = {"sessions": [], "header": {}}
    sessions: dict[int, dict] = {}
    end = None
    for kind, rec in d.records():
        rec = dict(rec)
        if kind == "header":
            summary["header"] = {
                k: rec[k] for k in ("argv", "inputs", "env", "rows", "writer_version")
            }
        elif kind == "end":
            end = rec
        # logged-only fields out of the hashed set
        lg = {k: rec.pop(k) for k in hd.LOGGED_FIELDS if k in rec}
        if lg:
            logged.append(
                {"seq": rec.get("seq"), "session": rec.get("session"), "kind": kind, **lg}
            )
        sid = rec.get("session", 0)
        s = (
            sessions.setdefault(sid, {"session": sid, "batches": [], "sssp": [], "compare": []})
            if sid
            else None
        )
        if kind == "session" and s is not None:
            s["caps"] = {k: rec[k] for k in rec if k.startswith("caps.")}
            s["num_vertices"] = rec["num_vertices"]
            s["exception"] = rec["exception"]
        elif kind == "load" and s is not None:
            if rec["phase"] == "enter":
                rows_off, rows_val, w = rec["rows.offsets"], rec["rows.values"], rec["weights"]
                load = {
                    "rows": rows_off.count - 1,
                    "pins": rows_val.count,
                    "rows.sha256": hd.sha256(_bytes(rows_off) + _bytes(rows_val)),
                    "weights.sha256": hd.sha256(_bytes(w)),
                    "source_he": rec["source_he"],
                    "target_he": rec["target_he"],
                }
                s["load"] = load
                if paper:
                    # the loaded rows once per dataset, in loads/<dataset>.hsd.zst
                    assert loads_dir is not None and dataset is not None
                    store_load(rec, loads_dir / f"{dataset}.hsd.zst", load)
                    for k in ("rows.offsets", "rows.values", "weights"):
                        rec.pop(k)
                    rec["rows.sha256"] = load["rows.sha256"]
                    rec["weights.sha256"] = load["weights.sha256"]
                    rec["rows.count"] = load["rows"]
                    rec["load_file"] = f"loads/{dataset}.hsd.zst"
            else:
                s.setdefault("load", {}).update(
                    {k: rec[k] for k in rec if k not in ("session", "phase", "seq")}
                )
        elif kind == "device_build" and s is not None:
            s["device_build"] = {k: rec[k] for k in rec if k not in ("session", "seq")}
        elif kind == "begin_batch" and s is not None:
            if rec["phase"] == "enter":
                s["batches"].append(
                    {
                        "index": rec["index"],
                        "ops": {
                            "he_delete": rec["he_delete"].count,
                            "vtx_delete": rec["vtx_delete.he"].count,
                            "vtx_insert": rec["vtx_insert.he"].count,
                            "he_insert": rec["he_insert.weight"].count,
                        },
                        "batch.sha256": hd.sha256(
                            b"".join(
                                _bytes(rec[k])
                                for k in (
                                    "he_delete",
                                    "vtx_delete.he",
                                    "vtx_delete.vertex",
                                    "vtx_insert.he",
                                    "vtx_insert.vertex",
                                    "he_insert.offsets",
                                    "he_insert.values",
                                    "he_insert.weight",
                                )
                            )
                        ),
                    }
                )
            elif s["batches"]:
                b = s["batches"][-1]
                b["exception"] = rec["exception"]
                if not rec["exception"]:
                    b["final_ids"] = rec["final_ids"].count
                    b["final_ids.sha256"] = hd.sha256(_bytes(rec["final_ids"]))
                    b["dead_he"] = rec["dead_he"].count
                    b["skipped_ops"] = rec["skipped_ops"]
                    b["max_id"] = rec["max_id"]
        elif kind == "apply_batch" and s is not None:
            a = {"index": rec["index"], "exception": rec["exception"]}
            if not rec["exception"]:
                data, nd, ni = hd.canonical_pairs(rec)
                a.update(
                    {
                        "pairs.deleted": nd,
                        "pairs.inserted": ni,
                        "pairs.sha256": hd.sha256(data),
                        "counters": {k: rec[k] for k in rec if k.startswith(("bt.", "delta."))},
                        "h2h_pair_count": rec["h2h_pair_count"],
                        "alive_count": rec["alive_count"],
                        "max_id": rec["max_id"],
                        "device": {k: rec[k] for k in rec if k.startswith("dev.")},
                        "rows.kind": rec["rows.kind"],
                        "rows": rec["rows.ids"].count,
                        "rows.sha256": hd.sha256(
                            b"".join(
                                _bytes(rec[k])
                                for k in (
                                    "rows.ids",
                                    "rows.offsets",
                                    "rows.values",
                                    "rows.weight",
                                    "rows.alive",
                                )
                            )
                        ),
                    }
                )
                if digest:
                    _replace_arrays(rec, DIGEST_ARRAYS["apply_batch"])
                    rec["pairs.sha256"] = a["pairs.sha256"]
            s.setdefault("apply", []).append(a)
        elif kind == "sssp" and s is not None:
            c = {
                k: rec[k]
                for k in (
                    "kind",
                    "state",
                    "batches",
                    "source",
                    "cfg.max_iterations",
                    "cfg.work_budget",
                    "cfg.block_size",
                    "exception",
                )
            }
            if not rec["exception"]:
                c["fallback"] = rec["stats.fallback_recompute"]
                c["seed_count"] = rec["stats.seed_count"]
                c["n"] = rec["n"]
                c["distances.sha256"] = hd.sha256(hd.canonical_distances(rec["distances"].data))
                c["parents.sha256"] = hd.sha256(hd.canonical_parents(rec["parents"].data))
                if digest:
                    _replace_arrays(rec, DIGEST_ARRAYS["sssp"])
            s["sssp"].append(c)
        elif kind == "compare" and s is not None:
            s["compare"].append({k: rec[k] for k in ("state_a", "state_b", "n", "mismatches")})
        out += hd.encode_record(kind, rec)
    if end is None or end.get("exit") != "normal":
        raise ExportError(f"{raw}: the dump did not end normally")
    summary["sessions"] = [sessions[k] for k in sorted(sessions)]
    summary["records"] = end.get("records")
    tmp = out_dump.with_suffix("")
    tmp.write_bytes(bytes(out))
    subprocess.run([*ZSTD, "-f", str(tmp), "-o", str(out_dump)], check=True)
    tmp.unlink()
    return summary, logged


def _bytes(a) -> bytes:
    data = a.data
    if sys.byteorder != "little":
        data = type(data)(data.typecode, data)
        data.byteswap()
    return data.tobytes()


def _replace_arrays(rec: dict, names: tuple[str, ...]) -> None:
    for name in names:
        a = rec.pop(name)
        rec[f"{name}.sha256"] = hd.sha256(_bytes(a))
        rec[f"{name}.count"] = a.count


def store_load(rec: dict, path: Path, load: dict) -> None:
    """The loaded rows of a dataset, once: a dump with the load record only. A second paper case
    of the same dataset must load the same rows."""
    if path.is_file():
        old = json.loads(path.with_suffix(".json").read_text())
        if (
            old["rows.sha256"] != load["rows.sha256"]
            or old["weights.sha256"] != load["weights.sha256"]
        ):
            raise ExportError(f"{path}: another case of the dataset loaded different rows")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    body = hd.file_head() + hd.encode_record("load", {k: v for k, v in rec.items() if k != "seq"})
    tmp = path.with_suffix("")
    tmp.write_bytes(body)
    subprocess.run([*ZSTD, "-f", str(tmp), "-o", str(path)], check=True)
    tmp.unlink()
    path.with_suffix(".json").write_text(json.dumps(load, indent=1, sort_keys=True) + "\n")


# --- Export ---------------------------------------------------------------------------------------


def write_case(
    raw: Path,
    out: Path,
    name: str,
    meta: dict,
    digest: bool,
    paper: bool,
    dataset: str | None,
    logged_dir: Path,
) -> dict:
    d = out / name
    d.mkdir(parents=True)
    summary, logged = rewrite_dump(raw, d / "dump.hsd.zst", digest, paper, out / "loads", dataset)
    case = {"case": name, "digest_mode": digest, **meta, **summary}
    (d / "case.json").write_text(json.dumps(case, indent=1, sort_keys=True) + "\n")
    lf = logged_dir / (name.replace("/", "__") + ".json")
    lf.parent.mkdir(parents=True, exist_ok=True)
    lf.write_text(json.dumps(logged, indent=0) + "\n")
    return case


def export_t(ref_dir: Path, ref: dict, out: Path, raw: Path, logged_dir: Path, info: dict) -> None:
    dumps = raw / "t-dumps"
    (raw / "tmp").mkdir(parents=True, exist_ok=True)
    env = run_env(
        ref,
        {
            "DYNG_HSOSP_DUMP_DIR": str(dumps),
            "DYNG_HSOSP_DUMP_ROWS": "all",
            "TMPDIR": str(raw / "tmp"),
        },
    )
    wait_gpu1_quiet()
    cmd = ref["smoke_test"]
    p = subprocess.run(["bash", "-c", cmd], cwd=ref_dir, env=env, capture_output=True, text=True)
    verdicts = [
        line
        for line in (p.stdout + p.stderr).splitlines()
        if re.match(r"^(PASS|FAIL|XFAIL|XPASS|===)", line)
    ]
    info["test"] = {"command": cmd, "status": p.returncode, "verdicts": verdicts}
    if p.returncode != 0:
        raise ExportError(f"{cmd} failed (exit {p.returncode}):\n" + "\n".join(verdicts[-20:]))
    files = sorted(dumps.glob("*.hsd"))
    if len(files) != len(T_CASES):
        raise ExportError(f"make test wrote {len(files)} H-SOSP dumps, expected {len(T_CASES)}")
    for (name, argv), f in zip(T_CASES, files, strict=True):
        head = next(rec for kind, rec in hd.Dump(f).records() if kind == "header")
        if head["argv"] != argv:
            raise ExportError(f"{f.name}: argv {head['argv']} is not {argv}")
        meta = {"argv": argv, "kind": "test"}
        inputs = []
        for item in head["inputs"]:
            idx, base, fnv, size, stored = item.split(" ", 4)
            entry = {
                "arg": int(idx),
                "name": base.rsplit("/", 1)[-1],
                "fnv1a64": fnv,
                "bytes": int(size),
            }
            if stored != "-":
                src = dumps / stored
                dst = out / "inputs" / Path(stored).name.rsplit(".", 1)[0]
                dst.parent.mkdir(parents=True, exist_ok=True)
                entry["sha256"] = sha256_file(src)
                if not dst.exists():
                    shutil.copyfile(src, dst)
                entry["file"] = f"inputs/{dst.name}"
            inputs.append(entry)
        meta["inputs"] = inputs
        write_case(
            f,
            out,
            f"test/{name}",
            meta,
            digest=False,
            paper=False,
            dataset=None,
            logged_dir=logged_dir,
        )
        print(f"  test/{name}: {f.stat().st_size / 1e6:.1f} MB raw", flush=True)


def export_p(
    ref_dir: Path,
    ref: dict,
    out: Path,
    raw: Path,
    logged_dir: Path,
    info: dict,
    scratch: Path,
    names: list[str],
) -> None:
    info.setdefault("paper", {})
    for name, dataset, kind, size in p_configs():
        if name not in names:
            continue
        free = shutil.disk_usage(scratch).free
        if free < 20e9:
            raise ExportError(f"'/' has {free / 1e9:.1f} GB free (< 20 GB): exports stop")
        path = check_dataset(scratch, dataset)
        dumps = raw / f"p-{name}"
        work = raw / f"w-{name}"
        work.mkdir(parents=True)
        full = name in P50
        env = run_env(
            ref,
            {
                "DYNG_HSOSP_DUMP_DIR": str(dumps),
                "DYNG_HSOSP_DUMP_ROWS": "all" if full else "touched",
            },
        )
        argv = p_argv(path, kind, size, "out.csv")
        wait_gpu1_quiet()
        t0 = time.monotonic()
        p = subprocess.run(
            [str(ref_dir / "bin" / "hsospBench"), *argv],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
        )
        seconds = time.monotonic() - t0
        if p.returncode != 0:
            raise ExportError(
                f"hsospBench {' '.join(argv)} failed (exit {p.returncode}):\n" + p.stderr[-3000:]
            )
        files = sorted(dumps.glob("*.hsd"))
        if len(files) != 1:
            raise ExportError(f"{name}: {len(files)} dumps")
        meta = {
            "argv": [
                "<abs>/hsospBench",
                *[hd.normalize(a) if a.startswith("/") else a for a in argv],
            ],
            "kind": "paper",
            "dataset": dataset,
            "input": {"file": DATASETS[dataset][0], "sha256": DATASETS[dataset][1]},
            "csv": hd.csv_deterministic((work / "out.csv").read_text()),
            "log": deterministic_log(p.stderr),
        }
        raw_bytes = files[0].stat().st_size
        write_case(
            files[0],
            out,
            f"paper/{name}",
            meta,
            digest=not full,
            paper=True,
            dataset=dataset,
            logged_dir=logged_dir,
        )
        info["paper"][name] = {"raw_dump_bytes": raw_bytes}
        (logged_dir / f"paper__{name}.run.json").write_text(
            json.dumps(
                {
                    "seconds": round(seconds, 1),
                    "stderr": p.stderr.splitlines()[-12:],
                    "csv": (work / "out.csv").read_text(),
                },
                indent=1,
            )
            + "\n"
        )
        shutil.rmtree(dumps)
        shutil.rmtree(work)
        print(f"  paper/{name}: {raw_bytes / 1e6:.0f} MB raw, {seconds:.0f} s", flush=True)


def write_manifest(out: Path) -> tuple[str, int, int]:
    lines, size = [], 0
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.sha256":
            lines.append(f"{sha256_file(p)}  {p.relative_to(out).as_posix()}\n")
            size += p.stat().st_size
    text = "".join(lines)
    (out / "MANIFEST.sha256").write_text(text)
    return hashlib.sha256(text.encode()).hexdigest(), len(lines), size


def case_digest(case_dir: Path) -> str:
    lines = [
        f"{sha256_file(p)}  {p.relative_to(case_dir).as_posix()}\n"
        for p in sorted(case_dir.rglob("*"))
        if p.is_file()
    ]
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def select(sel: str) -> tuple[bool, list[str]]:
    """(--cases) 'all', 't', 'p50', 'p' or a comma list of paper configurations."""
    if sel == "all":
        return True, [c[0] for c in p_configs()]
    parts = sel.split(",")
    t = "t" in parts
    names: list[str] = []
    for part in parts:
        if part == "p":
            names += [c[0] for c in p_configs()]
        elif part == "p50":
            names += list(P50)
        elif part != "t":
            if part not in [c[0] for c in p_configs()]:
                raise SystemExit(f"unknown case selection '{part}'")
            names.append(part)
    return t, list(dict.fromkeys(names))


def export_set(ref_dir: Path, out: Path, scratch: Path, sel: str) -> dict:
    ref = references()[REFERENCE]
    raw = out.parent / (out.name + ".raw")
    logged_dir = out.parent / (out.name + ".logged")
    for d in (raw, out, logged_dir):
        shutil.rmtree(d, ignore_errors=True)
    raw.mkdir(parents=True)
    out.mkdir(parents=True)
    info: dict = {
        "set": SET,
        "reference": REFERENCE,
        "commit": COMMIT,
        "kind": KIND,
        "tools": tool_versions(),
        "selection": sel,
        "datasets": {k: {"file": v[0], "sha256": v[1]} for k, v in DATASETS.items()},
        "coverage": COVERAGE,
        "modes": MODES,
        "logged_only": list(hd.LOGGED_FIELDS),
        "csv_run_varying": list(hd.CSV_RUN_VARYING),
    }
    try:
        t, names = select(sel)
        if t:
            export_t(ref_dir, ref, out, raw, logged_dir, info)
        if names:
            export_p(ref_dir, ref, out, raw, logged_dir, info, scratch, names)
        # the raw sizes are run facts, recorded outside the set
        (logged_dir / "raw_sizes.json").write_text(
            json.dumps(info.pop("paper", {}), indent=1) + "\n"
        )
        info["cases"] = sorted(str(p.parent.relative_to(out)) for p in out.rglob("case.json"))
        (out / "info.json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n")
    finally:
        shutil.rmtree(raw, ignore_errors=True)
    sha, n, size = write_manifest(out)
    return {"manifest_sha256": sha, "files": n, "bytes": size, "info": info}


def toml_section(out: Path, result: dict) -> str:
    rows = []
    for case in result["info"]["cases"]:
        c = json.loads((out / case / "case.json").read_text())
        batches = sum(len(s.get("apply", [])) for s in c["sessions"])
        rows.append(
            f'"{case}" = {{ sessions = {len(c["sessions"])}, batches = {batches}, '
            f"digest_mode = {str(c['digest_mode']).lower()}, "
            f'sha256 = "{case_digest(out / case)}" }}\n'
        )
    cov = "".join(f"{k} = {json.dumps(v)}\n" for k, v in COVERAGE.items())
    modes = "".join(
        f"{b}.{m} = {json.dumps(v)}\n" for b, ms in MODES.items() for m, v in ms.items()
    )
    loads = "".join(
        f'"{p.stem.split(".")[0]}" = {{ sha256 = "{sha256_file(p)}" }}\n'
        for p in sorted((out / "loads").glob("*.hsd.zst"))
    )
    converted = converted_section()
    return (
        f"[sets.{SET}]\n"
        f'reference = "{REFERENCE}"\n'
        f'commit = "{COMMIT}"\n'
        f'kind = "{KIND}"\n'
        "what = \"MOSP_ESCHER's make-test H-SOSP sessions (T) and the 16 paper configurations "
        "(P: coauth-DBLP and coauth-MAG-Geology, hyperedge and vertex batches of 25K-200K, 3 "
        'batches, --del 50, --verify none); P50 with every array, the rest with digests"\n'
        f'location = "$DYNG_SCRATCH/goldens/hyper_sssp/mosp_escher"\n'
        'generated_by = "parity/hsosp_goldens.py export --twice '
        '(parity/export_goldens.py hyper_sssp)"\n'
        f"num_cases = {len(result['info']['cases'])}\n"
        f"num_files = {result['files']}\n"
        f"num_bytes = {result['bytes']}\n"
        f'manifest_sha256 = "{result["manifest_sha256"]}"\n'
        f"\n[sets.{SET}.coverage]\n"
        "# Which backend replays which cases (HD2, decided by the author on 2026-10-28).\n"
        'status = "decided by the author on 2026-10-28 (HD2)"\n' + cov + f"\n[sets.{SET}.modes]\n"
        "# The parity modes per backend (ADR 0080 section 2; M11a criterion 14).\n"
        + modes
        + f"\n[sets.{SET}.loads]\n"
        + loads
        + f"\n[sets.{SET}.cases]\n"
        + "".join(rows)
        + converted
    )


# --- dyng-compat-hsosp to-dgb (M11a step 2, ADRs 0085 and 0086) -----------------------------------

CONVERTED_HEADER = (
    f"\n[sets.{SET}.converted]\n"
    "# dyng-compat-hsosp to-dgb of every case (hsosp_goldens.py convert): converted twice\n"
    "# with the same bytes and every session replayed by id; sha256 is the SHA-256 of the sorted\n"
    '# "<sha256>  <file>" lines of the converted files (.s<k>.hg / .dgb / .txt).\n'
)


def converted_section(path: Path = REPO / "parity" / "goldens.toml") -> str:
    """The [sets.<SET>.converted] table as goldens.toml has it (kept by a new export)."""
    doc = tomllib.loads(path.read_text()) if path.is_file() else {}
    rows = doc.get("sets", {}).get(SET, {}).get("converted", {})
    if not rows:
        return ""
    return CONVERTED_HEADER + converted_rows(rows)


def converted_rows(rows: dict) -> str:
    """The rows of the [sets.<SET>.converted] table, sorted by case."""
    return "".join(
        f'"{case}" = {{ sessions = {r["sessions"]}, files = {r["files"]}, '
        f'sha256 = "{r["sha256"]}" }}\n'
        for case, r in sorted(rows.items())
    )


def find_compat_hsosp() -> Path | None:
    """dyng-compat-hsosp: $DYNG_COMPAT_HSOSP, else the first build tree that has it."""
    env = os.environ.get("DYNG_COMPAT_HSOSP")
    if env:
        return Path(env)
    for preset in ("release", "cpu-only", "parity", "dev", "release-cuda", "dev-cuda"):
        exe = REPO / "build" / preset / "tools" / "compat" / "dyng-compat-hsosp"
        if exe.is_file():
            return exe
    return None


def convert_main(argv: list[str]) -> int:
    """Every selected case converted twice with dyng-compat-hsosp to-dgb (the same bytes), every
    converted session replayed (`dyng-compat-hsosp replay`: the expected ids and B's skipped
    counts), and the digests of the converted files; --write-toml records them in the
    [sets.<SET>.converted] table of parity/goldens.toml, and then a second run must give them."""
    p = argparse.ArgumentParser(prog="hsosp_goldens.py convert")
    p.add_argument("--scratch", type=Path, default=scratch_default())
    p.add_argument("--cases", default="t", help="'t', 'p50' or a comma list (default t)")
    p.add_argument("--exe", type=Path, help="dyng-compat-hsosp (default: find it in build/)")
    p.add_argument("--backend", choices=["sequential", "openmp"], default="sequential")
    p.add_argument("--write-toml", action="store_true", help="record the digests in goldens.toml")
    p.add_argument("--record", type=Path, help="write M11a-hsosp-converted.json here")
    args = p.parse_args(argv)
    exe = args.exe or find_compat_hsosp()
    if exe is None or not exe.is_file():
        print("dyng-compat-hsosp not found: build the target dyng-compat-hsosp", file=sys.stderr)
        return 2
    out = set_dir(args.scratch)
    if not (out / "MANIFEST.sha256").is_file():
        print(f"no goldens under {out}; skipping", file=sys.stderr)
        return 77
    t, names = select(args.cases)
    cases = ([f"test/{n}" for n, _ in T_CASES] if t else []) + [f"paper/{n}" for n in names]
    digest = [c for c in cases if c.startswith("paper/") and c.split("/", 1)[1] not in P50]
    if digest:
        print(f"{digest}: digest cases have no rows to convert", file=sys.stderr)
        return 2
    known = tomllib.loads((REPO / "parity" / "goldens.toml").read_text())
    known = known.get("sets", {}).get(SET, {}).get("converted", {})
    rows: dict[str, dict] = {}
    problems: list[str] = []
    (args.scratch / "runs").mkdir(parents=True, exist_ok=True)
    for case in cases:
        with tempfile.TemporaryDirectory(dir=args.scratch / "runs", prefix="hsosp-convert-") as tmp:
            tmp_dir = Path(tmp)
            dump = out / case / "dump.hsd.zst"
            if case.startswith("paper/"):
                dump = splice_load(out, case, tmp_dir)
            stem = case.split("/", 1)[1]
            digests = []
            for run in ("a", "b"):
                (tmp_dir / run).mkdir()
                subprocess.run(
                    [str(exe), "to-dgb", str(dump), str(tmp_dir / run / stem)],
                    check=True,
                    capture_output=True,
                )
                lines = sorted(
                    f"{sha256_file(f)}  {f.name}" for f in sorted((tmp_dir / run).iterdir())
                )
                digests.append(lines)
            if digests[0] != digests[1]:
                problems.append(f"{case}: the two conversions differ")
            sessions = sorted((tmp_dir / "a").glob("*.dgb"))
            for dgb in sessions:
                proc = subprocess.run(
                    [str(exe), "replay", str(dgb.with_suffix("")), "--backend", args.backend],
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    problems.append(f"{case} {dgb.name}: {(proc.stderr or proc.stdout).strip()}")
            sha = hashlib.sha256("".join(x + "\n" for x in digests[0]).encode()).hexdigest()
            rows[case] = {"sessions": len(sessions), "files": len(digests[0]), "sha256": sha}
            if case in known and not args.write_toml and known[case].get("sha256") != sha:
                problems.append(f"{case}: digest {sha} != goldens.toml {known[case].get('sha256')}")
            print(f"{case}: {len(sessions)} sessions, {len(digests[0])} files, {sha[:16]}")
    if args.write_toml and not problems:
        merged = {**known, **rows}
        text = (REPO / "parity" / "goldens.toml").read_text()
        block = CONVERTED_HEADER + converted_rows(merged)
        # The old table out, the new one after the set's last subtable.
        text = re.sub(rf"\n\[sets\.{SET}\.converted\]\n.*?(?=\n\[sets\.|\Z)", "", text, flags=re.S)
        here = text.find(f"[sets.{SET}]")
        later = re.compile(rf"^\[sets\.(?!{SET}[\].])", re.M).search(text, here)
        at = later.start() if later else len(text)
        text = text[:at].rstrip("\n") + "\n" + block + ("\n" + text[at:] if later else "")
        (REPO / "parity" / "goldens.toml").write_text(text)
        print(f"wrote [sets.{SET}.converted] ({len(merged)} cases)")
    if args.record:
        args.record.mkdir(parents=True, exist_ok=True)
        doc = {
            "record": "dyng-compat-hsosp to-dgb of the cases, twice, every session replayed by id",
            "backend": args.backend,
            "commit": git_head(),
            "passed": not problems,
            "cases": rows,
            "problems": problems,
        }
        (args.record / "M11a-hsosp-converted.json").write_text(json.dumps(doc, indent=1) + "\n")
    for pr in problems[:20]:
        print("  " + pr)
    print("convert: " + ("OK" if not problems else "FAILED"))
    return 0 if not problems else 1


def write_toml(section: str) -> None:
    import cycle_count_goldens

    cycle_count_goldens.write_toml(section, SET)


def export_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="hsosp_goldens.py export",
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--scratch", type=Path, default=scratch_default())
    p.add_argument("--cases", default="all", help="all | t | p50 | p | <config>,... (comma list)")
    p.add_argument(
        "--copy",
        type=Path,
        help="export from this patched copy (default: build the patched copy of $DYNG_SCRATCH/ref)",
    )
    p.add_argument(
        "--twice",
        type=Path,
        metavar="SCRATCH2",
        help="export again from a FRESH patched copy built under SCRATCH2 and require "
        "an identical MANIFEST.sha256",
    )
    p.add_argument("--out", type=Path, help="the set directory (default: the golden root's)")
    p.add_argument("--no-toml", action="store_true")
    args = p.parse_args(argv)
    out = args.out or set_dir(args.scratch)
    try:
        ref_dir = args.copy or reference_dir(args.scratch, "patched")
        t0 = time.monotonic()
        result = export_set(ref_dir, out, args.scratch, args.cases)
        print(
            f"{SET}: {len(result['info']['cases'])} cases, {result['files']} files, "
            f"{result['bytes'] / 1e6:.1f} MB, MANIFEST.sha256 {result['manifest_sha256']} "
            f"({time.monotonic() - t0:.0f} s)",
            flush=True,
        )
        if args.twice:
            ref2 = reference_dir(args.twice, "patched", fresh=True)
            out2 = args.twice / "goldens" / "hyper_sssp" / "mosp_escher"
            result2 = export_set(ref2, out2, args.scratch, args.cases)
            if result2["manifest_sha256"] != result["manifest_sha256"]:
                subprocess.run(
                    ["diff", str(out / "MANIFEST.sha256"), str(out2 / "MANIFEST.sha256")]
                )
                print(f"{SET}: SECOND EXPORT DIFFERS", file=sys.stderr)
                return 1
            print(
                f"{SET}: second export from a fresh patched copy ({portable(ref2)}): identical "
                f"manifest ({result2['manifest_sha256']})",
                flush=True,
            )
    except ExportError as e:
        print(f"hsosp_goldens.py export: {e}", file=sys.stderr)
        return 1
    if not args.no_toml and args.out is None:
        write_toml(toml_section(out, result))
        print(f"wrote the [sets.{SET}] section of parity/goldens.toml")
    return 0


# --- Verify ---------------------------------------------------------------------------------------


def verify_set(scratch: Path, full: bool = False, out: Path | None = None) -> list[str]:
    """The set against goldens.toml: the manifest, every file, every case digest; then every case's
    summary against its dump (the counts, the digests of the canonical encodings of every array
    the dump holds) and the structure (T complete and in order, every P configuration, P50 in
    full, the loads), so that a planted change in any file fails."""
    toml = tomllib.loads((REPO / "parity" / "goldens.toml").read_text())
    spec = toml.get("sets", {}).get(SET)
    out = out or set_dir(scratch)
    if spec is None:
        return [f"parity/goldens.toml has no [sets.{SET}]"]
    if not (out / "MANIFEST.sha256").is_file():
        return [f"{out}: no MANIFEST.sha256 (parity/hsosp_goldens.py export)"]
    problems = []
    if (
        hashlib.sha256((out / "MANIFEST.sha256").read_bytes()).hexdigest()
        != spec["manifest_sha256"]
    ):
        problems.append("MANIFEST.sha256 differs from goldens.toml")
    listed = set()
    for line in (out / "MANIFEST.sha256").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        listed.add(rel)
        if not (out / rel).is_file() or sha256_file(out / rel) != digest:
            problems.append(f"{rel} differs from MANIFEST.sha256")
    for f in out.rglob("*"):
        if (
            f.is_file()
            and f.name != "MANIFEST.sha256"
            and f.relative_to(out).as_posix() not in listed
        ):
            problems.append(f"{f.relative_to(out)}: not in MANIFEST.sha256")
    cases = spec.get("cases", {})
    want = [f"test/{n}" for n, _ in T_CASES] + [f"paper/{c[0]}" for c in p_configs()]
    if sorted(cases) != sorted(want):
        problems.append(
            f"the cases are not T and P: missing {sorted(set(want) - set(cases))}, "
            f"extra {sorted(set(cases) - set(want))}"
        )
    for case, c in cases.items():
        if not (out / case).is_dir() or case_digest(out / case) != c["sha256"]:
            problems.append(f"case {case} differs from goldens.toml")
            continue
        if full:
            problems += verify_case(out, case)
    return problems


def verify_case(out: Path, case: str) -> list[str]:
    """A case's summary recomputed from its dump."""
    c = json.loads((out / case / "case.json").read_text())
    problems = []
    paper = case.startswith("paper/")
    if paper and c["digest_mode"] == (case.split("/")[1] in P50):
        problems.append(f"{case}: digest_mode {c['digest_mode']} (P50 is stored in full)")
    d = hd.Dump(out / case / "dump.hsd.zst")
    idx: dict = {}
    for kind, rec in d.records():
        sid = rec.get("session", 0)
        if not sid:
            continue
        s = next((x for x in c["sessions"] if x["session"] == sid), None)
        if s is None:
            problems.append(f"{case}: session {sid} is not in case.json")
            continue
        if kind == "sssp" and not rec["exception"]:
            k = idx.setdefault((sid, "sssp"), 0)
            idx[(sid, "sssp")] += 1
            want = s["sssp"][k]
            if "distances" in rec:
                got = hd.sha256(hd.canonical_distances(rec["distances"].data))
                gotp = hd.sha256(hd.canonical_parents(rec["parents"].data))
            else:
                got, gotp = want["distances.sha256"], want["parents.sha256"]
                if rec["distances.count"] != want["n"]:
                    problems.append(f"{case}: session {sid} sssp {k}: count differs")
            if got != want["distances.sha256"] or gotp != want["parents.sha256"]:
                problems.append(f"{case}: session {sid} sssp {k}: digests differ from the dump")
            if rec["stats.fallback_recompute"] != want["fallback"]:
                problems.append(f"{case}: session {sid} sssp {k}: fallback differs")
        elif kind == "apply_batch" and not rec["exception"]:
            k = idx.setdefault((sid, "apply"), 0)
            idx[(sid, "apply")] += 1
            want = s["apply"][k]
            got = (
                hd.sha256(hd.canonical_pairs(rec)[0])
                if "pairs.deleted.a" in rec
                else rec["pairs.sha256"]
            )
            if got != want["pairs.sha256"]:
                problems.append(f"{case}: session {sid} batch {k}: pairs digest differs")
            for key in ("h2h_pair_count", "max_id"):
                if rec[key] != want[key]:
                    problems.append(f"{case}: session {sid} batch {k}: {key} differs")
    if paper:
        load = next(rec for kind, rec in d.records() if kind == "load" and rec["phase"] == "enter")
        lf = out / load["load_file"]
        lrec = next(rec for kind, rec in hd.Dump(lf).records() if kind == "load")
        got = hd.sha256(_bytes(lrec["rows.offsets"]) + _bytes(lrec["rows.values"]))
        if got != load["rows.sha256"]:
            problems.append(f"{case}: its load file holds other rows")
    return problems


def verify_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="hsosp_goldens.py verify")
    p.add_argument("--scratch", type=Path, default=scratch_default())
    p.add_argument("--full", action="store_true", help="also recompute every case from its dump")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    problems = verify_set(args.scratch, args.full, args.out)
    for pr in problems:
        print("  " + pr)
    print(f"{SET}: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


def manifest_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="hsosp_goldens.py manifest")
    p.add_argument("--scratch", type=Path, default=scratch_default())
    args = p.parse_args(argv)
    out = set_dir(args.scratch)
    text = (out / "MANIFEST.sha256").read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    spec = (
        tomllib.loads((REPO / "parity" / "goldens.toml").read_text()).get("sets", {}).get(SET, {})
    )
    print(
        f"{out / 'MANIFEST.sha256'}: {len(text.splitlines())} files, SHA-256 {sha} "
        f"({'equal to' if spec.get('manifest_sha256') == sha else 'NOT'} goldens.toml's)"
    )
    return 0 if spec.get("manifest_sha256") == sha else 1


# --- The inputs (HD15) ----------------------------------------------------------------------------


def datasets_main(argv: list[str]) -> int:
    """Re-derives the Geology inputs from both registered archives with B's and dynG's converters
    (python3 -I, paths as arguments; B's script taken from the commit with git show into a
    directory of its own) and checks every entry of datasets/registry.toml it names."""
    p = argparse.ArgumentParser(prog="hsosp_goldens.py datasets")
    p.add_argument("--scratch", type=Path, default=scratch_default())
    p.add_argument("--json", type=Path)
    args = p.parse_args(argv)
    reg = {
        d["name"]: d
        for d in tomllib.loads((REPO / "datasets" / "registry.toml").read_text())["dataset"]
    }
    record: dict = {"entries": {}}
    ok = True
    origin = os.path.expandvars(
        references()[REFERENCE]["origin"].replace(
            "$DYNG_ORIGINALS_DIR",
            os.environ.get("DYNG_ORIGINALS_DIR", str(Path.home() / "Projects")),
        )
    )
    runs = args.scratch / "runs"
    with tempfile.TemporaryDirectory(dir=runs, prefix="hsosp-datasets-") as t:
        tdir = Path(t)
        scripts = tdir / "scripts"
        scripts.mkdir()
        (scripts / "benson_to_hg.py").write_bytes(
            subprocess.check_output(
                ["git", "-C", origin, "show", f"{COMMIT}:scripts/benson_to_hg.py"]
            )
        )
        for name in (
            "coauth-MAG-Geology-full",
            "coauth-MAG-Geology",
            "coauth-MAG-Geology-hsosp",
            "coauth-DBLP-hsosp",
        ):
            d = reg[name]
            f = args.scratch / d["file"]
            got = sha256_file(f) if f.is_file() else None
            record["entries"][name] = {"file": d["file"], "sha256": got, "registry": d["sha256"]}
            print(f"{name}: {d['file']} {got} ({'OK' if got == d['sha256'] else 'DIFFERS'})")
            ok &= got == d["sha256"]
        for archive, prefix, want in (
            (
                "coauth-MAG-Geology-full",
                "coauth-MAG-Geology-full/coauth-MAG-Geology-full",
                reg["coauth-MAG-Geology-full"]["derives_to"],
            ),
            (
                "coauth-MAG-Geology",
                "coauth-MAG-Geology/coauth-MAG-Geology",
                reg["coauth-MAG-Geology-hsosp"]["sha256"],
            ),
        ):
            ex = tdir / f"x-{archive}"
            ex.mkdir()
            subprocess.run(
                [
                    "tar",
                    "-xzf",
                    str(args.scratch / reg[archive]["file"]),
                    "-C",
                    str(ex),
                    "--no-same-owner",
                    f"{prefix}-nverts.txt",
                    f"{prefix}-simplices.txt",
                ],
                check=True,
            )
            for label, cmd in (
                (
                    "B scripts/benson_to_hg.py",
                    [sys.executable, "-I", str(scripts / "benson_to_hg.py")],
                ),
                (
                    "dynG benson_to_hg.py --variant hsosp",
                    [
                        sys.executable,
                        "-I",
                        str(REPO / "datasets" / "preprocess" / "benson_to_hg.py"),
                        "--variant",
                        "hsosp",
                    ],
                ),
            ):
                o = tdir / "out.hg"
                r = subprocess.run(
                    [*cmd, str(ex / prefix), str(o)], capture_output=True, text=True, cwd=tdir
                )
                got = sha256_file(o) if r.returncode == 0 else None
                lines = sum(1 for _ in open(o)) if got else None
                record["entries"].setdefault(f"{archive} -> hg", []).append(
                    {
                        "converter": label,
                        "sha256": got,
                        "lines": lines,
                        "stderr": r.stderr.strip().replace(str(args.scratch), "$DYNG_SCRATCH"),
                    }
                )
                print(
                    f"{archive} via {label}: {got} ({lines} lines; "
                    f"{'OK' if got == want else 'DIFFERS from ' + want})"
                )
                ok &= got == want
                o.unlink(missing_ok=True)
            shutil.rmtree(ex)
    record["ok"] = ok
    if args.json:
        args.json.write_text(json.dumps(record, indent=1) + "\n")
    return 0 if ok else 1


# --- nm census (criterion 2) ----------------------------------------------------------------------


def census_main(argv: list[str]) -> int:
    """Every symbol of wrap_hsosp.txt is defined by one object of the patched copy and referenced by
    another (so -Xlinker --wrap reaches every call between objects); lists the other H-SOSP
    entry points the S tests call directly."""
    p = argparse.ArgumentParser(prog="hsosp_goldens.py census")
    p.add_argument("--copy", type=Path, required=True)
    p.add_argument("--json", type=Path)
    args = p.parse_args(argv)
    wraps = [
        s.split("#")[0].strip()
        for s in (REPO / "parity/export_patches/MOSP_ESCHER/wrap_hsosp.txt")
        .read_text()
        .splitlines()
    ]
    wraps = [s for s in wraps if s]
    objs = sorted((args.copy / "build").rglob("*.o"))
    defined: dict[str, list[str]] = {}
    undefined: dict[str, list[str]] = {}
    for o in objs:
        out = subprocess.run(["nm", str(o)], capture_output=True, text=True).stdout
        rel = o.relative_to(args.copy).as_posix()
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[-2] in ("T", "W") and "escher_mosp" in parts[-1]:
                defined.setdefault(parts[-1], []).append(rel)
            elif len(parts) == 2 and parts[0] == "U" and "escher_mosp" in parts[1]:
                undefined.setdefault(parts[1], []).append(rel)
    record: dict = {"wrapped": {}, "direct_entries": {}}
    ok = True
    for s in wraps:
        d, u = defined.get(s, []), undefined.get(s, [])
        record["wrapped"][s] = {"defined_in": d, "referenced_by": u}
        good = len(d) == 1 and bool(u)
        ok &= good
        print(
            f"{'OK' if good else 'NOT CROSS-OBJECT'} {s}: defined in {d}, "
            f"referenced by {len(u)} objects"
        )
    # The H-SOSP and hypergraph entries the tests call directly (not through applyBatch).
    for s, users in sorted(undefined.items()):
        tests = [u for u in users if u.startswith("build/tests/unit/")]
        if tests and ("hsosp" in s or "DynamicHypergraph" in s) and s not in wraps:
            record["direct_entries"][s] = tests
    record["ok"] = ok
    if args.json:
        args.json.write_text(json.dumps(record, indent=1) + "\n")
    return 0 if ok else 1


# --- The CBST traces (criterion 3) ----------------------------------------------------------------


def _cbst_values(a, trace) -> list[int] | int:
    if a.kind == "hash":
        return a.hash
    return list(a.values(trace))


def _same(mine, theirs, trace) -> bool:
    if isinstance(theirs, escher_trace.Array) and theirs.kind == "hash":
        data = mine.data.tobytes() if hasattr(mine, "data") else b""
        return escher_trace.fnv1a(data) == theirs.hash
    return list(mine.values()) == _cbst_values(theirs, trace)


def _cbst_rows(rec: dict, name: str, trace) -> list[list[int]]:
    vals = list(rec[name + ".values"].values(trace))
    pre = list(rec[name + ".prefix"].values(trace))
    out, start = [], 0
    for end in pre:
        out.append(vals[start:end])
        start = end
    return out


class BatchScan:
    """The batch records of a (large) CBST trace, streamed through `zstd -dc` with every other
    record skipped unparsed (coauth-DBLP's traces hold millions of op records); side-file arrays
    are read as escher_trace.Trace reads them."""

    TAG_BATCH = 5
    blob = escher_trace.Trace.blob

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._blob: bytes | None = None

    def records(self):
        p = subprocess.Popen(
            ["zstd", "-q", "-dc", "--long=31", str(self.path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # a scan stopped early closes the pipe: zstd complains
        )
        f = p.stdout
        assert f is not None
        try:
            if f.read(8) != escher_trace.MAGIC:
                raise ExportError(f"{self.path}: not a CBST trace")
            f.read(8)
            while True:
                h = f.read(12)
                if len(h) < 12:
                    break
                tag, length = struct.unpack("<IQ", h)
                if tag == self.TAG_BATCH:
                    buf = memoryview(f.read(length))
                    yield "batch", escher_trace.parse_fields(buf, 0, length)
                else:
                    while length:
                        n = len(f.read(min(length, 1 << 22)))
                        if n == 0:
                            raise ExportError(f"{self.path}: truncated")
                        length -= n
        finally:
            f.close()
            p.wait()


def compare_with_trace(dump: Path, trace: Path) -> dict:
    """The bulk_load and begin_batch records of a dump against those of a CBST trace of the same
    invocation, in order."""
    t = BatchScan(trace)
    theirs = [
        (rec["hook"], rec["phase"], rec)
        for kind, rec in t.records()
        if kind == "batch"
        and rec.get("hook") in ("mosp_escher.bulk_load", "mosp_escher.begin_batch")
    ]
    mine = []
    for kind, rec in hd.Dump(dump).records():
        if kind in ("load", "begin_batch"):
            mine.append(
                (
                    "mosp_escher.bulk_load" if kind == "load" else "mosp_escher.begin_batch",
                    rec["phase"],
                    rec,
                )
            )
    res = {"records": len(theirs), "differences": []}
    if len(mine) != len(theirs):
        res["differences"].append(f"{len(mine)} records here, {len(theirs)} in the trace")
        return res
    for n, ((h1, p1, a), (h2, p2, b)) in enumerate(zip(mine, theirs, strict=True)):
        if (h1, p1) != (h2, p2):
            res["differences"].append(f"record {n}: {h1} {p1} vs {h2} {p2}")
            continue
        if h1 == "mosp_escher.bulk_load" and p1 == "enter":
            if "rows.offsets" not in a:
                continue  # a paper case: the rows are in loads/ (compared there)
            rows = hd.rows_of(a, "rows")
            if rows != _cbst_rows(b, "rows", t) or not _same(a["weights"], b["weights"], t):
                res["differences"].append(f"record {n}: the loaded rows differ")
            for k in ("source_he", "target_he"):
                if a[k] != b[k]:
                    res["differences"].append(f"record {n}: {k} differs")
        elif h1 == "mosp_escher.bulk_load":
            if a["exception"] == "" and a["max_id"] != b.get("max_id"):
                res["differences"].append(f"record {n}: max_id differs")
        elif p1 == "enter":
            for k in (
                "he_delete",
                "vtx_delete.he",
                "vtx_delete.vertex",
                "vtx_insert.he",
                "vtx_insert.vertex",
                "he_insert.weight",
            ):
                if not _same(a[k], b[k], t):
                    res["differences"].append(f"record {n}: {k} differs")
            if hd.rows_of(a, "he_insert") != _cbst_rows(b, "he_insert", t):
                res["differences"].append(f"record {n}: he_insert differs")
        else:
            if a["exception"]:
                if "exception.what" not in b:
                    res["differences"].append(f"record {n}: an exception only here")
                continue
            for k1, k2 in (
                ("final_ids", "final_ids"),
                ("final_weights", "final_weights"),
                ("dead_he", "dead_he"),
            ):
                if not _same(a[k1], b[k2], t):
                    res["differences"].append(f"record {n}: {k1} differs")
            for k in ("skipped_ops", "max_id"):
                if a[k] != b[k]:
                    res["differences"].append(f"record {n}: {k} differs")
    return res


def cbst_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="hsosp_goldens.py cbst")
    p.add_argument("--scratch", type=Path, default=scratch_default())
    p.add_argument("--json", type=Path)
    args = p.parse_args(argv)
    out = set_dir(args.scratch)
    root = args.scratch / "goldens" / "cbst_trace"
    record: dict = {"pairs": {}}
    ok = True
    # make test: match the CBST set's cases to T by their argv.
    test_set = root / "mosp_escher_test"
    by_argv = {}
    for case_json in sorted(test_set.glob("*/case.json")):
        meta = json.loads(case_json.read_text())
        by_argv[tuple(meta["argv"])] = case_json.parent
    for name, argv in T_CASES:
        key = tuple(argv)
        if key not in by_argv:
            record["pairs"][f"test/{name}"] = "not in cbst_trace_mosp_escher_test"
            continue
        r = compare_with_trace(out / "test" / name / "dump.hsd.zst", by_argv[key] / "trace.cbt.zst")
        record["pairs"][f"test/{name}"] = {"trace": f"mosp_escher_test/{by_argv[key].name}", **r}
        ok &= not r["differences"]
        print(
            f"test/{name} vs {by_argv[key].name}: {r['records']} records, "
            f"{len(r['differences'])} differences",
            flush=True,
        )
    for kind in KINDS:
        case = f"paper/dblp_{kind}_50000"
        trace = root / f"mosp_escher_dblp50k_{kind}" / f"dblp_{kind}_50000" / "trace.cbt.zst"
        r = compare_with_trace(out / case / "dump.hsd.zst", trace)
        # the loaded rows of the paper cases are in loads/dblp.hsd.zst
        t = BatchScan(trace)
        b = next(
            rec
            for k, rec in t.records()
            if k == "batch" and rec.get("hook") == "mosp_escher.bulk_load"
        )
        a = next(rec for k, rec in hd.Dump(out / "loads" / "dblp.hsd.zst").records() if k == "load")
        if hd.rows_of(a, "rows") != _cbst_rows(b, "rows", t) or not _same(
            a["weights"], b["weights"], t
        ):
            r["differences"].append("the loaded rows (loads/dblp.hsd.zst) differ")
        record["pairs"][case] = {"trace": f"mosp_escher_dblp50k_{kind}", **r}
        ok &= not r["differences"]
        print(
            f"{case} vs mosp_escher_dblp50k_{kind}: {r['records']} records, "
            f"{len(r['differences'])} differences",
            flush=True,
        )
    record["ok"] = ok
    if args.json:
        args.json.write_text(json.dumps(record, indent=1) + "\n")
    return 0 if ok else 1


# --- Additivity (criterion 2) ---------------------------------------------------------------------


def additivity_main(argv: list[str]) -> int:
    """cbst_trace_mosp_escher_test re-exported through the twins of a patched copy with the H-SOSP
    dump linked in (DYNG_CBST_TRACE_DIR set, DYNG_HSOSP_DUMP_DIR not): its MANIFEST.sha256 must
    equal the golden's (goldens.toml)."""
    import cbst_trace_goldens as ctg

    p = argparse.ArgumentParser(prog="hsosp_goldens.py additivity")
    p.add_argument("--copy", type=Path, required=True)
    p.add_argument("--scratch", type=Path, default=scratch_default())
    p.add_argument("--json", type=Path)
    args = p.parse_args(argv)
    name = "cbst_trace_mosp_escher_test"
    spec = tomllib.loads((REPO / "parity" / "goldens.toml").read_text())["sets"][name]
    wait_gpu1_quiet()
    with tempfile.TemporaryDirectory(dir=args.scratch / "runs", prefix="hsosp-additivity-") as t:
        out = Path(t) / "mosp_escher_test"
        result = ctg.export_set(name, args.copy, out, args.scratch)
        same = result["manifest_sha256"] == spec["manifest_sha256"]
        if not same:
            subprocess.run(
                [
                    "diff",
                    str(ctg.set_dir(args.scratch, name) / "MANIFEST.sha256"),
                    str(out / "MANIFEST.sha256"),
                ]
            )
    record = {
        "set": name,
        "copy": portable(args.copy),
        "manifest_sha256": result["manifest_sha256"],
        "golden_manifest_sha256": spec["manifest_sha256"],
        "identical": same,
        "cases": len(result["metas"]),
    }
    print(json.dumps(record, indent=1))
    if args.json:
        args.json.write_text(json.dumps(record, indent=1) + "\n")
    return 0 if same else 1


# --- B's own cross-checks (criterion 4) -----------------------------------------------------------


def crosscheck_main(argv: list[str]) -> int:
    """With the UNPATCHED copy on GPU 1: a fresh `make test` (every case PASS); every P
    configuration with --verify all (mismatch_static 0, the oracle OK on every batch) and with
    --verify none; h2h_pairs, seeds and overflow_rebuilds of both equal the golden's CSV; the
    fallback decision per batch compared with the repeat rule of ADR 0080 section 3 (a differing
    run is repeated, at most doubling the runs); the --verify none CSV's deterministic columns
    equal the golden's."""
    p = argparse.ArgumentParser(prog="hsosp_goldens.py crosscheck")
    p.add_argument("--scratch", type=Path, default=scratch_default())
    p.add_argument("--cases", default="p")
    p.add_argument("--no-test", action="store_true")
    p.add_argument("--json", type=Path, required=True)
    args = p.parse_args(argv)
    ref = references()[REFERENCE]
    unpatched = reference_dir(args.scratch, "unpatched")
    out = set_dir(args.scratch)
    record: dict = json.loads(args.json.read_text()) if args.json.is_file() else {}
    record.update({"reference": REFERENCE, "commit": COMMIT, "copy": portable(unpatched)})
    ok = True
    env = run_env(ref, {})
    if not args.no_test:
        with tempfile.TemporaryDirectory(dir=args.scratch / "runs", prefix="hsosp-maketest-") as t:
            wait_gpu1_quiet()
            p0 = subprocess.run(
                ["bash", "-c", ref["smoke_test"]],
                cwd=unpatched,
                env=dict(env, TMPDIR=t),
                capture_output=True,
                text=True,
            )
        lines = (p0.stdout + p0.stderr).splitlines()
        verdicts = [x for x in lines if re.match(r"^(PASS|FAIL|XFAIL|XPASS|===)", x)]
        log = args.scratch / "runs" / "M11a-mosp-escher-make-test.log"
        log.write_text(p0.stdout + p0.stderr)
        record["make_test"] = {
            "status": p0.returncode,
            "verdicts": verdicts,
            "log": portable(log),
            "pass": sum(x.startswith("PASS") for x in verdicts),
            "fail": sum(x.startswith(("FAIL", "XPASS")) for x in verdicts),
        }
        ok &= p0.returncode == 0 and record["make_test"]["fail"] == 0
        print(
            f"make test (unpatched): exit {p0.returncode}, {record['make_test']['pass']} PASS, "
            f"{record['make_test']['fail']} FAIL",
            flush=True,
        )
    _, names = select(args.cases)
    record.setdefault("paper", {})
    for name, dataset, kind, size in p_configs():
        if name not in names:
            continue
        golden = json.loads((out / "paper" / name / "case.json").read_text())
        gcsv = golden["csv"]
        entry: dict = {"runs": []}
        path = check_dataset(args.scratch, dataset)

        def one(verify: str, path=path, kind=kind, size=size) -> dict:
            with tempfile.TemporaryDirectory(dir=args.scratch / "runs", prefix="hsosp-xc-") as t:
                wait_gpu1_quiet()
                t0 = time.monotonic()
                r = subprocess.run(
                    [
                        str(unpatched / "bin" / "hsospBench"),
                        *p_argv(path, kind, size, "out.csv", verify),
                    ],
                    cwd=t,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                rows = (
                    hd.csv_deterministic((Path(t) / "out.csv").read_text())
                    if r.returncode == 0
                    else []
                )
                return {
                    "verify": verify,
                    "status": r.returncode,
                    "seconds": round(time.monotonic() - t0, 1),
                    "rows": rows,
                    "log": deterministic_log(r.stderr),
                    "load_line": next(
                        (x for x in r.stderr.splitlines() if "hyperedges (cardinality" in x), None
                    ),
                }

        def check(run: dict, gcsv=gcsv) -> list[str]:
            bad = []
            if run["status"] != 0 or len(run["rows"]) != 3:
                return [f"exit {run['status']}, {len(run['rows'])} rows"]
            for b, (row, g) in enumerate(zip(run["rows"], gcsv, strict=True), 1):
                for col in ("h2h_pairs", "seeds", "overflow_rebuilds"):
                    if row[col] != g[col]:
                        bad.append(f"batch {b}: {col} {row[col]} != golden {g[col]}")
                if row["mismatch_static"] != "0":
                    bad.append(f"batch {b}: mismatch_static {row['mismatch_static']}")
                if run["verify"] == "all" and (
                    row["verified"] != "1"
                    or row["correct"] != "1"
                    or row["mismatch_oracle"] != "0"
                    or row["parent_errors"] != "0"
                ):
                    bad.append(
                        f"batch {b}: the oracle failed ({row['mismatch_oracle']} distances, "
                        f"{row['parent_errors']} parents)"
                    )
                if run["verify"] == "none":
                    # the fallback decision has its own repeat rule (HD7)
                    diff = sorted(k for k in g if k != "fallback" and row.get(k) != g[k])
                    if diff:
                        bad.append(
                            f"batch {b}: deterministic columns differ from the golden's: {diff}"
                        )
            return bad

        def fallbacks(run: dict) -> list[str]:
            return [r["fallback"] for r in run["rows"]]

        gfall = [g["fallback"] for g in gcsv]
        for verify in ("all", "none"):
            run = one(verify)
            run["problems"] = check(run)
            entry["runs"].append(run)
            # the fallback decision: repeated when it differs, at most doubling the runs (HD7)
            repeats = 0
            while fallbacks(run) != gfall and repeats < 1 and run["status"] == 0:
                repeats += 1
                run = one(verify)
                run["problems"] = check(run)
                run["repeat"] = repeats
                entry["runs"].append(run)
            entry[f"fallback_{verify}"] = {
                "golden": gfall,
                "final": fallbacks(run),
                "repeats": repeats,
                "equal": fallbacks(run) == gfall,
            }
        problems = [f"{r['verify']}: {x}" for r in entry["runs"] for x in r["problems"]]
        problems += [
            f"fallback ({v}) still differs after the repeat"
            for v in ("all", "none")
            if not entry[f"fallback_{v}"]["equal"]
        ]
        entry["problems"] = problems
        entry["fallback_golden"] = gfall
        record["paper"][name] = entry
        ok &= not problems
        print(
            f"{name}: {'OK' if not problems else problems} (fallback {gfall}; "
            f"{', '.join(str(r['seconds']) + ' s' for r in entry['runs'])})",
            flush=True,
        )
        args.json.write_text(json.dumps(record, indent=1) + "\n")
    record["ok"] = ok and all(not e["problems"] for e in record["paper"].values())
    args.json.write_text(json.dumps(record, indent=1) + "\n")
    return 0 if record["ok"] else 1


# --- The oracle go/no-go (criterion 5) ------------------------------------------------------------


def find_oracle() -> Path | None:
    env = os.environ.get("DYNG_HSOSP_ORACLE")
    if env:
        return Path(env)
    for p in sorted((REPO / "build").glob("*/cpp/tests/dyng_hsosp_oracle")):
        return p
    return None


def oracle_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="hsosp_goldens.py oracle")
    p.add_argument("--scratch", type=Path, default=scratch_default())
    p.add_argument("--cases", default="t,dblp_vertex_50000,geology_hyperedge_50000")
    p.add_argument("--json", type=Path)
    args = p.parse_args(argv)
    tool = find_oracle()
    if tool is None:
        print("no dyng_hsosp_oracle (build the tests, or set DYNG_HSOSP_ORACLE)", file=sys.stderr)
        return 2
    out = set_dir(args.scratch)
    t, names = select(args.cases)
    cases = ([f"test/{n}" for n, _ in T_CASES] if t else []) + [f"paper/{n}" for n in names]
    record: dict = {"tool": portable(tool), "cases": {}}
    ok = True
    with tempfile.TemporaryDirectory(dir=args.scratch / "runs", prefix="hsosp-oracle-") as tmp:
        for case in cases:
            dump = out / case / "dump.hsd.zst"
            if case.startswith("paper/"):
                # the paper dumps reference their loaded rows: splice them back in
                dump = splice_load(out, case, Path(tmp))
            t0 = time.monotonic()
            r = subprocess.run([str(tool), str(dump)], capture_output=True, text=True)
            res = json.loads(r.stdout.splitlines()[-1]) if r.stdout.strip() else {"error": r.stderr}
            res["file"] = f"{case}/dump.hsd.zst"
            res["seconds"] = round(time.monotonic() - t0, 1)
            record["cases"][case] = res
            good = r.returncode == 0 and res.get("ok") is True
            ok &= good
            print(
                f"{case}: {'GO' if good else 'NO-GO'} ({res.get('sssp_checked')} updates and "
                f"recomputes, {res.get('batches_checked')} batches, {res['seconds']} s) "
                f"{res.get('problems', '')}",
                flush=True,
            )
            if case.startswith("paper/"):
                dump.unlink()
    record["go"] = ok
    if args.json:
        args.json.write_text(json.dumps(record, indent=1) + "\n")
    return 0 if ok else 1


def splice_load(out: Path, case: str, tmp: Path) -> Path:
    """A paper case's dump with its loaded rows back in its load record (from loads/)."""
    d = hd.Dump(out / case / "dump.hsd.zst")
    body = bytearray(hd.file_head())
    for kind, rec in d.records():
        if kind == "load" and rec["phase"] == "enter" and "load_file" in rec:
            full = next(r for k, r in hd.Dump(out / rec["load_file"]).records() if k == "load")
            rec = {
                **{
                    k: v
                    for k, v in rec.items()
                    if k not in ("rows.sha256", "weights.sha256", "rows.count", "load_file")
                },
                "rows.offsets": full["rows.offsets"],
                "rows.values": full["rows.values"],
                "weights": full["weights"],
            }
        body += hd.encode_record(kind, rec)
    path = tmp / (case.replace("/", "__") + ".hsd")
    path.write_bytes(bytes(body))
    return path


# --- The committed subset (criterion 3, HD18) -----------------------------------------------------

SUBSET_DIR = REPO / "cpp" / "tests" / "data" / "goldens" / "hyper_sssp"
SUBSET_LIMIT = 100_000  # bytes (HD18: M11 commits at most 100 KB)


def subset_main(argv: list[str]) -> int:
    """The first sessions of test_hsosp_matches_dijkstra (from the golden T case), up to and with
    the first session that has both a vertex-kind batch and a forced fallback (workBudget 0), as
    one dump: hosted CI replays it (dyng_hsosp_dump_tests)."""
    p = argparse.ArgumentParser(prog="hsosp_goldens.py subset")
    p.add_argument("--scratch", type=Path, default=scratch_default())
    args = p.parse_args(argv)
    src = set_dir(args.scratch) / "test" / "03-test_hsosp_matches_dijkstra" / "dump.hsd.zst"
    d = hd.Dump(src)
    header, sessions, end, loose = hd.sessions(d)
    keep: list[int] = []
    for s in sessions:
        keep.append(s.id)
        vertex = any(
            r["vtx_delete.he"].count + r["vtx_insert.he"].count > 0
            for r in s.of("begin_batch")
            if r["phase"] == "enter"
        )
        fallback = any(
            r.get("stats.fallback_recompute") == 1 and r["cfg.work_budget"] == "0"
            for r in s.of("sssp")
        )
        if vertex and fallback:
            break
    else:
        print("no session has both a vertex batch and a forced fallback", file=sys.stderr)
        return 1
    body = bytearray(hd.file_head())
    for kind, rec in d.records():
        if kind in ("header", "end") or rec.get("session") in keep:
            body += hd.encode_record(kind, rec)
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBSET_DIR / "test_hsosp_matches_dijkstra.hsd"
    out.write_bytes(bytes(body))
    (SUBSET_DIR / "SUBSET.txt").write_text(
        "# SPDX-"
        "FileCopyrightText: 2026 The dynG Authors\n"  # split: not this file's tags
        "# SPDX-"
        "License-Identifier: Apache-2.0\n#\n"
        "# The committed subset of the golden set hyper_sssp_mosp_escher (M11a, HD18: at most\n"
        "# 100 KB), written by parity/hsosp_goldens.py subset from its case\n"
        "# test/03-test_hsosp_matches_dijkstra: the first sessions of MOSP_ESCHER@4b86159's\n"
        "# tests/unit/test_hsosp_matches_dijkstra.cu, up to the first one with a vertex-kind\n"
        "# batch and a forced fallback (workBudget 0). Read by dyng_hsosp_dump_tests in hosted\n"
        "# CI.\n"
        f"test_hsosp_matches_dijkstra.hsd  sessions {keep}  sha256 {sha256_file(out)}\n"
    )
    size = out.stat().st_size
    print(
        f"{out.relative_to(REPO)}: sessions {keep}, {size} bytes "
        f"({'OK' if size <= SUBSET_LIMIT else 'OVER the 100 KB budget'})"
    )
    return 0 if size <= SUBSET_LIMIT else 1


# --- CLI ------------------------------------------------------------------------------------------

COMMANDS = {
    "export": export_main,
    "verify": verify_main,
    "manifest": manifest_main,
    "datasets": datasets_main,
    "census": census_main,
    "cbst": cbst_main,
    "additivity": additivity_main,
    "crosscheck": crosscheck_main,
    "oracle": oracle_main,
    "subset": subset_main,
    "convert": convert_main,
}


def find_replay() -> Path | None:
    """dyng_hsosp_replay: $DYNG_HSOSP_REPLAY, else the first build tree that has it."""
    env = os.environ.get("DYNG_HSOSP_REPLAY")
    if env:
        return Path(env)
    for preset in (
        "release",
        "cpu-only",
        "parity",
        "dev",
        "release-cuda",
        "parity-cuda",
        "dev-cuda",
    ):
        exe = REPO / "build" / preset / "cpp" / "tests" / "dyng_hsosp_replay"
        if exe.is_file():
            return exe
    return None


DELTA_TOTALS = (
    "sessions",
    "stopped_sessions",
    "batches",
    "vertex_batches",
    "ids_compared",
    "rows_compared",
    "pair_batches",
    "deleted_pairs",
    "inserted_pairs",
    "filtered_virtual",
    "skipped_ops",
    "skipped_edits",
    "removals",
    "additions",
    "original_exceptions",
    "hyper_sssp_weight_limit",
    "oracle_states",
    "candidates_equal",
    "candidates_differ",
    "csr_rebuilds",
    "b_csr_rebuilds",
    "node_growths",
    "mismatch_count",
)


def delta_main(args) -> int:
    """--part delta (M11a criteria 8 and 10, ADRs 0085 and 0086): every session of the cases
    replayed whole through dyng::hypergraph on slack_csr storage by id with dyng_hsosp_replay (the
    ids, skipped ops, dead ids, rows, weights, the pair delta and count, check_integrity(), and
    with the oracle rebuild_line_graph and line_graph_delta after every batch). A paper case's
    loaded rows are spliced back in from loads/ (one case at a time: the spliced dump is a few
    GB for coauth-DBLP). --record writes M11a-hsosp-delta-<backend>[-t<N>].json."""
    exe = args.exe or find_replay()
    if exe is None or not exe.is_file():
        print("dyng_hsosp_replay not found: build dyng_hsosp_replay_tool", file=sys.stderr)
        return 2
    out = set_dir(args.scratch)
    if not (out / "MANIFEST.sha256").is_file():
        print(f"no goldens under {out}; skipping", file=sys.stderr)
        return 77
    problems: list[str] = []
    if not args.skip_verify:
        problems += verify_set(args.scratch)
    t, names = select(args.cases)
    cases = ([f"test/{n}" for n, _ in T_CASES] if t else []) + [f"paper/{n}" for n in names]
    digest = [c for c in cases if c.startswith("paper/") and c.split("/", 1)[1] not in P50]
    if digest:
        print(
            f"{digest}: digest cases (their pairs are hashes): the CUDA replay's", file=sys.stderr
        )
        return 2
    reports = []
    (args.scratch / "runs").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.scratch / "runs", prefix="hsosp-delta-") as tmp:
        for case in cases:
            dump = out / case / "dump.hsd.zst"
            if case.startswith("paper/"):
                dump = splice_load(out, case, Path(tmp))
            js = Path(tmp) / "report.json"
            cmd = [str(exe), "--backend", args.backend, "--json", str(js)]
            if args.threads:
                cmd += ["--threads", str(args.threads)]
            if args.no_oracle:
                cmd.append("--no-oracle")
            cmd.append(str(dump))
            t0 = time.monotonic()
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode not in (0, 1) or not js.is_file():
                rep = {"ok": False, "error": (r.stderr or r.stdout)[-2000:], "mismatch_count": 1}
            else:
                rep = json.loads(js.read_text())[0]
            rep["file"] = f"{case}/dump.hsd.zst"
            rep["wall_seconds"] = round(time.monotonic() - t0, 1)
            reports.append(rep)
            first = "" if rep.get("ok") else (rep.get("mismatches") or [rep.get("error", "")])[0]
            print(
                f"{case}: {'ok' if rep.get('ok') else 'MISMATCH'} ({rep.get('sessions')} sessions, "
                f"{rep.get('batches')} batches, {rep.get('vertex_batches')} with incidence edits, "
                f"{rep.get('ids_compared')} ids, {rep.get('deleted_pairs')} + "
                f"{rep.get('inserted_pairs')} pairs, {rep['wall_seconds']} s) "
                f"{first}",
                flush=True,
            )
            if case.startswith("paper/"):
                dump.unlink()
    totals = {k: sum(int(r.get(k, 0)) for r in reports) for k in DELTA_TOTALS}
    for r in reports:
        if not r.get("ok"):
            problems.append(f"{r['file']}: {(r.get('mismatches') or [r.get('error', '?')])[0]}")
    if totals["stopped_sessions"]:
        problems.append(f"{totals['stopped_sessions']} sessions were not replayed to their end")
    record = {
        "record": "hyper_sssp's container side by id: B's sessions replayed whole on slack_csr",
        "backend": args.backend,
        "threads": args.threads,
        "oracle": not args.no_oracle,
        "verified": not args.skip_verify,
        "commit": git_head(),
        "tool": portable(exe),
        "passed": not problems,
        "totals": totals,
        "cases": reports,
        "problems": problems,
    }
    print(
        f"hyper_sssp delta on {args.backend}"
        + (f" ({args.threads} threads)" if args.threads else "")
        + f": {len(reports)} cases, {totals['sessions']} sessions ({totals['stopped_sessions']} "
        f"stopped), {totals['batches']} batches ({totals['vertex_batches']} with incidence edits), "
        f"{totals['ids_compared']} ids, {totals['deleted_pairs']} + {totals['inserted_pairs']} "
        f"pairs, {totals['mismatch_count']} mismatches: " + ("OK" if not problems else "FAILED")
    )
    for pr in problems[:20]:
        print("  " + pr)
    if args.record:
        args.record.mkdir(parents=True, exist_ok=True)
        name = f"M11a-hsosp-delta-{args.backend}" + (f"-t{args.threads}" if args.threads else "")
        (args.record / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n")
    return 0 if not problems else 1


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def compare_main(argv: list[str]) -> int:
    """parity/compare.py hyper_sssp [--verify [--full]] [--list] [--part delta|results]: the
    goldens verified; --part delta replays the sessions on the container (M11a step 2, the host
    backends; delta_main); --part results compares the distances and parents (M11a step 4)."""
    p = argparse.ArgumentParser(prog="compare.py hyper_sssp")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--part", choices=["delta", "results"])
    p.add_argument("--scratch", type=Path, default=scratch_default())
    p.add_argument("--backend", choices=["sequential", "openmp"], default="sequential")
    p.add_argument("--threads", type=int, default=0, help="openmp threads (0: the default)")
    p.add_argument("--cases", default="t,p50", help="'t', 'p50' or a comma list (default t,p50)")
    p.add_argument("--no-oracle", action="store_true", help="leave out the oracle checks")
    p.add_argument("--skip-verify", action="store_true", help="do not re-hash the goldens")
    p.add_argument("--exe", type=Path, help="dyng_hsosp_replay (default: find it in build/)")
    p.add_argument("--record", type=Path, help="write M11a-hsosp-delta-*.json here")
    args = p.parse_args(argv)
    if args.part == "delta":
        return delta_main(args)
    if args.list:
        for name, _ in T_CASES:
            print(f"test/{name}")
        for c in p_configs():
            print(f"paper/{c[0]}{'' if c[0] in P50 else ' (digests)'}")
        return 0
    if args.verify or args.part is None:
        return verify_main(["--scratch", str(args.scratch), *(["--full"] if args.full else [])])
    print(
        f"compare.py hyper_sssp --part {args.part}: no port to replay yet (M11a steps 2-4 add "
        "dyng-compat-hsosp replay)",
        file=sys.stderr,
    )
    return 2


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        return 2
    return COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
