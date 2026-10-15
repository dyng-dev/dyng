#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The triad_count replays of ESCHER-GPU@abcf9b2's golden set (M9a step 2; PLAN 6.4.6, 8.3;
ADR 0050).

    parity/compare.py triad_count --backend sequential|openmp|cuda [--threads N]
                                  [--mode by_content|by_key|both] [--counter oracle|algorithm]
                                  [--cases LIST] [--jobs N] [--gpu G] [--oracle-threads N]
                                  [--record FILE] [--build DIR] [--keep] [--plant DEFECT]
    parity/triad_replay.py singles [--record FILE]     (the oracle against A, one triad per bin)
    parity/triad_replay.py draws [--backend B] [--groups LIST] [--record FILE]

`compare.py triad_count` (parity/triad_goldens.py compare_main) replays the cases of the set
`triad_count_escher_gpu` ($DYNG_SCRATCH/goldens/triad_count/escher_gpu) that the backend covers
(the coverage table, triad_goldens.COVERAGE; --cases narrows it: case names, group names, or the
coverage items make-test, grid, syn100k, syn1m_s1, ...) through `dyng_triad_replay`
(cpp/tests/parity/escher): per group one run of the tool, which builds the container from the
group's input (A's loaded rows; coauth-DBLP is the dataset file) with A's vertex count and H2V
payload capacity under {erase_first, escher_gpu(capacity), sorted_merge}, applies the group's
batch files in order (by key, by content or both), checks the integrity after every batch, and
reports the counter after the build and after every batch for the patterns of the selected cases.

A case passes when, in every mode run: the full count equals the golden's; there are as many
batches as the golden's, each batch file with the golden's SHA-256 and deletions / insertions;
every batch's net delta and counts after equal the golden's; check_integrity() passed after every
batch; by key every batch's inserted ids were the exported expected ids; by content every
exported deleted row resolved to a live hyperedge with that member set; and (--mode both) the two
modes report the same counts. Exit status: 0 every case passed, 1 a case failed, 2 a usage error
or the counter is not available, 77 the set is missing.

--plant DEFECT (the harness's own test, parity/tests/test_triad_replay.py: each must FAIL):
`bin` changes one bin of the golden full count, `row` perturbs a deleted row of batch 1 (by
content), `drop` drops batch 2, `swap` swaps batches 1 and 2, `id` changes an expected inserted
id of batch 1 (by key).

`singles` (M9a criterion 4): one single-triad `.hg` per h-motif class (the triple whose non-empty
Venn regions are the class's representative, one vertex per region), run through A's UNPATCHED
build/main (`--input F --batches 0`) and through the oracle (`dyng_triad_replay --patterns
h_motif` without batches): both must print the same 30 bins, a single 1 in bin k - 1 for the 24
reachable classes (1-20, 27-30) and nothing for the open triads (21-26).

`draws` (the generators' host-model path, ADR 0051): `dyng-compat-escher run --no-count
--dump-batches` with A's arguments of make-test, syn1m_b50k_s1-s3, syn2m_b50k_s1 and
coauth_b50k_s1-s3 draws its batches from escher_host_model; each batch file must have the
golden's SHA-256, its batch lines (and for make-test its Hypergraph line) must be A's.

Heavy runs take the shared perf lock: flock -s $DYNG_SCRATCH/perf.lock nice -n 10 ...; CUDA runs
use GPU --gpu (default 1, the development GPU).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "parity"))

import triad_goldens as tg  # noqa: E402

PLANTS = ("bin", "row", "drop", "swap", "id")
MODES = {"both": ("by_key", "by_content"), "by_key": ("by_key",), "by_content": ("by_content",)}


class ReplayError(Exception):
    pass


def find_tool(backend: str, build: Path | None = None) -> Path:
    env = os.environ.get("DYNG_TRIAD_REPLAY")
    if env:
        return Path(env)
    presets = ["parity-cuda", "dev-cuda", "release-cuda"]
    if backend != "cuda":
        presets += ["parity", "dev", "release", "cpu-only"]
    for b in [build] if build else [REPO / "build" / p for p in presets]:
        exe = b / "cpp" / "tests" / "dyng_triad_replay"
        if exe.is_file():
            return exe
    raise ReplayError(
        "dyng_triad_replay not built (cmake --build build/parity-cuda --target dyng_triad_replay),"
        " or set DYNG_TRIAD_REPLAY / --build"
    )


# --- The cases ---------------------------------------------------------------------------------


def matches(token: str, group: tg.Group, exe: str) -> bool:
    case = f"{group.name}/{exe}"
    return (
        token in (case, group.name, "all")
        or (token == "make-test" and group.kind == "test")
        or (token == "grid" and group.kind == "grid")
        or (
            group.kind == "grid"
            and token in (group.dataset, f"{group.dataset}_s{group.options.seed}")
        )
        or (group.kind == "grid" and token == f"{group.dataset}_s{group.options.seed}/{exe}")
    )


def select(backend: str, tokens: list[str] | None, coverage: bool = True) -> dict[str, list[str]]:
    """{group name: [exe, ...]} of the covered cases the tokens select (all covered ones without
    tokens); coverage=False selects among all 300 cases (the go/no-go of step 2 runs the syn1m
    seed-1 grid on every backend)."""
    covered = set(tg.coverage_cases(backend)) if coverage else set(tg.case_names())
    out: dict[str, list[str]] = {}
    for g in tg.all_groups():
        for exe in tg.EXES:
            case = f"{g.name}/{exe}"
            if case not in covered:
                continue
            if tokens and not any(matches(t, g, exe) for t in tokens):
                continue
            out.setdefault(g.name, []).append(exe)
    return out


# --- .dgb hypergraph records (docs/api/file_formats.md, version 1; int32 vertices, unweighted) --


def _align8(n: int) -> int:
    return (n + 7) // 8 * 8


def dgb_fields(record: bytes) -> dict:
    """The arrays of one hypergraph record (16-byte prefix + payload) with the flags
    deleted_members and expected_ids."""
    payload = record[16:]
    batch_id, d, i, m, q = struct.unpack_from("<5Q", payload, 0)
    pos = 40
    out = {"batch_id": batch_id}
    out["deleted_ids"] = list(struct.unpack_from(f"<{d}q", payload, pos))
    pos += 8 * d
    out["insert_offsets"] = list(struct.unpack_from(f"<{i + 1}q", payload, pos))
    pos += 8 * (i + 1)
    out["members"] = list(struct.unpack_from(f"<{m}i", payload, pos))
    pos += _align8(4 * m)
    out["deleted_offsets"] = list(struct.unpack_from(f"<{d + 1}q", payload, pos))
    pos += 8 * (d + 1)
    out["deleted_members"] = list(struct.unpack_from(f"<{q}i", payload, pos))
    pos += _align8(4 * q)
    out["expected_ids"] = list(struct.unpack_from(f"<{i}q", payload, pos))
    pos += 8 * i
    if pos != len(payload):
        raise ReplayError(f".dgb record: {len(payload) - pos} bytes after the arrays")
    return out


def dgb_record(f: dict) -> bytes:
    d, i = len(f["deleted_ids"]), len(f["expected_ids"])
    m, q = len(f["members"]), len(f["deleted_members"])

    def arr(fmt: str, values: list[int]) -> bytes:
        raw = struct.pack(f"<{len(values)}{fmt}", *values)
        return raw + b"\0" * (_align8(len(raw)) - len(raw))

    payload = (
        struct.pack("<5Q", f["batch_id"], d, i, m, q)
        + arr("q", f["deleted_ids"])
        + arr("q", f["insert_offsets"])
        + arr("i", f["members"])
        + arr("q", f["deleted_offsets"])
        + arr("i", f["deleted_members"])
        + arr("q", f["expected_ids"])
    )
    return struct.pack("<QI4x", len(payload), zlib.crc32(payload) & 0xFFFFFFFF) + payload


def rewrite_dgb(path: Path, change) -> None:
    """Apply `change(fields)` to the (single) record of a batch file and rewrite it."""
    header, records = tg.dgb_records(path.read_bytes())
    fields = dgb_fields(records[0])
    change(fields)
    path.write_bytes(tg.dgb_single(header, dgb_record(fields)))


# --- One group ---------------------------------------------------------------------------------


def prepare_group(out: Path, scratch: Path, group: str, work: Path, inputs: dict) -> dict:
    """The group's input (decompressed once per input into `work`) and its batch files."""
    gdir = out / "cases" / group
    doc = json.loads((gdir / "group.json").read_text())
    name = doc["input"]
    if name not in inputs:
        meta = json.loads((out / "inputs" / f"{name}.json").read_text())
        if meta.get("stored"):
            path = work / "inputs" / f"{name}.hg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tg.zstd_read(out / meta["stored"]))
        else:
            path = scratch / meta["file"]["path"]
        inputs[name] = path
    bdir = work / "cases" / group
    bdir.mkdir(parents=True, exist_ok=True)
    batches = []
    for b in doc["batches"]:
        path = bdir / b["file"].removesuffix(".zst")
        if not path.is_file():
            path.write_bytes(tg.zstd_read(gdir / b["file"]))
        batches.append(path)
    return {
        "hg": inputs[name],
        "vertices": doc["vertices"],
        "capacity": doc["payload_capacity"],
        "batches": batches,
        "dir": gdir,
        "doc": doc,
    }


def plant(defect: str, prep: dict, work: Path, goldens: dict[str, dict]) -> None:
    """Apply a planted defect to a group's inputs (copies) or its goldens (in memory)."""
    if defect == "bin":
        g = next(iter(goldens.values()))
        g["full_count"][0] += 1
        return
    pdir = work / "planted"
    pdir.mkdir(parents=True, exist_ok=True)
    files = []
    for p in prep["batches"]:
        q = pdir / p.name
        shutil.copyfile(p, q)
        files.append(q)
    if len(files) < 2:
        raise ReplayError("a planted batch defect needs a group of at least two batches")
    if defect == "row":

        def perturb(f: dict) -> None:
            f["deleted_members"][0] = (f["deleted_members"][0] + 1) % prep["vertices"]

        rewrite_dgb(files[0], perturb)
    elif defect == "id":

        def wrong(f: dict) -> None:
            f["expected_ids"][0] += 1

        rewrite_dgb(files[0], wrong)
    elif defect == "drop":
        del files[1]
    elif defect == "swap":
        files[0], files[1] = files[1], files[0]
    prep["batches"] = files


def compare_case(golden: dict, prep: dict, report: dict, pattern: str, modes: list[str]) -> list:
    problems = []
    by_mode = {m["mode"]: m for m in report["modes"]}
    gb = golden["batches"]
    shas = [tg.sha256_file(p) for p in prep["batches"]]
    if len(prep["batches"]) != len(gb):
        problems.append(f"{len(prep['batches'])} batch files, the golden has {len(gb)} batches")
    for k, (sha, b) in enumerate(zip(shas, gb, strict=False), start=1):
        if sha != b["batch_sha256"]:
            problems.append(f"batch {k}: the file's SHA-256 is not the golden's")
    for mode in modes:
        m = by_mode.get(mode)
        if m is None:
            problems.append(f"{mode}: not in the report")
            continue
        if m["full"].get(pattern) != golden["full_count"]:
            problems.append(
                f"{mode}: full count {m['full'].get(pattern)} != {golden['full_count']}"
            )
        if len(m["batches"]) != len(gb):
            problems.append(f"{mode}: {len(m['batches'])} batches replayed, golden {len(gb)}")
        for b, want in zip(m["batches"], gb, strict=False):
            k = b["index"]
            if (b["deletions"], b["insertions"]) != (want["deletions"], want["insertions"]):
                problems.append(
                    f"{mode} batch {k}: {b['deletions']} deletions, {b['insertions']} insertions"
                    f" (golden {want['deletions']}, {want['insertions']})"
                )
            if b["delta"].get(pattern) != want["delta"]:
                problems.append(
                    f"{mode} batch {k}: delta {b['delta'].get(pattern)} != {want['delta']}"
                )
            if b["counts_after"].get(pattern) != want["counts_after"]:
                problems.append(f"{mode} batch {k}: counts after differ from the golden's")
            if not b["integrity_ok"]:
                problems.append(f"{mode} batch {k}: check_integrity failed: {b['integrity_error']}")
            if mode == "by_key" and not b["ids_as_expected"]:
                problems.append(
                    f"by_key batch {k}: inserted ids differ from the expected ids (first at "
                    f"insertion {b['first_id_mismatch']})"
                )
            if mode == "by_content" and b["unresolved"]:
                problems.append(f"by_content batch {k}: {b['unresolved']} deleted rows unresolved")
    if len(modes) == 2 and all(m in by_mode for m in modes):
        a, c = by_mode["by_key"], by_mode["by_content"]
        same = a["full"].get(pattern) == c["full"].get(pattern) and [
            (x["delta"].get(pattern), x["counts_after"].get(pattern)) for x in a["batches"]
        ] == [(x["delta"].get(pattern), x["counts_after"].get(pattern)) for x in c["batches"]]
        if not same:
            problems.append("by_key and by_content report different counts")
    return problems


def run_group(args, tool: Path, out: Path, group: str, exes: list[str], work: Path, inputs, lock):
    with lock:
        prep = prepare_group(out, args.scratch, group, work, inputs)
    goldens = {exe: json.loads((prep["dir"] / f"{exe}.json").read_text()) for exe in exes}
    gwork = work / "cases" / group
    if args.plant:
        plant(args.plant, prep, gwork, goldens)
    report_path = gwork / f"report-{args.mode}.json"
    cmd = [
        str(tool),
        "--hg",
        str(prep["hg"]),
        "--vertices",
        str(prep["vertices"]),
        "--capacity",
        str(prep["capacity"]),
        "--backend",
        args.backend,
        "--mode",
        args.mode,
        "--counter",
        args.counter,
        "--patterns",
        ",".join(tg.PATTERN[e] for e in exes),
        "--json",
        str(report_path),
    ]
    if args.backend == "openmp":
        cmd += ["--threads", str(args.threads)]
    if args.oracle_threads is not None:
        cmd += ["--oracle-threads", str(args.oracle_threads)]
    cmd += [str(p) for p in prep["batches"]]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_MODULE_LOADING="EAGER")
    t0 = time.monotonic()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    seconds = time.monotonic() - t0
    if p.returncode != 0:
        return {
            f"{group}/{e}": {
                "passed": False,
                "problems": [f"dyng_triad_replay exit {p.returncode}: {p.stderr.strip()[-600:]}"],
                "tool_exit": p.returncode,
            }
            for e in exes
        }, seconds
    report = json.loads(report_path.read_text())
    results = {}
    for exe in exes:
        problems = compare_case(goldens[exe], prep, report, tg.PATTERN[exe], list(MODES[args.mode]))
        results[f"{group}/{exe}"] = {
            "passed": not problems,
            "problems": problems[:20],
            "batches": len(goldens[exe]["batches"]),
            "ambiguous": sum(b["ambiguous"] for m in report["modes"] for b in m["batches"]),
            "oracle_evaluations": {m["mode"]: m["oracle_evaluations"] for m in report["modes"]},
            "paths": report["paths"],
            "launches": report["launches"],
            "tool_seconds": report["seconds"],
        }
    if not args.keep:
        report_path.unlink(missing_ok=True)
    return results, seconds


def compare_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="compare.py triad_count")
    p.add_argument("--set", default=tg.SET, choices=[tg.SET])
    p.add_argument("--backend", required=True, choices=sorted(tg.COVERAGE))
    p.add_argument("--threads", type=int, default=8, help="openmp threads")
    p.add_argument("--mode", default="both", choices=sorted(MODES))
    p.add_argument("--counter", default="oracle", choices=["oracle", "algorithm"])
    p.add_argument("--cases", type=lambda s: [x for x in s.split(",") if x])
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--gpu", default=os.environ.get("DYNG_TEST_GPU", "1"))
    p.add_argument("--oracle-threads", type=int)
    p.add_argument("--record", type=Path)
    p.add_argument("--build", type=Path)
    p.add_argument("--scratch", type=Path, default=tg.scratch_default())
    p.add_argument("--keep", action="store_true", help="keep the work directory")
    p.add_argument("--plant", choices=PLANTS, help="a planted defect (the harness's test)")
    p.add_argument(
        "--no-coverage",
        action="store_true",
        help="select among all cases, not only those the backend covers",
    )
    args = p.parse_args(argv)
    out = tg.set_dir(args.scratch)
    if not (out / "MANIFEST.sha256").is_file():
        print(f"{out}: no golden set (parity/triad_goldens.py export --twice)", file=sys.stderr)
        return 77
    try:
        tool = find_tool(args.backend, args.build)
    except ReplayError as e:
        print(f"compare.py triad_count: {e}", file=sys.stderr)
        return 2
    selected = select(args.backend, args.cases, not args.no_coverage)
    if not selected:
        print("compare.py triad_count: no case selected", file=sys.stderr)
        return 2
    (args.scratch / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="triad-replay-", dir=args.scratch / "runs"))
    import threading

    lock = threading.Lock()
    inputs: dict = {}
    results: dict = {}
    timings: dict = {}
    t0 = time.monotonic()
    try:
        # Large groups first, so the pool ends with the small ones.
        order = sorted(
            selected,
            key=lambda g: (
                not g.startswith("coauth"),
                not g.startswith("syn2m"),
                not g.startswith("syn1m"),
                g,
            ),
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(run_group, args, tool, out, g, selected[g], work, inputs, lock): g
                for g in order
            }
            for f in concurrent.futures.as_completed(futures):
                g = futures[f]
                res, seconds = f.result()
                results.update(res)
                timings[g] = round(seconds, 1)
                bad = [c for c, r in res.items() if not r["passed"]]
                print(
                    f"  {g} ({','.join(selected[g])}): {'FAILED ' + str(len(bad)) if bad else 'ok'}"
                    f" {seconds:.1f} s",
                    flush=True,
                )
                for c in bad:
                    for x in res[c]["problems"][:4]:
                        print(f"    {c}: {x}", flush=True)
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    passed = all(r["passed"] for r in results.values())
    n_ok = sum(r["passed"] for r in results.values())
    print(
        f"triad_count {args.backend}{':' + str(args.threads) if args.backend == 'openmp' else ''} "
        f"{args.mode} {args.counter}{' plant=' + args.plant if args.plant else ''}: "
        f"{n_ok}/{len(results)} cases passed ({time.monotonic() - t0:.0f} s)"
    )
    if args.record:
        doc = {
            "schema": 1,
            "set": args.set,
            "manifest_sha256": (tg.golden_spec() or {}).get("manifest_sha256"),
            "backend": args.backend,
            "threads": args.threads if args.backend == "openmp" else None,
            "mode": args.mode,
            "counter": args.counter,
            "plant": args.plant,
            "coverage": not args.no_coverage,
            "tool": str(tool.relative_to(REPO)) if tool.is_relative_to(REPO) else tool.name,
            "commit": subprocess.run(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip(),
            "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "gpu": args.gpu if args.backend == "cuda" else None,
            "cases": dict(sorted(results.items())),
            "group_seconds": dict(sorted(timings.items())),
            "num_cases": len(results),
            "num_passed": n_ok,
            "passed": passed,
        }
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    if all(r.get("tool_exit") == 2 for r in results.values()):
        return 2  # the counter is not available (or every run was a usage error)
    return 0 if passed else 1


# --- singles (criterion 4) ---------------------------------------------------------------------

# The canonical Venn id of classes 1-30 (dyng::testing::triad_class_representatives(); bit 0..2:
# A, B, C only; 3: A and B only; 4: B and C only; 5: C and A only; 6: all three).
REPRESENTATIVES = [64, 65, 67, 71, 72, 73, 76, 75, 77, 79, 88, 90, 89, 91, 93, 95, 120, 121, 123,
                   127, 24, 26, 25, 27, 29, 31, 56, 57, 59, 63]  # fmt: skip
REGIONS = {0: (0,), 1: (1,), 2: (2,), 3: (0, 1), 4: (1, 2), 5: (2, 0), 6: (0, 1, 2)}


def single_triad(venn_id: int) -> str:
    """The three hyperedges (1-based vertices, one per non-empty region) as a .hg text."""
    rows: list[list[int]] = [[], [], []]
    for region, owners in REGIONS.items():
        if (venn_id >> region) & 1:
            for h in owners:
                rows[h].append(region + 1)
    return "".join(" ".join(map(str, sorted(r))) + "\n" for r in rows)


def singles_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="triad_replay.py singles")
    p.add_argument("--record", type=Path)
    p.add_argument("--build", type=Path)
    p.add_argument("--gpu", default=os.environ.get("DYNG_TEST_GPU", "1"))
    p.add_argument("--scratch", type=Path, default=tg.scratch_default())
    args = p.parse_args(argv)
    import perf_ab

    a_main = perf_ab.reference_copy("ESCHER-GPU") / "build" / "main"
    tool = find_tool("sequential", args.build)
    (args.scratch / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="triad-singles-", dir=args.scratch / "runs"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_MODULE_LOADING="EAGER")
    rows, ok = [], True
    try:
        for k, rep in enumerate(REPRESENTATIVES, start=1):
            hg = work / f"class{k:02d}.hg"
            hg.write_text(single_triad(rep))
            a = subprocess.run(
                [str(a_main), "--input", str(hg), "--batches", "0"],
                env=env,
                capture_output=True,
                text=True,
                cwd=work,
            )
            a_bins = None
            for line in a.stdout.splitlines():
                if m := tg.FULL_MAIN.match(line):
                    a_bins = [int(x) for x in m.group(1).split()]
            o = subprocess.run(
                [str(tool), "--hg", str(hg), "--patterns", "h_motif", "--mode", "by_key"],
                capture_output=True,
                text=True,
            )
            o_bins = (
                json.loads(o.stdout)["modes"][0]["full"]["h_motif"] if o.returncode == 0 else None
            )
            reachable = not (21 <= k <= 26)
            want = [0] * 30
            if reachable:
                want[k - 1] = 1
            good = a.returncode == 0 and a_bins == want and o_bins == want
            ok = ok and good
            rows.append(
                {
                    "class": k,
                    "bin": k - 1,
                    "venn_id": rep,
                    "reachable": reachable,
                    "hg": single_triad(rep).strip().split("\n"),
                    "hg_sha256": hashlib.sha256(single_triad(rep).encode()).hexdigest(),
                    "escher_gpu_bins_nonzero": {i: v for i, v in enumerate(a_bins or []) if v},
                    "oracle_bins_nonzero": {i: v for i, v in enumerate(o_bins or []) if v},
                    "agree": good,
                }
            )
            print(
                f"class {k:2d} (bin {k - 1:2d}, Venn id {rep:3d}): A "
                f"{rows[-1]['escher_gpu_bins_nonzero']} oracle {rows[-1]['oracle_bins_nonzero']}"
                f" {'ok' if good else 'DIFFERENT'}"
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(
        f"singles: {sum(r['agree'] for r in rows)}/30 agree, "
        f"{sum(r['reachable'] and r['agree'] for r in rows)}/24 reachable bins"
    )
    if args.record:
        doc = {
            "schema": 1,
            "what": "M9a criterion 4: one single-triad .hg per h-motif class through ESCHER-GPU's "
            "UNPATCHED build/main (--input F --batches 0) and through dyng::testing::"
            "triad_reference (dyng_triad_replay --patterns h_motif)",
            "reference": f"{tg.REFERENCE}@{tg.COMMIT[:7]} (the persistent unpatched copy)",
            "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "classes": rows,
            "reachable_bins_agree": sum(r["reachable"] and r["agree"] for r in rows),
            "passed": ok,
        }
        args.record.write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if ok else 1


# --- draws (the driver's host-model batches against the goldens) --------------------------------

DRAW_GROUPS = [
    *[g.name for g in tg.test_groups()],
    "syn1m_b50k_s1",
    "syn1m_b50k_s2",
    "syn1m_b50k_s3",
    "syn2m_b50k_s1",
    "coauth_b50k_s1",
    "coauth_b50k_s2",
    "coauth_b50k_s3",
]


def find_driver(backend: str, build: Path | None = None) -> Path:
    env = os.environ.get("DYNG_COMPAT_ESCHER")
    if env:
        return Path(env)
    presets = ["parity-cuda", "dev-cuda", "release-cuda"]
    if backend != "cuda":
        presets += ["parity", "dev", "release", "cpu-only"]
    for b in [build] if build else [REPO / "build" / p for p in presets]:
        exe = b / "tools" / "compat" / "dyng-compat-escher"
        if exe.is_file():
            return exe
    raise ReplayError(
        "dyng-compat-escher not built (cmake --build build/parity-cuda --target dyng-compat-escher)"
    )


def draw_group(
    driver: Path, out: Path, scratch: Path, group: tg.Group, backend: str, work: Path, gpu: str
) -> dict:
    """The driver with A's count arguments, --no-count and --dump-batches: its batch files (split
    one record per file, as the exporter does) must have the goldens' SHA-256, and its batch lines
    (and for make-test its Hypergraph line) must be A's."""
    gdir = out / "cases" / group.name
    doc = json.loads((gdir / "group.json").read_text())
    gwork = work / group.name.replace("/", "__")
    gwork.mkdir(parents=True, exist_ok=True)
    name = group.input_file
    if name == "coauth":
        (gwork / "coauth.hg").unlink(missing_ok=True)
        os.symlink(scratch / tg.COAUTH, gwork / "coauth.hg")
    elif name:
        (gwork / f"{name}.hg").write_bytes(tg.zstd_read(out / "inputs" / f"{name}.source.hg.zst"))
    dump = gwork / "drawn.dgb"
    cmd = [
        str(driver),
        "run",
        *group.argv("main")[1:],
        "--no-count",
        "--backend",
        backend,
        "--dump-batches",
        str(dump),
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_MODULE_LOADING="EAGER")
    p = subprocess.run(cmd, cwd=gwork, env=env, capture_output=True, text=True)
    problems = []
    if p.returncode != 0:
        return {"passed": False, "problems": [f"exit {p.returncode}: {p.stderr[-800:]}"]}
    want = [b["sha256"] for b in doc["batches"]]
    got = []
    if want:
        header, records = tg.dgb_records(dump.read_bytes())
        got = [tg.sha256_bytes(tg.dgb_single(header, r)) for r in records]
    if got != want:
        bad = [k + 1 for k, (a, b) in enumerate(zip(got, want, strict=False)) if a != b]
        problems.append(
            f"batch files differ from the goldens' (batches {bad}; {len(got)} drawn,"
            f" {len(want)} golden)"
        )
    lines = [x for x in p.stdout.splitlines() if tg.BATCH_LINE.match(x)]
    golden = json.loads((gdir / "main.json").read_text())
    want_lines = [
        f"Batch {b['index']} ({b['deletions']} deletions, {b['insertions']} insertions)"
        for b in golden["batches"]
        if b["batch_line_printed"]
    ]
    if lines != want_lines:
        problems.append(f"batch lines {lines[:3]} != {want_lines[:3]}")
    if (gdir / "main.stdout").is_file():
        a_line = next(
            x
            for x in (gdir / "main.stdout").read_text().splitlines()
            if tg.HYPERGRAPH_LINE.match(x)
        )
        mine = next((x for x in p.stdout.splitlines() if tg.HYPERGRAPH_LINE.match(x)), None)
        if mine != a_line:
            problems.append(f"Hypergraph line {mine!r} != A's {a_line!r}")
    shutil.rmtree(gwork, ignore_errors=True)
    return {"passed": not problems, "problems": problems, "batches": len(got), "batch_sha256": got}


def draws_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="triad_replay.py draws")
    p.add_argument("--backend", default="cuda", choices=["sequential", "openmp", "cuda"])
    p.add_argument("--groups", type=lambda x: x.split(","), default=DRAW_GROUPS)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--gpu", default=os.environ.get("DYNG_TEST_GPU", "1"))
    p.add_argument("--build", type=Path)
    p.add_argument("--record", type=Path)
    p.add_argument("--scratch", type=Path, default=tg.scratch_default())
    args = p.parse_args(argv)
    out = tg.set_dir(args.scratch)
    driver = find_driver(args.backend, args.build)
    groups = {g.name: g for g in tg.all_groups()}
    (args.scratch / "runs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="triad-draws-", dir=args.scratch / "runs"))
    results = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(
                    draw_group, driver, out, args.scratch, groups[g], args.backend, work, args.gpu
                ): g
                for g in args.groups
            }
            for f in concurrent.futures.as_completed(futures):
                g = futures[f]
                results[g] = f.result()
                r = results[g]
                print(
                    f"  {g}: {'ok' if r['passed'] else 'DIFFERENT'} {r.get('batches', 0)} "
                    f"batches {'; '.join(r['problems'])}",
                    flush=True,
                )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    ok = all(r["passed"] for r in results.values())
    print(
        f"draws ({args.backend}): {sum(r['passed'] for r in results.values())}/{len(results)} "
        "groups draw the goldens' batch files"
    )
    if args.record:
        doc = {
            "schema": 1,
            "what": "M9a step 2: dyng-compat-escher run --no-count draws its batches with "
            "generators::legacy::escher_batch_generator::next(escher_host_model) from one "
            "std::mt19937_64(seed); every batch file (--dump-batches, split one record per file "
            "as the exporter does) must have the SHA-256 of the golden's",
            "backend": args.backend,
            "driver": str(driver.relative_to(REPO)) if driver.is_relative_to(REPO) else driver.name,
            "commit": subprocess.run(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip(),
            "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "groups": dict(sorted(results.items())),
            "passed": ok,
        }
        args.record.write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if ok else 1


def main() -> int:
    if sys.argv[1:2] == ["draws"]:
        return draws_main(sys.argv[2:])
    if sys.argv[1:2] == ["singles"]:
        return singles_main(sys.argv[2:])
    if sys.argv[1:2] == ["compare"]:
        return compare_main(sys.argv[2:])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
