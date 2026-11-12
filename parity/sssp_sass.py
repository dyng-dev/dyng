#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The machine-code identity of sssp and mosp across the shared-engine generalization (M11b
criterion 2; PLAN 4.5.3, 6.3 step 8 and 8.6 "Kernel checks"; ADR 0095).

    parity/sssp_sass.py record --build <tree> [--json OUT]
    parity/sssp_sass.py compare BASE.json HEAD.json [--json OUT]
    parity/sssp_sass.py plant --build <tree> [--json OUT]

`record` reads every object of sssp and mosp in a build tree (the parity-cuda preset; OBJECTS):
for each CUDA object every kernel function of `cuobjdump -sass` (the fused persistent kernel, the
operators engine's kernels and mosp's, every DYNG_FOR_EACH_GRAPH_TYPE instantiation) with its
instruction stream and `cuobjdump -res-usage`'s registers, stack, shared, local and constant
memory, and for every object (host and CUDA) the host instructions of `objdump -d`. Only NAMES
are normalized: a kernel's own name becomes <self>, and the names nvcc derives from the build (its
temporary file's process id, the anonymous and internal namespaces' hashes:
sass_identity.strip_build_names) are removed. Addresses, offsets, immediates, registers, predicates
and constant-bank operands stay, so two kernels compare equal exactly when their instructions do.

For the CUDA objects it also digests every section of the embedded device code (cuobjdump -xelf)
apart from the debug sections (the line tables of -lineinfo and the PTX text, which follow the
source's line numbers) and the symbol and string tables (names), so that the kernels' info,
constant and shared-memory sections are compared byte for byte too.

`compare` pairs the kernels of two records by their normalized names and the objects by path:
identical when every object holds the same kernels with the same instruction streams and
resources, the same non-debug device sections and the same host instructions. Exit 1 otherwise,
listing what differs (whether the debug sections are identical is reported, not required).

`plant` shows that the normalization catches a one-instruction change: it copies
cpp/src/algorithms/sssp to a scratch directory, plants one change in the operators engine's pull
(PLANT: `nd <= max_distance` -> `nd < max_distance`, one comparison), compiles operators.cu from the
copy with the build tree's own nvcc command (compile_commands.json, the copy's directory first on
the include path) and compares the planted object's kernels with the tree's: the pull kernels must
differ (and only they), each by the same instruction count, while an unplanted recompile of the
same command must be identical. Exit 1 if either does not hold.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "parity"))

import sass_identity  # noqa: E402
import triad_sass  # noqa: E402

SSSP = "cpp/src/algorithms/sssp/CMakeFiles/dyng_sssp.dir"
MOSP = "cpp/src/algorithms/mosp/CMakeFiles/dyng_mosp.dir"
# Every object of sssp and mosp (the CUDA ones hold kernels and host code; the others host code).
OBJECTS = [
    f"{SSSP}/cuda.cu.o",
    f"{SSSP}/operators.cu.o",
    f"{SSSP}/sssp.cpp.o",
    f"{SSSP}/sequential.cpp.o",
    f"{SSSP}/openmp.cpp.o",
    f"{MOSP}/cuda.cu.o",
    f"{MOSP}/mosp.cpp.o",
    f"{MOSP}/sequential.cpp.o",
    f"{MOSP}/openmp.cpp.o",
    "tools/compat/CMakeFiles/dyng-compat-mosp.dir/mosp/mosp_compat.cpp.o",
]

# The planted change of `plant`: one comparison of the operators engine's pull.
PLANT_FILE = "operators.cuh"
PLANT_OLD = "if (nd <= max_distance) {  // larger: never shortest, may not fit"
PLANT_NEW = "if (nd < max_distance) {  // PLANTED (parity/sssp_sass.py plant)"
PLANT_KERNEL = "pull_kernel"


def normalize_name(name: str) -> str:
    """A function name with the names nvcc derives from the build removed (and the length prefix
    the mangling puts before them, which follows the hash's length)."""
    name = sass_identity.strip_build_names(sass_identity.ANONYMOUS.sub("_GLOBAL__N__", name))
    return re.sub(r"\d+(_GLOBAL__N__|_INTERNAL_)", r"\1", name)


def normalize_instruction(ins: str, name: str) -> str:
    """One SASS instruction with only names normalized: its function's own name -> <self>, the
    build-derived names removed, whitespace collapsed. Nothing else changes."""
    ins = ins.replace(name, "<self>")
    return re.sub(r"\s+", " ", normalize_name(ins)).strip()


def kernels(obj: Path) -> dict[str, dict]:
    """{normalized name: {digest, instructions, resources}} of every kernel function of `obj`."""
    res = triad_sass.resource_usage(obj)
    out = {}
    for name, code in triad_sass.sass(obj).items():
        stream = "\n".join(f"{addr:06x} {normalize_instruction(ins, name)}" for addr, ins in code)
        out[normalize_name(name)] = {
            "digest": hashlib.sha256(stream.encode()).hexdigest(),
            "instructions": len(code),
            "resources": res.get(name, {}),
        }
    return out


# Sections of the device code that hold debug data only (the line tables of -lineinfo and the PTX
# text with its file and line comments), and the symbol tables (names: the build-derived ones
# differ between two trees); every other section is compared byte for byte.
DEBUG_SECTION = re.compile(r"^\.(rel\.)?(debug_|nv_debug_)|^\.(strtab|symtab|shstrtab)$")


def device_sections(obj: Path) -> dict:
    """{normalized section name: SHA-256} of every non-debug section of every cubin embedded in
    `obj` (cuobjdump -xelf), and the digests of the debug sections apart."""
    out: dict = {"sections": {}, "debug": {}}
    with tempfile.TemporaryDirectory(prefix="sssp-sass-elf-") as tmp:
        subprocess.run(
            [triad_sass.cuobjdump(), "-xelf", "all", str(obj.resolve())],
            cwd=tmp,
            check=True,
            capture_output=True,
        )
        for cubin in sorted(Path(tmp).glob("*.cubin")):
            data = cubin.read_bytes()
            heads = subprocess.run(
                ["readelf", "-SW", str(cubin)], capture_output=True, text=True
            ).stdout
            for line in heads.splitlines():
                m = re.match(
                    r"\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+[0-9a-f]+\s+([0-9a-f]+)\s+"
                    r"([0-9a-f]+)",
                    line,
                )
                if not m:
                    continue
                name, kind = m.group(1), m.group(2)
                off, size = int(m.group(3), 16), int(m.group(4), 16)
                key = f"{cubin.name.split('.')[-2]}:{normalize_name(name)}"
                # A NOBITS section (shared memory, .bss) has a size and no bytes in the file.
                body = f"NOBITS {size}".encode() if kind == "NOBITS" else data[off : off + size]
                digest = hashlib.sha256(body).hexdigest()
                (out["debug"] if DEBUG_SECTION.search(name) else out["sections"])[key] = digest
    return out


def host_code(obj: Path) -> dict:
    """The digest of an object's host instructions and their relocations (`objdump -d -r`),
    build-derived names removed. objdump's `<symbol+offset>` annotations are dropped: in a
    relocatable object they name the nearest symbol of an address the relocation fills in (a
    label of the string section whose number depends on the strings' paths), and the relocation
    lines that follow them name the real target."""
    out = subprocess.run(
        ["objdump", "-d", "-r", "--no-show-raw-insn", str(obj)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    lines = [  # the file name line first
        normalize_name(re.sub(r"\s*<[^>]*>", "", line.rstrip())) for line in out.splitlines()[2:]
    ]
    stream = "\n".join(lines)
    return {"digest": hashlib.sha256(stream.encode()).hexdigest(), "lines": len(lines)}


def record(build: Path) -> dict:
    objects = {}
    for rel in OBJECTS:
        obj = build / rel
        if not obj.is_file():
            objects[rel] = {"missing": True}
            continue
        entry: dict = {"sha256": hashlib.sha256(obj.read_bytes()).hexdigest()}
        if rel.endswith(".cu.o"):
            entry["kernels"] = dict(sorted(kernels(obj).items()))
            entry["device"] = device_sections(obj)
        entry["host"] = host_code(obj)
        objects[rel] = entry
    return {
        "record": "sssp's and mosp's kernels (SASS, resources) and host code (M11b criterion 2)",
        "tool": "parity/sssp_sass.py",
        "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": provenance(build),
        "objects": objects,
        "kernels": sum(len(o.get("kernels", {})) for o in objects.values()),
        "instructions": sum(
            k["instructions"] for o in objects.values() for k in o.get("kernels", {}).values()
        ),
    }


def provenance(build: Path) -> dict:
    cache = build / "CMakeCache.txt"
    first = next((build / rel for rel in OBJECTS if (build / rel).is_file()), None)
    doc = {"build": str(build)}
    if cache.is_file():
        for key in ("CMAKE_HOME_DIRECTORY", "CMAKE_BUILD_TYPE", "CMAKE_CUDA_COMPILER"):
            doc[key] = triad_sass._cache_value(cache, key)
        src = doc.get("CMAKE_HOME_DIRECTORY")
        doc["commit"] = triad_sass._commit(Path(src)) if src else None
        # A git-archive copy (no .git) names its commit in src.commit beside it.
        marker = Path(src).parent / "src.commit" if src else None
        if doc["commit"] is None and marker is not None and marker.is_file():
            doc["commit"] = marker.read_text().strip() + " (src.commit of a git-archive copy)"
    if first is not None:
        doc["cuobjdump"] = triad_sass.provenance([first], False)["cuobjdump"]
    return doc


def compare_records(base: dict, head: dict) -> dict:
    """What differs between two records (empty lists: identical)."""
    out: dict = {"objects": {}, "identical": True}
    for rel in sorted(set(base["objects"]) | set(head["objects"])):
        a, b = base["objects"].get(rel), head["objects"].get(rel)
        if a is None or b is None or a.get("missing") or b.get("missing"):
            out["objects"][rel] = {"identical": False, "missing": True}
            out["identical"] = False
            continue
        ka, kb = a.get("kernels", {}), b.get("kernels", {})
        differ = sorted(n for n in set(ka) | set(kb) if ka.get(n) != kb.get(n))
        host_same = a["host"] == b["host"]
        da, db = a.get("device", {}).get("sections", {}), b.get("device", {}).get("sections", {})
        sections = sorted(n for n in set(da) | set(db) if da.get(n) != db.get(n))
        debug_same = a.get("device", {}).get("debug") == b.get("device", {}).get("debug")
        same = not differ and host_same and not sections
        out["objects"][rel] = {
            "identical": same,
            "kernels": len(kb),
            "instructions": sum(k["instructions"] for k in kb.values()),
            "differ": differ,
            "host_identical": host_same,
            "device_sections_differ": sections,
            "debug_sections_identical": debug_same,
            "byte_identical": a["sha256"] == b["sha256"],
        }
        out["identical"] = out["identical"] and same
    return out


def plant(build: Path, keep: Path | None = None) -> dict:
    """Recompile operators.cu twice from a copy of the sssp folder, unplanted and planted, and
    compare both objects' kernels with the build tree's."""
    db = json.loads((build / "compile_commands.json").read_text())
    target = f"{SSSP}/operators.cu.o"
    entry = next(e for e in db if e.get("output", "").endswith(target) or target in e["command"])
    src_root = Path(triad_sass._cache_value(build / "CMakeCache.txt", "CMAKE_HOME_DIRECTORY"))
    results = {}
    with tempfile.TemporaryDirectory(prefix="sssp-sass-plant-") as tmp:
        root = Path(tmp)
        for label, planted in (("unplanted", False), ("planted", True)):
            copy = root / label / "algorithms" / "sssp"
            shutil.copytree(src_root / "cpp/src/algorithms/sssp", copy)
            if planted:
                text = (copy / PLANT_FILE).read_text()
                if (count := text.count(PLANT_OLD)) != 1:
                    raise SystemExit(f"plant: the planted line matches {count} times")
                (copy / PLANT_FILE).write_text(text.replace(PLANT_OLD, PLANT_NEW))
            out = root / label / "operators.cu.o"
            argv = shlex.split(entry["command"])
            # The copy first on the include path; the copy's source; the object in scratch.
            i = argv.index("-o")
            argv[i + 1] = str(out)
            source = str(copy / "operators.cu")
            argv = [source if a.endswith("algorithms/sssp/operators.cu") else a for a in argv]
            argv.insert(1, f"-I{root / label}")
            subprocess.run(argv, cwd=entry["directory"], check=True, capture_output=True)
            ka, kb = kernels(build / target), kernels(out)
            differ = sorted(n for n in set(ka) | set(kb) if ka.get(n) != kb.get(n))
            results[label] = {
                "differ": differ,
                "instructions": {
                    n: [ka.get(n, {}).get("instructions"), kb.get(n, {}).get("instructions")]
                    for n in differ
                },
            }
            if keep is not None:
                keep.mkdir(parents=True, exist_ok=True)
                shutil.copy(out, keep / f"operators-{label}.cu.o")
    pulls = [n for n in results["planted"]["differ"] if PLANT_KERNEL in n]
    ok = (
        not results["unplanted"]["differ"] and bool(pulls) and pulls == results["planted"]["differ"]
    )
    return {
        "record": "the planted one-instruction change of parity/sssp_sass.py plant",
        "plant": {
            "file": f"cpp/src/algorithms/sssp/{PLANT_FILE}",
            "old": PLANT_OLD,
            "new": PLANT_NEW,
        },
        "results": results,
        "caught": ok,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("record")
    r.add_argument("--build", type=Path, required=True)
    r.add_argument("--json", type=Path)
    c = sub.add_parser("compare")
    c.add_argument("base", type=Path)
    c.add_argument("head", type=Path)
    c.add_argument("--json", type=Path)
    q = sub.add_parser("plant")
    q.add_argument("--build", type=Path, required=True)
    q.add_argument("--json", type=Path)
    args = p.parse_args(argv)
    if args.command == "record":
        doc = record(args.build)
        missing = [rel for rel, o in doc["objects"].items() if o.get("missing")]
        print(
            f"{doc['kernels']} kernels, {doc['instructions']} instructions in "
            f"{len(OBJECTS) - len(missing)} objects; missing {missing}"
        )
        rc = 1 if missing else 0
    elif args.command == "compare":
        base, head = (json.loads(f.read_text()) for f in (args.base, args.head))
        doc = compare_records(base, head)
        doc["base"], doc["head"] = base["provenance"], head["provenance"]
        for rel, o in doc["objects"].items():
            verdict = "identical" if o["identical"] else "DIFFERS"
            why = "" if o["identical"] else f" {o.get('differ')} host {o.get('host_identical')}"
            print(f"{verdict}  {rel} ({o.get('kernels', 0)} kernels){why}")
        print("identical" if doc["identical"] else "DIFFERS")
        rc = 0 if doc["identical"] else 1
    else:
        doc = plant(args.build)
        for label, res in doc["results"].items():
            print(f"{label}: {len(res['differ'])} kernels differ {res['instructions']}")
        print("caught" if doc["caught"] else "NOT CAUGHT")
        rc = 0 if doc["caught"] else 1
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=1) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
