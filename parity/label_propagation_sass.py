#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The build-flags go/no-go of label_propagation (M10a step 1; PLAN 6.4.7, 7.4, 8.6; ADR 0066):
O's six pipeline translation units compiled from a fresh `git archive a276a3a` twice, with O's
Makefile line and with the nvcc line dynG generates for a FAST_MATH CUDA TU under the parity-cuda
preset, and the SASS of every function compared.

    parity/label_propagation_sass.py compile --archive DIR --dyng-commands compile_commands.json
                                             --probe <path of the FAST_MATH probe .cu> --out DIR
    parity/label_propagation_sass.py compare --out DIR [--json FILE]
    parity/label_propagation_sass.py port --original-obj DIR --port-obj DIR --graph-obj FILE
                                          [--json FILE]

`compile` takes O's line from `make -n -B obj/<tu>.o` in the archive (the Makefile's defaults) and
dynG's from the compile_commands.json of a scratch configure (`cmake --preset parity-cuda` with a
throw-away `dyng_add_algorithm(... FAST_MATH)` target, whose .cu entry is --probe): every flag of
dynG's line is kept, its source and object replaced by O's, and O's include directory (-Iheaders)
added (O's sources need it; it changes no code generation). `compare` reads `cuobjdump -sass` and
`cuobjdump -res-usage` of both objects of each TU and compares every function: the instruction
text with only the address comments removed (the strictest normalization: a function is equal
only when its machine code is), plus the registers, stack, shared and local memory. It records
both command lines, the per-function instruction counts and digests, the cuobjdump version, and
exits 1 when any function differs or exists in one object only.

`port` (M10a step 4; criterion 12; ADR 0075): the fused engine's kernels in dyng_label_propagation's
objects of a parity-cuda build (--port-obj, the int32 instantiations) against the same kernels in
O's unpatched objects (--original-obj, O's obj/), named by PORT_KERNELS: per kernel the sequence of
floating-point instructions (every F*, D*, MUFU and H* opcode with its modifiers, in program
order: the arithmetic of the hot loops) must be equal; the whole instruction text, the instruction
text with the constant-bank operands c[0x0][...] blanked (the kernel parameters' offsets) and the
resources (registers, stack, shared and local memory) are recorded, with the differences listed.
And the graph module's device apply (--graph-obj, apply_slotted_device.cu.o) must hold no
floating-point instruction at all (it moves weight bits only). Exit 1 if either fails.
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
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# O's six pipeline translation units (PLAN 6.4.7, the step's list).
TUS = (
    "dynamicGraph",
    "sparsifySubgraph",
    "connectedComponents",
    "labelInitialization",
    "iterativePropagation",
    "validation",
)


def cuobjdump() -> str:
    for c in [shutil.which("cuobjdump"), "/usr/local/cuda-13.1/bin/cuobjdump"]:
        if c and Path(c).exists():
            return c
    raise SystemExit("cuobjdump not found")


def original_line(archive: Path, tu: str) -> list[str]:
    out = subprocess.run(
        ["make", "-n", "-B", f"obj/{tu}.o"], cwd=archive, capture_output=True, text=True, check=True
    ).stdout
    line = next((x for x in out.splitlines() if f"-o obj/{tu}.o" in x), None)
    if line is None:
        raise SystemExit(f"no compile line for obj/{tu}.o in `make -n`")
    return shlex.split(line)


def dyng_line(commands: Path, probe: str) -> list[str]:
    for e in json.loads(commands.read_text()):
        if e["file"].endswith(probe):
            return shlex.split(e["command"])
    raise SystemExit(f"{probe}: not in {commands}")


def retarget(line: list[str], source: str, obj: str, include: str) -> list[str]:
    """dynG's line for O's TU: its flags kept, the source and object replaced, -I<headers>
    added before the first -I."""
    out: list[str] = []
    skip = False
    added = False
    for i, tok in enumerate(line):
        if skip:
            skip = False
            continue
        if tok == "-o":
            out += ["-o", obj]
            skip = True
            continue
        if tok == "-c":
            out += ["-c", source]
            skip = True
            continue
        if tok.startswith("-I") and not added:
            out.append(f"-I{include}")
            added = True
        out.append(tok)
        if i == 0:
            continue
    return out


def compile_cmd(args: argparse.Namespace) -> int:
    archive = args.archive.resolve()
    out = args.out.resolve()
    (out / "original").mkdir(parents=True, exist_ok=True)
    (out / "dyng").mkdir(parents=True, exist_ok=True)
    lines = {"original": {}, "dyng": {}}
    base = dyng_line(args.dyng_commands, args.probe)
    for tu in TUS:
        o = original_line(archive, tu)
        o = [x if x != f"obj/{tu}.o" else str(out / "original" / f"{tu}.o") for x in o]
        o = [x for x in o if x not in ("-MMD", "-MP")]  # dependency files only
        subprocess.run(o, cwd=archive, check=True)
        d = retarget(
            base,
            str(archive / "src" / f"{tu}.cu"),
            str(out / "dyng" / f"{tu}.o"),
            str(archive / "headers"),
        )
        subprocess.run(d, cwd=archive, check=True)
        lines["original"][tu] = shlex.join(o)
        lines["dyng"][tu] = shlex.join(d)
        print(f"{tu}: compiled both ways", flush=True)
    (out / "lines.json").write_text(json.dumps(lines, indent=1) + "\n")
    return 0


def sass(obj: Path) -> dict[str, list[str]]:
    text = subprocess.run(
        [cuobjdump(), "-sass", str(obj)], capture_output=True, text=True, check=True
    ).stdout
    funcs: dict[str, list[str]] = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;", line)
        if cur and m:
            funcs[cur].append(re.sub(r"\s+", " ", m.group(1)).strip())
    return funcs


def resources(obj: Path) -> dict[str, dict]:
    text = subprocess.run(
        [cuobjdump(), "-res-usage", str(obj)], capture_output=True, text=True, check=True
    ).stdout
    res: dict[str, dict] = {}
    name = None
    for line in text.splitlines():
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            name = m.group(1)
            continue
        if name and "REG:" in line:
            res[name] = {k: int(v) for k, v in re.findall(r"(\w+):(\d+)", line)}
            name = None
    return res


def compare_cmd(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    lines = json.loads((out / "lines.json").read_text())
    record: dict = {
        "schema": 1,
        "what": "O's six pipeline TUs compiled with O's Makefile line and with dynG's FAST_MATH "
        "nvcc line under parity-cuda; every function's SASS (address comments removed) and "
        "resources compared (M10a step 1, ADR 0066)",
        "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cuobjdump": subprocess.run([cuobjdump(), "--version"], capture_output=True, text=True)
        .stdout.strip()
        .splitlines()[-1],
        "lines": lines,
        "tus": {},
    }
    differ = []
    for tu in TUS:
        a, b = out / "original" / f"{tu}.o", out / "dyng" / f"{tu}.o"
        sa, sb = sass(a), sass(b)
        ra, rb = resources(a), resources(b)
        funcs = {}
        for name in sorted(set(sa) | set(sb)):
            same = sa.get(name) == sb.get(name) and ra.get(name) == rb.get(name)
            funcs[name] = {
                "identical": same,
                "instructions": [len(sa.get(name, [])), len(sb.get(name, []))],
                "sha256": [
                    hashlib.sha256("\n".join(x.get(name, [])).encode()).hexdigest()
                    for x in (sa, sb)
                ],
                "resources": [ra.get(name), rb.get(name)],
            }
            if not same:
                differ.append(f"{tu}: {name}")
        record["tus"][tu] = {"functions": funcs, "count": len(funcs)}
        print(
            f"{tu}: {len(funcs)} functions, "
            f"{sum(1 for f in funcs.values() if not f['identical'])} differ",
            flush=True,
        )
    record["identical"] = not differ
    record["differ"] = differ
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=1) + "\n")
    print("SASS identical in every function" if not differ else f"DIFFER: {differ}")
    return 0 if not differ else 1


# O's kernel -> the port's int32 instantiation (ADR 0075's rename map).
PORT_KERNELS = {
    "markDeletionsKernel": "mark_deletions_kernel<int>",
    "buildSubgraphKernel": "build_subgraph_kernel",
    "markInsertedKernel": "mark_inserted_kernel",
    "initParentKernel": "init_parent_kernel",
    "hookKernel": "hook_kernel",
    "compressKernel": "compress_kernel",
    "markRootsKernel": "mark_roots_kernel",
    "numberRootsKernel": "number_roots_kernel",
    "assignComponentIdsKernel": "assign_component_ids_kernel",
    "computeSupernodeWeightsKernel": "compute_supernode_weights_kernel<int>",
    "initializeLabelsKernel": "initialize_labels_kernel",
    "initFrontierKernel": "init_frontier_kernel<int>",
    "propagateGroupKernel<8>": "propagate_group_kernel<8, int>",
    "propagateGroupKernel<32>": "propagate_group_kernel<32, int>",
    "propagateBlockKernel": "propagate_block_kernel<int>",
    "commitLabelsKernel": "commit_labels_kernel",
    "smallFrontierKernel": "small_frontier_kernel<int>",
    "enforceGroundTruthKernel": "enforce_ground_truth_kernel",
    "irLPKernel": "ir_lp_kernel<int>",
    "irLPEnforceGTKernel": "ir_lp_enforce_gt_kernel",
}
O_TUS = ("sparsifySubgraph", "connectedComponents", "labelInitialization",
         "iterativePropagation", "validation")  # fmt: skip
PORT_TUS = ("sparsify", "components", "label_init", "propagation", "irlp")
# The floating-point instructions: FP32 / FP64 / FP16 arithmetic, compares and selects (F*, D*,
# H*), MUFU, and the conversions from integers (I2F, I2FP; F2I and F2F are F*).
FP_OPCODE = re.compile(r"^(F[A-Z0-9]*|D[A-Z]+|MUFU|H[A-Z0-9]+|I2F[A-Z0-9]*)(\.|$)")
INTEGER_LOOKALIKES = ("FLO",)  # find leading one: an integer instruction


def demangle(names: list[str]) -> dict[str, str]:
    if not names:
        return {}
    out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True)
    return dict(zip(names, out.stdout.splitlines(), strict=True))


def base_name(demangled: str) -> str:
    """'void ns::(anonymous namespace)::kernel<8, int>(args)' -> 'kernel<8, int>'."""
    text = demangled.replace("(anonymous namespace)::", "")
    if text.startswith("void "):
        text = text[5:]
    depth, end = 0, len(text)
    for i, ch in enumerate(text):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            end = i
            break
    head = text[:end]
    depth, start = 0, 0
    for i, ch in enumerate(head):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == ":" and depth == 0 and i + 1 < len(head) and head[i + 1] == ":":
            start = i + 2
    return head[start:].strip()


def fp_sequence(instructions: list[str], operands: bool = False) -> list[str]:
    """The floating-point instructions of a kernel in order: their opcodes, or with `operands`
    the whole instructions (the predicate guard and the operands, the parameter offsets blanked:
    the association of the arithmetic, which the opcodes alone do not show; M10a review)."""
    out = []
    for ins in instructions:
        text = re.sub(r"^@!?U?P[T0-9]+\s+", "", ins)  # the predicate guard
        opcode = text.split(" ")[0]
        if FP_OPCODE.match(opcode) and opcode.split(".")[0] not in INTEGER_LOOKALIKES:
            out.append(blank_params([ins])[0] if operands else opcode)
    return out


def blank_params(instructions: list[str]) -> list[str]:
    return [re.sub(r"c\[0x[0-9a-f]+\]\[0x[0-9a-f]+\]", "c[*][*]", x) for x in instructions]


def kernels_of(objs: list[Path]) -> tuple[dict, dict]:
    code: dict[str, list[str]] = {}
    res: dict[str, dict] = {}
    for obj in objs:
        s, r = sass(obj), resources(obj)
        names = demangle(sorted(s))
        for mangled, ins in s.items():
            code[base_name(names[mangled])] = ins
            if mangled in r:
                res[base_name(names[mangled])] = r[mangled]
    return code, res


def port_cmd(args: argparse.Namespace) -> int:
    o_code, o_res = kernels_of([args.original_obj / f"{t}.o" for t in O_TUS])
    p_code, p_res = kernels_of([args.port_obj / f"{t}.cu.o" for t in PORT_TUS])
    kernels, bad = {}, []
    for o_name, p_name in PORT_KERNELS.items():
        a, b = o_code.get(o_name), p_code.get(p_name)
        if a is None or b is None:
            bad.append(f"{o_name}: missing ({'original' if a is None else 'port'})")
            continue
        fa, fb = fp_sequence(a, operands=True), fp_sequence(b, operands=True)
        same_fp = fa == fb
        same_code = blank_params(a) == blank_params(b)
        if not same_fp:
            bad.append(
                f"{o_name}: the floating-point instructions differ (opcodes, operands or order)"
            )
        if not same_code:
            # The go/no-go is the SASS instruction for instruction (ADR 0075 section 1): a
            # difference outside the floating-point instructions (a shuffle offset, an integer
            # index) changes what the arithmetic reads, so it fails too.
            bad.append(f"{o_name}: the SASS differs beyond the parameter offsets")
        kernels[o_name] = {
            "port": p_name,
            "fp_sequence_equal": same_fp,
            "fp_opcodes_equal": fp_sequence(a) == fp_sequence(b),
            "fp_instructions": [len(fa), len(fb)],
            "fp_sha256": [hashlib.sha256("\n".join(x).encode()).hexdigest() for x in (fa, fb)],
            "identical": a == b,
            "identical_but_parameter_offsets": same_code,
            "instructions": [len(a), len(b)],
            "resources": [o_res.get(o_name), p_res.get(p_name)],
            "resources_equal": o_res.get(o_name) == p_res.get(p_name),
        }
    # The graph module's device apply: no floating-point instruction.
    g_code, _ = kernels_of([args.graph_obj])
    graph_fp = {k: fp_sequence(v) for k, v in g_code.items() if fp_sequence(v)}
    if graph_fp:
        bad.append(f"the graph module's device apply has floating-point code in {sorted(graph_fp)}")
    record = {
        "schema": 1,
        "what": "the fused engine's kernels against LabelPropagation-CUDA@a276a3a's (criterion 12; "
        "ADR 0075): per kernel the SASS equal but for the parameter offsets and the "
        "floating-point instructions equal with their operands; the whole SASS and the resources "
        "recorded; and no floating-point instruction (conversions included) in the graph "
        "module's device apply",
        "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cuobjdump": subprocess.run([cuobjdump(), "--version"], capture_output=True, text=True)
        .stdout.strip()
        .splitlines()[-1],
        "original_objects": str(args.original_obj),
        "port_objects": str(args.port_obj),
        "kernels": kernels,
        "graph_device_apply": {
            "object": str(args.graph_obj),
            "functions": len(g_code),
            "with_floating_point": graph_fp,
        },
        "equal": not bad,
        "problems": bad,
    }
    for name, k in kernels.items():
        print(
            f"{name}: fp {'equal' if k['fp_sequence_equal'] else 'DIFFERENT'} "
            f"({k['fp_instructions'][0]} instructions), whole SASS "
            f"{'identical' if k['identical'] else 'differs'}"
            f"{'' if k['identical'] or not k['identical_but_parameter_offsets'] else ' (offs)'}, "
            f"resources {'equal' if k['resources_equal'] else k['resources']}",
            flush=True,
        )
    print(f"graph device apply: {len(g_code)} functions, floating point in {sorted(graph_fp)}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=1) + "\n")
    print("port SASS: PASSED" if not bad else f"port SASS: FAILED {bad}")
    return 0 if not bad else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--archive", type=Path, required=True)
    c.add_argument("--dyng-commands", type=Path, required=True)
    c.add_argument("--probe", required=True)
    c.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("compare")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--json", type=Path)
    p = sub.add_parser("port")
    p.add_argument("--original-obj", type=Path, required=True, help="O's unpatched obj/")
    p.add_argument("--port-obj", type=Path, required=True,
                   help="dyng_label_propagation.dir of a parity-cuda build")  # fmt: skip
    p.add_argument("--graph-obj", type=Path, required=True, help="apply_slotted_device.cu.o")
    p.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    return {"compile": compile_cmd, "compare": compare_cmd, "port": port_cmd}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
