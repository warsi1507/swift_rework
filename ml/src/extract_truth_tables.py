#!/usr/bin/env python3
"""
ml/src/extract_truth_tables.py
Evaluates the DAG structure of a .super library against the cell library
to produce per-supergate truth tables and physical metrics (area, max
delay, input count). Physical metrics come from the ABC generation dump;
truth tables are computed bottom-up from genlib gate expressions.
Generates: ml/data/all_supergates_truth_tables.csv
"""

import re
import csv
import json
import argparse
from pathlib import Path


def _tokenize(expr):
    return re.findall(r"[A-Za-z]\w*|[!*+()]", expr)


def _eval_expr(expr, env):
    """Evaluates a genlib function expression with variables mapped to 0/1."""
    toks = _tokenize(expr)
    pos = [0]

    def parse_or():
        value = parse_and()
        while pos[0] < len(toks) and toks[pos[0]] == "+":
            pos[0] += 1
            right = parse_and()
            value = value or right
        return value

    def parse_and():
        value = parse_unary()
        while pos[0] < len(toks) and toks[pos[0]] == "*":
            pos[0] += 1
            right = parse_unary()
            value = value and right
        return value

    def parse_unary():
        if pos[0] < len(toks) and toks[pos[0]] == "!":
            pos[0] += 1
            return not parse_unary()
        if pos[0] < len(toks) and toks[pos[0]] == "(":
            pos[0] += 1
            value = parse_or()
            if pos[0] < len(toks) and toks[pos[0]] == ")":
                pos[0] += 1
            return value
        if pos[0] < len(toks):
            value = env.get(toks[pos[0]], 0)
            pos[0] += 1
            return value
        return 0

    return bool(parse_or())


def parse_genlib(path):
    """Parses genlib gate expressions into truth tables.

    Returns {gate_name: (n_pins, tt)} where tt bit i is the output for
    input assignment i (pin k is input bit k). Pin order is the order
    of first appearance of the variables in the expression.
    """
    text = path.read_text()
    gates = {}
    for m in re.finditer(r"GATE\s+(\S+)\s+[0-9.]+\s+Y=([^;]+);", text):
        name, expr = m.group(1), m.group(2)
        vars_ = []
        for tok in re.findall(r"[A-Za-z]\w*", expr):
            if tok not in vars_:
                vars_.append(tok)
        if not vars_:
            gates[name] = (0, 1 if "1" in expr else 0)
            continue
        tt = 0
        for a in range(1 << len(vars_)):
            env = {v: (a >> i) & 1 for i, v in enumerate(vars_)}
            if _eval_expr(expr, env):
                tt |= 1 << a
        gates[name] = (len(vars_), tt)
    return gates


def read_super_file(path):
    """Parses a .super file into its header and gate lines.

    Returns (genlib_line, n_vars, entries) where entries is a list of
    (is_super, root_name, fanin_indices) in file order. Fanin indices
    use the file's numbering: PIs are 0..n_vars-1 and the gate at
    0-based position k has index n_vars + k.
    """
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s and not s.startswith("#"):
            break
        i += 1
    genlib_line = lines[i].strip()
    n_vars = int(lines[i + 1].strip())
    i += 4  # skip n_vars and the two declared-count lines that follow it
    entries = []
    for line in lines[i:]:
        s = line.strip()
        if not s or s.startswith("#") or s == "\x00":
            continue
        parts = s.split()
        if parts[0] == "*":
            is_super, root = True, parts[1]
            fans = [int(p) for p in parts[2:] if p.isdigit()]
        else:
            is_super, root = False, parts[0]
            fans = [int(p) for p in parts[1:] if p.isdigit()]
        entries.append((is_super, root, fans))
    return genlib_line, n_vars, entries


def evaluate_entries(entries, n_vars, gate_tables):
    """Computes the truth table of every gate line, bottom-up.

    Returns a list parallel to entries of (tt, n_pins). Raises
    ValueError if a root gate is unknown or the fanin count does not
    match the root gate's pin count.
    """
    tts = [
        sum(1 << a for a in range(1 << n_vars) if (a >> k) & 1)
        for k in range(n_vars)
    ]
    out = []
    for pos, (is_super, root, fans) in enumerate(entries):
        if root not in gate_tables:
            raise ValueError(f"Unknown root gate {root!r} at line {pos}")
        n_pins, gtt = gate_tables[root]
        if len(fans) != n_pins:
            raise ValueError(
                f"Fanin count mismatch for {root} at line {pos}: "
                f"got {len(fans)}, expected {n_pins}"
            )
        tt = 0
        for b in range(1 << n_vars):
            a = 0
            for k, f in enumerate(fans):
                if (tts[f] >> b) & 1:
                    a |= 1 << k
            if (gtt >> a) & 1:
                tt |= 1 << b
        tts.append(tt)
        out.append((tt, n_pins))
    return out


def load_generation_metadata(path):
    """Loads the ABC generation dump keyed by (truth_table, root_gate).

    Returns {(tt, root): [(num_inputs, area, max_delay), ...]} with one
    entry per recorded occurrence, since the same pair can repeat.
    """
    meta = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = (int(d["truth_table"], 16), d["root_gate"])
            rec = (int(d["num_inputs"]), float(d["area"]),
                   float(d["max_delay"]))
            meta.setdefault(key, []).append(rec)
    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate .super DAG logic into truth tables with "
                    "area/delay metrics from the generation dump")
    parser.add_argument("--super", type=Path,
                        default="data/library/library.super")
    parser.add_argument("--genlib", type=Path,
                        default="data/library/library.genlib")
    parser.add_argument("--generation-json", type=Path,
                        default="analysis/results/usage_stats/"
                                "supergate_generation.json")
    parser.add_argument("--output", type=Path,
                        default="ml/data/all_supergates_truth_tables.csv")
    args = parser.parse_args()

    for p in (args.super, args.genlib, args.generation_json):
        if not p.exists():
            raise SystemExit(f"ERROR: missing input file: {p}")

    gate_tables = parse_genlib(args.genlib)
    print(f"Parsed {len(gate_tables)} gates from {args.genlib.name}.")

    _, n_vars, entries = read_super_file(args.super)
    print(f"Parsed {len(entries)} DAG lines ({n_vars} inputs) from "
          f"{args.super.name}.")

    evals = evaluate_entries(entries, n_vars, gate_tables)
    meta = load_generation_metadata(args.generation_json)
    print(f"Loaded {sum(len(v) for v in meta.values())} generation "
          f"metadata records.")

    rows = []
    matched = total = 0
    for pos, ((is_super, root, fans), (tt, n_pins)) in enumerate(
            zip(entries, evals)):
        if not is_super:
            continue
        total += 1
        recs = meta.get((tt, root), [])
        if recs:
            num_inputs, area, delay = recs.pop()
            matched += 1
        else:
            raise ValueError(
                f"No generation metadata for supergate at DAG line "
                f"{pos} (root {root}, tt 0x{tt:08x})")
        rows.append({
            "supergate_id": pos,
            "truth_table_hex": f"0x{tt:08x}",
            "num_inputs": num_inputs,
            "area": f"{area:.4f}",
            "delay": f"{delay:.4f}",
        })

    print(f"Generation-dump cross-check: {matched}/{total} star lines "
          f"matched.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "supergate_id", "truth_table_hex", "num_inputs", "area",
            "delay"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} truth table rows to {args.output}")


if __name__ == "__main__":
    main()
