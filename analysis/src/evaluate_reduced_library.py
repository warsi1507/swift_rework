#!/usr/bin/env python3
"""Usage-ranked reduced supergate library evaluation.

Builds reduced .super libraries containing the top-K most-used
functions (ranked by canonical-group usage across mapped benchmark
circuits), then maps benchmark circuits with the full library and with
each reduced library. Only the mapping step is timed; per-circuit
I/O, AND count, levels, speedup, and QoR loss relative to the full
library are recorded in a CSV table.

The reduced library keeps, for every selected function, all of its
supergate lines in the original file, together with the transitive
closure of the lines they reference (kept as internal structure, not
as matchable supergates). Indices are renumbered to the kept lines so
the file stays a valid topologically-ordered library.

Usage:
    python3 evaluate_reduced_library.py --circuits div
    python3 evaluate_reduced_library.py [--k-values 500 1000]
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from truth_table_canon import Canonicalizer

# ---------------------------------------------------------------------------
# Directory configuration
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SRC_DIR.parent
PROJECT_ROOT = ANALYSIS_DIR.parent

ABC_BIN = PROJECT_ROOT / "abc" / "abc"
CIRCUITS_DIR = PROJECT_ROOT / "data" / "benchmarks"
LIB_DIR = PROJECT_ROOT / "data" / "library"

FULL_SUPER = LIB_DIR / "library.super"
FULL_GENLIB = LIB_DIR / "library.genlib"

USAGE_RANKING_CSV = ANALYSIS_DIR / "results" / "usage_stats" / "supergate_usage_analysis.csv"
GENERATION_JSON = ANALYSIS_DIR / "results" / "usage_stats" / "supergate_generation.json"

OUT_DIR = ANALYSIS_DIR / "results" / "reduced_library_eval"
LOG_DIR = OUT_DIR / "logs"
MAPPING_CSV = OUT_DIR / "mapping_results.csv"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Benchmark circuits of the two evaluation sets (one circuit is shared).
SEEN_CIRCUITS = [
    "div", "adder_apex7", "adder", "c6288", "arbiter", "alu4", "c3540",
    "c1908", "csela_orig", "b2", "apex3", "cavlc", "apex7", "bar", "ctrl",
]
UNSEEN_CIRCUITS = [
    "hyp", "div_alu2", "div_apex7", "div", "mem_ctrl_alu2", "mem_ctrl_apex7",
    "mem_ctrl", "mult_wallace_orig", "log2_alu2", "log2_apex7",
    "mult_dadda_orig", "log2", "multiplier_alu2", "multiplier_apex7",
    "multiplier",
]
ALL_CIRCUITS = sorted(set(SEEN_CIRCUITS) | set(UNSEEN_CIRCUITS))


# ---------------------------------------------------------------------------
# Genlib parsing
# ---------------------------------------------------------------------------

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
    of first appearance of the variables in the expression, matching
    the genlib reader.
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


# ---------------------------------------------------------------------------
# Supergate library reading, evaluation, and reduction
# ---------------------------------------------------------------------------

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


def cross_check_json(entries, evals):
    """Compares computed (tt, root) pairs against the generation dump.

    Returns (matched, total) or None if the dump is unavailable.
    """
    if not GENERATION_JSON.exists():
        return None
    counts = {}
    with GENERATION_JSON.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (int(d["truth_table"], 16), d["root_gate"])
            counts[key] = counts.get(key, 0) + 1
    matched = 0
    total = 0
    for (is_super, root, fans), (tt, n_pins) in zip(entries, evals):
        if not is_super:
            continue
        total += 1
        if counts.get((tt, root), 0) > 0:
            counts[(tt, root)] -= 1
            matched += 1
    return matched, total


def load_top_k_groups(ranking_csv, k):
    """Loads the top-K usage-ranked canonical groups.

    Returns {(n_vars, canonical_tt), ...} from the first K rows of the
    usage ranking CSV.
    """
    groups = set()
    with ranking_csv.open() as f:
        for row in csv.DictReader(f):
            if int(row["rank"]) > k:
                break
            groups.add((int(row["n_vars"]), int(row["canonical_tt"], 16)))
    return groups


def build_reduced_library(entries, evals, top_groups, canon, genlib_line,
                          n_vars, out_path):
    """Writes a reduced .super file containing the selected top-K groups.

    Every supergate line whose canonicalized truth table belongs to a
    selected group is kept, together with the transitive closure of the
    lines it references. Closure lines are written without the
    supergate mark so they cannot be matched directly. Indices are
    renumbered to the kept-line order.
    """
    selected = []
    for pos, ((is_super, root, fans), (tt, n_pins)) in enumerate(
            zip(entries, evals)):
        if is_super and canon.key(tt, n_pins) in top_groups:
            selected.append(pos)

    keep = set(selected)
    stack = list(selected)
    while stack:
        pos = stack.pop()
        for f in entries[pos][2]:
            if f >= n_vars:
                p = f - n_vars
                if p not in keep:
                    keep.add(p)
                    stack.append(p)

    order = sorted(keep)
    new_index = {f: f for f in range(n_vars)}
    new_index.update({n_vars + p: n_vars + i for i, p in enumerate(order)})

    lines = ["# Reduced supergate library (usage-ranked selection).",
             "# Selection keeps the top most-used functions with all of",
             "# their supergate lines; referenced lines are kept as",
             "# internal structure and are not matchable.", "#"]
    lines.append(genlib_line)
    lines.append(str(n_vars))
    n_star = sum(1 for p in order if p in selected)
    lines.append(str(n_star))
    lines.append(str(n_vars + len(order)))
    for p in order:
        is_super, root, fans = entries[p]
        mark = "* " if p in selected else ""
        new_fans = " ".join(str(new_index[f]) for f in fans)
        lines.append(f"{mark}{root} {new_fans}".rstrip())
    out_path.write_text("\n".join(lines) + "\n")

    realized = {canon.key(evals[p][0], evals[p][1]) for p in selected}
    return {
        "groups_requested": len(top_groups),
        "groups_realized": len(realized),
        "groups_missing": len(top_groups) - len(realized),
        "star_lines": n_star,
        "support_lines": len(order) - n_star,
        "total_lines": len(order),
    }


# ---------------------------------------------------------------------------
# Circuit statistics and mapping
# ---------------------------------------------------------------------------

def patch_super_genlib(src_path, dst_path, genlib_abs):
    """Copies a .super file with its genlib reference line replaced."""
    lines = src_path.read_text().splitlines()
    for i, s in enumerate(lines):
        t = s.strip()
        if t and not t.startswith("#"):
            lines[i] = genlib_abs
            break
    dst_path.write_text("\n".join(lines) + "\n")


def read_aig_header(path):
    """Reads input, output, and AND counts from an AIG header line."""
    with path.open("rb") as f:
        first = f.readline().decode("ascii", errors="replace").strip()
    parts = first.split()
    if len(parts) == 6 and parts[0] in ("aig", "aag"):
        return int(parts[2]), int(parts[4]), int(parts[5])
    raise ValueError(f"Unrecognized AIG header in {path}: {first!r}")


def parse_abc_metrics(output):
    """Extracts area, delay, levels, nodes, and edges from print_stats."""
    metrics = {"area": None, "delay": None, "levels": None,
               "nodes": None, "edges": None}
    mapped_line = None
    for line in reversed(output.splitlines()):
        if "area" in line.lower() and "delay" in line.lower():
            mapped_line = line.strip()
            break
    if mapped_line:
        m = re.search(
            r"nd\s*=\s*(\d+).*?edge\s*=\s*(\d+).*?"
            r"area\s*=\s*([0-9.]+).*?delay\s*=\s*([0-9.]+).*?lev\s*=\s*(\d+)",
            mapped_line, re.IGNORECASE)
        if m:
            metrics.update(nodes=int(m.group(1)), edges=int(m.group(2)),
                           area=float(m.group(3)), delay=float(m.group(4)),
                           levels=int(m.group(5)))
            return metrics
    for key, pattern in [("area", r"area\s*=\s*([0-9.]+)"),
                         ("delay", r"delay\s*=\s*([0-9.]+)"),
                         ("levels", r"lev\s*=\s*(\d+)"),
                         ("nodes", r"nd\s*=\s*(\d+)"),
                         ("edges", r"edge\s*=\s*(\d+)")]:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            metrics[key] = (float(matches[-1]) if key in ("area", "delay")
                            else int(matches[-1]))
    return metrics


def parse_map_time(output):
    """Returns the elapsed time between the two `time` commands (map only)."""
    elapses = re.findall(r"elapse:\s*([0-9.]+)\s*seconds", output)
    if len(elapses) < 2:
        return None
    return float(elapses[-1])


def run_mapping(circuit_path, super_lib, label, timeout):
    """Maps one circuit with the given library, timing only the map step."""
    script = (
        f"read {circuit_path.absolute()}\n"
        f"read_genlib {FULL_GENLIB.absolute()}\n"
        f"read_super {super_lib.absolute()}\n"
        f"time\nmap -v\ntime\nprint_stats\nquit\n"
    )
    log_path = LOG_DIR / f"{circuit_path.stem}_{label}.log"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".abc", delete=False) as tf:
        tf.write(script)
        script_path = Path(tf.name)
    try:
        proc = subprocess.run([str(ABC_BIN), "-f", str(script_path)],
                              capture_output=True, text=True, timeout=timeout)
        output = proc.stdout + "\n" + proc.stderr
        log_path.write_text(output)
        metrics = parse_abc_metrics(output)
        runtime = parse_map_time(output)
        return {
            "runtime_s": runtime,
            "area": metrics["area"], "delay": metrics["delay"],
            "levels": metrics["levels"], "nodes": metrics["nodes"],
            "edges": metrics["edges"],
            "success": (proc.returncode == 0 and metrics["area"] is not None
                        and metrics["delay"] is not None
                        and runtime is not None),
        }
    except subprocess.TimeoutExpired:
        log_path.write_text("TIMEOUT\n")
        return {"runtime_s": None, "area": None, "delay": None,
                "levels": None, "nodes": None, "edges": None,
                "success": False}
    finally:
        if script_path.exists():
            script_path.unlink()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-values", nargs="+", type=int, default=[500, 1000],
                        help="library sizes to evaluate (default: 500 1000)")
    parser.add_argument("--circuits", nargs="+", default=None,
                        help="circuit stems to evaluate (default: all)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="per-mapping timeout in seconds")
    args = parser.parse_args()

    circuits = args.circuits if args.circuits else ALL_CIRCUITS

    for path in (ABC_BIN, FULL_SUPER, FULL_GENLIB, USAGE_RANKING_CSV):
        if not path.exists():
            sys.exit(f"ERROR: missing {path}")

    gate_tables = parse_genlib(FULL_GENLIB)
    genlib_line, n_vars, entries = read_super_file(FULL_SUPER)
    evals = evaluate_entries(entries, n_vars, gate_tables)
    print(f"Parsed {len(entries)} gate lines ({n_vars} inputs) from "
          f"{FULL_SUPER.name}.")

    cross = cross_check_json(entries, evals)
    if cross is not None:
        matched, total = cross
        print(f"Generation-dump cross-check: {matched}/{total} lines matched "
              f"({100.0 * matched / total:.1f}%).")

    canon = Canonicalizer()
    full_lib_path = OUT_DIR / "full_library.super"
    patch_super_genlib(FULL_SUPER, full_lib_path, str(FULL_GENLIB.absolute()))
    lib_stats = {"json_crosscheck": ({"matched": cross[0], "total": cross[1]}
                                     if cross else None)}
    reduced_libs = {}
    for k in args.k_values:
        top_groups = load_top_k_groups(USAGE_RANKING_CSV, k)
        out_path = OUT_DIR / f"library_top{k}.super"
        stats = build_reduced_library(entries, evals, top_groups, canon,
                                      str(FULL_GENLIB.absolute()), n_vars,
                                      out_path)
        lib_stats[str(k)] = stats
        reduced_libs[k] = out_path
        print(f"K={k}: {stats['star_lines']} supergate lines, "
              f"{stats['support_lines']} support lines, "
              f"{stats['groups_missing']}/{stats['groups_requested']} "
              f"groups missing.")

    (OUT_DIR / "library_stats.json").write_text(
        json.dumps(lib_stats, indent=2))

    n_runs = len(circuits) * (1 + len(args.k_values))
    print(f"\nMapping {len(circuits)} circuits x "
          f"{1 + len(args.k_values)} libraries = {n_runs} runs.")

    rows = []
    for circuit in circuits:
        circuit_path = CIRCUITS_DIR / f"{circuit}.aig"
        if not circuit_path.exists():
            print(f"WARNING: {circuit}.aig not found; skipping.")
            continue
        inputs, outputs, ands = read_aig_header(circuit_path)
        print(f"\n=== {circuit} (I/O {inputs}/{outputs}, AND {ands}) ===")
        full = run_mapping(circuit_path, full_lib_path, "full", args.timeout)
        print(f"  full : runtime={full['runtime_s']}s area={full['area']} "
              f"delay={full['delay']} lev={full['levels']} "
              f"success={full['success']}")
        for k in args.k_values:
            res = run_mapping(circuit_path, reduced_libs[k], f"k{k}",
                              args.timeout)
            speedup = (full["runtime_s"] / res["runtime_s"]
                       if full["runtime_s"] and res["runtime_s"] else None)
            qor = (((res["area"] * res["delay"])
                    / (full["area"] * full["delay"]) - 1.0) * 100.0
                   if (full["area"] and full["delay"]
                       and res["area"] and res["delay"]) else None)
            rows.append({
                "circuit": circuit, "k": k,
                "inputs": inputs, "outputs": outputs, "ands": ands,
                "full_runtime_s": full["runtime_s"],
                "full_area": full["area"], "full_delay": full["delay"],
                "full_levels": full["levels"],
                "k_runtime_s": res["runtime_s"],
                "k_area": res["area"], "k_delay": res["delay"],
                "k_levels": res["levels"],
                "speedup": speedup,
                "qor_f_pct": qor,
                "success": bool(full["success"] and res["success"]),
            })
            speedup_s = (f"{speedup:.2f}x" if speedup is not None else "n/a")
            qor_s = (f"{qor:.2f}%" if qor is not None else "n/a")
            print(f"  k={k:<4}: runtime={res['runtime_s']}s "
                  f"area={res['area']} delay={res['delay']} "
                  f"lev={res['levels']} speedup={speedup_s} qor={qor_s} "
                  f"success={res['success']}")

    csv_path = MAPPING_CSV
    fields = ["circuit", "k", "inputs", "outputs", "ands",
              "full_runtime_s", "full_area", "full_delay", "full_levels",
              "k_runtime_s", "k_area", "k_delay", "k_levels",
              "speedup", "qor_f_pct", "success"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
