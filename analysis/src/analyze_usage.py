#!/usr/bin/env python3
"""Supergate usage analysis: aggregates per-circuit usage logs.

Inputs (analysis/results/usage_stats/):
  - usage_<circuit>.json          raw-TT usage counts per circuit (C logger)
  - supergate_generation.json     raw TT -> {num_inputs, area, max_delay, root_gate}

Outputs (analysis/results/usage_stats/):
  - circuit_supergate_frequency_table.csv   circuits x raw gates (usage counts)
  - supergate_usage_analysis.csv            canonical-level usage ranking
  - supergate_usage_metadata.csv            raw gate -> area/delay/usage
  - top_50_usage.csv                        top-50 canonical groups
  - cumulative_usage.csv                    cumulative usage curve
  - pie_shares.json                         top-10/11-30/31-50/others usage shares
  - statistics.json                         high-level usage statistics

Canonicalization follows ABC's mapper (mapperCanon.c): minimum truth table
over all input-phase assignments.  Groups are keyed by (num_inputs, canon_tt).
No numbers are hardcoded; everything is computed from the data.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from truth_table_canon import Canonicalizer

BASE_DIR = Path(__file__).resolve().parents[1]
STATS_DIR = BASE_DIR / "results" / "usage_stats"
LIBRARY_SUPER = BASE_DIR.parent / "data" / "library" / "library.super"

REPO_ROOT_NOTE = str(Path(__file__).resolve().parents[2])


def log(msg):
    print(msg, flush=True)


def load_generation_metadata():
    meta = {}
    with open(STATS_DIR / "supergate_generation.json") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tt = int(d["truth_table"], 16)
            meta[tt] = {
                "num_inputs": int(d["num_inputs"]),
                "area": float(d["area"]),
                "delay": float(d["max_delay"]),
                "root_gate": d.get("root_gate", ""),
            }
    return meta


def load_usage():
    """Return {circuit_name: Counter(tt -> count)} and files read."""
    usage = {}
    n_lines = 0
    for path in sorted((STATS_DIR).glob("usage_*.json")):
        circuit = path.stem[len("usage_"):]
        counter = Counter()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                counter[int(d["supergate_id"], 16)] += int(d["count"])
                n_lines += 1
        usage[circuit] = counter
    return usage, n_lines


def count_super_lines():
    with open(LIBRARY_SUPER) as f:
        return sum(1 for line in f if line.startswith("* "))


def build_frequency_table(usage):
    circuits = sorted(usage)
    all_tts = set()
    for counter in usage.values():
        all_tts.update(counter)
    totals = Counter({tt: sum(c.get(tt, 0) for c in usage.values()) for tt in all_tts})
    ordered = [tt for tt, _ in totals.most_common()]
    with open(STATS_DIR / "circuit_supergate_frequency_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["circuit"] + [f"0x{tt:08x}" for tt in ordered])
        for circuit in circuits:
            counter = usage[circuit]
            writer.writerow([circuit] + [counter.get(tt, 0) for tt in ordered])
    return totals, ordered, circuits


def build_canonical_ranking(usage, meta, totals):
    """Aggregate raw-TT usage into canonical groups keyed by (n_vars, canon_tt)."""
    canon = Canonicalizer()
    group_usage = defaultdict(int)
    group_tts = defaultdict(set)
    for tt, count in totals.items():
        if tt not in meta:
            continue
        key = canon.key(tt, meta[tt]["num_inputs"])
        group_usage[key] += count
        group_tts[key].add(tt)
    ranked = sorted(group_usage.items(), key=lambda kv: -kv[1])
    return ranked, group_tts


def spearman(x, y):
    """Rank-based Spearman correlation with average ranks for ties."""
    n = len(x)
    if n < 2:
        return float("nan")

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j + 2) / 2
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = (n + 1) / 2, (n + 1) / 2
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def write_usage_metadata(meta, totals):
    with open(STATS_DIR / "supergate_usage_metadata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["truth_table", "num_inputs", "area", "delay", "root_gate", "usage_count"])
        for tt, count in totals.items():
            m = meta[tt]
            writer.writerow([f"0x{tt:08x}", m["num_inputs"], m["area"], m["delay"], m["root_gate"], count])


def compute_spearman(meta, totals, usage):
    used = [tt for tt in totals if tt in meta]
    areas = [meta[tt]["area"] for tt in used]
    delays = [meta[tt]["delay"] for tt in used]
    global_unweighted = spearman(areas, delays)

    per_circuit = []
    for circuit, counter in usage.items():
        tts = [tt for tt in counter if tt in meta]
        if len(tts) < 2:
            continue
        per_circuit.append(spearman([meta[tt]["area"] for tt in tts], [meta[tt]["delay"] for tt in tts]))
    pc = {"mean": sum(per_circuit) / len(per_circuit) if per_circuit else None,
          "min": min(per_circuit) if per_circuit else None,
          "max": max(per_circuit) if per_circuit else None,
          "n_circuits": len(per_circuit)}
    return global_unweighted, pc


def main():
    log("=" * 70)
    log("SUPERGATE USAGE ANALYSIS")
    log("=" * 70)

    n_super = count_super_lines()
    meta = load_generation_metadata()
    usage, n_lines = load_usage()
    log(f"library.super entries: {n_super}")
    log(f"generation metadata entries: {len(meta)}")
    log(f"usage files: {len(usage)}")
    log(f"usage JSON records: {n_lines}")

    missing = {tt for c in usage.values() for tt in c} - set(meta)
    if missing:
        log(f"WARNING: {len(missing)} used TTs missing from metadata (dropped)")

    totals, ordered, circuits = build_frequency_table(usage)
    total_usage = sum(totals.values())
    log(f"total usage events: {total_usage:,}")
    log(f"distinct raw gates used: {len(totals)} ({100 * len(totals) / n_super:.1f}% of library)")

    ranked, group_tts = build_canonical_ranking(usage, meta, totals)
    n_groups = len(ranked)
    log(f"distinct canonical functions used: {n_groups}")

    # canonical-level usage analysis CSV
    with open(STATS_DIR / "supergate_usage_analysis.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "n_vars", "canonical_tt", "usage_count", "share", "cumulative_share",
                         "n_raw_gates", "example_raw_tts"])
        cum = 0
        for rank, (key, count) in enumerate(ranked, 1):
            share = count / total_usage
            cum += share
            tts = sorted(group_tts[key])
            writer.writerow([rank, key[0], f"0x{key[1]:08x}", count, f"{share:.6f}",
                             f"{cum:.6f}", len(tts), ";".join(f"0x{t:08x}" for t in tts[:5])])

    # cumulative curve data
    with open(STATS_DIR / "cumulative_usage.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "usage_count", "cumulative_share"])
        cum = 0
        for rank, (_, count) in enumerate(ranked, 1):
            cum += count
            writer.writerow([rank, count, cum / total_usage])

    # thresholds at 50/80/90%
    thresholds = {}
    cum = 0
    for rank, (_, count) in enumerate(ranked, 1):
        cum += count
        pct = cum / total_usage
        for target in (0.50, 0.80, 0.90):
            if target not in thresholds and pct >= target:
                thresholds[target] = rank
    log(f"coverage thresholds: { {int(t * 100): k for t, k in thresholds.items()} }")

    # top-50 (normalized by the max usage, so the top bar = 1.0)
    top50 = ranked[:50]
    max_usage = top50[0][1]
    with open(STATS_DIR / "top_50_usage.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "n_vars", "canonical_tt", "usage_count", "normalized_usage"])
        for rank, (key, count) in enumerate(top50, 1):
            writer.writerow([rank, key[0], f"0x{key[1]:08x}", count, f"{count / max_usage:.6f}"])

    # pie shares
    def share(a, b):
        return sum(c for _, c in ranked[a:b]) / total_usage

    pie = {
        "top_10": share(0, 10),
        "top_11_30": share(10, 30),
        "top_31_50": share(30, 50),
        "others": share(50, n_groups),
        "n_total_groups": n_groups,
    }
    with open(STATS_DIR / "pie_shares.json", "w") as f:
        json.dump(pie, f, indent=2)
    log(f"pie shares: { {k: f'{v * 100:.1f}%' for k, v in pie.items() if k != 'n_total_groups'} }")

    write_usage_metadata(meta, totals)

    global_spearman, per_circuit = compute_spearman(meta, totals, usage)
    log(f"Spearman(area, delay) over used gates: {global_spearman:.4f}")
    log(f"Spearman(area, delay) per circuit: mean {per_circuit['mean']:.4f} "
        f"(min {per_circuit['min']:.4f}, max {per_circuit['max']:.4f}, n={per_circuit['n_circuits']})")

    per_circuit_usage = [sum(c.values()) for c in usage.values()]
    per_circuit_distinct = [len(c) for c in usage.values()]
    statistics = {
        "library": {
            "super_lines_in_file": n_super,
            "generation_metadata_entries": len(meta),
            "raw_gates_used": len(totals),
            "pct_of_library_used": 100 * len(totals) / n_super,
        },
        "usage": {
            "n_circuits": len(usage),
            "total_usage_events": total_usage,
            "per_circuit_mean": sum(per_circuit_usage) / len(per_circuit_usage),
            "per_circuit_min": min(per_circuit_usage),
            "per_circuit_max": max(per_circuit_usage),
            "per_circuit_distinct_gates_mean": sum(per_circuit_distinct) / len(per_circuit_distinct),
            "per_circuit_distinct_gates_min": min(per_circuit_distinct),
            "per_circuit_distinct_gates_max": max(per_circuit_distinct),
        },
        "canonical": {
            "n_groups_used": n_groups,
            "top_10_share": pie["top_10"],
            "top_11_30_share": pie["top_11_30"],
            "top_31_50_share": pie["top_31_50"],
            "top_50_share": pie["top_10"] + pie["top_11_30"] + pie["top_31_50"],
            "others_share": pie["others"],
            "k_for_50pct": thresholds.get(0.50),
            "k_for_80pct": thresholds.get(0.80),
            "k_for_90pct": thresholds.get(0.90),
        },
        "spearman": {
            "global_area_delay_unweighted": global_spearman,
            "per_circuit_mean": per_circuit["mean"],
            "per_circuit_min": per_circuit["min"],
            "per_circuit_max": per_circuit["max"],
            "per_circuit_n": per_circuit["n_circuits"],
        },
    }
    with open(STATS_DIR / "statistics.json", "w") as f:
        json.dump(statistics, f, indent=2)

    log("outputs written to " + str(STATS_DIR))
    log("=" * 70)


if __name__ == "__main__":
    main()
