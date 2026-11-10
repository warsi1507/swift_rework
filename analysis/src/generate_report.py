#!/usr/bin/env python3
"""Markdown report for reduced-library mapping results.

Reads the per-circuit CSV written by evaluate_reduced_library.py and
compiles a Markdown report with per-circuit rows and group averages
for the seen and unseen circuit sets. Every number in the report is
computed from the CSV; nothing is hardcoded.

Usage:
    python3 generate_report.py
    python3 generate_report.py --csv path/to/mapping_results.csv
"""

import argparse
import csv
from pathlib import Path

from evaluate_reduced_library import (SEEN_CIRCUITS, UNSEEN_CIRCUITS,
                                      MAPPING_CSV)

SRC_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SRC_DIR.parent


def load_rows(csv_path):
    rows = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def fmt(value, ndigits=2, suffix=""):
    if value is None:
        return "n/a"
    return f"{float(value):.{ndigits}f}{suffix}"


def circuit_row(row, k):
    speedup = row.get("speedup")
    qor = row.get("qor_f_pct")
    success = row.get("success") == "True"
    if not success:
        return ["n/a", "n/a"]
    return [f"{float(speedup):.2f}x" if speedup else "n/a",
            f"{float(qor):+.2f}%" if qor else "n/a"]


def build_table(rows, circuits, k):
    by_circuit = {}
    for row in rows:
        if int(row["k"]) == k:
            by_circuit[row["circuit"]] = row

    header = ["Circuit", "I/O", "AND", "Lev", "Speedup", "QoR-F (%)"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]

    speedups, qors = [], []
    for circuit in circuits:
        row = by_circuit.get(circuit)
        if row is None:
            lines.append(f"| {circuit} | n/a | n/a | n/a | n/a | n/a |")
            continue
        io = f"{row['inputs']}/{row['outputs']}"
        speedup_s, qor_s = circuit_row(row, k)
        lines.append(f"| {circuit} | {io} | {row['ands']} | "
                     f"{row['full_levels']} | {speedup_s} | {qor_s} |")
        if row.get("success") == "True":
            if row.get("speedup"):
                speedups.append(float(row["speedup"]))
            if row.get("qor_f_pct"):
                qors.append(float(row["qor_f_pct"]))

    def avg(values):
        return sum(values) / len(values) if values else None

    lines.append(f"| Average | | | | {fmt(avg(speedups), 2, 'x')} | "
                 f"{fmt(avg(qors), 2, '%')} |")
    return "\n".join(lines), avg(speedups), avg(qors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=MAPPING_CSV,
                        help="mapping results CSV (default: reduced-library "
                             "evaluation output)")
    parser.add_argument("--out", type=Path, default=None,
                        help="report path (default: next to the CSV)")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: {args.csv} not found. Run evaluate_reduced_library.py "
              f"first.")
        return

    rows = load_rows(args.csv)
    k_values = sorted({int(row["k"]) for row in rows})
    n_failed = sum(1 for row in rows if row.get("success") != "True")
    n_total = len(rows)

    report = ["# Reduced-library mapping results", "",
              f"Source: `{args.csv}`",
              f"Runs: {n_total} (failed: {n_failed})",
              "Library sizes: " +
              ", ".join(f"K={k}" for k in k_values), "",
              "QoR-F (%) is the change in area-delay product relative to "
              "mapping with the full library; Speedup is the full-library "
              "mapping time divided by the reduced-library mapping time.",
              ""]

    for k in k_values:
        report.append(f"## Seen circuits (K={k})")
        table, avg_s, avg_q = build_table(rows, SEEN_CIRCUITS, k)
        report.append(table)
        report.append("")
        report.append(f"## Unseen circuits (K={k})")
        table, avg_s, avg_q = build_table(rows, UNSEEN_CIRCUITS, k)
        report.append(table)
        report.append("")

    out_path = args.out or (args.csv.parent / "mapping_report.md")
    out_path.write_text("\n".join(report))
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
