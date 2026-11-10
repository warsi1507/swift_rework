#!/usr/bin/env python3
"""
ml/src/build_dataset.py
Constructs the labeled binary training dataset (Label=1 for used gates, 0 for unused gates).
Generates: ml/data/supergate_usage_dataset.csv
"""

import os
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict


def normalize_tt(tt_hex: str) -> str:
    s = tt_hex.strip()
    if s.lower().startswith('0x'):
        s = s[2:]
    return f"0x{s.lower().zfill(8)}"


def build_dataset(truth_table_csv: Path, usage_dir: Path, output_csv: Path):
    if not truth_table_csv.exists():
        raise FileNotFoundError(f"Missing master truth table file: {truth_table_csv}")
    if not usage_dir.exists():
        raise FileNotFoundError(f"Missing usage statistics directory: {usage_dir}")

    print(f"Loading master supergate library from: {truth_table_csv}")
    master_gates = []
    with open(truth_table_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            master_gates.append({
                'supergate_id': int(row['supergate_id']),
                'truth_table_hex': normalize_tt(row['truth_table_hex']),
                'num_inputs': int(row['num_inputs']),
                'area': float(row['area']),
                'delay': float(row['delay'])
            })
    print(f"Loaded {len(master_gates)} master supergates.")

    usage_files = list(usage_dir.glob("usage_*.json"))
    if not usage_files:
        raise FileNotFoundError(f"No per-circuit JSON files found in: {usage_dir}")

    print(f"Found {len(usage_files)} per-circuit usage logs.")
    usage_map = defaultdict(lambda: defaultdict(int))
    circuit_paths = []

    for uf in usage_files:
        c_name = uf.stem.replace("usage_", "")
        c_path = f"data/benchmarks/{c_name}.aig"
        circuit_paths.append(c_path)

        with open(uf, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tt_hex = rec.get("supergate_id")
                if not tt_hex:
                    continue
                count = int(rec.get("count", 0))
                if count > 0:
                    usage_map[c_path][normalize_tt(str(tt_hex))] += count

    circuit_paths.sort()
    print(f"Aggregating dataset across {len(circuit_paths)} circuits...")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ['AIG_Path', 'Supergate_ID', 'Truth_Table_Hex', 'Num_Inputs', 'Area', 'Delay_Max', 'Usage_Count', 'Label']

    label_1_count = 0
    total_rows = 0

    with open(output_csv, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(header)

        for c_idx, c_path in enumerate(circuit_paths, 1):
            c_usage = usage_map[c_path]
            for sg in master_gates:
                norm_tt = sg['truth_table_hex']
                count = c_usage.get(norm_tt, 0)
                label = 1 if count > 0 else 0

                if label == 1:
                    label_1_count += 1

                writer.writerow([
                    c_path,
                    sg['supergate_id'],
                    sg['truth_table_hex'],
                    sg['num_inputs'],
                    f"{sg['area']:.4f}",
                    f"{sg['delay']:.4f}",
                    count,
                    label
                ])
                total_rows += 1

            if c_idx % 20 == 0 or c_idx == len(circuit_paths):
                print(f"  Processed {c_idx}/{len(circuit_paths)} circuits ({total_rows:,} rows)")

    print(f"\nTraining dataset successfully created: {output_csv}")
    print(f"  Total Rows:     {total_rows:,}")
    print(f"  Positive (1):   {label_1_count:,} ({100*label_1_count/total_rows:.2f}%)")
    print(f"  Negative (0):   {total_rows - label_1_count:,} ({100*(total_rows - label_1_count)/total_rows:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Expand usage data into labeled binary dataset")
    parser.add_argument('--truth-table', type=str, default="ml/data/all_supergates_truth_tables.csv")
    parser.add_argument('--usage-dir', type=str, default="analysis/results/usage_stats")
    parser.add_argument('--output', type=str, default="ml/data/supergate_usage_dataset.csv")
    args = parser.parse_args()

    build_dataset(Path(args.truth_table), Path(args.usage_dir), Path(args.output))


if __name__ == "__main__":
    main()