#!/usr/bin/env python3
"""
ml/src/plot_timing_overhead.py
Measures mapping runtimes across different library scales and stacks GNN inference latency.
Generates: ml/plots/timing_breakdown.png
"""

import os
import re
import glob
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def get_and_nodes(abc_bin: str, circuit: str) -> int:
    cmd = [abc_bin, "-c", f"read_aiger {circuit}; print_stats"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r'and\s*=\s*(\d+)', res.stdout)
    return int(m.group(1)) if m else 0


def count_supergates(super_path: str) -> int:
    with open(super_path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    return int(lines[2]) if len(lines) >= 4 else 0


def patch_super_genlib(src: str, dst: str, genlib: str) -> str:
    lines = Path(src).read_text(encoding="utf-8", errors="ignore").splitlines()
    for i, s in enumerate(lines):
        t = s.strip()
        if t and not t.startswith("#"):
            lines[i] = str(Path(genlib).resolve())
            break
    Path(dst).write_text("\n".join(lines) + "\n")
    return dst


def run_map_timing(abc_bin: str, genlib: str, superlib: str, circuit: str) -> float:
    script = f"""read_aiger {circuit}
read_library {Path(genlib).resolve()}
read_super {superlib}
echo __MAP_START__
time
map
time
echo __MAP_END__
quit
"""
    res = subprocess.run([abc_bin], input=script, capture_output=True, text=True)
    out = res.stdout + res.stderr
    # The last marker pair belongs to this run; ABC also echoes earlier
    # sessions' commands from abc.history at startup.
    segments = re.findall(r'__MAP_START__(.*?)__MAP_END__', out, re.DOTALL)
    if segments:
        elapsed = re.findall(r'elapse:\s*([\d.]+)\s*seconds', segments[-1])
        if elapsed:
            return float(elapsed[-1])
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Generate Timing Breakdown Plot")
    parser.add_argument('--abc', type=str, default="abc/abc")
    parser.add_argument('--genlib', type=str, default="data/library/library.genlib")
    parser.add_argument('--super-dir', type=str, default="data/library")
    parser.add_argument('--reduced-dir', type=str, default="analysis/results/reduced_library_eval")
    parser.add_argument('--circuits-dir', type=str, default="data/benchmarks")
    parser.add_argument('--plot-out', type=str, default="ml/plots/timing_breakdown.png")
    parser.add_argument('--max-circuits', type=int, default=50)
    parser.add_argument('--inf-time', type=float, default=None,
                        help="GNN inference latency (seconds) stacked onto reduced-library runtimes; "
                             "defaults to the mean inf_time of ml/results/seen_k500_results.csv if present")
    args = parser.parse_args()

    if args.inf_time is None:
        eval_csv = Path("ml/results/seen_k500_results.csv")
        if eval_csv.exists():
            args.inf_time = float(pd.read_csv(eval_csv)["inf_time"].mean())
            print(f"Using measured GNN inference latency from {eval_csv}: {args.inf_time:.3f}s")
        else:
            args.inf_time = 0.0
            print(f"Warning: {eval_csv} not found; using 0.0s GNN inference latency")

    libraries = [
        ("Small", f"{args.reduced_dir}/library_top500.super", args.inf_time),
        ("Medium", f"{args.reduced_dir}/library_top1000.super", args.inf_time),
        ("Full", f"{args.super_dir}/library.super", 0.0)
    ]

    tmp_lib_dir = Path("ml/results/tmp_libs")
    tmp_lib_dir.mkdir(parents=True, exist_ok=True)

    lib_specs = []
    for label, super_path, inf_time in libraries:
        if not os.path.exists(super_path):
            print(f"Warning: missing {super_path}; skipping {label}")
            continue
        patched = patch_super_genlib(super_path, str(tmp_lib_dir / Path(super_path).name), args.genlib)
        n_sg = count_supergates(patched)
        lib_specs.append((f"{label} ({n_sg})", patched, inf_time))

    all_circuits = sorted(glob.glob(f"{args.circuits_dir}/*.aig"), key=os.path.getsize, reverse=True)
    circuits = all_circuits[:args.max_circuits]

    print(f"Benchmarking {len(circuits)} circuits for timing breakdown...")
    results = []

    for idx, c in enumerate(circuits, 1):
        name = Path(c).stem
        nodes = get_and_nodes(args.abc, c)
        print(f"[{idx}/{len(circuits)}] Measuring {name:<20} ({nodes} nodes)")

        row = {'name': name, 'nodes': nodes}
        for lib_label, super_path, inf_time in lib_specs:
            t_map = run_map_timing(args.abc, args.genlib, super_path, c)
            row[f'{lib_label}_map'] = t_map
            row[f'{lib_label}_total'] = t_map + inf_time

        results.append(row)

    df = pd.DataFrame(results).sort_values('nodes')

    out_path = Path(args.plot_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel (a): Mapping time vs circuit size
    ax1 = axes[0]
    markers = ['x', '^', 'o'][:len(lib_specs)]
    colors_a = ['black', 'goldenrod', 'green'][:len(lib_specs)]
    for (lib_label, _, _), mk, col in zip(lib_specs, markers, colors_a):
        ax1.scatter(df['nodes'], df[f'{lib_label}_map'], color=col, marker=mk, label=lib_label, s=30)
    ax1.set_xlabel('Circuit Size (AND nodes)', fontsize=11)
    ax1.set_ylabel('Mapping Time (seconds)', fontsize=11)
    ax1.set_title('(a) Mapping Time vs Circuit Size', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel (b): Total mapping time across circuits, with GNN inference latency stacked
    ax2 = axes[1]
    labels = [spec[0].replace(' (', '\n(') for spec in lib_specs]
    totals = [df[f'{spec[0]}_map'].sum() for spec in lib_specs]
    inf_totals = [spec[2] * len(df) for spec in lib_specs]
    colors = ['green', 'goldenrod', 'red'][:len(lib_specs)]
    bars = ax2.bar(labels, totals, color=colors, width=0.55, edgecolor='black',
                   label='mapping time')
    inf_bars = ax2.bar(labels, inf_totals, bottom=totals, color='gray', width=0.55,
                       edgecolor='black', label='GNN inference')
    ax2.set_ylabel('Total Mapping Time (seconds)', fontsize=11)
    ax2.set_title('(b) Total Mapping Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    for bar, val, inf_val in zip(bars, totals, inf_totals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + inf_val + 2.0, f"{val + inf_val:.2f}s",
                 ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"\nTiming plot successfully saved to: {out_path}")


if __name__ == "__main__":
    main()