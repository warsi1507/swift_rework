"""
ml/src/sweep_qor_vs_k.py
Sweeps the reduced-library size K for a set of circuits and plots the
QoR (area-delay product) loss relative to the full library mapping.
For each K, the library combines the top-100 global-frequency core
gates with model-scored candidates from the non-core set.
Generates: ml/results/qor_vs_k_results.csv and ml/plots/qor_vs_k.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from inference import SupergateSelector
from abc_driver import ABCDriver
from supergate_lib import patch_genlib_line
from evaluate import SEEN_CIRCUITS
from dataset import UNSEEN_CIRCUITS


def plot_qor_vs_k(df: pd.DataFrame, plot_out: Path):
    plt.figure(figsize=(10, 6))
    for c_name, grp in df.groupby('circuit'):
        grp = grp.sort_values('K')
        plt.plot(grp['K'], grp['qor_loss'], marker='o', markersize=3, linewidth=1,
                 alpha=0.5, color='tab:gray')
    mean = df.groupby('K')['qor_loss'].mean().reset_index().sort_values('K')
    plt.plot(mean['K'], mean['qor_loss'], marker='s', linewidth=2.5,
             color='black', label='Average QoR loss')
    plt.xlabel('Library Size K', fontsize=12)
    plt.ylabel('QoR Loss (%)', fontsize=12)
    plt.title('QoR Loss vs Library Size K', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_out, dpi=200)
    plt.close()
    print(f"Plot saved to: {plot_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='ml/checkpoints/best_model.pth')
    parser.add_argument('--k-values', type=str, default='100,200,300,500')
    parser.add_argument('--unseen', action='store_true', help="run on the unseen circuit set")
    parser.add_argument('--abc', type=str, default='abc/abc')
    parser.add_argument('--genlib', type=str, default='data/library/library.genlib')
    parser.add_argument('--full-super', type=str, default='data/library/library.super')
    parser.add_argument('--supergates-csv', type=str, default='ml/data/all_supergates_truth_tables.csv')
    parser.add_argument('--freq-csv', type=str,
                        default='analysis/results/usage_stats/circuit_supergate_frequency_table.csv')
    parser.add_argument('--results-out', type=str, default='ml/results/qor_vs_k_results.csv')
    parser.add_argument('--plot-out', type=str, default='ml/plots/qor_vs_k.png')
    args = parser.parse_args()

    k_values = [int(x.strip()) for x in args.k_values.split(',') if x.strip()]
    circuits = UNSEEN_CIRCUITS if args.unseen else SEEN_CIRCUITS

    driver = ABCDriver(args.abc)
    selector = SupergateSelector(
        checkpoint_path=args.checkpoint,
        supergates_csv=args.supergates_csv,
        global_freq_csv=args.freq_csv,
        source_super_file=args.full_super,
        genlib_path=args.genlib,
    )

    tmp_dir = Path('ml/results/tmp_libs')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    full_super_patched = patch_genlib_line(
        Path(args.full_super), tmp_dir / 'full_library_patched.super', Path(args.genlib))

    print(f"\nSweeping K in {k_values} over {len(circuits)} circuits\n")

    rows = []
    for c_name in circuits:
        aig_path = f'data/benchmarks/{c_name}.aig'
        if not Path(aig_path).exists():
            continue

        full_res = driver.run_mapping(aig_path, args.genlib, str(full_super_patched))
        if not full_res.success:
            print(f"  [skip] {c_name}: full-library mapping failed: {full_res.error}")
            continue

        for k in k_values:
            model_ids, _ = selector.select_supergates(aig_path, k=k, m_core=100)
            lib_path = tmp_dir / f'sweep_k{k}_{c_name}.super'
            selector.generate_library_file(model_ids, str(lib_path))
            res = driver.run_mapping(aig_path, args.genlib, str(lib_path))
            if not res.success:
                print(f"  [skip] {c_name} K={k}: mapping failed: {res.error}")
                continue
            qor_loss = (res.adp - full_res.adp) / full_res.adp * 100.0
            rows.append({'circuit': c_name, 'K': k, 'qor_loss': qor_loss})
            print(f"{c_name:<18} K={k:<5} QoR loss={qor_loss:7.2f}%")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No results collected.")
        return
    df.to_csv(args.results_out, index=False)
    print(f"\nSaved results to: {args.results_out}")

    plot_qor_vs_k(df, Path(args.plot_out))


if __name__ == '__main__':
    main()
