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
    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = plt.cm.tab20.colors
    circuit_names = sorted(df['circuit'].unique())
    for i, (c_name, grp) in enumerate(df.groupby('circuit')):
        grp = grp.sort_values('K')
        ax.plot(grp['K'], grp['qor_loss'], marker='o', markersize=3.5, linewidth=1.2,
                alpha=0.8, color=colors[i % len(colors)], zorder=2)
    mean = df.groupby('K')['qor_loss'].mean().reset_index().sort_values('K')
    ax.plot(mean['K'], mean['qor_loss'], marker='s', linewidth=2.5,
            color='black', label='Average QoR loss', zorder=3)
    ax.set_xlabel('K (Top-K Selected Supergates)', fontsize=12)
    ax.set_ylabel('QoR Loss (%)', fontsize=12)
    ax.set_title('QoR Loss vs K for Top-15 Largest Circuits', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 50)
    ax.set_xticks(range(0, 2001, 250))
    ax.set_yticks(range(0, 51, 10))
    handles = [plt.Line2D([0], [0], color=colors[i % len(colors)], linewidth=1.2, label=name)
               for i, name in enumerate(circuit_names)]
    handles.append(plt.Line2D([0], [0], color='black', linewidth=2.5, label='Average QoR loss'))
    ax.legend(handles=handles, ncol=2, fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.text(0.5, 0.01,
             'Figure 11: QoR loss for 15 considered circuits with increase in '
             'Supergate Library size (K).',
             ha='center', va='bottom', fontsize=10)
    fig.subplots_adjust(bottom=0.12)
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_out, dpi=200)
    plt.close(fig)
    print(f"Plot saved to: {plot_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='ml/checkpoints/best_model.pth')
    parser.add_argument('--k-values', type=str, default='100,200,300,500,1000,2000')
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
