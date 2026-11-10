"""
ml/src/evaluate.py
Evaluates frequency-based vs model-based supergate selection on the
seen and unseen circuit sets.
"""

import os
import argparse
import pandas as pd
from pathlib import Path

from inference import SupergateSelector
from abc_driver import ABCDriver
from dataset import UNSEEN_CIRCUITS
from supergate_lib import patch_genlib_line

SEEN_CIRCUITS = [
    'div', 'adder_apex7', 'adder', 'c6288', 'arbiter',
    'alu4', 'c3540', 'c1908', 'csela_orig', 'b2',
    'apex3', 'cavlc', 'apex7', 'bar', 'ctrl'
]


def evaluate_circuits(circuits: list, selector: SupergateSelector, driver: ABCDriver, 
                      genlib: str, full_super: str, k: int = 500):
    rows = []
    tmp_dir = Path("ml/results/tmp_libs")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    full_super_patched = patch_genlib_line(Path(full_super), tmp_dir / "full_library_patched.super", Path(genlib))
    
    print(f"\n{'Circuit':<18} | {'AND':<7} | {'Lev':<5} | {'Full(s)':<8} | {'Spd(F)':<7} | {'QoR-f%':<8} | {'Spd(M)':<7} | {'QoR-m%':<8}")
    print("-" * 88)
    
    for c_name in circuits:
        aig_path = f"data/benchmarks/{c_name}.aig"
        if not os.path.exists(aig_path):
            continue
            
        full_res = driver.run_mapping(aig_path, genlib, str(full_super_patched))
        if not full_res.success:
            print(f"  [skip] {c_name}: full-library mapping failed: {full_res.error}")
            continue
            
        # Frequency-based selection: top-K by global usage
        top_k_freq = selector.global_rankings[:k]
        freq_lib = tmp_dir / f"freq_{k}_{c_name}.super"
        selector.generate_library_file(top_k_freq, str(freq_lib))
        freq_res = driver.run_mapping(aig_path, genlib, str(freq_lib))
        
        # Model-based selection: 100-gate global core plus model-scored candidates
        model_ids, inf_time = selector.select_supergates(aig_path, k=k, m_core=100)
        model_lib = tmp_dir / f"model_{k}_{c_name}.super"
        selector.generate_library_file(model_ids, str(model_lib))
        model_res = driver.run_mapping(aig_path, genlib, str(model_lib))
        
        if not freq_res.success or not model_res.success:
            print(f"  [skip] {c_name}: reduced-library mapping failed: "
                  f"freq={freq_res.error!r}, model={model_res.error!r}")
            continue
        
        # Area-Delay Product degradation relative to full library
        qor_freq = ((freq_res.adp - full_res.adp) / full_res.adp) * 100.0
        qor_model = ((model_res.adp - full_res.adp) / full_res.adp) * 100.0
        
        # Speedup (including ML inference latency)
        spd_freq = full_res.map_time / freq_res.map_time if freq_res.map_time > 0 else 1.0
        spd_model = full_res.map_time / (model_res.map_time + inf_time) if (model_res.map_time + inf_time) > 0 else 1.0
        
        print(f"{c_name:<18} | {full_res.num_gates:<7} | {full_res.num_levels:<5} | {full_res.map_time:<8.2f} | "
              f"{spd_freq:<7.2f} | {qor_freq:<8.2f} | {spd_model:<7.2f} | {qor_model:<8.2f}")
              
        rows.append({
            'circuit': c_name, 'nodes': full_res.num_gates, 'levels': full_res.num_levels,
            'full_time': full_res.map_time, 'spd_freq': spd_freq, 'qor_freq': qor_freq,
            'spd_model': spd_model, 'qor_model': qor_model, 'inf_time': inf_time
        })
        
    df = pd.DataFrame(rows)
    print("-" * 88)
    if df.empty:
        print("No circuits evaluated successfully.")
        return df
    print(f"{'AVERAGE':<18} | {'-':<7} | {'-':<5} | {df['full_time'].mean():<8.2f} | "
          f"{df['spd_freq'].mean():<7.2f} | {df['qor_freq'].mean():<8.2f} | "
          f"{df['spd_model'].mean():<7.2f} | {df['qor_model'].mean():<8.2f}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unseen', action='store_true', help="run on the unseen circuit set")
    parser.add_argument('--k', type=int, default=500, choices=[500, 1000])
    parser.add_argument('--checkpoint', type=str, default='ml/checkpoints/best_model.pth')
    parser.add_argument('--genlib', type=str, default='data/library/library.genlib')
    parser.add_argument('--full-super', type=str, default='data/library/library.super')
    parser.add_argument('--supergates-csv', type=str, default='ml/data/all_supergates_truth_tables.csv')
    parser.add_argument('--freq-csv', type=str, default='analysis/results/usage_stats/circuit_supergate_frequency_table.csv')
    args = parser.parse_args()

    driver = ABCDriver("abc/abc")
    selector = SupergateSelector(
        checkpoint_path=args.checkpoint,
        supergates_csv=args.supergates_csv,
        global_freq_csv=args.freq_csv,
        source_super_file=args.full_super,
        genlib_path=args.genlib
    )

    circuits = UNSEEN_CIRCUITS if args.unseen else SEEN_CIRCUITS
    mode = "Unseen circuits" if args.unseen else "Seen circuits"
    print(f"\n================ Evaluating {mode} with K={args.k} ================")
    
    res_df = evaluate_circuits(circuits, selector, driver, args.genlib, args.full_super, k=args.k)
    out_csv = f"ml/results/{'unseen' if args.unseen else 'seen'}_k{args.k}_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved results to: {out_csv}")


if __name__ == '__main__':
    main()