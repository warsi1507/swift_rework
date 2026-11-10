#!/usr/bin/env python3
"""Supergate library generation and usage-data collection.

Converts the ASAP7 library to genlib format, generates the supergate
library with the instrumented ABC binary, maps all benchmark circuits,
and records per-circuit supergate usage logs plus generation metadata.
"""

import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import argparse

# ==========================================
# 1. Directory Structure Configuration
# ==========================================
SRC_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SRC_DIR.parent
PROJECT_ROOT = ANALYSIS_DIR.parent

ABC_BIN = PROJECT_ROOT / "abc" / "abc"
CIRCUITS_DIR = PROJECT_ROOT / "data" / "benchmarks"
LIB_DIR = PROJECT_ROOT / "data" / "library"
LIB_FILE = LIB_DIR / "7nm.lib"

# Output directories specific to analysis
RESULTS_DIR = ANALYSIS_DIR / "results"
MAPPED_DIR = RESULTS_DIR / "mapped_blifs"
LOGS_DIR = RESULTS_DIR / "logs"
STATS_DIR = RESULTS_DIR / "usage_stats"
PLOTS_DIR = ANALYSIS_DIR / "plots"

# Temporary ABC scripts directory
TMP_DIR = ANALYSIS_DIR / "tmp"

# Ensure all directories exist
for dir_path in [LIB_DIR, MAPPED_DIR, LOGS_DIR, STATS_DIR, PLOTS_DIR, TMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class SupergateDataCollector:
    def __init__(self, verbose=False, force_all=False):
        self.verbose = verbose
        self.force_all = force_all
        self.circuits = []
        self.supergate_data = {}
        self.usage_data = defaultdict(lambda: defaultdict(int))
        
    def log(self, message):
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def find_circuits(self, pattern="*.aig", limit=150):
        """Locates all circuit files in the benchmarks directory."""
        self.circuits = list(CIRCUITS_DIR.rglob(pattern))[:limit]
        self.log(f"Found {len(self.circuits)} circuits in {CIRCUITS_DIR}")
        return self.circuits
    
    def generate_supergate_library(self, n_vars=5, n_levels=3, delay_max=100.0, area_max=100.0, gates_max=10000):
        """Converts .lib to .genlib and generates the .super library."""
        genlib_file = LIB_DIR / "library.genlib"
        super_file = LIB_DIR / "library.super"
        gen_json = STATS_DIR / "supergate_generation.json"

        # Skip supergate precomputation if already present unless force_all is specified
        if not self.force_all and super_file.exists() and gen_json.exists() and gen_json.stat().st_size > 0:
            self.log("Supergate library already exists. Skipping precomputation.")
            self._load_supergate_data()
            return True

        self.log("Generating supergate library...")

        # Step 1: Convert .lib to .genlib
        self.log("Converting .lib to .genlib format...")
        script_file_gen = TMP_DIR / "gen_genlib.abc"
        script_file_gen.write_text(f"read_lib {LIB_FILE.absolute()}\nwrite_genlib {genlib_file.absolute()}\nquit\n")
        
        try:
            subprocess.run([str(ABC_BIN), "-f", str(script_file_gen)], capture_output=True, text=True, timeout=60)
        except Exception as e:
            self.log(f"ERROR converting to genlib: {e}")
            return False
            
        # Step 2: Generate supergates
        self.log(f"Generating supergates (vars={n_vars}, levels={n_levels})...")
        script_file_super = TMP_DIR / "gen_super.abc"
        script_file_super.write_text(
            f"super -I {n_vars} -L {n_levels} -D {delay_max} -A {area_max} -N {gates_max} -v {genlib_file.absolute()}\nquit\n"
        )
        
        env = os.environ.copy()
        env['SUPERGATE_LOG_FILE'] = str(gen_json)
        
        try:
            result = subprocess.run([str(ABC_BIN), "-f", str(script_file_super)], capture_output=True, text=True, timeout=300, env=env)
            (LOGS_DIR / "supergate_generation.log").write_text(result.stdout + "\n" + result.stderr)
            self._load_supergate_data()
            return True
        except Exception as e:
            self.log(f"ERROR: Supergate generation failed: {e}")
            return False

    def _load_supergate_data(self):
        """Loads structural metadata of generated supergates from the ABC JSON dump."""
        log_file = STATS_DIR / "supergate_generation.json"
        if not log_file.exists():
            self.log("WARNING: No supergate generation log found.")
            return
            
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    tt = data.get('truth_table')
                    if tt:
                        self.supergate_data[tt] = data
        self.log(f"Loaded {len(self.supergate_data)} supergates from generation log.")
    
    def map_circuit(self, circuit_path):
        """Runs technology mapping for a single circuit and extracts usage data."""
        circuit_name = circuit_path.stem
        usage_file = STATS_DIR / f"usage_{circuit_name}.json"

        # Skip mapping if log already exists and force_all is False
        if not self.force_all and usage_file.exists() and usage_file.stat().st_size > 0:
            self.log(f"Skipping {circuit_name}: already processed.")
            self._load_usage_data(circuit_name)
            return True

        self.log(f"Mapping circuit: {circuit_name}")
        
        super_file = LIB_DIR / "library.super"
        genlib_file = LIB_DIR / "library.genlib"
        mapped_blif = MAPPED_DIR / f"{circuit_name}_mapped.blif"
        
        script_content = f"read {circuit_path.absolute()}\n"
        if super_file.exists() and genlib_file.exists():
            script_content += f"read_genlib {genlib_file.absolute()}\nread_super {super_file.absolute()}\n"
        else:
            script_content += f"read_lib {LIB_FILE.absolute()}\n"
            
        script_content += f"map -v\nwrite_blif {mapped_blif.absolute()}\nprint_stats\nquit\n"
        
        script_file = TMP_DIR / f"map_{circuit_name}.abc"
        script_file.write_text(script_content)
        
        env = os.environ.copy()
        env['SUPERGATE_USAGE_LOG'] = str(usage_file)
        env['CURRENT_CIRCUIT'] = circuit_name
        
        try:
            result = subprocess.run([str(ABC_BIN), "-f", str(script_file)], capture_output=True, text=True, timeout=900, env=env)
            (LOGS_DIR / f"{circuit_name}.log").write_text(result.stdout + "\n" + result.stderr)
            self._load_usage_data(circuit_name)
            return True
        except Exception as e:
            self.log(f"ERROR: Mapping failed for {circuit_name}: {e}")
            return False

    def _load_usage_data(self, circuit_name):
        """Loads runtime usage frequencies dumped by ABC's Map_MatchNodeCut."""
        usage_file = STATS_DIR / f"usage_{circuit_name}.json"
        if not usage_file.exists():
            return
            
        with open(usage_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    gate_id = data.get('supergate_id')
                    if gate_id:
                        self.usage_data[gate_id][circuit_name] += data.get('count', 1)

    def generate_dataset(self):
        """Merges structural metadata and usage frequencies into a final CSV."""
        self.log("Generating final dataset...")
        
        usage_stats = {}
        for truth_table, circuits in self.usage_data.items():
            usage_stats[truth_table] = {
                'usage_count': sum(circuits.values()),
                'num_circuits': len(circuits)
            }
            
        dataset = []
        for gate_id, gate_info in self.supergate_data.items():
            usage = usage_stats.get(gate_id, {'usage_count': 0, 'num_circuits': 0})
            dataset.append({
                'TruthTable': gate_id,
                'NumInputs': gate_info.get('num_inputs', 0),
                'Area': gate_info.get('area', 0.0),
                'MaxDelay': gate_info.get('max_delay', 0.0),
                'RootGate': gate_info.get('root_gate', ''),
                'GenerationLevel': gate_info.get('generation_level', 0),
                'UsageCount': usage['usage_count'],
                'NumCircuitsUsed': usage['num_circuits']
            })
            
        df = pd.DataFrame(dataset).sort_values('UsageCount', ascending=False)
        output_file = STATS_DIR / "supergate_dataset.csv"
        df.to_csv(output_file, index=False)
        
        stats = {
            'total_supergates_generated': len(df),
            'supergates_used': len(df[df['UsageCount'] > 0]),
            'total_lookups': int(df['UsageCount'].sum())
        }
        with open(STATS_DIR / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
            
        return df

    def generate_plots(self, df):
        """Generates analytical plots."""
        if df.empty:
            return
        self.log("Generating plots in plots/ directory...")
        sns.set_style("whitegrid")
        
        # 1. Cumulative usage
        plt.figure(figsize=(10, 6))
        sorted_usage = df[df['UsageCount'] > 0]['UsageCount'].sort_values(ascending=False).values
        if len(sorted_usage) > 0:
            cumsum_pct = (np.cumsum(sorted_usage) / np.sum(sorted_usage)) * 100
            plt.plot(range(1, len(cumsum_pct) + 1), cumsum_pct, linewidth=2, color='darkblue')
            plt.axhline(y=80, color='orange', linestyle='--', label='80% of usage')
            plt.xlabel('Number of Supergates (sorted by usage)')
            plt.ylabel('Cumulative Usage (%)')
            plt.title('Cumulative Usage Distribution')
            plt.legend()
            plt.savefig(PLOTS_DIR / 'cumulative_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Area vs Delay Map
        plt.figure(figsize=(10, 6))
        used = df[df['UsageCount'] > 0]
        if not used.empty:
            plt.scatter(used['Area'], used['MaxDelay'], alpha=0.6, s=30, c='green')
            plt.xlabel('Area')
            plt.ylabel('Max Delay')
            plt.title('Area vs Delay Distribution (Used Supergates)')
            plt.savefig(PLOTS_DIR / 'area_vs_delay.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run_pipeline(self):
        self.find_circuits()
        if not self.generate_supergate_library():
            return False
        for circuit in self.circuits:
            self.map_circuit(circuit)
        df = self.generate_dataset()
        self.generate_plots(df)
        self.log("Empirical Pipeline Complete.")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Empirical Data Pipeline')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-a', '--all', dest='all', action='store_true', help='Force run for all circuits (re-execute previously completed runs)')
    args = parser.parse_args()
    
    if not ABC_BIN.exists():
        print(f"ERROR: ABC binary not found at {ABC_BIN}. Compile it first.")
        sys.exit(1)
        
    collector = SupergateDataCollector(verbose=args.verbose, force_all=args.all)
    collector.run_pipeline()