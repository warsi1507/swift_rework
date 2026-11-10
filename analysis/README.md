# Empirical Analysis Pipeline

This directory contains the automated pipeline for the empirical analysis: supergate usage profiling, Pareto concentration, heatmap generation, runtime speedup, and QoR trade-offs.

---

## 1. Source Files Overview (`src/`)

Listed in execution order: usage-data collection -> usage analysis -> figures, then QoR baseline -> reduced-library sweep -> report.

* **`collect_usage_data.py`**
Converts the ASAP7 cell library into Genlib/Supergate format, orchestrates technology mapping across all benchmark circuits using the custom C-instrumented ABC, and records per-circuit supergate usage logs plus generation metadata.
* **`analyze_usage.py`**
Aggregates the per-circuit usage logs and generation metadata into the Circuit-vs-Supergate frequency table, canonical-level usage statistics (ABC-faithful phase canonicalization), coverage thresholds, pie shares, and Spearman correlations.
* **`plot_usage_figures.py`**
Renders the usage figures (circuit-vs-supergate usage heatmap, cumulative usage curve, top-50 bars, usage pie, area-delay heatmap) with the established scales and labelling.
* **`truth_table_canon.py`**
ABC-faithful truth-table canonicalization (minimum over all input-phase assignments, per `mapperCanon.c`). Imported by `analyze_usage.py`. Self-tests against a brute-force reference: `python3 truth_table_canon.py`.
* **`qor_baseline.py`**
Maps benchmark circuits against the standard unaugmented `7nm.lib` to extract baseline area, delay, ADP, node count, logic levels, and runtime metrics.
* **`evaluate_reduced_library.py`**
Builds reduced supergate libraries containing the top-K most-used functions (usage-ranked canonical groups plus the transitive closure of referenced lines), maps benchmark circuits with the full and each reduced library, and records per-circuit runtime speedups and Area-Delay Product (ADP) changes.
* **`generate_report.py`**
Reads the per-circuit CSV produced by `evaluate_reduced_library.py` and compiles a Markdown report with per-circuit rows and group averages for the seen and unseen circuit sets.

---

## 2. Environment Setup & Prerequisites

### Python Environment

Create and activate a virtual environment, then install the necessary dependencies:

```bash
cd /mnt/c/Users/HP/Downloads/archive
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib
```

### Required Binaries & Data

* **ABC Binary**:
Compiled executable instrumented with supergate-usage logging hooks located at:
`/mnt/c/Users/HP/Downloads/archive/abc/abc`
* **Standard Cell Library**:
ASAP7 technology library file located at:
`/mnt/c/Users/HP/Downloads/archive/data/library/7nm.lib`
* **Benchmark Circuits**:
Benchmark AIG files placed inside:
`/mnt/c/Users/HP/Downloads/archive/data/benchmarks/*.aig`

---

## 3. Execution Guide & Expected Outputs

Run the scripts in sequential order from the project root or the `analysis/` folder. Everything runs inside WSL.

### Step 1: Data Collection & Profiling

```bash
python3 analysis/src/collect_usage_data.py -v
```

* **Flags**:
* `-v`: Verbose logging.
* `-a` / `--all`: Force full re-execution (bypasses skipping of already mapped circuits).

* **Expected Outputs**:
* `analysis/results/usage_stats/supergate_generation.json`: Metadata for generated supergates.
* `analysis/results/usage_stats/usage_<circuit>.json`: Per-circuit supergate usage counts.
* `data/library/library.genlib` and `library.super`: Full generated supergate library files.

### Step 2: Usage Analysis & Figures

```bash
python3 analysis/src/analyze_usage.py
python3 analysis/src/plot_usage_figures.py
```

* **Expected Outputs**:
* `analysis/results/usage_stats/circuit_supergate_frequency_table.csv`: Frequency matrix (circuits vs supergates).
* `analysis/results/usage_stats/supergate_usage_analysis.csv`: Canonical-level usage ranking.
* `analysis/results/usage_stats/supergate_usage_metadata.csv`: Per-gate area/delay/usage.
* `analysis/results/usage_stats/cumulative_usage.csv`, `top_50_usage.csv`, `pie_shares.json`: Figure inputs.
* `analysis/results/usage_stats/statistics.json`: All computed usage statistics.
* `analysis/plots/usage_heatmap_*`, `cumulative_usage_curve.png`, `top50_usage_bars.png`, `usage_pie.png`, `area_delay_usage_heatmap.png`, `area_vs_delay.png`: Figures.

### Step 3: ASAP7 Standard Baseline Evaluation

```bash
python3 analysis/src/qor_baseline.py
```

* **Expected Outputs**:
* `analysis/results/asap7_baseline_qor.csv`: Baseline metrics (Area, Delay, ADP, levels, runtime) mapped using standard `7nm.lib`.

### Step 4: Library Reduction & QoR Trade-Off Sweeps

Run evaluations for the target supergate library sizes (e.g., K=500 and K=1000):

```bash
python3 analysis/src/evaluate_reduced_library.py --k-values 500 1000
```

* **Expected Outputs**:
* `analysis/results/reduced_library_eval/full_library.super`, `library_top<K>.super`: Full (genlib-path adjusted) and reduced supergate libraries.
* `analysis/results/reduced_library_eval/library_stats.json`: Selected-group and line counts per K.
* `analysis/results/reduced_library_eval/logs/`: Per-circuit, per-library ABC logs.
* `analysis/results/reduced_library_eval/mapping_results.csv`: Per-circuit table with runtime speedups and ADP changes.

### Step 5: Summary Report Generation

```bash
python3 analysis/src/generate_report.py
```

* **Expected Outputs**:
* `analysis/results/reduced_library_eval/mapping_report.md`: Markdown report with per-circuit rows and group averages (seen and unseen circuit sets).
