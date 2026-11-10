# Machine Learning Technology Mapping Subsystem (`ml/`)

This directory contains the machine learning subsystem for circuit-adaptive supergate library selection. It provides tools to extract supergate Boolean features, construct labeled training datasets from empirical profiling logs, train Graph Neural Network (GNN) compatibility models, prune `.super` libraries dynamically for target circuits, and benchmark technology mapping performance in ABC.

---

## 1. Directory Structure

```text
ml/
├── checkpoints/       # Saved PyTorch model checkpoints (.pth)
├── data/              # Feature CSVs, truth table definitions, and training datasets
├── plots/             # Visualizations and timing breakdown figures
├── results/           # Evaluation metrics, benchmark CSVs, and temporary libraries
└── src/
    ├── models.py                   # PyTorch/PyG neural architecture definitions & loss
    ├── dataset.py                  # Binary AIGER parser & PyG DataLoader collator
    ├── supergate_lib.py            # DAG-safe custom .super library emitter
    ├── abc_driver.py               # ABC subprocess wrapper for mapping execution & timing
    ├── inference.py                # Two-stage supergate selector engine
    ├── train.py                    # Model training and fine-tuning CLI
    ├── plot_training_history.py    # Training-dynamics figure from the history CSV
    ├── evaluate.py                 # Multi-circuit evaluation and benchmark reporter
    ├── sweep_qor_vs_k.py           # Library-size sweep and QoR-loss plot
    ├── extract_truth_tables.py     # Parser evaluating DAG logic into truth table CSVs
    ├── build_dataset.py            # Dataset builder combining usage logs with truth tables
    └── plot_timing_overhead.py     # Benchmark script plotting library size vs runtime

```

---

## 2. Environment Setup

### 2.1 Virtual Environment Activation

Activate the project virtual environment from the repository root:

```bash
cd /mnt/c/Users/HP/Downloads/archive
source .venv/bin/activate

```

If the virtual environment has not been created yet:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

```

The feature-extraction and dataset-building steps (`extract_truth_tables.py`, `build_dataset.py`) only need the system Python (numpy/pandas). Training, inference, and evaluation require the virtual environment (torch, torch_geometric).

### 2.2 Required Input Files

Verify the following assets exist before executing the pipeline:

* Standard cell library: `data/library/library.genlib`
* Master supergate library: `data/library/library.super`
* Supergate generation dump: `analysis/results/usage_stats/supergate_generation.json`
* Benchmark AIG circuits: `data/benchmarks/*.aig`
* Empirical profiling logs: `analysis/results/usage_stats/usage_*.json`
* Global frequency matrix: `analysis/results/usage_stats/circuit_supergate_frequency_table.csv`

---

## 3. Step-by-Step Execution Guide

### Step 1: Extract Truth Tables from Master Supergate Library

Parse `data/library/library.super`, evaluate the shared DAG logic for all gates up to 5 inputs, and output the master truth-table definition file. Area, delay, and input counts are taken from the ABC generation dump; truth tables are computed bottom-up from the genlib gate expressions.

```bash
python3 ml/src/extract_truth_tables.py \
    --super data/library/library.super \
    --genlib data/library/library.genlib \
    --generation-json analysis/results/usage_stats/supergate_generation.json \
    --output ml/data/all_supergates_truth_tables.csv

```

* **Inputs**: `data/library/library.super`, `data/library/library.genlib`, `analysis/results/usage_stats/supergate_generation.json`
* **Output**: `ml/data/all_supergates_truth_tables.csv` (`supergate_id`, `truth_table_hex`, `num_inputs`, `area`, `delay`)

---

### Step 2: Build the Labeled Training Dataset

Cross-reference circuit-level usage logs (`usage_*.json`) from empirical profiling with the master truth table list to generate binary labels (`Label=1` for gates used in mapping, `0` otherwise):

```bash
python3 ml/src/build_dataset.py \
    --truth-table ml/data/all_supergates_truth_tables.csv \
    --usage-dir analysis/results/usage_stats \
    --output ml/data/supergate_usage_dataset.csv

```

* **Inputs**: `ml/data/all_supergates_truth_tables.csv`, `analysis/results/usage_stats/usage_*.json`
* **Output**: `ml/data/supergate_usage_dataset.csv`

---

### Step 3: Train or Fine-Tune the Compatibility Model

Train the dual-branch GNN model using Focal Loss to predict circuit-supergate compatibility scores in $[0, 1]$. The focal-loss class weight is derived automatically from the training-label distribution.

#### Standard Training (10 Epochs):

```bash
.venv/bin/python ml/src/train.py \
    --data-csv ml/data/supergate_usage_dataset.csv \
    --epochs 10 \
    --batch-size 256 \
    --lr 0.0001 \
    --checkpoint-out ml/checkpoints/best_model.pth

```

#### Fine-Tuning an Existing Checkpoint:

```bash
.venv/bin/python ml/src/train.py \
    --data-csv ml/data/supergate_usage_dataset.csv \
    --finetune ml/checkpoints/best_model.pth \
    --epochs 5 \
    --batch-size 256 \
    --checkpoint-out ml/checkpoints/finetuned_model.pth

```

#### Quick Smoke Run:

```bash
.venv/bin/python ml/src/train.py \
    --data-csv ml/data/supergate_usage_dataset.csv \
    --epochs 1 \
    --num-samples 2000 \
    --num-workers 0 \
    --checkpoint-out ml/checkpoints/best_model.pth

```

* **Inputs**: `ml/data/supergate_usage_dataset.csv` (and optional `--finetune` checkpoint)
* **Outputs**: Checkpoint file in `ml/checkpoints/` containing model state dict and validation metrics, and the per-epoch history CSV `ml/results/training_history.csv` (epoch, train/val loss, F1, precision, recall, threshold).
* **Options**: `--num-samples` caps rows per split for quick runs; `--num-workers` controls DataLoader parallelism; `--history-out` changes the history CSV path; `--resume CHECKPOINT` continues training from a saved checkpoint instead of starting over (the checkpoint is written whenever the validation loss improves and carries weights, optimizer, scheduler, and the completed epoch count).

#### Train on a Remote GPU (Google Colab):

The full 10-epoch run takes hours on a CPU. A free Google Colab GPU makes it feasible in one sitting. The step-by-step workflow — bundling `ml/src` with the dataset and circuit files, uploading via rclone, training with automatic Drive mirroring, resuming after session disconnects, and pulling the checkpoint back — is documented in [train-collab.md](train-collab.md). The resulting `best_model.pth` and `training_history.csv` are identical to the local outputs used in the steps below.

#### Plot the Training Dynamics:

After training, generate the training-dynamics figure (loss curves, validation F1, and classification metrics per epoch):

```bash
python3 ml/src/plot_training_history.py \
    --history ml/results/training_history.csv \
    --plot-out ml/plots/training_dynamics.png

```

* **Inputs**: `ml/results/training_history.csv`
* **Output**: `ml/plots/training_dynamics.png`

---

### Step 4: Evaluate Technology Mapping Performance

Run comparative technology mapping benchmarks across circuit test suites to measure mapping speedup and Area-Delay Product (ADP) relative to the full library baseline. The model-scored library stacks the measured GNN inference latency onto the mapping time.

#### Evaluate Test Circuits (Unseen Circuit Set):

```bash
# Evaluate with reduced library size K = 500 (100 core gates + 400 ML gates)
.venv/bin/python ml/src/evaluate.py --unseen --k 500 \
    --checkpoint ml/checkpoints/best_model.pth

# Evaluate with reduced library size K = 1000 (100 core gates + 900 ML gates)
.venv/bin/python ml/src/evaluate.py --unseen --k 1000 \
    --checkpoint ml/checkpoints/best_model.pth

```

#### Evaluate Training Circuits (Seen Circuit Set):

```bash
.venv/bin/python ml/src/evaluate.py --k 500 \
    --checkpoint ml/checkpoints/best_model.pth

.venv/bin/python ml/src/evaluate.py --k 1000 \
    --checkpoint ml/checkpoints/best_model.pth

```

* **Outputs**: Terminal summary table and CSV reports saved to `ml/results/{seen|unseen}_k{500|1000}_results.csv`.

#### Sweep Library Size K vs. QoR Loss:

For each circuit, map with the full library as the QoR baseline, then map with reduced libraries of increasing size K (top-100 global core gates plus model-scored candidates) and record the relative QoR loss:

```bash
.venv/bin/python ml/src/sweep_qor_vs_k.py \
    --checkpoint ml/checkpoints/best_model.pth \
    --k-values 100,200,300,500 \
    --results-out ml/results/qor_vs_k_results.csv \
    --plot-out ml/plots/qor_vs_k.png

```

* **Inputs**: Trained checkpoint, genlib, full `.super` library, and benchmark circuits (seen set by default; `--unseen` switches to the unseen set).
* **Outputs**: `ml/results/qor_vs_k_results.csv` (circuit, K, QoR loss %) and the sweep plot `ml/plots/qor_vs_k.png` (per-circuit curves plus the average).
* **Options**: `--k-values` is a comma-separated sweep; K values below the 100-gate core are not supported.

---

### Step 5: Benchmark Library Size vs. Runtime

Measure ABC technology mapping runtimes across three library scales (the frequency-ranked top-500 and top-1000 libraries from the analysis evaluation, plus the full library), stack the measured GNN inference latency, and generate comparative timing plots:

```bash
python3 ml/src/plot_timing_overhead.py \
    --abc abc/abc \
    --genlib data/library/library.genlib \
    --super-dir data/library \
    --reduced-dir analysis/results/reduced_library_eval \
    --circuits-dir data/benchmarks \
    --plot-out ml/plots/timing_breakdown.png \
    --max-circuits 50

```

* **Inputs**: Circuit `.aig` files, genlib, and `.super` libraries of varying sizes (`library_top500.super`, `library_top1000.super`, `library.super`).
* **Output**: `ml/plots/timing_breakdown.png`.
* **Note**: library sizes in the plot labels are read from the `.super` headers, not hardcoded.
* **Note**: the GNN inference latency stacked onto the reduced-library runtimes is taken from the mean `inf_time` column of `ml/results/seen_k500_results.csv` (Step 4 output) by default; override it with `--inf-time SECONDS`.

---

## 4. Module Reference

| File | Primary Functions / Classes | Role in Pipeline |
| --- | --- | --- |
| `extract_truth_tables.py` | `parse_genlib()`, `read_super_file()`, `evaluate_entries()`, `load_generation_metadata()` | Step 1 offline tool: Evaluates the shared DAG logic in `library.super` to extract 32-bit truth tables and joins area/delay metrics from the generation dump. |
| `build_dataset.py` | `build_dataset()`, `normalize_tt()` | Step 2 offline tool: Combines per-circuit profiling JSON logs with master truth tables to generate the labeled training CSV. |
| `models.py` | `GNNCompatibilityModel`, `AIGGraphEncoder`, `AIGNodeEncoder`, `FocalLoss` | Model definitions: Dual-branch GCN + MLP fusion network predicting compatibility scores in $[0, 1]$. |
| `dataset.py` | `SupergateDataset`, `parse_aiger()`, `collate_batch` | Data loader: Parses binary AIGER circuits into PyG graph objects with a pure-Python decoder and collates batches for training. |
| `train.py` | `evaluate_model()` | Step 3 CLI: Training and fine-tuning engine using AdamW, Cosine Annealing, class-balanced Focal Loss, and multi-threshold F1 evaluation; writes the per-epoch history CSV and saves a resumable full-state checkpoint whenever the validation loss improves. |
| `plot_training_history.py` | `main()` | Step 3 plotting: Draws the training-dynamics panels (train/val loss, validation F1, precision/recall/F1) from the history CSV. |
| `supergate_lib.py` | `SupergateLibraryWriter`, `patch_genlib_line()` | Library generator: Slices `.super` files preserving all ancestor intermediate nodes and valid DAG header indices; patches the embedded genlib reference of copied libraries. |
| `abc_driver.py` | `ABCDriver`, `QoRMetrics` | ABC interface: Runs `map` subprocesses, tracks the mapping command's elapsed time, and parses mapped circuit area and delay. |
| `inference.py` | `SupergateSelector` | Selection engine: Selects the Top-100 global frequency core gates and uses the GNN to pick the remaining non-core gates, with top-up filling. |
| `evaluate.py` | `evaluate_circuits()`, `main()` | Step 4 CLI: Runs comparative technology mapping benchmarks (Full vs Frequency vs ML-guided) and outputs metrics. |
| `sweep_qor_vs_k.py` | `plot_qor_vs_k()`, `main()` | Step 4 sweep: Maps a circuit set with reduced libraries of increasing size K and plots the QoR loss relative to the full library. |
| `plot_timing_overhead.py` | `run_map_timing()`, `count_supergates()`, `patch_super_genlib()` | Step 5 CLI: Benchmarks mapping runtime across small, medium, and full supergate libraries and plots timing breakdowns. |
