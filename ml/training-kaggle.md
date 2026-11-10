# Training the Model on Kaggle (GPU)

This guide walks you through training the GNN supergate compatibility model on a **free Kaggle GPU** (30 hours/week) by streaming your training bundle from Google Drive and bringing the resulting model weights and metrics back to your local repository.

---

## 1. Training Plan

The model is trained on pairs of *(circuit, supergate)* examples from a dataset CSV.
Training prints metrics per completed epoch (train loss, val loss, F1, precision, recall).
Two artifacts are generated:

| File | Description |
| --- | --- |
| `best_model.pth` | The trained model weights + optimizer/scheduler state needed to resume training. Saved **only when validation loss improves**. |
| `training_history.csv` | One row per completed epoch containing recorded metrics. Used to generate the training-dynamics plot. |

---

## 2. Files Involved

These files are transferred to Kaggle inside `train_bundle.zip`.

| Path in repo | Description |
| --- | --- |
| `ml/src/train.py` | The main training entry point |
| `ml/src/models.py`, `ml/src/dataset.py` | Core model architecture and PyTorch Geometric data loaders |
| `ml/src/*.py` (everything else) | Evaluation and plotting helpers |
| `ml/data/supergate_usage_dataset.csv` | The training dataset (~570k example rows) |
| `data/benchmarks/*.aig` | The circuit graph files referenced by the dataset (117 files) |

The dataset CSV contains relative file paths such as `data/benchmarks/adder.aig`. On Kaggle, the bundle is unzipped directly into `/kaggle/working` to replicate this exact repository structure.

---

## Part A — Local Preparation & Cloud Storage

### A1. Build the Training Bundle (zip)

From your repository root in Git Bash (Windows):

```bash
powershell.exe -NoProfile -Command "Compress-Archive -Path 'ml\src','ml\data\supergate_usage_dataset.csv','data\benchmarks' -DestinationPath 'train_bundle.zip' -Force"
```

This packages `ml/` and `data/benchmarks/` into `train_bundle.zip` while preserving relative directory paths.

### A2. Upload Bundle to Google Drive & Retrieve File ID

1. Upload `train_bundle.zip` to your Google Drive.
2. Right-click `train_bundle.zip` $\rightarrow$ **Share** $\rightarrow$ **Share**.
3. Under **General access**, set permissions to **Anyone with the link**.
4. Click **Copy link**. The URL format will be:
   ```text
   https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0jklmnopqrst/view?usp=sharing
   ```
5. Extract the alphanumeric string between `/d/` and `/view` (for example: `1a2b3c4d5e6f7g8h9i0jklmnopqrst`). This is your `FILE_ID`.

---

## Part B — Run Training on Kaggle

### B1. Create Notebook & Configure Hardware

1. Go to [kaggle.com](https://www.kaggle.com) and click **Create** $\rightarrow$ **New Notebook**.
2. Open the **Notebook options** panel on the right sidebar:
   * **Accelerator:** Select **GPU T4 x1** (or **GPU P100**).
   * **Internet:** Toggle to **On** *(required for pip packages and file transfers)*.
3. Confirm that the status icon in the top navigation bar shows an active session with connected GPU resources.

### B2. Cell 1 — Install Dependencies

```bash
!pip install -q gdown torch-geometric
```

### B3. Cell 2 — Download and Extract the Training Bundle

Replace `YOUR_DRIVE_FILE_ID` with the file ID from Step A2:

```bash
# Download the archive directly into Kaggle's working directory
!gdown --id YOUR_DRIVE_FILE_ID -O /kaggle/working/train_bundle.zip

# Extract bundle contents
!unzip -q -o /kaggle/working/train_bundle.zip -d /kaggle/working

# Clean up archive to reclaim disk space
!rm -f /kaggle/working/train_bundle.zip
```

Once extracted, the Kaggle filesystem mirrors the local project root:

```text
/kaggle/working/ml/src/...        (code)
/kaggle/working/ml/data/...       (dataset CSV)
/kaggle/working/data/benchmarks/  (117 circuit AIG files)
```

### B4. Cell 3 — Environment Verification

```bash
!nvidia-smi | head -5
!python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
!ls /kaggle/working/data/benchmarks | wc -l
```

**Expected verification output:**
* An NVIDIA GPU (T4 or P100) listed in the process table.
* `CUDA available: True`
* Circuit file count prints `117`.

### B5. Cell 4 — Execute Training

Kaggle instances allocate 2 vCPUs; explicitly passing `--num-workers 2` optimizes DataLoader throughput and prevents thread deadlocks:

```bash
%cd /kaggle/working

!python ml/src/train.py \
    --data-csv ml/data/supergate_usage_dataset.csv \
    --epochs 10 \
    --batch-size 512 \
    --num-workers 2 \
    --checkpoint-out /kaggle/working/ml/checkpoints/best_model.pth \
    --history-out /kaggle/working/ml/results/training_history.csv
```

Outputs written under `/kaggle/working/` persist in the session output and are accessible directly via Kaggle's file browser.

Sample training progress:

```text
Epoch 1/10: 100%|██████████| 1114/1114 [14:22<00:00, loss: 0.0841]
Epoch 1 -> Train Loss: 0.1245 | Val Loss: 0.1089 | F1: 0.6712 | Prec: 0.6901 | Rec: 0.6533 (Thresh: 0.3)
  Saved checkpoint to /kaggle/working/ml/checkpoints/best_model.pth (val loss improved to 0.1089)
```

### B6. Resuming Training After Interruption

If an execution session hits its time limit or you want to train for additional epochs:

1. Re-run Cells **B1**, **B2**, and **B3**.
2. Verify that your previous `best_model.pth` and `training_history.csv` are present inside `/kaggle/working/ml/checkpoints/` and `/kaggle/working/ml/results/`.
3. Add the `--resume` flag to your command:

```bash
%cd /kaggle/working

!python ml/src/train.py \
    --data-csv ml/data/supergate_usage_dataset.csv \
    --epochs 10 \
    --batch-size 512 \
    --num-workers 2 \
    --checkpoint-out /kaggle/working/ml/checkpoints/best_model.pth \
    --history-out /kaggle/working/ml/results/training_history.csv \
    --resume /kaggle/working/ml/checkpoints/best_model.pth
```

The `--resume` flag restores the model weights, optimizer, and learning rate scheduler while continuing the epoch count and appending to `training_history.csv`.

---

## Part C — Download Artifacts and Verify Locally

### C1. Retrieve Artifacts from Kaggle

#### Method 1: Web Interface Download
1. In the right-hand notebook sidebar, locate the **Output** section (`/kaggle/working`).
2. Expand `ml/` $\rightarrow$ `checkpoints/` $\rightarrow$ click the three dots next to `best_model.pth` $\rightarrow$ **Download**.
3. Expand `ml/` $\rightarrow$ `results/` $\rightarrow$ click the three dots next to `training_history.csv` $\rightarrow$ **Download**.
4. Move `best_model.pth` into your local `ml/checkpoints/` directory and `training_history.csv` into `ml/results/`.

#### Method 2: Download Archive
To package all outputs into a single downloadable archive:

```bash
!tar -czvf /kaggle/working/trained_artifacts.tar.gz \
    -C /kaggle/working \
    ml/checkpoints/best_model.pth \
    ml/results/training_history.csv
```

Download `trained_artifacts.tar.gz` directly from the **Output** panel.

### C2. Terminate the Session

To preserve your **30 hours per week** GPU allocation:
1. Click the power button / session options icon in the top header bar.
2. Select **Stop Session** to halt GPU usage immediately.

### C3. Local Verification & Plotting (WSL)

Verify the retrieved model checkpoint inside your local WSL environment:

```bash
wsl.exe -e bash -lc 'cd /mnt/c/Users/HP/Downloads/archive && .venv/bin/python -c "import torch; ck = torch.load(\"ml/checkpoints/best_model.pth\", map_location=\"cpu\", weights_only=False); print(\"epoch:\", ck[\"epoch\"], \"best_val_loss:\", round(ck[\"best_val_loss\"], 4), \"f1:\", round(ck[\"f1\"], 4))"'
```

Generate the training-dynamics plot from the history CSV:

```bash
wsl.exe -e bash -lc 'cd /mnt/c/Users/HP/Downloads/archive && python3 ml/src/plot_training_history.py'
```

This writes the metrics visualization to `ml/plots/training_dynamics.png`.

---

## Troubleshooting

| Problem | Cause | Resolution |
| --- | --- | --- |
| `gdown: Cannot retrieve URL` | Drive link permissions restricted | In Google Drive, verify general access is set to **Anyone with the link**, and verify that only the file ID is passed to `--id`. |
| `CUDA out of memory` | Batch size exceeds available GPU memory | Restart kernel and re-run with `--batch-size 256` or `128`. |
| `Output files missing in sidebar` | UI file listing stale | Click the refresh icon next to the **Output** heading in the right sidebar. |
| `DataLoader worker exited unexpectedly` | CPU thread exhaustion | Explicitly specify `--num-workers 2` or `--num-workers 0` in the training command. |
| Weekly quota exhausted | 30-hour weekly GPU limit reached | Wait for the weekly counter to reset, switch notebook accelerator to **GPU P100**, or use an alternate Kaggle account. |
| Missing circuit AIG warning | Bundle extraction error | Verify `!ls /kaggle/working/data/benchmarks | wc -l` prints `117` before running `train.py`. |