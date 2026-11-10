# Training the Model on Google Colab (GPU)

This guide walks you through training the GNN supergate compatibility model on a **free Google Colab GPU** and bringing the results back to this repo.

---

## 1. Training Plan

The model is taught with pairs of *(circuit, supergate)* examples from a dataset CSV. 
Training prints one line per epoch (train loss, val loss, F1, precision, recall). 
Two files come out of it:

| File | Description |
| --- | --- |
| `best_model.pth` | the trained model weights + the state needed to resume training. Saved **only when the validation loss improves**. |
| `training_history.csv` | one row per finished epoch with all the printed metrics. Used later to draw the training-dynamics plot. |

---

## 2. The files involved
These files are send to Colab.

| Path in repo | Description |
| --- | --- |
| `ml/src/train.py` | the training script (the only script you run) |
| `ml/src/models.py`, `ml/src/dataset.py` | code that `train.py` imports (GNN model, data loading) |
| `ml/src/*.py` (everything else) | evaluation / plotting helpers, not needed for training but tiny, so we ship the whole folder |
| `ml/data/supergate_usage_dataset.csv` | the training dataset (~570k example rows) |
| `data/benchmarks/*.aig` | the circuit graph files the dataset rows point to (117 files) |

Why the circuits must travel too: the CSV stores relative paths like `data/benchmarks/adder.aig`, resolved against the folder you run training from. So on Colab we unzip the bundle so the same relative layout exists there.

---

## Part A — One-time setup on your PC

### A1. Install rclone and connect it to Google Drive

rclone is a command-line tool that copies files to/from cloud storage (Google Drive here). 
It runs fine from Git Bash on Windows.

1. Install rclone from <https://rclone.org/downloads/> 
(Windows zip: put `rclone.exe` somewhere on your PATH, e.g. `C:\tools\rclone`).
2. In Git Bash, configure a Drive remote:
   ```bash
   rclone config
   ```

   - `n` for "New remote"
   - name it `gdrive`
   - type: `drive` (the number shown next to "Google Drive")
   - client_id / client_secret: press Enter to accept the defaults
   - scope: choose `1` (full access)
   - root_folder_id / service_account_file / config edits: press Enter
     (defaults)
   - auto config: `y` — a browser tab opens, sign in to Google, allow
     access, come back and confirm `y`.

3. Test it:

   ```bash
   rclone lsd gdrive:
   ```

   This lists your Drive folders. If it works, setup is done forever.

### A2. Build the training bundle (zip)

From the repo root, in Git Bash:

```bash
powershell.exe -NoProfile -Command "Compress-Archive -Path 'ml\src','ml\data\supergate_usage_dataset.csv','data\benchmarks' -DestinationPath 'train_bundle.zip' -Force"
```

This creates `train_bundle.zip`.
The three entries keep their relative paths inside the zip, which is exactly what Colab needs.

> Rebuild this zip whenever the dataset or any file in `ml/src` changes.

### A3. Upload the bundle to Drive

```bash
rclone mkdir gdrive:ml_training
rclone copy train_bundle.zip gdrive:ml_training/bundle/ -P
```

`-P` shows progress. When done, you can delete `train_bundle.zip` locally — it lives on Drive now.

---

## Part B — Train in Colab

### B1. Create the notebook and turn on the GPU

1. Go to <https://colab.research.google.com> → **New notebook**.
   (This notebook is separate from the repo; it is not meant to be saved into it.)
2. Menu **Runtime → Change runtime type → T4 GPU → Save**.
   The free tier gives you a T4 GPU with 16 GB memory. Sessions are time limited (roughly 12 hours, idle timeouts too) — the workflow below is built to survive that.

### B2. Cell 1 — mount Drive and unpack the bundle

```python
from google.colab import drive
drive.mount('/content/drive')
```

Run it, follow the authorization popup. Then, in a second cell:

```bash
!cp /content/drive/MyDrive/ml_training/bundle/train_bundle.zip /content/
!unzip -q -o /content/train_bundle.zip -d /content
```

After this, the Colab file system mirrors the repo layout:

```
/content/ml/src/...        (the code)
/content/ml/data/...       (the dataset CSV)
/content/data/benchmarks/  (the 117 circuit files)
```

### B3. Cell 3 — install dependencies

```bash
!pip install -q torch-geometric
```

Colab already has PyTorch preinstalled; this adds the graph neural network library that the model uses.

### B4. Cell 4 — sanity checks (30 seconds, saves you an hour)

```bash
!nvidia-smi | head -5
!python -c "import torch; print('cuda available:', torch.cuda.is_available())"
!ls /content/data/benchmarks | wc -l
```

Expected: the GPU shows up, `cuda available: True`, and the last line prints `117`. 
If the count is wrong, the unzip in B2 failed — re-run it before training (training crashes on missing circuit files).

### B5. Cell 5 — train

```python
import os, shutil, threading, time

LOCAL_CKPT = '/content/checkpoints'
LOCAL_HIST = '/content/history'
DRIVE_DIR  = '/content/drive/MyDrive/ml_training'

os.makedirs(LOCAL_CKPT, exist_ok=True)
os.makedirs(LOCAL_HIST, exist_ok=True)
os.makedirs(f'{DRIVE_DIR}/checkpoints', exist_ok=True)
os.makedirs(f'{DRIVE_DIR}/history', exist_ok=True)

def mirror_to_drive():
  """Copy training outputs to Drive every 60 s (survives a session kill)."""
  while True:
    time.sleep(60)
    for src, dst in [(LOCAL_CKPT, f'{DRIVE_DIR}/checkpoints'), (LOCAL_HIST, f'{DRIVE_DIR}/history')]:
      for f in os.listdir(src):
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))

threading.Thread(target=mirror_to_drive, daemon=True).start()
```

Then the training command itself, in the next cell:

```bash
%cd /content
!python ml/src/train.py \
    --data-csv ml/data/supergate_usage_dataset.csv \
    --epochs 10 \
    --batch-size 512 \
    --num-workers 2 \
    --checkpoint-out /content/checkpoints/best_model.pth \
    --history-out /content/history/training_history.csv
```

What the flags mean:

- `--data-csv` — the dataset uploaded in the bundle.
- `--epochs 10` — full training length (one pass over the data per epoch).
- `--batch-size 512` — how many examples the GPU processes at once
  (512 fits the free T4; see troubleshooting if it doesn't).
- `--num-workers 2` — background processes that prepare data so the GPU
  never waits.
- `--checkpoint-out` / `--history-out` — where outputs go. They are written
  to Colab's local disk first (fast), and the mirror loop copies them to
  Drive every 60 seconds.

While it runs you will see a progress bar per epoch and a summary line like:

```
Epoch 5 -> Train Loss: 0.1234 | Val Loss: 0.1120 | F1: 0.6543 | Prec: ... | Rec: ... (Thresh: 0.3)
Saved checkpoint to /content/checkpoints/best_model.pth (val loss improved to 0.1120)
```

The "Saved checkpoint" line appears **only on epochs where the validation loss got better than every epoch before** — that is intentional. The history CSV records every epoch regardless.

> Keep the Colab tab open while training. Expect roughly 10–30 minutes per
> epoch on the free GPU, so 10 epochs take a few hours.

### B6. If the session dies or is interrupted — resume

Nothing is lost (except at most the last 60 s): the last saved checkpoint and the history are on Drive. 
Start a new Colab session, re-run cells B2 (unpack), B3 (deps), B4 (checks), then restore the outputs from Drive:

```bash
!mkdir -p /content/checkpoints /content/history
!cp /content/drive/MyDrive/ml_training/checkpoints/best_model.pth /content/checkpoints/
!cp /content/drive/MyDrive/ml_training/history/training_history.csv /content/history/
```

Then re-run the mirror cell (B5 part 1) and run training with
`--resume`, which continues from the checkpoint instead of starting over:

```bash
%cd /content
!python ml/src/train.py \
    --data-csv ml/data/supergate_usage_dataset.csv \
    --epochs 10 \
    --batch-size 512 \
    --num-workers 2 \
    --checkpoint-out /content/checkpoints/best_model.pth \
    --history-out /content/history/training_history.csv \
    --resume /content/checkpoints/best_model.pth
```

How `--resume` works: the checkpoint remembers how many epochs finished (e.g. 6). 
`--epochs 10` now means "do 10 *more* epochs", so training runs epochs 7–16. 
The history CSV is also loaded and extended, not overwritten.

> If a resumed run never improves the validation loss, `best_model.pth` is
> simply not updated again — that is fine, the old best one is still there.

---

## Part C — Bring the results home

### C1. Pull the outputs down with rclone (Git Bash, repo root)

```bash
rclone copy gdrive:ml_training/checkpoints ml/checkpoints/ -P
rclone copy gdrive:ml_training/history ml/results/ -P
```

After this you have:

- `ml/checkpoints/best_model.pth` — the trained model.
- `ml/results/training_history.csv` — the per-epoch metrics.

### C2. Verify the checkpoint locally (WSL)

This repo runs its Python inside WSL. Quick sanity check:

```bash
wsl.exe -e bash -lc 'cd /mnt/c/Users/HP/Downloads/archive && .venv/bin/python -c "import torch; ck = torch.load(\"ml/checkpoints/best_model.pth\", map_location=\"cpu\", weights_only=False); print(\"epoch:\", ck[\"epoch\"], \"best_val_loss:\", round(ck[\"best_val_loss\"], 4), \"f1:\", round(ck[\"f1\"], 4))"'
```

Then draw the training-dynamics plot from the history CSV:

```bash
wsl.exe -e bash -lc 'cd /mnt/c/Users/HP/Downloads/archive && python3 ml/src/plot_training_history.py'
```

This writes `ml/plots/training_dynamics.png` (loss curves, validation F1, precision/recall/F1 per epoch).

### C3. What to do next

The trained model is now usable by the rest of the pipeline:

- evaluate the model on the held-out circuit sets (`ml/src/evaluate.py`),
- sweep library size K vs. quality (`ml/src/sweep_qor_vs_k.py`),

both documented in `ml/README.md` (steps 4 and 5).

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `CUDA out of memory` during training | The T4 is full. In the menu: **Runtime → Restart session**, then re-run B2/B3/B4 and train with `--batch-size 256` instead of 512. |
| Colab says GPU not available | Free-tier GPUs are quota-limited. Wait a while, or switch to **Runtime → Change runtime type → T4 GPU** again. CPU fallback works but is many times slower. |
| `Warning: Missing circuit AIG: ...` then a crash | The bundle was not unpacked correctly. Re-run the B2 unzip cell and check that `!ls /content/data/benchmarks | wc -l` prints 117. |
| Colab session dies mid-epoch | Expected with the free tier. Follow B6 (resume) — the last saved checkpoint is on Drive. |
| I edited the code inside Colab and lost the fix | Never edit `ml/src` inside Colab. Make fixes in the local repo, rebuild the zip (A2), re-upload (A3), and re-run. |
| `rclone copy` says the folder does not exist | Drive folders are created lazily: run `rclone mkdir gdrive:ml_training` and the `bundle/checkpoints/history` subfolders first, or run the Colab cells once so the notebook creates them. |
