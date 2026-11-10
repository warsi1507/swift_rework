"""
ml/src/train.py
Trains the GNN supergate compatibility model with focal loss and a
class-balanced alpha. The checkpoint (weights, optimizer, scheduler,
and epoch) is saved whenever the validation loss improves, and a
per-epoch training-history CSV is maintained for the dynamics plot.
--resume continues training from such a checkpoint.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import GNNCompatibilityModel, FocalLoss
from dataset import SupergateDataset, collate_batch


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for graphs, sg_data, targets in loader:
            graphs = graphs.to(device)
            sg_data = {k: v.to(device) for k, v in sg_data.items()}
            targets = targets.to(device)
            
            preds = model(graphs, sg_data)
            loss = criterion(preds, targets)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    best_f1, best_th = 0.0, 0.5
    for th in [0.1, 0.2, 0.3, 0.4, 0.5]:
        bin_preds = (all_preds > th).astype(int)
        tp = np.sum((bin_preds == 1) & (all_targets == 1))
        pp = np.sum(bin_preds == 1)
        ap = np.sum(all_targets == 1)
        prec = tp / pp if pp > 0 else 0.0
        rec = tp / ap if ap > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            
    bin_preds = (all_preds > best_th).astype(int)
    tp = np.sum((bin_preds == 1) & (all_targets == 1))
    prec = tp / np.sum(bin_preds == 1) if np.sum(bin_preds == 1) > 0 else 0.0
    rec = tp / np.sum(all_targets == 1) if np.sum(all_targets == 1) > 0 else 0.0
    
    return total_loss / len(loader), best_f1, prec, rec, best_th


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-csv', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint-out', type=str, default='ml/checkpoints/best_model.pth')
    parser.add_argument('--history-out', type=str, default='ml/results/training_history.csv')
    parser.add_argument('--finetune', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help="resume from a full-state checkpoint (weights, optimizer, epoch)")
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_data = SupergateDataset(args.data_csv, split='train', num_samples=args.num_samples)
    val_data = SupergateDataset(args.data_csv, split='val', num_samples=args.num_samples)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=collate_batch, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=collate_batch, num_workers=args.num_workers)

    model = GNNCompatibilityModel().to(device)
    if args.finetune and os.path.exists(args.finetune):
        print(f"Loading weights from: {args.finetune}")
        ckpt = torch.load(args.finetune, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))

    train_labels = np.array([float(r['Label']) for r in train_data.data_rows])
    n_pos = float(train_labels.sum())
    alpha = (len(train_labels) - n_pos) / len(train_labels) if len(train_labels) else 0.25
    print(f"Training labels: {len(train_labels)} total, {int(n_pos)} positive -> focal alpha={alpha:.4f}")
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    start_epoch = 0
    best_val_loss = float('inf')
    history = []
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt.get('optimizer_state_dict', optimizer.state_dict()))
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = int(ckpt.get('epoch', 0))
        best_val_loss = float(ckpt.get('best_val_loss', float('inf')))
        if os.path.exists(args.history_out):
            history = pd.read_csv(args.history_out).to_dict('records')
            history = [r for r in history if int(r['epoch']) <= start_epoch]
        print(f"Resumed from {args.resume}: completed epochs={start_epoch}, "
              f"best val loss={best_val_loss:.4f}, history rows={len(history)}")
    elif args.resume:
        print(f"Warning: resume checkpoint not found at {args.resume}; starting fresh")

    total_epochs = start_epoch + args.epochs
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        for graphs, sg_data, targets in pbar:
            graphs = graphs.to(device)
            sg_data = {k: v.to(device) for k, v in sg_data.items()}
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(graphs, sg_data)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        train_loss = total_train_loss / len(train_loader)
        val_loss, f1, prec, rec, th = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} (Thresh: {th})")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'f1': f1,
            'precision': prec,
            'recall': rec,
            'threshold': th,
        })
        history_dir = os.path.dirname(args.history_out)
        if history_dir:
            os.makedirs(history_dir, exist_ok=True)
        pd.DataFrame(history).to_csv(args.history_out, index=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(args.checkpoint_out), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'f1': f1,
            }, args.checkpoint_out)
            print(f"  Saved checkpoint to {args.checkpoint_out} (val loss improved to {val_loss:.4f})")


if __name__ == '__main__':
    main()