"""
ml/src/plot_training_history.py
Plots the training dynamics (loss, F1, and classification metrics per
epoch) from the history CSV written by train.py.
Generates: ml/plots/training_dynamics.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=str, default='ml/results/training_history.csv')
    parser.add_argument('--plot-out', type=str, default='ml/plots/training_dynamics.png')
    args = parser.parse_args()

    df = pd.read_csv(args.history)
    epochs = df['epoch']

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(epochs, df['train_loss'], marker='o', label='Training loss')
    ax.plot(epochs, df['val_loss'], marker='s', label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(a) Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, df['f1'], marker='o', color='tab:blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('(b) Validation F1 Score')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, df['precision'], marker='o', label='Precision')
    ax.plot(epochs, df['recall'], marker='s', label='Recall')
    ax.plot(epochs, df['f1'], marker='^', label='F1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('(c) Classification Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(args.plot_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Training dynamics plot saved to: {out}")


if __name__ == '__main__':
    main()
