"""
ml/src/plot_training_history.py
Plots the training dynamics (loss, F1, and classification metrics per
epoch) from the history CSV written by train.py.
Generates: ml/plots/training_dynamics.png
"""

import argparse
import textwrap
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
    n_epochs = len(df)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    ax_a.plot(epochs, df['train_loss'], marker='o', color='tab:blue', label='Train Loss')
    ax_a.plot(epochs, df['val_loss'], marker='o', color='tab:orange', label='Validation Loss')
    ax_a.set_xlabel('Epoch')
    ax_a.set_ylabel('Loss')
    ax_a.set_title('Training and Validation Loss')
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)
    ax_a.text(-0.08, 1.05, '(a)', transform=ax_a.transAxes, fontsize=13,
              fontweight='bold', va='top')

    ax_b.plot(epochs, df['f1'], marker='o', color='tab:green', label='Validation F1')
    ax_b.set_xlabel('Epoch')
    ax_b.set_ylabel('F1 Score')
    ax_b.set_title('F1 by Epoch')
    ax_b.legend(loc='lower right')
    ax_b.grid(True, alpha=0.3)
    ax_b.text(-0.08, 1.05, '(b)', transform=ax_b.transAxes, fontsize=13,
              fontweight='bold', va='top')

    ax_c.plot(epochs, df['precision'], marker='o', color='tab:blue', label='Precision')
    ax_c.plot(epochs, df['recall'], marker='o', color='tab:orange', label='Recall')
    ax_c.plot(epochs, df['f1'], marker='o', color='tab:green', label='F1 Score')
    ax_c.set_xlabel('Epoch')
    ax_c.set_ylabel('Score')
    ax_c.set_title('Classification Metrics by Epoch')
    ax_c.legend(loc='center right')
    ax_c.grid(True, alpha=0.3)
    ax_c.text(-0.04, 1.05, '(c)', transform=ax_c.transAxes, fontsize=13,
              fontweight='bold', va='top')

    caption = (f'Figure 10: Training dynamics of the model across epochs: (a) training and '
               f'validation loss, (b) validation F1 score, and (c) classification metrics '
               f'(precision, recall, and F1). The model exhibits stable convergence and '
               f'balanced classification performance across the {n_epochs} training epochs.')
    caption_lines = textwrap.wrap(caption, 110)
    fig.text(0.5, 0.01, '\n'.join(caption_lines), ha='center', va='bottom', fontsize=10)

    fig.subplots_adjust(bottom=0.07 + 0.02 * len(caption_lines), top=0.93)
    out = Path(args.plot_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Training dynamics plot saved to: {out}")


if __name__ == '__main__':
    main()
