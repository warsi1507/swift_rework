#!/usr/bin/env python3
"""Supergate usage figure generation.

Reads the CSVs/JSONs produced by analyze_usage.py and writes PNGs to
analysis/plots/.  All annotations (50/80/90% thresholds, pie shares,
top-K cutoffs) are computed from the data - nothing is hardcoded.

Figures produced:
  circuit-vs-supergate usage heatmap, log(usage + 1), two variants
  cumulative usage curve, linear x-axis, 50/80/90% coverage annotations
  top-50 usage bar chart with 1e6 axis offset
  usage pie chart: top-10 / top-11-30 / top-31-50 / others
  normalized area-delay usage heatmap, log(usage + 1), vmax = data maximum
  area vs delay scatter colored by usage (supplementary)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
STATS_DIR = BASE_DIR / "results" / "usage_stats"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150


def save(fig, name):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def add_caption(fig, text):
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.subplots_adjust(bottom=0.14)


def usage_heatmap():
    df = pd.read_csv(STATS_DIR / "circuit_supergate_frequency_table.csv", index_col=0).astype(float)
    values = df.values
    # columns are already sorted by total usage (descending)

    def plot(cols, title_line, name):
        data = values[:, :cols]
        data_log = np.where(data > 0, np.log(data + 1.0), 0.0)
        fig, ax = plt.subplots(figsize=(24, 12), dpi=DPI)
        im = ax.imshow(data_log, cmap="YlOrRd", aspect="auto", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Supergates (sorted by usage)", fontsize=18, fontweight="bold")
        ax.set_ylabel("Circuits", fontsize=18, fontweight="bold")
        ax.set_title(f"Circuit vs Supergate Usage Frequency Heatmap\n{title_line}",
                     fontsize=18, fontweight="bold", pad=15)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("log(Usage + 1)", fontsize=18, fontweight="bold")
        add_caption(fig, "Usage heatmap of circuits (rows) versus supergates "
                         "(columns, sorted by total usage).")
        save(fig, name)

    n_cols = values.shape[1]
    plot(min(n_cols, 1000), "(Top 1000 Supergates, Log Scale)", "usage_heatmap_top1000.png")
    plot(n_cols, "(All Used Supergates, Log Scale)", "usage_heatmap_full.png")


def cumulative_usage_curve():
    df = pd.read_csv(STATS_DIR / "cumulative_usage.csv")
    stats = json.loads((STATS_DIR / "statistics.json").read_text())
    k = stats["canonical"]
    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)
    ax.plot(df["rank"], df["cumulative_share"] * 100, linewidth=2.5, color="steelblue")

    annotations = [
        (50, k["k_for_50pct"], "#d62728", "o", "50% of usage"),
        (80, k["k_for_80pct"], "#ff7f0e", "s", "80% of usage"),
        (90, k["k_for_90pct"], "#2ca02c", "^", "90% of usage"),
    ]
    for target, kk, color, marker, label in annotations:
        if kk is None:
            continue
        row = df.iloc[kk - 1]
        ax.axhline(target, linestyle="--", color=color, linewidth=1.5)
        ax.axvline(kk, linestyle="--", color=color, linewidth=1.5)
        ax.plot([kk], [row["cumulative_share"] * 100], marker=marker, color=color, markersize=9,
                label=label)
        ax.annotate(f"{kk} gates ({target}%)", xy=(kk, target), xytext=(kk * 1.1, target - 8),
                    fontsize=13, fontweight="bold", color=color)

    ax.set_xlabel("Number of Supergates (sorted by usage)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Cumulative Usage (%)", fontsize=16, fontweight="bold")
    xmax = int(np.ceil(df["rank"].max() / 500.0) * 500)
    ax.set_xticks(range(0, xmax + 1, 500))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=12)
    add_caption(fig, "Cumulative usage curve showing the number of supergates "
                     "required to cover gate utilization in the design set.")
    save(fig, "cumulative_usage_curve.png")


def top50_usage_bars():
    df = pd.read_csv(STATS_DIR / "top_50_usage.csv")
    fig, ax = plt.subplots(figsize=(12, 5), dpi=DPI)
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(df)))
    ax.bar(df["rank"], df["usage_count"], color=colors)
    ax.set_xlabel("Supergate Rank", fontsize=16, fontweight="bold")
    ax.set_ylabel("Usage Count", fontsize=16, fontweight="bold")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.set_xticks(range(1, 51, 5))
    ax.grid(axis="y", alpha=0.3)
    add_caption(fig, "Top 50 most used supergates and their usage counts")
    save(fig, "top50_usage_bars.png")


def usage_pie():
    pie = json.loads((STATS_DIR / "pie_shares.json").read_text())
    others_n = pie["n_total_groups"] - 50
    # layout: clockwise from top-right = others (gray), top 31-50 (blue),
    # top 11-30 (teal), top 10 (red)
    labels = [f"Others ({others_n}) ({pie['others'] * 100:.1f}%)",
              f"Top 31-50 ({pie['top_31_50'] * 100:.1f}%)",
              f"Top 11-30 ({pie['top_11_30'] * 100:.1f}%)",
              f"Top 10 ({pie['top_10'] * 100:.1f}%)"]
    sizes = [pie["others"], pie["top_31_50"], pie["top_11_30"], pie["top_10"]]
    colors = ["#7f7f7f", "#1f77b4", "#17becf", "#d62728"]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=DPI)
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
           startangle=90, counterclock=False,
           textprops={"fontsize": 13, "fontweight": "bold"})
    add_caption(fig, f"The distribution of top {pie['n_total_groups']} supergates "
                     f"in terms of usage")
    save(fig, "usage_pie.png")


def area_delay_usage_heatmap():
    df = pd.read_csv(STATS_DIR / "supergate_usage_metadata.csv")
    area = df["area"].to_numpy(dtype=float)
    delay = df["delay"].to_numpy(dtype=float)
    usage = df["usage_count"].to_numpy(dtype=float)

    def norm(v):
        lo, hi = float(np.min(v)), float(np.max(v))
        rng = hi - lo
        return (v - lo) / rng if rng > 0 else np.zeros_like(v)

    delay_n, area_n = norm(delay), norm(area)

    bins = 90
    heat, _, _ = np.histogram2d(delay_n, area_n, bins=bins, weights=usage)
    heat = heat.T  # rows = area, cols = delay

    transformed = np.log(heat + 1.0)
    vmax = float(np.ceil(transformed.max()))

    fig, ax = plt.subplots(figsize=(14, 10), dpi=DPI)
    im = ax.imshow(transformed, origin="lower", aspect="auto", cmap="YlOrRd",
                   extent=[0.0, 1.0, 0.0, 1.0], interpolation="nearest",
                   vmin=0.0, vmax=vmax)
    ax.set_xlabel("Normalized Delay (0-1)", fontsize=24, fontweight="bold")
    ax.set_ylabel("Normalized Area (0-1)", fontsize=24, fontweight="bold")
    ax.tick_params(axis="both", labelsize=20)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log(Total Usage + 1)", fontsize=24, fontweight="bold")
    cbar.ax.tick_params(labelsize=20)
    add_caption(fig, "Normalized Area-Normalized delay distribution of supergate "
                     "usage: a weighted heatmap where each bin accumulates total supergate "
                     "lookup counts for gates within the corresponding delay and area ranges "
                     "(log-scaled color).")
    save(fig, "area_delay_usage_heatmap.png")


def extra_area_vs_delay_scatter():
    df = pd.read_csv(STATS_DIR / "supergate_usage_metadata.csv")
    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)
    sc = ax.scatter(df["delay"], df["area"], c=np.log1p(df["usage_count"]),
                    cmap="YlOrRd", s=14, alpha=0.7)
    ax.set_xlabel("Delay", fontsize=16, fontweight="bold")
    ax.set_ylabel("Area", fontsize=16, fontweight="bold")
    ax.set_title("Area vs Delay of Used Supergates (color = log usage)",
                 fontsize=16, fontweight="bold", pad=12)
    fig.colorbar(sc, ax=ax, label="log(usage + 1)")
    save(fig, "area_vs_delay.png")


def main():
    print("Generating usage figures...")
    usage_heatmap()
    cumulative_usage_curve()
    top50_usage_bars()
    usage_pie()
    area_delay_usage_heatmap()
    extra_area_vs_delay_scatter()
    print("done")


if __name__ == "__main__":
    main()
