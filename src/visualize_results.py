"""
visualize_results.py — Generate all publication-quality result figures.

Creates:
  • figures/class_distribution.png        — patient- and hour-level imbalance
  • figures/missingness_heatmap.png       — variable missingness bar chart
  • figures/sequence_length_dist.png      — distribution of ICU stay lengths
  • figures/training_curves_*.png         — loss/AUROC over epochs
  • figures/results_comparison_bar.png    — AUROC/F1/Recall comparison
  • figures/ablation_heatmap.png          — ablation study grid

Run:
    python src/visualize_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
LOGS_DIR    = ROOT / "logs"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "LR":                    "#4A90D9",
    "XGB":                   "#E67E22",
    "LSTM":                  "#9B59B6",
    "GRU":                   "#1ABC9C",
    "TCN_Baseline":          "#E74C3C",
    "TCN_Baseline_PosWeight":"#C0392B",
    "TCN_Proposed":          "#2ECC71",
    "TCN_Proposed_PosWeight":"#27AE60",
}
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def save_fig(fig, name: str) -> None:
    p = FIGURES_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 1. Missingness bar chart
# ---------------------------------------------------------------------------

def plot_missingness() -> None:
    miss_csv = RESULTS_DIR / "missing_summary.csv"
    if not miss_csv.exists():
        print("  Missing summary CSV not found — skipping.")
        return

    ms = pd.read_csv(miss_csv, index_col="variable").sort_values(
        "missing_percent", ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, 9))
    colors = ["#E74C3C" if v > 90 else "#E67E22" if v > 50 else "#2ECC71"
              for v in ms["missing_percent"]]
    bars = ax.barh(ms.index, ms["missing_percent"], color=colors, alpha=0.85)
    ax.axvline(90, color="#C0392B", lw=1.2, ls="--", label=">90% missing")
    ax.axvline(50, color="#E67E22", lw=1.2, ls="--", label=">50% missing")
    ax.set_xlabel("Missingness (%)")
    ax.set_title("Feature Missingness Rates — PhysioNet CinC 2019")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 105)
    for bar, pct in zip(bars, ms["missing_percent"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=7.5)
    plt.tight_layout()
    save_fig(fig, "missingness_heatmap.png")


# ---------------------------------------------------------------------------
# 2. Class distribution (from manifest)
# ---------------------------------------------------------------------------

def plot_class_distribution() -> None:
    manifest_csv = ROOT / "data" / "processed" / "split_manifest.csv"
    if not manifest_csv.exists():
        print("  split_manifest.csv not found — skipping class distribution.")
        return

    m = pd.read_csv(manifest_csv)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Patient-level
    counts = m["sepsis_positive"].value_counts().sort_index()
    labels = ["Non-Sepsis", "Sepsis"]
    colors = ["#4A90D9", "#E74C3C"]
    axes[0].bar(labels, counts.values, color=colors, alpha=0.85, width=0.5)
    axes[0].set_title("Patient-Level Class Distribution")
    axes[0].set_ylabel("Number of Patients")
    for i, (lbl, cnt) in enumerate(zip(labels, counts.values)):
        axes[0].text(i, cnt + 50, f"{cnt:,}", ha="center", fontsize=11, fontweight="bold")

    # Per-split
    split_sep = m.groupby("split")["sepsis_positive"].agg(["sum", "count"])
    split_sep["non_sep"] = split_sep["count"] - split_sep["sum"]
    split_sep = split_sep.loc[["train", "val", "test"]]
    x = np.arange(3)
    axes[1].bar(x - 0.2, split_sep["non_sep"], 0.4, label="Non-Sepsis", color="#4A90D9", alpha=0.85)
    axes[1].bar(x + 0.2, split_sep["sum"],     0.4, label="Sepsis",     color="#E74C3C", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Train", "Val", "Test"])
    axes[1].set_title("Class Distribution by Split")
    axes[1].set_ylabel("Number of Patients")
    axes[1].legend()

    fig.suptitle("Class Imbalance Summary", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, "class_distribution.png")


# ---------------------------------------------------------------------------
# 3. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves() -> None:
    log_files = sorted(LOGS_DIR.glob("train_*.json"))
    if not log_files:
        print("  No training logs found — skipping curves.")
        return

    for lf in log_files:
        with open(lf) as f:
            rows = json.load(f)
        if not rows:
            continue
        df    = pd.DataFrame(rows)
        label = lf.stem.replace("train_", "")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        if "auroc" in df.columns:
            axes[0].plot(df["epoch"], df["auroc"], color="#2ECC71", lw=2)
            axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val AUROC")
            axes[0].set_title(f"{label} — Validation AUROC")

        if "f1" in df.columns:
            axes[1].plot(df["epoch"], df["f1"], color="#E74C3C", lw=2)
            axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val F1")
            axes[1].set_title(f"{label} — Validation F1")

        plt.tight_layout()
        save_fig(fig, f"training_curve_{label}.png")


# ---------------------------------------------------------------------------
# 4. Results comparison bar chart
# ---------------------------------------------------------------------------

def plot_results_comparison() -> None:
    files = {
        "baseline_metrics":    RESULTS_DIR / "baseline_metrics.json",
        "proposed_tcn_metrics": RESULTS_DIR / "proposed_tcn_metrics.json",
    }
    all_metrics = {}
    for key, p in files.items():
        if p.exists():
            with open(p) as f:
                all_metrics.update(json.load(f))

    if not all_metrics:
        print("  No results found — skipping comparison chart.")
        return

    models = list(all_metrics.keys())
    aurocs  = [all_metrics[m]["auroc"]  for m in models]
    f1s     = [all_metrics[m]["f1"]     for m in models]
    recalls = [all_metrics[m]["recall"] for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(models)), 5))
    c1 = ax.bar(x - width, aurocs,  width, label="AUROC",  color="#4A90D9", alpha=0.85)
    c2 = ax.bar(x,         f1s,     width, label="F1",     color="#2ECC71", alpha=0.85)
    c3 = ax.bar(x + width, recalls, width, label="Recall", color="#E74C3C", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set Performance")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))

    for bars in (c1, c2, c3):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                    f"{b.get_height():.3f}", ha="center", fontsize=7)

    plt.tight_layout()
    save_fig(fig, "results_comparison_bar.png")


# ---------------------------------------------------------------------------
# 5. Ablation heatmap
# ---------------------------------------------------------------------------

def plot_ablation_heatmap() -> None:
    p = RESULTS_DIR / "proposed_tcn_metrics.json"
    if not p.exists():
        print("  proposed_tcn_metrics.json not found — skipping ablation heatmap.")
        return

    with open(p) as f:
        metrics = json.load(f)

    rows = []
    for name, m in metrics.items():
        rows.append({
            "Experiment":    name,
            "Miss.Ind.":     "Yes" if m.get("uses_missingness") else "No",
            "Pos.Weight":    "Yes" if m.get("uses_pos_weight")  else "No",
            "AUROC":         m["auroc"],
            "AUPRC":         m["auprc"],
            "Recall":        m["recall"],
            "F1":            m["f1"],
        })

    df = pd.DataFrame(rows)

    metric_cols = ["AUROC", "AUPRC", "Recall", "F1"]
    heat_df = df[metric_cols].copy()
    heat_df.index = df["Experiment"]

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="YlGn",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Score"})
    ax.set_title("Ablation Study — TCN Variants")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    save_fig(fig, "ablation_heatmap.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Generating result visualisations …")
    plot_missingness()
    plot_class_distribution()
    plot_training_curves()
    plot_results_comparison()
    plot_ablation_heatmap()
    print("\nAll figures saved to:", FIGURES_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
