"""
evaluate.py — Phase 7: Unified evaluation utilities.

Provides:
  • evaluate_model()    — compute all metrics from predictions
  • optimal_threshold() — find best threshold on val set
  • evaluate_sequence_model() — run a PyTorch model over a DataLoader
  • evaluate_sklearn_model()  — run an sklearn/XGB model on flat arrays
  • confusion_matrix_plot()
  • roc_curve_plot()
  • pr_curve_plot()

Usage:
    from evaluate import evaluate_model
    metrics = evaluate_model(y_true, y_prob)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    threshold: float = 0.5,
    label:     str   = "model",
    verbose:   bool  = True,
) -> dict[str, float]:
    """
    Compute classification metrics from binary labels and probabilities.

    Returns a dict with: accuracy, precision, recall, specificity, f1,
    auroc, auprc, threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)

    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(   y_true, y_pred, zero_division=0)
    f1    = f1_score(        y_true, y_pred, zero_division=0)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec = tn / max(tn + fp, 1)

    metrics = {
        "accuracy":    round(float(acc),   4),
        "precision":   round(float(prec),  4),
        "recall":      round(float(rec),   4),
        "specificity": round(float(spec),  4),
        "f1":          round(float(f1),    4),
        "auroc":       round(float(auroc), 4),
        "auprc":       round(float(auprc), 4),
        "threshold":   round(float(threshold), 4),
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
    }

    if verbose:
        print(f"\n  [{label}]")
        print(f"  Accuracy:    {acc*100:6.2f}%")
        print(f"  Precision:   {prec*100:6.2f}%")
        print(f"  Recall:      {rec*100:6.2f}%")
        print(f"  Specificity: {spec*100:6.2f}%")
        print(f"  F1:          {f1*100:6.2f}%")
        print(f"  AUROC:       {auroc:.4f}")
        print(f"  AUPRC:       {auprc:.4f}")
        print(f"  Confusion Matrix:  TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    return metrics


def optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    """
    Find the decision threshold that maximises `metric` on a validation set.

    metric : 'f1' | 'recall' | 'precision' | 'balanced'
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_val = 0.5, -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            v = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            v = recall_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            v = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "balanced":
            rec  = recall_score(y_true, y_pred, zero_division=0)
            spec = (y_true[y_pred == 0] == 0).mean() if (y_pred == 0).any() else 0.0
            v    = (rec + spec) / 2
        else:
            raise ValueError(f"Unknown metric: {metric}")
        if v > best_val:
            best_val, best_t = v, t

    return float(best_t)


# ---------------------------------------------------------------------------
# PyTorch sequence model evaluator
# ---------------------------------------------------------------------------

def evaluate_sequence_model(
    model,
    loader,
    device: str = "cpu",
    label:  str = "model",
    threshold: float = 0.5,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Run a PyTorch model over a DataLoader and return metrics + raw predictions.

    Returns
    -------
    metrics   : dict from evaluate_model()
    y_true    : (N,) flat ground truth
    y_prob    : (N,) flat predicted probabilities
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    all_probs, all_true = [], []

    t0 = time.perf_counter()
    with torch.no_grad():
        for x_batch, y_batch, lengths in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch, lengths.to(device))   # (B, T)
            probs  = torch.sigmoid(logits)                # (B, T)

            # Only keep non-padded time steps
            for i, L in enumerate(lengths):
                all_probs.append(probs[i, :L].cpu().numpy())
                all_true.append(y_batch[i, :L].cpu().numpy())

    elapsed = time.perf_counter() - t0

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_true)

    metrics = evaluate_model(y_true, y_prob, threshold=threshold, label=label)
    metrics["inference_seconds"] = round(elapsed, 3)
    return metrics, y_true, y_prob


# ---------------------------------------------------------------------------
# sklearn / XGB evaluator
# ---------------------------------------------------------------------------

def evaluate_sklearn_model(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    label: str  = "model",
    threshold: float = 0.5,
) -> tuple[dict, np.ndarray, np.ndarray]:
    t0 = time.perf_counter()
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X)[:, 1]
    else:
        y_prob = clf.decision_function(X)
        # sigmoid-normalise
        y_prob = 1 / (1 + np.exp(-y_prob))
    elapsed = time.perf_counter() - t0

    metrics = evaluate_model(y, y_prob, threshold=threshold, label=label)
    metrics["inference_seconds"] = round(elapsed, 3)
    return metrics, y, y_prob


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def confusion_matrix_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    label: str = "model",
    save: bool = True,
) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion Matrix — {label}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()

    if save:
        p = FIGURES_DIR / f"cm_{label.lower().replace(' ', '_')}.png"
        fig.savefig(p, dpi=150)
        print(f"  Saved: {p.relative_to(ROOT)}")
    plt.close(fig)


def roc_curve_plot(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    save: bool = True,
    filename: str = "roc_curve.png",
) -> None:
    """
    results : {model_label: (y_true, y_prob)}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)

    for label, (yt, yp) in results.items():
        fpr, tpr, _ = roc_curve(yt, yp)
        auroc = roc_auc_score(yt, yp)
        ax.plot(fpr, tpr, label=f"{label} (AUROC={auroc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save:
        p = FIGURES_DIR / filename
        fig.savefig(p, dpi=150)
        print(f"  Saved: {p.relative_to(ROOT)}")
    plt.close(fig)


def pr_curve_plot(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    save: bool = True,
    filename: str = "pr_curve.png",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    for label, (yt, yp) in results.items():
        prec, rec, _ = precision_recall_curve(yt, yp)
        auprc = average_precision_score(yt, yp)
        ax.plot(rec, prec, label=f"{label} (AUPRC={auprc:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save:
        p = FIGURES_DIR / filename
        fig.savefig(p, dpi=150)
        print(f"  Saved: {p.relative_to(ROOT)}")
    plt.close(fig)


def save_metrics(metrics: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    y_t = rng.integers(0, 2, size=1000)
    y_p = rng.random(size=1000)
    m   = evaluate_model(y_t, y_p, label="random")
    t   = optimal_threshold(y_t, y_p, metric="f1")
    print(f"\n  Optimal F1 threshold: {t:.2f}")
    confusion_matrix_plot(y_t, y_p, threshold=t, label="test")
    roc_curve_plot({"Random": (y_t, y_p)})
    pr_curve_plot( {"Random": (y_t, y_p)})
    print("Evaluate self-test passed.")
