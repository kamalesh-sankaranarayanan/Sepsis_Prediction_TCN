"""
train_baselines.py — Phase 4: Train and evaluate baseline models.

Models trained:
  1. Logistic Regression  (flat hour-level features)
  2. XGBoost              (flat hour-level features)
  3. LSTM                 (sequence model)
  4. GRU                  (sequence model)
  5. TCN (baseline)       (sequence model, clinical features only)

All models use the same patient-level train/val/test split.
Logistic Regression and XGBoost use class_weight / scale_pos_weight
to handle imbalance. Sequence models use BCEWithLogitsLoss(pos_weight=…).

Results are saved to results/baseline_metrics.json.
Model checkpoints are saved to models/.

Run:
    python src/train_baselines.py
    python src/train_baselines.py --models lr xgb   # subset
    python src/train_baselines.py --epochs 5         # quick test
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import (
    LR_CFG, MODELS_DIR, N_FEATURES, RESULTS_DIR, RNN_CFG,
    SEED, TCN_CFG, TRAIN_CFG,
    MAX_SEQ_LEN,
)
from evaluate import (
    evaluate_sequence_model, evaluate_sklearn_model,
    optimal_threshold, roc_curve_plot, pr_curve_plot,
    confusion_matrix_plot, save_metrics,
)
from models import build_model, count_params
from sequence_builder import (
    SepsisDataset, flat_loader, make_loader, compute_pos_weight,
)

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUN_LOG_DIR = ROOT / "logs"


def banner(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def train_lr(args) -> dict:
    banner("LOGISTIC REGRESSION")
    from sklearn.linear_model import LogisticRegression

    X_tr, y_tr, fn = flat_loader("train")
    X_vl, y_vl, _  = flat_loader("val")
    X_te, y_te, _   = flat_loader("test")

    print(f"  Train: {X_tr.shape}  Val: {X_vl.shape}  Test: {X_te.shape}")
    print(f"  Train positive rate: {y_tr.mean()*100:.2f}%")

    clf = LogisticRegression(**LR_CFG)
    t0  = time.perf_counter()
    clf.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Tune threshold on validation
    _, y_vl_true, y_vl_prob = evaluate_sklearn_model(clf, X_vl, y_vl, "LR val")
    thresh = optimal_threshold(y_vl_true, y_vl_prob, metric="f1")

    # Evaluate on test
    te_metrics, y_te_true, y_te_prob = evaluate_sklearn_model(
        clf, X_te, y_te, "Logistic Regression", threshold=thresh
    )
    te_metrics["train_time_s"] = round(train_time, 2)

    confusion_matrix_plot(y_te_true, y_te_prob, threshold=thresh, label="LR")

    out = MODELS_DIR / "lr.pkl"
    with open(out, "wb") as f:
        pickle.dump(clf, f)
    print(f"  Saved model: {out.relative_to(ROOT)}")

    return te_metrics


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgb(args) -> dict:
    banner("XGBOOST")
    from xgboost import XGBClassifier

    X_tr, y_tr, _ = flat_loader("train")
    X_vl, y_vl, _ = flat_loader("val")
    X_te, y_te, _ = flat_loader("test")

    pos_w = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    cfg   = dict(XGB_CFG) if False else {}   # build from config
    from config import XGB_CFG as _xgb
    cfg   = dict(_xgb)
    cfg["scale_pos_weight"] = pos_w
    cfg.pop("use_label_encoder", None)

    clf = XGBClassifier(**cfg)
    t0  = time.perf_counter()
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        verbose=100,
    )
    train_time = time.perf_counter() - t0

    _, y_vl_true, y_vl_prob = evaluate_sklearn_model(clf, X_vl, y_vl, "XGB val")
    thresh = optimal_threshold(y_vl_true, y_vl_prob, metric="f1")

    te_metrics, y_te_true, y_te_prob = evaluate_sklearn_model(
        clf, X_te, y_te, "XGBoost", threshold=thresh
    )
    te_metrics["train_time_s"] = round(train_time, 2)

    confusion_matrix_plot(y_te_true, y_te_prob, threshold=thresh, label="XGB")

    out = MODELS_DIR / "xgb.json"
    clf.save_model(str(out))
    print(f"  Saved model: {out.relative_to(ROOT)}")

    return te_metrics


# ---------------------------------------------------------------------------
# Generic PyTorch sequence-model trainer
# ---------------------------------------------------------------------------

def train_sequence_model(
    model_name: str,
    n_input:    int,
    model_cfg:  dict,
    train_cfg:  dict,
    dataset_cls,
    label:      str,
    args,
) -> dict:
    banner(label)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    epochs     = args.epochs or train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]
    lr_val     = train_cfg["lr"]
    wd         = train_cfg["weight_decay"]
    patience   = train_cfg["patience"]
    grad_clip  = train_cfg["grad_clip"]

    tr_ds = dataset_cls("train", max_seq_len=MAX_SEQ_LEN)
    vl_ds = dataset_cls("val",   max_seq_len=MAX_SEQ_LEN)
    te_ds = dataset_cls("test",  max_seq_len=MAX_SEQ_LEN)

    tr_ldr = make_loader(tr_ds, batch_size=batch_size, shuffle=True)
    vl_ldr = make_loader(vl_ds, batch_size=batch_size)
    te_ldr = make_loader(te_ds, batch_size=batch_size)

    model = build_model(model_name, n_input, model_cfg).to(DEVICE)
    print(f"  Parameters: {count_params(model):,}")

    pos_w  = compute_pos_weight("train", max_seq_len=MAX_SEQ_LEN)
    print(f"  pos_weight: {pos_w:.2f}")
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_w, device=DEVICE)
    )

    opt   = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3
    )

    best_auroc   = -1.0
    best_path    = MODELS_DIR / f"{model_name.lower()}_{label.lower().replace(' ','_')}.pt"
    no_improve   = 0
    log_rows     = []
    t0           = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for x_b, y_b, lens in tr_ldr:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            opt.zero_grad()
            logits = model(x_b, lens.to(DEVICE))    # (B, T)
            # Build mask to ignore padded steps
            mask = torch.zeros_like(y_b, dtype=torch.bool)
            for i, L in enumerate(lens):
                mask[i, :L] = True
            loss = loss_fn(logits[mask], y_b[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ep_loss += loss.item() * mask.sum().item()

        # Validation
        vl_metrics, _, _ = evaluate_sequence_model(
            model, vl_ldr, DEVICE, f"{label} val", threshold=0.5
        )
        sched.step(vl_metrics["auroc"])

        log_rows.append({"epoch": epoch, **vl_metrics})
        print(f"  Epoch {epoch:3d}  loss={ep_loss:.4f}  "
              f"val_auroc={vl_metrics['auroc']:.4f}  "
              f"val_f1={vl_metrics['f1']:.4f}")

        if vl_metrics["auroc"] > best_auroc:
            best_auroc = vl_metrics["auroc"]
            torch.save(model.state_dict(), best_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    train_time = time.perf_counter() - t0

    # Save training log
    log_path = RUN_LOG_DIR / f"train_{model_name.lower()}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log_rows, f, indent=2)

    # Evaluate on test with best checkpoint
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))

    # Tune threshold on val
    vl_metrics, y_vl_t, y_vl_p = evaluate_sequence_model(
        model, vl_ldr, DEVICE, f"{label} (val)", threshold=0.5
    )
    thresh = optimal_threshold(y_vl_t, y_vl_p, metric="f1")

    te_metrics, y_te_t, y_te_p = evaluate_sequence_model(
        model, te_ldr, DEVICE, label, threshold=thresh
    )
    te_metrics["train_time_s"] = round(train_time, 2)
    te_metrics["best_val_auroc"] = round(best_auroc, 4)

    confusion_matrix_plot(y_te_t, y_te_p, threshold=thresh, label=label)
    print(f"  Saved model: {best_path.relative_to(ROOT)}")

    return te_metrics, y_te_t, y_te_p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    global MODELS_DIR, RESULTS_DIR, RUN_LOG_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["lr", "xgb", "lstm", "gru", "tcn"],
                        help="Which models to train")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max epochs for sequence models")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory under experiments/; legacy reproduction only, evaluates every model on test. Prefer run_experiments.py for selection.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if ROOT / "experiments" not in output.parents:
        parser.error("--output-dir must be a new directory under experiments/")
    output.mkdir(parents=True, exist_ok=False)
    MODELS_DIR, RESULTS_DIR, RUN_LOG_DIR = output / "models", output / "results", output / "logs"
    import evaluate
    evaluate.FIGURES_DIR = output / "figures"
    for directory in (MODELS_DIR, RESULTS_DIR, RUN_LOG_DIR, evaluate.FIGURES_DIR):
        directory.mkdir()

    proc_train = ROOT / "data" / "processed" / "train"
    if not proc_train.is_dir() or not any(proc_train.glob("*.npz")):
        print("ERROR: Processed .npz files not found.")
        print("  Run:  python src/preprocess.py")
        sys.exit(1)

    all_metrics   = {}
    roc_data      = {}   # {label: (y_true, y_prob)}
    pr_data       = {}

    models_to_run = [m.lower() for m in args.models]

    # ---- Logistic Regression ----
    if "lr" in models_to_run:
        m = train_lr(args)
        all_metrics["LR"] = m

    # ---- XGBoost ----
    if "xgb" in models_to_run:
        m = train_xgb(args)
        all_metrics["XGB"] = m

    # ---- LSTM ----
    if "lstm" in models_to_run:
        m, y_t, y_p = train_sequence_model(
            "lstm", N_FEATURES, RNN_CFG, TRAIN_CFG,
            SepsisDataset, "LSTM", args
        )
        all_metrics["LSTM"] = m
        roc_data["LSTM"] = (y_t, y_p)
        pr_data["LSTM"]  = (y_t, y_p)

    # ---- GRU ----
    if "gru" in models_to_run:
        m, y_t, y_p = train_sequence_model(
            "gru", N_FEATURES, RNN_CFG, TRAIN_CFG,
            SepsisDataset, "GRU", args
        )
        all_metrics["GRU"] = m
        roc_data["GRU"] = (y_t, y_p)
        pr_data["GRU"]  = (y_t, y_p)

    # ---- Baseline TCN ----
    if "tcn" in models_to_run:
        m, y_t, y_p = train_sequence_model(
            "tcn", N_FEATURES, TCN_CFG, TRAIN_CFG,
            SepsisDataset, "TCN_Baseline", args
        )
        all_metrics["TCN_Baseline"] = m
        roc_data["TCN Baseline"] = (y_t, y_p)
        pr_data["TCN Baseline"]  = (y_t, y_p)

    # Save metrics
    out = RESULTS_DIR / "baseline_metrics.json"
    save_metrics(all_metrics, out)
    print(f"\n\nAll baseline metrics saved to: {out.relative_to(ROOT)}")

    # Combined plots
    if roc_data:
        roc_curve_plot(roc_data, filename="baselines_roc.png")
        pr_curve_plot( pr_data,  filename="baselines_pr.png")

    # Print comparison table
    banner("BASELINE COMPARISON TABLE")
    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "Model":      name,
            "Accuracy":   f"{m['accuracy']*100:.1f}%",
            "Precision":  f"{m['precision']*100:.1f}%",
            "Recall":     f"{m['recall']*100:.1f}%",
            "F1":         f"{m['f1']*100:.1f}%",
            "AUROC":      f"{m['auroc']:.4f}",
            "AUPRC":      f"{m['auprc']:.4f}",
        })
    import pandas as pd
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
