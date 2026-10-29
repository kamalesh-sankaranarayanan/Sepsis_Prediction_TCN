"""
train_proposed_tcn.py — Phase 5 + 8: Proposed missingness-aware TCN.

Trains and evaluates:
  • TCN_Baseline  : clinical values only (X[:, :D])
  • TCN_Proposed  : clinical values + missingness indicators (X[:, :2D])

Both are trained under identical conditions for a controlled comparison.

Ablation Study (Phase 8):
  Experiment 1: TCN_Baseline          (no imbalance handling)
  Experiment 2: TCN_Baseline + pos_weight
  Experiment 3: TCN_Proposed          (missingness indicators)
  Experiment 4: TCN_Proposed + pos_weight

Results are saved to results/proposed_tcn_metrics.json.

Run:
    python src/train_proposed_tcn.py
    python src/train_proposed_tcn.py --epochs 5    # quick test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import (
    MODELS_DIR, N_FEATURES, RESULTS_DIR, SEED, TCN_CFG, TRAIN_CFG,
)
from evaluate import (
    evaluate_sequence_model, optimal_threshold,
    roc_curve_plot, pr_curve_plot, confusion_matrix_plot, save_metrics,
)
from models import build_model, count_params
from sequence_builder import (
    SepsisDataset, SepsisDatasetMiss, compute_pos_weight, make_loader,
)

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D = N_FEATURES


def banner(title: str) -> None:
    print(f"\n{'='*65}\n  {title}\n{'='*65}")


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def train_tcn(
    model_name:     str,
    dataset_cls,
    use_pos_weight: bool,
    label:          str,
    epochs:         int,
) -> tuple[dict, np.ndarray, np.ndarray]:

    banner(label)

    n_input = dataset_cls("train", max_seq_len=72).n_input
    print(f"  Input features: {n_input}")

    tr_ds = dataset_cls("train", max_seq_len=72)
    vl_ds = dataset_cls("val",   max_seq_len=72)
    te_ds = dataset_cls("test",  max_seq_len=72)

    tr_ldr = make_loader(tr_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=True)
    vl_ldr = make_loader(vl_ds, batch_size=TRAIN_CFG["batch_size"])
    te_ldr = make_loader(te_ds, batch_size=TRAIN_CFG["batch_size"])

    model = build_model(model_name, n_input, TCN_CFG).to(DEVICE)
    print(f"  Parameters: {count_params(model):,}")

    if use_pos_weight:
        pos_w  = compute_pos_weight("train")
        print(f"  pos_weight: {pos_w:.2f}")
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_w, device=DEVICE)
        )
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    opt   = torch.optim.Adam(
        model.parameters(), lr=TRAIN_CFG["lr"], weight_decay=TRAIN_CFG["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3
    )

    safe_label  = label.lower().replace(" ", "_").replace("+", "plus")
    best_path   = MODELS_DIR / f"{safe_label}.pt"
    best_auroc  = -1.0
    no_improve  = 0
    log_rows    = []
    t0          = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for x_b, y_b, lens in tr_ldr:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            opt.zero_grad()
            logits = model(x_b, lens.to(DEVICE))
            mask   = torch.zeros_like(y_b, dtype=torch.bool)
            for i, L in enumerate(lens):
                mask[i, :L] = True
            loss = loss_fn(logits[mask], y_b[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG["grad_clip"])
            opt.step()
            ep_loss += loss.item() * mask.sum().item()

        vl_m, _, _ = evaluate_sequence_model(
            model, vl_ldr, DEVICE, f"{label} val", threshold=0.5
        )
        sched.step(vl_m["auroc"])
        log_rows.append({"epoch": epoch, **vl_m})
        print(f"  Epoch {epoch:3d}  loss={ep_loss:.4f}  "
              f"val_auroc={vl_m['auroc']:.4f}  val_f1={vl_m['f1']:.4f}")

        if vl_m["auroc"] > best_auroc:
            best_auroc = vl_m["auroc"]
            torch.save(model.state_dict(), best_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= TRAIN_CFG["patience"]:
                print(f"  Early stopping at epoch {epoch}.")
                break

    train_time = time.perf_counter() - t0

    log_path = ROOT / "logs" / f"train_{safe_label}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log_rows, f, indent=2)

    # Best checkpoint → test evaluation
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))

    vl_m, y_vl_t, y_vl_p = evaluate_sequence_model(
        model, vl_ldr, DEVICE, f"{label} (val)", threshold=0.5
    )
    thresh = optimal_threshold(y_vl_t, y_vl_p, metric="f1")

    te_m, y_te_t, y_te_p = evaluate_sequence_model(
        model, te_ldr, DEVICE, label, threshold=thresh
    )
    te_m["train_time_s"]   = round(train_time, 2)
    te_m["best_val_auroc"] = round(best_auroc, 4)
    te_m["uses_missingness"] = (dataset_cls != SepsisDataset)
    te_m["uses_pos_weight"]  = use_pos_weight

    confusion_matrix_plot(y_te_t, y_te_p, threshold=thresh, label=safe_label)
    return te_m, y_te_t, y_te_p


# ---------------------------------------------------------------------------
# Main: ablation study
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    args  = parser.parse_args()

    proc_train = ROOT / "data" / "processed" / "train"
    if not proc_train.is_dir() or not any(proc_train.glob("*.npz")):
        print("ERROR: Processed .npz files not found.")
        print("  Run:  python src/preprocess.py")
        sys.exit(1)

    epochs = args.epochs or TRAIN_CFG["epochs"]

    experiments = [
        # (model_name, dataset_cls, use_pos_weight, label)
        ("tcn", SepsisDataset,     False, "TCN_Baseline"),
        ("tcn", SepsisDataset,     True,  "TCN_Baseline_PosWeight"),
        ("tcn", SepsisDatasetMiss, False, "TCN_Proposed"),
        ("tcn", SepsisDatasetMiss, True,  "TCN_Proposed_PosWeight"),
    ]

    all_metrics = {}
    roc_data    = {}
    pr_data     = {}

    for model_name, ds_cls, use_pw, label in experiments:
        m, y_t, y_p = train_tcn(model_name, ds_cls, use_pw, label, epochs)
        all_metrics[label] = m
        roc_data[label]    = (y_t, y_p)
        pr_data[label]     = (y_t, y_p)

    # Save
    out = RESULTS_DIR / "proposed_tcn_metrics.json"
    save_metrics(all_metrics, out)
    print(f"\nResults saved: {out.relative_to(ROOT)}")

    # Plots
    roc_curve_plot(roc_data, filename="proposed_tcn_roc.png")
    pr_curve_plot( pr_data,  filename="proposed_tcn_pr.png")

    # Comparison table
    banner("ABLATION STUDY RESULTS")
    import pandas as pd
    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "Experiment":   name,
            "Miss.Ind.":    "Yes" if m["uses_missingness"] else "No",
            "Pos.Weight":   "Yes" if m["uses_pos_weight"]  else "No",
            "AUROC":        f"{m['auroc']:.4f}",
            "AUPRC":        f"{m['auprc']:.4f}",
            "Recall":       f"{m['recall']*100:.1f}%",
            "F1":           f"{m['f1']*100:.1f}%",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # Δ improvement: proposed vs baseline
    if "TCN_Baseline" in all_metrics and "TCN_Proposed" in all_metrics:
        b = all_metrics["TCN_Baseline"]
        p = all_metrics["TCN_Proposed"]
        banner("MISSINGNESS INDICATOR EFFECT (controlled, same pos_weight=False)")
        print(f"  ΔAUROC:  {p['auroc'] - b['auroc']:+.4f}")
        print(f"  ΔAUPRC:  {p['auprc'] - b['auprc']:+.4f}")
        print(f"  ΔRecall: {p['recall'] - b['recall']:+.4f}")
        print(f"  ΔF1:     {p['f1']    - b['f1']:+.4f}")

    if "TCN_Baseline_PosWeight" in all_metrics and "TCN_Proposed_PosWeight" in all_metrics:
        b = all_metrics["TCN_Baseline_PosWeight"]
        p = all_metrics["TCN_Proposed_PosWeight"]
        banner("MISSINGNESS INDICATOR EFFECT (with pos_weight)")
        print(f"  ΔAUROC:  {p['auroc'] - b['auroc']:+.4f}")
        print(f"  ΔAUPRC:  {p['auprc'] - b['auprc']:+.4f}")
        print(f"  ΔRecall: {p['recall'] - b['recall']:+.4f}")
        print(f"  ΔF1:     {p['f1']    - b['f1']:+.4f}")


if __name__ == "__main__":
    main()
