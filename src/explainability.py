"""
explainability.py — Phase 9: SHAP-based feature importance analysis.

Uses SHAP to explain:
  • XGBoost (TreeExplainer — fast, exact)
  • Baseline TCN  (DeepExplainer or GradientExplainer — sampling-based)
  • Proposed TCN  (same, with separate clinical and missingness indicator groups)

Outputs:
  • figures/shap_xgb_summary.png       — global beeswarm plot
  • figures/shap_tcn_baseline.png
  • figures/shap_tcn_proposed_clin.png — clinical features only
  • figures/shap_tcn_proposed_miss.png — missingness indicators only
  • results/shap_xgb_values.npy        — raw SHAP values

Run:
    python src/explainability.py
    python src/explainability.py --model xgb   # XGBoost only (fastest)
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import (
    FEATURE_NAMES, FIGURES_DIR, MODELS_DIR,
    N_FEATURES, RESULTS_DIR, SEED, SHAP_MAX_SAMPLES, TCN_CFG,
)
from sequence_builder import flat_loader

D = N_FEATURES
np.random.seed(SEED)


def section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ---------------------------------------------------------------------------
# XGBoost SHAP (TreeExplainer)
# ---------------------------------------------------------------------------

def explain_xgb() -> None:
    section("SHAP — XGBoost (TreeExplainer)")

    model_path = MODELS_DIR / "xgb.json"
    if not model_path.exists():
        print("  XGBoost model not found. Run train_baselines.py first.")
        return

    from xgboost import XGBClassifier
    clf = XGBClassifier()
    clf.load_model(str(model_path))

    X_te, y_te, feat_names = flat_loader("test")

    # Sub-sample for speed
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_te), min(SHAP_MAX_SAMPLES, len(X_te)), replace=False)
    X_s, y_s = X_te[idx], y_te[idx]

    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_s)

    np.save(str(RESULTS_DIR / "shap_xgb_values.npy"), shap_values)
    np.save(str(RESULTS_DIR / "shap_xgb_X.npy"),      X_s)

    # Global summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_s, feature_names=feat_names,
                      show=False, plot_size=None, max_display=20)
    plt.title("SHAP Feature Importance — XGBoost", fontsize=13)
    plt.tight_layout()
    out = FIGURES_DIR / "shap_xgb_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.relative_to(ROOT)}")

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_s, feature_names=feat_names,
                      plot_type="bar", show=False, max_display=20)
    plt.title("Mean |SHAP| — XGBoost", fontsize=13)
    plt.tight_layout()
    out2 = FIGURES_DIR / "shap_xgb_bar.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out2.relative_to(ROOT)}")

    # Top features summary
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(-mean_abs)
    print("\n  Top 10 features by mean |SHAP|:")
    for rank, i in enumerate(order[:10], 1):
        print(f"    {rank:2d}. {feat_names[i]:<25s} {mean_abs[i]:.4f}")


# ---------------------------------------------------------------------------
# TCN SHAP (GradientExplainer)
# ---------------------------------------------------------------------------

def explain_tcn(
    model_file: str,
    label: str,
    use_missingness: bool,
) -> None:
    section(f"SHAP — {label} (GradientExplainer)")

    model_path = MODELS_DIR / model_file
    if not model_path.exists():
        for alt in [
            model_file.replace("tcn_tcn_", "tcn_"),
            model_file.replace("tcn_", ""),
            f"tcn_{model_file}",
        ]:
            if (MODELS_DIR / alt).exists():
                model_path = MODELS_DIR / alt
                break
    if not model_path.exists():
        print(f"  Model not found: {model_path.name} — skip.")
        return

    import torch
    from models import build_model
    from sequence_builder import flat_loader

    n_input = 2 * D if use_missingness else D
    feat_names = (
        FEATURE_NAMES + [f"{f}_miss" for f in FEATURE_NAMES]
        if use_missingness else list(FEATURE_NAMES)
    )

    model = build_model("tcn", n_input, TCN_CFG)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    X_te, y_te, _ = flat_loader("test", use_missingness=use_missingness)
    X_tr, _,    _ = flat_loader("train", use_missingness=use_missingness)

    # Sub-sample
    rng    = np.random.default_rng(SEED)
    bg_idx = rng.choice(len(X_tr), min(100, len(X_tr)), replace=False)
    te_idx = rng.choice(len(X_te), min(SHAP_MAX_SAMPLES, len(X_te)), replace=False)

    # Reshape to (N, 1, C) — treat each hour as a single-timestep sequence
    X_bg = torch.tensor(X_tr[bg_idx, None, :], dtype=torch.float32)
    X_ex = torch.tensor(X_te[te_idx, None, :], dtype=torch.float32)

    # Wrap model: (N, 1, C) → (N, 1) probability
    class _Wrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            logit = self.m(x)   # (N, 1)
            return torch.sigmoid(logit)

    wrapper = _Wrapper(model)

    try:
        explainer   = shap.GradientExplainer(wrapper, X_bg)
        shap_values = explainer.shap_values(X_ex)  # (N, 1, C)
        # Squeeze time dimension
        if isinstance(shap_values, list):
            sv = shap_values[0].squeeze(1)
        else:
            sv = np.array(shap_values).squeeze(2) if np.array(shap_values).ndim == 4 else shap_values.squeeze(1)
    except Exception as exc:
        print(f"  GradientExplainer failed: {exc}")
        print("  Falling back to KernelExplainer (slow — sub-sampling)")
        wrapper_fn = lambda x: torch.sigmoid(
            model(torch.tensor(x[:, None, :], dtype=torch.float32))
        ).detach().numpy().squeeze(1)
        X_bg_np = X_tr[bg_idx]
        X_ex_np = X_te[te_idx[:50]]
        explainer   = shap.KernelExplainer(wrapper_fn, X_bg_np[:20])
        shap_values = explainer.shap_values(X_ex_np, nsamples=50)
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values

    # Plot
    safe_label = label.lower().replace(" ", "_")
    X_plot = X_te[te_idx[:len(sv)]]

    if not use_missingness:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, X_plot, feature_names=feat_names,
                          show=False, max_display=20)
        plt.title(f"SHAP Feature Importance — {label}", fontsize=13)
        plt.tight_layout()
        out = FIGURES_DIR / f"shap_{safe_label}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.relative_to(ROOT)}")
    else:
        # Clinical features
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv[:, :D], X_plot[:, :D],
                          feature_names=FEATURE_NAMES,
                          show=False, max_display=20)
        plt.title(f"SHAP — {label}: Clinical Features", fontsize=13)
        plt.tight_layout()
        out_c = FIGURES_DIR / f"shap_{safe_label}_clinical.png"
        plt.savefig(out_c, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_c.relative_to(ROOT)}")

        # Missingness indicators
        fig, ax = plt.subplots(figsize=(10, 8))
        miss_names = [f"{f}_miss" for f in FEATURE_NAMES]
        shap.summary_plot(sv[:, D:], X_plot[:, D:],
                          feature_names=miss_names,
                          show=False, max_display=20)
        plt.title(f"SHAP — {label}: Missingness Indicators", fontsize=13)
        plt.tight_layout()
        out_m = FIGURES_DIR / f"shap_{safe_label}_missingness.png"
        plt.savefig(out_m, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_m.relative_to(ROOT)}")

    mean_abs = np.abs(sv).mean(axis=0)
    order    = np.argsort(-mean_abs)
    print(f"\n  Top 10 features for {label}:")
    for rank, i in enumerate(order[:10], 1):
        print(f"    {rank:2d}. {feat_names[i]:<30s} {mean_abs[i]:.4f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["xgb", "tcn_baseline", "tcn_proposed", "all"],
        default="all"
    )
    args = parser.parse_args()

    run_all     = args.model == "all"
    run_xgb     = run_all or args.model == "xgb"
    run_base    = run_all or args.model == "tcn_baseline"
    run_propose = run_all or args.model == "tcn_proposed"

    proc_test = ROOT / "data" / "processed" / "test"
    if not proc_test.is_dir() or not any(proc_test.glob("*.npz")):
        print("Processed .npz files not found — run src/preprocess.py first.")
        sys.exit(1)

    if run_xgb:
        explain_xgb()

    if run_base:
        explain_tcn(
            model_file    = "tcn_tcn_baseline.pt",
            label         = "TCN Baseline",
            use_missingness = False,
        )

    if run_propose:
        explain_tcn(
            model_file    = "tcn_tcn_proposed.pt",
            label         = "TCN Proposed",
            use_missingness = True,
        )

    print("\nExplainability analysis complete.")


if __name__ == "__main__":
    main()
