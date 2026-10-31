"""
config.py — Central configuration for the Missingness-Aware TCN project.

All paths are resolved relative to the repository root so the project
can be run from any working directory.
"""

from __future__ import annotations
from pathlib import Path
import json

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW      = ROOT / "data" / "raw"
DATA_PROC     = ROOT / "data" / "processed"
RESULTS_DIR   = ROOT / "results"
MODELS_DIR    = ROOT / "models"
LOGS_DIR      = ROOT / "logs"
FIGURES_DIR   = ROOT / "figures"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
MANIFEST_CSV        = DATA_PROC / "split_manifest.csv"
PREPROCESSING_CFG   = DATA_PROC / "preprocessing_config.json"
SPLIT_CONFIG_JSON   = DATA_PROC / "split_config.json"

LABEL_COL = "SepsisLabel"

# Load preprocessing artefacts (feature list, means, stds).
with open(PREPROCESSING_CFG, encoding="utf-8") as _f:
    _cfg = json.load(_f)

FEATURE_NAMES: list[str] = _cfg["feature_names"]
TRAIN_MEANS: list[float] = _cfg["training_means"]
TRAIN_STDS:  list[float] = _cfg["training_stds"]
N_FEATURES = len(FEATURE_NAMES)   # 40 clinical features

# Each patient .npz has X of shape (T, 3*D):
#   columns [0:D]     = scaled clinical values
#   columns [D:2D]    = observation mask  (1 = originally observed, 0 = imputed)
#   columns [2D:3D]   = log1p(time since last observation)
# The observation mask IS the missingness indicator (1-mask → missing).

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Sequence / model inputs
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = 72          # cap ICU stay to 72 h for fixed-length batching
STRIDE       = 1           # reserved; retained study uses prefixes, not sliding windows

# ---------------------------------------------------------------------------
# TCN hyper-parameters
# ---------------------------------------------------------------------------
TCN_CFG = {
    "num_channels":    [64, 64, 64, 64],   # channels per residual block
    "kernel_size":     3,
    "dropout":         0.2,
}

# ---------------------------------------------------------------------------
# LSTM / GRU hyper-parameters
# ---------------------------------------------------------------------------
RNN_CFG = {
    "hidden_size":  128,
    "num_layers":   2,
    "dropout":      0.2,
}

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
TRAIN_CFG = {
    "batch_size":        64,
    "epochs":            50,
    "lr":                1e-3,
    "weight_decay":      1e-4,
    "patience":          8,          # early-stopping patience (val AUROC)
    "pos_weight_factor": None,       # None → compute from training data
    "grad_clip":         1.0,
}

# ---------------------------------------------------------------------------
# XGBoost hyper-parameters
# ---------------------------------------------------------------------------
XGB_CFG = {
    "n_estimators":    300,
    "max_depth":       6,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":     "auc",
    "tree_method":     "hist",
    "random_state":    SEED,
}

# ---------------------------------------------------------------------------
# Logistic Regression hyper-parameters
# ---------------------------------------------------------------------------
LR_CFG = {
    "C":              0.1,
    "max_iter":       1000,
    "solver":         "lbfgs",
    "class_weight":   "balanced",
    "random_state":   SEED,
}

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_THRESHOLD = 0.5   # default decision threshold (tuned per-model on val)

# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
SHAP_MAX_SAMPLES = 500   # max rows sent to SHAP explainer (speed / memory)
