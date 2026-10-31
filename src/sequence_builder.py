"""
sequence_builder.py — Phase 3: Build patient-level sequences from processed .npz files.

Each processed .npz contains:
  X       : (T, 3*D) — [scaled_clinical | observed_mask | log_delta]
  y       : (T,)     — binary SepsisLabel per hour
  observed: (T, D)   — uint8 binary mask (1=observed, 0=imputed)

This module provides:
  • SepsisDataset      — PyTorch Dataset for (X, y) sequences
  • SepsisDatasetMiss  — same, but with missingness indicators (X[:, :2D])
  • collate_fn         — zero-pads a batch of variable-length sequences
  • flat_loader        — returns flattened (n_hours, features) arrays for
                         sklearn / XGBoost baselines

Run a self-test:
    python src/sequence_builder.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import (
    DATA_PROC,
    FEATURE_NAMES,
    MANIFEST_CSV,
    N_FEATURES,
    PREPROCESSING_CFG,
    SEED,
    MAX_SEQ_LEN,
)

D = N_FEATURES   # 40 features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_patient(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a single patient .npz and return (X, y, observed)."""
    with np.load(path, allow_pickle=False) as z:
        X   = z["X"].astype(np.float32)       # (T, 3D)
        y   = z["y"].astype(np.float32)        # (T,)
        obs = z["observed"].astype(np.float32) # (T, D)
    return X, y, obs


def _get_split_paths(
    split: Literal["train", "val", "test"],
) -> list[Path]:
    from study import split_paths
    return split_paths(split, ROOT)


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class SepsisDataset(Dataset):
    """
    Baseline dataset: clinical values only (X[:, :D]).

    Each item is (x, y, length):
      x      : (T, D) float32  — scaled clinical features
      y      : (T,)   float32  — binary labels
      length : int             — actual sequence length T
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        max_seq_len: int | None = None,
    ) -> None:
        self.paths       = _get_split_paths(split)
        self.max_seq_len = max_seq_len
        self.n_input     = D

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        X, y, _ = load_patient(self.paths[idx])
        x_clin  = X[:, :D]                    # clinical values only
        if self.max_seq_len:
            x_clin = x_clin[:self.max_seq_len]
            y      = y[:self.max_seq_len]
        T = len(y)
        return (
            torch.from_numpy(x_clin),
            torch.from_numpy(y),
            T,
        )


class SepsisDatasetMiss(Dataset):
    """
    Missingness-aware dataset: clinical values + missingness indicators.

    Input: X[:, :D] (clinical) ∥ X[:, D:2D] (observation mask)
    The mask bit is 1 when a value WAS observed (0 = originally missing / imputed).
    We convert to a missingness indicator (1 = missing) for interpretability:
      miss_indicator = 1 - observed_mask

    Each item is (x, y, length):
      x      : (T, 2D) float32 — [scaled_clinical | missingness_indicator]
      y      : (T,)   float32
      length : int
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        max_seq_len: int | None = None,
    ) -> None:
        self.paths       = _get_split_paths(split)
        self.max_seq_len = max_seq_len
        self.n_input     = 2 * D

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        X, y, _ = load_patient(self.paths[idx])
        x_clin  = X[:, :D]
        x_obs   = X[:, D:2 * D]             # observation mask (1=observed)
        x_miss  = 1.0 - x_obs               # missingness indicator (1=missing)
        x_full  = np.concatenate([x_clin, x_miss], axis=1)

        if self.max_seq_len:
            x_full = x_full[:self.max_seq_len]
            y      = y[:self.max_seq_len]
        T = len(y)
        return (
            torch.from_numpy(x_full),
            torch.from_numpy(y),
            T,
        )


class SepsisDatasetDelta(Dataset):
    """
    Full-feature dataset: clinical + obs mask + log-delta time.
    Shape per patient: (T, 3D).
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        max_seq_len: int | None = None,
    ) -> None:
        self.paths       = _get_split_paths(split)
        self.max_seq_len = max_seq_len
        self.n_input     = 3 * D

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        X, y, _ = load_patient(self.paths[idx])
        if self.max_seq_len:
            X = X[:self.max_seq_len]
            y = y[:self.max_seq_len]
        T = len(y)
        return (
            torch.from_numpy(X),
            torch.from_numpy(y),
            T,
        )


# ---------------------------------------------------------------------------
# Collate function (variable-length → padded batch)
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    Pad a list of (x, y, length) tuples to the max length in the batch.

    Returns:
      x_pad    : (B, T_max, C) float32
      y_pad    : (B, T_max)    float32
      lengths  : (B,)          int64
    """
    xs, ys, lengths = zip(*batch)
    T_max = max(lengths)
    C     = xs[0].shape[1]

    x_pad = torch.zeros(len(xs), T_max, C, dtype=torch.float32)
    y_pad = torch.zeros(len(ys), T_max,    dtype=torch.float32)

    for i, (x, y, L) in enumerate(zip(xs, ys, lengths)):
        x_pad[i, :L] = x
        y_pad[i, :L] = y

    return x_pad, y_pad, torch.tensor(lengths, dtype=torch.long)


def make_loader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Flat loader for sklearn / XGBoost
# ---------------------------------------------------------------------------

def flat_loader(
    split: Literal["train", "val", "test"],
    use_missingness: bool = False,
    max_seq_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Return flattened arrays for tree / linear baselines.

    Returns
    -------
    X_flat : (N_hours, n_features)
    y_flat : (N_hours,)
    feat_names : list[str]
    """
    paths = _get_split_paths(split)
    Xs, ys = [], []
    for p in paths:
        X, y, _ = load_patient(p)
        if max_seq_len:
            X = X[:max_seq_len]
            y = y[:max_seq_len]
        if use_missingness:
            x_clin = X[:, :D]
            x_miss = 1.0 - X[:, D:2 * D]
            X_use  = np.concatenate([x_clin, x_miss], axis=1)
        else:
            X_use = X[:, :D]
        Xs.append(X_use)
        ys.append(y)

    X_flat = np.concatenate(Xs, axis=0).astype(np.float32)
    y_flat = np.concatenate(ys, axis=0).astype(np.float32)

    if use_missingness:
        feat_names = FEATURE_NAMES + [f"{f}_miss" for f in FEATURE_NAMES]
    else:
        feat_names = list(FEATURE_NAMES)

    return X_flat, y_flat, feat_names


# ---------------------------------------------------------------------------
# Positive-weight for BCEWithLogitsLoss
# ---------------------------------------------------------------------------

def compute_pos_weight(split: str = "train", max_seq_len: int | None = MAX_SEQ_LEN) -> float:
    """Count exactly the retained valid timesteps; None explicitly means full stay."""
    paths = _get_split_paths(split)
    total_pos = total_neg = 0
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            y = z["y"]
        if max_seq_len is not None:
            y = y[:max_seq_len]
        total_pos += int((y == 1).sum())
        total_neg += int((y == 0).sum())
    return total_neg / max(total_pos, 1)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    proc_dir = DATA_PROC / "train"
    if not proc_dir.is_dir() or not any(proc_dir.glob("*.npz")):
        print("Processed .npz files not found — run src/preprocess.py first.")
        sys.exit(0)

    print("=== Baseline dataset ===")
    ds = SepsisDataset("train", max_seq_len=72)
    print(f"  Patients: {len(ds)}")
    x, y, L = ds[0]
    print(f"  x shape: {x.shape}, y shape: {y.shape}, length: {L}")

    print("\n=== Missingness-aware dataset ===")
    ds_m = SepsisDatasetMiss("train", max_seq_len=72)
    x2, y2, L2 = ds_m[0]
    print(f"  x shape: {x2.shape}, y shape: {y2.shape}, length: {L2}")
    miss_cols = x2[:, D:]
    print(f"  Missingness indicator range: [{miss_cols.min():.0f}, {miss_cols.max():.0f}]")

    print("\n=== DataLoader (batch) ===")
    loader = make_loader(ds, batch_size=4, shuffle=True)
    xb, yb, lb = next(iter(loader))
    print(f"  x batch: {xb.shape}, y batch: {yb.shape}, lengths: {lb.tolist()}")

    print("\n=== Flat loader ===")
    Xf, yf, fn = flat_loader("train")
    print(f"  X_flat: {Xf.shape}, y_flat: {yf.shape}")
    print(f"  Positive rate: {yf.mean()*100:.2f}%")

    pw = compute_pos_weight("train")
    print(f"\n  Positive weight (neg/pos): {pw:.2f}")
    print("\nSelf-test complete.")
