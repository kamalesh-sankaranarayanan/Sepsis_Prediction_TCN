"""Shared, side-effect-free data and metrics for the retained 72-row study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
MAX_ROWS = 72
MODEL_CFG = {"num_channels": [64, 64, 64, 64], "kernel_size": 3, "dropout": 0.2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_paths(split: str, root: Path = ROOT) -> list[Path]:
    if split not in {"train", "val", "test"}:
        raise ValueError("Unknown split")
    base = root / "data/processed"
    manifest = pd.read_csv(base / "split_manifest.csv")
    if manifest.patient_id.isna().any() or manifest.patient_id.duplicated().any():
        raise ValueError("Manifest has missing or duplicate patient IDs")
    if set(manifest.split) != {"train", "val", "test"}:
        raise ValueError("Manifest must contain exactly train, val, test")
    ids = set(manifest.loc[manifest.split == split, "patient_id"])
    found = {p.stem: p for p in (base / split).glob("*.npz")}
    if set(found) != ids:
        raise ValueError(f"{split}: manifest/file mismatch: missing {len(ids-set(found))}, extra {len(set(found)-ids)}")
    return [found[pid] for pid in sorted(ids)]


class CachedPatients(Dataset):
    """Load one split once; truncate before counting class weights."""
    def __init__(self, split: str, root: Path = ROOT):
        cfg = json.loads((root / "data/processed/preprocessing_config.json").read_text())
        self.d = len(cfg["feature_names"])
        self.items = []
        self.ids = []
        for path in split_paths(split, root):
            with np.load(path, allow_pickle=False) as z:
                x = z["X"][:MAX_ROWS].astype(np.float32)
                y = z["y"][:MAX_ROWS].astype(np.float32)
            if x.shape != (len(y), 3*self.d) or not len(y) or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
                raise ValueError(f"Invalid processed patient: {path.stem}")
            features = np.concatenate([x[:, :self.d], 1-x[:, self.d:2*self.d]], axis=1)
            self.items.append((torch.from_numpy(features), torch.from_numpy(y)))
            self.ids.append(path.stem)
        self.positives = sum(int(y.sum()) for _, y in self.items)
        self.hours = sum(len(y) for _, y in self.items)
        if not self.positives or self.positives == self.hours:
            raise ValueError("Split needs both label classes")

    @property
    def pos_weight(self):
        return (self.hours - self.positives) / self.positives

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch):
    lengths = torch.tensor([len(y) for _, y in batch])
    x = torch.nn.utils.rnn.pad_sequence([x for x, _ in batch], batch_first=True)
    y = torch.nn.utils.rnn.pad_sequence([y for _, y in batch], batch_first=True)
    mask = torch.arange(y.shape[1])[None, :] < lengths[:, None]
    return x, y, mask


def best_f1_threshold(y, probabilities):
    """Exact validation PR thresholds, including scores below 0.05; low tie wins."""
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    if not len(thresholds):
        raise ValueError("No validation thresholds")
    scores = np.divide(2*precision[:-1]*recall[:-1], precision[:-1]+recall[:-1],
                       out=np.zeros_like(precision[:-1]), where=(precision[:-1]+recall[:-1]) > 0)
    return float(thresholds[int(np.argmax(scores))])


def metrics(y, probabilities, threshold):
    y = np.asarray(y)
    probabilities = np.asarray(probabilities)
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {"accuracy": float(accuracy_score(y, predicted)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "specificity": float(tn/max(tn+fp, 1)),
            "auroc": float(roc_auc_score(y, probabilities)) if both else None,
            "auprc": float(average_precision_score(y, probabilities)) if both else None,
            "threshold": float(threshold), "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp), "hours": len(y), "positive_hours": int(y.sum())}


@torch.inference_mode()
def predict(model, loader):
    model.eval()
    truths, probabilities = [], []
    for x, y, mask in loader:
        probabilities.append(torch.sigmoid(model(x))[mask].numpy())
        truths.append(y[mask].numpy())
    return np.concatenate(truths), np.concatenate(probabilities)
