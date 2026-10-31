"""Read-only patient parsing, exact saved preprocessing, and real TCN inference."""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models import TCN
from .study import ROOT, MAX_ROWS, MODEL_CFG, sha256

MAX_UPLOAD_BYTES = 2 * 1024 * 1024


@dataclass
class PreparedPatient:
    inputs: np.ndarray
    hours: np.ndarray
    observed_fraction: np.ndarray
    labels: np.ndarray | None
    total_rows: int
    invalid_values: int
    missing_values: int


def read_statistics(root: Path = ROOT):
    path = root / "data/processed/preprocessing_config.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    names = cfg["feature_names"]
    means = np.asarray(cfg["training_means"], dtype=np.float64)
    scales = np.asarray(cfg["training_stds"], dtype=np.float64)
    if (len(names) != 40 or len(set(names)) != 40 or means.shape != (40,) or
            scales.shape != (40,) or not np.isfinite(means).all() or
            not np.isfinite(scales).all() or (scales <= 0).any()):
        raise ValueError("Saved preprocessing statistics are invalid")
    return names, means, scales


def prepare_patient(content: bytes, root: Path = ROOT) -> PreparedPatient:
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload a nonempty CSV or PSV file smaller than 2 MiB.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The file must use UTF-8 text encoding.") from exc
    first = text.splitlines()[0] if text.splitlines() else ""
    delimiter = "|" if "|" in first else ","
    try:
        records = list(csv.reader(io.StringIO(text), delimiter=delimiter, strict=True))
    except csv.Error as exc:
        raise ValueError("Invalid CSV quoting or delimiter.") from exc
    records = [row for row in records if row and any(cell.strip() for cell in row)]
    if len(records) < 2:
        raise ValueError("The file needs a header and at least one hourly row.")
    header = [name.strip() for name in records[0]]
    if len(header) != len(set(header)):
        raise ValueError("Duplicate column names are not allowed.")
    names, means, scales = read_statistics(root)
    missing = set(names) - set(header)
    extra = set(header) - set(names) - {"SepsisLabel"}
    if missing or extra:
        raise ValueError(f"Schema mismatch. Missing: {', '.join(sorted(missing)) or 'none'}. Unexpected: {', '.join(sorted(extra)) or 'none'}.")
    if any(len(row) != len(header) for row in records[1:]):
        raise ValueError("Every row must have the same number of columns as the header.")
    if len(records) > 10001:
        raise ValueError("At most 10,000 input rows are supported; only the first 72 are scored.")
    # Pandas' default NA vocabulary matches the inherited preprocessing reader.
    frame = pd.read_csv(io.StringIO(text), sep=delimiter)
    frame.columns = header
    if len(frame) == 0:
        raise ValueError("No hourly rows were found.")
    original = frame[names]
    numeric = original.apply(pd.to_numeric, errors="coerce")
    invalid = int((original.notna() & numeric.isna()).to_numpy().sum())
    values = numeric.to_numpy(dtype=np.float64)
    invalid += int(np.isinf(values).sum())
    values[~np.isfinite(values)] = np.nan
    hours = values[:, names.index("ICULOS")]
    if (not np.isfinite(hours).all() or (hours < 1).any() or
            not np.equal(hours, np.floor(hours)).all() or not np.equal(np.diff(hours), 1).all()):
        raise ValueError("ICULOS must contain positive integer hours increasing by exactly one; starting above 1 is allowed.")
    labels = None
    if "SepsisLabel" in frame:
        labels = pd.to_numeric(frame.SepsisLabel, errors="coerce").to_numpy(dtype=np.float32)
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Optional SepsisLabel must contain only 0 or 1 without missing values.")
        labels = labels[:MAX_ROWS]
    total_rows = len(values)
    values = values[:MAX_ROWS]
    observed = np.isfinite(values)
    filled = pd.DataFrame(values).ffill().to_numpy()
    filled = np.where(np.isfinite(filled), filled, means)
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = ((filled-means)/scales).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("Values exceed the supported numerical range.")
    inputs = np.concatenate([scaled, (~observed).astype(np.float32)], axis=1)
    return PreparedPatient(inputs, hours[:MAX_ROWS].copy(), observed.mean(axis=1), labels,
                           total_rows, invalid, int((~observed).sum()))


def available_models(root: Path = ROOT):
    historical = json.loads((root/"results/proposed_tcn_metrics.json").read_text())
    specs = [{"id": "historical_proposed", "name": "Historical proposed TCN",
              "checkpoint": root/"models/tcn_proposed.pt", "model_cfg": MODEL_CFG,
              "threshold": historical["TCN_Proposed"]["threshold"],
              "test": historical["TCN_Proposed"], "kind": "Reproduced historical result"}]
    for selection_path in sorted((root/"experiments").glob("*/selection.json"), reverse=True):
        directory = selection_path.parent
        status_path = directory/"status.json"
        if not status_path.exists() or json.loads(status_path.read_text()).get("status") != "complete":
            continue
        selection = json.loads(selection_path.read_text())
        if selection.get("selection_used_test") is not False or selection.get("max_rows") != MAX_ROWS or selection.get("n_input") != 80:
            continue
        checkpoint = (directory/selection["checkpoint"]).resolve()
        if directory.resolve() not in checkpoint.parents:
            continue
        result = json.loads((directory/"test_comparison.json").read_text())
        specs.insert(0, {"id": directory.name, "name": "Validation-selected TCN: "+selection["name"],
                         "checkpoint": checkpoint, "model_cfg": selection["model_cfg"],
                         "threshold": selection["validation"]["threshold"], "test": result["test"],
                         "checkpoint_sha256": selection["checkpoint_sha256"],
                         "preprocessing_sha256": selection["preprocessing_sha256"],
                         "kind": "New validation-selected experiment"})
        break
    return specs


def load_model(spec, root: Path = ROOT):
    checkpoint = Path(spec["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError("Selected model checkpoint is missing.")
    if spec.get("checkpoint_sha256") and sha256(checkpoint) != spec["checkpoint_sha256"]:
        raise ValueError("Checkpoint integrity check failed.")
    if spec.get("preprocessing_sha256") and sha256(root/"data/processed/preprocessing_config.json") != spec["preprocessing_sha256"]:
        raise ValueError("Preprocessing does not match the selected experiment.")
    model = TCN(80, **spec["model_cfg"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    return model


@torch.inference_mode()
def score_patient(model, patient: PreparedPatient, threshold: float):
    scores = torch.sigmoid(model(torch.from_numpy(patient.inputs)[None]))[0].numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Inference returned invalid scores.")
    data = {"ICULOS": patient.hours, "Model score": scores,
            "Study threshold": np.full(len(scores), threshold),
            "Above study threshold": scores >= threshold,
            "Observed feature fraction": patient.observed_fraction}
    if patient.labels is not None:
        data["Supplied label"] = patient.labels
    return pd.DataFrame(data)
