"""Create reproducible patient-level train/val/test splits for Stage 1.

Reads official set A/B files only. Does not write tensors or train models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cohort import (
    LABEL,
    RAW_DIR,
    ROOT,
    duplicate_patient_ids,
    iter_official_patients,
    unexpected_patient_ids,
)

SEED = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
OUT_DIR = ROOT / "data" / "processed"
MANIFEST_CSV = OUT_DIR / "split_manifest.csv"
CONFIG_JSON = OUT_DIR / "split_config.json"


def read_sepsis_flag(path: Path) -> int:
    df = pd.read_csv(path, sep="|", usecols=[LABEL])
    if df[LABEL].isna().any():
        raise ValueError(f"Missing {LABEL} values in {path}")
    unique = set(df[LABEL].unique().tolist())
    if not unique.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Invalid {LABEL} values {unique} in {path}")
    return int(df[LABEL].eq(1).any())


def build_cohort_table() -> pd.DataFrame:
    rows = iter_official_patients()

    unexpected = unexpected_patient_ids(rows)
    if unexpected:
        raise ValueError(f"Unexpected patient IDs: {unexpected[:20]}")

    duplicates = duplicate_patient_ids(rows)
    if duplicates:
        raise ValueError(f"Duplicate patient IDs: {duplicates[:20]}")

    records = []
    for i, row in enumerate(rows, start=1):
        sepsis_positive = read_sepsis_flag(row["path"])
        records.append(
            {
                "patient_id": row["patient_id"],
                "source_group": row["source_group"],
                "sepsis_positive": sepsis_positive,
                "relative_path": row["relative_path"],
            }
        )
        if i % 5000 == 0:
            print(f"Read labels: {i}/{len(rows)}")

    return pd.DataFrame.from_records(records)


def assign_splits(cohort: pd.DataFrame) -> pd.DataFrame:
    if abs(TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1")

    strata = (
        cohort["source_group"].astype(str)
        + "_"
        + cohort["sepsis_positive"].astype(str)
    )
    indices = np.arange(len(cohort))
    rest_fraction = VAL_FRACTION + TEST_FRACTION

    train_idx, rest_idx = train_test_split(
        indices,
        test_size=rest_fraction,
        random_state=SEED,
        stratify=strata,
    )
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=TEST_FRACTION / rest_fraction,
        random_state=SEED,
        stratify=strata.iloc[rest_idx],
    )

    assigned = cohort.copy()
    split = np.empty(len(assigned), dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    assigned["split"] = split
    return assigned


def print_crosstab(manifest: pd.DataFrame) -> None:
    counts = (
        manifest.groupby(["split", "source_group", "sepsis_positive"])
        .size()
        .unstack(fill_value=0)
    )
    print("\nPatients by split, hospital, and sepsis-positive flag:")
    print(counts.to_string())
    print("\nPatients per split:")
    print(manifest["split"].value_counts().loc[["train", "val", "test"]].to_string())


def validate_manifest(manifest: pd.DataFrame) -> None:
    if manifest["patient_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate patient IDs")

    split_ids = {
        name: set(manifest.loc[manifest["split"] == name, "patient_id"])
        for name in ("train", "val", "test")
    }
    if split_ids["train"] & split_ids["val"]:
        raise ValueError("Train/val patient overlap")
    if split_ids["train"] & split_ids["test"]:
        raise ValueError("Train/test patient overlap")
    if split_ids["val"] & split_ids["test"]:
        raise ValueError("Val/test patient overlap")

    if len(manifest) != 40336:
        raise ValueError(f"Expected 40336 patients, found {len(manifest)}")

    missing_split = manifest["split"].eq("").sum()
    if missing_split:
        raise ValueError(f"{missing_split} patients have no split")


def main() -> None:
    print("Building official cohort from training_setA and training_setB only")
    print("Ignoring any .psv files directly under data/raw/training/")

    cohort = build_cohort_table()
    print("Official patients:", len(cohort))
    print("Set A:", int((cohort["source_group"] == "A").sum()))
    print("Set B:", int((cohort["source_group"] == "B").sum()))
    print("Septic patients:", int(cohort["sepsis_positive"].sum()))

    manifest = assign_splits(cohort)
    validate_manifest(manifest)
    print_crosstab(manifest)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_columns = [
        "patient_id",
        "source_group",
        "sepsis_positive",
        "split",
        "relative_path",
    ]
    manifest[csv_columns].to_csv(MANIFEST_CSV, index=False)

    config = {
        "seed": SEED,
        "label_column": LABEL,
        "train_fraction": TRAIN_FRACTION,
        "val_fraction": VAL_FRACTION,
        "test_fraction": TEST_FRACTION,
        "stratify_by": ["source_group", "sepsis_positive"],
        "unit": "patient",
        "raw_root": "data/raw",
        "set_a_dir": "data/raw/training/training_setA",
        "set_b_dir": "data/raw/training/training_setB",
        "excluded": "data/raw/training/*.psv (not in training_setA or training_setB)",
        "n_patients": int(len(manifest)),
        "n_set_a": int((manifest["source_group"] == "A").sum()),
        "n_set_b": int((manifest["source_group"] == "B").sum()),
        "sklearn_train_test_split": {
            "first_call": "test_size=0.30, random_state=42, stratify=source_group+sepsis_positive",
            "second_call": "split remaining 50/50 into val and test, same seed and strata",
        },
        "notes": [
            "Each patient appears in exactly one split.",
            "All hours from a patient stay in that patient's split.",
            "Preprocessing statistics in later stages must be fit on train only.",
            "The test split must not be used to choose model settings.",
        ],
    }
    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\nWrote", MANIFEST_CSV.relative_to(ROOT).as_posix())
    print("Wrote", CONFIG_JSON.relative_to(ROOT).as_posix())
    print("Stage 1 splits complete. No tensors were written.")


if __name__ == "__main__":
    main()
