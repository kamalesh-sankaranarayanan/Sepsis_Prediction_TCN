"""Validate the Stage 1 official cohort and saved patient-level splits."""

from __future__ import annotations

import json

import pandas as pd

from cohort import (
    EXPECTED_COUNTS,
    RAW_DIR,
    ROOT,
    duplicate_patient_ids,
    iter_official_patients,
    list_official_files,
    unexpected_patient_ids,
)

MANIFEST_CSV = ROOT / "data" / "processed" / "split_manifest.csv"
CONFIG_JSON = ROOT / "data" / "processed" / "split_config.json"


def main() -> None:
    print("=== Official folder counts ===")
    for group in ("A", "B"):
        n = len(list_official_files(group))
        expected = EXPECTED_COUNTS[group]
        status = "OK" if n == expected else "MISMATCH"
        print(f"Set {group}: {n} files (expected {expected}) [{status}]")

    loose = sorted((RAW_DIR / "training").glob("*.psv"))
    print(f"Loose files under data/raw/training/ (excluded): {len(loose)}")

    rows = iter_official_patients()
    unexpected = unexpected_patient_ids(rows)
    duplicates = duplicate_patient_ids(rows)
    print("Official unique patients:", len(rows))
    print("Unexpected IDs:", unexpected if unexpected else "none")
    print("Duplicate IDs:", duplicates if duplicates else "none")

    if not MANIFEST_CSV.exists() or not CONFIG_JSON.exists():
        raise FileNotFoundError(
            "Split files missing. Run: python src/make_splits.py"
        )

    manifest = pd.read_csv(MANIFEST_CSV)
    with open(CONFIG_JSON, encoding="utf-8") as f:
        config = json.load(f)

    print("\n=== Split config ===")
    print("seed:", config["seed"])
    print("fractions:", config["train_fraction"], config["val_fraction"], config["test_fraction"])
    print("stratify_by:", config["stratify_by"])

    required = {"patient_id", "source_group", "sepsis_positive", "split"}
    missing_cols = required - set(manifest.columns)
    if missing_cols:
        raise ValueError(f"Manifest missing columns: {missing_cols}")

    official_ids = {row["patient_id"] for row in rows}
    manifest_ids = set(manifest["patient_id"])

    print("\n=== Coverage ===")
    print("Manifest rows:", len(manifest))
    print("IDs in official folders but not manifest:", sorted(official_ids - manifest_ids)[:10] or "none")
    print("IDs in manifest but not official folders:", sorted(manifest_ids - official_ids)[:10] or "none")
    print("Duplicate IDs in manifest:", int(manifest["patient_id"].duplicated().sum()))

    split_sets = {
        name: set(manifest.loc[manifest["split"] == name, "patient_id"])
        for name in ("train", "val", "test")
    }
    print("\n=== Overlaps ===")
    print("train and val overlap:", len(split_sets["train"] & split_sets["val"]))
    print("train and test overlap:", len(split_sets["train"] & split_sets["test"]))
    print("val and test overlap:", len(split_sets["val"] & split_sets["test"]))

    print("\n=== Patients per split ===")
    print(manifest["split"].value_counts().loc[["train", "val", "test"]].to_string())

    print("\n=== Split x hospital x sepsis_positive ===")
    table = pd.crosstab(
        [manifest["split"], manifest["source_group"]],
        manifest["sepsis_positive"],
        margins=True,
    )
    print(table.to_string())

    ok = (
        len(rows) == 40336
        and not unexpected
        and not duplicates
        and official_ids == manifest_ids
        and not manifest["patient_id"].duplicated().any()
        and not (split_sets["train"] & split_sets["val"])
        and not (split_sets["train"] & split_sets["test"])
        and not (split_sets["val"] & split_sets["test"])
        and set(manifest["split"].unique()) == {"train", "val", "test"}
    )
    print("\nVALIDATION:", "PASSED" if ok else "FAILED")


if __name__ == "__main__":
    main()
