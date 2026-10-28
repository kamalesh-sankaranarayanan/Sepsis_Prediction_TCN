"""Official PhysioNet 2019 cohort: training_setA and training_setB only.

Loose .psv files under data/raw/training/ are ignored. Raw files are never
modified.
"""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TRAINING_DIR = RAW_DIR / "training"
SET_A_DIR = TRAINING_DIR / "training_setA"
SET_B_DIR = TRAINING_DIR / "training_setB"

EXPECTED_COUNTS = {"A": 20336, "B": 20000}
PATIENT_ID_PATTERN = re.compile(r"^p\d+$")
LABEL = "SepsisLabel"

SOURCE_DIRS = {
    "A": SET_A_DIR,
    "B": SET_B_DIR,
}


def list_official_files(source_group: str) -> list[Path]:
    directory = SOURCE_DIRS[source_group]
    if not directory.is_dir():
        raise FileNotFoundError(f"Official folder missing: {directory}")
    return sorted(directory.glob("*.psv"))


def iter_official_patients() -> list[dict]:
    """Return one record per official patient file (paths only, no labels)."""
    rows = []
    for source_group in ("A", "B"):
        files = list_official_files(source_group)
        expected = EXPECTED_COUNTS[source_group]
        if len(files) != expected:
            raise ValueError(
                f"Set {source_group} has {len(files)} .psv files, "
                f"expected {expected} in {SOURCE_DIRS[source_group]}"
            )
        for path in files:
            patient_id = path.stem
            rows.append(
                {
                    "patient_id": patient_id,
                    "source_group": source_group,
                    "relative_path": path.relative_to(RAW_DIR).as_posix(),
                    "path": path,
                }
            )
    return rows


def unexpected_patient_ids(rows: list[dict]) -> list[str]:
    unexpected = []
    for row in rows:
        patient_id = row["patient_id"]
        if not PATIENT_ID_PATTERN.match(patient_id):
            unexpected.append(patient_id)
            continue
        if row["source_group"] == "A" and not patient_id.startswith("p0"):
            unexpected.append(patient_id)
        elif row["source_group"] == "B" and not patient_id.startswith("p1"):
            unexpected.append(patient_id)
    return unexpected


def duplicate_patient_ids(rows: list[dict]) -> list[str]:
    seen = {}
    duplicates = []
    for row in rows:
        patient_id = row["patient_id"]
        if patient_id in seen:
            duplicates.append(patient_id)
        else:
            seen[patient_id] = row["source_group"]
    return sorted(set(duplicates))
