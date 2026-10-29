"""
preprocessing.py — Phase 3: Preprocessing pipeline runner.

This script re-runs the full preprocessing pipeline on the raw PhysioNet 2019
.psv files, producing per-patient .npz tensors under data/processed/{split}/.

Each .npz contains:
  X        : (T, 3D) float32  [scaled_clinical | observed_mask | log_delta]
  y        : (T,)    float32  binary SepsisLabel
  observed : (T, D)  uint8    1=originally observed, 0=imputed

The observed mask is the key missingness indicator for Phase 5.

This wraps the original preprocess.py with additional logging and verification.
If .npz files already exist, the script skips re-processing unless --force is given.

Run:
    python src/preprocessing.py
    python src/preprocessing.py --force    # re-process even if files exist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if .npz files exist")
    args = parser.parse_args()

    proc_train = ROOT / "data" / "processed" / "train"
    already_done = proc_train.is_dir() and any(proc_train.glob("*.npz"))

    if already_done and not args.force:
        print("Processed .npz files already exist.")
        print("  → Running validate_processed.py to verify integrity …")
        import subprocess, sys as _sys
        result = subprocess.run(
            [_sys.executable, str(ROOT / "src" / "validate_processed.py")],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print("VALIDATION FAILED:", result.stderr)
            _sys.exit(1)
        print("All processed files validated successfully.")
        print("Run with --force to re-process.")
        return

    raw_dir = ROOT / "data" / "raw" / "training"
    if not raw_dir.is_dir():
        print("ERROR: Raw data directory not found:", raw_dir)
        print("  Please place the PhysioNet CinC 2019 raw data under:")
        print("    data/raw/training/training_setA/")
        print("    data/raw/training/training_setB/")
        sys.exit(1)

    print("Running preprocessing pipeline …")
    import importlib.util, subprocess, sys as _sys
    result = subprocess.run(
        [_sys.executable, str(ROOT / "src" / "preprocess.py")],
        capture_output=False
    )
    if result.returncode != 0:
        print("Preprocessing failed.")
        _sys.exit(result.returncode)

    print("\nValidating processed files …")
    result2 = subprocess.run(
        [_sys.executable, str(ROOT / "src" / "validate_processed.py")],
        capture_output=True, text=True
    )
    print(result2.stdout)
    if result2.returncode != 0:
        print("POST-PROCESSING VALIDATION FAILED:", result2.stderr)
        _sys.exit(1)
    print("Preprocessing pipeline complete.")


if __name__ == "__main__":
    main()
