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
import os
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def find_best_python() -> str:
    """Find a Python executable with numpy and pandas installed."""
    try:
        res = subprocess.run(
            [sys.executable, "-c", "import numpy, pandas"],
            capture_output=True,
            timeout=5,
        )
        if res.returncode == 0:
            return sys.executable
    except Exception:
        pass

    known_path = Path(r"C:\Users\haris\AppData\Local\Programs\Python\Python310\python.exe")
    if known_path.is_file():
        try:
            res = subprocess.run(
                [str(known_path), "-c", "import numpy, pandas"],
                capture_output=True,
                timeout=5,
            )
            if res.returncode == 0:
                return str(known_path)
        except Exception:
            pass

    return sys.executable


PY_EXE = find_best_python()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if .npz files exist")
    args = parser.parse_args()

    proc_train = ROOT / "data" / "processed" / "train"
    already_done = proc_train.is_dir() and any(proc_train.glob("*.npz"))

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    if already_done and not args.force:
        print("Processed .npz files already exist.")
        print("  → Running validate_processed.py to verify integrity …")
        result = subprocess.run(
            [PY_EXE, "-X", "utf8", str(ROOT / "src" / "validate_processed.py")],
            capture_output=True, text=True, env=env
        )
        print(result.stdout)
        if result.returncode != 0:
            print("VALIDATION FAILED:", result.stderr)
            sys.exit(1)
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
    result = subprocess.run(
        [PY_EXE, "-X", "utf8", str(ROOT / "src" / "preprocess.py")],
        capture_output=False, env=env
    )
    if result.returncode != 0:
        print("Preprocessing failed.")
        sys.exit(result.returncode)

    print("\nValidating processed files …")
    result2 = subprocess.run(
        [PY_EXE, "-X", "utf8", str(ROOT / "src" / "validate_processed.py")],
        capture_output=True, text=True, env=env
    )
    print(result2.stdout)
    if result2.returncode != 0:
        print("POST-PROCESSING VALIDATION FAILED:", result2.stderr)
        sys.exit(1)
    print("Preprocessing pipeline complete.")


if __name__ == "__main__":
    main()
