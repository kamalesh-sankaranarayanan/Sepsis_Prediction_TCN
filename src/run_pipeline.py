"""
run_pipeline.py — Master orchestration script.

Runs the entire research pipeline in order:

  Stage 0: Install requirements
  Stage 1: Data inspection (Phase 1)
  Stage 2: Data-quality audit (Phase 1, Sections 2-8)
  Stage 3: Preprocessing (Phase 3) — requires raw .psv files
  Stage 4: Train baselines (Phase 4)
  Stage 5: Train proposed TCN / ablation (Phase 5 + 8)
  Stage 6: SHAP explainability (Phase 9)
  Stage 7: Visualize results (Phase 10)

Usage:
    python src/run_pipeline.py                      # full pipeline
    python src/run_pipeline.py --stages 1 2         # audit only
    python src/run_pipeline.py --stages 4 5         # training only
    python src/run_pipeline.py --epochs 5           # short training run
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


def find_best_python() -> list[str]:
    """Find a Python executable with numpy and torch installed."""
    # 1. Test current sys.executable
    try:
        res = subprocess.run(
            [sys.executable, "-c", "import numpy, torch"],
            capture_output=True,
            timeout=5,
        )
        if res.returncode == 0:
            return [sys.executable]
    except Exception:
        pass

    # 2. Known local Python 3.10 installation with all ML dependencies
    known_path = Path(r"C:\Users\haris\AppData\Local\Programs\Python\Python310\python.exe")
    if known_path.is_file():
        try:
            res = subprocess.run(
                [str(known_path), "-c", "import numpy, torch"],
                capture_output=True,
                timeout=5,
            )
            if res.returncode == 0:
                print(f"[INFO] Using ML Python environment: {known_path}")
                return [str(known_path)]
        except Exception:
            pass

    # 3. Windows py launcher for 3.10
    try:
        res = subprocess.run(
            ["py", "-3.10", "-c", "import numpy, torch"],
            capture_output=True,
            timeout=5,
        )
        if res.returncode == 0:
            print("[INFO] Using Python 3.10 via 'py -3.10'")
            return ["py", "-3.10"]
    except Exception:
        pass

    return [sys.executable]


PY_CMD = find_best_python()


def run(cmd: list[str], check: bool = True) -> int:
    print(f"\n>>> {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if check and result.returncode != 0:
        print(f"FAILED with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode


STAGES = {
    1: ("Data Inspection",        PY_CMD + ["-X", "utf8", "src/data_inspection.py"]),
    2: ("Data-Quality Audit",     PY_CMD + ["-X", "utf8", "src/data_quality_audit.py"]),
    3: ("Preprocessing",          PY_CMD + ["-X", "utf8", "src/preprocessing.py"]),
    4: ("Train Baselines",        None),   # built dynamically
    5: ("Train Proposed TCN",     None),
    6: ("SHAP Explainability",    PY_CMD + ["-X", "utf8", "src/explainability.py"]),
    7: ("Visualize Results",      PY_CMD + ["-X", "utf8", "src/visualize_results.py"]),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", type=int,
                        default=list(STAGES.keys()),
                        help="Which stages to run (default: all)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Max epochs for deep learning models")
    parser.add_argument("--models", nargs="+",
                        default=["lr", "xgb", "lstm", "gru", "tcn"],
                        help="Which baselines to train in stage 4")
    args = parser.parse_args()

    print("=" * 65)
    print("  Missingness-Aware TCN — Full Research Pipeline")
    print("=" * 65)

    for stage_id in sorted(args.stages):
        if stage_id not in STAGES:
            print(f"Unknown stage {stage_id} — skip.")
            continue

        name, cmd = STAGES[stage_id]
        print(f"\n{'='*65}")
        print(f"  STAGE {stage_id}: {name}")
        print(f"{'='*65}")

        if stage_id == 4:
            cmd = PY_CMD + ["-X", "utf8", "src/train_baselines.py", "--models"] + args.models
            if args.epochs:
                cmd += ["--epochs", str(args.epochs)]

        elif stage_id == 5:
            cmd = PY_CMD + ["-X", "utf8", "src/train_proposed_tcn.py"]
            if args.epochs:
                cmd += ["--epochs", str(args.epochs)]

        run(cmd)

    print("\n" + "="*65)
    print("  Pipeline complete.")
    print("  Results:  results/")
    print("  Figures:  figures/")
    print("  Models:   models/")
    print("  Logs:     logs/")
    print("="*65)


if __name__ == "__main__":
    main()
