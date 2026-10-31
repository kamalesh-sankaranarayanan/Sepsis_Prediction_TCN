"""
run_pipeline.py — Master orchestration script.

Defaults to isolated validation-selected experiments (stage 8).
Stages 1-7 describe legacy artifact-writing workflows:

  Stage 0: Install requirements
  Stage 1: Data inspection (Phase 1)
  Stage 2: Data-quality audit (Phase 1, Sections 2-8)
  Stage 3: Preprocessing (Phase 3) — requires raw .psv files
  Stage 4: Train baselines (Phase 4)
  Stage 5: Train proposed TCN / ablation (Phase 5 + 8)
  Stage 6: SHAP explainability (Phase 9)
  Stage 7: Visualize results (Phase 10)

Usage:
    python src/run_pipeline.py                      # isolated experiments
    python src/run_pipeline.py --stages 1 2         # audit only
    python src/run_pipeline.py --stages 8 --epochs 6 # bounded training run
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
    8: ("Isolated validation-selected TCN experiments", PY_CMD + ["-B", "-X", "utf8", "src/run_experiments.py"]),
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
                        default=[8],
                        help="Default: safe isolated experiments (8). Stages 1-7 are legacy artifact writers.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Max epochs for deep learning models")
    args = parser.parse_args()
    if any(stage in args.stages for stage in (4, 5)):
        parser.error("Legacy training stages 4/5 evaluate every model on test. Use stage 8 for validation-only selection; direct legacy scripts require a new --output-dir.")

    print("=" * 65)
    print("  Missingness-Aware TCN — Full Research Pipeline")
    print("=" * 65)

    for stage_id in sorted(args.stages):
        if stage_id not in STAGES:
            print(f"Unknown stage {stage_id} — skip.")
            continue

        name, cmd = STAGES[stage_id]
        if stage_id == 8 and args.epochs:
            cmd = cmd + ["--epochs", str(args.epochs)]
        print(f"\n{'='*65}")
        print(f"  STAGE {stage_id}: {name}")
        print(f"{'='*65}")

        run(cmd)

    print("\n" + "="*65)
    print("  Pipeline complete.")
    print("  Stage 8 artifacts: experiments/<unique-run>/")
    print("  Legacy stages may write results/, figures/, models/, logs/.")
    print("="*65)


if __name__ == "__main__":
    main()
