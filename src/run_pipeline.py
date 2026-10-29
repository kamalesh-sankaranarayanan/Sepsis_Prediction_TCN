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
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY   = sys.executable


def run(cmd: list[str], check: bool = True) -> int:
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if check and result.returncode != 0:
        print(f"FAILED with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode


STAGES = {
    1: ("Data Inspection",        [PY, "src/data_inspection.py"]),
    2: ("Data-Quality Audit",     [PY, "src/data_quality_audit.py"]),
    3: ("Preprocessing",          [PY, "src/preprocessing.py"]),
    4: ("Train Baselines",        None),   # built dynamically
    5: ("Train Proposed TCN",     None),
    6: ("SHAP Explainability",    [PY, "src/explainability.py"]),
    7: ("Visualize Results",      [PY, "src/visualize_results.py"]),
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
            cmd = [PY, "src/train_baselines.py", "--models"] + args.models
            if args.epochs:
                cmd += ["--epochs", str(args.epochs)]

        elif stage_id == 5:
            cmd = [PY, "src/train_proposed_tcn.py"]
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
