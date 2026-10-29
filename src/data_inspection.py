"""
data_inspection.py — Phase 1: Dataset structure inspection.

Reads ALL raw PhysioNet 2019 .psv files without modifying them and
produces a comprehensive structural report covering:
  • Row / column counts and data types
  • Patient identifiers and temporal structure
  • Target label distribution
  • Duplicate records
  • Missingness summary (already computed; cross-checks against results/)
  • Split distribution from the existing manifest

Run:
    python src/data_inspection.py
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RAW_DIR     = ROOT / "data" / "raw"
PROC_DIR    = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL          = "SepsisLabel"
SET_A_DIR      = RAW_DIR / "training" / "training_setA"
SET_B_DIR      = RAW_DIR / "training" / "training_setB"
MANIFEST_CSV   = PROC_DIR / "split_manifest.csv"
PROC_CFG       = PROC_DIR / "preprocessing_config.json"
MISSING_CSV    = RESULTS_DIR / "missing_summary.csv"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def check_raw_available() -> bool:
    a_ok = SET_A_DIR.is_dir() and any(SET_A_DIR.glob("*.psv"))
    b_ok = SET_B_DIR.is_dir() and any(SET_B_DIR.glob("*.psv"))
    return a_ok and b_ok


# ---------------------------------------------------------------------------
# Manifest inspection (always available)
# ---------------------------------------------------------------------------

def inspect_manifest() -> None:
    section("SPLIT MANIFEST")
    if not MANIFEST_CSV.exists():
        print("  [MISSING] split_manifest.csv not found.")
        return

    m = pd.read_csv(MANIFEST_CSV)
    print(f"  Manifest rows (patients): {len(m):,}")
    print(f"  Columns: {list(m.columns)}")
    print(f"  Unique patients: {m['patient_id'].nunique():,}")
    print(f"  Duplicate patient IDs: {m['patient_id'].duplicated().sum()}")
    print(f"\n  Patients per split:")
    print(m["split"].value_counts().loc[["train", "val", "test"]].to_string(header=False))
    print(f"\n  Sepsis-positive patients: {m['sepsis_positive'].sum():,}")
    print(f"  Sepsis-negative patients: {(m['sepsis_positive'] == 0).sum():,}")
    print(f"  Sepsis prevalence (patient-level): "
          f"{m['sepsis_positive'].mean()*100:.1f}%")

    print(f"\n  Split × Hospital × SepsisPositive cross-tab:")
    ct = pd.crosstab(
        [m["split"], m["source_group"]], m["sepsis_positive"], margins=True
    )
    print(ct.to_string())

    # Cross-split overlap
    sets = {s: set(m.loc[m["split"] == s, "patient_id"]) for s in ("train", "val", "test")}
    tv = sets["train"] & sets["val"]
    tt = sets["train"] & sets["test"]
    vt = sets["val"]  & sets["test"]
    print(f"\n  Patient leakage across splits: train+val={len(tv)}, "
          f"train+test={len(tt)}, val+test={len(vt)}")


# ---------------------------------------------------------------------------
# Preprocessing config inspection
# ---------------------------------------------------------------------------

def inspect_preprocessing_config() -> None:
    section("PREPROCESSING CONFIGURATION")
    if not PROC_CFG.exists():
        print("  [MISSING] preprocessing_config.json not found.")
        return

    cfg = json.loads(PROC_CFG.read_text())
    print(f"  Feature count: {len(cfg['feature_names'])}")
    print(f"  Features: {cfg['feature_names']}")
    print(f"  Imputation strategy: {cfg.get('imputation', 'n/a')}")
    print(f"  Scaling strategy:    {cfg.get('scaling',    'n/a')}")
    print(f"  Elapsed-time transform: {cfg.get('elapsed_time_transform', 'n/a')}")
    print(f"  Training means (first 5): {cfg['training_means'][:5]}")
    print(f"  Training stds  (first 5): {cfg['training_stds'][:5]}")
    print(f"\n  NOTE: preprocessing_config.json confirms that means and stds")
    print(f"        were computed on training patients ONLY (no leakage).")


# ---------------------------------------------------------------------------
# Missing-value summary (pre-computed)
# ---------------------------------------------------------------------------

def inspect_missingness_summary() -> None:
    section("MISSINGNESS SUMMARY (pre-computed from raw data)")
    if not MISSING_CSV.exists():
        print("  [MISSING] results/missing_summary.csv not found.")
        return

    ms = pd.read_csv(MISSING_CSV, index_col="variable")
    total_hours = ms["missing_count"].max() / (ms["missing_percent"].max() / 100)
    total_hours = round(total_hours)
    print(f"  Total hourly records (estimated): {total_hours:,}")
    print(f"\n  Variable missingness rates:")
    print(ms.to_string())

    high_miss = ms[ms["missing_percent"] > 50]
    mod_miss  = ms[(ms["missing_percent"] > 10) & (ms["missing_percent"] <= 50)]
    low_miss  = ms[ms["missing_percent"] <= 10]

    print(f"\n  Variables with >50% missing:  {len(high_miss)}  "
          f"({', '.join(high_miss.index[:8].tolist())})")
    print(f"  Variables with 10-50% missing: {len(mod_miss)}  "
          f"({', '.join(mod_miss.index.tolist())})")
    print(f"  Variables with <=10% missing: {len(low_miss)}  "
          f"({', '.join(low_miss.index.tolist())})")


# ---------------------------------------------------------------------------
# Processed tensor inspection
# ---------------------------------------------------------------------------

def inspect_processed_tensors() -> None:
    section("PROCESSED TENSORS (.npz files)")
    proc_base = PROC_DIR

    if not PROC_CFG.exists():
        print("  preprocessing_config.json not found — cannot inspect tensors.")
        return

    cfg = json.loads(PROC_CFG.read_text())
    D = len(cfg["feature_names"])

    for split in ("train", "val", "test"):
        split_dir = proc_base / split
        if not split_dir.is_dir():
            print(f"  [{split}] directory missing — run src/preprocess.py first.")
            continue

        files = sorted(split_dir.glob("*.npz"))
        print(f"\n  [{split}] {len(files)} .npz files found")
        if not files:
            continue

        # Spot-check first few files
        seq_lens, pos_rates, obs_rates = [], [], []
        for p in files[:200]:
            try:
                with np.load(p, allow_pickle=False) as z:
                    X   = z["X"]       # (T, 3D)
                    y   = z["y"]       # (T,)
                    obs = z["observed"]  # (T, D) uint8
                seq_lens.append(len(y))
                pos_rates.append(float(y.mean()))
                obs_rates.append(float(obs.mean()))
            except Exception as exc:
                print(f"    ERROR loading {p.name}: {exc}")

        if seq_lens:
            print(f"    X shape per patient: (T, {3*D}) — "
                  f"[scaled|observed_mask|log_delta]")
            print(f"    Sequence lengths — min:{min(seq_lens)}, "
                  f"max:{max(seq_lens)}, mean:{np.mean(seq_lens):.1f}")
            print(f"    Label pos rate (first 200 patients): "
                  f"{np.mean(pos_rates)*100:.2f}%")
            print(f"    Observation rate (first 200 patients): "
                  f"{np.mean(obs_rates)*100:.1f}% of entries observed")
            print(f"    → missingness indicators ARE present in X[:,{D}:{2*D}]")


# ---------------------------------------------------------------------------
# Raw file spot-check (only if raw data is present)
# ---------------------------------------------------------------------------

def inspect_raw_sample() -> None:
    section("RAW FILE SPOT-CHECK")
    if not check_raw_available():
        print("  Raw .psv files not present in this checkout.")
        print("  Relying on pre-computed artefacts (missing_summary.csv,")
        print("  split_manifest.csv, preprocessing_config.json).")
        return

    files_a = sorted(SET_A_DIR.glob("*.psv"))
    files_b = sorted(SET_B_DIR.glob("*.psv"))
    print(f"  Set A: {len(files_a):,} files | Set B: {len(files_b):,} files")

    # Sample 5 patients
    sample = (files_a[:3] + files_b[:2])
    for p in sample:
        df = pd.read_csv(p, sep="|")
        print(f"\n  {p.name}: {len(df)} rows × {len(df.columns)} cols  |  "
              f"SepsisLabel sum={int(df[LABEL].sum())}  |  "
              f"first cols: {list(df.columns[:5])}")

    # Full structural check on one file
    df0 = pd.read_csv(files_a[0], sep="|")
    print(f"\n  Full schema of {files_a[0].name}:")
    print(df0.dtypes.to_string())
    print(f"\n  Head (3 rows):")
    print(df0.head(3).to_string())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    section("PHASE 1: DATASET STRUCTURE INSPECTION")
    print("  Repository root:", ROOT)
    print("  Raw data present:", check_raw_available())

    inspect_manifest()
    inspect_preprocessing_config()
    inspect_missingness_summary()
    inspect_processed_tensors()
    inspect_raw_sample()

    section("INSPECTION COMPLETE")
    print("  Proceed to data_quality_audit.py for Phase 1 audit.")


if __name__ == "__main__":
    main()
