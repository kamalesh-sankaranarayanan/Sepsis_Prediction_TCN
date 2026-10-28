
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "processed"
manifest = pd.read_csv(BASE / "split_manifest.csv")

required = {"patient_id", "split"}
if not required.issubset(manifest.columns):
    sys.exit(f"FAIL: manifest columns are {list(manifest.columns)}")

if manifest["patient_id"].duplicated().any():
    sys.exit("FAIL: duplicate patient IDs in manifest")

config = json.loads((BASE / "preprocessing_config.json").read_text())
D = len(config["feature_names"])

expected_counts = {"train": 28235, "val": 6050, "test": 6051}
total = 0
errors = []

for split, expected_count in expected_counts.items():
    ids = set(manifest.loc[manifest["split"] == split, "patient_id"].astype(str))
    if len(ids) != expected_count:
        errors.append(f"{split}: expected {expected_count} IDs, found {len(ids)}")

    folder = BASE / split
    files = {p.stem: p for p in folder.glob("*.npz")}

    if set(files) != ids:
        errors.append(
            f"{split}: missing {len(ids - set(files))}, "
            f"extra {len(set(files) - ids)} files"
        )

    checked = 0
    for pid in sorted(ids & set(files)):
        try:
            with np.load(files[pid], allow_pickle=False) as z:
                X, y, obs = z["X"], z["y"], z["observed"]

                if X.ndim != 2 or X.shape[1] != 3 * D:
                    raise ValueError(f"Wrong X shape: {X.shape}")
                if len(y) != len(X) or obs.shape != (len(y), D):
                    raise ValueError("X, y and observation mask shapes differ")
                if not np.isfinite(X).all() or not np.isfinite(y).all():
                    raise ValueError("Non-finite values")
                if not np.isin(y, [0, 1]).all():
                    raise ValueError("Labels are not binary")
                if not np.isin(obs, [0, 1]).all():
                    raise ValueError("Observation mask is not binary")
                if not np.array_equal(X[:, D:2*D], obs.astype(np.float32)):
                    raise ValueError("Observation mask does not match X")

                checked += 1
                total += 1
        except Exception as exc:
            errors.append(f"{pid}: {exc}")

    print(f"{split}: {checked}/{expected_count} files validated")

print(f"Total validated: {total}")

if errors:
    print("\nVALIDATION FAILED:")
    for error in errors[:30]:
        print("-", error)
    sys.exit(1)

print("\nPROCESSED DATA VALIDATION: PASSED")
