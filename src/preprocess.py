
from pathlib import Path
import json

import numpy as np
import pandas as pd

from cohort import iter_official_patients

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
MANIFEST = OUT_DIR / "split_manifest.csv"
LABEL = "SepsisLabel"


def load_manifest():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing validated manifest: {MANIFEST}")

    df = pd.read_csv(MANIFEST)
    required = {"patient_id", "source_group", "sepsis_positive", "split"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Manifest columns are {list(df.columns)}; "
            f"expected at least {sorted(required)}. Stop without changing it."
        )

    if df["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient IDs in manifest.")
    if set(df["split"]) != {"train", "val", "test"}:
        raise ValueError("Manifest must contain train, val and test splits.")

    return df


def get_patient_map():
    result = {}
    for row in iter_official_patients():
        path = Path(row["path"])
        patient_id = path.stem
        if patient_id in result:
            raise ValueError(f"Duplicate official patient ID: {patient_id}")
        result[patient_id] = path
    return result


def read_patient(path):
    df = pd.read_csv(path, sep="|")
    if LABEL not in df.columns:
        raise ValueError(f"Missing {LABEL}: {path}")

    y = df[LABEL].to_numpy(dtype=np.float32)
    features = df.drop(columns=[LABEL]).apply(
        pd.to_numeric, errors="coerce"
    )
    values = features.to_numpy(dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    return list(features.columns), values, y


def main():
    manifest = load_manifest()
    paths = get_patient_map()

    if len(paths) != 40336:
        raise ValueError(f"Expected 40336 official patients, found {len(paths)}")
    if set(manifest["patient_id"]) != set(paths):
        raise ValueError("Manifest and official patient IDs do not match.")

    assignments = {
        row.patient_id: row.split
        for row in manifest.itertuples(index=False)
    }
    for split in ("train", "val", "test"):
        ids = manifest.loc[manifest["split"] == split, "patient_id"].tolist()
        print(f"{split}: {len(ids)} patients")

    train_ids = manifest.loc[
        manifest["split"] == "train", "patient_id"
    ].tolist()

    feature_names = None
    sums = counts = None

    # Fit feature means and standard deviations on training patients only.
    for i, pid in enumerate(train_ids, 1):
        names, values, _ = read_patient(paths[pid])
        if feature_names is None:
            feature_names = names
            sums = np.zeros(len(names), dtype=np.float64)
            counts = np.zeros(len(names), dtype=np.int64)
        elif names != feature_names:
            raise ValueError(f"Feature columns differ for patient {pid}")

        observed = np.isfinite(values)
        sums += np.where(observed, values, 0).sum(axis=0)
        counts += observed.sum(axis=0)

        if i % 2000 == 0:
            print(f"Fitting training statistics: {i}/{len(train_ids)}")

    means = np.divide(
        sums, counts, out=np.zeros_like(sums), where=counts > 0
    )

    # Second pass: forward-fill, then fill leading gaps with train means.
    # Calculate variance on observed training values after mean imputation.
    sum_sq = np.zeros(len(feature_names), dtype=np.float64)
    n_scaled = 0
    for pid in train_ids:
        _, values, _ = read_patient(paths[pid])
        observed = np.isfinite(values)
        filled = pd.DataFrame(values).ffill().to_numpy()
        filled = np.where(np.isfinite(filled), filled, means)
        sum_sq += ((filled - means) ** 2).sum(axis=0)
        n_scaled += len(filled)

    stds = np.sqrt(sum_sq / max(n_scaled, 1))
    stds[~np.isfinite(stds) | (stds < 1e-8)] = 1.0

    # Save preprocessing parameters.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "preprocessing_config.json", "w") as f:
        json.dump({
            "feature_names": feature_names,
            "training_means": means.tolist(),
            "training_stds": stds.tolist(),
            "elapsed_time_transform": "log1p(hours)",
            "elapsed_time_initialization": "1, 2, 3, ... hours before first observation",
            "imputation": "forward fill, then training mean for leading gaps",
            "scaling": "training mean and standard deviation only",
            "split_manifest": str(MANIFEST.relative_to(ROOT))
        }, f, indent=2)

    # Process every patient according to the existing manifest.
    for split in ("train", "val", "test"):
        split_dir = OUT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        ids = manifest.loc[manifest["split"] == split, "patient_id"].tolist()

        for i, pid in enumerate(ids, 1):
            names, values, y = read_patient(paths[pid])
            if names != feature_names:
                raise ValueError(f"Feature columns differ for patient {pid}")

            observed = np.isfinite(values)
            T, D = values.shape
            delta = np.zeros((T, D), dtype=np.float32)
            last_seen = np.full(D, -1, dtype=np.int64)

            for t in range(T):
                current = observed[t]
                delta[t] = np.where(
                    current, 0,
                    np.where(last_seen >= 0, t - last_seen, t + 1)
                )
                last_seen[current] = t

            filled = pd.DataFrame(values).ffill().to_numpy()
            filled = np.where(np.isfinite(filled), filled, means)
            scaled = ((filled - means) / stds).astype(np.float32)

            X = np.concatenate([
                scaled,
                observed.astype(np.float32),
                np.log1p(delta).astype(np.float32)
            ], axis=1)

            if X.shape != (len(y), 3 * D):
                raise ValueError(f"Invalid shape for patient {pid}")
            if not np.isfinite(X).all() or not np.isfinite(y).all():
                raise ValueError(f"Non-finite processed data for patient {pid}")
            if len(y) == 0 or not np.isin(y, [0, 1]).all():
                raise ValueError(f"Invalid labels for patient {pid}")

            np.savez_compressed(
                split_dir / f"{pid}.npz",
                X=X,
                y=y,
                observed=observed.astype(np.uint8)
            )

            if i % 2000 == 0:
                print(f"{split}: {i}/{len(ids)} patients processed")

        print(f"Saved {len(ids)} patients to {split_dir}")

    print("Preprocessing complete. Raw files were not modified.")


if __name__ == "__main__":
    main()
