
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(RAW_DIR.rglob("*.psv"))

total_patients = 0
patients_with_sepsis = 0
positive_hours = 0
total_hours = 0
bad_files = []

missing_values = None
column_names = None

for i, file in enumerate(files, start=1):
    try:
        df = pd.read_csv(file, sep="|")

        if "SepsisLabel" not in df.columns:
            bad_files.append((str(file), "Missing SepsisLabel"))
            continue

        if column_names is None:
            column_names = df.columns.tolist()
            missing_values = pd.Series(0, index=column_names, dtype="int64")

        if set(df.columns) != set(column_names):
            bad_files.append((str(file), "Inconsistent columns"))
            continue

        df = df[column_names]

        total_patients += 1
        total_hours += len(df)
        positive_hours += int(df["SepsisLabel"].sum())

        if df["SepsisLabel"].max() == 1:
            patients_with_sepsis += 1

        missing_values += df.isna().sum().astype("int64")

    except Exception as exc:
        bad_files.append((str(file), str(exc)))

    if i % 5000 == 0:
        print(f"Checked {i:,} / {len(files):,} files")

print("\n===== DATASET SUMMARY =====")
print("Files discovered:", len(files))
print("Valid patient files:", total_patients)
print("Patients with a positive sepsis label:", patients_with_sepsis)
print("Patients without a positive sepsis label:",
      total_patients - patients_with_sepsis)
print("Total hourly records:", total_hours)
print("Positive hourly labels:", positive_hours)
print("Unreadable or invalid files:", len(bad_files))

if missing_values is not None and total_hours:
    missing_summary = pd.DataFrame({
        "missing_count": missing_values,
        "missing_percent": (missing_values / total_hours * 100).round(2)
    }).sort_values("missing_percent", ascending=False)

    missing_summary.to_csv(
        RESULTS_DIR / "missing_summary.csv",
        index_label="variable"
    )

    print("\n===== TOP 15 MISSING VARIABLES =====")
    print(missing_summary.head(15).to_string())

if bad_files:
    pd.DataFrame(
        bad_files, columns=["file", "error"]
    ).to_csv(RESULTS_DIR / "invalid_files.csv", index=False)

print("\nSaved missing-value report to results/missing_summary.csv")
