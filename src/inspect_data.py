
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

files = sorted(RAW_DIR.rglob("*.psv"))

print("Total patient files:", len(files))

if not files:
    raise FileNotFoundError(f"No .psv files found in {RAW_DIR}")

# Inspect a few files without loading the entire dataset
for file in files[:5]:
    try:
        df = pd.read_csv(file, sep="|")
        print(f"\nFile: {file.name}")
        print("Shape:", df.shape)
        print("Has SepsisLabel:", "SepsisLabel" in df.columns)
        print("Hourly records:", len(df))

        if "SepsisLabel" in df.columns:
            print("Positive sepsis hours:", int(df["SepsisLabel"].sum()))

    except Exception as exc:
        print(f"ERROR reading {file}: {exc}")
