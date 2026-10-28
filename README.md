# Time-Aware Missing Data Representation and Temporal Attention for TCN-Based Early Sepsis Prediction

PhysioNet/CinC Challenge 2019 data. Official patients are only:

- `data/raw/training/training_setA/` (20,336 files)
- `data/raw/training/training_setB/` (20,000 files)

Loose `.psv` files directly under `data/raw/training/` are **not** part of the cohort. Raw files are never modified.

## Stage 1 — splits only

This stage creates a patient-level 70/15/15 train/validation/test split. It does not write model tensors and does not train.

Stratification uses hospital (A or B) and whether the patient has any positive `SepsisLabel`. Seed is `42`.

### PowerShell (from the project folder)

```powershell
cd C:\Users\sanka\OneDrive\Desktop\SepsisPrediction
python -m pip install -r requirements.txt
python src/make_splits.py
python src/verify_dataset.py
```

Outputs:

- `data/processed/split_manifest.csv` — patient ID, source group, sepsis-positive flag, split, relative path
- `data/processed/split_config.json` — seed, fractions, and how the split was made

Later stages must fit preprocessing on the **train** split only and must not tune on the test split.
