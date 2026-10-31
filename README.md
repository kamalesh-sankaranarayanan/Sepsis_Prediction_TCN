# Missingness-Aware TCN: Sepsis Research Demo

A local, working research prototype for hourly SepsisLabel classification using each patient's first 72 available rows. Rows need not start at ICU hour 1. The proposed causal TCN uses 40 imputed/scaled clinical values plus 40 original-missingness indicators. The baseline TCN uses only the 40 clinical values. Neither uses the stored elapsed-time channels or temporal attention.

## Launch

One-command launcher (checks dependencies, model integrity, and real sample inference before starting):

```powershell
python -B -m src.demo
```

Check readiness without starting a server: `python -B -m src.demo --check-only`. To verify preserved original artifacts as well: `python -B -m src.check_demo --verify-originals`. The launcher performs no installation, training, or artifact writes. Use `--port 8502` if the default port is occupied. It works directly from PowerShell without changing script-execution policy.

From the repository root, using the current Python environment:

```powershell
python -B -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open the printed localhost URL. Upload one UTF-8 CSV or PhysioNet pipe-delimited PSV (maximum 2 MiB). All 40 feature columns are required; SepsisLabel is optional and never enters predictions. Input columns are reordered to the saved feature list. ICULOS must advance by one hour. Longer stays are visibly truncated to the first 72 rows. Invalid numeric feature values are treated as missing with a warning, matching training; malformed schemas and time axes are rejected.

The app performs real checkpoint inference and shows scores, a timeline, threshold comparisons, and optional reference labels. Patient uploads and scores are not saved. Aggregate evaluation metrics are shown separately from individual predictions. This is a research prototype, not a diagnostic system, calibrated probability estimator, or validated six-hour forecast.

Existing dependencies were sufficient; no packages were installed for this work. `requirements-demo.txt` lists minimal demo/experiment dependencies, and `requirements-dev.txt` adds pytest. `requirements.txt` includes optional legacy research tooling. On another machine, review and authorize installation before using pip; XGBoost is not needed for the UI.

## Checks

```powershell
python -B -m pytest -q -p no:cacheprovider tests
python -B -X utf8 src/validate_processed.py
```

Tests cover exact preprocessing against real processed patients, checkpoint inference, CSV column ordering, label exclusion, causality, 72-row truncation, invalid inputs, manifest enforcement, losses, and Streamlit interactions. Pytest requires a writable temporary directory. Structural validation checks all processed tensors but does not independently reproduce the complete historical raw-data audit.

## Isolated experiments

```powershell
python -B -X utf8 src/run_experiments.py --epochs 6 --patience 2 --threads 4 --max-estimated-seconds 1200
```

The runner creates a unique directory under `experiments/` and refuses existing output directories. It benchmarks a disposable model before bounded CPU training. Three same-seed candidates use unweighted BCE, corrected weighted BCE, and focal loss. The fixed historical proposed checkpoint is also a validation reference. Checkpoints/early stopping use validation AUPRC; thresholds maximize validation F1 over all distinct scores, including below 0.05. The final candidate is locked in `selection.json` before test data is loaded. Only that candidate receives new test evaluation.

Each run records configuration, versions, source and artifact hashes, logs, checkpoints, validation comparison, final selection, and test metrics/predictions. A completed run appears automatically as the UI default; otherwise the verified historical proposed model remains available. Original checkpoints, results, logs, figures, preprocessing outputs, and splits are preserved. Never use an incomplete experiment as the demo default.

`run_pipeline.py` defaults to isolated stage 8. Legacy training scripts require a new `--output-dir experiments/<new-name>`; their historical per-model test evaluation is unsuitable for validation-only model selection. Legacy audit, preprocessing, explanation, and visualization stages may write artifacts and should not be rerun merely to inspect existing results.

## Study and validation

- Fixed patient-level split: 28,235 train / 6,050 validation / 6,051 test, hospital-by-full-stay-label stratification, seed 42.
- Original sequence logs contain five epochs. The new experiment has its own epoch logs/configuration.
- Prior read-only auditing checked all 40,336 raw patients: membership, labels, original masks, clinical values, elapsed-time recurrence, and training-only means/scales matched. There is no patient-ID overlap across splits.
- LR and seven sequence checkpoints reproduced saved test metrics and validation thresholds; XGBoost remains unverified because its package is absent. The historical LR version mismatch is documented in the report.
- Sequence evaluation uses 222,297 test hours (1.377% positive); LR/XGBoost use 232,215 hours (1.810% positive), so cross-family scores have different populations.
- First-72-row truncation removes all positive labels from the retained prefixes of 508 train, 113 validation, and 111 test patients positive in their full stay.
- Historical sequence weighting used 54.6046 from uncapped labels; corrected runs use 71.8471 from valid capped training labels. Original artifacts retain their original behavior.
- New seeds are fixed per candidate. Exact training portability across environments, repeated-seed uncertainty, external validation, calibration, clinical value/sentinel checks, and original reference-paper claims remain unresolved. The historical test set has already been examined; new selection is validation-only, but this is not a fresh external test cohort.

See `research_report.md` for historical/new result distinctions, `PROJECT_CONTEXT_HANDOFF.md` for audit history and implementation updates, and `UI_PLAN.md` for the implemented UI contract and constraints.

## Latest measured test result

Validation selected corrected weighted BCE at epoch 6, threshold 0.7500504851. All rows below use the same capped test population. Only the final selected new candidate was newly evaluated on test.

| Model | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Historical unweighted ablation baseline | 97.38% | 8.69% | 9.54% | 9.10% | 0.7520 | 0.0527 |
| Historical proposed | 96.82% | 8.97% | 14.28% | 11.02% | 0.7726 | 0.0584 |
| Selected new weighted BCE | 96.78% | 9.84% | 16.41% | 12.30% | 0.7823 | 0.0633 |

New confusion counts: TN 214,639; FP 4,598; FN 2,558; TP 502. Precision, recall, F1, AUROC, and AUPRC improve over the historical proposed result; accuracy and specificity decrease. These are single-seed results, and the new protocol also changes the epoch budget, seed reset, checkpoint criterion, and threshold search. Do not attribute the gain to weighting alone.

See `experiments/tcn_losses_20261029T154725515099Z/REPORT.md` for validation comparisons and preservation evidence. All 21 tests passed. For a real local demonstration, upload an existing anonymized patient file such as `data/raw/training/training_setA/p000005.psv`; the app does not replace missing inputs with simulated predictions.
