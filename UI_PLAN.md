# UI Implementation and Demonstration Contract

Status: implemented locally with Streamlit in `app.py`; inference is in `src/inference.py`. The later master task authorized implementation and bounded new first-72-row experiments, superseding the earlier planning-only restriction. No packages were installed and no deployment is performed.

## Workflow

1. Launch `python -B -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false`.
2. Select the completed validation-selected experiment (default when present) or the verified historical proposed TCN.
3. Upload one UTF-8 CSV or pipe-delimited PSV, at most 2 MiB, with the exact 40 feature names. SepsisLabel is optional and excluded from model inputs.
4. Review retained/excluded rows, observation coverage, and invalid/missing-value warnings. Only the first 72 rows are scored. Recorded ICULOS starts above 1 are accepted; nonconsecutive/invalid hours are rejected.
5. Generate real model scores. Inspect the timeline, saved threshold, hourly table, and optional supplied-label overlay. Invalid uploads, missing checkpoints, and inference errors never trigger fake predictions.
6. Use the separate Research evaluation tab for aggregate test metrics, confusion counts, method, and limitations. These metrics do not describe the uploaded patient.

## Exact inference behavior

Column order follows preprocessing_config.json. Original finite-value masks are captured before imputation. Invalid numeric values/infinities become missing with an explicit warning. Within-patient forward fill and saved training means handle gaps; scaling uses the exact saved means/scales. Model inputs are clinical values concatenated with `1 - observed`. Elapsed-time channels are not inputs to these checkpoints.

The existing TCN architecture is loaded with saved weights, evaluated on CPU without gradients, and produces one sigmoid score per retained row. New experiment metadata supplies its architecture, threshold, checkpoint hash, preprocessing hash, and measured metrics. Only completed runs are selectable; incomplete runs do not replace the historical default.

Uploads and predictions remain in memory, with no application persistence, patient filenames in charts, external services, or automatic exports. Immutable models are cached. No training or plotting pipeline is imported by inference.

## Verified checks

Pytest covers real raw-to-processed prefix equivalence, checkpoint predictions, optional-label exclusion, column reordering, causal behavior, leading missingness, longer stays, invalid schemas/time axes, numerical errors, manifest enforcement, and Streamlit startup/prediction/error paths. Local HTTP root and health endpoints returned 200. No visual browser screenshot review was performed; automated Streamlit interactions provide functional coverage.

## Scope and constraints

This is a first-72-available-row research demonstration, not a diagnosis, calibrated clinical probability, or validated six-hour forecast. Thresholds are fixed by validation, not tuned using the uploaded patient. Single-patient AUROC is not displayed. Single-hour SHAP panels are omitted because they do not explain temporal predictions correctly.

The original model's historical metrics remain distinct from new experiments. XGBoost verification, exact training portability, repeated-seed uncertainty, clinical validity/calibration, and reference verification remain research tasks. All original data, splits, checkpoints, logs, figures, and result files are preserved.
