# Project Handoff & Full Context: Missingness-Aware TCN for Early Sepsis Prediction

> **Purpose:** This document provides the complete, self-contained context of this project, codebase, experiments, empirical results, resolved technical issues, and next steps. You can provide this file to any new AI chat session to seamlessly resume work without loss of context.

> **Current audit update (2026-10-29):** See Section 9 for verified local findings. Paths, Python environments, and Git/remote details below describe the historical workspace and must not be assumed current. The current workspace is `C:\Users\sanka\OneDrive\Desktop\SepsisPrediction`, on branch `harish`.

---

## 1. Project Overview & Research Objective

* **Project Title:** Missingness-Aware Temporal Convolutional Network (MA-TCN) for Early Sepsis Prediction
* **Domain:** Medical AI, Clinical Time-Series Modeling, Electronic Health Records (EHR)
* **Repository Location:** `D:\Final Year Project\Sepsis_Prediction_TCN`
* **Active Working Branch:** `harish`
* **Upstream Git Origin:** `https://github.com/kamalesh-sankaranarayanan/Sepsis_Prediction_TCN.git`
* **Local User Identity:** `Harish3051` (`23z114@psgitech.ac.in`)

### Core Research Hypothesis
In ICU electronic health records, diagnostic tests (e.g., blood gas, lactate, coagulation panels) are **not missing at random (MNAR)**. Clinicians order tests because they suspect deterioration; stable patients are tested far less frequently. 

While conventional models treat missingness as a nuisance to be imputed away, our proposed model (**MA-TCN**) explicitly preserves and feeds the **binary observation indicators** ($M_{i,t} \in \{0, 1\}$) alongside the imputed clinical measurements into a causal, dilated Temporal Convolutional Network.

### Reference Paper Target
* **Reference Work:** Sankaranarayanan (2025) — TCN with SMOTE and weighted loss for early sepsis prediction on PhysioNet 2019.
* **Published Targets:** ~88% Accuracy, ~0.92 AUROC.
* **Key Baseline Comparisons:** Logistic Regression (LR), XGBoost, LSTM, GRU, Baseline TCN.
* **Implemented Imbalance Strategies:** Class-weighted LR, XGBoost scale_pos_weight, and weighted BCE; the ablation also includes unweighted BCE. SMOTE is not implemented.
* **Interpretability:** SHAP feature importance.

---

## 2. Dataset & Cohort Specifications

* **Source:** PhysioNet / Computing in Cardiology (CinC) Challenge 2019 (*Early Prediction of Sepsis from Clinical Data*).
* **Official Cohort:** 40,336 patients (~1.55 million hourly records).
  * `training_setA`: 20,336 patients (Hospital System A)
  * `training_setB`: 20,000 patients (Hospital System B)
* **Raw File Format:** Pipe-delimited `.psv` files per patient.
* **Split Configuration:** Fixed 70% / 15% / 15% patient-level split (stratified by hospital and sepsis status, `seed=42`).
  * **Train:** 28,235 patients (`data/processed/train/`)
  * **Validation:** 6,050 patients (`data/processed/val/`)
  * **Test:** 6,051 patients (`data/processed/test/`)
  * **Split Leakage:** 0 overlap across all splits (`data/processed/split_manifest.csv`).
* **Target Label (`SepsisLabel`):**
  * Sepsis-3 clinical criteria.
  * Raw labels are copied unchanged. The claimed six-hour lead time and clinical onset definition have not been independently verified.
  * **Patient-level prevalence:** 7.3% (2,932 positive, 37,404 negative).
  * **Overall retrospective hourly prevalence:** **1.798%** positive hours (27,916 / 1,552,210); overall negative-to-positive ratio 54.60:1.
* **Feature Dimensionality ($D = 40$ features):**
  * **8 Vital signs:** HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2
  * **26 Laboratory values:** BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets
  * **6 Demographics / ICU:** Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS
* **Extreme Missingness Profile:**
  * 26 of 40 features are missing >90% of hours (e.g., Bilirubin_direct 99.81%, Fibrinogen 99.34%, TroponinI 99.05%, Lactate 97.32%).
* **Processed Tensor Representation (`.npz` files):**
  * `X`: $(T, 3D)$ matrix containing:
    * Columns $0 \dots D-1$: Standardized, forward-filled clinical values (leading gaps filled with train mean).
    * Columns $D \dots 2D-1$: Binary observation mask ($1 = \text{observed}, 0 = \text{imputed}$).
    * Columns $2D \dots 3D-1$: Log elapsed time since last observation $\log(1 + \Delta t)$.
  * `y`: $(T,)$ binary ground truth label per hour.
  * `observed`: $(T, D)$ uint8 observation mask.

---

## 3. Codebase Architecture

```
Sepsis_Prediction_TCN/
├── data/
│   ├── raw/training/                  # 40,336 raw .psv files in training_setA/ & training_setB/
│   └── processed/                     # split_manifest.csv, configs, and per-split .npz files
├── figures/                           # All generated publication charts and SHAP beeswarm plots
├── logs/                              # Epoch-by-epoch training logs (.json)
├── models/                            # Checkpoints (.pt, .json, .pkl)
├── results/                           # Evaluation metrics (.json) and audit reports
├── src/
│   ├── config.py                      # Master hyperparameters, paths, seeds
│   ├── cohort.py                      # Official patient listing and validation
│   ├── verify_dataset.py              # Cohort and split integrity test
│   ├── preprocess.py                  # Low-level tensor builder
│   ├── preprocessing.py               # Preprocessing orchestrator & validator
│   ├── sequence_builder.py            # PyTorch Datasets (40-dim & 80-dim), flat_loader, pos_weight
│   ├── models.py                      # Causal Dilated Residual TCN, LSTM, GRU
│   ├── train_baselines.py             # LR, XGBoost, LSTM, GRU, Baseline TCN training
│   ├── train_proposed_tcn.py          # 4-condition ablation study runner
│   ├── evaluate.py                    # Evaluation metrics, ROC/PR curves, threshold tuning
│   ├── explainability.py              # SHAP TreeExplainer & KernelExplainer
│   ├── visualize_results.py           # Publication visual generation
│   └── run_pipeline.py                # Master CLI with auto-detecting Python resolver
├── research_report.md                 # Full 14-section academic research paper
└── requirements.txt                   # Complete dependencies
```

---

## 4. Empirical Results & Findings

### A. Baseline Comparison Table (Test Set: 6,051 patients, ~225,000 test hours)
Retrospective evaluation: LR/XGBoost use all 232,215 test hours (1.810% positive). Sequence models use each patient's first 72 available rows, totaling 222,297 hours (1.377% positive). Cross-family metrics therefore use different populations. XGBoost metrics remain saved results rather than independently reproduced results in the current audit.

| Model | Accuracy | Precision | Recall | F1-Score | AUROC | AUPRC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 95.31% | 11.89% | 24.82% | 16.08% | 0.7487 | 0.0756 |
| **XGBoost** | 95.84% | 13.71% | 24.48% | 17.58% | **0.8084** | **0.0976** |
| **LSTM** | 97.23% | 9.51% | 11.86% | 10.56% | 0.7382 | 0.0569 |
| **GRU** | 95.68% | 7.02% | 17.48% | 10.02% | 0.7415 | 0.0504 |
| **TCN (Baseline)** | 94.34% | 6.34% | 22.58% | 9.90% | 0.7553 | 0.0483 |
| **Reference Paper** `[REF]` | **~88%** | — | — | — | **~0.92** | — |

### B. Controlled Ablation Study: Validating the Proposed MA-TCN
Isolating the explicit effect of missingness indicators under identical conditions:

| Experiment | Miss. Indicators | Pos. Weight | AUROC | AUPRC | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TCN_Baseline** | No | No | 0.7520 | 0.0527 | 9.54% | 9.10% |
| **TCN_Baseline_PosWeight** | No | Yes | 0.7342 | 0.0441 | 12.78% | 9.45% |
| **TCN_Proposed (Our Contribution)**| **Yes** | **No** | **0.7726** | **0.0584** | **14.28%** | **11.02%** |
| **TCN_Proposed_PosWeight** | **Yes** | **Yes** | **0.7705** | **0.0540** | **17.58%** | **10.73%** |

#### Measured Controlled Improvement (Proposed vs Baseline, same pos_weight=No):
* **$\Delta$AUROC:** $+0.0206$ (**$+2.06\%$ absolute gain**)
* **$\Delta$AUPRC:** $+0.0057$ (**$+10.8\%$ relative gain**)
* **$\Delta$Recall:** $+4.74\%$ (**$+49.7\%$ relative gain: from 9.54% to 14.28%**)
* **$\Delta$F1:** $+1.92\%$ (**$+21.1\%$ relative gain: from 9.10% to 11.02%**)
* **Recall with Positive Weighting:** Climbs to **17.58%** (critical for early clinical warning).

### C. SHAP Explainability Findings
* **XGBoost Top Features:** ICULOS (0.5255), Temperature (0.2355), HospAdmTime (0.2292), FiO2 (0.1888), Lactate (0.1859), Heart Rate (0.1641), Unit1 (0.1563), BUN (0.1245), WBC (0.1163), Creatinine (0.1111).
* **Proposed TCN Top Features:** ICULOS, BUN, Temperature, Unit2, Unit1, SaO2, Heart Rate, **`Resp_miss`**, pH, Creatinine.
* **Attribution limitation:** The prior run reported `Resp_miss` in the top eight; this ranking was not independently reproduced. TCN SHAP uses isolated single-hour inputs without sequence history and does not prove MNAR or clinical causality.

---

## 5. Understanding the "88% Accuracy vs 0.92 AUROC" Question

* **Why did our models get ~96% accuracy with 0.77 AUROC, while the paper reported 88% accuracy with 0.92 AUROC?**
  * **Class Imbalance:** Always-negative accuracy is 98.190% on the full test stream and 98.623% on the capped sequence test stream. Accuracy alone cannot determine false-positive rate.
  * **Reference uncertainty:** The reference scores and evaluation protocol require original-source verification. Do not infer balanced test evaluation or attribute its AUROC to SMOTE without checking the source.
  * **Evaluation scope:** Current results are retrospective hourly classification, not prospective deployment validation. Clinical lead time and official challenge utility have not been evaluated in this implementation.

---

## 6. Technical Issues Solved in this Workspace

1. **Python Environment Mismatch:**
   * PowerShell was defaulting to `C:\Users\haris\.hermes\hermes-agent\venv\Scripts\python.exe` (missing ML libraries).
   * All ML packages (`torch`, `xgboost`, `scikit-learn`, `shap`, `numpy`, `pandas`) are in `Python 3.10` (`C:\Users\haris\AppData\Local\Programs\Python\Python310\python.exe` or `py -3.10`).
   * **Fix:** Implemented `find_best_python()` in [`src/run_pipeline.py`](file:///d:/Final%20Year%20Project/Sepsis_Prediction_TCN/src/run_pipeline.py) and [`src/preprocessing.py`](file:///d:/Final%20Year%20Project/Sepsis_Prediction_TCN/src/preprocessing.py) to automatically locate and bind the ML environment regardless of shell default.
2. **Windows Unicode Console Encoding (`cp1252` Crash):**
   * Printing `\u2264` or em-dashes caused `UnicodeEncodeError: 'charmap' codec can't encode character`.
   * **Fix:** Added automatic UTF-8 stream reconfiguration and injected `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` in subprocess runners.
3. **Raw Dataset Directory Nesting:**
   * Extraction placed files inside `training_setA/training/` and `training_setB/training_setB/`.
   * **Fix:** Flattened directories so all 20,336 files are directly in `training_setA/` and 20,000 in `training_setB/`. Validated via `src/verify_dataset.py` (Passed).
4. **SHAP Array Slice Mismatch:**
   * KernelExplainer fallback evaluated 50 samples, while `summary_plot` was passed 200 samples (`AssertionError: Feature and SHAP matrices must have the same number of rows!`).
   * **Fix:** Synchronized feature matrix slicing: `X_plot = X_te[te_idx[:len(sv)]]`.

---

## 7. Git & Remote Status

* **Branch:** `harish`
* **Local Commits Ahead of Origin:**
  ```
  38ab0b0 docs: populate research report with empirical metrics and ablation findings
  ee9cc0b feat(results): save trained model checkpoints, training logs, and evaluation metrics
  5ab3033 feat(visuals): generate complete publication figures and training curves
  8543020 fix(explainability): correct sample slice alignment in SHAP summary plot
  33e6973 fix(preprocessing): resolve duplicate subprocess call and ensure clean syntax
  7954b51 fix(preprocess): auto-resolve ML Python and enforce UTF-8 in preprocessing runner
  31a0a31 fix(pipeline): auto-detect and select ML Python environment when invoked from bare venv
  e1f5a16 docs: add research report and fix Windows UTF-8 console encoding in pipeline runner
  b99d0e2 feat: complete missingness-aware TCN research pipeline
  ```
* **Git Push Error 403:**
  Direct push to `origin` (`kamalesh-sankaranarayanan/Sepsis_Prediction_TCN.git`) failed because user `Harish3051` needs write access.
  * **Solution:** Fork the repository on GitHub to `Harish3051/Sepsis_Prediction_TCN`, update remote URL, and push:
    ```powershell
    git remote set-url origin https://github.com/Harish3051/Sepsis_Prediction_TCN.git
    git push -u origin harish
    ```

---

## 8. Next Steps & Recommended Actions

1. **Push branch to GitHub:** Fork repo on GitHub and push the `harish` branch.
2. **Implement SMOTE Experiment:**
   * First verify the original reference protocol. Any later SMOTE experiment must preserve patient separation and leave the held-out evaluation cohort unchanged; no AUROC improvement is guaranteed.
3. **Incorporate Derived Clinical Vitals:**
   * Add Shock Index ($\text{HR} / \text{SBP}$), Pulse Pressure ($\text{SBP} - \text{DBP}$), and 6-hour moving vital rates of change ($\Delta\text{HR}/\Delta t$).
4. **Thesis / Paper Finalization:**
   * `research_report.md` has 14 drafted sections. Reference verification, uncertainty analysis, and the documented limitations remain before submission.

---

## 9. Verified Local Audit Status (2026-10-29)

**Accepted scope:** Retain and finalize the first-72-available-row sequence study. Preserve checkpoints and metrics; do not initiate full-stay retraining. The later master task authorizes a local Streamlit UI and bounded new first-72-row experiments while preserving historical artifacts. The UI is now implemented; no dependency installation or deployment was performed. See the implementation update below.

* All 40,336 raw patient IDs exactly match the manifest and processed split files, with no cross-split patient-ID overlap.
* Raw-to-processed labels, original observation masks, scaled clinical values, and elapsed-time channels match for every patient. Saved training means and scales match a training-only recalculation exactly.
* Total hours: 1,552,210 (train 1,085,958; validation 234,037; test 232,215). Positive hours: 27,916. No label reversions were found.
* ICULOS advances by one hour per row in every patient; 8,793 stays start above hour 1. Leading missingness ages start at one from the first available row.
* Scaling uses observed training means and RMS deviations of filled values from those means, not conventional standard deviations around filled-data means.
* Every saved sequence-training log contains five epochs; 50 is the configured maximum. Original training invocation and hardware are not established by these logs.
* Logistic Regression and seven sequence checkpoints reproduced saved test classification metrics to four decimal places and confusion-matrix counts exactly using saved thresholds. LR emitted a scikit-learn version warning (saved 1.6.1; audit 1.7.2).
* XGBoost remains unverified because its package is absent from registered Python environments. No packages were installed.
* All seven sequence checkpoints reproduced their best logged validation AUROC and saved thresholds using the original validation F1 grid search (6,050 patients; 223,640 capped hours). LR threshold 0.78 was also reproduced on 234,037 uncapped validation hours (F1 0.1557; AUROC 0.7484). Training reproduction remains unverified. Single-run gains need uncertainty analysis.
* Remaining implementation limitations: loaders do not enforce manifest membership at load time; STRIDE is unused; training hardcodes the first 72 rows; class weights count uncapped hours; threshold search is limited to 0.05 through 0.95, with both unweighted ablation thresholds at 0.05.
* Clinical value ranges, sentinel zeros, source-paper claims, clinical lead time, and SHAP rankings remain unverified. A local Streamlit research UI now exists; there is no deployed clinical inference API.
* The raw-data and checkpoint audits changed no artifacts. Subsequent documentation corrections affect only this handoff and the research report.
* Quantified cap effect: the first-72-row policy excludes 5,255 / 1,154 / 1,143 positive hours in train / validation / test. It removes all positive labels from the retained prefixes of 508 / 113 / 111 otherwise-positive patients. Weighted sequence training uses ratio 54.6046 from all training hours, versus 71.8471 in retained training labels (approximately 24.0% lower). No weights or models were changed; the performance effect of changing this policy remains untested.



## 10. Master Task Implementation Update (2026-10-29)

The later master task authorized bounded new first-72-row experiments and a working local UI. This supersedes earlier planning-only restrictions, while preserving original data, splits, and historical experiment artifacts.

* New shared study utilities enforce manifest membership and valid-timestep masking. Weights now count retained training hours; new thresholds cover all validation scores; legacy specificity calculation was corrected. Legacy training resets seeds, uses the configured cap, and requires a fresh experiments/ output directory. Pipeline stage 8 is the safe default.
* Completed run: `experiments/tcn_losses_20261029T154725515099Z`. Three losses each ran six epochs. Weighted BCE was selected using validation AUPRC 0.064274; its epoch-6 checkpoint and threshold 0.7500504851 were locked before test evaluation.
* Selected test scores: accuracy 96.78%, precision 9.84%, recall 16.41%, F1 12.30%, AUROC 0.7823, AUPRC 0.0633. Counts: TN 214639, FP 4598, FN 2558, TP 502. Precision, recall, F1, AUROC, and AUPRC improve over the historical proposed run; accuracy and specificity decrease. Changes versus historical training are not limited to the class weight, so do not claim a causal weighting-only improvement.
* Exact configs, environment versions, source snapshots, logs, checkpoints, validation comparisons, final selection, test predictions, and REPORT.md are retained inside the isolated run. All 56 protected original artifact hashes match. Raw/processed patient files were only read.
* Implemented `app.py` and `src/inference.py`: local CSV/PSV upload, exact preprocessing, real checkpoint scoring, hourly charts, aggregate comparison, clear limitations, and graceful invalid-input/missing-model errors. Patient data is not persisted. The completed selected candidate is the default; historical proposed remains available.
* Final checks: 21 pytest tests passed; HTTP root and health endpoints returned 200; legacy overwrite guards passed; no visual browser review. No packages installed, no deployment, no Git commits/pushes/merges/deletions.
* Commands: `python -B -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false`; `python -B -m pytest -q -p no:cacheprovider tests`.
* Remaining: XGBoost package absent, single-seed uncertainty, exact training portability, external validation/calibration, clinical value/sentinel checks, and verification of reference-paper and lead-time claims. This remains a research demonstration.


## 11. Performance-Only V2 Follow-up (2026-10-29)

* Actual architecture remains a causal missingness-aware TCN with 80 clinical/missingness inputs, without temporal attention or elapsed-time channels. No UI changes were made in this task.
* Artifact directories: `experiments_v2/tcn_refine_20261029T170738771536Z` and `experiments_v2/independent_seeds_20261029T171824288587Z`, each with REPORT.md. All prior experiment outputs were preserved.
* Reverified saved loss checkpoints on validation, manifest/statistics hashes, fixed populations, and capped-label weight 71.847075. Reused prior six-epoch BCE/weighted-BCE/focal evidence.
* Three new common-warm-start refinements (control, lower LR, stronger decay) each stopped after two epochs. Best trained validation AUPRC: 0.061305 / 0.062443 / 0.061822, versus incumbent 0.064274. No refinement won; epoch-zero incumbent was retained.
* New validation-only operating protocol: F1 maximum with FPR <=3%, minimum precision/recall/accuracy and AUROC guard; multi-metric selection score emphasizes AUPRC/F1. Full alternative operating points are saved. This is not a validated clinical threshold policy.
* Final checkpoint and threshold are unchanged: V1 weighted seed42, threshold 0.7500504851. Test scores remain accuracy 96.78%, precision 9.84%, recall 16.41%, F1 12.30%, AUROC 0.7823, AUPRC 0.0633. No additional V2 improvement; score arrays exactly match V1. No losing or replication checkpoint was evaluated on test.
* Independently trained the weighted six-epoch configuration from scratch for seeds43/44, selecting epochs6/5. These replications never change the locked final seed. Across42/43/44, validation AUPRC mean +/- SD = 0.063062 +/- 0.001258; F1 = 0.123952 +/- 0.005433; AUROC = 0.788273 +/- 0.003589.
* Added 200-draw paired patient bootstrap: test F1 CI [0.099689,0.143636], AUROC [0.758182,0.805250], AUPRC [0.050005,0.077730]. Paired improvement intervals versus the original proposed model include zero. Do not describe the earlier gains as statistically conclusive.
* Alarm burden: hourly FPR2.097%; 685/5722 (11.97%) negative-label prefixes have at least one threshold crossing. Prefix-level crossings are not clinical alarm episodes.
* Added `src/experiment_protocol_v2.py`, `src/run_experiments_v2.py`, `src/repeat_selected_seeds.py`, and `tests/test_experiments_v2.py`. Updated only report/handoff documentation among existing sources for this round. All29 tests pass; all79 captured original artifact/UI hashes and38 first-phase snapshot hashes match. Raw/processed patient files were only read. No installs, dependency changes, resets, merges, commits, or pushes.
* Commands: `python -B -m src.run_experiments_v2 --epochs 3 --patience 2 --repeat-seeds 43 44 --bootstrap 200 --threads 4 --max-estimated-seconds 1500`; `python -B -m src.repeat_selected_seeds --seeds 43 44 --threads 4 --max-estimated-seconds 1000`. Reproduce the original winning loss matrix with `python -B -X utf8 src/run_experiments.py --epochs 6 --patience 2 --seed 42 --threads 4 --max-estimated-seconds 1200`.
* Next research step: preregister a larger training/architecture study and obtain genuinely new external data. Do not keep tuning against the historically examined test cohort. UI work remains outside this performance-only task.
