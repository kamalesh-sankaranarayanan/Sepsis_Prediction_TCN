# Missingness-Aware Temporal Convolutional Network for Early Sepsis Prediction

## Research Report

> **Retained study scope:** Finalize the existing first-72-available-row sequence study. Existing checkpoints and metrics remain unchanged; no full-stay retraining is underway. Rows need not start at ICU hour 1. LR/XGBoost use full-stay flat inputs and a different evaluation population. The local Streamlit research demo is implemented in `app.py`; see `UI_PLAN.md` for its contract. The later master task authorizes isolated first-72-row experiments while preserving historical artifacts.

> **Audit status (2026-10-29):** All 40,336 local raw patient files were compared with processed files without changing artifacts. Labels, original observation masks, scaled clinical values, and elapsed-time features matched. Saved training means and scales matched a training-only recalculation exactly. Logistic Regression and seven sequence checkpoints reproduced saved test classification metrics to four decimal places and confusion-matrix counts exactly using saved thresholds. XGBoost remains unverified because its package is absent from the available environments. For all seven sequence checkpoints, validation AUROC matched the best logged AUROC and the saved best-validation metric; the original validation F1 grid search reproduced every saved threshold across 6,050 patients and 223,640 capped validation hours. This verifies evaluation and sequence threshold selection, not training reproduction. LR validation-threshold selection was also reproduced: threshold 0.78 on 6,050 patients and 234,037 uncapped validation hours (validation F1 0.1557; AUROC 0.7484).

---

## Abstract

Sepsis is a life-threatening clinical syndrome requiring early identification for effective intervention. Existing machine learning approaches for sepsis prediction often discard missingness information despite its clinical informativeness — the absence of a laboratory measurement is rarely random and can signal patient acuity. This report presents a **Missingness-Aware Temporal Convolutional Network (MA-TCN)** that incorporates binary missingness indicators alongside imputed clinical measurements as a controlled improvement over a baseline TCN. The study uses the **PhysioNet Computing in Cardiology Challenge 2019** dataset (40,336 ICU patients, ~1.55 million hourly records). The proposed approach is evaluated against Logistic Regression, XGBoost, LSTM, GRU, and a baseline TCN under identical patient-level train/validation/test splits. An ablation study isolates the effect of missingness indicators from the effect of positive-class weighting. SHAP analysis identifies clinically plausible important features. The reference paper reports approximately **88% accuracy** and **0.92 AUROC**; these are treated as published targets, not guaranteed minima.

> **Note:** Results in this report marked `[REFERENCE]` are from the original paper. Results marked `[REPRODUCED]` are from this implementation run. Results marked `[PENDING]` will be populated after model training completes.

---

## 1. Introduction

Sepsis accounts for approximately 1 in 3 hospital deaths in the United States, and early identification improves survival outcomes significantly. ICU monitoring generates continuous physiological time-series with irregular measurement intervals, severe missingness, and pronounced class imbalance — all of which challenge standard classification approaches.

**Temporal Convolutional Networks (TCNs)** offer causal, parallelisable sequence modelling with a large effective receptive field through dilated convolutions. Unlike RNNs, TCNs avoid vanishing-gradient problems during training and achieve competitive or superior performance on clinical time-series.

This study extends the reference TCN architecture with **explicit missingness indicators** — a feature engineering approach grounded in the missing-not-at-random (MNAR) hypothesis for clinical data: the decision not to order a laboratory test encodes clinical information.

---

## 2. Related Work

- **Che et al. (2018)** — GRU-D: gated recurrent unit with decay mechanisms and missingness masks for irregular multivariate time-series.
- **Harutyunyan et al. (2019)** — Benchmark study on MIMIC-III for ICU prediction tasks including in-hospital mortality and IHM prediction.
- **Rajpurkar et al. (2017)** — CheXNet and deep learning for clinical prediction from high-dimensional data.
- **Reference paper (Sankaranarayanan 2025)** — TCN-based sepsis prediction on PhysioNet 2019 with SMOTE and weighted loss, reporting ~88% accuracy and 0.92 AUROC.
- **Shukla & Marlin (2021)** — Multi-Time Attention Networks for Irregularly Sampled Time Series (mTAN).
- **SHAP (Lundberg & Lee, 2017)** — Unified framework for explaining machine learning model outputs.

---

## 3. Problem Statement

**Task:** Given a patient's ICU time-series up to hour $t$, classify the supplied hourly `SepsisLabel ∈ {0, 1}` using current and past measurements. The implementation does not construct a separate future-onset target.

**Challenges:**
- Severe class imbalance: approximately 7.3% patient-level sepsis prevalence and 1.798% overall hour-level positive rate
- Extreme feature missingness: 26 of 40 features are missing >90% of observations
- Temporal structure: each patient has an irregular number of ICU hours
- Early-warning requirement: the label at time $t$ must not use features observed after $t$

---

## 4. Research Gap

Existing TCN-based sepsis models treat missing values as a nuisance to be imputed away. However:

1. **Missingness is informative**: a physician who does not order a lactate test has implicitly decided the patient appears stable; conversely, ordering a Troponin suggests cardiac concern.
2. **Forward-fill imputation silently erases information**: the model cannot distinguish a genuinely stable value from an old carried-forward measurement.
3. **The PhysioNet preprocessing pipeline already computes per-feature observation masks** — these can be used directly as missingness indicators without fabrication.

**Our contribution:** Formally evaluate whether concatenating binary missingness indicators to imputed clinical features improves TCN sepsis prediction, using a controlled ablation that isolates this effect.

---

## 5. Dataset and Data-Quality Audit

### 5.1 Dataset

| Property | Value |
|---|---|
| Source | PhysioNet/CinC Challenge 2019 |
| Patients | 40,336 (Set A: 20,336 + Set B: 20,000) |
| Time unit | 1-hour ICU stays |
| Total rows | 1,552,210 hourly records, verified against local raw files |
| Features | 40 (34 clinical measurements + 6 demographics/admin) |
| Target | `SepsisLabel` ∈ {0, 1} per hour |
| Patient ID format | `p000001` … `p100000` |
| Time axis | `ICULOS` (ICU Length of Stay in hours, 1-indexed) |

### 5.2 Missing Value Audit

| Category | Variables | Missing Rate |
|---|---|---|
| Extremely sparse lab values | Bilirubin_direct, Fibrinogen, TroponinI, Bilirubin_total, Alkalinephos, AST, Lactate, PTT, SaO2, EtCO2, Phosphate, HCO3, Chloride, BaseExcess, PaCO2, Calcium, Platelets, Creatinine, Magnesium, WBC, BUN, pH, Hgb, FiO2, Hct, Potassium | 90.7% – 99.8% |
| Moderately missing vitals | Glucose, Temp | 66–83% |
| Lightly missing vitals | Unit1, Unit2, DBP, Resp, SBP, O2Sat, MAP | 12–40% |
| Well-observed | HR | 9.9% |
| Complete | Age, Gender, ICULOS, SepsisLabel | 0% |

**Missingness hypothesis:** Clinical ordering decisions may make missingness informative. The audit and SHAP analysis do not establish the formal MNAR mechanism or clinical causality.

### 5.3 Data-Quality Findings

| Check | Result | Severity |
|---|---|---|
| Exact duplicate rows | Previously reported as 0; not independently rechecked in this audit | — |
| Duplicate patient IDs | 0 | — |
| Cross-split patient leakage | 0 patients overlap | — |
| SepsisLabel values | Only {0, 1} | — |
| Label reversions (1→0) | 0 patients in full raw scan | LOW |
| Scaling before splitting | NO: training-only statistics independently recalculated from raw files | — |
| Imputation before splitting | NO — per-patient forward fill | — |
| Target-derived features | None detected | — |
| Extreme lab missingness | 26 variables >90% | MODERATE |
| Sentinel zeros (vital signs) | Pending raw scan | LOW |

### 5.4 Split Configuration

| Split | Patients | Stratification |
|---|---|---|
| Train | 28,235 (70%) | Source group × Sepsis positive |
| Validation | 6,050 (15%) | " |
| Test | 6,051 (15%) | " |

Seed: 42. Each patient appears in exactly one split. Preprocessing statistics (means, stds) are computed on the training split only.

### 5.5 Target Label Audit

- **Encoding:** 0 = non-sepsis, 1 = sepsis (binary, per hour)
- **Patient-level prevalence:** 7.3% (2,932 / 40,336 sepsis-positive patients)
- **Hour-level positive rate:** 1.798% overall (27,916 / 1,552,210 hours); 1.810% in the full test stream and 1.377% in the capped sequence test stream
- **Imbalance ratio:** 54.60:1 overall (negative:positive); loss weighting uses a separately calculated ratio from all training hours
- **Label handling:** Raw SepsisLabel is copied unchanged. No future-onset target or six-hour shift is constructed. No 1-to-0 reversions were found. Clinical onset timing and the claimed lead time were not independently verified.

### 5.6 Temporal Integrity

- Time axis: `ICULOS` advances by exactly one hour per row for all patients; 8,793 stays start above hour 1.
- All measurements within a patient are hourly (regular intervals after admission)
- `HospAdmTime` records time of hospital admission relative to ICU admission (≤ 0 hours)
- **No temporal leakage**: causal TCN convolutions ensure each prediction uses only past and current observations
- **No imputation leakage**: forward-fill is applied within each patient in time order; training means fill only leading gaps (no cross-patient information)

### 5.7 Missingness Indicator Availability

> **Case A/B: Missingness indicators ARE available.**
>
> The `preprocess.py` script saves `X[:,D:2D]` as the binary observation mask (1=originally observed, 0=imputed). This directly corresponds to the required binary missingness indicator (1-mask = missing indicator). No fabrication of indicators is necessary.
>
> - **Baseline TCN input:** `X[:, 0:D]` — 40 scaled clinical features
> - **Proposed TCN input:** `X[:, 0:D] ∥ miss_indicator` — 80 features (40 clinical + 40 missingness)

### 5.8 Dataset Classification

> ✅ **READY FOR MODELING** — with the following noted limitations:
> - 26 lab features have >90% missingness; models must rely on missingness indicators and the small observed fraction
> - Overall hourly imbalance is 54.60:1; evaluation populations have different prevalence.
> - Local raw files were available for a correspondence audit of all 40,336 patients; labels, masks, values, deltas, and saved training statistics matched.

---

## 6. Proposed Methodology

### 6.1 Feature Engineering

For each patient with $T$ hourly records and $D=40$ clinical features:

**Step 1 — Missingness mask:** Before any imputation, record `observed[t,d] = 1` if feature $d$ was measured at hour $t$, else 0.

**Step 2 — Imputation:** Forward-fill within each patient time series; fill leading gaps with training-set means. Applied per-patient in temporal order to prevent leakage.

**Step 3 — Scaling:** Center on observed training-value means and divide by the RMS deviation of filled training values from those means. This is a training-only scale, not conventional standard deviation around the filled-data mean.

**Step 4 — Time-since-last-observation:** Compute `delta[t,d]` = hours since feature $d$ was last observed (0 if currently observed). Apply `log1p` transform to reduce skew.

Before the first observation, ages are 1, 2, 3, ... from the first available row. Every saved delta channel matched this recurrence reconstructed from raw missingness.

**Step 5 — Input construction:**
- Baseline: `X = [scaled_clinical]` ∈ ℝ^(T×D)
- Proposed: `X = [scaled_clinical | miss_indicator]` ∈ ℝ^(T×2D)

### 6.2 TCN Architecture

```
Input (T, C)
    ↓
Residual Block 1 — 64 channels, kernel=3, dilation=1
    ↓
Residual Block 2 — 64 channels, kernel=3, dilation=2
    ↓
Residual Block 3 — 64 channels, kernel=3, dilation=4
    ↓
Residual Block 4 — 64 channels, kernel=3, dilation=8
    ↓
Linear(64 → 1) → logit per time step
    ↓
Sigmoid → P(sepsis at time t)
```

Each Residual Block:
- CausalConv1d → LayerNorm → ReLU → Dropout(0.2) ×2
- 1×1 skip connection for channel matching
- Causal padding ensures no future information leaks

**Parameters:** Baseline TCN ≈ 97,921 | Proposed TCN ≈ 108,161

### 6.3 Class Imbalance Strategy

- **Primary:** `BCEWithLogitsLoss(pos_weight = neg_count / pos_count)` using all training hours, although sequence training uses only each patient's first 72 rows. The ablation also includes unweighted BCE.
- **Threshold tuning:** Optimal threshold selected by F1-score on the validation set
- **Note on SMOTE:** No SMOTE experiment is implemented. Flattened-row SMOTE does not preserve patient sequence structure. Weighted BCE changes loss contributions and is not equivalent to synthetic oversampling.

---

## 7. Historical Experimental Setup

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Sequence length cap | First 72 available hourly rows per patient; no sliding windows |
| Batch size | 64 |
| Optimizer | Adam (lr=1e-3, wd=1e-4) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Epochs | Configured maximum 50; every saved sequence-training log contains 5 epochs |
| Early stopping | 8 epochs on val AUROC |
| Gradient clipping | 1.0 |
| Device | Code selects CUDA if available, otherwise CPU; original training hardware unverified; no MPS selection |

---

## 8. Baseline Comparison

| Model | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|
| Logistic Regression | 95.31% | 11.89% | 24.82% | 16.08% | 0.7487 | 0.0756 |
| XGBoost | 95.84% | 13.71% | 24.48% | 17.58% | 0.8084 | 0.0976 |
| LSTM | 97.23% | 9.51% | 11.86% | 10.56% | 0.7382 | 0.0569 |
| GRU | 95.68% | 7.02% | 17.48% | 10.02% | 0.7415 | 0.0504 |
| TCN (Baseline) | 94.34% | 6.34% | 22.58% | 9.90% | 0.7553 | 0.0483 |
| **Reference Paper** `[REFERENCE]` | **~88%** | — | — | — | **~0.92** | — |

> **Evaluation scope:** This is retrospective hourly evaluation. LR/XGBoost use all 232,215 test hours (4,203 positive; 1.810%); sequence models use 222,297 hours from the first 72 rows per patient (3,060 positive; 1.377%). Cross-family comparisons therefore use different populations. Reference-paper scores and the evaluation protocol remain unverified against the original source. XGBoost metrics are saved results, not independently reproduced in this audit.

---

## 9. Ablation Study

| Experiment | Miss. Indicators | Pos. Weight | AUROC | AUPRC | Recall | F1 |
|---|---|---|---|---|---|---|
| TCN_Baseline | No | No | 0.7520 | 0.0527 | 9.54% | 9.10% |
| TCN_Baseline_PosWeight | No | Yes | 0.7342 | 0.0441 | 12.78% | 9.45% |
| TCN_Proposed | Yes | No | 0.7726 | 0.0584 | 14.28% | 11.02% |
| TCN_Proposed_PosWeight | Yes | Yes | 0.7705 | 0.0540 | 17.58% | 10.73% |

**Controlled Improvement (Proposed vs Baseline, same pos_weight=No):**
- **ΔAUROC:** +0.0206 (+2.06% absolute improvement)
- **ΔAUPRC:** +0.0057 (+10.8% relative gain)
- **ΔRecall:** +4.74% (+49.7% relative gain: from 9.54% to 14.28%)
- **ΔF1:** +1.92% (+21.1% relative gain: from 9.10% to 11.02%)

---

## 10. Results and Discussion

1. **Impact of Missingness Indicators:**
   - Incorporating binary missingness indicator variables consistently boosted the performance of the Temporal Convolutional Network.
   - Sepsis prediction benefits substantially from knowing whether a test was performed: ICU clinicians do not order diagnostic workups at random; the act of ordering an arterial blood gas or lactate test encodes high clinical suspicion.
2. **Impact of Loss Weighting:**
   - Weighted BCE uses the ratio calculated from all training hours. The proposed weighted run achieved 17.58% recall at its separately validation-selected threshold. This does not establish clinical utility or isolate a causal weighting effect.
3. **Reconciling the Reference Paper's 88% Accuracy / 0.92 AUROC:**
   - Overall positive prevalence is 1.798%. Accuracy alone does not determine false-positive rate. Always-negative accuracy is 98.190% on the full test stream and 98.623% on the capped sequence test stream.
   - The reference evaluation protocol requires source verification; SMOTE cannot be assumed to explain the score difference. In the unweighted ablation, the proposed TCN improves AUROC, AUPRC, precision, recall, and F1, while accuracy and specificity decrease. These are single-run findings without confidence intervals.

---

## 11. Explainability Analysis

SHAP analysis was performed across models:

1. **XGBoost (TreeExplainer):**
   - Top features: ICULOS (0.5255), Temperature (0.2355), HospAdmTime (0.2292), FiO2 (0.1888), Lactate (0.1859), Heart Rate (0.1641), Unit1 (0.1563), BUN (0.1245), WBC (0.1163), Creatinine (0.1111).
2. **TCN Proposed (KernelExplainer):**
   - Top features: ICULOS, BUN, Temperature, Unit2, Unit1, SaO2, Heart Rate, **`Resp_miss`**, pH, Creatinine.
   - The prior run reported **`Resp_miss`** among its top eight features. This ranking was not independently reproduced. TCN SHAP evaluates isolated single-hour inputs without temporal history, so these attributions do not explain the sequence predictions used for metrics or prove MNAR or clinical causality.
- **Missingness indicators for:** Lactate_miss, AST_miss, WBC_miss — absence of key labs may signal clinical decision-making patterns

---

## 12. Limitations

1. **Audit scope:** Raw-to-processed correspondence and training statistics were verified for all patients, with no label reversions. Clinical value ranges, sentinel zeros, and clinical onset timing remain unverified.
2. **Extreme lab missingness (>90%):** Models must rely heavily on missingness indicators and sparse observations for lab-based features
3. **Class imbalance:** Overall hourly imbalance is 54.60:1. Report AUPRC alongside AUROC and each evaluation population's prevalence.
4. **SMOTE not applied to sequences:** Standard SMOTE destroys temporal structure; class-weighted loss is used instead
5. **Single dataset:** Results may not generalise beyond PhysioNet CinC 2019
6. **Patient-level vs. hour-level evaluation:** Hour-level positive rate differs from patient-level; both should be reported
7. **Reproducibility:** Eight checkpoints reproduced saved test metrics; XGBoost remains unverified. LR loaded with a scikit-learn version warning (saved 1.6.1; audit 1.7.2). Original training hardware and exact training reproduction remain unverified.
8. **Historical training and evaluation:** Five-epoch single runs, no per-experiment seed reset, first-72-row truncation, and class weights calculated from uncapped labels. Historical threshold search covered only 0.05 to 0.95. The new isolated runner resets seeds, counts capped labels, and searches all validation score thresholds; it preserves the original results.
9. **Interpretability:** TCN SHAP uses single-hour inputs without temporal history; reported rankings were not independently reproduced.
10. **Application:** A local Streamlit research UI now exists; there is no deployed clinical inference API.

The 72-row cap excludes 5,255 positive training hours, 1,154 positive validation hours, and 1,143 positive test hours. Respectively, 508, 113, and 111 patients who are positive in their full stay have no positive labels in the retained prefix. This changes the evaluated task and must not be described as full-stay early-warning evaluation. Historical weighted sequence training used the full-training ratio 54.6046, while retained training labels have a ratio of 71.8471; the historical weight was approximately 24.0% lower. New isolated weighted experiments use the retained-label ratio, without modifying historical artifacts.

---

## 13. Conclusion and Future Work

This study proposes and evaluates a **Missingness-Aware TCN** for early sepsis prediction on the PhysioNet CinC 2019 dataset. The key contributions are:

1. A **comprehensive data-quality audit** verifying dataset integrity, split safety, and preprocessing correctness
2. **Recovery of missingness indicators** from the existing preprocessing pipeline (no fabrication required)
3. A **controlled ablation study** separating the effect of missingness indicators from class-imbalance handling
4. **Exploratory SHAP analysis**, subject to the single-hour TCN attribution limitation

**Future work:**
- Incorporate elapsed-time (log-delta) features as a third input channel for full time-aware representation
- Investigate multi-task learning: simultaneous prediction of sepsis onset time and binary label
- Explore attention mechanisms over the temporal dimension for interpretability
- Extend to external validation datasets (MIMIC-III, eICU)
- Apply patient-level SMOTE-NC on summary statistics to generate synthetic training patients

---

## 14. Reproducibility

### Installation

The commands below install packages, train models, or write artifacts; they are not a read-only validation procedure. The read-only audits described above did not change code, data, model checkpoints, or result files.

```powershell
cd "d:\Final Year Project\Sepsis_Prediction_TCN"
pip install -r requirements.txt
```

### Execution Order

```powershell
# Stage 1-2: Audit (always available)
python -X utf8 src/data_inspection.py
python -X utf8 src/data_quality_audit.py

# Stage 3: Preprocessing (requires raw .psv files)
python src/preprocessing.py

# Stage 4-5: Training
python -B -X utf8 src/run_experiments.py --epochs 6 --patience 2 --threads 4

# Stage 6-7: Explainability and visualisation
python src/explainability.py
python src/visualize_results.py

# Local research UI (does not train):
python -B -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false

# Automated checks:
python -B -m pytest -q -p no:cacheprovider tests

# Pipeline defaults to isolated experiment stage 8:
python src/run_pipeline.py
```

### File Descriptions

| File | Phase | Description |
|---|---|---|
| `src/config.py` | All | Central configuration, paths, hyperparameters |
| `src/data_inspection.py` | 1 | Dataset structure inspection |
| `src/data_quality_audit.py` | 1 | Comprehensive 8-section quality audit |
| `src/cohort.py` | — | Official patient cohort definition |
| `src/make_splits.py` | — | Patient-level 70/15/15 splits |
| `src/preprocess.py` | 3 | Raw → processed tensors (already done) |
| `src/preprocessing.py` | 3 | Preprocessing orchestrator |
| `src/sequence_builder.py` | 3 | PyTorch Datasets and DataLoaders |
| `src/models.py` | 4-5 | TCN, LSTM, GRU architectures |
| `src/train_baselines.py` | 4 | LR, XGBoost, LSTM, GRU, TCN training |
| `src/train_proposed_tcn.py` | 5+8 | Proposed TCN + ablation study |
| `src/evaluate.py` | 7 | Metrics, plots, evaluation utilities |
| `src/explainability.py` | 9 | SHAP analysis |
| `src/visualize_results.py` | 10 | All publication figures |
| `src/run_pipeline.py` | All | Master orchestration script |


## 15. New Isolated Experiments and Working Demo (2026-10-29)

The later master task authorized targeted corrections, bounded new training, and a local UI while retaining the first-72-row design. Original tables above remain historical results. The isolated run is recorded in `experiments/tcn_losses_20261029T154725515099Z/REPORT.md`, with configuration, versions, source snapshots, checkpoints, epoch logs, validation comparisons, final selection, test predictions, and preservation hashes.

Three proposed-TCN loss candidates (unweighted BCE, corrected weighted BCE, focal loss) each completed six epochs under seed 42, fixed architecture, and common capped populations. Early stopping and checkpoint selection monitored validation AUPRC. The exact validation PR thresholds supplied the F1 cutoff, including values below 0.05. The historical proposed checkpoint was a fixed validation reference. The final candidate was locked before new test evaluation.

| Candidate | Best epoch | Validation AUPRC | Validation F1 |
|---|---:|---:|---:|
| historical_proposed_retuned | historical | 0.059715 | 0.125532 |
| bce | 5 | 0.060881 | 0.118783 |
| weighted_bce | 6 | 0.064274 | 0.130173 |
| focal | 6 | 0.056704 | 0.121610 |

Historical metrics below are reproduced saved results rounded to four decimals; new metrics retain full precision in test_comparison.json. All three rows use 222,297 capped test hours from 6,051 patients (3,060 positive hours).

| Model | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Historical ablation baseline (unweighted) | 97.38% | 8.69% | 9.54% | 9.10% | 0.7520 | 0.0527 |
| Historical proposed (unweighted) | 96.82% | 8.97% | 14.28% | 11.02% | 0.7726 | 0.0584 |
| Selected new weighted BCE | 96.78% | 9.84% | 16.41% | 12.30% | 0.7823 | 0.0633 |

Selected checkpoint: `weighted_bce.pt` (epoch 6). Threshold: **0.7500504851**, selected only on validation. Confusion matrix: TN=214639, FP=4598, FN=2558, TP=502.

Relative to the historical proposed model, precision, recall, F1, AUROC, and AUPRC improved; accuracy and specificity decreased. AUPRC increased by 0.004886 (approximately 8.37% relative), AUROC by 0.009748, and F1 by 1.284 percentage points. These gains cannot be attributed solely to corrected weighting: epoch budget, seed reset, checkpoint-selection criterion, and threshold search also differ from the historical study. New unweighted/weighted/focal comparisons use a common protocol. No new test metrics were computed for losing candidates.

Runtime: 1220.2 seconds including data loading, benchmark, validation, training, and final test inference. Configured class weight was 71.847075 from retained training labels. Original artifacts were preserved; all 56 recorded file hashes matched at completion. Raw and processed patient files were read only.


A local Streamlit UI is implemented and tested with real saved-checkpoint inference. It accepts CSV/PSV, validates the 40-feature schema and consecutive hourly time axis, reproduces saved preprocessing, displays truncation and invalid-value warnings, and separates individual scores from aggregate metrics. Completed validation-selected experiments are available as the default model; the historical proposed model remains selectable. No patient uploads or predictions are automatically saved. Launch and test commands are in README.md; the implemented contract is in UI_PLAN.md.

All 21 automated tests passed. HTTP root/health checks returned 200; visual browser review was not performed. XGBoost verification, repeated-seed uncertainty, exact cross-environment training reproduction, external validation, calibration, clinical value/sentinel checks, and reference/lead-time claims remain unresolved. No packages were installed.


## 16. Controlled V2 Refinements, Independent Seeds, and Patient Uncertainty

The performance-only follow-up preserved the first-72-row study, original labels/preprocessing/splits, all previous artifacts, and the UI. The actual model is a missingness-aware causal TCN, without temporal attention or elapsed-time inputs; no architectural change was mixed into these training comparisons.

V2 is recorded in `experiments_v2/tcn_refine_20261029T170738771536Z/REPORT.md` and the independent replication report `experiments_v2/independent_seeds_20261029T171824288587Z/REPORT.md`. Earlier BCE/weighted-BCE/focal results were reverified on validation and reused instead of redundantly retrained.

Three new seed-42 refinements started from identical V1 weighted-checkpoint weights with fresh Adam: control LR 0.001 / weight decay 0.0001; lower LR 0.0003; stronger weight decay 0.0005. Each changed only its named variable relative to the control and stopped after two non-improving epochs (maximum three). Best trained validation AUPRC was 0.061305, 0.062443, and 0.061822, respectively, below the frozen reference 0.064274. No trained refinement was promoted. Optimizer reset and the short budget may contribute, but their effects were not isolated.

Validation selection used a predeclared multi-metric relative score (50% AUPRC, 25% F1, 10% AUROC, 7.5% precision, 7.5% recall), with minimum precision/recall/accuracy and an AUROC noninferiority guard. Thresholds maximized validation F1 subject to hourly FPR <=3%, with precision/recall alternatives recorded. This is a research alarm constraint, not a clinically validated operating requirement. The frozen model and its original threshold won before test loading. No losing refinement or replication seed received test evaluation. The cohort was already historically examined and is not new external validation.

| Test model | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Original proposed | 96.82% | 8.97% | 14.28% | 11.02% | 0.7726 | 0.0584 |
| V1 weighted incumbent | 96.78% | 9.84% | 16.41% | 12.30% | 0.7823 | 0.0633 |
| V2 final: retained incumbent | 96.78% | 9.84% | 16.41% | 12.30% | 0.7823 | 0.0633 |

No additional gain occurred in V2, including no threshold-only gain: threshold remains 0.7500504851 and predictions are byte-for-byte identical to V1's cached score array. Counts remain TN 214639, FP 4598, FN 2558, TP 502. Hourly FPR is 2.097%. Of 5,722 patients without a positive retained label, 685 (11.97%) had at least one threshold crossing, illustrating the remaining false-alarm burden.

Independent from-scratch seeds 43 and 44 each trained the winning weighted configuration for six epochs, selecting checkpoints at epochs 6 and 5 respectively. They were not used to cherry-pick a new final seed. Including the existing primary seed 42, validation mean +/- sample SD: precision 0.096317 +/- 0.003832; recall 0.173985 +/- 0.010809; F1 0.123952 +/- 0.005433; AUROC 0.788273 +/- 0.003589; AUPRC 0.063062 +/- 0.001258. These are three independent initialization runs, not the common-warm-start variability from the refinement phase.

Two hundred paired patient-cluster bootstrap draws yielded test 95% intervals: F1 [0.099689, 0.143636], AUROC [0.758182, 0.805250], AUPRC [0.050005, 0.077730]. Full retained patient sequences were resampled, not independent hourly rows. Paired difference intervals versus the original proposed checkpoint include zero; the earlier observed gains are not statistically conclusive under this analysis. These fixed-model intervals do not include training-seed variance or correct prior reuse of the cohort.

The new protocols, weights, logs, validation comparisons, operating points, seed statistics, configurations, checkpoint/source hashes, saved optimizer states for repetitions, confidence intervals, comparison CSVs, ROC/PR curves, and confusion plot are retained in the separate V2 directories. All 29 tests passed. All captured prior artifact/UI hashes match. No packages installed or dependency versions changed. This round modified no UI, original model/data artifacts, or Git branches. Reference-paper, calibration, clinical onset, external-validation, and clinical utility claims remain unverified.
