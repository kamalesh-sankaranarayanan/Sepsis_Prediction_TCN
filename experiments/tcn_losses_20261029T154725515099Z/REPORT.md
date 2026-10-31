# Isolated TCN Loss Comparison: Completed

Run: `tcn_losses_20261029T154725515099Z`. Fixed first-72-row design, seed 42, CPU with four threads. All three new candidates completed six epochs. Architecture, optimizer, patient splits, and feature representation were held fixed across the new candidates. Early stopping monitored validation AUPRC with patience two; it did not trigger before the epoch cap. The scheduler reduced learning rate on validation AUPRC plateaus. Focal loss used gamma 2 and positive alpha 0.75.

The original proposed checkpoint was a fixed validation reference with an unrestricted validation-F1 threshold search. Each new loss candidate was initialized from the same seed. Validation AUPRC selected checkpoints and the final candidate; ties were resolved by validation F1. Only the selected candidate received new test evaluation after selection.json was written. The historical test set was previously examined, so this is not independent external validation.

## Validation selection

| Candidate | Best epoch | Epochs trained | Validation AUPRC | Validation F1 | Threshold |
|---|---:|---:|---:|---:|---:|
| historical_proposed_retuned | historical | 0 | 0.059715 | 0.125532 | 0.05235592 |
| bce | 5 | 6 | 0.060881 | 0.118783 | 0.04384930 |
| weighted_bce | 6 | 6 | 0.064274 | 0.130173 | 0.75005049 |
| focal | 6 | 6 | 0.056704 | 0.121610 | 0.29789615 |

## Test comparison

Historical metrics below are reproduced saved results rounded to four decimals; new metrics retain full precision in test_comparison.json. All three rows use 222,297 capped test hours from 6,051 patients (3,060 positive hours).

| Model | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Historical ablation baseline (unweighted) | 97.38% | 8.69% | 9.54% | 9.10% | 0.7520 | 0.0527 |
| Historical proposed (unweighted) | 96.82% | 8.97% | 14.28% | 11.02% | 0.7726 | 0.0584 |
| Selected new weighted BCE | 96.78% | 9.84% | 16.41% | 12.30% | 0.7823 | 0.0633 |

Selected checkpoint: `weighted_bce.pt` (epoch 6). Threshold: **0.7500504851**, selected only on validation. Confusion matrix: TN=214639, FP=4598, FN=2558, TP=502.

Relative to the historical proposed model, precision, recall, F1, AUROC, and AUPRC improved; accuracy and specificity decreased. AUPRC increased by 0.004886 (approximately 8.37% relative), AUROC by 0.009748, and F1 by 1.284 percentage points. These gains cannot be attributed solely to corrected weighting: epoch budget, seed reset, checkpoint-selection criterion, and threshold search also differ from the historical study. New unweighted/weighted/focal comparisons use a common protocol. No new test metrics were computed for losing candidates.

Runtime: 1220.2 seconds including data loading, benchmark, validation, training, and final test inference. Configured class weight was 71.847075 from retained training labels. Original artifacts were preserved; all 56 recorded file hashes matched at completion. Raw and processed patient files were read only.

## Verification and limits

Final automated checks: 21 pytest tests passed, including selected-checkpoint UI inference, preprocessing consistency, causal behavior, bad uploads, missing checkpoints, capped class weighting, and manifest enforcement. Local HTTP root and health checks returned 200. No visual browser screenshot review was performed. No packages were installed, and no commits, pushes, merges, or branch deletion were performed.

Versions are in environment.json; the exact training source matching the recorded source hashes is archived in source/. The current runner additionally automates environment/source capture for future runs. Original artifact hashes, configuration, epoch logs, selection, and test predictions are retained in this directory.

Single seed, small epoch budget, previously used test cohort, no uncertainty estimates, no measured calibration or onset-specific utility, and no external validation. XGBoost remains unverified because its package is unavailable. The UI presents scores as research outputs, not clinical probabilities or diagnoses.
