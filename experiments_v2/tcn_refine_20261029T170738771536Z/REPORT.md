# Controlled Refinement Experiment: No Additional Gain

The saved V1 weighted TCN remains the final validation-selected model. All three bounded refinements were rejected. No UI, architecture, patient split, label, preprocessing output, or previous experiment artifact was changed.

## Architecture and baseline

The implemented model is a causal missingness-aware TCN with 80 inputs: 40 scaled clinical values plus 40 missingness indicators. It does not implement temporal attention and does not consume stored elapsed-time channels. Those architectural changes were not mixed into this training experiment.

Before training, all four saved loss/reference checkpoints were re-evaluated on validation only and their ranking metrics matched. The earlier BCE/weighted-BCE/focal experiments were reused rather than retrained. Class weight 71.847075 was recalculated from 1,039,892 retained training hours, including 14,275 positive labels. Validation uses 6,050 patients and 223,640 hours. Test uses 6,051 patients and 222,297 hours. The manifest and preprocessing hashes match the prior run.

## Registered selection and threshold policy

Threshold: exact maximum validation F1 subject to hourly FPR <= 3%, with the lowest threshold winning ties. This research constraint is not a validated clinical alarm budget. Precision/recall trade-offs were saved for minimum precision 10%/15% and recall 20%. A model was eligible only with validation precision >=8%, recall >=15%, accuracy >=95%, and AUROC no more than 0.005 below the reference. Eligible candidates were ranked by 0.50 relative AUPRC + 0.25 relative F1 + 0.10 relative AUROC + 0.075 relative precision + 0.075 relative recall, normalized to the frozen reference.

All refinements used weighted BCE and seed 42, identical initial weights from the V1 six-epoch weighted checkpoint, fresh Adam state, batch size 64, and gradient clipping 1. Adam state was not available from V1, so this is explicit warm-start refinement, not exact resumed training. Each variant changed only LR or weight decay relative to the continuation control. Maximum three epochs, early-stopping patience two. Epoch zero remained eligible to avoid replacing the reference with a worse checkpoint.

## Completed new refinements

The values below describe the best trained epoch by validation AUPRC, not the frozen epoch-zero fallback. Every refinement stopped at epoch two and its deployable best remained epoch zero. Full per-epoch metrics and eligibility scores are in each run's log.json.

| Run | LR | Weight decay | Best trained epoch | Stop epoch | Validation AUPRC | F1 | AUROC | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| continuation_control | 0.001 | 0.0001 | 2 | 2 | 0.061305 | 0.117927 | 0.789086 | 109.8 |
| lower_lr | 0.0003 | 0.0001 | 1 | 2 | 0.062443 | 0.123181 | 0.790174 | 171.2 |
| stronger_decay | 0.001 | 0.0005 | 2 | 2 | 0.061822 | 0.121176 | 0.789920 | 131.7 |

The frozen validation reference has AUPRC 0.064274, F1 0.130173, and AUROC 0.788924. Lower LR and stronger decay recovered some AUROC compared with the continuation control, but did not improve the overall validation criterion or reference AUPRC/F1. Optimizer reset and the short refinement window are possible explanations, not established causes. No further hyperparameter search was performed after the final model was locked.

## Final held-out comparison

The final choice was written before test loading. Only the final candidate was evaluated; losing new checkpoints and later seed replications were not tested. This is the historically used test cohort, not an independent untouched external cohort.

| Model | Accuracy | Precision | Recall | F1 | AUROC | AUPRC | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original proposed | 96.82% | 8.97% | 14.28% | 11.02% | 0.7726 | 0.0584 | 2.023% |
| Prior weighted incumbent | 96.78% | 9.84% | 16.41% | 12.30% | 0.7823 | 0.0633 | 2.097% |
| V2 final: retained incumbent | 96.78% | 9.84% | 16.41% | 12.30% | 0.7823 | 0.0633 | 2.097% |

Threshold 0.7500504851. Confusion counts: TN=214639, FP=4598, FN=2558, TP=502. Hourly FPR is 2.097% (20.97 false-positive hours per 1,000 negative hours). Among 5722 patients without a positive retained label, 685 (11.97%) had at least one threshold crossing. This is a prefix-level burden measure, not a clinical alarm episode or onset-specific sensitivity measure.

No metric improved over the immediately preceding weighted incumbent in this round: it was retained unchanged. Against the original proposed checkpoint, changes calculated from full-precision reproduced values are accuracy -0.0441 pp, precision +0.8735 pp, recall +2.1242 pp, F1 +1.2853 pp, AUROC +0.009723, and AUPRC +0.004840. The gains versus that older model were established in V1, not discovered by these refinements.

The alarm-constrained validation search selected the same threshold as V1, so there is no threshold-only gain in this round. In general, threshold changes affect precision/recall/F1 and false alarms, not AUROC/AUPRC ranking metrics. Differences between V1 and the original historical study include training, epoch budget, seeds, checkpoint criterion, and threshold tuning; do not attribute the original gain to weighting alone.

## Patient-level uncertainty

200 paired patient-cluster bootstrap draws (seed 2026) resampled complete retained patient sequences, preserving within-patient correlation. Pointwise percentile 95% intervals:

| Metric | Lower | Upper |
|---|---:|---:|
| accuracy | 0.965131 | 0.970067 |
| precision | 0.079833 | 0.116062 |
| recall | 0.132058 | 0.194700 |
| f1 | 0.099689 | 0.143636 |
| auroc | 0.758182 | 0.805250 |
| auprc | 0.050005 | 0.077730 |

Paired difference intervals versus the original proposed model include zero for precision, recall, F1, AUROC, and AUPRC. The observed gains are not statistically conclusive under this resampling analysis. These intervals do not include training-seed variability, correct repeated reuse of this test cohort, or establish clinical validity. Independent seed repeats are recorded separately and do not select a new final model.

## Reproduction and preservation

```powershell
python -B -m src.run_experiments_v2 --epochs 3 --patience 2 --repeat-seeds 43 44 --bootstrap 200 --threads 4 --max-estimated-seconds 1500
python -B -m pytest -q -p no:cacheprovider tests
```

The frozen winner has no new training recipe: its six-epoch weighted-BCE source/configuration/checkpoint is preserved in `experiments/tcn_losses_20261029T154725515099Z`. V2 creates a new unique experiments_v2 directory, never overwriting the original run. Source snapshots, versions, configs, logs, validation predictions, operating points, final selection, selected test predictions, confidence intervals, ROC/PR plots, confusion matrix, and preservation hashes are retained here.

All 29 tests passed. The 79 captured original artifact/UI hashes matched at completion; raw and processed patient files were read only. No packages installed, dependency versions changed, UI redesigned, commits, pushes, resets, or merges. Core refinement runtime was 501.2 seconds. Independent full-seed replications have their own runtime and reports.
