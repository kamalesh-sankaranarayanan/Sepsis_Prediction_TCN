# Independent Training Seeds and Overall V2 Conclusion

The incumbent six-epoch weighted TCN remains selected. Three new short refinements did not beat it on the registered validation criterion. See the companion report at `../tcn_refine_20261029T170738771536Z/REPORT.md` for their complete results, exact operating policies, plots, final test comparison, and patient bootstrap.

## Independent seeds

Seeds 43 and 44 trained from scratch using the incumbent's weighted-BCE configuration: 80 inputs, fixed architecture, capped-label weight 71.847075, Adam LR 0.001, weight decay 0.0001, batch 64, gradient clip 1, up to six epochs, validation-AUPRC checkpoint selection/early stopping (patience two), and ReduceLROnPlateau on validation AUPRC (factor 0.5, patience one). Seed 42 is reused from the verified original run; it was not redundantly retrained.

These repeats are independent initializations, unlike the preceding common-warm-start refinements. Their validation thresholds maximize F1 under the same FPR <=3% cap; unrestricted alternatives are also saved. The chosen thresholds are unrestricted optima too, so the cap did not alter their cutoffs. Neither new seed was selected for deployment or evaluated on test. Primary seed 42 was fixed before replication; no favorable seed was cherry-picked.

| Seed | Best epoch | Stopping epoch | Validation precision | Recall | F1 | AUROC | AUPRC | Threshold | Training seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 6 | 6 | 0.100160 | 0.185870 | 0.130173 | 0.788924 | 0.064274 | 0.75005049 | reused |
| 43 | 6 | 6 | 0.096295 | 0.164741 | 0.121544 | 0.791491 | 0.063150 | 0.74478519 | 158.09715960000176 |
| 44 | 5 | 6 | 0.092497 | 0.171344 | 0.120139 | 0.784403 | 0.061763 | 0.79019862 | 171.65444939999725 |

Sample mean and standard deviation across three seeds (validation only):

| Metric | Mean | Sample SD |
|---|---:|---:|
| accuracy | 0.966704 | 0.000920 |
| precision | 0.096317 | 0.003832 |
| recall | 0.173985 | 0.010809 |
| f1 | 0.123952 | 0.005433 |
| auroc | 0.788273 | 0.003589 |
| auprc | 0.063062 | 0.001258 |

## Final result and uncertainty

No additional final-model gain in V2: seed 42's existing weighted checkpoint and threshold 0.7500504851 remain unchanged. Test accuracy 96.78%, precision 9.84%, recall 16.41%, F1 12.30%, AUROC 0.7823, AUPRC 0.0633; TN 214639, FP 4598, FN 2558, TP 502. Changes relative to the immediate V1 incumbent are exactly zero. Relative to the much older proposed checkpoint, the previously measured gains remain; they are not new V2 improvements.

Patient-level 95% bootstrap intervals are F1 [0.099689, 0.143636], AUROC [0.758182, 0.805250], AUPRC [0.050005, 0.077730]. Paired difference intervals versus the older proposed checkpoint include zero. Report these gains as observed, not statistically conclusive. Three seeds and 200 bootstrap draws give limited uncertainty evidence, not external/clinical validation. Bootstrap intervals condition on the fixed model and do not include model-training randomness.

Hourly test FPR is 2.097%; 11.97% of patients without positive labels in the retained prefix had at least one threshold crossing. This burden remains substantial. Thresholds affect decision metrics, not underlying AUROC/AUPRC. No threshold-only gain was found in V2; original training and threshold changes cannot be uniquely credited for the earlier F1 difference.

## Commands

Reproduce the V2 refinements and bootstrap (creates a fresh output directory):

```powershell
python -B -m src.run_experiments_v2 --epochs 3 --patience 2 --repeat-seeds 43 44 --bootstrap 200 --threads 4 --max-estimated-seconds 1500
```

When the incumbent wins, run the independent training-seed analysis:

```powershell
python -B -m src.repeat_selected_seeds --seeds 43 44 --threads 4 --max-estimated-seconds 1000
```

Reproduce the original winning seed-42 weighted configuration as part of its original controlled loss matrix:

```powershell
python -B -X utf8 src/run_experiments.py --epochs 6 --patience 2 --seed 42 --threads 4 --max-estimated-seconds 1200
```

Run checks:

```powershell
python -B -m pytest -q -p no:cacheprovider tests
```

All 29 tests passed. Both independent runs completed six epochs; their measured training durations are listed above, excluding shared data loading. Each run retains its best checkpoint, optimizer/scheduler/Torch/loader RNG state, config, epoch log, validation predictions, and operating-point alternatives. No test predictions exist for the repeat seeds. Exact source snapshots and dependency versions are recorded by the companion experiment protocol; this repeat's protocol also records Torch/NumPy and hashes of the split/statistics artifacts.

All captured original artifact/UI hashes and the first-phase V2 snapshot hashes still match. No datasets, splits, previous checkpoints, metrics, logs, or figures were overwritten. UI sources are untouched. No installation, dependency-version changes, destructive Git actions, commits, or pushes were performed.

## Remaining work

The current architecture has neither explicit temporal attention nor elapsed-time inputs. Adding either requires a separately controlled architectural study. Stronger conclusions require a larger preregistered training budget, validation stability analysis, and genuinely new external data. Full optimizer continuation of older V1 is unavailable because its original optimizer state was not saved; V2 refinement explicitly reset Adam. No clinical onset lead-time, calibration, clinical utility, or paper-surpassing claims are supported. This historically examined test split must not be repeatedly used for model search.
