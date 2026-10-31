"""Validation-only operating policy and patient-cluster uncertainty calculations."""
from __future__ import annotations

import numpy as np

from .study import metrics


def operating_metrics(y, score, threshold):
    result = metrics(y, score, threshold)
    result["false_positive_rate"] = result["fp"] / max(result["fp"] + result["tn"], 1)
    result["alert_fraction"] = (result["tp"] + result["fp"]) / result["hours"]
    result["false_positives_per_1000_negative_hours"] = 1000*result["false_positive_rate"]
    return result


def threshold_options(y, score, max_fpr=.03):
    """Exact score thresholds, with ties grouped and an explicit false-alarm cap."""
    y, score = np.asarray(y), np.asarray(score)
    if y.ndim != 1 or score.shape != y.shape or not len(y) or not np.isin(y, [0, 1]).all() or not np.isfinite(score).all():
        raise ValueError("Invalid validation labels or scores")
    pos = int(y.sum())
    neg = len(y)-pos
    if not pos or not neg or not 0 <= max_fpr <= 1:
        raise ValueError("Both validation classes and a valid false-positive cap are required")
    order = np.argsort(-score, kind="stable")
    ordered = score[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered) != 0), len(y)-1]
    tp = np.cumsum(y[order], dtype=np.float64)[ends][::-1]
    fp = (ends+1)[::-1]-tp
    thresholds = ordered[ends][::-1].astype(np.float64)
    thresholds = np.r_[thresholds, np.nextafter(float(score.max()), np.inf)]
    tp, fp = np.r_[tp, 0.], np.r_[fp, 0.]
    precision = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=tp+fp > 0)
    recall, fpr = tp/pos, fp/neg
    f1 = 2*tp/(pos+tp+fp)
    constrained = fpr <= max_fpr

    def choose(mask, objective):
        if not mask.any():
            return None
        indices = np.flatnonzero(mask)
        i = indices[np.argmax(objective[indices])]
        return {"threshold": float(thresholds[i]), "precision": float(precision[i]),
                "recall": float(recall[i]), "f1": float(f1[i]),
                "false_positive_rate": float(fpr[i]), "tp": int(tp[i]), "fp": int(fp[i])}

    return {"selected": choose(constrained, f1),
            "reason": f"Maximum validation F1 subject to hourly FPR <= {max_fpr:.3f}; lowest threshold wins ties",
            "unrestricted_max_f1": choose(np.ones(len(f1), dtype=bool), f1),
            "precision_at_least_10pct": choose(constrained & (precision >= .10), recall),
            "precision_at_least_15pct": choose(constrained & (precision >= .15), recall),
            "recall_at_least_20pct": choose(constrained & (recall >= .20), -fpr)}


def validation_score(value, reference, max_fpr=.03):
    """Predeclared multi-metric objective; no test information is accepted."""
    eligible = (value["false_positive_rate"] <= max_fpr and value["precision"] >= .08 and
                value["recall"] >= .15 and value["accuracy"] >= .95 and
                value["auroc"] >= reference["auroc"]-.005)
    if not eligible:
        return None
    weights = {"auprc": .50, "f1": .25, "auroc": .10, "precision": .075, "recall": .075}
    return float(sum(weight*value[key]/reference[key] for key, weight in weights.items()))


class ClusterMetricCache:
    """Weighted ranking metrics with precomputed tie groups; no hourly resampling."""
    def __init__(self, y, scores, threshold, lengths):
        y, scores = np.asarray(y), np.asarray(scores)
        lengths = np.asarray(lengths, dtype=np.int64)
        if (y.ndim != 1 or scores.shape != y.shape or lengths.sum() != len(y) or
                (lengths <= 0).any() or not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all()):
            raise ValueError("Invalid patient grouping or predictions")
        order = np.argsort(-scores, kind="stable")
        self.patient = np.repeat(np.arange(len(lengths)), lengths)[order]
        self.y = y[order].astype(np.float64)
        self.predicted = scores[order] >= threshold
        ordered = scores[order]
        self.ends = np.r_[np.flatnonzero(np.diff(ordered) != 0), len(y)-1]

    def calculate(self, patient_counts):
        weight = np.asarray(patient_counts)[self.patient]
        positive = weight*self.y
        negative = weight-positive
        tp_path = np.cumsum(positive)[self.ends]
        fp_path = np.cumsum(negative)[self.ends]
        pos, neg = tp_path[-1], fp_path[-1]
        if pos == 0 or neg == 0:
            return None
        precision_path = np.divide(tp_path, tp_path+fp_path, out=np.zeros_like(tp_path), where=tp_path+fp_path > 0)
        ap = np.sum(np.diff(np.r_[0., tp_path])/pos*precision_path)
        tpr, fpr = np.r_[0., tp_path/pos], np.r_[0., fp_path/neg]
        auroc = np.sum((tpr[1:]+tpr[:-1])*.5*np.diff(fpr))
        tp = positive[self.predicted].sum()
        fp = negative[self.predicted].sum()
        fn, tn = pos-tp, neg-fp
        return {"accuracy": float((tp+tn)/(pos+neg)), "precision": float(tp/max(tp+fp, 1)),
                "recall": float(tp/pos), "f1": float(2*tp/max(2*tp+fp+fn, 1)),
                "auroc": float(auroc), "auprc": float(ap),
                "specificity": float(tn/neg), "false_positive_rate": float(fp/neg)}


def patient_bootstrap(y, predictions, lengths, replicates=200, seed=2026):
    """Paired patient bootstrap with all retained hours of each sampled patient."""
    caches = {name: ClusterMetricCache(y, score, threshold, lengths)
              for name, (score, threshold) in predictions.items()}
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in caches}
    for _ in range(replicates):
        counts = np.bincount(rng.integers(len(lengths), size=len(lengths)), minlength=len(lengths))
        values = {name: cache.calculate(counts) for name, cache in caches.items()}
        if any(v is None for v in values.values()):
            continue
        for name in caches:
            samples[name].append(values[name])
    def summarize(rows):
        return {key: {"lower": float(np.percentile([r[key] for r in rows], 2.5)),
                      "upper": float(np.percentile([r[key] for r in rows], 97.5))}
                for key in rows[0]}
    if not samples or not next(iter(samples.values())):
        raise ValueError("No bootstrap draws contained both classes")
    deltas = {}
    for name, rows in samples.items():
        if name == "selected":
            continue
        paired = [{k: a[k]-b[k] for k in a} for a, b in zip(samples["selected"], rows)]
        deltas["selected_minus_"+name] = summarize(paired)
    return {"method": "paired patient-cluster percentile bootstrap; 95% intervals",
            "seed": seed, "requested_replicates": replicates,
            "valid_replicates": len(next(iter(samples.values()))), "patients": len(lengths),
            "intervals": {name: summarize(rows) for name, rows in samples.items()},
            "paired_differences": deltas}
