"""Bounded, isolated continuation experiments; UI and historical files are untouched.

Run: python -B -m src.run_experiments_v2 --epochs 3 --repeat-seeds 43 44 --bootstrap 200
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import TCN
from .study import ROOT, MODEL_CFG, CachedPatients, collate, predict, sha256
from .experiment_protocol_v2 import operating_metrics, threshold_options, validation_score, patient_bootstrap

V1 = ROOT/"experiments/tcn_losses_20261029T154725515099Z"
MATRIX = [{"name": "continuation_control", "lr": .001, "weight_decay": .0001},
          {"name": "lower_lr", "lr": .0003, "weight_decay": .0001},
          {"name": "stronger_decay", "lr": .001, "weight_decay": .0005}]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def protected_hashes():
    paths = []
    for directory in ("models", "results", "logs", "figures", "experiments"):
        paths += [p for p in (ROOT/directory).rglob("*") if p.is_file()]
    paths += list((ROOT/"data/processed").glob("*.json"))
    paths += [ROOT/"data/processed/split_manifest.csv", ROOT/"app.py", ROOT/"UI_PLAN.md"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def model_from(state):
    model = TCN(80, **MODEL_CFG)
    model.load_state_dict(state)
    return model


def validation(model, loader, cap):
    y, score = predict(model, loader)
    options = threshold_options(y, score, cap)
    return operating_metrics(y, score, options["selected"]["threshold"]), options, y, score


def fit(cfg, seed, state, reference, train, vl, out, args):
    seed_all(seed)
    model = model_from(state)
    generator = torch.Generator().manual_seed(seed)
    tr = DataLoader(train, batch_size=64, shuffle=True, generator=generator, collate_fn=collate)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=1, factor=.5)
    name = f"{cfg['name']}_seed{seed}"
    directory = out/name
    directory.mkdir()
    dump(directory/"config.json", {**cfg, "seed": seed, "loss": "weighted_bce", "pos_weight": train.pos_weight,
                                  "optimizer_reset": True, "warm_start": str(V1/"weighted_bce.pt"),
                                  "max_epochs": args.epochs, "patience": args.patience})
    # Epoch zero is a valid frozen reference; do not replace it with worse refinement.
    best_value = reference.copy()
    best_score = validation_score(reference, reference, args.max_fpr)
    best_epoch = 0
    checkpoint = directory/"best.pt"
    torch.save(model.state_dict(), checkpoint)
    history = []
    stalled = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        model.train()
        total = 0.
        for x, y, mask in tr:
            opt.zero_grad()
            logits = model(x)[mask]
            loss = F.binary_cross_entropy_with_logits(logits, y[mask], pos_weight=logits.new_tensor(train.pos_weight))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            total += float(loss.detach())*int(mask.sum())
        vm, options, _, _ = validation(model, vl, args.max_fpr)
        score = validation_score(vm, reference, args.max_fpr)
        history.append({"epoch": epoch, "mean_train_loss": total/train.hours, "validation": vm,
                        "validation_score": score, "operating_points": options})
        dump(directory/"log.json", history)
        scheduler.step(score if score is not None else 0.)
        print(f"{name} epoch {epoch}: AP={vm['auprc']:.6f} F1={vm['f1']:.6f} ROC={vm['auroc']:.6f} FPR={vm['false_positive_rate']:.5f} score={score}", flush=True)
        if score is not None and score > best_score:
            best_value, best_score, best_epoch = vm, score, epoch
            torch.save(model.state_dict(), checkpoint)
            stalled = 0
        else:
            stalled += 1
        if stalled >= args.patience:
            break
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    verified, options, y, probability = validation(model, vl, args.max_fpr)
    assert np.isclose(verified["auprc"], best_value["auprc"])
    np.savez_compressed(directory/"validation_predictions.npz", y=y, score=probability)
    summary = {"id": name, "configuration": cfg, "seed": seed, "checkpoint": str(checkpoint.relative_to(out)),
               "best_epoch": best_epoch, "stopping_epoch": len(history), "validation": verified,
               "validation_score": best_score, "operating_points": options,
               "training_seconds": time.perf_counter()-started}
    dump(directory/"summary.json", summary)
    return summary


def plots(out, y, predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    for name, curve in [("precision_recall", precision_recall_curve), ("roc", roc_curve)]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, (probability, _) in predictions.items():
            a, b, _ = curve(y, probability)
            ax.plot(b, a, label=label) if name == "precision_recall" else ax.plot(a, b, label=label)
        ax.set(xlabel="Recall" if name == "precision_recall" else "False-positive rate",
               ylabel="Precision" if name == "precision_recall" else "True-positive rate")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out/(name+".png"), dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat-seeds", nargs="*", type=int, default=[43, 44])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-fpr", type=float, default=.03)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--max-estimated-seconds", type=float, default=1500)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.bootstrap < 20 or args.threads < 1:
        parser.error("Positive epochs/patience/threads and at least 20 bootstrap draws required")
    if len(set([args.seed]+args.repeat_seeds)) != len([args.seed]+args.repeat_seeds):
        parser.error("All seeds must be distinct")
    torch.set_num_threads(args.threads)
    out = ROOT/"experiments_v2"/dt.datetime.now(dt.timezone.utc).strftime("tcn_refine_%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    frozen_hashes = protected_hashes()
    dump(out/"preservation_hashes.json", frozen_hashes)
    source = out/"source"
    source.mkdir()
    for name in ("run_experiments_v2.py", "experiment_protocol_v2.py", "study.py", "models.py"):
        (source/name).write_bytes((ROOT/"src"/name).read_bytes())
    config = {**vars(args), "matrix": MATRIX, "model": MODEL_CFG, "n_input": 80, "max_rows": 72,
              "loss": "weighted_bce", "batch_size": 64, "grad_clip": 1.,
              "warm_start": str(V1/"weighted_bce.pt"), "warm_start_sha256": sha256(V1/"weighted_bce.pt"),
              "optimizer": "fresh Adam per refinement; previous optimizer state was not saved",
              "scheduler": "ReduceLROnPlateau, validation composite score, factor .5, patience 1",
              "criterion": "0.50 relative AP + 0.25 relative F1 + 0.10 relative ROC + 0.075 relative precision + 0.075 relative recall",
              "eligibility": {"min_precision": .08, "min_recall": .15, "min_accuracy": .95,
                              "max_auroc_decline_from_reference": .005, "max_fpr": args.max_fpr},
              "selection": "declared primary seed chooses configuration; replication never selects a seed; primary checkpoint is final",
              "replication_note": "Refinement randomness only; all seeds share a seed42-pretrained checkpoint",
              "test_note": "Historically examined test cohort; no test data used for new selection",
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pandas", "scikit-learn", "matplotlib")},
              "python": sys.version, "device": "cpu", "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "manifest_sha256": sha256(ROOT/"data/processed/split_manifest.csv"),
              "preprocessing_sha256": sha256(ROOT/"data/processed/preprocessing_config.json")}
    dump(out/"protocol.json", config)
    prior_config = json.loads((V1/"config.json").read_text())
    assert config["manifest_sha256"] == prior_config["manifest_sha256"]
    assert config["preprocessing_sha256"] == prior_config["preprocessing_sha256"]
    print(f"OUTPUT {out}", flush=True)
    train, val = CachedPatients("train"), CachedPatients("val")
    assert not set(train.ids) & set(val.ids)
    config.update(pos_weight=train.pos_weight, train_hours=train.hours, train_positive_hours=train.positives,
                  val_hours=val.hours, val_positive_hours=val.positives)
    assert train.hours == prior_config["train_hours"] and train.positives == prior_config["train_positive_hours"]
    assert np.isclose(train.pos_weight, prior_config["pos_weight"])
    dump(out/"protocol.json", config)
    vl = DataLoader(val, batch_size=64, collate_fn=collate)
    initial_state = torch.load(V1/"weighted_bce.pt", weights_only=True, map_location="cpu")
    reference_model = model_from(initial_state)
    reference, options, yv, pv = validation(reference_model, vl, args.max_fpr)
    recorded = json.loads((V1/"validation_comparison.json").read_text())
    recorded = next(row["validation"] for row in recorded if row["name"] == "weighted_bce")
    assert np.isclose(reference["auprc"], recorded["auprc"], atol=1e-9)
    assert np.isclose(reference["auroc"], recorded["auroc"], atol=1e-9)
    if validation_score(reference, reference, args.max_fpr) is None:
        raise ValueError("Reference does not satisfy preregistered operating policy")
    dump(out/"baseline_verification.json", {"validation": reference, "operating_points": options,
                                           "saved_validation_ranking_metrics_match": True})
    np.savez_compressed(out/"reference_validation_predictions.npz", y=yv, score=pv)
    summaries = []
    # Re-evaluate existing loss checkpoints on validation only, no expensive retraining.
    reused = []
    for row in json.loads((V1/"validation_comparison.json").read_text()):
        model = TCN(80, **MODEL_CFG)
        model.load_state_dict(torch.load(V1/row["checkpoint"], weights_only=True, map_location="cpu"))
        vm, points, _, _ = validation(model, vl, args.max_fpr)
        assert np.isclose(vm["auprc"], row["validation"]["auprc"], atol=1e-9)
        reused.append({"name": row["name"], "reused_prior_experiment": True,
                       "validation": vm, "operating_points": points})
    dump(out/"reused_loss_comparison.json", reused)
    # Disposable CPU benchmark; reset all RNGs for each real refinement.
    seed_all(args.seed)
    bench = model_from(initial_state)
    opt = torch.optim.Adam(bench.parameters(), lr=.001)
    t0 = time.perf_counter()
    for i, (x, y, mask) in enumerate(DataLoader(train, batch_size=64, collate_fn=collate)):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(bench(x)[mask], y[mask], pos_weight=torch.tensor(train.pos_weight)).backward()
        opt.step()
        if i == 11:
            break
    per_batch = (time.perf_counter()-t0)/12
    estimate = per_batch*((len(train)+63)//64)*args.epochs*(len(MATRIX)+len(args.repeat_seeds))*1.3
    dump(out/"runtime_estimate.json", {"seconds_per_batch": per_batch, "estimated_training_seconds": estimate})
    print(f"Estimated bounded refinements: {estimate:.0f}s; capped weight {train.pos_weight:.6f}", flush=True)
    if estimate > args.max_estimated_seconds:
        dump(out/"status.json", {"status": "blocked_compute_budget"})
        return
    del bench, opt, reference_model
    baseline = {"id": "frozen_v1", "checkpoint": "frozen_reference.pt", "validation": reference,
                "validation_score": 1., "best_epoch": 0, "stopping_epoch": 0}
    torch.save(initial_state, out/"frozen_reference.pt")
    summaries.append(baseline)
    for cfg in MATRIX:
        summaries.append(fit(cfg, args.seed, initial_state, reference, train, vl, out, args))
        dump(out/"validation_comparison.json", summaries)
    selected = max(summaries, key=lambda r: r["validation_score"])
    # Configuration is locked before validation seed repeats; repetitions do not pick a seed.
    dump(out/"selection_before_repeats.json", {**selected, "used_test": False, "primary_seed": args.seed})
    repeats = []
    if selected["id"] != "frozen_v1":
        for seed in args.repeat_seeds:
            repeats.append(fit(selected["configuration"], seed, initial_state, reference, train, vl, out, args))
    rows = [selected]+repeats
    seed_stats = {key: {"mean": float(np.mean([row["validation"][key] for row in rows])),
                        "std": float(np.std([row["validation"][key] for row in rows], ddof=1)) if len(rows)>1 else None}
                  for key in ("accuracy", "precision", "recall", "f1", "auroc", "auprc", "false_positive_rate")}
    dump(out/"seed_variability.json", {"runs": rows, "statistics": seed_stats,
                                       "scope": "warm-start refinement variability, not independent end-to-end pretraining"})
    selection = {**selected, "used_test": False, "primary_seed": args.seed,
                 "checkpoint_sha256": sha256(out/selected["checkpoint"]), "model_cfg": MODEL_CFG,
                 "preprocessing_sha256": config["preprocessing_sha256"]}
    dump(out/"final_selection.json", selection)
    print(f"LOCKED {selected['id']}; now reading test once for selected new candidate", flush=True)
    del train, val
    test = CachedPatients("test")
    test_loader = DataLoader(test, batch_size=64, collate_fn=collate)
    model = model_from(torch.load(out/selected["checkpoint"], weights_only=True, map_location="cpu"))
    yt, pt = predict(model, test_loader)
    lengths = np.array([len(y) for _, y in test.items])
    np.savez_compressed(out/"selected_test_predictions.npz", y=yt, score=pt,
                        lengths=lengths, patient_ids=np.array(test.ids))
    with np.load(V1/"selected_test_predictions.npz", allow_pickle=False) as z:
        assert np.array_equal(yt, z["y"]) and np.array_equal(lengths, z["lengths"]) and np.array_equal(test.ids, z["patient_ids"])
        prior_probability = z["score"].copy()
    old_model = model_from(torch.load(ROOT/"models/tcn_proposed.pt", weights_only=True, map_location="cpu"))
    old_y, old_probability = predict(old_model, test_loader)
    assert np.array_equal(yt, old_y)
    predictions = {"selected": (pt, selected["validation"]["threshold"]),
                   "v1_weighted": (prior_probability, recorded["threshold"]),
                   "original_proposed": (old_probability, .05)}
    result = {name: operating_metrics(yt, prob, threshold) for name, (prob, threshold) in predictions.items()}
    saved_old = json.loads((ROOT/"results/proposed_tcn_metrics.json").read_text())["TCN_Proposed"]
    saved_v1 = json.loads((V1/"test_comparison.json").read_text())["test"]
    for key in ("accuracy", "precision", "recall", "f1", "auroc", "auprc", "tp", "tn", "fp", "fn"):
        assert round(result["original_proposed"][key], 4) == saved_old[key], key
        assert np.isclose(result["v1_weighted"][key], saved_v1[key]), key
    # Separates changed ranking from threshold effects: same new scores at old cutoff.
    result["selected_at_v1_threshold"] = operating_metrics(yt, pt, recorded["threshold"])
    dump(out/"test_comparison.json", result)
    uncertainty = patient_bootstrap(yt, predictions, lengths, args.bootstrap)
    dump(out/"patient_bootstrap.json", uncertainty)
    plots(out, yt, predictions)
    import matplotlib.pyplot as plt
    m = result["selected"]
    fig, ax = plt.subplots(figsize=(4, 4))
    matrix = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted label", ylabel="Supplied label")
    fig.tight_layout()
    fig.savefig(out/"confusion_matrix.png", dpi=150)
    plt.close(fig)
    assert protected_hashes() == frozen_hashes, "Existing artifact or UI changed"
    dump(out/"status.json", {"status": "complete", "original_artifacts_and_ui_unchanged": True,
                             "elapsed_seconds": time.perf_counter()-started})
    print(json.dumps({"selection": selected["id"], "test": result, "seed_statistics": seed_stats}), flush=True)


if __name__ == "__main__":
    main()
