"""Isolated, bounded TCN loss comparison. Never trains/selects on test data."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import TCN
from study import (ROOT, MODEL_CFG, MAX_ROWS, CachedPatients, best_f1_threshold,
                   collate, metrics, predict, sha256)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def objective(logits, labels, kind, pos_weight):
    if kind == "weighted_bce":
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=logits.new_tensor(pos_weight))
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if kind == "focal":
        # gamma=2, alpha=0.75 positive / 0.25 negative. No added pos_weight.
        pt = torch.exp(-bce)
        alpha = torch.where(labels == 1, .75, .25)
        return (alpha*(1-pt).pow(2)*bce).mean()
    return bce.mean()


def protected_hashes():
    paths = []
    for directory in ("models", "results", "logs", "figures"):
        paths.extend(p for p in (ROOT/directory).rglob("*") if p.is_file())
    paths += list((ROOT/"data/processed").glob("*.json"))
    paths += [ROOT/"data/processed/split_manifest.csv"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-estimated-seconds", type=float, default=1200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.threads < 1:
        parser.error("epochs, patience, and threads must be positive")
    out = args.output or ROOT/"experiments"/dt.datetime.now(dt.timezone.utc).strftime("tcn_losses_%Y%m%dT%H%M%S%fZ")
    out = out.resolve()
    if ROOT/"experiments" not in out.parents:
        parser.error("Output must be a new directory under experiments/")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    preserved = protected_hashes()
    dump(out/"original_artifact_hashes.json", preserved)
    config = {**vars(args), "output": str(out), "model": MODEL_CFG, "input": "clinical + 1-observed",
              "n_input": 80, "max_rows": MAX_ROWS, "batch_size": 64, "lr": .001,
              "weight_decay": .0001, "grad_clip": 1., "selection": "maximum validation AUPRC; ties by F1",
              "threshold_selection": "exact validation maximum F1, lowest threshold wins ties",
              "focal": {"gamma": 2., "positive_alpha": .75}, "device": "cpu",
              "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "manifest_sha256": sha256(ROOT/"data/processed/split_manifest.csv"),
              "preprocessing_sha256": sha256(ROOT/"data/processed/preprocessing_config.json"),
              "source_sha256": {p.name: sha256(p) for p in [Path(__file__), ROOT/"src/study.py", ROOT/"src/models.py"]}}
    dump(out/"config.json", config)
    print(f"OUTPUT {out}", flush=True)
    train = CachedPatients("train")
    val = CachedPatients("val")
    config.update(train_hours=train.hours, train_positive_hours=train.positives,
                  pos_weight=train.pos_weight, val_hours=val.hours, val_positive_hours=val.positives)
    dump(out/"config.json", config)
    vl = DataLoader(val, batch_size=64, collate_fn=collate)
    # Benchmark uses a disposable model and RNG state is reset before every candidate.
    seed_all(args.seed)
    bench = TCN(80, **MODEL_CFG)
    optimizer = torch.optim.Adam(bench.parameters(), lr=.001)
    t0 = time.perf_counter()
    for i, (x, y, mask) in enumerate(DataLoader(train, batch_size=64, collate_fn=collate)):
        optimizer.zero_grad()
        objective(bench(x)[mask], y[mask], "bce", train.pos_weight).backward()
        optimizer.step()
        if i == 11:
            break
    seconds_per_batch = (time.perf_counter()-t0)/12
    estimated = seconds_per_batch*((len(train)+63)//64)*args.epochs*3*1.25
    dump(out/"runtime_estimate.json", {"seconds_per_batch": seconds_per_batch, "estimated_training_seconds": estimated})
    print(f"Estimated bounded training: {estimated:.0f}s; capped train pos_weight={train.pos_weight:.6f}", flush=True)
    if estimated > args.max_estimated_seconds:
        dump(out/"status.json", {"status": "blocked_compute_budget", "estimated_seconds": estimated})
        print("STOP: runtime estimate exceeds the allowed compute budget", flush=True)
        return
    del bench, optimizer
    summaries = []
    # Historical proposed checkpoint is a fixed reference, with validation-only retuning.
    reference = TCN(80, **MODEL_CFG)
    reference.load_state_dict(torch.load(ROOT/"models/tcn_proposed.pt", weights_only=True, map_location="cpu"))
    yv, pv = predict(reference, vl)
    threshold = best_f1_threshold(yv, pv)
    ref = {"name": "historical_proposed_retuned", "checkpoint": "reference.pt", "epochs_trained": 0,
           "validation": metrics(yv, pv, threshold), "source_checkpoint_sha256": sha256(ROOT/"models/tcn_proposed.pt")}
    torch.save(reference.state_dict(), out/"reference.pt")
    summaries.append(ref)
    del reference
    for kind in ("bce", "weighted_bce", "focal"):
        seed_all(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        tr = DataLoader(train, batch_size=64, shuffle=True, generator=generator, collate_fn=collate)
        model = TCN(80, **MODEL_CFG)
        opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=1, factor=.5)
        history = []
        best = -1.
        stalled = 0
        best_state = None
        checkpoint = out/f"{kind}.pt"
        candidate_start = time.perf_counter()
        for epoch in range(1, args.epochs+1):
            model.train()
            total_loss = 0.
            for x, y, mask in tr:
                opt.zero_grad()
                loss = objective(model(x)[mask], y[mask], kind, train.pos_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                total_loss += float(loss.detach())*int(mask.sum())
            yv, pv = predict(model, vl)
            vm = metrics(yv, pv, best_f1_threshold(yv, pv))
            history.append({"epoch": epoch, "mean_train_loss": total_loss/train.hours, "validation": vm})
            dump(out/f"{kind}_log.json", history)
            scheduler.step(vm["auprc"])
            print(f"{kind} epoch {epoch}: val AP={vm['auprc']:.6f} F1={vm['f1']:.6f} threshold={vm['threshold']:.8f}", flush=True)
            if vm["auprc"] > best:
                best = vm["auprc"]
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, checkpoint)
                best_metrics, best_epoch = vm, epoch
                stalled = 0
            else:
                stalled += 1
            if stalled >= args.patience:
                break
        summaries.append({"name": kind, "checkpoint": checkpoint.name, "validation": best_metrics,
                          "best_epoch": best_epoch, "epochs_trained": len(history),
                          "training_seconds": time.perf_counter()-candidate_start})
        dump(out/"validation_comparison.json", summaries)
    selected = max(summaries, key=lambda r: (r["validation"]["auprc"], r["validation"]["f1"]))
    # Persist the irrevocable validation-only choice BEFORE loading any test data.
    selection = {**selected, "model_cfg": MODEL_CFG, "n_input": 80, "max_rows": MAX_ROWS,
                 "checkpoint_sha256": sha256(out/selected["checkpoint"]),
                 "preprocessing_sha256": config["preprocessing_sha256"],
                 "selection_used_test": False}
    dump(out/"selection.json", selection)
    print(f"SELECTED {selected['name']} by validation AUPRC; now evaluating test once", flush=True)
    del train, val
    test = CachedPatients("test")
    model = TCN(80, **MODEL_CFG)
    model.load_state_dict(torch.load(out/selected["checkpoint"], weights_only=True, map_location="cpu"))
    yt, pt = predict(model, DataLoader(test, batch_size=64, collate_fn=collate))
    test_metrics = metrics(yt, pt, selected["validation"]["threshold"])
    np.savez_compressed(out/"selected_test_predictions.npz", y=yt, score=pt,
                        patient_ids=np.array(test.ids), lengths=np.array([len(y) for _, y in test.items]))
    historical = json.loads((ROOT/"results/proposed_tcn_metrics.json").read_text())
    result = {"selected": selected["name"], "test": test_metrics,
              "historical_ablation_baseline": historical["TCN_Baseline"],
              "historical_proposed": historical["TCN_Proposed"],
              "test_protocol_note": "Previously used historical test split; no new candidate test scores used for selection."}
    dump(out/"test_comparison.json", result)
    current = protected_hashes()
    if current != preserved:
        raise RuntimeError("Protected artifact hash mismatch")
    dump(out/"status.json", {"status": "complete", "original_artifacts_unchanged": True,
                             "elapsed_seconds": time.perf_counter()-started})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
