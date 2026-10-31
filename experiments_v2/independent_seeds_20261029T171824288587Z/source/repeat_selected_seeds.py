"""Validation-only independent seeds for the locked V1 weighted configuration.

These repeats never select a new final model and never read test predictions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import TCN
from .study import ROOT, MODEL_CFG, CachedPatients, collate, predict, sha256
from .experiment_protocol_v2 import operating_metrics, threshold_options
from .run_experiments_v2 import protected_hashes

V1 = ROOT/"experiments/tcn_losses_20261029T154725515099Z"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-estimated-seconds", type=float, default=1000)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or 42 in args.seeds or any(s < 0 for s in args.seeds):
        parser.error("Use distinct nonnegative replication seeds other than the fixed primary seed 42")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    out = ROOT/"experiments_v2"/dt.datetime.now(dt.timezone.utc).strftime("independent_seeds_%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    frozen = protected_hashes()
    write(out/"preservation_hashes.json", frozen)
    source = out/"source"
    source.mkdir()
    for name in ("repeat_selected_seeds.py", "run_experiments_v2.py", "experiment_protocol_v2.py", "models.py", "study.py"):
        (source/name).write_bytes((ROOT/"src"/name).read_bytes())
    cfg = json.loads((V1/"config.json").read_text())
    protocol = {"seeds": args.seeds, "reference_seed": 42, "model": MODEL_CFG,
                "loss": "weighted_bce", "lr": .001, "weight_decay": .0001,
                "epochs": 6, "patience": 2, "checkpoint_selection": "validation AUPRC",
                "threshold_selection": "maximum validation F1 with FPR <= .03; unrestricted also recorded",
                "scheduler": "ReduceLROnPlateau validation AUPRC, patience 1, factor .5",
                "fixed_final_model": str(V1/"weighted_bce.pt"),
                "final_checkpoint_sha256": sha256(V1/"weighted_bce.pt"),
                "purpose": "independent training-seed variability only; no seed selection and no test evaluation",
                "torch": torch.__version__, "numpy": np.__version__, "threads": args.threads,
                "manifest_sha256": sha256(ROOT/"data/processed/split_manifest.csv"),
                "preprocessing_sha256": sha256(ROOT/"data/processed/preprocessing_config.json")}
    assert protocol["manifest_sha256"] == cfg["manifest_sha256"]
    assert protocol["preprocessing_sha256"] == cfg["preprocessing_sha256"]
    write(out/"protocol.json", protocol)
    print(f"OUTPUT {out}", flush=True)
    train, val = CachedPatients("train"), CachedPatients("val")
    assert train.hours == cfg["train_hours"] and train.positives == cfg["train_positive_hours"]
    assert np.isclose(train.pos_weight, cfg["pos_weight"])
    protocol.update(pos_weight=train.pos_weight, train_hours=train.hours, val_hours=val.hours)
    write(out/"protocol.json", protocol)
    # Reuse the measured V1 weighted training duration for a conservative bounded estimate.
    prior_rows = json.loads((V1/"validation_comparison.json").read_text())
    prior = next(row for row in prior_rows if row["name"] == "weighted_bce")
    estimated = prior["training_seconds"]*len(args.seeds)*1.3
    write(out/"runtime_estimate.json", {"estimated_seconds": estimated, "reference": str(V1/"validation_comparison.json")})
    print(f"Estimated independent seed runs: {estimated:.0f}s", flush=True)
    if estimated > args.max_estimated_seconds:
        write(out/"status.json", {"status": "blocked_compute_budget"})
        return
    vl = DataLoader(val, batch_size=64, collate_fn=collate)
    baseline = {"seed": 42, "validation": prior["validation"], "best_epoch": prior["best_epoch"],
                "stopping_epoch": prior["epochs_trained"], "reused": True}
    rows = [baseline]
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        tr = DataLoader(train, batch_size=64, shuffle=True, generator=generator, collate_fn=collate)
        model = TCN(80, **MODEL_CFG)
        opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=.5, patience=1)
        directory = out/f"seed{seed}"
        directory.mkdir()
        best = -1.
        stalled = 0
        history = []
        started = time.perf_counter()
        for epoch in range(1, 7):
            model.train()
            loss_sum = 0.
            for x, y, mask in tr:
                opt.zero_grad()
                logit = model(x)[mask]
                loss = F.binary_cross_entropy_with_logits(logit, y[mask], pos_weight=logit.new_tensor(train.pos_weight))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                loss_sum += float(loss.detach())*int(mask.sum())
            y, score = predict(model, vl)
            points = threshold_options(y, score)
            vm = operating_metrics(y, score, points["selected"]["threshold"])
            history.append({"epoch": epoch, "mean_train_loss": loss_sum/train.hours,
                            "validation": vm, "operating_points": points})
            write(directory/"log.json", history)
            scheduler.step(vm["auprc"])
            print(f"independent seed {seed} epoch {epoch}: AP={vm['auprc']:.6f} F1={vm['f1']:.6f} ROC={vm['auroc']:.6f}", flush=True)
            if vm["auprc"] > best:
                best = vm["auprc"]
                best_epoch = epoch
                best_metrics, best_points = vm, points
                torch.save(model.state_dict(), directory/"best.pt")
                # Preserve optimizer and RNG state for future documented continuation.
                torch.save({"optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(),
                            "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
                            "epoch": epoch}, directory/"training_state.pt")
                np.savez_compressed(directory/"validation_predictions.npz", y=y, score=score)
                stalled = 0
            else:
                stalled += 1
            if stalled >= 2:
                break
        row = {"seed": seed, "validation": best_metrics, "operating_points": best_points,
               "best_epoch": best_epoch, "stopping_epoch": len(history),
               "training_seconds": time.perf_counter()-started,
               "checkpoint": str((directory/"best.pt").relative_to(out)), "test_evaluated": False}
        write(directory/"summary.json", row)
        rows.append(row)
        write(out/"seed_runs.json", rows)
    statistics = {key: {"mean": float(np.mean([r["validation"][key] for r in rows])),
                        "std": float(np.std([r["validation"][key] for r in rows], ddof=1))}
                  for key in ("accuracy", "precision", "recall", "f1", "auroc", "auprc")}
    write(out/"seed_statistics.json", {"statistics": statistics, "seeds": [r["seed"] for r in rows],
                                      "fixed_primary_seed": 42, "test_evaluated_for_repeats": False})
    assert protected_hashes() == frozen, "Prior artifact or UI changed"
    write(out/"status.json", {"status": "complete", "original_artifacts_and_ui_unchanged": True})
    print(json.dumps(statistics), flush=True)


if __name__ == "__main__":
    main()
