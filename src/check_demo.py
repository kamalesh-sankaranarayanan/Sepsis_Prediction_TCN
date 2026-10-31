"""Read-only readiness check: python -B -m src.check_demo [--verify-originals]."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-originals", action="store_true")
    args = parser.parse_args(argv)
    missing = [name for name in ("numpy", "pandas", "torch", "sklearn", "streamlit")
               if importlib.util.find_spec(name) is None]
    if missing:
        print("NOT READY: missing dependencies: " + ", ".join(missing) +
              ". Installation requires your approval; nothing was installed.", file=sys.stderr)
        return 1
    try:
        import torch
        from .inference import available_models, load_model, prepare_patient, read_statistics, score_patient
        from .study import ROOT, sha256
        torch.set_num_threads(4)
        read_statistics()
        spec = available_models()[0]
        model = load_model(spec)
        result = {"status": "ready", "model": spec["name"], "threshold": spec["threshold"],
                  "ui": "app.py", "packages_installed": False, "artifacts_written": False}
        sample = ROOT/"data/raw/training/training_setA/p000005.psv"
        if sample.is_file():
            output = score_patient(model, prepare_patient(sample.read_bytes()), spec["threshold"])
            result["real_patient_inference_hours"] = len(output)
        if args.verify_originals:
            index = ROOT/"experiments"/spec["id"]/"original_artifact_hashes.json"
            if not index.is_file():
                raise ValueError("No preservation hash index exists for the selected model")
            hashes = json.loads(index.read_text())
            changed = [p for p, digest in hashes.items() if not (ROOT/p).is_file() or sha256(ROOT/p) != digest]
            if changed:
                raise ValueError("Original artifact verification failed: " + ", ".join(changed))
            result["original_files_verified"] = len(hashes)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(f"NOT READY: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
