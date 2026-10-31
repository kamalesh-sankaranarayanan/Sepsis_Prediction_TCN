import csv
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from streamlit.testing.v1 import AppTest

from src.inference import available_models, load_model, prepare_patient, read_statistics, score_patient
from src.study import ROOT, MODEL_CFG, CachedPatients, best_f1_threshold, collate, metrics, split_paths
from src.models import TCN


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    return load_model(available_models()[-1])


def patient_bytes(pid="p000005"):
    path = ROOT/"data/raw/training/training_setA"/(pid+".psv")
    return path.read_bytes()


@pytest.mark.parametrize("pid", ["p000005", "p000006", "p000010", "p000001"])
def test_matches_processed_and_checkpoint(pid, model):
    raw = patient_bytes(pid)
    prepared = prepare_patient(raw)
    manifest = pd.read_csv(ROOT/"data/processed/split_manifest.csv")
    split = manifest.loc[manifest.patient_id == pid, "split"].iloc[0]
    with np.load(ROOT/"data/processed"/split/(pid+".npz"), allow_pickle=False) as z:
        x = z["X"][:72]
        expected = np.concatenate([x[:, :40], 1-x[:, 40:80]], axis=1)
        np.testing.assert_array_equal(prepared.inputs, expected)
        np.testing.assert_array_equal(prepared.labels, z["y"][:72])
    scored = score_patient(model, prepared, .05)
    with torch.inference_mode():
        direct = torch.sigmoid(model(torch.tensor(expected)[None]))[0].numpy()
    np.testing.assert_allclose(scored["Model score"], direct, atol=1e-7)


def test_csv_order_and_labels_do_not_change_scores(model):
    frame = pd.read_csv(io.BytesIO(patient_bytes()), sep="|")
    original = prepare_patient(frame.to_csv(index=False).encode())
    frame.SepsisLabel = 1-frame.SepsisLabel
    changed = prepare_patient(frame[frame.columns[::-1]].to_csv(index=False).encode())
    unlabeled = prepare_patient(frame.drop(columns="SepsisLabel").to_csv(index=False).encode())
    np.testing.assert_array_equal(original.inputs, changed.inputs)
    np.testing.assert_array_equal(original.inputs, unlabeled.inputs)
    assert unlabeled.labels is None
    np.testing.assert_array_equal(score_patient(model, original, .05)["Model score"],
                                  score_patient(model, unlabeled, .05)["Model score"])


def test_future_rows_cannot_change_earlier_predictions(model):
    patient = prepare_patient(patient_bytes())
    before = score_patient(model, patient, .05)["Model score"].to_numpy()
    patient.inputs[10:, :40] += 10
    after = score_patient(model, patient, .05)["Model score"].to_numpy()
    np.testing.assert_allclose(before[:10], after[:10], atol=1e-7)


def test_leading_missingness_and_72_row_cap():
    names, means, scales = read_statistics()
    values = np.full((80, 40), np.nan)
    values[:, names.index("ICULOS")] = np.arange(3, 83)
    values[2, 0] = means[0]+scales[0]
    patient = prepare_patient(pd.DataFrame(values, columns=names).to_csv(index=False).encode())
    assert patient.total_rows == 80
    assert len(patient.inputs) == 72
    np.testing.assert_allclose(patient.inputs[:4, 0], [0, 0, 1, 1])
    np.testing.assert_array_equal(patient.inputs[:4, 40], [1, 1, 0, 1])
    assert patient.hours[0] == 3


@pytest.mark.parametrize("change,match", [
    (lambda f: f.drop(columns="HR"), "Missing"),
    (lambda f: f.assign(Unexpected=1), "Unexpected"),
    (lambda f: f.assign(ICULOS=1), "ICULOS"),
    (lambda f: f.assign(SepsisLabel=2), "SepsisLabel"),
])
def test_bad_schema(change, match):
    frame = pd.read_csv(io.BytesIO(patient_bytes()), sep="|")
    with pytest.raises(ValueError, match=match):
        prepare_patient(change(frame).to_csv(index=False).encode())


def test_duplicate_header_empty_ragged_and_invalid_values():
    content = patient_bytes().decode()
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_patient(content.replace("HR|", "O2Sat|", 1).encode())
    with pytest.raises(ValueError):
        prepare_patient(b"")
    with pytest.raises(ValueError, match="same number"):
        prepare_patient((content+"\n1|2\n").encode())
    frame = pd.read_csv(io.StringIO(content), sep="|")
    frame["HR"] = frame["HR"].astype(object)
    frame.loc[0, "HR"] = "broken"
    frame.loc[1, "HR"] = np.inf
    prepared = prepare_patient(frame.to_csv(index=False).encode())
    assert prepared.invalid_values == 2
    assert np.isfinite(prepared.inputs).all()


def test_padding_and_exact_threshold():
    batch = [(torch.zeros(2, 80), torch.tensor([0., 1.])),
             (torch.ones(1, 80), torch.tensor([0.]))]
    _, y, mask = collate(batch)
    assert int(mask.sum()) == 3
    assert y[mask].tolist() == [0, 1, 0]
    threshold = best_f1_threshold(np.array([0, 1, 1]), np.array([.001, .01, .02]))
    assert threshold == .01
    assert metrics([0, 1, 1], [.001, .01, .02], threshold)["f1"] == 1
    assert metrics([0, 0], [.1, .2], .5)["auroc"] is None


def test_loader_rejects_manifest_extras(tmp_path):
    base = tmp_path/"data/processed"
    base.mkdir(parents=True)
    pd.DataFrame({"patient_id": ["a", "b", "c"], "split": ["train", "val", "test"]}).to_csv(base/"split_manifest.csv", index=False)
    (base/"train").mkdir()
    (base/"train/a.npz").touch()
    assert len(split_paths("train", tmp_path)) == 1
    (base/"train/stray.npz").touch()
    with pytest.raises(ValueError, match="mismatch"):
        split_paths("train", tmp_path)


def test_missing_checkpoint_error():
    with pytest.raises(FileNotFoundError, match="missing"):
        load_model({"checkpoint": ROOT/"models/does_not_exist.pt"})


def test_capped_class_weight_counts_only_retained_hours(tmp_path):
    base = tmp_path/"data/processed"
    (base/"train").mkdir(parents=True)
    pd.DataFrame({"patient_id": ["a", "b", "c"], "split": ["train", "val", "test"]}).to_csv(base/"split_manifest.csv", index=False)
    (base/"preprocessing_config.json").write_text(json.dumps({"feature_names": list(range(40))}))
    y = np.zeros(73, dtype=np.float32)
    y[[0, 72]] = 1
    np.savez(base/"train/a.npz", X=np.zeros((73, 120), dtype=np.float32), y=y)
    dataset = CachedPatients("train", tmp_path)
    assert dataset.hours == 72
    assert dataset.positives == 1
    assert dataset.pos_weight == 71


def test_streamlit_app_loads():
    app = AppTest.from_file(str(ROOT/"app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Sepsis research explorer"
    assert len(app.selectbox) == 1
    assert len(app.error) == 0


def test_streamlit_real_prediction_path(monkeypatch):
    import streamlit
    monkeypatch.setattr(streamlit, "file_uploader", lambda *a, **kw: io.BytesIO(patient_bytes()))
    app = AppTest.from_file(str(ROOT/"app.py"), default_timeout=30).run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.error) == 0
    assert any(m.label == "Latest model score" for m in app.metric)
    assert len(app.dataframe) >= 2


def test_streamlit_invalid_upload(monkeypatch):
    import streamlit
    monkeypatch.setattr(streamlit, "file_uploader", lambda *a, **kw: io.BytesIO(b"bad,data\n1,2\n"))
    app = AppTest.from_file(str(ROOT/"app.py"), default_timeout=30).run()
    assert not app.exception
    assert any("Schema mismatch" in error.value for error in app.error)


def test_streamlit_missing_model(monkeypatch):
    import streamlit
    import src.inference as inference
    spec = available_models()[-1].copy()
    spec['checkpoint'] = ROOT/'models/does_not_exist.pt'
    monkeypatch.setattr(inference, 'available_models', lambda: [spec])
    monkeypatch.setattr(streamlit, "file_uploader", lambda *a, **kw: io.BytesIO(patient_bytes()))
    app = AppTest.from_file(str(ROOT/"app.py"), default_timeout=30).run()
    app.button[0].click().run()
    assert not app.exception
    assert any("checkpoint is missing" in error.value for error in app.error)
    assert not any(m.label == "Latest model score" for m in app.metric)


def test_focal_and_weighted_losses_use_valid_labels():
    # Script modules use sibling imports when run from src/.
    import sys
    sys.path.insert(0, str(ROOT/"src"))
    from run_experiments import objective
    logits = torch.tensor([-2., 0., 2.], requires_grad=True)
    labels = torch.tensor([0., 1., 1.])
    for kind in ["bce", "weighted_bce", "focal"]:
        loss = objective(logits, labels, kind, 71.847075)
        assert torch.isfinite(loss)
        loss.backward()
    weighted = objective(torch.zeros(2), torch.tensor([0., 1.]), "weighted_bce", 3.)
    np.testing.assert_allclose(weighted, 2*np.log(2), rtol=1e-6)
