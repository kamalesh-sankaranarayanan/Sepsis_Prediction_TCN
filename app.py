"""Local research UI. Launch: python -m streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from src.inference import available_models, load_model, prepare_patient, score_patient

st.set_page_config(page_title="Sepsis | Research Explorer", page_icon="◌", layout="wide")
torch.set_num_threads(4)
st.markdown("""<style>
.stApp {background:#f4f7fb;color:#142b40;color-scheme:light}
.block-container{max-width:1320px;padding-top:2.5rem;padding-bottom:4rem}
h1,h2,h3{color:#142b40!important;letter-spacing:-.025em}
h1{font-size:clamp(2rem,4vw,3rem)!important;font-weight:800!important}
[data-testid="stMarkdownContainer"] p{font-size:1.03rem;line-height:1.65}
[data-testid="stCaptionContainer"] p{color:#475b70!important;font-size:.94rem!important;line-height:1.6}
[data-testid="stMetric"]{background:#fff;border:1px solid #cbd8e4;border-top:4px solid #087e8b;border-radius:14px;padding:20px;box-shadow:0 4px 16px #142b4008}
[data-testid="stMetricLabel"]{color:#475b70!important}
[data-testid="stMetricValue"]{color:#142b40!important;font-weight:750;font-variant-numeric:tabular-nums}
[data-testid="stSidebar"]{background:#e8eff5;border-right:1px solid #cbd8e4}
[data-testid="stSidebar"] [data-testid="stMetric"]{padding:16px}
[data-baseweb="tab-list"]{gap:24px;border-bottom:1px solid #cbd8e4;margin-bottom:20px}
[data-baseweb="tab"]{height:54px;color:#475b70;font-size:1rem;font-weight:650}
[data-baseweb="tab"][aria-selected="true"]{color:#076b76}
[data-testid="stFileUploader"]{background:#fff;border:2px dashed #9bb7ca;border-radius:14px;padding:16px}
[data-testid="stBaseButton-primary"]{min-height:48px;border-radius:10px;font-weight:700}
button:focus-visible,a:focus-visible{outline:3px solid #087e8b!important;outline-offset:3px}
.eyebrow{color:#076b76;font-weight:750;letter-spacing:.12em;font-size:.8rem;margin-bottom:8px}
.welcome{background:linear-gradient(120deg,#e4f3f3,#eef3fb);border:1px solid #bcd5df;border-radius:16px;padding:28px;margin:20px 0}
.welcome h3{margin:0 0 10px;font-size:1.35rem}.welcome p{color:#354e64;margin:0;line-height:1.7}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:24px}
.step{background:#fff;border:1px solid #cbd8e4;border-radius:12px;padding:18px;color:#354e64;line-height:1.6}
.step strong{display:block;color:#076b76;margin-bottom:6px}
.hero{background:linear-gradient(115deg,#102c44,#164a5b);border:1px solid #28586a;border-radius:20px;padding:30px 32px;margin:10px 0 24px;box-shadow:0 12px 32px #102c4412}
.hero .hero-title{font-size:1.6rem;font-weight:750;color:#fff;margin:12px 0}
.hero p{color:#d8e8f0;line-height:1.7;max-width:760px;margin:0}
.badge{display:inline-block;color:#d9fbf5;background:#ffffff12;border:1px solid #ffffff38;border-radius:30px;padding:5px 12px;font-size:.78rem;letter-spacing:.06em;font-weight:650}
.hero-facts{display:flex;gap:32px;flex-wrap:wrap;border-top:1px solid #ffffff26;margin-top:24px;padding-top:20px}
.hero-facts span{color:#d8e8f0;font-size:.86rem}.hero-facts b{display:block;color:#fff;font-size:1.45rem;margin-bottom:3px}
.brand{font-size:1.45rem;font-weight:800;color:#163c50;letter-spacing:-.04em;margin-bottom:3px}
.brand span{color:#087e8b}.section-label{color:#53687b;font-size:.75rem;letter-spacing:.12em;font-weight:700;margin:22px 0 10px}
.matrix{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:16px 0}.matrix-cell{border-radius:12px;padding:20px;background:#e6f3f1;border:1px solid #b8d9d4;color:#164e48}
.matrix-cell.error{background:#fff4e8;border-color:#e7c9a2;color:#79400e}.matrix-cell b{display:block;font-size:1.8rem;font-variant-numeric:tabular-nums}.matrix-cell small{display:block;margin-top:6px;font-size:.85rem}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:14px}
.footer{border-top:1px solid #cbd8e4;color:#53687b;font-size:.85rem;padding-top:20px;margin-top:32px}
@media(max-width:700px){.steps{grid-template-columns:1fr}.block-container{padding-left:1rem;padding-right:1rem}.welcome{padding:20px}}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">CLINICAL TIME SERIES · RESEARCH PROTOTYPE</div>', unsafe_allow_html=True)
st.title("Sepsis research explorer")
st.caption("A missingness-aware TCN study · First 72 available hourly rows · Local inference")
st.markdown('''<div class="hero"><span class="badge">FINAL YEAR PROJECT · RESEARCH DASHBOARD</span>
<div class="hero-title">Explore the signal. Understand the uncertainty.</div>
<p>Follow a patient’s hourly model scores and see how missing measurements are represented in a causal temporal convolutional network.</p>
<div class="hero-facts"><span><b>40,336</b>Patients in the study</span><span><b>72 rows</b>Maximum patient window</span>
<span><b>40 + 40</b>Clinical values + missingness indicators</span></div></div>''', unsafe_allow_html=True)
st.info("Research demonstration only. Scores are not calibrated clinical probabilities, diagnoses, or validated six-hour forecasts.")


@st.cache_resource
def cached_model(checkpoint, modified, model_cfg_json, checkpoint_hash, preprocessing_hash):
    import json
    return load_model({"checkpoint": Path(checkpoint), "model_cfg": json.loads(model_cfg_json),
                       "checkpoint_sha256": checkpoint_hash, "preprocessing_sha256": preprocessing_hash})


try:
    choices = available_models()
except Exception:
    st.error("Model metadata could not be read. Check that the saved metrics and experiment metadata are present.")
    st.stop()
with st.sidebar:
    st.markdown('<div class="brand">Sepsis<span> · Explorer</span></div>', unsafe_allow_html=True)
    st.caption("Temporal modelling research")
    st.markdown('<div class="section-label">MODEL CONFIGURATION</div>', unsafe_allow_html=True)
    chosen = st.selectbox("Model", choices, format_func=lambda s: s["name"])
    st.caption(chosen["kind"])
    st.metric("Saved study threshold", f"{chosen['threshold']:.6f}")
    st.caption("40 clinical values + 40 missingness indicators. Threshold fixed from validation, not adjusted on this patient.")
    st.divider()
    st.markdown("**Data handling**\n\nUploads stay in this local process. Patient data and scores are not saved by the app. Clear the upload to remove it from the view.")
    st.divider()
    st.markdown("**Reading the output**\n\nA threshold crossing is a model flag. A low score does not establish the absence of sepsis.")

upload_tab, study_tab, method_tab = st.tabs(["Patient explorer", "Research evaluation", "Study overview"])
with upload_tab:
    st.markdown('<div class="section-label">01 / PATIENT INPUT</div>', unsafe_allow_html=True)
    st.subheader("Explore one patient")
    st.write("Upload a CSV or pipe-delimited PSV containing the 40 PhysioNet features. SepsisLabel is optional and is used only as a reference overlay.")
    upload = st.file_uploader("Patient time series", type=["csv", "psv"], help="UTF-8, at most 2 MiB. ICULOS must advance by exactly one hour.")
    if upload is None:
        st.markdown('''<div class="welcome"><h3>Your patient timeline starts here</h3>
        <p>Upload one patient’s hourly measurements to explore model scores and data availability.</p>
        <div class="steps"><div class="step"><strong>01 · Upload a file</strong>Choose a CSV or PSV with the 40 required clinical features.</div>
        <div class="step"><strong>02 · Review the data</strong>Check the retained hours and how many measurements are available.</div>
        <div class="step"><strong>03 · Explore scores</strong>Generate a timeline using the saved model and study threshold.</div></div></div>''', unsafe_allow_html=True)
        st.caption("Ready for a patient file. No predictions are shown until a real checkpoint runs.")
    else:
        try:
            patient = prepare_patient(upload.getvalue())
        except (ValueError, KeyError, OSError) as exc:
            st.error(str(exc))
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows to score", len(patient.inputs))
            c2.metric("Trailing rows excluded", patient.total_rows-len(patient.inputs))
            c3.metric("Observed features", f"{patient.observed_fraction.mean():.1%}")
            if patient.total_rows > 72:
                st.warning(f"Only the first 72 of {patient.total_rows} hourly rows will be scored, matching the retained study.")
            if patient.invalid_values:
                st.warning(f"{patient.invalid_values} nonnumeric or infinite feature values were treated as missing, matching training preprocessing. Check the source data.")
            st.caption(f"{patient.missing_values} missing feature cells in the retained rows will use within-patient forward fill, then saved training means for leading gaps.")
            if st.button("Generate hourly scores", type="primary"):
                try:
                    import json
                    path = chosen["checkpoint"]
                    with st.spinner("Running the saved TCN on this patient..."):
                        model = cached_model(str(path), path.stat().st_mtime_ns, json.dumps(chosen["model_cfg"], sort_keys=True),
                                             chosen.get("checkpoint_sha256"), chosen.get("preprocessing_sha256"))
                        output = score_patient(model, patient, chosen["threshold"])
                except FileNotFoundError:
                    st.error("The selected checkpoint is missing. Restore its matching artifact before running inference.")
                except Exception:
                    st.error("Inference failed. Verify the checkpoint and saved preprocessing configuration. No substitute predictions were generated.")
                else:
                    st.markdown('<div class="section-label">02 / PREDICTION RESULTS</div>', unsafe_allow_html=True)
                    st.subheader("Patient score summary")
                    a, b, c = st.columns(3)
                    a.metric("Latest model score", f"{output['Model score'].iloc[-1]:.4f}")
                    b.metric("Peak model score", f"{output['Model score'].max():.4f}")
                    c.metric("Hours above study threshold", int(output['Above study threshold'].sum()))
                    chart_columns = ["Model score", "Study threshold"]
                    if "Supplied label" in output:
                        chart_columns.append("Supplied label")
                    st.subheader("Hourly model scores")
                    st.line_chart(output.set_index("ICULOS")[chart_columns], x_label="Recorded ICU hour", y_label="Score / supplied binary label", height=380,
                                  color=["#087e8b", "#b45309", "#7c3aed"][:len(chart_columns)])
                    st.caption("These are individual hourly predictions. Aggregate research metrics are shown in the separate Research evaluation tab.")
                    with st.expander("Measurement availability over time"):
                        st.caption("Fraction of the 40 features originally observed at each hour, before filling missing values.")
                        st.bar_chart(output.set_index("ICULOS")[["Observed feature fraction"]],
                                     color="#087e8b", height=230, x_label="Recorded ICU hour", y_label="Observed fraction")
                    st.subheader("Hourly results")
                    st.dataframe(output, hide_index=True, width="stretch", column_config={
                        "Model score": st.column_config.NumberColumn(format="%.4f"),
                        "Study threshold": st.column_config.NumberColumn(format="%.4f"),
                        "Observed feature fraction": st.column_config.ProgressColumn("Measurement availability", min_value=0, max_value=1, format="%.2f"),
                    })

with study_tab:
    st.subheader("Aggregate held-out evaluation")
    st.caption("Retrospective first-72-row study: 6,051 patients, 222,297 test hours, 3,060 positive hours (1.377%). These metrics do not describe the uploaded patient.")
    m = chosen["test"]
    cols = st.columns(3)
    for col, key in zip(cols, ["auprc", "auroc", "f1"]):
        col.metric({"auprc": "PR-AUC", "auroc": "ROC-AUC", "f1": "F1-score"}[key], f"{m[key]:.4f}")
    cols = st.columns(3)
    for col, key in zip(cols, ["precision", "recall", "accuracy"]):
        col.metric(key.capitalize(), f"{m[key]:.2%}")
    st.caption("Positive hours are rare: accuracy alone can be misleading. Read precision and recall alongside PR-AUC.")
    import json
    from src.study import ROOT
    history = json.loads((ROOT/"results/proposed_tcn_metrics.json").read_text())
    comparison = []
    for label, values in [("Historical baseline (unweighted)", history["TCN_Baseline"]),
                           ("Historical proposed", history["TCN_Proposed"]), (chosen["name"], m)]:
        if comparison and label == "Historical proposed TCN":
            continue
        comparison.append({"Model": label, **{k: values[k] for k in ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]}})
    st.subheader("Model comparison")
    st.dataframe(pd.DataFrame(comparison), hide_index=True, width="stretch", column_config={
        key: st.column_config.NumberColumn(label, format="%.4f")
        for key, label in [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                           ("f1", "F1-score"), ("auroc", "ROC-AUC"), ("auprc", "PR-AUC")]
    })
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Prediction outcomes")
        st.caption("Retained test hours · Fixed validation-selected threshold")
        st.markdown(f'''<div class="matrix">
        <div class="matrix-cell"><b>{m['tn']:,}</b>True negatives<small>Negative label · Below threshold</small></div>
        <div class="matrix-cell error"><b>{m['fp']:,}</b>False positives<small>Negative label · Threshold crossed</small></div>
        <div class="matrix-cell error"><b>{m['fn']:,}</b>False negatives<small>Positive label · Below threshold</small></div>
        <div class="matrix-cell"><b>{m['tp']:,}</b>True positives<small>Positive label · Threshold crossed</small></div>
        </div>''', unsafe_allow_html=True)
    with right:
        st.subheader("Understanding the metrics")
        with st.container(border=True):
            st.markdown("**Precision** — Of the flagged hours, how many have a positive label?\n\n"
                        "**Recall** — Of all positive-labelled hours, how many were flagged?\n\n"
                        "**PR-AUC** — Summarises the precision–recall trade-off across thresholds.\n\n"
                        "**F1-score** — Balances precision and recall at the selected threshold.")
    st.markdown("""**Method.** Training-only saved statistics, patient-level splits, causal dilated convolutions, and original missingness indicators. New candidates are selected using validation AUPRC; their thresholds maximize validation F1.

**Limits.** The displayed test metrics use one selected checkpoint; additional training seeds were assessed on validation data. The 72-row cap excludes later positive labels. The historical weight used uncapped labels; new weighted runs correct this. The historical test split has been used before, so this is not independent external validation. No clinical calibration, onset-specific lead-time validation, or causal SHAP interpretation is established. XGBoost remains unchecked in this environment.""")

with method_tab:
    st.subheader("From clinical measurements to hourly scores")
    st.caption("The implemented research workflow, explained for the project demonstration.")
    st.markdown('''<div class="steps">
    <div class="step"><strong>01 · Prepare</strong>Preserve original missingness, forward-fill earlier measurements, then fill leading gaps with saved training means.</div>
    <div class="step"><strong>02 · Represent</strong>Combine 40 scaled clinical features with 40 indicators: 1 means originally missing, 0 means observed.</div>
    <div class="step"><strong>03 · Predict</strong>Run causal dilated convolutions and compare each hourly score with the saved validation-selected threshold.</div>
    </div>''', unsafe_allow_html=True)
    st.subheader("Study design")
    a, b, c = st.columns(3)
    a.metric("Training patients", "28,235")
    b.metric("Validation patients", "6,050")
    c.metric("Test patients", "6,051")
    with st.container(border=True):
        st.markdown("**Patient-level separation.** Patients do not overlap between splits. Preprocessing statistics come only from training data.\n\n"
                    "**Temporal model.** Four 64-channel TCN blocks use dilations of 1, 2, 4 and 8. Causal convolutions use current and earlier inputs.\n\n"
                    "**Study window.** Up to the first 72 available rows are scored. The first recorded row may occur after ICU hour 1.\n\n"
                    "**Prediction target.** Original hourly sepsis labels are retained. No new six-hour-ahead target is constructed.")
    st.info("Elapsed-time features are stored in the research data but are not inputs to this model. Temporal attention is a future extension.")

st.markdown('<div class="footer">Sepsis Research Explorer · Missingness-aware TCN · Local research demonstration</div>', unsafe_allow_html=True)
