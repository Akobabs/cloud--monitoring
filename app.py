"""Read-only local frontend for completed and in-progress research runs."""
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Cloud Anomaly Research", page_icon="◈", layout="wide")
st.markdown("""<style>
    .stApp {background:#f6f8fc;}
    h1,h2,h3 {color:#172844;}
    [data-testid="stMetric"] {background:white; padding:18px; border:1px solid #dde5ef; border-radius:10px;}
    .eyebrow {color:#2563a6;font-size:12px;letter-spacing:2px;font-weight:700;}
    .stMainBlockContainer {padding-top:4.2rem; max-width:1440px;}
    [data-testid="stMetricValue"] {font-size:1.65rem;}
    [data-testid="stSidebar"] {background:#eef2f8;}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">OFFLINE EXPERIMENTS / CLOUD TELEMETRY</div>', unsafe_allow_html=True)
st.title("Cloud anomaly research")
st.caption("Compare detection quality and computing cost across four machine learning approaches.")

runs = sorted((ROOT / "artifacts/runs").glob("*/run.json"), reverse=True)
completed = [p for p in runs if json.loads(p.read_text(encoding="utf-8")).get("status") == "complete"]
default_run = runs.index(completed[0]) if completed else 0
if not runs:
    st.info("No experiments yet. Run run_project.bat to download data and produce results.")
    st.code("run_project.bat quick", language="batch")
    st.stop()
with st.sidebar:
    st.subheader("Experiment explorer")
    chosen = st.selectbox("Experiment", runs, index=default_run, format_func=lambda p: p.parent.name)
    page = st.radio("View", ["Overview", "Model comparison", "Prediction explorer", "Dataset details", "Reproducibility"])
    st.divider()
    st.caption("Google: failure proxy classification\n\nSMD: adapted chronological anomaly benchmark")
    st.caption("Read-only results. No production alerts or live cloud connections.")

run_dir = chosen.parent
run = json.loads(chosen.read_text(encoding="utf-8"))
if run.get("calibration_revision"):
    st.caption("Validation-calibrated thresholds · exploratory follow-up · frozen baseline models and resource measurements")
    st.caption("Calibration raises false alarms and does not improve every model. Compare the preserved baseline before choosing an operating point.")
if run.get("experiment_revision"):
    st.caption("Retrained experiment: resource utilisation features and validation-calibrated candidate selection")
    st.caption("Same evaluation period as the baseline; exploratory findings requiring independent confirmation.")
config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
files = list(run_dir.glob("models/*/*/*/result.json"))
if not files:
    st.warning(f"Run status: {run['status']}. Results appear after the first model completes.")
    if run.get("error"):
        st.error(run["error"])
    st.stop()
metrics = pd.DataFrame([json.loads(p.read_text(encoding="utf-8")) for p in files])
metadata = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in run_dir.glob("datasets/*/dataset.json")}

if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Completed evaluations", len(metrics))
    c2.metric("Data sources", 2)
    c3.metric("Algorithms", metrics.model.nunique())
    c4.metric("Run status", run["status"].capitalize())
    st.subheader("Detection performance")
    comparison_path = run_dir / "improvement_comparison.csv"
    if run.get("experiment_revision") and comparison_path.exists():
        st.markdown("**Before and after retraining**")
        comparison = pd.read_csv(comparison_path)
        displayed = comparison[["family", "model_name", "f1_baseline", "f1_revised", "f1_change",
                                "false_positive_rate_baseline", "false_positive_rate_revised"]].copy()
        displayed["family"] = displayed.family.map({"Google failure proxy": "Google", "SMD adapted benchmark": "SMD"})
        displayed = displayed.rename(columns={"family": "Dataset", "model_name": "Model",
            "f1_baseline": "F1 before", "f1_revised": "F1 after", "f1_change": "F1 change",
            "false_positive_rate_baseline": "FPR before", "false_positive_rate_revised": "FPR after"})
        st.dataframe(displayed.style.format(precision=4), hide_index=True, use_container_width=True)
        st.caption("FPR is the fraction of normal-labelled records flagged as positive. Lower is better.")
        st.download_button("Download before-and-after comparison", comparison.to_csv(index=False),
                           "improvement_comparison.csv", "text/csv")
    st.caption("SMD bars average the selected machines and seeds. Google and SMD measure different targets and must be interpreted separately.")
    left, right = st.columns(2)
    for col, family, title in [(left, "smd", "SMD adapted benchmark"), (right, "google", "Google failure proxy")]:
        with col:
            st.markdown(f"**{title}**")
            frame = metrics[metrics.dataset.str.startswith(family)]
            if not frame.empty:
                figure = run_dir / "figures" / f"{family}_comparison.png"
                if figure.exists():
                    st.image(str(figure), use_container_width=True)
                else:
                    st.dataframe(frame.groupby("model_name")[["precision", "recall", "f1"]].mean())
    st.info("Google LSTM: not applicable. After deduplication, tasks have at most four observations; the research model requires 30.")
    st.subheader("Experiment protocol")
    st.write("Chronological partitions · training-only preprocessing · validation-selected thresholds · CPU execution")
    if run.get("calibration_revision"):
        st.caption("The reviewed test period is reused. Baseline results are retained; an independent evaluation is still needed.")
    st.caption("No unit-test suite. Verification consists of schema checks, full pipeline execution, saved prediction checks, and manual interface review.")
elif page == "Model comparison":
    dataset = st.selectbox("Dataset", list(metadata), format_func=lambda key: metadata[key]["display_name"])
    frame = metrics[metrics.dataset == dataset]
    st.subheader(metadata[dataset]["display_name"])
    selected_columns = ["precision", "recall", "f1", "roc_auc", "average_precision", "false_positive_rate"]
    table = frame.groupby("model_name")[selected_columns].mean().sort_values("f1", ascending=False)
    st.caption("Mean across completed seeds. Positive means anomaly for SMD and failure-labelled record for Google.")
    st.dataframe(table.style.format("{:.4f}"), use_container_width=True)
    st.subheader("Computing cost")
    cost = frame.groupby("model_name")[["train_seconds", "latency_ms_per_observation", "peak_rss_mb"]].mean()
    st.dataframe(cost.style.format("{:.3f}"), use_container_width=True)
    st.caption("Training excludes preprocessing and model construction. Latency is warm, amortised batch scoring. Memory is sampled total process RSS during fitting.")
    if run.get("calibration_revision"):
        st.caption("Resource measurements are retained from the frozen baseline models. Only decision thresholds were revised.")
        baseline = pd.read_csv(run_dir / "calibration_comparison.csv")
        family = "SMD adapted benchmark" if dataset.startswith("smd_") else "Google failure proxy"
        with st.expander("Baseline and revised family F1"):
            st.dataframe(baseline.loc[baseline.family == family, ["model_name", "f1_baseline", "f1_revised", "false_positive_rate_revised"]], hide_index=True)
    if run.get("experiment_revision"):
        st.caption("All models were refitted in this run; resource measurements were collected again.")
    st.download_button("Download model results", frame.to_csv(index=False), f"{dataset}_results.csv", "text/csv")
    for suffix in ["confusion", "curves"]:
        figure = run_dir / "figures" / f"{dataset}_{suffix}.png"
        if figure.exists():
            st.image(str(figure), use_container_width=True)
elif page == "Prediction explorer":
    dataset = st.selectbox("Dataset", sorted(metrics.dataset.unique()))
    frame = metrics[metrics.dataset == dataset]
    model = st.selectbox("Algorithm", sorted(frame.model_name.unique()))
    frame = frame[frame.model_name == model]
    seed = st.selectbox("Seed", sorted(frame.seed.unique()))
    record = frame[frame.seed == seed].iloc[0]
    predictions = pd.read_csv(run_dir / "models" / dataset / record.model / str(seed) / "predictions.csv")
    st.subheader(f"{model} predictions")
    a, b, c = st.columns(3)
    a.metric("Actual positives", int(predictions.label.sum()))
    b.metric("Raised detections", int(predictions.prediction.sum()))
    c.metric("Missed positives", int(((predictions.label == 1) & (predictions.prediction == 0)).sum()))
    n = st.slider("Observations to display", 100, min(3000, len(predictions)), min(1000, len(predictions)), 100)
    st.line_chart(pd.DataFrame({"Anomaly score": predictions.score.iloc[:n].to_numpy(),
                                "Decision threshold": [record.threshold] * n}), color=["#2463a8", "#e6a23c"])
    st.caption("Horizontal axis is held-out sequence position. Google positions are ordered records from separate tasks, not a continuous task signal.")
    errors_only = st.checkbox("Show misclassified records only", value=True)
    shown = predictions[predictions.label != predictions.prediction] if errors_only else predictions
    st.dataframe(shown.head(500), use_container_width=True, hide_index=True)
    st.download_button("Download all predictions", predictions.to_csv(index=False), f"{dataset}_{record.model}_{seed}.csv", "text/csv")
elif page == "Dataset details":
    dataset = st.selectbox("Dataset", list(metadata), format_func=lambda key: metadata[key]["display_name"])
    info = metadata[dataset]
    st.subheader(info["display_name"])
    st.write(info["protocol"])
    st.info(info["label_semantics"])
    st.dataframe(pd.DataFrame(info["partitions"]).T, use_container_width=True)
    st.caption("Counts above precede the 29-record sequence warm-up on SMD. All models evaluate the same eligible endpoints.")
    st.warning(info["limitations"])
    with st.expander("Feature dictionary and preparation metadata"):
        st.json(info)
    manifest = json.loads((run_dir / "data_manifest.json").read_text(encoding="utf-8"))
    st.download_button("Download dataset manifest", json.dumps(manifest, indent=2), "data_manifest.json", "application/json")
else:
    st.subheader("Reproduce this experiment")
    st.code(f'python -m cloud_anomaly.cli run --config "{run_dir.relative_to(ROOT) / "config.json"}"', language="bash")
    st.json({"run_id": run["run_id"], "status": run["status"], "hardware": run["hardware"], "versions": run["versions"]})
    with st.expander("Exact configuration", expanded=True):
        st.json(config)
    st.download_button("Download configuration", json.dumps(config, indent=2), "config.json", "application/json")
    st.caption("Changing the code, dependencies, hardware or input files can change measurements. Repeated seeds are recorded rather than claimed to remove all numerical variation.")
