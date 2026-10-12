"""Validation-only threshold revision of frozen models; preserve the baseline.

All threshold decisions are persisted before revised test predictions are made.
This is an exploratory follow-up on an already reviewed evaluation period.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from cloud_anomaly.data import write_json
from cloud_anomaly.evaluation import metrics
from cloud_anomaly.calibration import validation_threshold

ROOT = Path(__file__).resolve().parents[1]


def calibrate_one(source_model: Path, dataset: Path, destination: Path):
    saved = json.loads((source_model / "result.json").read_text(encoding="utf-8"))
    info = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    data = np.load(dataset / "prepared.npz")
    length = info["window"]
    values = data["x_validation"]
    y = data["y_validation"][length - 1:]
    start = time.perf_counter()
    if saved["model"] == "lstm_autoencoder":
        import tensorflow as tf
        from cloud_anomaly.models import LSTMAutoencoder
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        model = object.__new__(LSTMAutoencoder)
        model.model = tf.keras.models.load_model(source_model / "model.keras", compile=False)
        x = np.lib.stride_tricks.sliding_window_view(values, length, axis=0).transpose(0, 2, 1)
    else:
        import joblib
        model = joblib.load(source_model / "model.joblib")
        x = values[length - 1:]
    scores = model.score(x).astype(np.float64)
    threshold, policy = validation_threshold(y, scores, saved["threshold"])
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": data["id_validation"][length - 1:], "label": y, "score": scores}).to_csv(
        destination / "validation_scores.csv", index=False)
    decision = {"threshold": threshold, "policy": policy,
        "baseline_threshold": saved["threshold"], "validation": metrics(y, scores, threshold),
        "baseline_validation": metrics(y, scores, saved["threshold"]),
        "calibration_seconds_including_load": time.perf_counter() - start,
        "validation_rows": len(y), "validation_positives": int(y.sum()),
        "decided_utc": datetime.now(timezone.utc).isoformat()}
    write_json(destination / "calibration.json", decision)
    print(f"Calibrated {saved['dataset']} {saved['model']} seed {saved['seed']}", flush=True)


def revise(source: Path):
    source = source.resolve()
    baseline = json.loads((source / "run.json").read_text(encoding="utf-8"))
    if baseline["status"] != "complete" or baseline.get("calibration_revision"):
        raise ValueError("Select a completed original baseline run")
    destination = ROOT / "artifacts/runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_calibrated")
    destination.mkdir(parents=True)
    revised = dict(baseline)
    revised.update({"run_id": destination.name, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_run": str(source.relative_to(ROOT)), "calibration_revision": True,
        "calibration_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurement_origin": "Frozen weights, scores and fit/resource measurements reused from baseline; only thresholds change.",
        "evaluation_status": "Exploratory follow-up: the evaluation period had already been reviewed; not an independent confirmation.",
        "new_model_fits": 0})
    write_json(destination / "run.json", revised)
    for filename in ["config.json", "data_manifest.json", "environment.txt"]:
        shutil.copy2(source / filename, destination / filename)
    shutil.copytree(source / "datasets", destination / "datasets")
    shutil.copytree(source / "models", destination / "models")
    entries = sorted(source.glob("models/*/*/*/result.json"))
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "2", "TF_ENABLE_ONEDNN_OPTS": "0",
                "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    print(f"REVISED RUN: {destination}", flush=True)
    try:
        # Phase one: choose every threshold without opening test prediction files.
        for entry in entries:
            model_dir = entry.parent
            dataset_name = model_dir.parent.parent.name
            target = destination / model_dir.relative_to(source)
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                "--source", str(model_dir), "--dataset", str(source / "datasets" / dataset_name),
                "--output", str(target)], env=env, check=True)
        write_json(destination / "thresholds_frozen.json", {"frozen_utc": datetime.now(timezone.utc).isoformat(), "decisions": len(entries)})
        # Phase two: apply the already frozen thresholds to unchanged test scores.
        for entry in entries:
            target = destination / entry.parent.relative_to(source)
            saved = json.loads(entry.read_text(encoding="utf-8"))
            decision = json.loads((target / "calibration.json").read_text(encoding="utf-8"))
            frame = pd.read_csv(entry.with_name("predictions.csv"), float_precision="round_trip")
            threshold = decision["threshold"]
            frame["prediction"] = (frame.score > threshold).astype(np.int8)
            frame.to_csv(target / "predictions.csv", index=False)
            result = dict(saved)
            result.update(metrics(frame.label.to_numpy(), frame.score.to_numpy(), threshold))
            result.update({"threshold": threshold, "threshold_policy": decision["policy"],
                "validation_f1": decision["validation"]["f1"], "baseline_f1": saved["f1"],
                "baseline_false_positive_rate": saved["false_positive_rate"], "calibration_revision": True,
                "calibration_seconds": decision["calibration_seconds_including_load"],
                "measurement_origin": "Fit time, latency and memory reused from frozen baseline model"})
            if saved["model"] != "lstm_autoencoder":
                import joblib
                model = joblib.load(target / "model.joblib")
                model.threshold = threshold
                joblib.dump(model, target / "model.joblib")
            write_json(target / "result.json", result)
        from cloud_anomaly.reporting import generate
        frame = generate(destination)
        revised.update({"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(), "completed_evaluations": len(frame)})
        write_json(destination / "run.json", revised)
        write_json(ROOT / "artifacts/latest.json", {"run_dir": str(destination.relative_to(ROOT))})
        before = pd.read_csv(source / "summary.csv")
        after = pd.read_csv(destination / "summary.csv")
        paired = before.merge(after, on=["family", "model_name"], suffixes=("_baseline", "_revised"))
        paired.to_csv(destination / "calibration_comparison.csv", index=False)
        print(paired[["family", "model_name", "f1_baseline", "f1_revised", "false_positive_rate_revised"]].to_string(index=False), flush=True)
    except Exception as error:
        revised.update({"status": "failed", "error": str(error)})
        write_json(destination / "run.json", revised)
        raise


def refresh_frozen(destination: Path):
    """Reapply saved decisions with lossless float parsing; never recalibrate."""
    destination = destination.resolve()
    run = json.loads((destination / "run.json").read_text(encoding="utf-8"))
    source = ROOT / run["baseline_run"]
    for path in sorted(destination.glob("models/*/*/*/result.json")):
        original_path = source / path.relative_to(destination)
        before = json.loads(original_path.read_text(encoding="utf-8"))
        decision = json.loads(path.with_name("calibration.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(original_path.with_name("predictions.csv"), float_precision="round_trip")
        frame["prediction"] = (frame.score > decision["threshold"]).astype(np.int8)
        frame.to_csv(path.with_name("predictions.csv"), index=False)
        result = json.loads(path.read_text(encoding="utf-8"))
        result.update(metrics(frame.label.to_numpy(), frame.score.to_numpy(), decision["threshold"]))
        write_json(path, result)
    from cloud_anomaly.reporting import generate
    generate(destination)
    before = pd.read_csv(source / "summary.csv")
    after = pd.read_csv(destination / "summary.csv")
    before.merge(after, on=["family", "model_name"], suffixes=("_baseline", "_revised")).to_csv(destination / "calibration_comparison.csv", index=False)
    run.setdefault("threshold_decision_script_sha256", run["calibration_script_sha256"])
    run["calibration_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    run["numeric_io_note"] = "Frozen thresholds reapplied using round-trip float parsing to preserve tied score ranks. No threshold, label or model was changed."
    write_json(destination / "run.json", run)
    print("Reapplied frozen thresholds with round-trip float parsing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--refresh", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.refresh:
        refresh_frozen(arguments.refresh)
    elif arguments.worker:
        calibrate_one(arguments.source, arguments.dataset, arguments.output)
    else:
        source = arguments.source
        if arguments.latest:
            source = ROOT / json.loads((ROOT / "artifacts/latest.json").read_text(encoding="utf-8"))["run_dir"]
        if source is None:
            parser.error("Use --source RUN_DIRECTORY or --latest")
        revise(source)

