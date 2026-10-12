"""Isolated model process; called sequentially by the experiment runner."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import write_json
from .calibration import validation_threshold
from .evaluation import MemoryMonitor, benchmark_score, metrics
from .models import LSTMAutoencoder, NAMES, TabularDetector


def windows(x, length):
    return np.lib.stride_tricks.sliding_window_view(x, length, axis=0).transpose(0, 2, 1)


def run(dataset_dir: Path, output: Path, config: dict, kind: str, seed: int):
    output.mkdir(parents=True, exist_ok=True)
    prepared = np.load(dataset_dir / "prepared.npz")
    metadata = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    length = metadata["window"]
    if kind == "lstm_autoencoder" and not metadata["temporal"]:
        raise ValueError("LSTM cannot train on unrelated Google tasks")
    raw_x = {part: prepared[f"x_{part}"] for part in ("train", "validation", "test")}
    raw_y = {part: prepared[f"y_{part}"] for part in raw_x}
    aligned_y = {part: y[length - 1:] for part, y in raw_y.items()}
    normal_windows = {part: np.lib.stride_tricks.sliding_window_view(y, length).sum(axis=1) == 0
                      for part, y in raw_y.items()}
    rng = np.random.default_rng(seed)
    eligible = np.arange(len(aligned_y["train"]))
    chosen = np.sort(rng.choice(eligible, min(len(eligible), config["max_fit_samples"]), replace=False))
    supervised = kind == "random_forest"
    fit_indices = chosen if supervised else chosen[normal_windows["train"][chosen]]
    if len(fit_indices) < 32:
        raise ValueError("Insufficient normal training windows")
    if supervised and len(np.unique(aligned_y["train"][fit_indices])) < 2:
        raise ValueError("The common training sample lacks a positive class")
    # Models share identical candidate endpoints and held-out evaluation endpoints.
    x = ({part: windows(values, length) for part, values in raw_x.items()}
         if kind == "lstm_autoencoder" else {part: values[length - 1:] for part, values in raw_x.items()})
    x_fit = np.ascontiguousarray(x["train"][fit_indices])
    y_fit = aligned_y["train"][fit_indices] if supervised else None
    valid_normal_mask = normal_windows["validation"]
    if valid_normal_mask.sum() < 20:
        raise ValueError("Too few normal validation windows to calibrate a threshold")
    x_validation_normal = np.ascontiguousarray(x["validation"][valid_normal_mask])
    # Bound only the reconstruction-loss validation set, not threshold calibration or metric evaluation.
    x_early_stop = x_validation_normal[:min(2000, len(x_validation_normal))]
    np.savez_compressed(output / "fit_indices.npz", common_endpoints=chosen + length - 1,
                        fitted_endpoints=fit_indices + length - 1)
    candidates = config["models"][kind]
    has_validation_positives = bool(aligned_y["validation"].sum())
    if not has_validation_positives:
        candidates = candidates[:1]  # Predeclared first candidate, not selected using the held-out test.
    best_quality = -np.inf
    selected = None
    tuning_records = []
    tuning_start = time.perf_counter()
    for candidate in candidates:
        print(f"{metadata['dataset']} seed={seed} {kind}: {candidate}", flush=True)
        model = (LSTMAutoencoder(candidate, seed, config["threads"], x_fit.shape[1:])
                 if kind == "lstm_autoencoder" else TabularDetector(kind, candidate, seed, config["threads"]))
        with MemoryMonitor() as memory:
            start = time.perf_counter()
            model.fit(x_fit, y_fit, validation=x_early_stop)
            elapsed = time.perf_counter() - start
        scores = np.asarray(model.score(x["validation"]), dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite validation scores")
        threshold = float(np.quantile(scores[valid_normal_mask], 1 - config["normal_fpr_target"]))
        percentile_threshold = threshold
        policy = f"{100 * (1 - config['normal_fpr_target']):g}th percentile of normal validation scores"
        if config.get("threshold_policy", "normal_percentile") == "validation_f1":
            threshold, policy = validation_threshold(aligned_y["validation"], scores, threshold)
        validation_metrics = metrics(aligned_y["validation"], scores, threshold)
        quality = validation_metrics["f1"] if has_validation_positives else 0.0
        record = {"parameters": candidate, "threshold": threshold,
                  "threshold_policy": policy, "percentile_threshold": percentile_threshold,
                  "validation": validation_metrics, "fit_seconds": elapsed,
                  "peak_rss_mb": memory.peak / 1024**2,
                  "incremental_peak_rss_mb": (memory.peak - memory.baseline) / 1024**2,
                  "lifetime_peak_rss_mb": memory.lifetime_peak / 1024**2}
        tuning_records.append(record)
        if quality > best_quality:
            best_quality = quality
            model.threshold = threshold
            model.save(output / "model")
            selected = record.copy()
            pd.DataFrame({"row_id": prepared["id_validation"][length - 1:],
                          "label": aligned_y["validation"], "score": scores}).to_csv(
                              output / "validation_scores.csv", index=False)
            if kind == "lstm_autoencoder":
                write_json(output / "training_history.json", model.history)
        del model
        gc.collect()
    tuning_seconds = time.perf_counter() - tuning_start
    # Persist selection before scoring the test partition.
    write_json(output / "selection.json", {"parameters": selected["parameters"],
        "threshold": selected["threshold"], "threshold_policy": selected["threshold_policy"],
        "validation": selected["validation"]})
    # Reload the selected frozen model; no test-based model or threshold choice.
    if kind == "lstm_autoencoder":
        import tensorflow as tf
        model = object.__new__(LSTMAutoencoder)
        model.model = tf.keras.models.load_model(output / "model.keras", compile=False)
        model.threshold = selected["threshold"]
    else:
        import joblib
        model = joblib.load(output / "model.joblib")
    test_scores, latency, timings = benchmark_score(model, x["test"])
    test_scores = np.asarray(test_scores, dtype=np.float64)
    if not np.isfinite(test_scores).all():
        raise ValueError("Non-finite test scores")
    test_y = aligned_y["test"]
    threshold = selected["threshold"]
    result = {
        "dataset": metadata["dataset"], "model": kind, "model_name": NAMES[kind],
        "seed": seed, "status": "complete", "parameters": selected["parameters"],
        "threshold": threshold, "threshold_policy": selected["threshold_policy"],
        "selection_policy": "Maximum validation F1" if has_validation_positives else "First predefined candidate: validation contains no positives",
        "fit_rows": len(fit_indices), "fit_normal": int(normal_windows["train"][fit_indices].sum()),
        "test_rows": len(test_y), "feature_count": metadata["feature_count"],
        "train_seconds": selected["fit_seconds"], "tuning_seconds": tuning_seconds,
        "latency_ms_per_observation": latency, "inference_repetition_seconds": timings,
        "peak_rss_mb": selected["peak_rss_mb"], "incremental_peak_rss_mb": selected["incremental_peak_rss_mb"],
        "lifetime_peak_rss_mb": selected["lifetime_peak_rss_mb"],
        "validation_f1": selected["validation"]["f1"], **metrics(test_y, test_scores, threshold),
    }
    prediction = (test_scores > threshold).astype(np.int8)
    pd.DataFrame({"row_id": prepared["id_test"][length - 1:], "label": test_y,
                  "score": test_scores, "prediction": prediction}).to_csv(output / "predictions.csv", index=False)
    write_json(output / "tuning.json", tuning_records)
    write_json(output / "result.json", result)
    f1_text = "not applicable" if result["f1"] is None else f"{result['f1']:.4f}"
    print(f"Completed: F1={f1_text}, fit={result['train_seconds']:.2f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run(args.dataset, args.output, config, args.model, args.seed)


if __name__ == "__main__":
    main()

