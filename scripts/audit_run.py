"""Audit saved research evidence without a unit-test framework.

Checks cross-model row alignment and independently recomputes confusion counts
and F1. Run after an experiment; this does not train or choose a model.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    directory = args.run.resolve()
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if run["status"] != "complete":
        raise ValueError("Cannot verify an incomplete run")
    records = sorted(directory.glob("models/*/*/*/result.json"))
    reference = {}
    for path in records:
        saved = json.loads(path.read_text(encoding="utf-8"))
        prediction = pd.read_csv(path.with_name("predictions.csv"), float_precision="round_trip")
        y = prediction.label.to_numpy()
        p = prediction.prediction.to_numpy()
        if not np.isfinite(prediction.score).all():
            raise ValueError(f"Non-finite scores in {path}")
        expected = (prediction.score.to_numpy() > saved["threshold"]).astype(int)
        if not np.array_equal(p, expected):
            raise ValueError(f"Threshold mismatch in {path}")
        counts = {"tp": int(((y == 1) & (p == 1)).sum()), "tn": int(((y == 0) & (p == 0)).sum()),
                  "fp": int(((y == 0) & (p == 1)).sum()), "fn": int(((y == 1) & (p == 0)).sum())}
        if any(saved[key] != value for key, value in counts.items()) or len(y) != saved["test_rows"]:
            raise ValueError(f"Confusion counts mismatch in {path}")
        denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
        f1 = 2 * counts["tp"] / denominator if denominator else 0.0
        if saved["f1"] is not None and abs(f1 - saved["f1"]) > 1e-12:
            raise ValueError(f"F1 mismatch in {path}")
        if saved.get("calibration_revision"):
            from recalibrate_results import validation_threshold
            calibration = json.loads(path.with_name("calibration.json").read_text(encoding="utf-8"))
            validation = pd.read_csv(path.with_name("validation_scores.csv"), float_precision="round_trip")
            recalculated, _ = validation_threshold(validation.label.to_numpy(), validation.score.to_numpy(), calibration["baseline_threshold"])
            if recalculated != saved["threshold"]:
                raise ValueError(f"Validation threshold cannot be reproduced: {path}")
            baseline_dir = Path(__file__).resolve().parents[1] / run["baseline_run"]
            baseline_result = baseline_dir / path.relative_to(directory)
            baseline_prediction = pd.read_csv(baseline_result.with_name("predictions.csv"), float_precision="round_trip")
            if not np.allclose(prediction.score, baseline_prediction.score, rtol=0, atol=1e-14):
                raise ValueError(f"Frozen test scores changed: {path}")
            before = json.loads(baseline_result.read_text(encoding="utf-8"))
            if abs(saved["roc_auc"] - before["roc_auc"]) > 1e-10 or abs(saved["average_precision"] - before["average_precision"]) > 1e-10:
                raise ValueError(f"Score-ranking metrics changed: {path}")
        if run.get("experiment_revision"):
            from cloud_anomaly.calibration import validation_threshold
            from cloud_anomaly.evaluation import metrics
            selection = json.loads(path.with_name("selection.json").read_text(encoding="utf-8"))
            tuning = json.loads(path.with_name("tuning.json").read_text(encoding="utf-8"))
            validation = pd.read_csv(path.with_name("validation_scores.csv"), float_precision="round_trip")
            winner = max(tuning, key=lambda row: row["validation"]["f1"] if row["validation"]["f1"] is not None else 0)
            if winner["parameters"] != saved["parameters"] or selection["threshold"] != saved["threshold"]:
                raise ValueError(f"Candidate selection mismatch: {path}")
            threshold, _ = validation_threshold(validation.label, validation.score, winner["percentile_threshold"])
            measured = metrics(validation.label, validation.score, threshold)
            if threshold != saved["threshold"] or measured != selection["validation"]:
                raise ValueError(f"Validation decision cannot be reproduced: {path}")
        paired = prediction[["row_id", "label"]].to_numpy()
        key = saved["dataset"]
        if key in reference and not np.array_equal(paired, reference[key]):
            raise ValueError(f"Cross-model row alignment mismatch for {key}")
        reference[key] = paired
    for prepared in directory.glob("datasets/*/prepared.npz"):
        data = np.load(prepared)
        ids = [set(data[f"id_{partition}"].tolist()) for partition in ["train", "validation", "test"]]
        if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
            raise ValueError(f"Partition overlap in {prepared}")
        if not all(np.isfinite(data[f"x_{part}"]).all() for part in ["train", "validation", "test"]):
            raise ValueError(f"Non-finite prepared features in {prepared}")
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for file in sorted((root / "src/cloud_anomaly").glob("*.py")):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    if digest.hexdigest() != run["source_sha256"]:
        raise ValueError("Experimental source differs from the recorded source hash")
    if run.get("calibration_revision"):
        calibration_script = Path(__file__).with_name("recalibrate_results.py")
        if hashlib.sha256(calibration_script.read_bytes()).hexdigest() != run["calibration_script_sha256"]:
            raise ValueError("Calibration script differs from the recorded source hash")
    report = {"status": "passed", "evaluations": len(records), "datasets": len(reference),
              "checks": ["finite scores", "threshold decisions", "confusion counts", "F1 recalculation",
                         "cross-model endpoint alignment", "partition disjointness", "finite features", "source hash"]}
    (directory / "evidence_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

