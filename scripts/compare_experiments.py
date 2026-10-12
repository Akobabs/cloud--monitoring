"""Compare complete experiments on identical evaluation records, retaining losses."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cloud_anomaly.data import write_json

ROOT = Path(__file__).resolve().parents[1]


def compare(directory):
    directory = directory.resolve()
    run = json.loads((directory / "run.json").read_text())
    baseline = ROOT / run["comparison_run"]
    old_run = json.loads((baseline / "run.json").read_text())
    if run["status"] != "complete" or old_run["status"] != "complete":
        raise ValueError("Both experiments must be complete")
    config = json.loads((directory / "config.json").read_text())
    expected = {(dataset, model, seed)
                for dataset in ["google_cluster", *[f"smd_{m}" for m in config["smd_machines"]]]
                for model in config["models"] for seed in config["seeds"]
                if not (dataset == "google_cluster" and model == "lstm_autoencoder")}
    observed = set()
    selected_changes = 0
    for path in directory.glob("models/*/*/*/result.json"):
        saved = json.loads(path.read_text())
        observed.add((saved["dataset"], saved["model"], saved["seed"]))
        previous = baseline / path.relative_to(directory)
        old = json.loads(previous.read_text())
        selected_changes += old["parameters"] != saved["parameters"]
        before = pd.read_csv(previous.with_name("predictions.csv"), float_precision="round_trip")
        after = pd.read_csv(path.with_name("predictions.csv"), float_precision="round_trip")
        if not before[["row_id", "label"]].equals(after[["row_id", "label"]]):
            raise ValueError(f"Changed held-out records: {path}")
        with np.load(path.with_name("fit_indices.npz")) as new_fit, np.load(previous.with_name("fit_indices.npz")) as old_fit:
            if any(not np.array_equal(new_fit[key], old_fit[key]) for key in new_fit.files):
                raise ValueError(f"Changed fitting endpoints: {path}")
    if observed != expected:
        raise ValueError("Missing or unexpected evaluations")
    for metadata in directory.glob("datasets/*/dataset.json"):
        dataset = metadata.parent.name
        with np.load(metadata.with_name("prepared.npz")) as new, np.load(baseline / "datasets" / dataset / "prepared.npz") as old:
            for key in old.files:
                current = new[key]
                if key.startswith("x_") and dataset == "google_cluster":
                    current = current[:, :old[key].shape[1]]
                if not np.array_equal(current, old[key]):
                    raise ValueError(f"Changed original input: {dataset}/{key}")
    before = pd.read_csv(baseline / "summary.csv")
    after = pd.read_csv(directory / "summary.csv")
    comparison = before.merge(after, on=["family", "model_name"], suffixes=("_baseline", "_revised"), validate="one_to_one")
    for metric in ("f1", "precision", "recall", "false_positive_rate", "roc_auc", "average_precision", "train_seconds"):
        comparison[f"{metric}_change"] = comparison[f"{metric}_revised"] - comparison[f"{metric}_baseline"]
    comparison.to_csv(directory / "improvement_comparison.csv", index=False)
    write_json(directory / "comparison_audit.json", {"status": "passed", "evaluations": len(observed),
        "changed_candidate_selections": selected_changes, "baseline_run": old_run["run_id"],
        "identical_test_rows_and_labels": True, "identical_training_endpoints": True,
        "identical_original_prepared_features": True, "google_extra_features": 2})
    print(comparison[["family", "model_name", "f1_baseline", "f1_revised", "f1_change", "false_positive_rate_revised"]].to_string(index=False))
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, nargs="?")
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args()
    if args.latest:
        args.run = ROOT / json.loads((ROOT / "artifacts/latest.json").read_text())["run_dir"]
    if args.run is None:
        parser.error("Supply a run directory or --latest")
    compare(args.run)
