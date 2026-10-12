"""Dataset acquisition, provenance, and leakage-aware partitioning.

SMD uses a declared supervised adaptation of its labelled test sequence.
Google is a tabular failure-label study: its sparse task traces are not LSTM data.
"""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

SMD_REVISION = "7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4"
SMD_URL = f"https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/{SMD_REVISION}"
GOOGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/derrickmwiti/google-2019-cluster-sample?datasetVersionNumber=1"
GOOGLE_FEATURES = [
    "average_cpu", "average_memory", "maximum_cpu", "maximum_memory",
    "sample_cpu", "requested_cpu", "requested_memory", "assigned_memory",
    "page_cache_memory", "cycles_per_instruction", "memory_accesses_per_instruction",
]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {destination.name}", flush=True)
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
    temporary.replace(destination)


def acquire(root: Path, config: dict) -> dict:
    google_dir = root / "data/raw/google"
    csv = google_dir / "borg_traces_data.csv"
    if not csv.exists():
        archive = google_dir / "google-sample.zip"
        download(GOOGLE_URL, archive)
        if not zipfile.is_zipfile(archive):
            raise ValueError("Kaggle did not return a ZIP. Download version 1 manually into data/raw/google.")
        with zipfile.ZipFile(archive) as zipped:
            # Extract only the expected member; do not extract arbitrary archive paths.
            with zipped.open("borg_traces_data.csv") as source, csv.open("wb") as target:
                shutil.copyfileobj(source, target)
    paths = [(csv, GOOGLE_URL)]
    for machine in config["smd_machines"]:
        if not __import__("re").fullmatch(r"machine-[123]-[1-9][0-9]*", machine):
            raise ValueError(f"Invalid SMD machine name: {machine}")
        for part in ("train", "test", "test_label"):
            destination = root / "data/raw/smd" / part / f"{machine}.txt"
            url = f"{SMD_URL}/ServerMachineDataset/{part}/{machine}.txt"
            download(url, destination)
            paths.append((destination, url))
    manifest = {
        "google_version": 1, "smd_revision": SMD_REVISION,
        "google_license": "Kaggle listing: CC BY-NC-SA 4.0; upstream Google trace: CC BY 4.0",
        "smd_license_note": "Repository contains MIT LICENSE; no separate dataset-specific licence was identified.",
        "files": [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                   "sha256": sha256(p), "source": url} for p, url in paths],
    }
    return manifest


def _partition_counts(y: np.ndarray) -> dict:
    return {"rows": len(y), "normal": int((y == 0).sum()), "positive": int((y == 1).sum())}


def _process_and_save(parts: list[pd.DataFrame], labels: list[np.ndarray],
                      ids: list[np.ndarray], directory: Path, metadata: dict,
                      config: dict, temporal: bool) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    train = parts[0].replace([np.inf, -np.inf], np.nan)
    retained = train.columns[train.notna().any()].tolist()
    if not retained:
        raise ValueError("No observed training features")
    parts = [part[retained].replace([np.inf, -np.inf], np.nan) for part in parts]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.fit(parts[0])
    arrays = [imputer.transform(part) for part in parts]
    features = retained[:]
    if temporal:
        features += [f"{f}_mean_{config['rolling_window']}" for f in retained]
        features += [f"{f}_delta" for f in retained]
        engineered = []
        for values in arrays:
            frame = pd.DataFrame(values)
            rolling = frame.rolling(config["rolling_window"], min_periods=1).mean()
            # Every window is causal and starts again at a partition boundary.
            delta = frame.diff().fillna(0)
            engineered.append(np.concatenate([values, rolling.to_numpy(), delta.to_numpy()], axis=1))
        arrays = engineered
    scaler = StandardScaler().fit(arrays[0])
    arrays = [scaler.transform(values).astype(np.float32) for values in arrays]
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("Non-finite values remain after preprocessing")
    if len(np.unique(labels[0])) != 2:
        raise ValueError("Supervised training requires both labels in the training partition")
    window = config["sequence_length"] if temporal else 1
    if any(len(y) < window + 10 for y in labels):
        raise ValueError("Insufficient observations after partitioning")
    metadata.update({
        "features": features, "raw_features": retained, "feature_count": len(features),
        "temporal": temporal, "window": window,
        "partitions": {name: _partition_counts(y) for name, y in zip(("train", "validation", "test"), labels)},
        "eligible_partitions": {name: _partition_counts(y[window-1:]) for name, y in zip(("train", "validation", "test"), labels)},
        "raw_missing_values": {name: {col: int(val) for col, val in part.isna().sum().items()}
                               for name, part in zip(("train", "validation", "test"), parts)},
    })
    payload = {}
    for name, array, y, row_ids in zip(("train", "validation", "test"), arrays, labels, ids):
        payload[f"x_{name}"] = array
        payload[f"y_{name}"] = np.asarray(y, dtype=np.int8)
        payload[f"id_{name}"] = np.asarray(row_ids, dtype=np.int64)
    np.savez_compressed(directory / "prepared.npz", **payload)
    joblib.dump({"imputer": imputer, "scaler": scaler, "raw_features": retained,
                 "features": features, "temporal": temporal, "rolling_window": config["rolling_window"]},
                directory / "preprocessor.joblib")
    write_json(directory / "dataset.json", metadata)
    return metadata


def prepare_smd(root: Path, directory: Path, machine: str, config: dict) -> dict:
    base = root / "data/raw/smd"
    values = np.loadtxt(base / "test" / f"{machine}.txt", delimiter=",", dtype=np.float32)
    # Validate before casting: integer conversion can truncate fractional labels.
    labels = np.loadtxt(base / "test_label" / f"{machine}.txt", dtype=np.float64, ndmin=1)
    if (values.ndim != 2 or labels.ndim != 1 or len(values) != len(labels)
            or not np.isin(labels, [0, 1]).all() or values.shape[1] != 38):
        raise ValueError(f"Invalid SMD schema or label alignment: {machine}")
    labels = labels.astype(np.int8)
    cuts = [int(len(values) * config["split_fractions"][0]),
            int(len(values) * sum(config["split_fractions"][:2]))]
    frame = pd.DataFrame(values, columns=[f"metric_{i:02d}" for i in range(1, 39)])
    bounds = [0, *cuts, len(frame)]
    metadata = {
        "dataset": f"smd_{machine}", "display_name": f"SMD {machine}",
        "source_rows": len(frame), "source_features": 38,
        "label_semantics": "Published point anomaly label: 0 normal, 1 anomaly",
        "protocol": "Adapted supervised benchmark: published labelled test sequence split chronologically 60/20/20. Official normal-only training file downloaded for provenance but not fitted.",
        "split_bounds_exclusive": bounds,
        "time_unit": "Ordered observation index; sampling interval and named metric units not supplied in these files.",
        "limitations": "Not directly comparable with published SMD official-split scores. Three predefined machines do not represent all 28 machines.",
    }
    return _process_and_save([frame.iloc[a:b].copy() for a, b in zip(bounds, bounds[1:])],
        list(np.split(labels, cuts)), list(np.split(np.arange(len(frame)), cuts)), directory, metadata, config, True)


def _resource(text) -> dict:
    if pd.isna(text):
        return {}
    value = ast.literal_eval(text)
    if not isinstance(value, dict):
        raise ValueError("Resource field must contain a dictionary")
    return value


def prepare_google(root: Path, directory: Path, config: dict) -> dict:
    raw = pd.read_csv(root / "data/raw/google/borg_traces_data.csv")
    required = {"cluster", "collection_id", "instance_index", "start_time", "end_time", "failed", "event"}
    required.update({"average_usage", "maximum_usage", "random_sample_usage", "resource_request",
                     "assigned_memory", "page_cache_memory", "cycles_per_instruction",
                     "memory_accesses_per_instruction"})
    if not required.issubset(raw.columns) or not raw.failed.isin([0, 1]).all():
        raise ValueError("Google sample schema or labels invalid")
    rows_before = len(raw)
    keys = ["cluster", "collection_id", "instance_index", "start_time", "end_time"]
    if raw.empty or raw[keys + ["event"]].isna().any().any():
        raise ValueError("Google telemetry keys and events must be present on every row")
    for column in ("start_time", "end_time"):
        raw[column] = pd.to_numeric(raw[column], errors="raise")
        if not np.isfinite(raw[column]).all():
            raise ValueError(f"Google {column} must be finite")
    if (raw.end_time < raw.start_time).any():
        raise ValueError("Google usage end_time precedes start_time")
    conflicts = int((raw.groupby(keys).failed.nunique() > 1).sum())
    if conflicts:
        raise ValueError(f"Conflicting failure labels on {conflicts} telemetry keys")
    event_agreement = float((raw.failed == raw.event.eq("FAIL").astype(int)).mean())
    if event_agreement != 1.0:
        raise ValueError("Google failed labels disagree with event=FAIL")
    raw = raw.drop_duplicates(keys).sort_values(["start_time", *keys[:3]], kind="stable").copy()
    sizes = raw.groupby(keys[:3]).size()
    max_sequence = int(sizes.max())
    for column, prefix in [("average_usage", "average"), ("maximum_usage", "maximum"),
                           ("random_sample_usage", "sample"), ("resource_request", "requested")]:
        parsed = raw[column].map(_resource)
        raw[f"{prefix}_cpu"] = parsed.map(lambda x: x.get("cpus", np.nan))
        raw[f"{prefix}_memory"] = parsed.map(lambda x: x.get("memory", np.nan))
    features = GOOGLE_FEATURES.copy()
    if config.get("google_resource_ratios", False):
        # Row-local ratios use no outcomes, identifiers, or future observations.
        # A non-positive request has no meaningful utilisation ratio; impute it
        # using training-only medians instead of dividing by an arbitrary epsilon.
        for resource in ("cpu", "memory"):
            denominator = raw[f"requested_{resource}"]
            name = f"{resource}_utilisation"
            raw[name] = raw[f"average_{resource}"] / denominator.where(denominator > 0)
            features.append(name)
    cuts = [int(len(raw) * config["split_fractions"][0]), int(len(raw) * sum(config["split_fractions"][:2]))]
    times = [int(raw.iloc[i].start_time) for i in cuts]
    # Equal timestamps stay together. Purge tasks spanning partitions to stop task identity leakage.
    raw["partition"] = np.searchsorted(times, raw.start_time.to_numpy(), side="right")
    spanning = raw.groupby(keys[:3]).partition.transform("nunique") > 1
    purged = int(spanning.sum())
    unique_rows = len(raw)
    raw = raw.loc[~spanning].copy()
    parts = [raw.loc[raw.partition == i] for i in range(3)]
    metadata = {
        "dataset": "google_cluster", "display_name": "Google Cluster 2019 sample",
        "source_rows": rows_before, "source_features": 34,
        "resource_ratios": bool(config.get("google_resource_ratios", False)),
        "duplicate_telemetry_rows_removed": rows_before - unique_rows,
        "unique_telemetry_rows": unique_rows, "cross_partition_task_rows_purged": purged,
        "conflicting_label_keys": conflicts, "event_fail_label_agreement": event_agreement,
        "max_observations_per_task_after_deduplication": max_sequence,
        "split_time_boundaries_microseconds": times,
        "label_semantics": "Kaggle failed flag (matches event=FAIL), a failure proxy; label 0 includes other event types and is not verified healthy operation.",
        "protocol": "Chronological split by usage start_time with entire cross-boundary tasks purged; failure outcomes and identifiers excluded from inputs.",
        "time_unit": "Microseconds relative to trace origin; CPU and memory are normalised trace units.",
        "excluded_features": ["failed", "event", "instance_events_type", "collections_events_type", "time", "end_time", "start_time", "cluster", "collection_id", "instance_index", "machine_id", "user", "collection_name", "collection_logical_name"],
        "lstm_status": f"Not applicable: maximum {max_sequence} distinct observations per task; {config['sequence_length']} required. No unrelated tasks joined into sequences.",
        "limitations": "Secondary tabular failure discrimination, not prospective failure prediction. Kaggle event-to-usage join and label timing are not independently verified. No disk or network telemetry. The original trace's representativeness is not inherited by this sample.",
    }
    return _process_and_save([part[features].copy() for part in parts],
        [part.failed.to_numpy() for part in parts], [part.index.to_numpy() for part in parts],
        directory, metadata, config, False)

