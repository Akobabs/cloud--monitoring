# Comparative Analysis of Anomaly Detection for Cloud Infrastructure Using Selected Machine Learning Algorithms

Implementation plan and research continuation

Chapters Four and Five with Appendices A to C

September 2026

This edition reports the retrained experiment 20260930T155937Z_research_improved, completed on 2026-09-30, and compares it with the preserved validation-calibrated baseline 20260928T092901Z_calibrated. All 45 model and seed evaluations were fitted again. The original Chapters One to Three are retained separately and unchanged.

The four-algorithm comparison uses three selected Server Machine Dataset series. Google Cluster records support a supplementary comparison of three tabular algorithms; their short task histories do not support the specified LSTM. Findings remain exploratory because the evaluation period had been reviewed before this revision.

# Implementation Plan

The revision addresses two implementation choices: resource representation and consistency between candidate selection and the final decision threshold. The experiment plan and candidate grid were fixed before the revised run. Data acquisition, partitioning, common fitting endpoints and the seed schedule were retained so differences could be interpreted against the existing baseline.

Plan 1 Work packages and evidence

| Work package | Implementation | Evidence |
| --- | --- | --- |
| Feature engineering | CPU and memory usage-to-request ratios | 13 Google predictors |
| Candidate selection | Calibrate each candidate before comparing validation F1 | tuning.json and selection.json |
| Fresh experiment | Three seeds and four datasets | 45 evaluations; 78 candidate fits |
| Verification | Recompute decisions and match baseline records | Two passed evidence audits |
| Reporting | Retain gains, regressions and fresh resource costs | Comparison CSV, figures and dashboard |

The baseline artifacts remain unchanged. The improved profile is reproduced with run_project.bat improved, and the local dashboard is opened with run_frontend.bat. The configuration, raw-file hashes, dependency versions and source fingerprint are retained in the new run directory. No separate unit-test suite was introduced; verification uses execution and artifact audits.

# CHAPTER FOUR

# IMPLEMENTATION AND RESULTS

## 4.1 Introduction

The framework compares Isolation Forest, One-Class SVM, Random Forest and an LSTM Autoencoder using chronological partitions and common evaluation endpoints. F1 is the primary detection metric; precision, recall, false-positive rate, ROC-AUC and average precision provide context. Computational measurements describe offline CPU execution, not production serving performance.

## 4.2 Implementation Environment

Table 4.1 Recorded execution environment

| Component | Value |
| --- | --- |
| Operating system | Windows-11-10.0.26200-SP0 |
| Python | 3.12.14 |
| Memory | 7.85 GiB |
| Scikit-learn | 1.7.2 |
| TensorFlow and Keras | 2.20.0 and 3.15.1 |
| Execution | CPU; 2 compute threads; sequential model processes |
| Random seeds | 42, 43, 44 |

Deterministic TensorFlow operations and random seeds were requested. Such controls do not guarantee bitwise equivalence across library or hardware changes. The environment snapshot is the authoritative dependency record for this run.

## 4.3 Methodological Amendments

The earlier baseline selected model settings using F1 at a fixed percentile threshold and then recalibrated only the selected model. The revised implementation calibrates every candidate on labelled validation data before selecting the candidate with the highest validation F1. This aligns the selection criterion with the final classification rule. It uses the same candidate grid and fitting budget as the reference experiment.

The second change adds two row-local Google resource ratios. All models were refitted, so weights, continuous scores and resource measurements may change. This is a modelling experiment, unlike the earlier threshold-only follow-up. Both modifications were applied together on Google; their separate causal contributions cannot be identified without a further ablation. SMD retains its original features and evaluates the changed selection procedure.

## 4.4 Dataset Acquisition and Audit

The Google source is Kaggle version 1 of the Google 2019 Cluster sample. Its 405,894 records included 152,300 repeated telemetry rows. Deduplication preceded chronological partitioning, and 11,000 rows were purged because their task crossed a partition boundary. Failure labels agree with event=FAIL. A zero label includes other event types and is not a verified healthy-operation label.

SMD supplies ordered, labelled metric records. This study re-partitions the published labelled test sequence into 60/20/20 chronological segments to permit supervised Random Forest training. The official normal-only training files are downloaded and hashed but not fitted. This adaptation is not directly comparable with the official SMD protocol. Only machine-1-1, machine-2-1 and machine-3-1 are included.

Table 4.2 Eligible endpoints after partitioning and sequence warm-up

| Dataset | Partition | Rows | Positives |
| --- | --- | --- | --- |
| Google | train | 146,406 | 38,809 |
| Google | validation | 48,807 | 15,038 |
| Google | test | 47,381 | 12,683 |
| SMD 1-1 | train | 17,058 | 670 |
| SMD 1-1 | validation | 5,667 | 1,988 |
| SMD 1-1 | test | 5,667 | 7 |
| SMD 2-1 | train | 14,187 | 156 |
| SMD 2-1 | validation | 4,710 | 577 |
| SMD 2-1 | test | 4,710 | 437 |
| SMD 3-1 | train | 17,191 | 183 |
| SMD 3-1 | validation | 5,711 | 0 |
| SMD 3-1 | test | 5,711 | 125 |

Google tasks contain at most 4 distinct observations after deduplication. Joining unrelated tasks into a 30-step sequence would create artificial histories, so the Google LSTM remains not applicable. Google outcome timing is not independently established; the task is retrospective failure-proxy discrimination, not demonstrated prospective prediction.

## 4.5 Preprocessing and Feature Engineering

The split precedes fitted transformations. Missing values are filled using training medians; entirely unobserved training columns are excluded. Standard scaling is fitted on training data and applied without refitting to validation and test partitions. Infinite engineered values are treated as missing. Input checks reject non-binary SMD labels before integer conversion, inconsistent Google labels, missing telemetry keys and invalid usage times.

Google retains its 11 resource predictors and adds CPU utilisation = average_cpu / requested_cpu and memory utilisation = average_memory / requested_memory. A ratio is missing when the request is non-positive and is then imputed from training values. No arbitrary epsilon is inserted into the denominator. These features describe resource use relative to allocation and contain no outcome flags, identifiers, timestamps or observations from another row.

SMD retains 38 anonymous metric channels, their trailing five-observation means and first differences, giving 114 inputs. Rolling calculations restart at each partition boundary. All four algorithms are evaluated at the same endpoints after excluding the first 29 records of each partition; LSTM inputs are genuine 30-step windows. No physical channel names or sampling interval are invented.

## 4.6 Model Training and Selection

Table 4.3 Fixed candidate settings

| Algorithm | Candidates |
| --- | --- |
| Isolation Forest | 100 or 200 trees; max_samples 256; normal windows only |
| One-Class SVM | RBF kernel; nu 0.01 or 0.05; gamma scale; normal windows only |
| Random Forest | 100 trees/depth 10 or 200 trees/depth 20; balanced class weights; leaf minimum 2 |
| LSTM Autoencoder | Two encoder and two decoder layers; latent width 8 or 16; Adam; MSE; up to 15 epochs; batch 64; patience 3 |

Each seed samples up to 6,000 shared training endpoints. Random Forest uses both labels; the other algorithms fit only endpoints whose entire input window is normal. Thus the study compares learning regimes as well as model families. LSTM early stopping uses up to 2,000 entirely normal validation windows; its score is reconstruction MSE at the last step.

## 4.7 Evaluation Procedure

Each candidate is scored on validation records. When both labels are present, its threshold maximises validation F1, with the higher threshold preferred on exact ties. Predictions use score > threshold consistently. Candidate ties retain the first predefined setting. Where validation lacks a class, the percentile of entirely normal validation scores is retained; machine-3-1 has no validation positives and uses the first predefined candidate. Selected settings, thresholds and validation scores are saved before test scoring.

F1, precision and recall are calculated on raw point predictions with no point adjustment. ROC-AUC and average precision use continuous scores. SMD family results first macro-average machines within each seed, then report means and sample standard deviations across seeds. These deviations do not establish statistical significance or uncertainty across unseen environments. The reused test period makes this an exploratory follow-up even though no test labels enter candidate or threshold selection.

## 4.8 Detection Results and Baseline Comparison

Table 4.4 F1 before and after retraining

| Dataset | Algorithm | Before | After | Change |
| --- | --- | --- | --- | --- |
| Google | Isolation Forest | 0.4141 | 0.4188 | +0.0048 |
| Google | One-Class SVM | 0.4473 | 0.4306 | -0.0167 |
| Google | Random Forest | 0.4339 | 0.4438 | +0.0100 |
| SMD | Isolation Forest | 0.2596 | 0.2643 | +0.0047 |
| SMD | LSTM AE | 0.1942 | 0.1954 | +0.0012 |
| SMD | One-Class SVM | 0.1566 | 0.1557 | -0.0009 |
| SMD | Random Forest | 0.1043 | 0.1043 | +0.0000 |

Across the seven dataset-family and algorithm combinations, F1 increased in 4, decreased in 2, and was unchanged in 1. The audit identified 12 changed candidate selections among 45 evaluations. No model or seed was excluded because of a regression.

Google Random Forest F1 changed from 0.4339 to 0.4438; its false-positive rate changed from 0.7612 to 0.5928. SMD Isolation Forest F1 changed from 0.2596 to 0.2643, with false-positive rate changing from 0.1210 to 0.1120. These are measured improvements in the selected operating points, not evidence of broad generalisation.

Table 4.5 False-positive rates before and after retraining

| Dataset | Algorithm | Before | After |
| --- | --- | --- | --- |
| Google | Isolation Forest | 0.8587 | 0.8596 |
| Google | One-Class SVM | 0.8386 | 0.6330 |
| Google | Random Forest | 0.7612 | 0.5928 |
| SMD | Isolation Forest | 0.1210 | 0.1120 |
| SMD | LSTM AE | 0.0838 | 0.0838 |
| SMD | One-Class SVM | 0.0779 | 0.0840 |
| SMD | Random Forest | 0.2607 | 0.2607 |

Table 4.6 Retrained detection results for SMD adapted benchmark

| Algorithm | Precision | Recall | F1 | ROC-AUC | FPR |
| --- | --- | --- | --- | --- | --- |
| Isolation Forest | 0.2161 | 0.7771 | 0.2643 | 0.9080 | 0.1120 |
| LSTM AE | 0.3881 | 0.7012 | 0.1954 | 0.9843 | 0.0838 |
| One-Class SVM | 0.3707 | 0.6967 | 0.1557 | 0.9874 | 0.0840 |
| Random Forest | 0.0769 | 0.6171 | 0.1043 | 0.8170 | 0.2607 |

Table 4.7 Retrained detection results for Google failure proxy

| Algorithm | Precision | Recall | F1 | ROC-AUC | FPR |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 0.3193 | 0.7443 | 0.4438 | 0.6595 | 0.5928 |
| One-Class SVM | 0.3021 | 0.7558 | 0.4306 | 0.5326 | 0.6330 |
| Isolation Forest | 0.2762 | 0.8895 | 0.4188 | 0.5474 | 0.8596 |
| LSTM AE | N/A | N/A | N/A | N/A | N/A |

The highest revised mean F1 was 0.2643 for Isolation Forest on SMD and 0.4438 for Random Forest on Google. A trivial all-positive Google rule obtains F1 0.4223 with recall and false-positive rate both equal to 1. The Google findings must be assessed against this reference and their still substantial false-alarm burden.

Table 4.8 Seed variation and score ranking

| Dataset | Algorithm | F1 SD | Average precision |
| --- | --- | --- | --- |
| Google | Isolation Forest | 0.0074 | 0.2823 |
| Google | One-Class SVM | 0.0456 | 0.2653 |
| Google | Random Forest | 0.0187 | 0.4087 |
| SMD | Isolation Forest | 0.0101 | 0.2937 |
| SMD | LSTM AE | 0.0071 | 0.8633 |
| SMD | One-Class SVM | 0.0021 | 0.8163 |
| SMD | Random Forest | 0.0247 | 0.0884 |

Table 4.9 Mean F1 by SMD machine

| Machine | Isolation Forest | One-Class SVM | Random Forest | LSTM AE |
| --- | --- | --- | --- | --- |
| 1-1 | 0.0134 | 0.0175 | 0.0150 | 0.0137 |
| 2-1 | 0.4489 | 0.1642 | 0.2140 | 0.1863 |
| 3-1 | 0.3306 | 0.2855 | 0.0838 | 0.3862 |

Machine-1-1 has only seven positive test endpoints, while machine-3-1 has no positive validation endpoints. These distributions explain why a threshold selected earlier may transfer poorly. Aggregate results must be read alongside the machine-level table.

![Figure 4.1 Retrained SMD F1 and false-positive rates with seed standard deviation](../artifacts/runs/20260930T155937Z_research_improved/figures/smd_comparison.png)

![Figure 4.2 Retrained Google F1 and false-positive rates with seed standard deviation](../artifacts/runs/20260930T155937Z_research_improved/figures/google_comparison.png)

## 4.9 Computational Results

Table 4.10 Newly measured computational costs

| Dataset | Algorithm | Fit seconds | ms per point | Peak RSS MB |
| --- | --- | --- | --- | --- |
| Google | Isolation Forest | 0.35 | 0.0075 | 175.3 |
| Google | One-Class SVM | 0.10 | 0.0259 | 180.2 |
| Google | Random Forest | 0.99 | 0.0059 | 180.1 |
| SMD | Isolation Forest | 0.37 | 0.0103 | 176.1 |
| SMD | LSTM AE | 58.79 | 1.9940 | 887.5 |
| SMD | One-Class SVM | 0.30 | 0.0469 | 190.3 |
| SMD | Random Forest | 1.86 | 0.0119 | 176.6 |

All resource values were measured during this new run. Fit time is the selected candidate's fit call; it excludes preprocessing and model construction. Tuning time is separately recorded. Inference is warmed and timed three times, with median batch time divided by the observation count. RSS is sampled every 10 ms and includes library and data overhead; brief peaks may be missed. These measurements should not be interpreted as single-request serving latency or exact per-model allocations.

## 4.10 Interface and Verification

The evidence audit verified all 45 saved evaluations across 4 datasets, including confusion counts, F1, reproducible validation decisions, aligned endpoints, finite features, partition separation and the source fingerprint. A second audit confirmed identical test row identifiers, labels, original prepared feature values and sampled training endpoints against the baseline. Only the two intended Google columns were added.

The dashboard reads saved artifacts, displays before-and-after F1 and false-positive rates, and retains access to older runs. All five views were exercised and a results CSV download was checked without browser errors. Screenshots in this edition were captured from the completed retrained experiment. The interface does not retrain models or issue production alerts.

![Figure 4.3 Dashboard showing the retrained run and baseline comparison](../artifacts/runs/20260930T155937Z_research_improved/screenshots/01_overview.png)

## 4.11 Discussion and Limitations

Aligning candidate selection with the final threshold rule corrects an optimisation mismatch, but it does not guarantee higher test F1. The One-Class SVM regressions are retained as evidence of this limitation. Resource ratios provide a representation of allocation pressure, but the combined Google experiment does not isolate their effect from candidate selection. Validation is reused for tuning and calibration, and for LSTM early stopping; this may overfit validation decisions.

The Google false-positive rates remain too high to justify an operational deployment claim. The best revised Google model must also be compared with the strongest earlier model, rather than only its own baseline. Small numerical gains across three seeds are not proof of statistical significance. No additional test-driven tuning round was performed after this comparison.

The experiment uses a reviewed evaluation period, three selected SMD machines, an adapted supervised split and a bounded candidate grid. It does not establish prospective failure prediction, robustness to drift, incident-level detection delay or production cost. Fresh environments and evaluation periods are required before generalisation can be assessed.

## 4.12 Chapter Summary

The revision produced 45 fresh model evaluations and an auditable before-and-after comparison. It changed both learned scores and candidate choices where supported by validation. The outcomes include 4 F1 gains, 2 regressions and 1 unchanged family-level result. Reporting these differences together with false alarms and resource costs provides the basis for the conclusions.

# CHAPTER FIVE

# SUMMARY CONCLUSION AND RECOMMENDATION

## 5.1 Summary of Study and Findings

The study implemented and compared four cloud telemetry anomaly-detection algorithms. Chronological preprocessing, shared endpoints, seed repetition and separate dataset interpretations support a controlled offline comparison. Google remains a supplementary tabular failure-proxy task because its task histories cannot support the requested sequence model.

The retrained experiment added resource-utilisation features and calibrated every candidate before selection. Google Random Forest achieved F1 0.4438 against 0.4339 previously, with a lower false-positive rate. SMD Isolation Forest achieved F1 0.2643 against 0.2596. Across all seven family-level comparisons, 4 improved, 2 regressed and 1 remained unchanged. The original experiment was preserved and resource measurements were repeated.

## 5.2 Conclusion

Within the revised experiment, Isolation Forest had the highest mean SMD F1 and Random Forest had the highest mean Google F1. The changes produced selective, modest gains rather than a universal improvement. A validation-selected operating point can reduce false alarms or improve F1, yet still transfer poorly to a later period. The study therefore supports careful validation and explicit reporting of regressions, not a claim that one algorithm is universally superior.

The framework meets the implementation and offline comparison objectives. High false-alarm rates, uncertain Google label timing and reuse of an already examined test period limit practical conclusions. These findings require independent confirmation before deployment decisions.

## 5.3 Recommendation for Future Research

Future work should separate the effects of resource features and candidate calibration through an ablation, reserve a distinct calibration segment, and evaluate the resulting choices on a new untouched time period. Rolling temporal validation would provide a stronger view of threshold stability and changing anomaly prevalence. A predeclared false-alarm budget should accompany F1 when operational alert volume matters.

The dataset coverage should expand to all SMD machines and independent infrastructure traces. Google usage records should be linked to verified future event outcomes with explicit prediction horizons and sufficient within-task history. A separate official-split SMD experiment should distinguish normal-only learning from supervised learning. Production studies should then measure drift, event-level delay, end-to-end latency and retraining cost.

# APPENDIX A

# CODEBASE AND REPRODUCTION

The current implementation is supplied as runnable source. run_project.bat improved executes configs/improved.json, preserving the original research and quick profiles. scripts/audit_run.py validates saved decisions; scripts/compare_experiments.py verifies baseline comparability and writes improvement_comparison.csv. scripts/build_improved_document.py creates this continuation from those audited results. Historical runs cannot be resumed with changed source because their fingerprints are intentionally enforced.

The shared calibration routine below implements the labelled-validation threshold search. Google ratios are implemented in src/cloud_anomaly/data.py; training and selection are implemented in worker.py. The printed listings are for inspection; runnable files remain authoritative.

## A.1 Validation Threshold Selection

```python
# src/cloud_anomaly/calibration.py
"""Decision thresholds selected exclusively from labelled validation records."""
import numpy as np

from .evaluation import validate_scores


def validation_threshold(y, scores, old_threshold):
    """Maximise validation F1; prefer the highest threshold on exact ties."""
    y, scores = validate_scores(y, scores, old_threshold)
    if len(np.unique(y)) != 2:
        return float(old_threshold), "Baseline threshold retained: validation lacks both classes"
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    cumulative_tp = np.cumsum(y[order])
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(ordered) - 1]
    tp = cumulative_tp[ends]
    fp = ends + 1 - tp
    fn = y.sum() - tp
    f1 = 2 * tp / (2 * tp + fp + fn)
    chosen = int(np.argmax(f1))
    threshold = float(np.nextafter(ordered[ends[chosen]], -np.inf))
    if not np.isfinite(threshold):
        raise ValueError("Calibration boundary cannot be represented as a finite threshold")
    return threshold, "Maximum labelled-validation F1; largest threshold resolves exact ties"

```

## A.2 Model Implementations

```python
# src/cloud_anomaly/models.py
"""Common scoring convention: larger scores always mean more anomalous."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
import joblib

NAMES = {
    "isolation_forest": "Isolation Forest", "one_class_svm": "One-Class SVM",
    "random_forest": "Random Forest", "lstm_autoencoder": "LSTM Autoencoder",
}


class TabularDetector:
    def __init__(self, kind: str, parameters: dict, seed: int, threads: int):
        self.kind = kind
        self.threshold = None
        if kind == "isolation_forest":
            self.model = IsolationForest(**parameters, contamination="auto", random_state=seed, n_jobs=threads)
        elif kind == "one_class_svm":
            self.model = OneClassSVM(**parameters, kernel="rbf", cache_size=256)
        elif kind == "random_forest":
            self.model = RandomForestClassifier(**parameters, class_weight="balanced",
                min_samples_leaf=2, random_state=seed, n_jobs=threads)
        else:
            raise ValueError(kind)

    def fit(self, x, y=None, validation=None):
        self.model.fit(x, y)
        return self

    def score(self, x):
        if self.kind == "random_forest":
            index = list(self.model.classes_).index(1)
            return self.model.predict_proba(x)[:, index]
        return -self.model.score_samples(x)

    def predict(self, x):
        if self.threshold is None:
            raise ValueError("Threshold has not been calibrated")
        return (self.score(x) > self.threshold).astype(np.int8)

    def save(self, path):
        joblib.dump(self, path.with_suffix(".joblib"))


class LSTMAutoencoder:
    def __init__(self, parameters, seed, threads, shape):
        import tensorflow as tf
        self.tf = tf
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        try:
            tf.config.threading.set_intra_op_parallelism_threads(threads)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            pass  # Thread settings are process-wide and already set for subsequent candidates.
        self.parameters = parameters
        self.threshold = None
        width = parameters["latent_dim"]
        inputs = tf.keras.Input(shape=shape)
        x = tf.keras.layers.LSTM(width * 2, return_sequences=True)(inputs)
        x = tf.keras.layers.LSTM(width)(x)
        x = tf.keras.layers.RepeatVector(shape[0])(x)
        x = tf.keras.layers.LSTM(width, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(width * 2, return_sequences=True)(x)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(shape[1]))(x)
        self.model = tf.keras.Model(inputs, output)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        self.history = {}

    def fit(self, x, y=None, validation=None):
        tf = self.tf
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        def dataset(values):
            return tf.data.Dataset.from_tensor_slices((values, values)).batch(
                self.parameters["batch_size"]).with_options(options)
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                     restore_best_weights=True)]
        history = self.model.fit(dataset(x), validation_data=dataset(validation),
            epochs=self.parameters["epochs"], callbacks=callbacks, verbose=0, shuffle=False)
        self.history = {key: [float(v) for v in values] for key, values in history.history.items()}
        return self

    def score(self, x):
        # Last-step error produces a point score aligned with each window endpoint.
        scores = []
        for start in range(0, len(x), 256):
            batch = x[start:start + 256]
            reconstruction = self.model(batch, training=False).numpy()
            scores.append(np.mean(np.square(batch[:, -1, :] - reconstruction[:, -1, :]), axis=1))
        return np.concatenate(scores)

    def predict(self, x):
        if self.threshold is None:
            raise ValueError("Threshold has not been calibrated")
        return (self.score(x) > self.threshold).astype(np.int8)

    def save(self, path):
        self.model.save(path.with_suffix(".keras"))

```

## A.3 Experiment Worker

```python
# src/cloud_anomaly/worker.py
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


```

# APPENDIX B

# INTERFACE EVIDENCE

![Figure B.1 Fresh detection and resource results](../artifacts/runs/20260930T155937Z_research_improved/screenshots/02_model_comparison.png)

![Figure B.2 Held-out score and error inspection](../artifacts/runs/20260930T155937Z_research_improved/screenshots/03_prediction_explorer.png)

![Figure B.3 Dataset interpretation and partitions](../artifacts/runs/20260930T155937Z_research_improved/screenshots/04_dataset_details.png)

# APPENDIX C

# DATA PROVENANCE AND INTEGRITY

The Google sample is Kaggle version 1; the SMD source revision is 7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4. The Google listing identifies CC BY-NC-SA 4.0, while the original Google trace is CC BY 4.0. The SMD repository includes MIT licensing; no separate dataset-specific licence was identified. Source URLs, raw-file sizes and SHA-256 values are stored in data_manifest.json.

Table C.1 Google predictor definitions

| Features | Definition |
| --- | --- |
| average_cpu, average_memory | Mean usage in normalised trace units |
| maximum_cpu, maximum_memory | Maximum usage in normalised trace units |
| sample_cpu | Supplied random-sample CPU usage |
| requested_cpu, requested_memory | Requested resources in normalised units |
| assigned_memory, page_cache_memory | Assigned limit and page-cache usage |
| cycles_per_instruction, memory_accesses_per_instruction | Supplied instruction-level ratios |
| cpu_utilisation, memory_utilisation | Mean usage divided by positive requested resources |

SMD features are metric_01 through metric_38 and their causal mean and difference terms. Outcome flags, task identifiers and timestamps do not enter model features. Complete missingness counts, retained feature names, partition counts and exclusions are stored in each dataset.json alongside prepared.npz and the fitted preprocessor.

Recorded source SHA-256: c02558fcf5ba1f33aad22bf836add83416700fa4999cc2e05ad8030d23619159

Recorded configuration SHA-256: 44b7b0f9614ef6989d3289f8edf4142a1117526338ca56601311177a737a3e9f

# References for the Implementation

Google. ClusterData 2019 traces. https://github.com/google/cluster-data/blob/master/ClusterData2019.md

Mwiti, D. Google 2019 Cluster sample, version 1. https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample

NetManAIOps. OmniAnomaly and Server Machine Dataset. https://github.com/NetManAIOps/OmniAnomaly

Scikit-learn developers. Common pitfalls and recommended practices. https://scikit-learn.org/stable/common_pitfalls.html

Scikit-learn developers. Tuning the decision threshold for class prediction. https://scikit-learn.org/stable/modules/classification_threshold.html

TensorFlow developers. Enable operation determinism. https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism
