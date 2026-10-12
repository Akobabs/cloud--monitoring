"""Metrics, measured process memory, and publication figures."""
from __future__ import annotations

import threading
import time

import numpy as np
import psutil
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                            precision_score, recall_score, roc_auc_score)


def validate_scores(y, scores, threshold):
    """Validate aligned binary labels and scores before any decisions are made."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or scores.ndim != 1 or not len(y) or y.shape != scores.shape:
        raise ValueError("Labels and scores must be non-empty aligned one-dimensional arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Expected binary labels and finite scores")
    if np.ndim(threshold) != 0 or not np.isfinite(threshold):
        raise ValueError("Threshold must be a finite scalar")
    return y.astype(np.int8), scores


def metrics(y, scores, threshold):
    y, scores = validate_scores(y, scores, threshold)
    prediction = (scores > threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) == 2
    return {
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)) if (tp + fn) else None,
        "f1": float(f1_score(y, prediction, zero_division=0)) if (tp + fn) else None,
        "roc_auc": float(roc_auc_score(y, scores)) if both_classes else None,
        "average_precision": float(average_precision_score(y, scores)) if both_classes else None,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


class MemoryMonitor:
    """Sample process RSS, including native allocations, every 10 ms.

    This is a sampled maximum and can miss short-lived allocation spikes.
    Windows lifetime peak working-set is also reported separately.
    """
    def __enter__(self):
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(0.01):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()
        memory = self.process.memory_info()
        self.lifetime_peak = getattr(memory, "peak_wset", self.peak)


def benchmark_score(model, x, repetitions=3):
    model.score(x[:min(256, len(x))])  # untimed warm-up
    timings = []
    for _ in range(repetitions):
        start = time.perf_counter()
        scores = model.score(x)
        timings.append(time.perf_counter() - start)
    return scores, float(np.median(timings) * 1000 / len(x)), timings

