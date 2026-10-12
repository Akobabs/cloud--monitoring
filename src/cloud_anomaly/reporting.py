"""Generate tables and figures directly from immutable run outputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .data import write_json
from .models import NAMES

METRICS = ["precision", "recall", "f1", "roc_auc", "average_precision", "false_positive_rate",
           "train_seconds", "latency_ms_per_observation", "peak_rss_mb"]
COLORS = ["#2463a8", "#22a699", "#e6a23c", "#8b5fbf"]


def generate(run_dir: Path):
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(run_dir.glob("models/*/*/*/result.json"))]
    if not records:
        raise ValueError("No completed results to report")
    full = pd.DataFrame(records)
    full.to_csv(run_dir / "metrics.csv", index=False)
    means = full.groupby(["dataset", "model_name"])[METRICS].mean().reset_index()
    deviations = full.groupby(["dataset", "model_name"])[METRICS].std(ddof=1).add_suffix("_sd").reset_index()
    means.merge(deviations, on=["dataset", "model_name"]).to_csv(run_dir / "comparison.csv", index=False)
    full["family"] = np.where(full.dataset.str.startswith("smd_"), "SMD adapted benchmark", "Google failure proxy")
    # Macro-average machines within each seed, then quantify variation across seeds.
    seed_means = full.groupby(["family", "model_name", "seed"])[METRICS].mean().reset_index()
    summary = seed_means.groupby(["family", "model_name"])[METRICS].mean().reset_index()
    summary = summary.merge(seed_means.groupby(["family", "model_name"])[METRICS].std(ddof=1).add_suffix("_sd").reset_index(), on=["family", "model_name"])
    summary.to_csv(run_dir / "summary.csv", index=False)
    figs = run_dir / "figures"
    figs.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.95)
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 180, "font.family": "DejaVu Sans"})
    for family, frame in summary.groupby("family"):
        frame = frame.set_index("model_name").reindex([name for name in NAMES.values() if name in set(frame.model_name)])
        fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.0), constrained_layout=True)
        for axis, metric, title in zip(axes, ["f1", "false_positive_rate"], ["F1 score", "False positive rate"]):
            axis.barh(frame.index, frame[metric], xerr=frame[metric + "_sd"].fillna(0), color=COLORS[:len(frame)], capsize=3)
            axis.set_xlim(0, 1)
            axis.set_xlabel(title)
            axis.invert_yaxis()
        fig.suptitle(family + " — mean across seeds")
        fig.savefig(figs / ("smd_comparison.png" if family.startswith("SMD") else "google_comparison.png"))
        plt.close(fig)
    for dataset, subset in full.groupby("dataset"):
        seed = int(subset.seed.min())
        selected = subset[subset.seed == seed]
        fig, axes = plt.subplots(2, 2, figsize=(7, 6.5), constrained_layout=True)
        axes = axes.ravel()
        for unused in axes[len(selected):]:
            unused.axis("off")
        curve_fig, curve_axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        for index, (_, record) in enumerate(selected.iterrows()):
            prediction = pd.read_csv(run_dir / "models" / dataset / record.model / str(seed) / "predictions.csv")
            matrix = confusion_matrix(prediction.label, prediction.prediction, labels=[0, 1])
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[index],
                        xticklabels=["Normal", "Positive"], yticklabels=["Normal", "Positive"])
            axes[index].set_title(record.model_name)
            axes[index].set_xlabel("Predicted")
            axes[index].set_ylabel("Actual")
            if prediction.label.nunique() == 2:
                fpr, tpr, _ = roc_curve(prediction.label, prediction.score)
                precision, recall, _ = precision_recall_curve(prediction.label, prediction.score)
                curve_axes[0].plot(fpr, tpr, label=record.model_name, color=COLORS[index])
                curve_axes[1].plot(recall, precision, label=record.model_name, color=COLORS[index])
        fig.suptitle(f"{dataset} · seed {seed}")
        fig.savefig(figs / f"{dataset}_confusion.png")
        plt.close(fig)
        curve_axes[0].plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
        curve_axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
        curve_axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve")
        for axis in curve_axes:
            axis.legend(fontsize=8)
        curve_fig.suptitle(f"{dataset} · seed {seed}")
        curve_fig.savefig(figs / f"{dataset}_curves.png")
        plt.close(curve_fig)
    return full
