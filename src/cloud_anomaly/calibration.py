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
