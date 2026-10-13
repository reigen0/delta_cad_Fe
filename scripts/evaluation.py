import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def masked_scores(probs, mask=None):
    probs = np.asarray(probs)
    return probs if mask is None else np.where(mask, probs, 0.0)


def predict_labels(probs, threshold, mask=None):
    pred = np.asarray(probs) >= threshold
    if mask is not None:
        pred &= np.asarray(mask, dtype=bool)
    return pred


def best_mcc_threshold(y, probs):
    """Choose the validation threshold that maximizes MCC."""
    thresholds = np.r_[
        np.nextafter(probs.max(), np.inf),
        np.unique(probs),
    ]

    scores = [
        matthews_corrcoef(y, probs >= threshold)
        for threshold in thresholds
    ]

    i = int(np.argmax(scores))

    return float(thresholds[i]), float(scores[i])

#compute evaluation metrics
def compute_metrics(y, probs, threshold, mask=None):
    y = np.asarray(y, dtype=int)
    scores = masked_scores(probs, mask)
    pred = predict_labels(probs, threshold, mask)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "ap": average_precision_score(y, scores),
        "mcc": matthews_corrcoef(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
