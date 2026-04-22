import numpy as np

def roc_curve(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Sort by descending scores
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    # Count positives and negatives
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    # Cumulative sums
    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)

    # Indices where score changes (handle ties)
    distinct = np.where(np.diff(y_score))[0]
    idx = np.r_[distinct, len(y_score) - 1]

    # Select points
    tp = tp[idx]
    fp = fp[idx]
    thresholds = y_score[idx]

    # Compute rates
    tpr = tp / P if P > 0 else np.zeros_like(tp, dtype=float)
    fpr = fp / N if N > 0 else np.zeros_like(fp, dtype=float)

    # Add starting point (0,0, inf)
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[np.inf, thresholds]

    return fpr, tpr, thresholds