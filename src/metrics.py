"""P6-CAL point estimates and fixed-calibration operating-point metrics."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def best_f1(gt, scores):
    precision, recall, _ = precision_recall_curve(gt, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(np.max(f1))


def confusion_metrics(gt, scores, threshold):
    gt = np.asarray(gt)
    predicted = np.asarray(scores) >= threshold
    tp = int(np.sum(predicted & (gt == 1)))
    fp = int(np.sum(predicted & (gt == 0)))
    tn = int(np.sum(~predicted & (gt == 0)))
    fn = int(np.sum(~predicted & (gt == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(threshold), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": float(precision), "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall) if precision + recall else 0.0),
        "fpr": float(fp / (fp + tn)),
    }


def evaluate(gt, scores, calibration_scores, k_values):
    result = {
        "auroc": float(roc_auc_score(gt, scores)),
        "ap": float(average_precision_score(gt, scores)),
        "best_f1": best_f1(gt, scores),
        "calibration_mean": float(np.mean(calibration_scores)),
        "calibration_std_ddof0": float(np.std(calibration_scores, ddof=0)),
        "thresholds": {},
    }
    for k in k_values:
        threshold = result["calibration_mean"] + k * result["calibration_std_ddof0"]
        result["thresholds"][f"mean+{k:g}sigma"] = confusion_metrics(gt, scores, threshold)
    return result


def metric_vector(gt, scores, threshold):
    return np.array([
        roc_auc_score(gt, scores), average_precision_score(gt, scores),
        best_f1(gt, scores), confusion_metrics(gt, scores, threshold)["f1"],
    ])
