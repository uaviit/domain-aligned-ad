"""Stratified paired bootstrap, paired DeLong, and exact McNemar tests."""
import numpy as np
from scipy.stats import binomtest, norm

from metrics import metric_vector

METRICS = ["AUROC", "AP", "best_f1", "f1@mean+2sigma"]


def paired_bootstrap(gt_by_method, scores_by_method, thresholds, n_bootstrap, seed, reference):
    names = list(scores_by_method)
    reference_gt = gt_by_method[reference]
    if not all(np.array_equal(gt_by_method[name], reference_gt) for name in names):
        raise ValueError("Paired bootstrap requires an identical ordered final test set")
    normal_idx = np.where(reference_gt == 0)[0]
    defective_idx = np.where(reference_gt == 1)[0]
    rng = np.random.RandomState(seed)
    values = {name: [] for name in names}
    differences = {name: [] for name in names if name != reference}

    for _ in range(n_bootstrap):
        idx = np.concatenate([
            rng.choice(normal_idx, len(normal_idx), replace=True),
            rng.choice(defective_idx, len(defective_idx), replace=True),
        ])
        current = {}
        for name in names:
            current[name] = metric_vector(
                gt_by_method[name][idx], scores_by_method[name][idx], thresholds[name],
            )
            values[name].append(current[name])
        for name in differences:
            differences[name].append(current[name] - current[reference])

    def summarize(samples):
        array = np.asarray(samples)
        return {
            "mean": array.mean(axis=0).tolist(),
            "ci95_lo": np.percentile(array, 2.5, axis=0).tolist(),
            "ci95_hi": np.percentile(array, 97.5, axis=0).tolist(),
            "metric_order": METRICS,
        }

    raw = {name: np.asarray(samples) for name, samples in values.items()}
    return {
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "resampling": "stratified paired by image",
        "intervals": {name: summarize(samples) for name, samples in values.items()},
        "paired_difference_from_reference": {
            name: summarize(samples) for name, samples in differences.items()
        },
    }, raw


def _midrank(values):
    order = np.argsort(values)
    sorted_values = values[order]
    n = len(values)
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    output = np.empty(n, dtype=np.float64)
    output[order] = ranks
    return output


def _delong_components(gt, scores):
    positive = gt == 1
    negative = ~positive
    n1, n0 = int(positive.sum()), int(negative.sum())
    all_ranks = _midrank(scores)
    auc = (all_ranks[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    positive_ranks = _midrank(scores[positive])
    negative_ranks = _midrank(scores[negative])
    v10 = (all_ranks[positive] - positive_ranks) / n0
    v01 = (all_ranks[negative] - negative_ranks) / n1
    return auc, v10, v01


def delong_paired(gt, first_scores, second_scores):
    auc1, v10_1, v01_1 = _delong_components(gt, first_scores)
    auc2, v10_2, v01_2 = _delong_components(gt, second_scores)
    d10, d01 = v10_1 - v10_2, v01_1 - v01_2
    variance = np.var(d10, ddof=1) / len(d10) + np.var(d01, ddof=1) / len(d01)
    delta = auc1 - auc2
    z = float(delta / np.sqrt(variance)) if variance > 0 else 0.0
    return {
        "auc_1": float(auc1), "auc_2": float(auc2), "delta": float(delta),
        "z": z, "p": float(2.0 * norm.sf(abs(z))),
    }


def mcnemar_exact(reference_prediction, other_prediction):
    b = int(np.sum(reference_prediction & ~other_prediction))
    c = int(np.sum(~reference_prediction & other_prediction))
    p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    return {"b_reference_only": b, "c_other_only": c, "exact_two_sided_p": p}
