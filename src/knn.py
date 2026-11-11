"""Cosine kNN memory-bank scoring and decision-level ensemble."""
import numpy as np
import torch


def memory_bank(train_features, device):
    """Flatten normalized [image, patch, channel] features into a GPU bank."""
    return train_features.reshape(-1, train_features.shape[-1]).to(device)


@torch.no_grad()
def score(features, bank, neighbours=3, top_fraction=0.01, batch_images=4,
          distance_topk=False):
    """Score each image by the mean of its largest top-fraction patch distances."""
    output = []
    patch_count = features.shape[1]
    top_count = max(int(patch_count * top_fraction), 1)
    for start in range(0, len(features), batch_images):
        batch = features[start:start + batch_images]
        b, n, d = batch.shape
        flat = batch.reshape(b * n, d).to(bank.device)
        similarity = flat @ bank.T
        if distance_topk:
            distances = (1 - similarity).clamp(min=0.0, max=2.0)
            selected = distances.topk(neighbours, dim=-1, largest=False).values
        else:
            selected = 1 - similarity.topk(neighbours, dim=-1).values
        patch_scores = selected.mean(dim=-1).reshape(b, n)
        output.append(patch_scores.topk(top_count, dim=1).values.mean(dim=1).cpu())
    return torch.cat(output).numpy()


def minmax_geometric_mean(first, second, first_fit, second_fit):
    """Apply the ViT-B L6/L10 ensemble with calibration-fitted min/max."""
    def fit(values):
        values = np.asarray(values, dtype=np.float64)
        return float(values.min()), float(values.max())

    def scale(values, limits):
        values = np.asarray(values, dtype=np.float64)
        fit_min, fit_max = limits
        z = (values - fit_min) / (fit_max - fit_min + 1e-9)
        return np.maximum(z, 0.0) + 1e-6

    first_limits = fit(first_fit)
    second_limits = fit(second_fit)
    scores = np.exp((np.log(scale(first, first_limits))
                     + np.log(scale(second, second_limits))) / 2)
    return scores, first_limits, second_limits
