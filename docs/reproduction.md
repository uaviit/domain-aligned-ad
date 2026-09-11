# Reproduction guide: paper tables ↔ repository files

This document maps the released files to the experimental tables of the paper and
explains how to recompute the metrics offline (no images required).

## Split (the canonical reference)

All k-NN results in the paper are computed on **`splits/split.json`** (division-1, the
V4 training split): 2,167 training / memory-bank images, 542 held-out normal images,
plus all 19 defective images as the test set. This JSON file is the single source of
truth for the split; regenerate it with `python splits/generate_splits.py` and verify
with `python splits/generate_splits.py --verify`.

## Score file formats

- `scores/p1/scores_v4.npz` — V4 (self-trained) L11 top-1% scores:
  `files_test` (561), `gt` (0=normal, 1=defective), `s_L11` (test scores),
  `loo_L11` + `files_loo` (leave-one-out scores of the 2,167 training images; basis of
  the mean+k·σ thresholds, no test leakage).
- `scores/p1/scores_vitb.npz` — pre-trained ViT-B A-DINO pipeline:
  `s_L6`, `s_L10`, `s_geo`, `s_rank` (layer and ensemble scores) plus the LOO
  counterparts.
- `scores/p1/vits_ref/vits_split1.json` — officially pre-trained ViT-S on the same
  split, with `scores_test` / `scores_loo`.
- `scores/p1/p1_threshold_ap_ci.json` — threshold analysis: thresholds, AP, bootstrap
  CIs and the per-defect-sample dissection of the 2×2 k-NN units.
- `scores/p1/per_sample_19bad.csv` — rank/decision summary of the 19 defective samples
  for the V4, ViT-B and Dinomaly2 units.

## Table-by-table mapping

**Tables 2, 5 (reconstruction units), 6, 7, 8.** Compiled from the robustness and
baseline experiments of the paper (baseline implementations, reconstruction-head
re-training, multi-split variance, independent re-training, zero-shot cross-component
FPR). The associated code and per-sample scores are not part of this release; the
numbers are reported in the paper.

**Tables 3 and 4 (2×2 backbone × framework matrix; same-scale control).**
Source: `scores/p1/p1_threshold_ap_ci.json`, keys:

| Unit | Key | npz |
|---|---|---|
| ViT-B + kNN (A-DINO L6/L10 rank ensemble) | `adino_vitb_L6L10_rank` | `scores_vitb.npz` (`s_rank`) |
| V4 + kNN (L11 top-1%) | `ads0_v4_L11top1` | `scores_v4.npz` (`s_L11`) |
| ViT-B + Dinomaly2 (official) | `dinomaly2_official` | — |
| V4 + Dinomaly2 (reconstruction) | `dinomaly2_v4` | — |

Table 4's ViT-S row comes from `scores/p1/vits_ref/vits_split1.json` (`metrics` field).
The kNN reference row of Table 5 equals the `ads0_v4_L11top1` unit above.

Offline recomputation example (reproduces the 98.3% AUROC / 71.8% best-F1 of V4):

```python
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve

d = np.load('scores/p1/scores_v4.npz', allow_pickle=True)
gt, s = d['gt'], d['s_L11']
print(roc_auc_score(gt, s))          # 0.9832  -> Table 4: AUROC 98.3
prec, rec, thr = precision_recall_curve(gt, s)
f1 = 2 * prec * rec / np.clip(prec + rec, 1e-10, None)
print(f1.max())                      # 0.7179  -> Table 4: best-F1 71.8

# F1 @ mean+2σ (deployment operating point, threshold from training LOO scores only)
mu, sd = d['loo_L11'].mean(), d['loo_L11'].std()
pred = (s >= mu + 2 * sd).astype(int)
```

The same procedure applied to `scores_vitb.npz['s_rank']` yields the ViT-B rank
ensemble (AUROC 0.9672, best-F1 0.6222), and to `vits_ref/vits_split1.json`
(`scores_test`, with `scores_loo` for thresholds) yields the ViT-S reference row.

## Script chain (requires the image data)

```
code/train_gb_dinov2_scratch_v2.py   # V4 self-supervised pre-training (2,167 images)
code/paper_p1a_scores.py             # P1a: per-image scores of the 2×2 kNN units -> scores/p1/*.npz
code/paper_p1b_analysis.py           # P1b: thresholds, AP, bootstrap CIs, 19-bad dissection -> p1_threshold_ap_ci.json
```

Training reproduces on a single GPU (RTX 4090, 24 GB): V4 ≈ 9 h for 18,400 iterations;
each k-NN evaluation ≈ 15 min. The scripts keep the exact hyperparameters, seeds and
data isolation used for the paper.
