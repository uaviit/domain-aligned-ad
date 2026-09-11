# Domain-Aligned Self-Supervised Backbones for Training-Free Anomaly Detection

Code and numerical results for the paper *"Domain-Aligned Self-Supervised Backbones for
Training-Free Anomaly Detection: A Case Study on Ground-Wire Hanging Plate Inspection"*.

This repository accompanies the paper and makes available:

- **Code of the proposed pipeline**: from-scratch self-supervised pre-training of a
  DINOv2-style ViT-S backbone on defect-free images of the target component (V4), and
  the k-NN anomaly-scoring evaluation of the 2×2 (backbone × scoring-framework)
  comparison (experiment P1: pre-trained ViT-B and self-trained V4 under the kNN
  discriminative framework, per-image scores and threshold analysis).
- **The GBS1 data split** (file-name list only, no images): the canonical training
  split `split.json`, together with a script that regenerates it exactly.
- **Per-sample anonymized numerical results**: raw anomaly scores for every test image
  (542 normal + 19 defective), LOO scores of the training images, and the officially
  pre-trained ViT-S reference — sufficient to independently recompute the AUROC /
  AP / best-F1 / F1@mean+2σ values reported in the paper without any access to the
  images (see `docs/reproduction.md`).
- **Anonymized and cropped representative sample images** (the paper figures).

Robustness experiments of the paper (multi-split variance, independent re-training,
zero-shot cross-component FPR) are reported in the paper; their code and per-sample
scores are not part of this release.

The inspection images themselves involve power grid operational security and cannot be
made public due to enterprise confidentiality agreements; the normal images and the
self-trained backbone weights can be made available through reasonable channels upon
negotiation with the enterprise.

## Repository layout

```
code/       V4 training script + P1 evaluation/analysis chain + shared components
splits/     GBS1 split file (JSON) + generator/verification script
scores/     Per-sample scores of experiment P1 (V4, ViT-B, ViT-S reference)
samples/    Anonymized figures of the paper
docs/       reproduction.md — mapping from paper tables to files and scripts
```

## Setup

- Python 3.10+, PyTorch 2.x, torchvision, numpy, scikit-learn, pandas, matplotlib, Pillow.
- The backbone training script (`code/train_gb_dinov2_scratch_v2.py`) builds on the
  official DINOv2 repository (https://github.com/facebookresearch/dinov2; see its
  LICENSE). Place it inside a clone of that repository (top level, next to the
  `dinov2/` package) to run training.
- The evaluation chain (`code/paper_common.py`, `code/paper_p1a_scores.py`,
  `code/paper_p1b_analysis.py`) additionally requires
  [anomalib](https://github.com/open-edge-platform/anomalib) (only for the DINOv2
  ViT-B loader `anomalib.models.components.dinov2.DinoV2Loader`).
- Expect the raw image data under `datasets/lj_gb/good` and `datasets/lj_gb/bad`
  (see the Data section below).

## Data

The dataset cannot be redistributed. To re-run the pipeline end to end, the following
layout is expected (file names must match those recorded in `splits/split.json`):

```
datasets/lj_gb/good/    2709 defect-free ground-wire hanging plate images (JPG)
datasets/lj_gb/bad/       19 defective images (JPG)
```

Without the images you can still:

1. verify the split file (`python splits/generate_splits.py --verify`), and
2. recompute the k-NN metrics of the paper from `scores/` (see `docs/reproduction.md`).

## Quick verification of the paper numbers

```bash
python splits/generate_splits.py --verify              # split file consistency
```

The shipped `scores/p1/scores_v4.npz` contains `files_test`, `gt`, `s_L11` (per-test-image
anomaly scores) and `loo_L11` (leave-one-out training scores used for the mean+k·σ
thresholds); `roc_auc_score(gt, s_L11)` reproduces the 98.3% AUROC of the paper.

## Citation

If you find this useful, please cite the paper (citation entry will be added upon
publication).

## License

Code is released under the MIT License (see `LICENSE`). Numerical result files
(`splits/`, `scores/`, `samples/`) are released under CC BY 4.0.
