# Domain-Aligned Anomaly Detection

Self-supervised anomaly detection for ground-wire hanging plate inspection: a file-disjoint comparison of a domain-aligned backbone and official DINOv2 models under kNN and reconstruction scoring.

## Overview

This repository implements the complete evaluation pipeline behind the paper's main experiment: a file-disjoint, common-calibration comparison of three self-supervised backbones — a domain-aligned ViT backbone (**V4**) and the official DINOv2 ViT-B/ViT-S — under two scoring paradigms:

- **kNN scoring** over frozen patch features using memory banks of defect-free images;
- **Reconstruction scoring** with fixed-iteration Dinomaly2 heads trained without any test-based selection.

Running the pipeline regenerates the paper's **Table 3** (P6-CAL point estimates and mean + kσ F1 values) and **Table 10** (paired-bootstrap intervals and significance analyses), together with the full-precision metric record. Runtime artifacts are written under `outputs/p6_cal/` (the directory name refers to the paper's file-disjoint common-calibration protocol).

## Highlights

- Deterministic frozen partition (seed 606): 2,167 training / 271 calibration / 271 normal final-test / 19 defective final-test images, disjoint at the file level.
- Thresholds estimated only from normal calibration images; no final-test scores or labels enter calibration.
- Fixed-iteration reconstruction training with no test-based model selection.
- 10,000-resample stratified paired bootstrap plus paired DeLong and exact McNemar tests.
- Single-command end-to-end reproduction, or evaluation-only recomputation from cached per-image scores.

## Requirements and installation

Python 3.10+ and a CUDA-capable PyTorch environment are recommended; Dinomaly2 head training requires CUDA.

```bash
python -m pip install -r requirements.txt
```

## Data and prerequisites

The inspection dataset is confidential, so images, split file-name lists, per-image results, and trained weights are **not distributed** with this repository. Reproducing the paper's numbers requires the following external inputs, supplied by arguments or environment variables:

| Environment variable | Meaning |
|---|---|
| `AD_DATA_ROOT` | GBS1 root containing `good/` and `bad/` |
| `GBS1_SPLIT` | Original GBS1 split JSON |
| `V4_CHECKPOINT` | Private V4 iteration-11000 checkpoint |
| `DINOV2_REPO` | Clone of the official [DINOv2](https://github.com/facebookresearch/dinov2) repository |
| `DINOMALY2_REPO` | Clone of the official [Dinomaly2](https://github.com/guojiajeremy/Dinomaly2) repository |

The user-supplied GBS1 root must follow:

```text
<AD_DATA_ROOT>/
  good/    normal images named exactly as in the original GBS1 split JSON
  bad/     19 fracture images named exactly as in that split JSON
```

Official DINOv2 ViT-B/ViT-S loading uses `anomalib`; Dinomaly2 official ViT-B weights follow the upstream repository's `backbones/weights` convention. The self-supervised pretraining code for the V4 backbone is outside the scope of this release; its training configuration is recorded in `configs/p6_cal.json`. No absolute path is hard-coded.

Because the images, split lists, and checkpoints are confidential, **no manuscript number can be regenerated from this repository alone**. With those external inputs, the scripts regenerate the six compared per-image score files and all Table 3/Table 10 outputs.

## Quick start

Run from the repository root, replacing the placeholders with local locations:

```bash
python scripts/reproduce_p6.py \
  --data-root path/to/lj_gb \
  --split path/to/split.json \
  --v4-checkpoint path/to/iter_11000.pth \
  --dinov2-repo path/to/dinov2 \
  --dinomaly2-repo path/to/Dinomaly2
```

The command runs the full pipeline and writes runtime artifacts under `outputs/p6_cal/` (ignored by Git). The stages can also be run separately using the environment variables above:

```bash
python scripts/prepare_protocol.py --split path/to/split.json
python scripts/score_knn.py --backbone v4
python scripts/score_knn.py --backbone official_vitb
python scripts/score_knn.py --backbone official_vits
python scripts/score_reconstruction.py --branch official_vitb_dinomaly2
python scripts/score_reconstruction.py --branch v4_dinomaly2_252
python scripts/score_reconstruction.py --branch v4_dinomaly2_224
python scripts/evaluate_p6.py
```

To recompute the tables from previously generated scores:

```bash
python scripts/reproduce_p6.py --evaluate-only
```

## Generated outputs

| Output | Content |
|---|---|
| `outputs/p6_cal/protocol.json` | Frozen 2,167 / 271 / 271 + 19 file-disjoint partition |
| `outputs/p6_cal/scores/*.npz` | Per-image scores for the six compared methods (never committed) |
| `outputs/p6_cal/tables/table3.csv` | Table 3: P6-CAL point estimates and mean + kσ F1 values |
| `outputs/p6_cal/tables/table11.csv` | Table 10: Table 3 point estimates plus 95% paired-bootstrap intervals |
| `outputs/p6_cal/tables/p6_metrics.json` | Full-precision metrics, thresholds, counts, bootstrap summaries, DeLong and McNemar tests |
| `outputs/p6_cal/tables/table11_bootstrap_raw.npz` | Raw bootstrap replicates for independent auditing (never committed) |

The internal filename `table11.csv` is retained for compatibility and corresponds to manuscript Table 10.

Table 10 point estimates are computed once on all 290 final-test images. Each interval uses 10,000 stratified paired resamples with seed 606. Deployment thresholds are fixed at each method's calibration mean + 2σ; best-F1 is re-optimized inside each resample and remains an oracle analysis metric.

To plot a generated score file:

```bash
python figures/plot_scores.py outputs/p6_cal/scores/v4_knn.npz
```

## Repository structure

```text
domain-aligned-ad/
├── README.md                         # Overview, installation, data contract, reproduction, and result mapping
├── LICENSE                           # MIT license for the released source code
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Excludes data, weights, scores, caches, and generated figures
├── configs/
│   └── p6_cal.json                   # Protocol, V4 record, kNN/reconstruction hyperparameters, bootstrap settings
├── src/
│   ├── common.py                     # Loads configuration/protocols and resolves argument-or-environment paths
│   ├── splits.py                     # Builds and validates the deterministic seed-606 P6-CAL partition
│   ├── features.py                   # Preprocesses images and extracts normalized frozen patch features
│   ├── backbones.py                  # Loads official DINOv2 models or the V4 teacher backbone
│   ├── knn.py                        # Builds memory banks, computes kNN scores, and applies the ViT-B ensemble
│   ├── reconstruction.py             # Trains and scores fixed-iteration Dinomaly2 heads without test-based selection
│   ├── metrics.py                    # Computes AUROC, AP, best-F1, thresholds, P/R/F1, FPR, and counts
│   └── bootstrap.py                  # Implements paired bootstrap, DeLong, and exact McNemar analyses
├── scripts/
│   ├── prepare_protocol.py           # Writes or verifies the frozen protocol
│   ├── score_knn.py                  # Runs one of the three native kNN branches
│   ├── score_reconstruction.py       # Runs one of the three reconstruction branches
│   ├── evaluate_p6.py                # Produces Table 3, Table 10, and paired statistical outputs
│   └── reproduce_p6.py               # Runs the complete pipeline or evaluation-only stage
└── figures/
    └── plot_scores.py                # Plots score distributions without shipping images
```

## Reproducibility notes

Randomized GPU kernels, dependency versions, and upstream repository revisions can cause small numerical differences. For strict archival reproduction, record the environment and upstream commit identifiers alongside the generated `p6_metrics.json`. Results on the confidential GBS1/UXGH images, official-weight downloads, and the private V4 checkpoint are not independently reproducible without those external inputs.

## License

Source code is released under the MIT License. External DINOv2, Dinomaly2, and anomalib code, model weights, and the confidential datasets remain subject to their own licenses and access conditions.
