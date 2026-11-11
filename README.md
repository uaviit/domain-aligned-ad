# P6-CAL Minimal Reproduction Package

Minimal code for the revised manuscript **“Domain-Aligned Self-Supervised Backbones for Anomaly Detection with Defect-Free Training: A Case Study on Ground-Wire Hanging Plate Inspection.”**

This release targets only the file-disjoint P6-CAL backbone × scoring comparison: **Table 3, Table 10, and the associated paired DeLong/McNemar analyses**. It implements deterministic split construction, frozen-feature extraction, native kNN scoring, fixed-iteration Dinomaly2 head training/scoring, point estimates, and stratified paired bootstrap intervals. The previously proposed representation–scoring alignment guideline is not implemented as a claim; the clean comparison is the object of this release.

## What is not shipped

- GBS1 and UXGH images, split file-name lists, and per-image result files are private and are **not included**.
- V4 checkpoints, official backbone weights, and Dinomaly2 weights are **not included**.
- The DINOv2 and Dinomaly2 upstream repositories are external dependencies, not vendored copies.
- V4 self-supervised pretraining code is outside this minimal release; its configuration is recorded in `configs/p6_cal.json`, and evaluation requires the iteration-11000 V4 checkpoint.
- Tables 1–2, 4–9, and 11–13, covariate analyses, UXGH experiments, and qualitative heatmaps are outside this package.

**Consequence:** no manuscript number can be regenerated from this repository alone without external data and checkpoints. With the private GBS1 images, the original GBS1 split JSON, the V4 checkpoint, and downloadable official dependencies, the scripts regenerate the six P6-CAL score files and the Table 3/Table 10 outputs. `--evaluate-only` recomputes tables only after those six score files have been generated.

## Data and dependency convention

The user-supplied GBS1 root must contain:

```text
<AD_DATA_ROOT>/
  good/    normal images named exactly as in the original GBS1 split JSON
  bad/     19 fracture images named exactly as in that split JSON
```

Clone the official [DINOv2](https://github.com/facebookresearch/dinov2) and [Dinomaly2](https://github.com/guojiajeremy/Dinomaly2) repositories separately. Official DINOv2 ViT-B/ViT-S loading uses `anomalib`; Dinomaly2 official ViT-B weights use the upstream repository's `backbones/weights` convention.

All local locations are supplied by arguments or environment variables. No local absolute path is hard-coded.

| Environment variable | Meaning |
|---|---|
| `AD_DATA_ROOT` | GBS1 root containing `good/` and `bad/` |
| `GBS1_SPLIT` | Original GBS1 split JSON |
| `V4_CHECKPOINT` | Private V4 iteration-11000 checkpoint |
| `DINOV2_REPO` | Clone of the official DINOv2 repository |
| `DINOMALY2_REPO` | Clone of the official Dinomaly2 repository |

## Installation

Python 3.10+ and a CUDA-capable PyTorch environment are recommended. Dinomaly2 head training requires CUDA.

```bash
python -m pip install -r requirements.txt
```

## One-command reproduction

Run from the repository root, replacing the relative placeholders with local locations:

```bash
python scripts/reproduce_p6.py \
  --data-root path/to/lj_gb \
  --split path/to/split.json \
  --v4-checkpoint path/to/iter_11000.pth \
  --dinov2-repo path/to/dinov2 \
  --dinomaly2-repo path/to/Dinomaly2
```

The command creates runtime artifacts under `outputs/p6_cal/`; that directory is ignored by Git. It performs the following stages, which can also be run separately:

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

The separate commands use the environment variables listed above. To recompute tables from previously generated scores:

```bash
python scripts/reproduce_p6.py --evaluate-only
```

## Outputs and paper mapping

| Output | Paper result |
|---|---|
| `outputs/p6_cal/protocol.json` | Frozen 2,167 train / 271 calibration / 271 normal test / 19 defective test partition |
| `outputs/p6_cal/scores/*.npz` | Generated per-image scores for the six compared methods; never committed |
| `outputs/p6_cal/tables/table3.csv` | P6-CAL point estimates and mean + kσ F1 values |
| `outputs/p6_cal/tables/table11.csv` | Point estimates identical to Table 3 plus 95% paired-bootstrap intervals |
| `outputs/p6_cal/tables/p6_metrics.json` | Full-precision metrics, thresholds, counts, bootstrap summaries, DeLong and McNemar tests |
| `outputs/p6_cal/tables/table11_bootstrap_raw.npz` | Generated bootstrap replicates for independent auditing; never committed |

The internal filename `table11.csv` is retained for compatibility and corresponds to manuscript Table 10.

Table 10 point estimates are computed once on all 290 final-test images. Each interval uses 10,000 stratified paired resamples with seed 606. Deployment thresholds are fixed at each method's calibration mean + 2σ; best-F1 is re-optimized inside each resample and remains an oracle analysis metric.

To plot a generated score file:

```bash
python figures/plot_scores.py outputs/p6_cal/scores/v4_knn.npz
```

## File guide

```text
open_source_v8/
├── README.md                         # Scope, installation, data contract, reproduction, and result mapping
├── LICENSE                           # MIT license for the released source code
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Excludes data, weights, scores, caches, secrets, and generated figures
├── configs/
│   └── p6_cal.json                   # Protocol, V4 record, kNN/reconstruction hyperparameters, and bootstrap settings
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

## Reproducibility limits

Randomized GPU kernels, dependency versions, and upstream repository revisions can cause small numerical differences. For strict archival reproduction, record the environment and upstream commit identifiers alongside the generated `p6_metrics.json`. The repository does not claim that results on private GBS1/UXGH data, official-weight downloads, or the private V4 checkpoint are independently reproducible without those external inputs.

## License

Source code is released under the MIT License. External DINOv2, Dinomaly2, anomalib, model weights, and private datasets remain subject to their own licenses and access conditions.
