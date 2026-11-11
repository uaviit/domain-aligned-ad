"""Extract frozen features and compute native kNN scores for one P6-CAL backbone."""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backbones import load_official, load_v4_teacher  # noqa: E402
from common import (DEFAULT_CONFIG, load_config, load_protocol,  # noqa: E402
                    protocol_image_paths, save_json, supplied_path)
from features import extract_layers  # noqa: E402
from knn import memory_bank, minmax_geometric_mean, score  # noqa: E402


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=["v4", "official_vitb", "official_vits"])
    parser.add_argument("--protocol", default="outputs/p6_cal/protocol.json")
    parser.add_argument("--data-root", help="GBS1 root with good/ and bad/; or set AD_DATA_ROOT")
    parser.add_argument("--output", default="outputs/p6_cal/scores")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--v4-checkpoint", help="V4 checkpoint; or set V4_CHECKPOINT")
    parser.add_argument("--dinov2-repo", help="Official DINOv2 clone; or set DINOV2_REPO")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    branch = config["knn"][args.backbone]
    protocol = load_protocol(args.protocol)
    data_root = supplied_path(args.data_root, "AD_DATA_ROOT")
    train_paths, eval_paths = protocol_image_paths(protocol, data_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.backbone == "v4":
        checkpoint = supplied_path(args.v4_checkpoint, "V4_CHECKPOINT")
        dinov2_repo = supplied_path(args.dinov2_repo, "DINOV2_REPO")
        backbone, iteration = load_v4_teacher(checkpoint, dinov2_repo, device)
        print(f"Loaded V4 teacher at iteration {iteration}")
        official_api = False
    else:
        backbone = load_official(branch["anomalib_name"], device)
        official_api = args.backbone == "official_vitb"

    kwargs = dict(layers=branch["layers"], image_size=branch["image_size"],
                  interpolation=branch["interpolation"], device=device,
                  fp16=branch["fp16"], batch_size=args.batch_size,
                  workers=args.workers, official_api=official_api)
    print("Extracting training memory-bank features", flush=True)
    train_features = extract_layers(backbone, train_paths, **kwargs)
    print("Extracting held-out calibration/final-test features", flush=True)
    eval_features = extract_layers(backbone, eval_paths, **kwargs)

    layer_scores = {}
    for layer in branch["layers"]:
        bank = memory_bank(train_features[layer], device)
        layer_scores[layer] = score(
            eval_features[layer], bank, neighbours=branch["neighbours"],
            top_fraction=branch["top_fraction"], distance_topk=args.backbone == "official_vitb",
        )
    normalization = None
    if branch.get("ensemble") == "minmax_geometric_mean":
        fit_key = branch["ensemble_fit"]
        if fit_key != "calibration_normal":
            raise ValueError("The L6/L10 ensemble must fit on calibration_normal only")
        fit_names = protocol[fit_key]
        n_cal = len(fit_names)
        if n_cal != config["protocol"]["calibration_normal"]:
            raise ValueError(f"Unexpected calibration count: {n_cal}")
        actual_fit_names = [path.name for path in eval_paths[:n_cal]]
        if actual_fit_names != fit_names:
            raise AssertionError("Evaluation paths do not match calibration names and order")
        scores, first_limits, second_limits = minmax_geometric_mean(
            layer_scores[6], layer_scores[10],
            layer_scores[6][:n_cal], layer_scores[10][:n_cal],
        )
        normalization = {
            "fit_split": fit_key,
            "fit_count": n_cal,
            "layers": {
                "6": {"min": first_limits[0], "max": first_limits[1]},
                "10": {"min": second_limits[0], "max": second_limits[1]},
            },
            "negative_clip": "max(z, 0.0) + 1e-6; values above fit_max are not clipped",
        }
    else:
        scores = layer_scores[branch["layers"][0]]

    labels = np.asarray([0] * (len(eval_paths) - len(protocol["test_defective"]))
                        + [1] * len(protocol["test_defective"]), dtype=np.int64)
    output = Path(args.output) / branch["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, files_test=np.asarray([path.name for path in eval_paths]),
                        gt=labels, pred_scores=np.asarray(scores, dtype=np.float64))
    if normalization is not None:
        repo = Path(__file__).resolve().parents[1]
        code_files = ["src/knn.py", "scripts/score_knn.py", "configs/p6_cal.json"]
        save_json(output.with_suffix(".run.json"), {
            **normalization,
            "branch": args.backbone,
            "num_scores": len(scores),
            "all_scores_finite": bool(np.isfinite(scores).all()),
            "protocol_sha256": hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest(),
            "code_version": "open_source_v8 non-git working tree",
            "code_sha256": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                            for name in code_files},
            "elapsed_seconds": time.perf_counter() - started,
        })
    print(f"Wrote {len(scores)} scores: {output}")


if __name__ == "__main__":
    main()
