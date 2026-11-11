"""Train one fixed-iteration Dinomaly2 head and save held-out image scores."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import (DEFAULT_CONFIG, load_config, load_protocol,  # noqa: E402
                    save_json, supplied_path)
from reconstruction import run_reconstruction  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", required=True,
                        choices=["official_vitb_dinomaly2", "v4_dinomaly2_252", "v4_dinomaly2_224"])
    parser.add_argument("--protocol", default="outputs/p6_cal/protocol.json")
    parser.add_argument("--data-root", help="GBS1 root with good/ and bad/; or set AD_DATA_ROOT")
    parser.add_argument("--output", default="outputs/p6_cal/scores")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dinomaly2-repo", help="Official Dinomaly2 clone; or set DINOMALY2_REPO")
    parser.add_argument("--dinov2-repo", help="Official DINOv2 clone; or set DINOV2_REPO")
    parser.add_argument("--v4-checkpoint", help="V4 checkpoint; or set V4_CHECKPOINT")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    branch = config["reconstruction"]["branches"][args.branch]
    dinomaly2_repo = supplied_path(args.dinomaly2_repo, "DINOMALY2_REPO")
    needs_v4 = branch["backbone"] == "v4_student"
    dinov2_repo = supplied_path(args.dinov2_repo, "DINOV2_REPO", required=needs_v4)
    checkpoint = supplied_path(args.v4_checkpoint, "V4_CHECKPOINT", required=needs_v4)
    output = Path(args.output) / branch["output"]
    metadata = run_reconstruction(
        args.branch, branch, config["reconstruction"], load_protocol(args.protocol),
        supplied_path(args.data_root, "AD_DATA_ROOT"), output, dinomaly2_repo,
        dinov2_repo=dinov2_repo, v4_checkpoint=checkpoint, workers=args.workers,
    )
    save_json(output.with_suffix(".run.json"), metadata)
    print(f"Wrote {metadata['num_scores']} scores: {output}")


if __name__ == "__main__":
    main()
