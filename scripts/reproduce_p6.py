"""One-command driver for P6-CAL protocol, six scoring branches, and main tables."""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def run(script, *arguments):
    command = [sys.executable, str(SCRIPTS / script), *map(str, arguments)]
    subprocess.run(command, cwd=REPO, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", help="GBS1 root with good/ and bad/; or set AD_DATA_ROOT")
    parser.add_argument("--split", help="GBS1 split JSON; or set GBS1_SPLIT")
    parser.add_argument("--output", default="outputs/p6_cal")
    parser.add_argument("--config", default=str(REPO / "configs" / "p6_cal.json"))
    parser.add_argument("--dinov2-repo", help="Official DINOv2 clone; or set DINOV2_REPO")
    parser.add_argument("--dinomaly2-repo", help="Official Dinomaly2 clone; or set DINOMALY2_REPO")
    parser.add_argument("--v4-checkpoint", help="V4 checkpoint; or set V4_CHECKPOINT")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Only recompute tables from existing score NPZ files")
    args = parser.parse_args()
    root = Path(args.output)
    protocol = root / "protocol.json"
    scores = root / "scores"
    tables = root / "tables"

    if not args.evaluate_only:
        split = args.split or __import__("os").environ.get("GBS1_SPLIT")
        if not split:
            parser.error("Supply --split or set GBS1_SPLIT")
        run("prepare_protocol.py", "--split", split, "--config", args.config, "--output", protocol)
        shared = ["--protocol", protocol, "--output", scores, "--config", args.config]
        if args.data_root:
            shared += ["--data-root", args.data_root]
        for backbone in ["v4", "official_vitb", "official_vits"]:
            extra = []
            if backbone == "v4":
                if args.dinov2_repo:
                    extra += ["--dinov2-repo", args.dinov2_repo]
                if args.v4_checkpoint:
                    extra += ["--v4-checkpoint", args.v4_checkpoint]
            run("score_knn.py", "--backbone", backbone, *shared, *extra)
        for branch in ["official_vitb_dinomaly2", "v4_dinomaly2_252", "v4_dinomaly2_224"]:
            extra = []
            if args.dinomaly2_repo:
                extra += ["--dinomaly2-repo", args.dinomaly2_repo]
            if args.dinov2_repo:
                extra += ["--dinov2-repo", args.dinov2_repo]
            if args.v4_checkpoint:
                extra += ["--v4-checkpoint", args.v4_checkpoint]
            run("score_reconstruction.py", "--branch", branch, *shared, *extra)

    run("evaluate_p6.py", "--protocol", protocol, "--scores", scores,
        "--output", tables, "--config", args.config)


if __name__ == "__main__":
    main()
