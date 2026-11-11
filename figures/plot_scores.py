"""Plot final-test normal/defective score distributions from a generated score NPZ."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", help="Generated NPZ containing files_test, gt, and pred_scores")
    parser.add_argument("--protocol", default="outputs/p6_cal/protocol.json")
    parser.add_argument("--output", default="outputs/p6_cal/score_distributions.png")
    parser.add_argument("--title", default="P6-CAL image-level anomaly scores")
    args = parser.parse_args()
    import json

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    final_names = protocol["test_normal"] + protocol["test_defective"]
    data = np.load(args.scores, allow_pickle=False)
    lookup = {str(name): float(score) for name, score in zip(data["files_test"], data["pred_scores"])}
    scores = np.asarray([lookup[name] for name in final_names])
    normal = scores[:len(protocol["test_normal"])]
    defective = scores[len(protocol["test_normal"]):]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(scores.min(), scores.max(), 30)
    ax.hist(normal, bins=bins, alpha=0.65, label=f"Normal (n={len(normal)})")
    ax.hist(defective, bins=bins, alpha=0.75, label=f"Defective (n={len(defective)})")
    ax.set(xlabel="Image-level anomaly score", ylabel="Image count", title=args.title)
    ax.legend()
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
