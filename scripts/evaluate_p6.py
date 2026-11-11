"""Recompute Table 3, Table 10, and paired tests from aligned per-image scores."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bootstrap import delong_paired, mcnemar_exact, paired_bootstrap  # noqa: E402
from common import DEFAULT_CONFIG, load_config, load_protocol, save_json  # noqa: E402
from metrics import evaluate  # noqa: E402


def source_map(config):
    sources = {}
    for branch in config["knn"].values():
        sources[branch["label"]] = branch["output"]
    for branch in config["reconstruction"]["branches"].values():
        sources[branch["label"]] = branch["output"]
    return sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="outputs/p6_cal/protocol.json")
    parser.add_argument("--scores", default="outputs/p6_cal/scores")
    parser.add_argument("--output", default="outputs/p6_cal/tables")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--bootstrap-samples", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    evaluation = config["evaluation"]
    protocol = load_protocol(args.protocol)
    names = evaluation["method_order"]
    filenames = source_map(config)
    calibration_names = protocol["calibration_normal"]
    final_names = protocol["test_normal"] + protocol["test_defective"]
    methods, gt_map, score_map, thresholds = {}, {}, {}, {}

    for name in names:
        path = Path(args.scores) / filenames[name]
        data = np.load(path, allow_pickle=False)
        lookup = {str(file): (int(gt), float(score))
                  for file, gt, score in zip(data["files_test"], data["gt"], data["pred_scores"])}
        expected = set(calibration_names) | set(final_names)
        if len(lookup) != len(data["files_test"]) or set(lookup) != expected:
            raise ValueError(f"Score file does not align with P6-CAL: {path}")
        calibration_scores = np.asarray([lookup[file][1] for file in calibration_names])
        gt = np.asarray([lookup[file][0] for file in final_names], dtype=np.int64)
        scores = np.asarray([lookup[file][1] for file in final_names])
        if list(gt) != [0] * len(protocol["test_normal"]) + [1] * len(protocol["test_defective"]):
            raise ValueError(f"Unexpected labels in {path}")
        methods[name] = evaluate(gt, scores, calibration_scores, evaluation["k_values"])
        gt_map[name], score_map[name] = gt, scores
        thresholds[name] = methods[name]["thresholds"]["mean+2sigma"]["threshold"]

    n_bootstrap = args.bootstrap_samples or evaluation["bootstrap_samples"]
    bootstrap, raw = paired_bootstrap(
        gt_map, score_map, thresholds, n_bootstrap,
        evaluation["bootstrap_seed"], evaluation["reference_method"],
    )
    reference = evaluation["reference_method"]
    delong = {name: delong_paired(gt_map[reference], score_map[reference], score_map[name])
              for name in names if name != reference}
    reference_prediction = score_map[reference] >= thresholds[reference]
    mcnemar = {name: mcnemar_exact(reference_prediction, score_map[name] >= thresholds[name])
               for name in names if name != reference}

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "p6_metrics.json", {
        "protocol": "P6-CAL: common 271 normal calibration and 271 normal + 19 defective final test",
        "point_estimate_rule": "computed on all 290 final-test images; Table 10 equals Table 3 (paper numbering)",
        "interval_rule": "2.5th and 97.5th percentiles of stratified paired bootstrap replicates",
        "methods": methods, "bootstrap": bootstrap,
        "delong_from_v4_knn": delong, "mcnemar_mean+2sigma_from_v4_knn": mcnemar,
    })
    np.savez_compressed(output / "table11_bootstrap_raw.npz",
                        **{f"values_{index}": raw[name] for index, name in enumerate(names)},
                        method_order=np.asarray(names))

    k_values = evaluation["k_values"]
    with (output / "table3.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "AUROC", "AP", "best_f1"] + [f"f1@mean+{k:g}sigma" for k in k_values]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in names:
            row = {"method": name, "AUROC": f"{methods[name]['auroc'] * 100:.1f}",
                   "AP": f"{methods[name]['ap'] * 100:.1f}",
                   "best_f1": f"{methods[name]['best_f1'] * 100:.1f}"}
            row.update({f"f1@mean+{k:g}sigma":
                        f"{methods[name]['thresholds'][f'mean+{k:g}sigma']['f1'] * 100:.1f}"
                        for k in k_values})
            writer.writerow(row)

    metric_keys = ["auroc", "ap", "best_f1"]
    with (output / "table11.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "AUROC [95% CI]", "AP [95% CI]",
                         "best_f1 [95% CI]", "f1@mean+2sigma [95% CI]"])
        for index, name in enumerate(names):
            point_values = [methods[name][key] for key in metric_keys] + [
                methods[name]["thresholds"]["mean+2sigma"]["f1"]]
            interval = bootstrap["intervals"][name]
            cells = [f"{point * 100:.1f} [{lo * 100:.1f}, {hi * 100:.1f}]"
                     for point, lo, hi in zip(point_values, interval["ci95_lo"], interval["ci95_hi"])]
            writer.writerow([name] + cells)
    print(f"Wrote Table 3, Table 10, raw metrics, and bootstrap replicates to {output}")


if __name__ == "__main__":
    main()
