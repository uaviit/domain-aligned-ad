# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as C  # noqa: E402

BOOT_N = 10000


def analyze_cell(name, scores, gt, files, normal_scores, thresh_source, bootstrap=True):
    thresholds, stats = C.stat_thresholds(normal_scores)
    m = C.pr_metrics(scores, gt, thresholds)
    ci = C.bootstrap_ci(scores, gt, thresholds, n_boot=BOOT_N) if bootstrap else None
    per = C.per_sample_bad(scores, gt, files, thresholds)
    out = {
        "n_test": int(len(gt)), "n_bad": int(gt.sum()),
        "threshold_source": thresh_source,
        "normal_score_stats": stats,
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "metrics": m,
        "bootstrap_ci_95": ci,
        "per_sample_bad": per,
    }
    print(f"\n[{name}]  threshold source: {thresh_source}")
    print(f"  AUROC={m['auroc']:.4f}  AP={m['ap']:.4f}  bestF1={m['best_f1']:.4f} "
          f"(thr={m['best_threshold']:.4f})")
    for k in thresholds:
        print(f"  {k}: F1={m[f'f1@{k}']:.4f} P={m[f'p@{k}']:.4f} R={m[f'r@{k}']:.4f} "
              f"cm={m[f'cm@{k}']}")
    if ci:
        print(f"  CI95: AUROC=[{ci['auroc']['lo95']:.4f},{ci['auroc']['hi95']:.4f}] "
              f"AP=[{ci['ap']['lo95']:.4f},{ci['ap']['hi95']:.4f}] "
              f"F1@mean+3s=[{ci['f1@mean+3s']['lo95']:.4f},{ci['f1@mean+3s']['hi95']:.4f}]")
    return out


def main():
    out_dir = C.PAPER_DIR / "p1"
    results = {"bootstrap_n": BOOT_N, "protocol_note": {
        "knn_cells": "threshold from mean+k*sigma of the 2167 training-image LOO scores (no test leakage)",
        "dinomaly_cells": "threshold from mean+k*sigma of the test-good scores (transductive; must be flagged in the paper)",
        "ap": "Average Precision, threshold-free, primary metric under the unbalanced 19/561 setting",
    }}

    z = np.load(out_dir / "scores_vitb.npz", allow_pickle=True)
    gt, files = z["gt"], [str(x) for x in z["files_test"]]
    results["adino_vitb_L6L10_rank"] = analyze_cell(
        "A-DINO ViT-B L6+L10 rank", z["s_rank"].astype(np.float64), gt, files,
        z["loo_rank"].astype(np.float64), "train-LOO rank ensemble scores")
    results["adino_vitb_L6L10_geo"] = analyze_cell(
        "A-DINO ViT-B L6+L10 geo", z["s_geo"].astype(np.float64), gt, files,
        z["loo_geo"].astype(np.float64), "train-LOO geo ensemble scores")

    z = np.load(out_dir / "scores_v4.npz", allow_pickle=True)
    results["ads0_v4_L11top1"] = analyze_cell(
        "AD-S0-11k V4 L11 top1", z["s_L11"].astype(np.float64), z["gt"],
        [str(x) for x in z["files_test"]],
        z["loo_L11"].astype(np.float64), "train-LOO L11 scores")

    r1_split = C.make_split_file(1, force=True)
    r1 = json.loads(r1_split.read_text(encoding="utf-8"))
    for key, d, label, division_note in [
        ("dinomaly2_official", C.DINOMALY2_OFFICIAL_DIR, "Dinomaly2 official ViT-S 252",
         "RandomState(1) split"),
        ("dinomaly2_v4", C.DINOMALY2_V4_DIR, "Dinomaly2 + V4 iter11000",
         "RandomState(1) split; V4 pre-trained on the split.json train set, so 426/542 "
         "test-good images were seen during V4 pre-training (transductive leakage, favors "
         "the self-trained backbone; that cell is nevertheless the worst, so the "
         "conclusion is robust)"),
    ]:
        scores = np.load(d / "pred_scores.npy").astype(np.float64)
        gt2 = np.load(d / "gt_labels.npy").astype(int)
        te_g_names = r1["eval_heldout_files"]
        te_b_names = r1["bad_files"]
        files2 = te_g_names + te_b_names
        gt_assumed = np.array([0] * len(te_g_names) + [1] * len(te_b_names))
        if len(files2) != len(scores) or not np.array_equal(gt_assumed, gt2):
            print(f"[warning] {key}: gt layout differs from assumption, rebuilding file names by index")
            files2 = [f"idx_{i}" for i in range(len(scores))]
        results[key] = analyze_cell(label, scores, gt2, files2,
                                    scores[gt2 == 0], "test-good transductive scores")
        results[key]["division"] = division_note
        results[key] = analyze_cell(label, scores, gt2, files2,
                                    scores[gt2 == 0], "test-good transductive scores")

    C.save_json(out_dir / "p1_threshold_ap_ci.json", results)

    csv_lines = ["file," + ",".join([
        "rank_score_v4", "rank_all_v4", "detected_3s_v4",
        "rank_score_vitb", "rank_all_vitb", "detected_3s_vitb",
        "rank_score_din2", "detected_3s_din2"])]
    zv = np.load(out_dir / "scores_v4.npz", allow_pickle=True)
    zb = np.load(out_dir / "scores_vitb.npz", allow_pickle=True)
    zd = np.load(C.DINOMALY2_OFFICIAL_DIR / "pred_scores.npy").astype(np.float64)
    gtd = np.load(C.DINOMALY2_OFFICIAL_DIR / "gt_labels.npy").astype(int)
    t_v4 = results["ads0_v4_L11top1"]["thresholds"]["mean+3s"]
    t_vb = results["adino_vitb_L6L10_rank"]["thresholds"]["mean+3s"]
    t_d2 = results["dinomaly2_official"]["thresholds"]["mean+3s"]
    _, te_g, te_b = C.split_paths(C.SEED_ORIG)
    files_v = [str(x) for x in zv["files_test"]]
    rows_v = {r["file"]: r for r in results["ads0_v4_L11top1"]["per_sample_bad"]}
    rows_b = {r["file"]: r for r in results["adino_vitb_L6L10_rank"]["per_sample_bad"]}
    d_order = np.argsort(-zd)
    d_ranks = np.empty_like(d_order); d_ranks[d_order] = np.arange(len(zd))
    d_files = [p.name for p in te_g] + [p.name for p in te_b]
    for i, fn in enumerate(files_v):
        if zv["gt"][i] != 1:
            continue
        rv, rb = rows_v[fn], rows_b.get(fn, {})
        di = d_files.index(fn) if fn in d_files else None
        csv_lines.append(",".join([
            fn,
            f"{rv['score']:.6f}", str(rv["rank_all"]), str(rv.get("detected@mean+3s", "")),
            f"{rb.get('score', float('nan')):.6f}", str(rb.get("rank_all", "")),
            str(rb.get("detected@mean+3s", "")),
            f"{zd[di]:.6f}" if di is not None else "",
            str(bool(zd[di] >= t_d2)) if di is not None else "",
        ]))
    (out_dir / "per_sample_19bad.csv").write_text("\n".join(csv_lines), encoding="utf-8")
    print(f"\nsaved: {out_dir / 'p1_threshold_ap_ci.json'}")
    print(f"saved: {out_dir / 'per_sample_19bad.csv'}")


if __name__ == "__main__":
    main()