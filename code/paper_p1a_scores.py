# -*- coding: utf-8 -*-
import argparse
import os
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", required=True, choices=["vitb", "v4"])
parser.add_argument("--gpu", type=int, default=0)
ARGS = parser.parse_args()

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as C  # noqa: E402


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", str(C.ROOT / ".hf_cache"))
    device = torch.device("cuda")

    out_dir = C.PAPER_DIR / "p1"
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_paths, te_g, te_b = C.split_paths(C.SEED_ORIG)
    te_paths = te_g + te_b
    files_test = [p.name for p in te_paths]
    files_loo = [p.name for p in tr_paths]
    gt = np.array([0] * len(te_g) + [1] * len(te_b))
    print(f"GBS1: bank={len(tr_paths)} | test {len(te_g)} good + {len(te_b)} bad")

    if ARGS.backbone == "vitb":
        backbone = C.load_vitb(device)
        layers = [6, 10]
        print("building ViT-B L6/L10 memory bank...")
        banks = C.build_bank_vitb(backbone, tr_paths, layers, device)
        print("inferring test set...")
        s_test = C.knn_scores_vitb(backbone, te_paths, banks, layers, device)
        print("inferring LOO training set (for statistical thresholds)...")
        s_loo = C.knn_scores_vitb_loo(backbone, tr_paths, banks, layers, device)
        s_geo, s_rk = C.ens_geo(s_test[6], s_test[10]), C.ens_rank(s_test[6], s_test[10])
        loo_geo, loo_rk = C.ens_geo(s_loo[6], s_loo[10]), C.ens_rank(s_loo[6], s_loo[10])
        np.savez(out_dir / "scores_vitb.npz",
                 files_test=np.array(files_test), gt=gt,
                 s_L6=s_test[6], s_L10=s_test[10], s_geo=s_geo, s_rank=s_rk,
                 loo_L6=s_loo[6], loo_L10=s_loo[10], loo_geo=loo_geo, loo_rank=loo_rk,
                 files_loo=np.array(files_loo))
        for name, s in [("L6", s_test[6]), ("L10", s_test[10]), ("geo", s_geo), ("rank", s_rk)]:
            m = C.pr_metrics(s, gt)
            print(f"  [test] {name:5s}: AUROC={m['auroc']:.4f} AP={m['ap']:.4f} "
                  f"bestF1={m['best_f1']:.4f}")
        print(f"saved: {out_dir / 'scores_vitb.npz'}")

    else:
        backbone, it = C.load_v4(C.V4_CKPT, device)
        print(f"loading V4 iter_{it}")
        print("building V4 L11 memory bank...")
        bank = C.build_bank_v4(backbone, tr_paths, 11, device)
        print("inferring test set...")
        s_test = C.knn_scores_v4(backbone, te_paths, bank, 11, device)
        print("inferring LOO training set (for statistical thresholds)...")
        s_loo = C.knn_scores_v4_loo(backbone, tr_paths, bank, 11, device)
        np.savez(out_dir / "scores_v4.npz",
                 files_test=np.array(files_test), gt=gt, s_L11=s_test,
                 loo_L11=s_loo, files_loo=np.array(files_loo))
        m = C.pr_metrics(s_test, gt)
        print(f"  [test] L11top1: AUROC={m['auroc']:.4f} AP={m['ap']:.4f} bestF1={m['best_f1']:.4f}")
        print(f"saved: {out_dir / 'scores_v4.npz'}")


if __name__ == "__main__":
    main()