# -*- coding: utf-8 -*-
import os
import sys
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.v2 import InterpolationMode
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
GOOD_DIR = ROOT / "datasets" / "lj_gb" / "good"
BAD_DIR = ROOT / "datasets" / "lj_gb" / "bad"
UXGH_GOOD_DIR = ROOT / "datasets" / "lj_uxgh" / "good"
SPLIT_ORIG = ROOT / "datasets" / "lj_gb" / "split.json"
V4_CKPT = ROOT / "results" / "gb" / "dinov2_scratch_v4" / "checkpoints_backup" / "iter_11000.pth"
PAPER_DIR = ROOT / "results" / "paper"
LOG_DIR = PAPER_DIR / "logs"
DINOMALY2_OFFICIAL_DIR = ROOT / "results" / "analysis_gb" / "dinomaly2"
DINOMALY2_V4_DIR = ROOT / "results" / "analysis_gb" / "dinomaly2_small_selftrained_11000"

IMG_EXTS = ("*.JPG", "*.jpg", "*.jpeg", "*.png")
NUM_NEIGHBOURS = 3
TOP_FRAC = 0.01
SEED_ORIG = 1


def _list_images(directory: Path):
    files = {}
    for pat in IMG_EXTS:
        for p in glob.glob(str(directory / pat)):
            files[p.lower()] = p
    return sorted(files.values())


def list_good():
    return [Path(p) for p in _list_images(GOOD_DIR)]


def list_bad():
    return [Path(p) for p in _list_images(BAD_DIR)]


def list_uxgh_good():
    return [Path(p) for p in _list_images(UXGH_GOOD_DIR)]


def make_split_file(seed: int, force: bool = False) -> Path:
    out = GOOD_DIR.parent / f"split_seed{seed}.json"
    if out.exists() and not force:
        return out
    good = [p.name for p in list_good()]
    bad = [p.name for p in list_bad()]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(good))
    n_train = int(len(good) * 0.8)
    d = {
        "seed": seed, "eval_ratio": 0.2, "good_total": len(good), "bad_total": len(bad),
        "main_pretrain_files": [good[i] for i in idx[:n_train]],
        "eval_heldout_files": [good[i] for i in idx[n_train:]],
        "bad_files": bad,
    }
    out.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load_split(seed: int) -> dict:
    if seed == SEED_ORIG:
        return json.loads(SPLIT_ORIG.read_text(encoding="utf-8"))
    p = make_split_file(seed)
    return json.loads(p.read_text(encoding="utf-8"))


def split_paths(seed: int):
    sp = load_split(seed)
    tr = [GOOD_DIR / n for n in sp["main_pretrain_files"]]
    te_g = [GOOD_DIR / n for n in sp["eval_heldout_files"]]
    te_b = [BAD_DIR / n for n in sp["bad_files"]]
    return tr, te_g, te_b


def v4_train_names() -> set:
    sp = json.loads(SPLIT_ORIG.read_text(encoding="utf-8"))
    return set(sp["main_pretrain_files"])


def verify_seed1_reproducibility() -> bool:
    sp = json.loads(SPLIT_ORIG.read_text(encoding="utf-8"))
    good = [p.name for p in list_good()]
    rng = np.random.RandomState(SEED_ORIG)
    idx = rng.permutation(len(good))
    n_train = int(len(good) * 0.8)
    return ([good[i] for i in idx[:n_train]] == sp["main_pretrain_files"]
            and [good[i] for i in idx[n_train:]] == sp["eval_heldout_files"])


class VitbDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.transform = transforms.Compose([
            transforms.Resize((252, 252), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


class V4Dataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB").resize((224, 224), Image.BILINEAR)
        x = transforms.functional.to_tensor(img)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (x - mean) / std


def load_vitb(device):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    from anomalib.models.components.dinov2 import DinoV2Loader
    m = DinoV2Loader().load("dinov2_vit_base_14").to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def load_v4(ckpt, device):
    prev_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    import train_gb_dinov2_scratch_v2 as T
    os.environ["CUDA_VISIBLE_DEVICES"] = prev_gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    from functools import partial
    import torch.nn as nn
    from dinov2.layers.dino_head import DINOHead

    def make_backbone(dp):
        return T.vit_small(patch_size=T.PATCH_SIZE, img_size=T.IMG_SIZE,
                           init_values=T.LAYERSCALE, num_register_tokens=0,
                           drop_path_rate=dp, drop_path_uniform=True,
                           interpolate_antialias=False, interpolate_offset=0.1)

    head_fn = partial(DINOHead, in_dim=T.EMBED_DIM, out_dim=T.DINO_N_PROTOTYPES,
                      hidden_dim=T.DINO_HIDDEN_DIM, bottleneck_dim=T.DINO_BOTTLENECK_DIM,
                      nlayers=T.DINO_NLAYERS)
    teacher = nn.ModuleDict({"backbone": make_backbone(0.0), "dino_head": head_fn()}).to(device)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    teacher.load_state_dict(state["teacher"])
    backbone = teacher["backbone"].eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone, int(state.get("iteration", -1))


@torch.no_grad()
def build_bank_vitb(backbone, paths, layers, device, batch_size=4):
    loader = DataLoader(VitbDataset(paths), batch_size=batch_size, shuffle=False, num_workers=2)
    banks = {l: [] for l in layers}
    for batch in loader:
        feats = backbone.get_intermediate_layers(batch.to(device), n=list(layers), norm=True,
                                                 reshape=False, return_class_token=False)
        for l, f in zip(layers, feats):
            b, n, d = f.shape
            banks[l].append(F.normalize(f.reshape(b * n, d), p=2, dim=1).cpu())
    return {l: torch.vstack(v).to(device) for l, v in banks.items()}


@torch.no_grad()
def knn_scores_vitb(backbone, paths, banks, layers, device, batch_size=4,
                    nn=NUM_NEIGHBOURS, top_frac=TOP_FRAC):
    loader = DataLoader(VitbDataset(paths), batch_size=batch_size, shuffle=False, num_workers=2)
    out = {l: [] for l in layers}
    for batch in loader:
        feats = backbone.get_intermediate_layers(batch.to(device), n=list(layers), norm=True,
                                                 reshape=False, return_class_token=False)
        for l, f in zip(layers, feats):
            b_, n_, d_ = f.shape
            fl = F.normalize(f.reshape(b_ * n_, d_), p=2, dim=1)
            sim = torch.matmul(fl, banks[l].T)
            dists = (1 - sim).clamp(min=0.0, max=2.0)
            tvals, _ = torch.topk(dists, k=nn, dim=1, largest=False)
            md = tvals.mean(dim=1).reshape(b_, n_)
            num_top = max(int(n_ * top_frac), 1)
            tk, _ = torch.topk(md, num_top, dim=1, largest=True)
            out[l].append(tk.mean(dim=1).cpu())
    return {l: torch.cat(v).numpy() for l, v in out.items()}


@torch.no_grad()
def build_bank_v4(backbone, paths, layer, device, batch_size=24):
    loader = DataLoader(V4Dataset(paths), batch_size=batch_size, shuffle=False, num_workers=2)
    cols = []
    for batch in loader:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            f = backbone.get_intermediate_layers(batch.to(device), n=[layer], norm=True)[0]
        f = F.normalize(f, dim=-1).float().cpu()
        cols.append(f.reshape(-1, f.shape[-1]))
    return torch.cat(cols).to(device)


@torch.no_grad()
def knn_scores_v4(backbone, paths, bank, layer, device, batch_size=24,
                  nn=NUM_NEIGHBOURS, top_frac=TOP_FRAC):
    loader = DataLoader(V4Dataset(paths), batch_size=batch_size, shuffle=False, num_workers=2)
    results = []
    for batch in loader:
        x = batch.to(device)
        b = x.shape[0]
        with torch.amp.autocast("cuda", dtype=torch.float16):
            f = backbone.get_intermediate_layers(x, n=[layer], norm=True)[0]
        f = F.normalize(f, dim=-1).float()
        bn, n, d = f.shape
        f = f.reshape(bn * n, d)
        sim = torch.matmul(f, bank.T)
        best = sim.topk(nn, dim=-1).values
        md = (1 - best.mean(-1)).reshape(b, n)
        num_top = max(int(n * top_frac), 1)
        results.append(md.topk(num_top, dim=1).values.mean(-1).cpu())
    return torch.cat(results).numpy()


@torch.no_grad()
def knn_scores_vitb_loo(backbone, paths, banks, layers, device, batch_size=2,
                        nn=NUM_NEIGHBOURS, top_frac=TOP_FRAC):
    loader = DataLoader(VitbDataset(paths), batch_size=batch_size, shuffle=False, num_workers=2)
    n_patch = None
    out = {l: [] for l in layers}
    base = 0
    for batch in loader:
        feats = backbone.get_intermediate_layers(batch.to(device), n=list(layers), norm=True,
                                                 reshape=False, return_class_token=False)
        b_ = batch.shape[0]
        for l, f in zip(layers, feats):
            b, n, d = f.shape
            n_patch = n
            fl = F.normalize(f.reshape(b * n, d), p=2, dim=1)
            sim = torch.matmul(fl, banks[l].T)
            for j in range(b):
                rows = slice(j * n, (j + 1) * n)
                cols = slice((base + j) * n, (base + j + 1) * n)
                sim[rows, cols] = -1e9
            tvals, _ = torch.topk(sim, k=nn, dim=1, largest=True)
            md = (1 - tvals.mean(dim=1)).reshape(b, n)
            num_top = max(int(n * top_frac), 1)
            tk, _ = torch.topk(md, num_top, dim=1, largest=True)
            out[l].append(tk.mean(dim=1).cpu())
        base += b_
    return {l: torch.cat(v).numpy() for l, v in out.items()}


@torch.no_grad()
def knn_scores_v4_loo(backbone, paths, bank, layer, device, batch_size=8,
                      nn=NUM_NEIGHBOURS, top_frac=TOP_FRAC):
    loader = DataLoader(V4Dataset(paths), batch_size=batch_size, shuffle=False, num_workers=2)
    results = []
    base = 0
    for batch in loader:
        x = batch.to(device)
        b = x.shape[0]
        with torch.amp.autocast("cuda", dtype=torch.float16):
            f = backbone.get_intermediate_layers(x, n=[layer], norm=True)[0]
        f = F.normalize(f, dim=-1).float()
        bn, n, d = f.shape
        f = f.reshape(bn * n, d)
        sim = torch.matmul(f, bank.T).float()
        for j in range(b):
            rows = slice(j * n, (j + 1) * n)
            cols = slice((base + j) * n, (base + j + 1) * n)
            sim[rows, cols] = -1e9
        best = sim.topk(nn, dim=-1).values
        md = (1 - best.mean(-1)).reshape(b, n)
        num_top = max(int(n * top_frac), 1)
        results.append(md.topk(num_top, dim=1).values.mean(-1).cpu())
        base += b
    return torch.cat(results).numpy()


def ens_geo(s_a, s_b):
    def _sc(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9) + 1e-6
    return np.exp((np.log(_sc(s_a)) + np.log(_sc(s_b))) / 2)


def ens_rank(s_a, s_b):
    ra = np.argsort(np.argsort(s_a)).astype(float)
    rb = np.argsort(np.argsort(s_b)).astype(float)
    return (ra + rb) / 2.0


def pr_metrics(scores, gt, thresholds=None):
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
    scores = np.asarray(scores, dtype=np.float64)
    gt = np.asarray(gt).astype(int)
    prec, rec, thr = precision_recall_curve(gt, scores)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-10, None)
    i = int(np.argmax(f1[:-1]))
    out = {
        "auroc": float(roc_auc_score(gt, scores)),
        "ap": float(average_precision_score(gt, scores)),
        "best_f1": float(f1[i]), "best_p": float(prec[i]), "best_r": float(rec[i]),
        "best_threshold": float(thr[i]),
    }
    for name, t in (thresholds or {}).items():
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (gt == 1)).sum()); fp = int(((pred == 1) & (gt == 0)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum()); tn = int(((pred == 0) & (gt == 0)).sum())
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        out[f"f1@{name}"] = float(2 * p * r / max(p + r, 1e-10))
        out[f"p@{name}"] = float(p); out[f"r@{name}"] = float(r)
        out[f"cm@{name}"] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
    return out


def stat_thresholds(normal_scores, ks=(2.0, 3.0, 4.0)):
    mu = float(np.mean(normal_scores)); sd = float(np.std(normal_scores))
    return {f"mean+{int(k)}s": mu + k * sd for k in ks}, {"mean": mu, "std": sd}


def bootstrap_ci(scores, gt, thresholds=None, n_boot=10000, seed=12345):
    from sklearn.metrics import roc_auc_score, average_precision_score
    scores = np.asarray(scores, dtype=np.float64)
    gt = np.asarray(gt).astype(int)
    n = len(gt)
    keys = ["auroc", "ap"] + [f"f1@{k}" for k in (thresholds or {})]
    vals = {k: [] for k in keys}
    rng = np.random.RandomState(seed)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        s, y = scores[idx], gt[idx]
        if y.sum() in (0, n):
            continue
        vals["auroc"].append(roc_auc_score(y, s))
        vals["ap"].append(average_precision_score(y, s))
        for name, t in (thresholds or {}).items():
            pred = s >= t
            tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
            vals[f"f1@{name}"].append(2 * p * r / max(p + r, 1e-10))
    return {k: {"mean": float(np.mean(v)), "lo95": float(np.percentile(v, 2.5)),
                 "hi95": float(np.percentile(v, 97.5))} for k, v in vals.items()}


def per_sample_bad(scores, gt, files, thresholds=None):
    scores = np.asarray(scores, dtype=np.float64)
    gt = np.asarray(gt).astype(int)
    order = np.argsort(-scores)
    ranks = np.empty_like(order); ranks[order] = np.arange(len(scores))
    rows = []
    bad_idx = np.where(gt == 1)[0]
    bad_order = np.argsort(-scores[bad_idx])
    bad_ranks = np.empty_like(bad_order); bad_ranks[bad_order] = np.arange(len(bad_idx))
    for ii, i in enumerate(bad_idx):
        row = {
            "file": Path(files[i]).name,
            "score": float(scores[i]),
            "rank_all": int(ranks[i]) + 1,
            "rank_bad": int(bad_ranks[ii]) + 1,
        }
        for name, t in (thresholds or {}).items():
            row[f"detected@{name}"] = bool(scores[i] >= t)
            row[f"margin@{name}"] = float(scores[i] - t)
        rows.append(row)
    return rows


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)