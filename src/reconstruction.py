"""Train and score a P6-CAL Dinomaly2 head using the official external implementation."""
import random
import re
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ReconstructionDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), self.labels[index], Path(self.paths[index]).name


def _seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _v4_student(checkpoint, device):
    from dinov2.models import vision_transformer as vtd2

    encoder = vtd2.vit_small(
        patch_size=14, img_size=224, block_chunks=0, init_values=1e-5,
        num_register_tokens=0, interpolate_antialias=False, interpolate_offset=0.1,
    )
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    backbone = {key[len("backbone."):]: value for key, value in state["student"].items()
                if key.startswith("backbone.")}
    remapped = {re.sub(r"^blocks\.0\.", "blocks.", key): value for key, value in backbone.items()}
    message = encoder.load_state_dict(remapped, strict=False)
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(f"V4 reconstruction checkpoint mismatch: {message}")
    return encoder.to(device)


def run_reconstruction(branch_name, branch_config, reconstruction_config, protocol,
                       data_root, output_path, dinomaly2_repo, dinov2_repo=None,
                       v4_checkpoint=None, workers=2):
    if not torch.cuda.is_available():
        raise RuntimeError("P6-CAL Dinomaly2 training requires CUDA")
    sys.path.insert(0, str(Path(dinomaly2_repo)))
    if dinov2_repo:
        sys.path.insert(0, str(Path(dinov2_repo)))
    from dataset import get_data_transforms
    from models import vit_encoder
    from models.uad import Dinomaly
    from models.vision_transformer import Block as VitBlock, LinearAttention2
    from optimizers import StableAdamW
    from utils import WarmupCosineScheduler, cal_anomaly_maps, get_gaussian_kernel, global_cosine_hm_percent

    device = torch.device("cuda:0")
    _seed(reconstruction_config["seed"])
    image_size = branch_config["image_size"]
    if branch_config["backbone"] == "official_vitb":
        weights_dir = Path(dinomaly2_repo) / "backbones" / "weights"
        encoder = vit_encoder.load("dinov2reg_vit_base_14", WEIGHTS_DIR=str(weights_dir))
    else:
        if not dinov2_repo or not v4_checkpoint:
            raise ValueError("V4 reconstruction requires DINOV2_REPO and V4_CHECKPOINT")
        encoder = _v4_student(v4_checkpoint, device)

    dimension = branch_config["embedding_dimension"]
    heads = branch_config["attention_heads"]
    dropout = branch_config["dropout"]
    bottleneck = nn.ModuleList([
        nn.Sequential(nn.Linear(dimension, 256), nn.Dropout(p=dropout)),
        nn.Sequential(nn.Linear(256, dimension * 4), nn.GELU(), nn.Dropout(p=dropout),
                      nn.Linear(dimension * 4, dimension), nn.Dropout(p=dropout)),
    ])
    decoder = nn.ModuleList([
        VitBlock(dim=dimension, num_heads=heads, mlp_ratio=4.0, qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-8),
                 attn=partial(LinearAttention2, eps=1e-8))
        for _ in range(8)
    ])
    model = Dinomaly(
        encoder=encoder, bottleneck=bottleneck, decoder=decoder,
        target_layers=reconstruction_config["target_layers"], remove_class_token=False,
        fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]], context_aware_recenter=True,
    ).to(device)
    model.init_weights()

    transform, _ = get_data_transforms(image_size, image_size)
    train_names = protocol["train_normal"]
    heldout_names = protocol["calibration_normal"] + protocol["test_normal"]
    defective_names = protocol["test_defective"]
    train_paths = [Path(data_root) / "good" / name for name in train_names]
    eval_paths = ([Path(data_root) / "good" / name for name in heldout_names]
                  + [Path(data_root) / "bad" / name for name in defective_names])
    eval_labels = [0] * len(heldout_names) + [1] * len(defective_names)
    batch_size = reconstruction_config["batch_size"]
    train_loader = DataLoader(ReconstructionDataset(train_paths, [0] * len(train_paths), transform),
                              batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    eval_loader = DataLoader(ReconstructionDataset(eval_paths, eval_labels, transform),
                             batch_size=batch_size, shuffle=False, num_workers=workers)

    trainable = nn.ModuleList([bottleneck, decoder])
    optimizer = StableAdamW(
        [{"params": bottleneck[0].parameters(), "lr": reconstruction_config["bottleneck0_learning_rate"]},
         {"params": bottleneck[1].parameters()}, {"params": decoder.parameters()}],
        lr=reconstruction_config["learning_rate"], betas=(0.9, 0.999),
        weight_decay=reconstruction_config["weight_decay"], amsgrad=False, eps=1e-10,
    )
    total = reconstruction_config["total_iterations"]
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=reconstruction_config["warmup_iterations"],
        total_epochs=total, final_ratio=reconstruction_config["final_learning_rate_ratio"],
    )

    iteration = 0
    while iteration < total:
        model.train()
        for images, _, _ in train_loader:
            if iteration >= total:
                break
            images = images.to(device)
            encoder_features, decoder_features = model(images)
            hardness = min(0.9 * iteration / 1000, 0.9)
            loss = global_cosine_hm_percent(encoder_features, decoder_features, p=hardness, factor=0.1)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=reconstruction_config["gradient_clip"])
            optimizer.step()
            scheduler.step()
            iteration += 1
            if iteration % 500 == 0:
                print(f"{branch_name}: iteration {iteration}/{total}, loss={loss.item():.6f}", flush=True)

    model.eval()
    files, labels, scores = [], [], []
    kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    with torch.no_grad():
        for images, label, names in eval_loader:
            images = images.to(device)
            output = model(images)
            anomaly_map, _ = cal_anomaly_maps(output[0], output[1], images.shape[-1])
            anomaly_map = F.interpolate(
                anomaly_map, size=reconstruction_config["evaluation_map_size"],
                mode="bilinear", align_corners=False,
            )
            anomaly_map = kernel(anomaly_map)
            flat = anomaly_map.flatten(1)
            count = int(flat.shape[1] * reconstruction_config["evaluation_top_fraction"])
            top = torch.sort(flat, dim=1, descending=True)[0][:, :count]
            files.extend(list(names))
            labels.extend(label.numpy().reshape(-1).tolist())
            scores.extend(top.mean(dim=1).cpu().numpy().tolist())

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, files_test=np.asarray(files),
                        gt=np.asarray(labels, dtype=np.int64),
                        pred_scores=np.asarray(scores, dtype=np.float32))
    return {"branch": branch_name, "iterations": total, "image_size": image_size,
            "seed": reconstruction_config["seed"], "num_scores": len(scores),
            "test_evaluation_during_training": False}
