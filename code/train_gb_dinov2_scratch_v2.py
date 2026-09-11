# -*- coding: utf-8 -*-
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
import json
import math
import random
import glob
import logging
import warnings
import numpy as np
from functools import partial
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

warnings.filterwarnings('ignore')

DINOV2_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DINOV2_PATH)

from dinov2.models.vision_transformer import vit_small
from dinov2.layers.dino_head import DINOHead
from dinov2.loss.dino_clstoken_loss import DINOLoss
from dinov2.loss.ibot_patch_loss import iBOTPatchLoss
from dinov2.loss.koleo_loss import KoLeoLoss
from dinov2.data.augmentations import DataAugmentationDINO
from dinov2.data.masking import MaskingGenerator
from dinov2.data.transforms import make_normalize_transform, GaussianBlur
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_JSON = os.path.join(_REPO_ROOT, 'splits', 'split.json')
DATA_ROOT = os.path.join(_REPO_ROOT, 'datasets', 'lj_gb', 'good')
OUTPUT_DIR = os.path.join(_REPO_ROOT, 'results', 'dinov2_scratch_v4')
CKPT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')

PATCH_SIZE = 14
IMG_SIZE = 224
EMBED_DIM = 384
DEPTH = 12
NUM_HEADS = 6
NUM_REGISTER_TOKENS = 0
DROP_PATH_RATE = 0.2
LAYERSCALE = 1e-5

TOTAL_ITERS = 18400
BATCH_SIZE = 96
BASE_LR = 1.22e-3
MIN_LR = 1e-6
WARMUP_ITERS = 920
WEIGHT_DECAY = 0.04
WEIGHT_DECAY_END = 0.4
CLIP_GRAD = 3.0
FREEZE_LAST_LAYER_ITERS = 184
LAYERWISE_DECAY = 0.9
PATCH_EMBED_LR_MULT = 0.2
SEED = 0
NUM_WORKERS = 2
SAVE_EVERY = 1000
LOG_EVERY = 50

DINO_LOSS_WEIGHT = 1.0
DINO_N_PROTOTYPES = 64
DINO_BOTTLENECK_DIM = 256
DINO_HIDDEN_DIM = 2048
DINO_NLAYERS = 3
STUDENT_TEMP = 0.1
CENTER_MOMENTUM = 0.9

IBOT_LOSS_WEIGHT = 1.0
IBOT_MASK_PROB = 0.5
IBOT_MASK_RATIO_MIN_MAX = [0.1, 0.5]

KOLEO_LOSS_WEIGHT = 0.1

MOMENTUM_TEACHER = 0.992
FINAL_MOMENTUM_TEACHER = 1.0

WARMUP_TEACHER_TEMP = 0.04
TEACHER_TEMP = 0.04
WARMUP_TEACHER_TEMP_ITERS = 1380

GLOBAL_CROPS_SIZE = 224
LOCAL_CROPS_SIZE = 98
LOCAL_CROPS_NUMBER = 10
GLOBAL_CROPS_SCALE = [0.25, 1.0]
LOCAL_CROPS_SCALE = [0.05, 0.25]

USE_FP16 = True


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(save_path):
    os.makedirs(save_path, exist_ok=True)
    logger = logging.getLogger('dinov2_scratch_v2')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(save_path, 'train.log'), encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


class CosineScheduler:
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))
        assert len(self.schedule) == total_iters, f"{len(self.schedule)} != {total_iters}"

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        return self.schedule[it]


class GBImageDataset(Dataset):
    def __init__(self, root, split_json, transform=None):
        with open(split_json, 'r', encoding='utf-8') as f:
            split = json.load(f)
        self.img_paths = [
            os.path.join(root, fname) for fname in split['main_pretrain_files']
        ]
        self.transform = transform
        logging.info(f"GBImageDataset (strict split): {len(self.img_paths)} images from {root}")
        logging.info(f"  - main_pretrain: {len(split['main_pretrain_files'])} (used for training)")
        logging.info(f"  - eval_heldout: {len(split['eval_heldout_files'])} (held out)")
        logging.info(f"  - bad: {split['bad_total']} (held out)")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, n_tokens, mask_generator, dtype=torch.half):
    n_global_crops = len(samples_list[0][0]['global_crops'])
    n_local_crops = len(samples_list[0][0]['local_crops'])

    collated_global_crops = torch.stack(
        [s[0]['global_crops'][i] for i in range(n_global_crops) for s in samples_list])
    collated_local_crops = torch.stack(
        [s[0]['local_crops'][i] for i in range(n_local_crops) for s in samples_list])

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))
    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        'collated_global_crops': collated_global_crops,
        'collated_local_crops': collated_local_crops,
        'collated_masks': collated_masks,
        'mask_indices_list': mask_indices_list,
        'masks_weight': masks_weight,
        'upperbound': upperbound,
        'n_masked_patches': torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }


class DINOv2Trainer:
    def __init__(self, logger):
        self.logger = logger
        self.device = torch.device('cuda:0')
        self.fp16_scaler = torch.amp.GradScaler('cuda') if USE_FP16 else None

        student_backbone = vit_small(
            patch_size=PATCH_SIZE, img_size=IMG_SIZE,
            init_values=LAYERSCALE, num_register_tokens=NUM_REGISTER_TOKENS,
            drop_path_rate=DROP_PATH_RATE, drop_path_uniform=True,
            interpolate_antialias=False, interpolate_offset=0.1,
        )
        teacher_backbone = vit_small(
            patch_size=PATCH_SIZE, img_size=IMG_SIZE,
            init_values=LAYERSCALE, num_register_tokens=NUM_REGISTER_TOKENS,
            drop_path_rate=0.0, drop_path_uniform=True,
            interpolate_antialias=False, interpolate_offset=0.1,
        )
        self.embed_dim = EMBED_DIM

        dino_head_fn = partial(DINOHead, in_dim=EMBED_DIM, out_dim=DINO_N_PROTOTYPES,
                               hidden_dim=DINO_HIDDEN_DIM, bottleneck_dim=DINO_BOTTLENECK_DIM,
                               nlayers=DINO_NLAYERS)
        self.student = nn.ModuleDict({
            'backbone': student_backbone,
            'dino_head': dino_head_fn(),
        }).to(self.device)
        self.teacher = nn.ModuleDict({
            'backbone': teacher_backbone,
            'dino_head': dino_head_fn(),
        }).to(self.device)

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.teacher.load_state_dict(self.student.state_dict())

        self.dino_loss = DINOLoss(DINO_N_PROTOTYPES, student_temp=STUDENT_TEMP,
                                   center_momentum=CENTER_MOMENTUM).to(self.device)
        self.ibot_patch_loss = iBOTPatchLoss(DINO_N_PROTOTYPES, student_temp=STUDENT_TEMP,
                                              center_momentum=CENTER_MOMENTUM).to(self.device)
        self.koleo_loss = KoLeoLoss().to(self.device)

        self.dtype = torch.float16 if USE_FP16 else torch.float32

    def update_teacher(self, m):
        with torch.no_grad():
            for k in self.student.keys():
                student_params = list(self.student[k].parameters())
                teacher_params = list(self.teacher[k].parameters())
                torch._foreach_mul_(teacher_params, m)
                torch._foreach_add_(teacher_params, student_params, alpha=1 - m)

    def forward_backward(self, images, teacher_temp):
        n_global_crops = 2
        n_local_crops = LOCAL_CROPS_NUMBER

        global_crops = images['collated_global_crops'].to(self.device)
        local_crops = images['collated_local_crops'].to(self.device)
        masks = images['collated_masks'].to(self.device)
        mask_indices_list = images['mask_indices_list'].to(self.device)
        n_masked_patches = images['n_masked_patches'].to(self.device)
        upperbound = images['upperbound']
        masks_weight = images['masks_weight'].to(self.device)

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops
        ibot_loss_scale = 1.0 / n_global_crops

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            teacher_backbone_output = self.teacher.backbone(global_crops, is_training=True)
            teacher_cls_tokens = teacher_backbone_output['x_norm_clstoken']
            teacher_cls_tokens_chunked = teacher_cls_tokens.chunk(n_global_crops)
            teacher_cls_tokens = torch.cat((teacher_cls_tokens_chunked[1], teacher_cls_tokens_chunked[0]))
            teacher_patch_tokens = teacher_backbone_output['x_norm_patchtokens']
            _dim = teacher_patch_tokens.shape[-1]
            n_cls = teacher_cls_tokens.shape[0]

            buffer_tensor_teacher = teacher_patch_tokens.new_zeros(upperbound + n_cls, _dim)
            buffer_tensor_teacher[:n_cls].copy_(teacher_cls_tokens)
            torch.index_select(teacher_patch_tokens.flatten(0, 1), dim=0,
                               index=mask_indices_list,
                               out=buffer_tensor_teacher[n_cls:n_cls + n_masked_patches])
            tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
            teacher_cls_tokens_after_head = tokens_after_head[:n_cls]
            masked_teacher_patch_tokens_after_head = tokens_after_head[n_cls:n_cls + n_masked_patches]

            teacher_dino_softmaxed_centered = self.dino_loss.softmax_center_teacher(
                teacher_cls_tokens_after_head, teacher_temp=teacher_temp
            ).view(n_global_crops, -1, DINO_N_PROTOTYPES)
            self.dino_loss.update_center(teacher_cls_tokens_after_head)

            masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
            masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
            ).squeeze(0)
            self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])

        with torch.amp.autocast('cuda', dtype=torch.float16):
            student_global_output = self.student.backbone(global_crops, masks=masks, is_training=True)
            student_local_output = self.student.backbone(local_crops, is_training=True)

            student_global_cls = student_global_output['x_norm_clstoken']
            student_local_cls = student_local_output['x_norm_clstoken']

            student_global_patch_tokens = student_global_output['x_norm_patchtokens']
            _dim_s = student_global_patch_tokens.shape[-1]
            buffer_tensor_student = student_global_patch_tokens.new_zeros(upperbound, _dim_s)
            buffer_tensor_student[:n_masked_patches].copy_(
                torch.index_select(student_global_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )

            student_local_cls_after_head = self.student.dino_head(student_local_cls)
            student_global_cls_after_head = self.student.dino_head(student_global_cls)
            student_global_masked_patch_after_head = self.student.dino_head(buffer_tensor_student)[:n_masked_patches]

        loss_dict = {}
        loss_accumulator = 0

        if n_local_crops > 0:
            dino_local_loss = self.dino_loss(
                student_output_list=student_local_cls_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            loss_dict['dino_local'] = dino_local_loss
            loss_accumulator += DINO_LOSS_WEIGHT * dino_local_loss

        loss_scales = 2
        dino_global_loss = self.dino_loss(
            student_output_list=[student_global_cls_after_head],
            teacher_out_softmaxed_centered_list=[teacher_dino_softmaxed_centered.flatten(0, 1)],
        ) * loss_scales / (n_global_crops_loss_terms + n_local_crops_loss_terms)
        loss_dict['dino_global'] = dino_global_loss
        loss_accumulator += DINO_LOSS_WEIGHT * dino_global_loss

        if KOLEO_LOSS_WEIGHT > 0:
            koleo_loss = KOLEO_LOSS_WEIGHT * sum(
                self.koleo_loss(p) for p in student_global_cls.chunk(2)
            )
            loss_accumulator += koleo_loss
            loss_dict['koleo'] = koleo_loss / loss_scales

        ibot_loss = self.ibot_patch_loss.forward_masked(
            student_global_masked_patch_after_head,
            masked_teacher_ibot_softmaxed_centered,
            student_masks_flat=masks,
            n_masked_patches=n_masked_patches,
            masks_weight=masks_weight,
        ) * loss_scales * ibot_loss_scale
        loss_dict['ibot'] = ibot_loss / 2
        loss_accumulator += IBOT_LOSS_WEIGHT * ibot_loss

        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss_accumulator).backward()
        else:
            loss_accumulator.backward()

        return loss_dict

    def clip_grad(self, clip_val):
        if self.fp16_scaler is not None:
            self.fp16_scaler.unscale_(self.optimizer)
        for v in self.student.values():
            torch.nn.utils.clip_grad_norm_(v.parameters(), clip_val)

    def step(self):
        if self.fp16_scaler is not None:
            self.fp16_scaler.step(self.optimizer)
            self.fp16_scaler.update()
        else:
            self.optimizer.step()

    def get_params_groups(self):
        all_groups = []
        for m in self.student.values():
            groups = get_params_groups_with_decay(
                model=m, lr_decay_rate=LAYERWISE_DECAY,
                patch_embed_lr_mult=PATCH_EMBED_LR_MULT)
            all_groups += groups
        return fuse_params_groups(all_groups)

    def save_checkpoint(self, path, iteration):
        state = {
            'iteration': iteration,
            'student': self.student.state_dict(),
            'teacher': self.teacher.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'dino_center': self.dino_loss.center,
            'ibot_center': self.ibot_patch_loss.center,
        }
        if self.fp16_scaler is not None:
            state['scaler'] = self.fp16_scaler.get_scale()
        torch.save(state, path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        self.student.load_state_dict(state['student'])
        self.teacher.load_state_dict(state['teacher'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.dino_loss.center = state['dino_center'].to(self.device)
        self.ibot_patch_loss.center = state['ibot_center'].to(self.device)
        return state.get('iteration', 0)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    logger = get_logger(OUTPUT_DIR)
    setup_seed(SEED)

    patches = IMG_SIZE // PATCH_SIZE
    n_tokens = patches * patches

    logger.info("=" * 70)
    logger.info("DINOv2 from-scratch training - gb dataset")
    logger.info("=" * 70)
    logger.info(f"Backbone:     ViT-Small (patch{PATCH_SIZE}, dim={EMBED_DIM}, depth={DEPTH}, heads={NUM_HEADS})")
    logger.info(f"Image size:   {IMG_SIZE} ({patches}x{patches}={n_tokens} patches)")
    logger.info(f"Register tokens: {NUM_REGISTER_TOKENS}")
    logger.info(f"Drop path:    {DROP_PATH_RATE}")
    logger.info(f"Total iters:  {TOTAL_ITERS} (~{TOTAL_ITERS * BATCH_SIZE // 2167} epochs)")
    logger.info(f"Batch size:   {BATCH_SIZE} (effective crops: {BATCH_SIZE * (2 + LOCAL_CROPS_NUMBER)})")
    logger.info(f"LR:           {BASE_LR} (warmup {WARMUP_ITERS}, min {MIN_LR})")
    logger.info(f"WD:           {WEIGHT_DECAY} -> {WEIGHT_DECAY_END}")
    logger.info(f"Clip grad:    {CLIP_GRAD}")
    logger.info(f"Layerwise decay: {LAYERWISE_DECAY}, patch_embed_lr_mult: {PATCH_EMBED_LR_MULT}")
    logger.info(f"Freeze last layer: {FREEZE_LAST_LAYER_ITERS} iters")
    logger.info(f"DINO:         loss_w={DINO_LOSS_WEIGHT}, prototypes={DINO_N_PROTOTYPES}")
    logger.info(f"iBOT:         loss_w={IBOT_LOSS_WEIGHT}, mask_prob={IBOT_MASK_PROB}, ratio={IBOT_MASK_RATIO_MIN_MAX}")
    logger.info(f"KoLeo:        loss_w={KOLEO_LOSS_WEIGHT}")
    logger.info(f"Teacher EMA:  {MOMENTUM_TEACHER} -> {FINAL_MOMENTUM_TEACHER}")
    logger.info(f"Teacher temp: warmup={WARMUP_TEACHER_TEMP}, final={TEACHER_TEMP}, warmup_iters={WARMUP_TEACHER_TEMP_ITERS}")
    logger.info(f"Multi-crop:   global={GLOBAL_CROPS_SIZE}, local={LOCAL_CROPS_SIZE}, n_local={LOCAL_CROPS_NUMBER}")
    logger.info(f"Global scale: {GLOBAL_CROPS_SCALE}")
    logger.info(f"Local scale:  {LOCAL_CROPS_SCALE}")
    logger.info(f"FP16:         {USE_FP16}")
    logger.info(f"Seed:         {SEED}")
    logger.info(f"Split json:   {SPLIT_JSON}")
    logger.info(f"Output dir:   {OUTPUT_DIR}")
    logger.info("=" * 70)

    data_transform = DataAugmentationDINO(
        global_crops_scale=GLOBAL_CROPS_SCALE,
        local_crops_scale=LOCAL_CROPS_SCALE,
        local_crops_number=LOCAL_CROPS_NUMBER,
        global_crops_size=GLOBAL_CROPS_SIZE,
        local_crops_size=LOCAL_CROPS_SIZE,
    )
    dataset = GBImageDataset(DATA_ROOT, SPLIT_JSON, transform=data_transform)
    logger.info(f"Dataset: {len(dataset)} images (strict split: main_pretrain only)")

    mask_generator = MaskingGenerator(
        input_size=(patches, patches),
        max_num_patches=0.5 * patches * patches,
    )
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=IBOT_MASK_RATIO_MIN_MAX,
        mask_probability=IBOT_MASK_PROB,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=torch.float32,
    )

    from torch.utils.data import RandomSampler
    class InfiniteSampler(RandomSampler):
        def __iter__(self):
            while True:
                yield from super().__iter__()

    data_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=InfiniteSampler(dataset),
        num_workers=NUM_WORKERS, drop_last=True,
        collate_fn=collate_fn, pin_memory=True, persistent_workers=False,
    )

    trainer = DINOv2Trainer(logger)

    params_groups = trainer.get_params_groups()
    for g in params_groups:
        g['foreach'] = True
    trainer.optimizer = torch.optim.AdamW(
        params_groups, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)

    lr_schedule = CosineScheduler(BASE_LR, MIN_LR, TOTAL_ITERS, WARMUP_ITERS, 0)
    wd_schedule = CosineScheduler(WEIGHT_DECAY, WEIGHT_DECAY_END, TOTAL_ITERS)
    momentum_schedule = CosineScheduler(MOMENTUM_TEACHER, FINAL_MOMENTUM_TEACHER, TOTAL_ITERS)
    teacher_temp_schedule = CosineScheduler(
        TEACHER_TEMP, TEACHER_TEMP, WARMUP_TEACHER_TEMP_ITERS,
        WARMUP_TEACHER_TEMP_ITERS, WARMUP_TEACHER_TEMP)
    last_layer_lr_schedule = CosineScheduler(BASE_LR, MIN_LR, TOTAL_ITERS, WARMUP_ITERS, 0)
    last_layer_lr_schedule.schedule[:FREEZE_LAST_LAYER_ITERS] = 0

    n_params = sum(p.numel() for p in trainer.student.parameters())
    logger.info(f"Student params: {n_params / 1e6:.2f} M")
    logger.info(f"Teacher params: {sum(p.numel() for p in trainer.teacher.parameters()) / 1e6:.2f} M")
    logger.info("=" * 70)

    start_iter = 0
    all_ckpts = glob.glob(os.path.join(CKPT_DIR, 'iter_*.pth'))
    if all_ckpts:
        all_ckpts.sort(key=lambda p: int(p.split('iter_')[-1].split('.pth')[0]))
        latest = all_ckpts[-1]
        start_iter = trainer.load_checkpoint(latest)
        logger.info(f"[Resume] resume from {latest}, start_iter={start_iter}")
    else:
        logger.info("[Resume] no checkpoint, training from scratch")

    data_iter = iter(data_loader)
    loss_history = []
    import time
    t0 = time.time()

    logger.info("start training...")
    for it in range(start_iter, TOTAL_ITERS):
        lr = lr_schedule[it]
        wd = wd_schedule[it]
        mom = momentum_schedule[it]
        teacher_temp = teacher_temp_schedule[it]
        last_layer_lr = last_layer_lr_schedule[it]

        for pg in trainer.optimizer.param_groups:
            is_last = pg.get('is_last_layer', False)
            lr_mult = pg.get('lr_multiplier', 1.0)
            wd_mult = pg.get('wd_multiplier', 1.0)
            pg['weight_decay'] = wd * wd_mult
            pg['lr'] = (last_layer_lr if is_last else lr) * lr_mult

        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            data = next(data_iter)

        trainer.optimizer.zero_grad(set_to_none=True)
        loss_dict = trainer.forward_backward(data, teacher_temp=teacher_temp)

        if CLIP_GRAD > 0:
            trainer.clip_grad(CLIP_GRAD)

        trainer.step()

        trainer.update_teacher(mom)

        loss_val = sum(v.item() for v in loss_dict.values())
        loss_history.append(loss_val)

        if (it + 1) % LOG_EVERY == 0:
            avg_loss = np.mean(loss_history[-LOG_EVERY:])
            lr_val = trainer.optimizer.param_groups[0]['lr']
            elapsed = time.time() - t0
            iters_per_min = (it + 1 - start_iter) / max(elapsed, 1) * 60
            eta_min = (TOTAL_ITERS - it - 1) / max(iters_per_min / 60, 1e-6)
            eta_h = int(eta_min // 60)
            eta_m = int(eta_min % 60)
            logger.info(
                f"iter [{it+1}/{TOTAL_ITERS}] loss: {avg_loss:.4f} "
                f"(dino_g: {loss_dict.get('dino_global', torch.tensor(0)).item():.4f}, "
                f"dino_l: {loss_dict.get('dino_local', torch.tensor(0)).item():.4f}, "
                f"ibot: {loss_dict.get('ibot', torch.tensor(0)).item():.4f}, "
                f"koleo: {loss_dict.get('koleo', torch.tensor(0)).item():.4f}) "
                f"lr: {lr_val:.6f} wd: {wd:.4f} mom: {mom:.5f} t_temp: {teacher_temp:.4f} "
                f"[{iters_per_min:.1f} it/min, ETA {eta_h}h{eta_m}m]"
            )

        if (it + 1) % SAVE_EVERY == 0 or (it + 1) == TOTAL_ITERS:
            ckpt_path = os.path.join(CKPT_DIR, f'iter_{it+1}.pth')
            trainer.save_checkpoint(ckpt_path, it + 1)
            all_ckpts = glob.glob(os.path.join(CKPT_DIR, 'iter_*.pth'))
            all_ckpts.sort(key=lambda p: int(p.split('iter_')[-1].split('.pth')[0]))
            if len(all_ckpts) > 3:
                for old in all_ckpts[:-3]:
                    try:
                        os.remove(old)
                    except OSError:
                        pass
            logger.info(f"[Checkpoint] saved {ckpt_path}")

    logger.info("=" * 70)
    logger.info("training finished!")
    logger.info(f"final checkpoint: {os.path.join(CKPT_DIR, f'iter_{TOTAL_ITERS}.pth')}")
    logger.info("=" * 70)

    config = {
        'model': 'dinov2_vit_small',
        'version': 'v4',
        'patch_size': PATCH_SIZE,
        'img_size': IMG_SIZE,
        'embed_dim': EMBED_DIM,
        'depth': DEPTH,
        'num_heads': NUM_HEADS,
        'num_register_tokens': NUM_REGISTER_TOKENS,
        'drop_path_rate': DROP_PATH_RATE,
        'total_iters': TOTAL_ITERS,
        'batch_size': BATCH_SIZE,
        'effective_crops_per_step': BATCH_SIZE * (2 + LOCAL_CROPS_NUMBER),
        'lr': BASE_LR,
        'min_lr': MIN_LR,
        'warmup_iters': WARMUP_ITERS,
        'weight_decay': WEIGHT_DECAY,
        'weight_decay_end': WEIGHT_DECAY_END,
        'clip_grad': CLIP_GRAD,
        'layerwise_decay': LAYERWISE_DECAY,
        'patch_embed_lr_mult': PATCH_EMBED_LR_MULT,
        'freeze_last_layer_iters': FREEZE_LAST_LAYER_ITERS,
        'dino_loss_weight': DINO_LOSS_WEIGHT,
        'dino_n_prototypes': DINO_N_PROTOTYPES,
        'dino_bottleneck_dim': DINO_BOTTLENECK_DIM,
        'dino_hidden_dim': DINO_HIDDEN_DIM,
        'ibot_loss_weight': IBOT_LOSS_WEIGHT,
        'ibot_mask_prob': IBOT_MASK_PROB,
        'ibot_mask_ratio': IBOT_MASK_RATIO_MIN_MAX,
        'koleo_loss_weight': KOLEO_LOSS_WEIGHT,
        'momentum_teacher': MOMENTUM_TEACHER,
        'final_momentum_teacher': FINAL_MOMENTUM_TEACHER,
        'teacher_temp': TEACHER_TEMP,
        'warmup_teacher_temp': WARMUP_TEACHER_TEMP,
        'warmup_teacher_temp_iters': WARMUP_TEACHER_TEMP_ITERS,
        'global_crops_size': GLOBAL_CROPS_SIZE,
        'local_crops_size': LOCAL_CROPS_SIZE,
        'local_crops_number': LOCAL_CROPS_NUMBER,
        'global_crops_scale': GLOBAL_CROPS_SCALE,
        'local_crops_scale': LOCAL_CROPS_SCALE,
        'fp16': USE_FP16,
        'seed': SEED,
        'split_json': SPLIT_JSON,
        'dataset_size': len(dataset),
        'data_isolation': 'strict (main_pretrain_files only, eval_heldout + bad excluded)',
    }
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"config saved: {os.path.join(OUTPUT_DIR, 'config.json')}")


if __name__ == '__main__':
    main()