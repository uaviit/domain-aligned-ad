"""Load the official DINOv2 backbones or the private V4 teacher checkpoint."""
import sys
from pathlib import Path

import torch


def load_official(anomalib_name, device):
    from anomalib.models.components.dinov2 import DinoV2Loader

    model = DinoV2Loader().load(anomalib_name).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def load_v4_teacher(checkpoint, dinov2_repo, device):
    sys.path.insert(0, str(Path(dinov2_repo)))
    from dinov2.models.vision_transformer import vit_small

    backbone = vit_small(
        patch_size=14, img_size=224, init_values=1e-5,
        num_register_tokens=0, drop_path_rate=0.0, drop_path_uniform=True,
        interpolate_antialias=False, interpolate_offset=0.1,
    )
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    teacher = state["teacher"]
    backbone_state = {key[len("backbone."):]: value for key, value in teacher.items()
                      if key.startswith("backbone.")}
    backbone.load_state_dict(backbone_state, strict=True)
    backbone = backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone, int(state.get("iteration", -1))
