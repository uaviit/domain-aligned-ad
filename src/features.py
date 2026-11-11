"""Image preprocessing and frozen-backbone patch-feature extraction."""
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.v2 import InterpolationMode


class InspectionDataset(Dataset):
    def __init__(self, paths, image_size, interpolation):
        self.paths = list(paths)
        mode = InterpolationMode.BILINEAR if interpolation == "bilinear" else InterpolationMode.BICUBIC
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=mode),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), Path(self.paths[index]).name


@torch.no_grad()
def extract_layers(backbone, paths, layers, image_size, interpolation, device,
                   fp16=False, batch_size=8, workers=2, official_api=False):
    """Return normalized float32 CPU features as {layer: [images, patches, channels]}."""
    loader = DataLoader(
        InspectionDataset(paths, image_size, interpolation),
        batch_size=batch_size, shuffle=False, num_workers=workers,
    )
    collected = {layer: [] for layer in layers}
    for images, _ in loader:
        images = images.to(device)
        if official_api:
            features = backbone.get_intermediate_layers(
                images, n=list(layers), norm=True, reshape=False, return_class_token=False,
            )
        else:
            if fp16 and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    features = backbone.get_intermediate_layers(images, n=list(layers), norm=True)
            else:
                features = backbone.get_intermediate_layers(images, n=list(layers), norm=True)
        for layer, feature in zip(layers, features):
            feature = F.normalize(feature.float(), dim=-1).cpu()
            collected[layer].append(feature)
    return {layer: torch.cat(chunks) for layer, chunks in collected.items()}
