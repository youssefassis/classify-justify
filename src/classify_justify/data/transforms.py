"""Transforms, split into the ones that may be random and the ones that may not.

The notebook applied `RandomRotation` and `ColorJitter` to its **test** transform, so
every evaluation saw a differently mangled image and no two runs agreed. It also
applied `ColorJitter(saturation=..., hue=...)` after `Grayscale()`, where those two
arguments do nothing at all.

Here the split is explicit: `train_transform` is random by design, `eval_transform` is
a pure function. Attribution work depends on the second one — a heatmap explains the
exact tensor the model saw, so that tensor has to be reproducible.
"""

from __future__ import annotations

from torch import nn
from torchvision.transforms import v2


def eval_transform(mean: float, std: float) -> nn.Module:
    """Deterministic: normalisation and nothing else.

    Resizing already happened at load time, where the mask could be resampled with
    NEAREST alongside the image.
    """
    return v2.Normalize(mean=[mean], std=[std])


def train_transform(mean: float, std: float, augment: bool = True) -> nn.Module:
    """Normalisation plus augmentation that leaves the defect where it is.

    Flips and small intensity shifts only. No rotation: KolektorSDD2 parts are imaged
    in a fixed orientation, and a 90-degree rotation of a 256x640 strip would teach
    the model a geometry it will never be tested on. Nothing here moves a defect
    relative to its mask, so the localisation ground truth stays valid.
    """
    if not augment:
        return eval_transform(mean, std)
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.Normalize(mean=[mean], std=[std]),
        ]
    )
