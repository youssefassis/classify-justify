"""A generated stand-in for X-ray defect data.

Its purpose is not realism. It is that every test, every example in the README and
every CI run can execute the whole pipeline without downloading anything — and,
because the defect's position is known exactly, the localisation metrics have ground
truth to score against.

A "defect" is a small Gaussian blob on a textured background. That is enough structure
for a small CNN to reach high accuracy in seconds, which is what the sanity checks
need: a model that has genuinely learned something.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils.data import Dataset


class SyntheticDefects(Dataset):
    """Textured grayscale images, half of which carry a small bright blob.

    Each item is `(image, label, mask)` where `mask` marks the blob — empty for the
    defect-free class.
    """

    def __init__(
        self,
        size: int = 256,
        image_size: int = 64,
        blob_radius: int = 4,
        noise: float = 0.30,
        seed: int = 0,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        indices: list[int] | None = None,
    ) -> None:
        if size < 2:
            raise ValueError("need at least one image per class")
        self.size = size
        self.image_size = image_size
        self.blob_radius = blob_radius
        self.noise = noise
        self.seed = seed
        self.transform = transform
        # `transform` and `indices` mirror KolektorSDD2 so the two are interchangeable
        # everywhere: the trainer, the CLI and the evaluation loop take either.
        self.indices = list(range(size)) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def labels(self) -> list[int]:
        """Labels for every item, for stratified splitting and class weighting."""
        return [index % 2 for index in self.indices]

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        if not 0 <= position < len(self.indices):
            raise IndexError(position)
        index = self.indices[position]
        # Seeded per item so the dataset is a pure function of (seed, index) and a
        # split stays identical across processes and workers.
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
        side = self.image_size

        image = torch.rand(1, side, side, generator=generator) * self.noise
        # Low-frequency background, so the model cannot win by looking at one pixel.
        coords = torch.linspace(0, 3.1416 * 2, side)
        image += 0.25 * (torch.sin(coords).view(1, -1, 1) * torch.cos(coords).view(1, 1, -1))

        label = index % 2
        mask = torch.zeros(1, side, side, dtype=torch.bool)
        if label == 1:
            margin = self.blob_radius * 2
            centre_y = int(torch.randint(margin, side - margin, (1,), generator=generator))
            centre_x = int(torch.randint(margin, side - margin, (1,), generator=generator))
            grid_y = torch.arange(side).view(-1, 1)
            grid_x = torch.arange(side).view(1, -1)
            squared = (grid_y - centre_y) ** 2 + (grid_x - centre_x) ** 2
            blob = torch.exp(-squared.float() / (2 * self.blob_radius**2))
            image += blob.unsqueeze(0)
            mask = (squared <= self.blob_radius**2).unsqueeze(0)

        if self.transform is not None:
            image = self.transform(image)
        return image, label, mask


def collate_without_masks(
    batch: list[tuple[torch.Tensor, int, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop masks so the dataset can feed an ordinary training loop."""
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels
