"""Splitting, and the two rules that keep the numbers honest.

The notebook this project replaces broke both, which is why its reported metrics
could not be believed:

1. **Augment after splitting, never before.** It built augmented copies of every
   image, concatenated them onto the originals, and split the result at random — so an
   image could land in train while its own flipped copy landed in test. Augmentation
   here is a transform applied per sample at load time, so a copy can never outlive
   the split that separated it.

2. **Fit normalisation on train alone.** It computed the mean and standard deviation
   over train *and* validation together, leaking a summary of the validation set into
   every training batch.

Splits are stratified because KolektorSDD2 is heavily imbalanced — 246 defective
against 2085 clean in the training set — and a uniform random split of that leaves
the validation defect count small enough to swing several points between seeds.
"""

from __future__ import annotations

import torch

from classify_justify.data.kolektor import KolektorSDD2


def stratified_split(
    labels: list[int], validation_fraction: float = 0.2, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Split indices in two, preserving the class balance in both halves.

    Returns `(train_indices, validation_indices)` into the original ordering.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")
    if not labels:
        raise ValueError("cannot split an empty dataset")

    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []

    for class_label in sorted(set(labels)):
        members = [i for i, value in enumerate(labels) if value == class_label]
        shuffled = torch.randperm(len(members), generator=generator).tolist()
        held_out = round(len(members) * validation_fraction)
        if len(members) > 1:
            # Never let a class vanish from either side, however small it is.
            held_out = max(1, min(held_out, len(members) - 1))
        validation_indices += [members[i] for i in shuffled[:held_out]]
        train_indices += [members[i] for i in shuffled[held_out:]]

    return sorted(train_indices), sorted(validation_indices)


def channel_statistics(dataset: KolektorSDD2, limit: int | None = 512) -> tuple[float, float]:
    """Mean and standard deviation over the **training** images only.

    `limit` caps how many images are read; the estimate is stable long before the full
    set and this runs on every training job.
    """
    count = len(dataset) if limit is None else min(limit, len(dataset))
    if count == 0:
        raise ValueError("cannot compute statistics over an empty dataset")

    total = 0.0
    total_squared = 0.0
    pixels = 0
    for index in range(count):
        image = dataset[index][0]
        total += float(image.sum())
        total_squared += float((image**2).sum())
        pixels += image.numel()

    mean = total / pixels
    variance = max(total_squared / pixels - mean**2, 0.0)
    return mean, variance**0.5
