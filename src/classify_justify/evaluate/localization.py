"""Does the heatmap land on the defect?

Faithfulness asks whether a map reflects the model; localisation asks whether the
model is looking at the right thing. They disagree more often than you would like —
a classifier that keys on an imaging artefact can produce perfectly faithful maps that
point nowhere near the defect, and only ground truth reveals it.

Both metrics take a binary mask of the defect region and ignore negative relevance:
the question is where the *evidence for the class* fell.
"""

from __future__ import annotations

import torch


def _check(relevance: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    relevance = relevance.squeeze()
    mask = mask.squeeze().bool()
    if relevance.shape != mask.shape:
        raise ValueError(
            f"relevance {tuple(relevance.shape)} does not match mask {tuple(mask.shape)}"
        )
    if not mask.any():
        raise ValueError("mask is empty — there is no region to score against")
    return relevance, mask


def pointing_game(relevance: torch.Tensor, mask: torch.Tensor) -> bool:
    """Does the single most-relevant pixel fall inside the defect?

    Coarse by design, and robust to the fact that different methods spread relevance
    over wildly different areas. Averaged over a dataset it becomes a hit rate.
    """
    relevance, mask = _check(relevance, mask)
    peak = int(relevance.argmax())
    return bool(mask.flatten()[peak])


def relevance_mass(relevance: torch.Tensor, mask: torch.Tensor) -> float:
    """Share of positive relevance falling inside the defect, in [0, 1].

    Unlike the pointing game this punishes a map that hits the defect but also lights
    up half the image.
    """
    relevance, mask = _check(relevance, mask)
    positive = relevance.clamp(min=0)
    total = positive.sum()
    if total == 0:
        return 0.0
    return float(positive[mask].sum() / total)


def relevance_rank(relevance: torch.Tensor, mask: torch.Tensor) -> float:
    """Precision of the top-|mask| pixels against the mask, in [0, 1].

    Size-normalised: a large defect no longer scores well simply for being large,
    which is the main bias in `relevance_mass`.
    """
    relevance, mask = _check(relevance, mask)
    size = int(mask.sum())
    top = relevance.flatten().argsort(descending=True)[:size]
    return float(mask.flatten()[top].float().mean())
