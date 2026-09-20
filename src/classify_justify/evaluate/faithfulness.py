"""Does the heatmap point at pixels the model actually used?

A heatmap can look convincing and still be unrelated to the decision. Deletion and
insertion (Petsiuk et al., 2018) test it directly: rank the pixels by relevance, then
either remove them in order and watch the class probability fall, or add them to an
empty image in order and watch it rise.

- **Deletion AUC** — area under the probability curve as pixels are removed.
  *Lower is better*: a faithful map destroys the prediction immediately.
- **Insertion AUC** — area under the curve as pixels are restored.
  *Higher is better*: a faithful map rebuilds the prediction from few pixels.

Report both. Deletion alone rewards a map that simply finds adversarially fragile
pixels, and insertion alone rewards one that highlights everything.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FaithfulnessScore:
    deletion_auc: float
    insertion_auc: float
    deletion_curve: list[float]
    insertion_curve: list[float]

    def __str__(self) -> str:
        return (
            f"deletion {self.deletion_auc:.3f} (lower better), "
            f"insertion {self.insertion_auc:.3f} (higher better)"
        )


def _ranked_pixels(relevance: torch.Tensor) -> torch.Tensor:
    """Pixel indices ordered most- to least-relevant."""
    return relevance.flatten().argsort(descending=True)


@torch.no_grad()
def _probability_curve(
    model: nn.Module,
    image: torch.Tensor,
    target: int,
    order: torch.Tensor,
    start: torch.Tensor,
    revealed: torch.Tensor,
    steps: int,
    batch_size: int,
) -> list[float]:
    """Walk `order`, moving pixels from `start` towards `revealed`, recording p(target).

    Deletion passes the image as `start` and a baseline as `revealed`; insertion swaps
    them. Sharing one walk keeps the two metrics exactly symmetric.
    """
    pixels = order.numel()
    per_step = max(1, pixels // steps)
    cuts = list(range(0, pixels + 1, per_step))

    frames = []
    for cut in cuts:
        frame = start.clone()
        if cut:
            flat = frame.flatten(1)
            flat[:, order[:cut]] = revealed.flatten(1)[:, order[:cut]]
        frames.append(frame)

    probabilities: list[float] = []
    for begin in range(0, len(frames), batch_size):
        block = torch.cat(frames[begin : begin + batch_size])
        probabilities.extend(model(block).softmax(dim=1)[:, target].tolist())
    return probabilities


def _auc(curve: list[float]) -> float:
    """Trapezoidal area under a curve sampled at equal intervals, normalised to [0, 1]."""
    if len(curve) < 2:
        return float(curve[0]) if curve else 0.0
    interior = sum(curve[1:-1])
    return float((curve[0] / 2 + interior + curve[-1] / 2) / (len(curve) - 1))


def faithfulness(
    model: nn.Module,
    image: torch.Tensor,
    relevance: torch.Tensor,
    target: int,
    steps: int = 64,
    baseline: torch.Tensor | None = None,
    batch_size: int = 32,
) -> FaithfulnessScore:
    """Score one heatmap on one image.

    `baseline` is what a removed pixel becomes; it defaults to zeros, which for a
    normalised X-ray is the absence of signal rather than a grey smudge.
    """
    if image.shape[0] != 1:
        raise ValueError("faithfulness scores one image at a time; use batch size 1")
    if relevance.shape[-2:] != image.shape[-2:]:
        raise ValueError(
            f"relevance {tuple(relevance.shape[-2:])} does not match image "
            f"{tuple(image.shape[-2:])}"
        )

    was_training = model.training
    model.eval()
    try:
        empty = torch.zeros_like(image) if baseline is None else baseline
        order = _ranked_pixels(relevance)

        deletion = _probability_curve(
            model, image, target, order, start=image, revealed=empty,
            steps=steps, batch_size=batch_size,
        )
        insertion = _probability_curve(
            model, image, target, order, start=empty, revealed=image,
            steps=steps, batch_size=batch_size,
        )
    finally:
        model.train(was_training)

    return FaithfulnessScore(
        deletion_auc=_auc(deletion),
        insertion_auc=_auc(insertion),
        deletion_curve=deletion,
        insertion_curve=insertion,
    )
