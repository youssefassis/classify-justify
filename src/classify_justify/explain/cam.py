"""CAM-family methods, which Captum does not ship.

All six weight the target layer's channels and sum them; they differ only in where the
weights come from. That is the whole taxonomy, and writing them against one base makes
the differences visible instead of burying them in six near-identical files:

| method       | weights come from                                          |
|--------------|------------------------------------------------------------|
| Grad-CAM     | spatially averaged gradient                                  |
| Grad-CAM++   | gradient, reweighted so many small activations are not drowned out |
| XGrad-CAM    | gradient, weighted by each position's share of its channel    |
| LayerCAM     | per-position positive gradient (weights vary within a channel) |
| Score-CAM    | forward passes with the channel as a mask — no gradients      |
| Ablation-CAM | the logit drop when the channel is removed — no gradients     |

The last two cost one forward pass per channel but survive the gradient-saturation
failure that makes the first four unreliable on a confident model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from classify_justify.explain.base import Explainer, hooked_activations, register


class _CAM(Explainer):
    """Shared machinery: capture the target layer, weight channels, sum, ReLU."""

    needs_target_layer = True

    def _forward_backward(
        self, inputs: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (activations, gradients) of the target layer for the target logit."""
        activations_of, gradients_of, remove = hooked_activations(self.target_layer)
        try:
            with torch.enable_grad():
                # Cloning with requires_grad guarantees a graph even when every
                # parameter is frozen, which is the case during the sanity checks.
                x = inputs.clone().requires_grad_(True)
                logits = self.model(x)
                selected = logits.gather(1, target.view(-1, 1)).sum()
                self.model.zero_grad(set_to_none=True)
                selected.backward()
            return activations_of().detach(), gradients_of().detach()
        finally:
            remove()

    @staticmethod
    def _combine(activations: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Weighted channel sum, keeping only positive evidence."""
        return F.relu((weights * activations).sum(dim=1, keepdim=True))


@register
class GradCAM(_CAM):
    name = "grad-cam"

    def _attribute(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        activations, gradients = self._forward_backward(inputs, target)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        return self._combine(activations, weights)


@register
class GradCAMPlusPlus(_CAM):
    name = "grad-cam++"

    def _attribute(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        activations, gradients = self._forward_backward(inputs, target)
        grad_2 = gradients.pow(2)
        grad_3 = grad_2 * gradients
        denom = 2.0 * grad_2 + (activations * grad_3).sum(dim=(2, 3), keepdim=True)
        alpha = grad_2 / torch.where(denom != 0, denom, torch.ones_like(denom))
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        return self._combine(activations, weights)


@register
class XGradCAM(_CAM):
    name = "xgrad-cam"

    def _attribute(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        activations, gradients = self._forward_backward(inputs, target)
        channel_sum = activations.sum(dim=(2, 3), keepdim=True)
        share = activations / channel_sum.clamp(min=1e-12)
        weights = (gradients * share).sum(dim=(2, 3), keepdim=True)
        return self._combine(activations, weights)


@register
class LayerCAM(_CAM):
    name = "layer-cam"

    def _attribute(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        activations, gradients = self._forward_backward(inputs, target)
        # Weights vary per position, so a channel can matter in one region and not
        # another — which is why LayerCAM stays sharp on shallower layers.
        return F.relu((F.relu(gradients) * activations).sum(dim=1, keepdim=True))


class _GradientFreeCAM(_CAM):
    """CAMs that score channels by re-running the model, not by differentiating it."""

    #: channels scored per forward pass; caps memory on wide layers
    chunk_size: int = 32

    def _channel_scores(
        self, inputs: torch.Tensor, target: torch.Tensor, activations: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def _attribute(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        activations_of, _, remove = hooked_activations(self.target_layer)
        try:
            with torch.no_grad():
                self.model(inputs)
                activations = activations_of().detach()
            weights = self._channel_scores(inputs, target, activations)
        finally:
            remove()
        return self._combine(activations, weights)


@register
class ScoreCAM(_GradientFreeCAM):
    name = "score-cam"

    @torch.no_grad()
    def _channel_scores(
        self, inputs: torch.Tensor, target: torch.Tensor, activations: torch.Tensor
    ) -> torch.Tensor:
        batch, channels = activations.shape[:2]
        if batch != 1:
            raise ValueError("score-cam masks the input per channel; use batch size 1")

        masks = F.interpolate(
            activations, size=inputs.shape[-2:], mode="bilinear", align_corners=False
        )[0]
        flat = masks.flatten(1)
        lo = flat.min(dim=1).values.view(-1, 1, 1)
        hi = flat.max(dim=1).values.view(-1, 1, 1)
        masks = (masks - lo) / (hi - lo).clamp(min=1e-12)

        scores = []
        cls = int(target[0])
        for start in range(0, channels, self.chunk_size):
            block = masks[start : start + self.chunk_size].unsqueeze(1)
            logits = self.model(inputs * block)
            scores.append(logits.softmax(dim=1)[:, cls])
        weights = torch.cat(scores)
        # A channel that never raises the class probability should not vote.
        weights = F.relu(weights - weights.mean())
        return weights.view(1, channels, 1, 1)


@register
class AblationCAM(_GradientFreeCAM):
    name = "ablation-cam"

    @torch.no_grad()
    def _channel_scores(
        self, inputs: torch.Tensor, target: torch.Tensor, activations: torch.Tensor
    ) -> torch.Tensor:
        batch, channels = activations.shape[:2]
        if batch != 1:
            raise ValueError("ablation-cam removes one channel at a time; use batch size 1")

        cls = int(target[0])
        baseline = self.model(inputs)[0, cls]

        ablated: list[int] = []

        def zero_channels(_m: nn.Module, _i: object, out: torch.Tensor) -> torch.Tensor:
            out = out.clone()
            for row, channel in enumerate(ablated):
                out[row, channel] = 0.0
            return out

        handle = self.target_layer.register_forward_hook(zero_channels)
        try:
            drops = []
            for start in range(0, channels, self.chunk_size):
                ablated = list(range(start, min(start + self.chunk_size, channels)))
                repeated = inputs.expand(len(ablated), -1, -1, -1)
                logits = self.model(repeated)
                drops.append(baseline - logits[:, cls])
        finally:
            handle.remove()

        weights = F.relu(torch.cat(drops) / baseline.abs().clamp(min=1e-12))
        return weights.view(1, channels, 1, 1)
