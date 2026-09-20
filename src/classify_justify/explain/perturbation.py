"""Perturbation-based attribution: change the input, watch the logit move.

These methods never differentiate the model, so they are immune to gradient
saturation and to the architecture tricks that confuse backward hooks. They pay for
that with forward passes — RISE and Occlusion are one to two orders of magnitude
slower than Grad-CAM, which is itself a finding worth reporting rather than hiding.

Occlusion, LIME and KernelSHAP come from Captum. RISE does not exist there, so it is
implemented here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr import KernelShap, Lime, Occlusion

from classify_justify.explain.base import Explainer, register


@register
class OcclusionExplainer(Explainer):
    name = "occlusion"

    #: side of the sliding patch, and its stride, in pixels
    window: int = 32
    stride: int = 16

    def _attribute(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channels = inputs.shape[1]
        attribution = Occlusion(self.model).attribute(
            inputs,
            target=target,
            sliding_window_shapes=(channels, self.window, self.window),
            strides=(channels, self.stride, self.stride),
            baselines=0.0,
            show_progress=False,
        )
        return attribution.sum(dim=1, keepdim=True)


@register
class RISEExplainer(Explainer):
    """Randomised Input Sampling for Explanation (Petsiuk et al., 2018).

    Score many random masks, then average them weighted by the class probability each
    one produced. The masks are generated small and upsampled bilinearly with a random
    sub-pixel shift, which is what gives RISE smooth blobs instead of a visible grid.
    """

    name = "rise"

    #: number of random masks, the resolution they are drawn at, and how many survive
    n_masks: int = 2000
    cell_size: int = 8
    keep_probability: float = 0.5
    batch_size: int = 64

    @torch.no_grad()
    def _attribute(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if inputs.shape[0] != 1:
            raise ValueError("rise masks one image at a time; use batch size 1")

        height, width = inputs.shape[-2:]
        cell_h = -(-height // self.cell_size)  # ceil, so the upsample can be cropped
        cell_w = -(-width // self.cell_size)
        device = inputs.device
        cls = int(target[0])

        total = torch.zeros(1, 1, height, width, device=device)
        weight_sum = 0.0

        for start in range(0, self.n_masks, self.batch_size):
            count = min(self.batch_size, self.n_masks - start)
            grid = (
                torch.rand(count, 1, self.cell_size, self.cell_size, device=device)
                < self.keep_probability
            ).float()
            upsampled = F.interpolate(
                grid,
                size=((self.cell_size + 1) * cell_h, (self.cell_size + 1) * cell_w),
                mode="bilinear",
                align_corners=False,
            )
            # Random crop shifts each mask by a sub-cell offset, so no pixel sits at a
            # fixed position in the grid across the whole sample.
            offset_y = torch.randint(0, cell_h, (1,)).item()
            offset_x = torch.randint(0, cell_w, (1,)).item()
            masks = upsampled[:, :, offset_y : offset_y + height, offset_x : offset_x + width]

            scores = self.model(inputs * masks).softmax(dim=1)[:, cls]
            total += (masks * scores.view(-1, 1, 1, 1)).sum(dim=0, keepdim=True)
            weight_sum += float(scores.sum())

        # Normalising by the summed weights (not the mask count) keeps the result
        # comparable across images of very different confidence.
        return total / max(weight_sum, 1e-12)


class _SurrogateExplainer(Explainer):
    """LIME and KernelSHAP: fit an interpretable model on superpixel on/off samples."""

    #: side of the square superpixels the image is divided into
    patch: int = 16
    n_samples: int = 512

    def _feature_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        """Label each square patch with an index, so pixels toggle in blocks.

        Without this both methods treat every pixel as an independent feature, which
        needs orders of magnitude more samples to fit anything stable.
        """
        _, _, height, width = inputs.shape
        rows = torch.arange(height, device=inputs.device) // self.patch
        cols = torch.arange(width, device=inputs.device) // self.patch
        mask = rows.view(-1, 1) * (width // self.patch + 1) + cols.view(1, -1)
        return mask.expand(1, 1, height, width).long()

    def _attributor(self) -> Lime | KernelShap:
        raise NotImplementedError

    def _attribute(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        attribution = self._attributor().attribute(
            inputs,
            target=target,
            feature_mask=self._feature_mask(inputs),
            n_samples=self.n_samples,
            show_progress=False,
        )
        return attribution.sum(dim=1, keepdim=True)


@register
class LimeExplainer(_SurrogateExplainer):
    name = "lime"

    def _attributor(self) -> Lime:
        # Captum defaults to Lasso at alpha=0.01, which on a normalised single-channel
        # image shrinks every coefficient to exactly zero — a blank heatmap that looks
        # like a working method. Plain least squares on the patch indicators is the
        # honest surrogate here; sparsity is already imposed by the patch size.
        return Lime(self.model, interpretable_model=SkLearnLinearRegression())


@register
class KernelShapExplainer(_SurrogateExplainer):
    name = "kernel-shap"

    def _attributor(self) -> KernelShap:
        return KernelShap(self.model)
