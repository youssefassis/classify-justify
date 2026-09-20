"""Gradient-based attribution, delegated to Captum.

These are thin adapters, on purpose. Captum's implementations are well tested and
widely reviewed; re-deriving them here would add risk without adding insight. What the
adapters contribute is the shared contract — (B, 1, H, W) at input resolution, one
`name` in the registry — so these methods and the hand-written CAMs can be scored side
by side in `classify_justify.evaluate`.

Captum returns attribution per input channel; summing over channels (rather than
taking a mean or a max) keeps the total relevance intact, which the deletion and
insertion metrics assume when they rank pixels.
"""

from __future__ import annotations

import torch
from captum.attr import (
    DeepLift,
    GradientShap,
    GuidedBackprop,
    InputXGradient,
    IntegratedGradients,
    NoiseTunnel,
    Saliency,
)

from classify_justify.explain.base import Explainer, register


class _CaptumExplainer(Explainer):
    """Run a Captum attribution and reduce it to one channel."""

    #: number of steps/samples for the methods that approximate an integral
    n_steps: int = 32

    def _run(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _attribute(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            attribution = self._run(inputs.clone().requires_grad_(True), target)
        return attribution.sum(dim=1, keepdim=True).detach()


@register
class SaliencyExplainer(_CaptumExplainer):
    name = "saliency"

    def _run(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # abs=False keeps the sign; `to_heatmap` decides how to display it.
        return Saliency(self.model).attribute(inputs, target=target, abs=False)


@register
class InputXGradientExplainer(_CaptumExplainer):
    name = "input-x-gradient"

    def _run(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return InputXGradient(self.model).attribute(inputs, target=target)


@register
class IntegratedGradientsExplainer(_CaptumExplainer):
    name = "integrated-gradients"

    def _run(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # A zero baseline is a black X-ray: the absence of a reading, which is the
        # right "nothing happened here" reference for this modality.
        return IntegratedGradients(self.model).attribute(
            inputs, baselines=torch.zeros_like(inputs), target=target, n_steps=self.n_steps
        )


@register
class SmoothGradExplainer(_CaptumExplainer):
    name = "smoothgrad"

    def _run(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Computed off the detached input: `inputs` carries a grad here, and casting a
        # grad-tracking tensor to float warns and pins the value into the graph.
        reference = inputs.detach()
        noise = 0.1 * float(reference.max() - reference.min())
        return NoiseTunnel(Saliency(self.model)).attribute(
            inputs,
            target=target,
            abs=False,
            nt_type="smoothgrad",
            nt_samples=self.n_steps,
            stdevs=noise,
        )


@register
class GuidedBackpropExplainer(_CaptumExplainer):
    name = "guided-backprop"

    def _run(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Included deliberately as a control: it draws clean edges but is known to be
        # largely independent of the weights. The sanity checks are expected to catch
        # it, and a run where they do not means the checks are broken.
        return GuidedBackprop(self.model).attribute(inputs, target=target)


@register
class DeepLiftExplainer(_CaptumExplainer):
    name = "deeplift"

    def _run(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return DeepLift(self.model).attribute(
            inputs, baselines=torch.zeros_like(inputs), target=target
        )


@register
class GradientShapExplainer(_CaptumExplainer):
    name = "gradient-shap"

    def _run(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # GradientSHAP samples between the input and a distribution of baselines; a
        # zero image plus its noisy neighbours is a reasonable stand-in for one.
        baselines = torch.cat([torch.zeros_like(inputs), torch.randn_like(inputs) * 0.1])
        return GradientShap(self.model).attribute(
            inputs, baselines=baselines, target=target, n_samples=self.n_steps, stdevs=0.09
        )
