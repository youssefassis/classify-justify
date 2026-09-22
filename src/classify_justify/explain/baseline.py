"""A control: an attribution with no information in it, for the table to be read against.

A deletion AUC of 0.26 is not good or bad on its own. It is good or bad relative to
what a map that knows nothing scores on the same model, the same images and the same
metric — and that number is not obvious in advance, because each metric has a floor
that comes from the geometry of the problem rather than from any method.

`guided-backprop` is already in the table as a control of a subtler kind: a real
method that turns out to depend only weakly on the weights. This is the blunt version,
and the floor every other row should clear.
"""

from __future__ import annotations

import torch

from classify_justify.explain.base import Explainer, register


@register
class RandomExplainer(Explainer):
    """Uniform noise, drawn without consulting the model or the image.

    It also marks the limit of the sanity test. At a fixed seed it draws the same noise
    before and after the weights are destroyed, so it fails that test, correctly. Drawn
    fresh on every call it would move further than anything and pass it outright:
    cascading randomisation rewards a map for moving, and moving is not the same as
    depending on the weights. Weight-dependence is necessary for a trustworthy
    explanation and nowhere near sufficient, which is why the sanity column is
    reported beside the faithfulness and localisation ones rather than instead of them.
    """

    name = "random"

    def _attribute(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.rand(inputs.shape[0], 1, *inputs.shape[-2:], device=inputs.device)
