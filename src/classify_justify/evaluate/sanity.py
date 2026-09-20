"""Is the explanation a function of the model at all?

Adebayo et al. (2018) showed that several popular methods produce near-identical maps
before and after a network's weights are destroyed — they are edge detectors wearing
an explanation's clothes. Guided Backprop is the standard example, which is why it is
registered here as a deliberate control.

The test progressively randomises layers from the output backwards and measures how
far the attribution moves. A trustworthy method decorrelates quickly. A method whose
correlation stays near 1.0 after the classifier head has been randomised is telling
you about the image, not about the decision.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from classify_justify.explain.base import Explainer


@dataclass(frozen=True)
class SanityResult:
    """Rank correlation with the original map, after each cumulative randomisation."""

    layers: list[str]
    correlations: list[float]

    @property
    def final_correlation(self) -> float:
        """Similarity once every randomised layer has been destroyed."""
        return self.correlations[-1] if self.correlations else float("nan")

    def passed(self, threshold: float = 0.5) -> bool:
        """True when the map has decorrelated — i.e. the method depends on the weights."""
        return abs(self.final_correlation) < threshold


def _tied_ranks(x: torch.Tensor) -> torch.Tensor:
    """Ranks with ties averaged, as Spearman's coefficient requires.

    Plain `argsort().argsort()` breaks ties by position, which invents an ordering the
    data does not have. That matters here more than it usually would: CAM maps are
    mostly exact zeros after the ReLU, so most pixels are tied, and a positional
    tiebreak would correlate two unrelated maps simply because both are mostly zero.
    """
    _, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True)
    ends = counts.cumsum(0)
    starts = ends - counts
    # Mean of the 0-based positions each tied group occupies.
    average = (starts + ends - 1).float() / 2.0
    return average[inverse]


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Rank correlation between two maps, in [-1, 1].

    Rank-based rather than Pearson because attribution scales differ wildly between
    methods and a monotone rescaling should not count as a difference — what matters
    is whether the same pixels are ranked important.

    A constant map scores 0: it expresses no ordering, so it agrees with nothing.
    """
    x, y = a.flatten().float(), b.flatten().float()
    if x.numel() != y.numel():
        raise ValueError(f"maps differ in size: {x.numel()} vs {y.numel()}")

    rank_x = _tied_ranks(x)
    rank_y = _tied_ranks(y)
    rank_x = rank_x - rank_x.mean()
    rank_y = rank_y - rank_y.mean()
    denominator = rank_x.norm() * rank_y.norm()
    if denominator == 0:  # at least one map is constant
        return 0.0
    return float((rank_x @ rank_y) / denominator)


def _randomisable_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Weight-carrying layers, output-first, which is the order the test destroys them."""
    layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d | nn.Linear)
    ]
    return list(reversed(layers))


def model_randomization_test(
    model: nn.Module,
    explainer_class: type[Explainer],
    image: torch.Tensor,
    target: int,
    target_layer_name: str | None = None,
    max_layers: int | None = None,
) -> SanityResult:
    """Cascading randomisation, as in Adebayo et al.

    The model is deep-copied, so the caller's trained weights are never touched. The
    explainer is rebuilt against each damaged copy — rebuilding rather than reusing
    matters, because CAM methods hold a reference to a specific layer object.
    """
    reference_model = copy.deepcopy(model).eval()
    reference = _attribute(reference_model, explainer_class, image, target, target_layer_name)

    damaged = copy.deepcopy(model).eval()
    layers = _randomisable_layers(damaged)
    if max_layers is not None:
        layers = layers[:max_layers]

    names: list[str] = []
    correlations: list[float] = []
    for name, module in layers:
        _reinitialise(module)
        attribution = _attribute(damaged, explainer_class, image, target, target_layer_name)
        names.append(name)
        correlations.append(spearman(reference, attribution))

    return SanityResult(layers=names, correlations=correlations)


def _reinitialise(module: nn.Module) -> None:
    """Re-draw a layer's parameters from its default initialisation."""
    module.reset_parameters()  # type: ignore[operator]  # Conv2d and Linear both have it


def _attribute(
    model: nn.Module,
    explainer_class: type[Explainer],
    image: torch.Tensor,
    target: int,
    target_layer_name: str | None,
) -> torch.Tensor:
    """Build the explainer against `model` and attribute one image."""
    target_layer = None
    if explainer_class.needs_target_layer:
        if target_layer_name is None:
            target_layer = model.target_layer  # type: ignore[union-attr]
        else:
            target_layer = dict(model.named_modules())[target_layer_name]
    return explainer_class(model, target_layer).attribute(image, target)
