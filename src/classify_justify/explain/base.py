"""The one interface every attribution method implements, and the registry that lists them.

A method takes a batch and a target class and returns one relevance value per input
pixel, at input resolution. Keeping that contract identical across gradient, CAM and
perturbation methods is what lets `classify_justify.evaluate` score all of them in the
same loop without special cases — and what makes the comparison in the README fair.

Relevance is returned **signed**. Methods that are non-negative by construction (the
CAM family, occlusion of a positive logit) simply never emit negatives. Callers that
want a picture use `to_heatmap`, which makes the sign convention explicit at the point
of display rather than burying it in each method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator

import torch
import torch.nn.functional as F
from torch import nn

_REGISTRY: dict[str, type[Explainer]] = {}


class Explainer(ABC):
    """Base class for attribution methods.

    Subclasses set `name`, and set `needs_target_layer = True` if they attribute to a
    convolutional feature map rather than to the input.
    """

    name: str = ""
    needs_target_layer: bool = False

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        if self.needs_target_layer and target_layer is None:
            raise ValueError(
                f"{type(self).__name__} attributes to a feature map and needs a "
                "target_layer; pass model.target_layer"
            )
        self.model = model
        self.target_layer = target_layer

    @abstractmethod
    def _attribute(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return relevance shaped (B, 1, H, W) at input resolution."""

    def attribute(self, inputs: torch.Tensor, target: torch.Tensor | int) -> torch.Tensor:
        """Attribute `inputs` to `target`, validating shapes on the way in and out."""
        if inputs.ndim != 4:
            raise ValueError(f"expected inputs shaped (B, C, H, W), got {tuple(inputs.shape)}")
        target = _as_target(target, batch_size=inputs.shape[0], device=inputs.device)

        was_training = self.model.training
        self.model.eval()
        try:
            relevance = self._attribute(inputs, target)
        finally:
            self.model.train(was_training)

        if relevance.shape[-2:] != inputs.shape[-2:]:
            relevance = F.interpolate(
                relevance, size=inputs.shape[-2:], mode="bilinear", align_corners=False
            )
        if relevance.shape[:2] != (inputs.shape[0], 1):
            raise RuntimeError(
                f"{self.name} produced {tuple(relevance.shape)}; "
                f"expected ({inputs.shape[0]}, 1, H, W)"
            )
        return relevance


def _as_target(target: torch.Tensor | int, batch_size: int, device: torch.device) -> torch.Tensor:
    """Accept a scalar class index or a per-sample tensor; always return (B,) long."""
    if isinstance(target, int):
        return torch.full((batch_size,), target, dtype=torch.long, device=device)
    target = target.to(device=device, dtype=torch.long).reshape(-1)
    if target.numel() == 1 and batch_size > 1:
        return target.expand(batch_size)
    if target.numel() != batch_size:
        raise ValueError(f"target has {target.numel()} entries for a batch of {batch_size}")
    return target


def to_heatmap(relevance: torch.Tensor, signed: bool = False) -> torch.Tensor:
    """Normalise relevance to [0, 1] per sample, for display.

    `signed=False` takes the absolute value first, which is what you want for methods
    whose sign is not meaningful (plain gradients). `signed=True` keeps only positive
    evidence, which is what you want when the question is "what supported this class".
    Normalisation is per sample: heatmaps are compared within an image, never across.
    """
    relevance = relevance.abs() if not signed else relevance.clamp(min=0)
    flat = relevance.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    # A constant map (every method produces one occasionally) normalises to zero
    # rather than dividing by zero and returning NaN.
    scaled = (relevance - lo) / (hi - lo).clamp(min=1e-12)
    return torch.where(hi > lo, scaled, torch.zeros_like(relevance))


def register(cls: type[Explainer]) -> type[Explainer]:
    """Class decorator: add an explainer to the registry under its `name`."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a non-empty `name`")
    if cls.name in _REGISTRY:
        raise ValueError(f"{cls.name!r} is already registered to {_REGISTRY[cls.name].__name__}")
    _REGISTRY[cls.name] = cls
    return cls


def available() -> list[str]:
    """Every registered method name, sorted."""
    return sorted(_REGISTRY)


def get(name: str) -> type[Explainer]:
    """Look up an explainer class by name, with a useful error when it is missing."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown method {name!r}; available: {', '.join(available())}") from None


def build(name: str, model: nn.Module, target_layer: nn.Module | None = None) -> Explainer:
    """Instantiate a registered explainer against a model."""
    return get(name)(model, target_layer)


def build_all(
    model: nn.Module,
    target_layer: nn.Module | None = None,
    names: list[str] | None = None,
) -> Iterator[Explainer]:
    """Instantiate several explainers, defaulting to every registered one."""
    for name in names if names is not None else available():
        yield build(name, model, target_layer)


def hooked_activations(
    layer: nn.Module,
) -> tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor], Callable[[], None]]:
    """Capture a layer's forward output and its gradient.

    Returned as (activations, gradients, remove). The gradient hook is registered on
    the captured tensor rather than on the module, because module-level backward hooks
    are unreliable for modules that are reused or run in-place.
    """
    store: dict[str, torch.Tensor] = {}

    def forward_hook(_module: nn.Module, _inp: object, out: torch.Tensor) -> None:
        store["activations"] = out
        if out.requires_grad:
            out.register_hook(lambda grad: store.__setitem__("gradients", grad))

    handle = layer.register_forward_hook(forward_hook)

    def activations() -> torch.Tensor:
        if "activations" not in store:
            raise RuntimeError("no activations captured — run a forward pass first")
        return store["activations"]

    def gradients() -> torch.Tensor:
        if "gradients" not in store:
            raise RuntimeError("no gradients captured — run a backward pass first")
        return store["gradients"]

    return activations, gradients, handle.remove
