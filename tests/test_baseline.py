"""The control row, and the two properties that make it one."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from classify_justify.evaluate import relevance_mass
from classify_justify.explain import available, build


def test_it_is_registered_like_any_other_method():
    assert "random" in available()


def test_it_never_runs_the_model(defect_sample):
    """A control has to be independent of the thing it is a control for."""

    class _Exploding(nn.Module):
        def forward(self, _inputs: torch.Tensor) -> torch.Tensor:
            raise AssertionError("the random baseline must not call the model")

    image, _ = defect_sample
    relevance = build("random", _Exploding()).attribute(image, 1)
    assert relevance.shape == (1, 1, *image.shape[-2:])


def test_its_relevance_mass_is_the_mask_s_share_of_the_frame(trained_model, defect_sample):
    """The floor the localisation column needs.

    Uniform relevance puts a share of its mass inside the mask equal to the mask's
    share of the pixels, so `relevance_mass` for this method measures the geometry of
    the dataset and nothing else. A real method that scores near this number has found
    nothing, whatever its heatmap looks like.
    """
    image, mask = defect_sample
    explainer = build("random", trained_model)
    masses = [relevance_mass(explainer.attribute(image, 1), mask) for _ in range(20)]

    assert sum(masses) / len(masses) == pytest.approx(float(mask.float().mean()), abs=0.01)
