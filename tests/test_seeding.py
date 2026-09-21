"""A published number that cannot be reproduced is not a measurement.

Five of the seventeen methods sample, and so does the layer re-initialisation in the
sanity test. These tests pin the two properties that matter: a seed makes a method
repeat, and setting it never reaches back into the caller's own random stream.
"""

from __future__ import annotations

import pytest
import torch

from classify_justify.evaluate import model_randomization_test
from classify_justify.explain import available, get
from classify_justify.seeding import seeded

#: The methods that draw random numbers. Kept explicit so the test below fails loudly
#: if one of them ever stops sampling and the seed test starts passing vacuously.
STOCHASTIC = ["gradient-shap", "kernel-shap", "lime", "rise", "smoothgrad"]


class TestSeeded:
    def test_no_seed_leaves_the_stream_alone(self):
        torch.manual_seed(0)
        expected = torch.rand(3)
        torch.manual_seed(0)
        with seeded(None):
            pass
        assert torch.equal(torch.rand(3), expected)

    def test_the_body_runs_from_the_seed(self):
        with seeded(7):
            first = torch.rand(3)
        with seeded(7):
            assert torch.equal(torch.rand(3), first)

    def test_the_caller_s_stream_survives(self):
        """A training run that explains a batch mid-epoch must not be perturbed by it."""
        torch.manual_seed(0)
        expected = torch.rand(3)

        torch.manual_seed(0)
        with seeded(99):
            torch.rand(50)
        assert torch.equal(torch.rand(3), expected)


@pytest.mark.parametrize("name", available())
def test_a_seed_makes_every_method_repeat(name, trained_model, defect_sample, make_explainer):
    image, _ = defect_sample
    first = make_explainer(name, trained_model, seed=0).attribute(image, 1)
    second = make_explainer(name, trained_model, seed=0).attribute(image, 1)
    assert torch.equal(first, second), f"{name} did not reproduce under a seed"


@pytest.mark.parametrize("name", STOCHASTIC)
def test_the_stochastic_methods_really_do_sample(
    name, trained_model, defect_sample, make_explainer
):
    """Without this, the seed test above could pass because nothing was random."""
    image, _ = defect_sample
    first = make_explainer(name, trained_model).attribute(image, 1)
    second = make_explainer(name, trained_model).attribute(image, 1)
    assert not torch.equal(first, second), f"{name} is listed as stochastic but is not"


def test_attribution_does_not_disturb_the_caller_s_stream(
    trained_model, defect_sample, make_explainer
):
    image, _ = defect_sample
    torch.manual_seed(0)
    expected = torch.rand(3)

    torch.manual_seed(0)
    make_explainer("rise", trained_model, seed=123).attribute(image, 1)
    assert torch.equal(torch.rand(3), expected)


def test_the_sanity_test_reproduces(trained_model, defect_sample):
    """Both halves are seeded: the sampling, and the draw that destroys each layer."""
    image, _ = defect_sample
    first = model_randomization_test(trained_model, get("saliency"), image, 1, seed=0)
    second = model_randomization_test(trained_model, get("saliency"), image, 1, seed=0)
    assert first.correlations == second.correlations
