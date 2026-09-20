import pytest
import torch

from classify_justify.explain import available, build, get, to_heatmap
from classify_justify.explain.base import _as_target

# RISE, LIME and KernelSHAP need hundreds of forward passes; they are exercised with
# reduced settings rather than skipped, because their contract is what is under test.
SLOW = {"rise": {"n_masks": 64}, "lime": {"n_samples": 32}, "kernel-shap": {"n_samples": 32}}


def _explainer(name, model):
    explainer = build(name, model, model.target_layer)
    for attribute, value in SLOW.get(name, {}).items():
        setattr(explainer, attribute, value)
    return explainer


@pytest.mark.parametrize("name", available())
def test_every_method_honours_the_contract(name, trained_model, defect_sample):
    """(B, 1, H, W) at input resolution, finite, for every registered method."""
    image, _ = defect_sample
    relevance = _explainer(name, trained_model).attribute(image, 1)

    assert relevance.shape == (1, 1, *image.shape[-2:])
    assert torch.isfinite(relevance).all()
    assert not torch.equal(relevance, torch.zeros_like(relevance)), f"{name} is uniformly zero"


@pytest.mark.parametrize("name", available())
def test_methods_leave_the_model_as_they_found_it(name, trained_model, defect_sample):
    """Hooks are removed and training mode restored, so methods can run in any order."""
    image, _ = defect_sample
    before = len(list(trained_model.target_layer._forward_hooks))
    trained_model.train()
    _explainer(name, trained_model).attribute(image, 1)

    assert trained_model.training, f"{name} left the model in eval mode"
    assert len(list(trained_model.target_layer._forward_hooks)) == before
    trained_model.eval()


def test_registry_rejects_a_duplicate_name():
    from classify_justify.explain.base import Explainer, register

    with pytest.raises(ValueError, match="already registered"):

        @register
        class Duplicate(Explainer):
            name = "grad-cam"

            def _attribute(self, inputs, target):
                raise NotImplementedError


def test_registry_reports_what_it_has():
    with pytest.raises(KeyError, match="unknown method"):
        get("no-such-method")
    assert {"grad-cam", "rise", "integrated-gradients"} <= set(available())


def test_heatmap_is_bounded_and_survives_a_constant_map():
    assert to_heatmap(torch.zeros(1, 1, 4, 4)).max() == 0.0  # no NaN from dividing by zero

    heatmap = to_heatmap(torch.randn(3, 1, 8, 8))
    assert float(heatmap.min()) >= 0.0 and float(heatmap.max()) <= 1.0
    # Normalisation is per sample, so every sample reaches both ends of the range.
    assert torch.allclose(heatmap.flatten(1).max(dim=1).values, torch.ones(3))


def test_signed_heatmap_keeps_only_positive_evidence():
    relevance = torch.tensor([[[[-5.0, 1.0]]]])
    assert to_heatmap(relevance, signed=True)[0, 0, 0, 0] == 0.0
    assert to_heatmap(relevance, signed=False)[0, 0, 0, 0] == 1.0  # |-5| is the largest


def test_target_accepts_a_scalar_or_a_tensor():
    device = torch.device("cpu")
    assert _as_target(1, 3, device).tolist() == [1, 1, 1]
    assert _as_target(torch.tensor([0, 1]), 2, device).tolist() == [0, 1]
    assert _as_target(torch.tensor([1]), 3, device).tolist() == [1, 1, 1]
    with pytest.raises(ValueError, match="entries for a batch"):
        _as_target(torch.tensor([0, 1]), 3, device)


def test_cam_methods_demand_a_target_layer(trained_model):
    with pytest.raises(ValueError, match="needs a"):
        get("grad-cam")(trained_model, None)


def test_attribute_rejects_an_unbatched_image(trained_model):
    with pytest.raises(ValueError, match=r"\(B, C, H, W\)"):
        build("saliency", trained_model).attribute(torch.randn(1, 64, 64), 1)


@pytest.mark.parametrize("name", ["score-cam", "ablation-cam", "rise"])
def test_single_image_methods_say_so(name, trained_model):
    """These mask or ablate per sample; a silent wrong answer would be worse."""
    with pytest.raises(ValueError, match="batch size 1"):
        _explainer(name, trained_model).attribute(torch.randn(2, 1, 64, 64), 1)


def test_grad_cam_may_legitimately_return_an_empty_map(trained_model):
    """ReLU(sum of weighted channels) is empty when every channel argues against the class.

    This is Grad-CAM working as specified, not a failure — but it is invisible in a
    normalised heatmap, where an empty map and a flat one look identical. It is
    asserted here so the behaviour stays known: a blank result means "no positive
    evidence at this layer", and the CAM family goes quiet on small feature maps.
    """
    tiny = torch.randn(1, 1, 16, 16)  # four pooling stages leave a 1x1 feature map
    relevance = build("grad-cam", trained_model, trained_model.target_layer).attribute(tiny, 0)
    assert torch.isfinite(relevance).all()
    assert relevance.min() >= 0.0  # never negative, whatever else it is


def test_grad_cam_matches_captums_reference(trained_model, defect_sample):
    """The hand-written CAM base is checked against an implementation we did not write.

    Grad-CAM is the one method in the CAM family that Captum also ships, which makes
    it the anchor for the other five: they share `_CAM._forward_backward` and
    `_combine`, so an error in that shared machinery shows up here.
    """
    from captum.attr import LayerGradCam

    image, _ = defect_sample
    target = torch.tensor([1])
    mine = build("grad-cam", trained_model, trained_model.target_layer)._attribute(image, target)
    reference = LayerGradCam(trained_model, trained_model.target_layer).attribute(
        image, target=1, relu_attributions=True
    )
    assert torch.allclose(mine, reference, atol=1e-6)
