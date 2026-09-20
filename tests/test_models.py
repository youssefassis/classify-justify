import pytest
import torch
from torch import nn

from classify_justify.models import (
    DefectClassifier,
    ModelConfig,
    build_model,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_round_trip_needs_no_dataset(tmp_path):
    """The bug this project started with: the old model could not be rebuilt alone."""
    model = DefectClassifier(ModelConfig(widths=(8, 16)))
    path = tmp_path / "model.pt"
    save_checkpoint(model, path)

    restored = load_checkpoint(path)
    x = torch.randn(2, 1, 32, 32)
    assert torch.allclose(model.eval()(x), restored(x))
    assert restored.config == model.config


def test_load_rejects_a_foreign_checkpoint(tmp_path):
    path = tmp_path / "plain.pt"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(ValueError, match="not a classify-justify checkpoint"):
        load_checkpoint(path)


def test_head_is_pooled_so_input_size_is_free():
    """A GAP head is what makes CAM weighting defensible, and frees the resolution."""
    model = DefectClassifier().eval()
    assert model(torch.randn(1, 1, 32, 32)).shape == (1, 2)
    assert model(torch.randn(1, 1, 64, 64)).shape == (1, 2)


def test_target_layer_is_the_last_relu():
    model = DefectClassifier()
    assert isinstance(model.target_layer, nn.ReLU)
    assert model.target_layer is [m for m in model.features[-1] if isinstance(m, nn.ReLU)][-1]


def test_relus_are_not_inplace():
    """In-place ReLU silently corrupts Guided Backprop and DeepLIFT attributions."""
    model = DefectClassifier()
    assert all(not m.inplace for m in model.modules() if isinstance(m, nn.ReLU))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_classes": 1}, "num_classes"),
        ({"in_channels": 0}, "in_channels"),
        ({"widths": ()}, "widths"),
        ({"dropout": 1.0}, "dropout"),
    ],
)
def test_config_validates_at_the_boundary(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_model(**kwargs)


def test_build_model_rejects_unknown_options():
    with pytest.raises(ValueError, match="unknown model options"):
        build_model(widht=8)
