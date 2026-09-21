"""Shared fixtures.

The model is trained once per session on a handful of synthetic images. Training
rather than randomising matters: the sanity checks compare a trained model against a
destroyed one, and a randomly initialised model has nothing to destroy.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from classify_justify.data import SyntheticDefects, collate_without_masks
from classify_justify.explain import build
from classify_justify.models import DefectClassifier


@pytest.fixture(scope="session")
def trained_model() -> DefectClassifier:
    torch.manual_seed(0)
    # 64px, not 32: four pooling stages take 32px down to a 2x2 feature map, which is
    # too degenerate for the CAM family to say anything.
    dataset = SyntheticDefects(size=256, image_size=64, seed=1)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_without_masks)
    model = DefectClassifier()
    # lr=2e-3 collapses to a single class at this dataset size; 1e-3 converges.
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(10):
        for images, labels in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimiser.step()
    model.eval()

    # Assert the fixture actually learned. A model that collapses to one class still
    # produces heatmaps of the right shape, so every contract test would pass while
    # measuring nothing — and Grad-CAM would return an empty map for the class the
    # model never predicts, looking like a bug in the CAM code. Fail loudly instead.
    with torch.no_grad():
        images, labels = collate_without_masks([dataset[i] for i in range(len(dataset))])
        predictions = model(images).argmax(dim=1)
    accuracy = float((predictions == labels).float().mean())
    assert accuracy > 0.9, f"fixture model did not train (accuracy {accuracy:.2f})"
    assert 0 < int(predictions.sum()) < len(predictions), "fixture model predicts one class"
    return model


@pytest.fixture(scope="session")
def defect_sample() -> tuple[torch.Tensor, torch.Tensor]:
    """One defective image and its ground-truth mask, batched."""
    image, label, mask = SyntheticDefects(size=128, image_size=64, seed=1)[1]
    assert label == 1
    return image.unsqueeze(0), mask


#: RISE, LIME and KernelSHAP need hundreds of forward passes. Tests turn them down
#: rather than skipping them, because their contract is what is under test.
SLOW = {"rise": {"n_masks": 64}, "lime": {"n_samples": 32}, "kernel-shap": {"n_samples": 32}}


@pytest.fixture
def make_explainer():
    """Build a registered explainer, with the slow methods reduced to test size."""

    def make(name: str, model: DefectClassifier, seed: int | None = None):
        explainer = build(name, model, model.target_layer, seed=seed)
        for attribute, value in SLOW.get(name, {}).items():
            setattr(explainer, attribute, value)
        return explainer

    return make
