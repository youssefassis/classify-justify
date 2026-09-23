"""The command line, covering the device plumbing end to end.

There were no CLI tests before this; the commands were checked by hand. These run on
the CPU, which is what CI has. The accelerator paths differ only in where the tensors
live, and the mistakes that plumbing invites — a mask left behind on the CPU while the
relevance map moved, a device tensor handed to matplotlib — are what the assertions
here are aimed at.
"""

from __future__ import annotations

import json

import pytest
import torch
from PIL import Image

from classify_justify.cli import build_parser, main
from classify_justify.models import save_checkpoint


@pytest.fixture
def checkpoint(tmp_path, trained_model):
    path = tmp_path / "model.pt"
    save_checkpoint(
        trained_model,
        path,
        {"image_size": [64, 64], "normalization": {"mean": 0.0, "std": 1.0}},
    )
    return path


@pytest.fixture
def png(tmp_path):
    path = tmp_path / "sample.png"
    Image.fromarray((torch.rand(64, 64) * 255).byte().numpy(), mode="L").save(path)
    return path


class TestDeviceFlag:
    def test_it_defaults_to_auto(self):
        arguments = build_parser().parse_args(
            ["explain", "--checkpoint", "m.pt", "--image", "i.png"]
        )
        assert arguments.device == "auto"

    def test_evaluate_takes_it_too(self):
        arguments = build_parser().parse_args(
            ["evaluate", "--checkpoint", "m.pt", "--device", "cpu"]
        )
        assert arguments.device == "cpu"


def test_evaluate_scores_a_method_on_an_explicit_device(tmp_path, checkpoint):
    output = tmp_path / "evaluation.json"
    code = main(
        [
            "evaluate",
            "--checkpoint", str(checkpoint),
            "--dataset", "synthetic",
            "--methods", "grad-cam",
            "--images", "2",
            "--steps", "4",
            "--device", "cpu",
            "--output", str(output),
        ]
    )

    assert code == 0
    results = json.loads(output.read_text())["results"]
    assert [row["method"] for row in results] == ["grad-cam"]
    assert 0.0 <= results[0]["mass"] <= 1.0


def test_explain_writes_a_grid_on_an_explicit_device(tmp_path, checkpoint, png):
    code = main(
        [
            "explain",
            "--checkpoint", str(checkpoint),
            "--image", str(png),
            "--methods", "grad-cam", "saliency",
            "--device", "cpu",
            "--output", str(tmp_path),
        ]
    )

    assert code == 0
    assert (tmp_path / "sample_explanations.png").exists()
