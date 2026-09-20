"""The classifier under explanation.

Two properties matter more than accuracy here, because every attribution method in
`classify_justify.explain` depends on them:

1. **The head is a global average pool followed by a single linear layer.** CAM-family
   methods weight a convolutional feature map by how much each channel moves a class
   logit; that argument only holds when the logit is a linear function of pooled
   channels. Flattening the feature map into a `Linear` instead — as a plain
   `nn.Flatten` head does — ties every logit to absolute spatial position and makes the
   resulting heatmaps hard to defend.
2. **Construction never touches the data.** The model is defined by its config alone,
   so a checkpoint can be loaded on a machine that has no dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    """Everything needed to rebuild the architecture, stored beside the weights."""

    in_channels: int = 1
    num_classes: int = 2
    widths: tuple[int, ...] = (16, 32, 64, 128)
    dropout: float = 0.3

    def __post_init__(self) -> None:
        if self.in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {self.in_channels}")
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if not self.widths:
            raise ValueError("widths must name at least one convolutional block")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv-BN-ReLU twice, then halve the resolution.

    BatchNorm rather than the plain conv stack of the original notebook: without it the
    deeper widths here train far more slowly, and attribution maps taken mid-training
    are dominated by scale differences between channels.

    The ReLUs are deliberately **not** in-place. Guided Backprop and DeepLIFT attach
    backward hooks to each ReLU to rewrite its gradient; an in-place ReLU overwrites
    the tensor those hooks need and silently yields wrong attributions.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class DefectClassifier(nn.Module):
    """A small CNN whose last convolutional block is the natural CAM target layer."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        blocks = []
        in_ch = self.config.in_channels
        for width in self.config.widths:
            blocks.append(_block(in_ch, width))
            in_ch = width
        self.features = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc = nn.Linear(self.config.widths[-1], self.config.num_classes)

    @property
    def target_layer(self) -> nn.Module:
        """The layer CAM methods attribute against: the last block's final ReLU.

        Taking the ReLU rather than the `MaxPool2d` after it keeps the map at the
        higher resolution, which matters because these defects are small.
        """
        last_block = self.features[-1]
        relus = [m for m in last_block if isinstance(m, nn.ReLU)]
        return relus[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def build_model(**kwargs: object) -> DefectClassifier:
    """Build a classifier from loose keyword arguments (as parsed from YAML)."""
    field_names = {f for f in ModelConfig.__dataclass_fields__}
    unknown = set(kwargs) - field_names
    if unknown:
        raise ValueError(f"unknown model options: {sorted(unknown)}")
    if "widths" in kwargs and kwargs["widths"] is not None:
        kwargs["widths"] = tuple(kwargs["widths"])  # type: ignore[arg-type]
    return DefectClassifier(ModelConfig(**kwargs))  # type: ignore[arg-type]


def save_checkpoint(model: DefectClassifier, path: str | Path) -> None:
    """Store weights together with the config that rebuilds the architecture."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> DefectClassifier:
    """Rebuild a model from a checkpoint alone — no dataset, no globals."""
    blob = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not {"config", "state_dict"} <= blob.keys():
        raise ValueError(
            f"{path} is not a classify-justify checkpoint "
            "(expected 'config' and 'state_dict' keys)"
        )
    config = ModelConfig(**{**blob["config"], "widths": tuple(blob["config"]["widths"])})
    model = DefectClassifier(config)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model
