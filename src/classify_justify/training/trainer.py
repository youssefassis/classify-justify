"""The training loop, and the three decisions that keep it honest.

**Early stopping tracks validation average precision, not loss or accuracy.** At 10.6%
defective, a model that predicts "clean" for everything scores 0.894 accuracy and can
improve its loss while getting worse at the only job it has.

**Augmentation is a per-sample transform**, applied when an image is loaded, so it can
never duplicate an image across the train/validation boundary.

**Normalisation statistics come from the training split alone** and are saved with the
checkpoint, so explanation preprocesses exactly as training did.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from classify_justify.data import (
    KolektorSDD2,
    channel_statistics,
    eval_transform,
    stratified_split,
    train_transform,
)
from classify_justify.models import DefectClassifier, ModelConfig, save_checkpoint
from classify_justify.training.metrics import ClassificationReport, class_weights, report


@dataclass
class TrainConfig:
    data_root: str
    output: str = "runs/kolektor"
    image_size: tuple[int, int] = (256, 640)
    widths: tuple[int, ...] = (16, 32, 64, 128)
    dropout: float = 0.3
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    patience: int = 8
    seed: int = 0
    device: str = "auto"
    # 0 by default: on macOS, forked DataLoader workers deadlock against the MPS
    # backend and the run hangs before the first epoch with the workers idle. Loading
    # this dataset costs ~17s an epoch single-threaded, so there is little to win.
    num_workers: int = 0
    augment: bool = True


@dataclass
class TrainResult:
    best_average_precision: float
    best_epoch: int
    checkpoint: Path
    normalization: tuple[float, float]
    history: list[dict] = field(default_factory=list)
    test: ClassificationReport | None = None


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick a device, preferring the GPU but never guessing wrong silently."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    def collate(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return images, labels

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> ClassificationReport:
    """Defect-class scores over a whole loader."""
    model.eval()
    scores, targets = [], []
    for images, labels in loader:
        probabilities = model(images.to(device)).softmax(dim=1)[:, 1]
        scores.append(probabilities.cpu())
        targets.append(labels)
    return report(torch.cat(scores), torch.cat(targets))


def train(config: TrainConfig, verbose: bool = True) -> TrainResult:
    """Train a defect classifier and return the best checkpoint's scores."""
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)

    # Split first, then fit statistics on the training half only.
    full = KolektorSDD2(config.data_root, "train", image_size=config.image_size)
    train_indices, validation_indices = stratified_split(
        full.labels, config.validation_fraction, seed=config.seed
    )
    statistics_source = KolektorSDD2(
        config.data_root, "train", image_size=config.image_size, indices=train_indices
    )
    mean, std = channel_statistics(statistics_source)

    train_set = KolektorSDD2(
        config.data_root,
        "train",
        image_size=config.image_size,
        transform=train_transform(mean, std, augment=config.augment),
        indices=train_indices,
    )
    validation_set = KolektorSDD2(
        config.data_root,
        "train",
        image_size=config.image_size,
        transform=eval_transform(mean, std),
        indices=validation_indices,
    )
    test_set = KolektorSDD2(
        config.data_root,
        "test",
        image_size=config.image_size,
        transform=eval_transform(mean, std),
    )

    train_loader = _loader(train_set, config.batch_size, True, config.num_workers)
    validation_loader = _loader(validation_set, config.batch_size, False, config.num_workers)
    test_loader = _loader(test_set, config.batch_size, False, config.num_workers)

    model = DefectClassifier(
        ModelConfig(
            in_channels=1,
            num_classes=2,
            widths=tuple(config.widths),
            dropout=config.dropout,
        )
    ).to(device)
    weights = class_weights([full.labels[i] for i in train_indices]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    checkpoint = output / "model.pt"
    metadata = {
        "normalization": {"mean": mean, "std": std},
        "image_size": list(config.image_size),
        "dataset": "KolektorSDD2",
    }

    best = -1.0
    best_epoch = -1
    since_improvement = 0
    history: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        started = time.time()
        running = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimiser.step()
            running += float(loss) * len(labels)
            seen += len(labels)

        scores = evaluate(model, validation_loader, device)
        history.append(
            {"epoch": epoch, "loss": running / max(seen, 1), "val_ap": scores.average_precision}
        )
        if verbose:
            print(
                f"epoch {epoch:3d}  loss {running / max(seen, 1):.4f}  "
                f"val {scores}  {time.time() - started:.0f}s",
                flush=True,
            )

        if scores.average_precision > best:
            best = scores.average_precision
            best_epoch = epoch
            since_improvement = 0
            save_checkpoint(model.cpu(), checkpoint, metadata)
            model.to(device)
        else:
            since_improvement += 1
            if since_improvement >= config.patience:
                if verbose:
                    print(f"early stop: no gain for {config.patience} epochs", flush=True)
                break

    # Report the test split with the best checkpoint, never the last one.
    from classify_justify.models import load_checkpoint

    best_model = load_checkpoint(checkpoint).to(device)
    test_scores = evaluate(best_model, test_loader, device)
    if verbose:
        print(f"\nbest epoch {best_epoch} (val AP {best:.3f})\ntest  {test_scores}", flush=True)

    return TrainResult(
        best_average_precision=best,
        best_epoch=best_epoch,
        checkpoint=checkpoint,
        normalization=(mean, std),
        history=history,
        test=test_scores,
    )
