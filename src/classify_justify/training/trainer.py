"""The training loop, and the three decisions that keep it honest.

**Early stopping tracks validation average precision, then F1.** At 10.6% defective, a
model that predicts "clean" for everything scores 0.894 accuracy and can improve its
loss while getting worse at the only job it has, so accuracy and loss are both out.

AP alone is not enough either: it is threshold-free, so it saturates at 1.000 the
moment the ranking is perfect, and a strict "better than best" test then locks in the
first epoch that got there — including one whose 0.5 threshold is still badly
calibrated. Ranking by `(AP, F1)` keeps AP as the objective and lets F1 separate
epochs it cannot distinguish.

**Augmentation is a per-sample transform**, applied when an image is loaded, so it can
never duplicate an image across the train/validation boundary.

**Normalisation statistics come from the training split alone** and are saved with the
checkpoint, so explanation preprocesses exactly as training did.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from classify_justify.data import (
    KolektorSDD2,
    SyntheticDefects,
    channel_statistics,
    eval_transform,
    stratified_split,
    train_transform,
)
from classify_justify.models import DefectClassifier, ModelConfig, save_checkpoint
from classify_justify.progress import progress
from classify_justify.training.metrics import ClassificationReport, class_weights, report


@dataclass
class TrainConfig:
    #: "kolektor" for the real dataset, "synthetic" for the no-download demo
    dataset: str = "kolektor"
    data_root: str | None = None
    output: str = "runs/kolektor"
    image_size: tuple[int, int] = (256, 640)
    #: only for the synthetic dataset: images per split
    synthetic_size: int = 512
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


def _dataset_factory(config: TrainConfig) -> Callable[..., Dataset]:
    """One builder for either dataset, so the training loop never branches.

    The synthetic set exists so the whole pipeline runs with no download and no GPU;
    the real one needs both. They expose the same `(image, label, mask)` items,
    `labels` property and `indices`/`transform` arguments, which is what lets a single
    call site serve both.
    """
    if config.dataset == "synthetic":
        # Distinct seeds give genuinely disjoint train and test images.
        seeds = {"train": 1, "test": 2}

        def build(split: str, **kwargs: object) -> Dataset:
            return SyntheticDefects(
                size=config.synthetic_size,
                image_size=min(config.image_size),
                seed=seeds[split],
                **kwargs,  # type: ignore[arg-type]
            )

        return build

    if config.dataset != "kolektor":
        raise ValueError(f"unknown dataset {config.dataset!r}; use 'kolektor' or 'synthetic'")
    if config.data_root is None:
        raise ValueError("the kolektor dataset needs --data-root")

    def build(split: str, **kwargs: object) -> Dataset:
        return KolektorSDD2(
            config.data_root,  # type: ignore[arg-type]
            split,
            image_size=config.image_size,
            **kwargs,  # type: ignore[arg-type]
        )

    return build


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
    make = _dataset_factory(config)
    full = make("train")
    train_indices, validation_indices = stratified_split(
        full.labels, config.validation_fraction, seed=config.seed
    )
    mean, std = channel_statistics(make("train", indices=train_indices))

    train_set = make(
        "train",
        indices=train_indices,
        transform=train_transform(mean, std, augment=config.augment),
    )
    validation_set = make(
        "train", indices=validation_indices, transform=eval_transform(mean, std)
    )
    test_set = make("test", transform=eval_transform(mean, std))

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
        "dataset": config.dataset,
    }

    best = (-1.0, -1.0)
    best_epoch = -1
    since_improvement = 0
    history: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        started = time.time()
        running = 0.0
        seen = 0
        # leave=False: the bar erases itself so the epoch's summary line below is the
        # only thing that survives in the scrollback.
        batches = progress(
            train_loader,
            enabled=verbose,
            desc=f"epoch {epoch}/{config.epochs}",
            unit="batch",
            leave=False,
        )
        for images, labels in batches:
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

        ranked = (scores.average_precision, scores.f1)
        if ranked > best:
            best = ranked
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
        print(
            f"\nbest epoch {best_epoch} (val AP {best[0]:.3f}, F1 {best[1]:.3f})"
            f"\ntest  {test_scores}",
            flush=True,
        )

    return TrainResult(
        best_average_precision=best[0],
        best_epoch=best_epoch,
        checkpoint=checkpoint,
        normalization=(mean, std),
        history=history,
        test=test_scores,
    )
