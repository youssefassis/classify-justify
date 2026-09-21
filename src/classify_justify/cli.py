"""Command line: `python -m classify_justify download | train | explain | evaluate`.

Four verbs, because the project makes four claims and each one should be checkable
from a shell without reading the source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from classify_justify import __version__
from classify_justify.progress import progress, write


def _add_download(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("download", help="fetch KolektorSDD2 (853 MB)")
    parser.add_argument("--root", default="data", help="where to put the dataset")
    parser.add_argument("--force", action="store_true", help="re-download even if present")


def _add_train(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train", help="train the defect classifier")
    parser.add_argument(
        "--dataset", default="kolektor", choices=("kolektor", "synthetic"),
        help="synthetic needs no download and no GPU",
    )
    parser.add_argument("--data-root", help="the KolektorSDD2 directory (kolektor only)")
    parser.add_argument("--output", default="runs/kolektor")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    parser.add_argument(
        "--num-workers", type=int, default=0, help="0 avoids the macOS/MPS DataLoader hang"
    )
    parser.add_argument("--no-augment", action="store_true")


def _add_explain(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("explain", help="write heatmaps for one image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, help="a PNG from the dataset")
    parser.add_argument("--methods", nargs="+", default=None, help="default: every method")
    parser.add_argument("--target", type=int, default=1, help="class to explain")
    parser.add_argument("--output", default="runs/explanations")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="fixes the sampling in rise, smoothgrad, gradient-shap, lime and kernel-shap",
    )


def _add_evaluate(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("evaluate", help="score every method and rank them")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset", default="kolektor", choices=("kolektor", "synthetic")
    )
    parser.add_argument("--data-root", help="the KolektorSDD2 directory (kolektor only)")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--images", type=int, default=25, help="defective images to score")
    parser.add_argument("--steps", type=int, default=64, help="deletion/insertion steps")
    parser.add_argument("--sanity", action="store_true", help="also run the randomisation test")
    parser.add_argument("--output", default="runs/evaluation.json")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="fixes the sampling in rise, smoothgrad, gradient-shap, lime and kernel-shap",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m classify_justify",
        description="Train an X-ray defect classifier, explain it, and measure the explanations.",
    )
    parser.add_argument("--version", action="version", version=f"classify-justify {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_download(subparsers)
    _add_train(subparsers)
    _add_explain(subparsers)
    _add_evaluate(subparsers)
    return parser


def _run_download(args: argparse.Namespace) -> int:
    from classify_justify.data import download

    path = download(args.root, force=args.force)
    print(f"dataset ready at {path}")
    return 0


def _run_train(args: argparse.Namespace) -> int:
    from classify_justify.training import TrainConfig, train

    synthetic = args.dataset == "synthetic"
    result = train(
        TrainConfig(
            dataset=args.dataset,
            data_root=args.data_root,
            image_size=(64, 64) if synthetic else (256, 640),
            output=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            augment=not args.no_augment,
        )
    )
    print(f"\nbest checkpoint: {result.checkpoint}")
    return 0


def _load_image(path: str, checkpoint: str) -> torch.Tensor:
    """Load a PNG exactly as training preprocessed it, using the checkpoint's metadata."""
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor

    from classify_justify.models import read_metadata

    metadata = read_metadata(checkpoint)
    height, width = metadata.get("image_size", [256, 640])
    normalization = metadata.get("normalization", {"mean": 0.0, "std": 1.0})

    image = Image.open(path).convert("L").resize((width, height), Image.BILINEAR)
    tensor = pil_to_tensor(image).float() / 255.0
    tensor = (tensor - normalization["mean"]) / normalization["std"]
    return tensor.unsqueeze(0)


def _run_explain(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from classify_justify.explain import available, build, to_heatmap
    from classify_justify.models import load_checkpoint

    model = load_checkpoint(args.checkpoint)
    image = _load_image(args.image, args.checkpoint)
    methods = args.methods or available()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    columns = min(len(methods) + 1, 4)
    rows = -(-(len(methods) + 1) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 2 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    axes[0].imshow(image[0, 0], cmap="gray")
    axes[0].set_title("input")
    axes[0].axis("off")

    explaining = progress(methods, desc="explaining", unit="method", leave=False)
    for axis, name in zip(axes[1:], explaining, strict=False):
        explainer = build(name, model, model.target_layer, seed=args.seed)
        relevance = explainer.attribute(image, args.target)
        axis.imshow(image[0, 0], cmap="gray")
        axis.imshow(to_heatmap(relevance)[0, 0], cmap="inferno", alpha=0.55)
        axis.set_title(name, fontsize=9)
        axis.axis("off")
    for axis in axes[len(methods) + 1 :]:
        axis.axis("off")

    destination = output / f"{Path(args.image).stem}_explanations.png"
    figure.tight_layout()
    figure.savefig(destination, dpi=140, bbox_inches="tight")
    print(f"wrote {destination}")
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    from classify_justify.data import KolektorSDD2, SyntheticDefects, eval_transform
    from classify_justify.evaluate import (
        faithfulness,
        model_randomization_test,
        pointing_game,
        relevance_mass,
        relevance_rank,
    )
    from classify_justify.explain import available, build, get
    from classify_justify.models import load_checkpoint, read_metadata

    model = load_checkpoint(args.checkpoint)
    metadata = read_metadata(args.checkpoint)
    normalization = metadata.get("normalization", {"mean": 0.0, "std": 1.0})
    image_size = tuple(metadata.get("image_size", [256, 640]))

    transform = eval_transform(normalization["mean"], normalization["std"])
    if args.dataset == "synthetic":
        dataset = SyntheticDefects(
            size=256, image_size=min(image_size), seed=2, transform=transform
        )
    else:
        if not args.data_root:
            parser_error = "the kolektor dataset needs --data-root"
            raise SystemExit(parser_error)
        dataset = KolektorSDD2(
            args.data_root, args.split, image_size=image_size, transform=transform
        )
    defective = [i for i, label in enumerate(dataset.labels) if label == 1][: args.images]
    if not defective:
        print("no defective images in this split")
        return 1

    methods = args.methods or available()
    results = []
    for name in progress(methods, desc="methods", unit="method", leave=False):
        explainer = build(name, model, model.target_layer, seed=args.seed)
        totals = dict.fromkeys(("deletion", "insertion", "pointing", "mass", "rank"), 0.0)
        for index in progress(defective, desc=name, unit="image", leave=False):
            image, _, mask = dataset[index]
            batched = image.unsqueeze(0)
            relevance = explainer.attribute(batched, 1)
            scores = faithfulness(model, batched, relevance, 1, steps=args.steps)
            totals["deletion"] += scores.deletion_auc
            totals["insertion"] += scores.insertion_auc
            totals["pointing"] += float(pointing_game(relevance, mask))
            totals["mass"] += relevance_mass(relevance, mask)
            totals["rank"] += relevance_rank(relevance, mask)

        row = {"method": name, **{k: v / len(defective) for k, v in totals.items()}}
        if args.sanity:
            image, _, _ = dataset[defective[0]]
            sanity = model_randomization_test(
                model, get(name), image.unsqueeze(0), 1, seed=args.seed
            )
            row["sanity_correlation"] = sanity.final_correlation
            row["sanity_passed"] = sanity.passed()
        results.append(row)
        write(
            f"{name:22} del {row['deletion']:.3f}  ins {row['insertion']:.3f}  "
            f"point {row['pointing']:.2f}  mass {row['mass']:.3f}  rank {row['rank']:.3f}"
            + (f"  sanity {row['sanity_correlation']:+.2f}" if args.sanity else "")
        )

    results.sort(key=lambda r: r["insertion"] - r["deletion"], reverse=True)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"split": args.split, "images": len(defective), "results": results}, indent=2)
    )
    print(f"\nwrote {destination}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "download": _run_download,
        "train": _run_train,
        "explain": _run_explain,
        "evaluate": _run_evaluate,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
