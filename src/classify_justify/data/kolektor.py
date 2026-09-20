"""KolektorSDD2: surface defects on production parts, with pixel-level masks.

Chosen over GDXray — which the original notebook's data resembled — for one practical
reason: GDXray's distribution is dead. Every official download returns "File Deleted"
and the only live mirror is gated behind manual approval. A project whose point is
that you can clone it and reproduce the result cannot depend on that.

KolektorSDD2 is an ungated 853 MB download, and every image ships a `_GT` mask rather
than a bounding box, so `evaluate.localization` scores against the defect's actual
shape instead of a rectangle around it.

    Bozic, Tabernik, Skocaj (2021), "Mixed supervision for surface-defect detection:
    from weakly to fully supervised learning", Computers in Industry.

Licensed CC BY-NC-SA 4.0: non-commercial, share-alike, attribution required. The data
is never vendored into this repository — `download` fetches it on demand.
"""

from __future__ import annotations

import shutil
import ssl
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

URL = "https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip"
ARCHIVE_BYTES = 853_126_555
#: The archive's own top-level directories; there is no official validation split.
SPLITS = ("train", "test")


def _ssl_context() -> ssl.SSLContext:
    """A context with a CA bundle that actually exists.

    A python.org install on macOS does not populate a system certificate store, so
    `urlopen` fails with CERTIFICATE_VERIFY_FAILED on an otherwise fine connection —
    the same URL downloads without complaint through curl. Using certifi's bundle
    makes the download work out of the box instead of failing in a way that reads
    like the dataset is unreachable.
    """
    import certifi

    return ssl.create_default_context(cafile=certifi.where())


def download(root: str | Path, force: bool = False) -> Path:
    """Fetch and extract KolektorSDD2 into `root`, returning the dataset directory.

    Skips the download when the data is already present, so this is safe to call from
    a training run. The archive is removed after extraction.
    """
    root = Path(root)
    target = root / "KolektorSDD2"
    if target.exists() and not force:
        return target

    root.mkdir(parents=True, exist_ok=True)
    archive = root / "KolektorSDD2.zip"

    # Streamed to disk rather than held in memory: this is an 853 MB file.
    with urllib.request.urlopen(URL, context=_ssl_context()) as response:
        declared = int(response.headers.get("Content-Length", 0))
        if declared and declared != ARCHIVE_BYTES:
            raise RuntimeError(
                f"{URL} served {declared} bytes, expected {ARCHIVE_BYTES}. "
                "The dataset may have been republished; verify before trusting it."
            )
        with archive.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)

    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(target)
    archive.unlink()
    return target


def _is_image(stem: str) -> bool:
    """True for an input image, false for a mask or one of the archive's stray files.

    The published archive contains `10301 (copy).png` and `10301_GT (copy).png`. Both
    matter, and both are easy to miss:

    - `10301_GT (copy)` does not end in `_GT`, so the obvious "skip anything ending in
      _GT" filter loads a **mask** as an input image and labels it defect-free.
    - `10301 (copy)` is a byte-identical duplicate of `10301`. Left in, the two can
      land on opposite sides of a train/validation split — the exact leakage this
      project exists to avoid.

    Requiring a purely numeric stem excludes both, and every genuine KolektorSDD2
    image is numbered.
    """
    return stem.isdigit()


class KolektorSDD2(Dataset):
    """Image, binary label, and defect mask.

    The label is derived from the mask rather than read from a separate annotation:
    an image is defective exactly when its `_GT` mask has a non-zero pixel. That keeps
    the classification target and the localisation ground truth from ever disagreeing.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (256, 640),
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        indices: list[int] | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.directory = Path(root) / split
        if not self.directory.is_dir():
            raise FileNotFoundError(
                f"{self.directory} not found — run `python -m classify_justify download` or "
                "classify_justify.data.kolektor.download() first"
            )

        self.image_size = image_size
        self.transform = transform
        # Sorted so the order is identical on every machine, which a seeded split relies on.
        stems = sorted(p.stem for p in self.directory.glob("*.png") if _is_image(p.stem))
        if not stems:
            raise FileNotFoundError(f"no images in {self.directory}")
        self.stems = [stems[i] for i in indices] if indices is not None else stems

    def __len__(self) -> int:
        return len(self.stems)

    def _load(self, path: Path, nearest: bool) -> Image.Image:
        # Masks resample with NEAREST: bilinear would invent grey values at the defect
        # boundary and turn a binary mask into something that needs thresholding.
        resample = Image.NEAREST if nearest else Image.BILINEAR
        height, width = self.image_size
        return Image.open(path).resize((width, height), resample)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        stem = self.stems[index]
        image = self._load(self.directory / f"{stem}.png", nearest=False).convert("L")
        mask_path = self.directory / f"{stem}_GT.png"

        tensor = pil_to_tensor(image).float() / 255.0

        if mask_path.exists():
            mask_image = self._load(mask_path, nearest=True).convert("L")
            mask = pil_to_tensor(mask_image) > 0
        else:
            mask = torch.zeros(1, *self.image_size, dtype=torch.bool)

        label = int(mask.any())
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, label, mask

    @property
    def labels(self) -> list[int]:
        """Labels for every item, for stratified splitting and class weighting.

        Reads only the masks, which are far smaller than the images.
        """
        labels = []
        for stem in self.stems:
            mask_path = self.directory / f"{stem}_GT.png"
            if not mask_path.exists():
                labels.append(0)
                continue
            with Image.open(mask_path) as mask:
                labels.append(int(mask.getbbox() is not None))
        return labels
