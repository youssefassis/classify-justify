"""Progress bars, with the one rule that keeps them from polluting logs.

Four commands here run long and silently: the 853 MB dataset download, an epoch of
training, and the `explain` and `evaluate` loops, which run every method in turn and
are dominated by the perturbation methods' thousands of forward passes. On
KolektorSDD2 that is hours with nothing on screen.

Everything routes through this module so the disable rule lives in one place: a bar
is drawn only when stderr is a terminal. Redirected output and CI logs — where a
progress bar becomes thousands of lines of carriage returns — get nothing.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import Any, TypeVar

from tqdm import tqdm

T = TypeVar("T")

#: Printing goes to stdout while bars go to stderr, so a caller can still pipe the
#: results of `evaluate` somewhere without catching the bar as well.
write = tqdm.write


def _drawable(enabled: bool = True) -> bool:
    return enabled and sys.stderr.isatty()


def progress(iterable: Iterable[T], enabled: bool = True, **kwargs: Any) -> Iterable[T]:
    """Wrap `iterable` in a progress bar, or return it untouched when nothing can see it."""
    return tqdm(iterable, disable=not _drawable(enabled), **kwargs)


def reading(stream: Any, total: int, **kwargs: Any) -> Any:
    """A context manager yielding `stream` wrapped so reads advance a byte-counting bar.

    `shutil.copyfileobj` takes no callback, so the count has to come from the stream
    itself rather than from the copy loop.
    """
    return tqdm.wrapattr(
        stream, "read", total=total or None, disable=not _drawable(), **kwargs
    )
