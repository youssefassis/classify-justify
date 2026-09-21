"""Making the stochastic attribution methods repeat.

Five of the seventeen methods draw random numbers: `rise` samples masks, `smoothgrad`
perturbs the input, `gradient-shap` samples baselines, and `lime` and `kernel-shap`
sample which superpixels to switch off. The model-randomisation test draws too, when
it re-initialises a layer. Left alone, every one of those numbers moves between runs —
in a project whose claim is that an explanation can be measured, a measurement that
does not repeat is not one.

Captum takes its randomness from the two global generators, torch's and numpy's, so
seeding those is the only handle there is. Seeding them outright would be a side
effect on the caller — a training loop that explains a batch mid-run would have its
own stream silently reset, and its next epoch would differ — so both are saved first
and put back on the way out.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy
import torch


@contextmanager
def seeded(seed: int | None, device: torch.device | None = None) -> Iterator[None]:
    """Run the body from a fixed RNG state, leaving the caller's stream untouched.

    `seed=None` does nothing at all, so a caller can pass an optional seed straight
    through without branching around it.
    """
    if seed is None:
        yield
        return

    # The CPU generator is forked unconditionally. An accelerator keeps its own, and
    # has to be named for it to be forked as well — RISE draws its masks on the
    # input's device, so on MPS or CUDA that is where the randomness lives.
    device = device or torch.device("cpu")
    devices = [] if device.type == "cpu" else [device]

    # numpy as well as torch: Captum's GradientShap draws its interpolation
    # coefficient from `np.random.uniform`, so seeding torch alone leaves that method
    # moving between runs. numpy ships no fork helper, so its state is saved by hand.
    numpy_state = numpy.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices, device_type=device.type):
            torch.manual_seed(seed)
            numpy.random.seed(seed)
            yield
    finally:
        numpy.random.set_state(numpy_state)
