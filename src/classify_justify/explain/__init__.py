"""Attribution methods, all behind one interface.

Importing this package is what populates the registry: each module registers its
methods on import, so `available()` is only complete once they have all been loaded.
"""

from classify_justify.explain import cam, gradient, perturbation  # noqa: F401  (registration)
from classify_justify.explain.base import (
    Explainer,
    available,
    build,
    build_all,
    get,
    register,
    to_heatmap,
)

__all__ = [
    "Explainer",
    "available",
    "build",
    "build_all",
    "get",
    "register",
    "to_heatmap",
]
