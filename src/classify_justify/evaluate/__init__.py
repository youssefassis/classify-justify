"""Metrics that decide whether an explanation deserves to be believed."""

from classify_justify.evaluate.faithfulness import FaithfulnessScore, faithfulness
from classify_justify.evaluate.localization import pointing_game, relevance_mass, relevance_rank
from classify_justify.evaluate.sanity import SanityResult, model_randomization_test, spearman

__all__ = [
    "FaithfulnessScore",
    "SanityResult",
    "faithfulness",
    "model_randomization_test",
    "pointing_game",
    "relevance_mass",
    "relevance_rank",
    "spearman",
]
