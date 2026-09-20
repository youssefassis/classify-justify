"""Metrics for a heavily imbalanced binary problem.

KolektorSDD2's training set is 10.6% defective. Accuracy is therefore close to
useless — a model that calls everything clean scores 0.894 — and so is early stopping
on validation loss, which a majority-class collapse can improve. Everything here is
reported for the defect class specifically.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ClassificationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    average_precision: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    def __str__(self) -> str:
        return (
            f"acc {self.accuracy:.3f}  P {self.precision:.3f}  R {self.recall:.3f}  "
            f"F1 {self.f1:.3f}  AP {self.average_precision:.3f}  "
            f"[tp {self.true_positives} fp {self.false_positives} "
            f"tn {self.true_negatives} fn {self.false_negatives}]"
        )


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Area under the precision-recall curve, computed without sklearn.

    Threshold-free, which is what makes it the right early-stopping signal: it does
    not reward a model for happening to sit well against a 0.5 cut-off.
    """
    if labels.sum() == 0:
        return 0.0
    order = scores.argsort(descending=True)
    ordered = labels[order].float()
    true_positives = ordered.cumsum(0)
    precision = true_positives / torch.arange(1, len(ordered) + 1, dtype=torch.float)
    # Sum precision only where a positive was retrieved: the standard AP estimator.
    return float((precision * ordered).sum() / ordered.sum())


def report(
    scores: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5
) -> ClassificationReport:
    """Score defect-class predictions against the truth."""
    if scores.shape != labels.shape:
        raise ValueError(f"scores {tuple(scores.shape)} do not match labels {tuple(labels.shape)}")
    predicted = (scores >= threshold).long()
    positive = labels == 1

    true_positives = int((predicted[positive] == 1).sum())
    false_negatives = int((predicted[positive] == 0).sum())
    false_positives = int((predicted[~positive] == 1).sum())
    true_negatives = int((predicted[~positive] == 0).sum())

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return ClassificationReport(
        accuracy=(true_positives + true_negatives) / max(len(labels), 1),
        precision=precision,
        recall=recall,
        f1=f1,
        average_precision=average_precision(scores, labels),
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
    )


def class_weights(labels: list[int], num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1.

    The notebook used `[p_defect, 1 - p_defect]`, which is not inverse frequency and
    happens to point the right way only by coincidence. This is the standard estimator:
    a class seen half as often gets twice the weight.
    """
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    if (counts == 0).any():
        raise ValueError(f"every class needs at least one example, got counts {counts.tolist()}")
    weights = counts.sum() / (num_classes * counts)
    return weights / weights.mean()
