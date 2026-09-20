import pytest
import torch

from classify_justify.training import average_precision, class_weights, report, resolve_device


class TestClassWeights:
    def test_a_rarer_class_gets_more_weight(self):
        weights = class_weights([0] * 2085 + [1] * 246)
        assert weights[1] > weights[0]
        # Inverse frequency: 2085/246 is about 8.5x.
        assert float(weights[1] / weights[0]) == pytest.approx(2085 / 246, rel=0.01)

    def test_a_balanced_dataset_gets_equal_weights(self):
        assert torch.allclose(class_weights([0] * 10 + [1] * 10), torch.ones(2))

    def test_a_missing_class_is_an_error(self):
        with pytest.raises(ValueError, match="at least one example"):
            class_weights([0] * 10)


class TestAveragePrecision:
    def test_a_perfect_ranking_scores_one(self):
        scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
        assert average_precision(scores, torch.tensor([1, 1, 0, 0])) == pytest.approx(1.0)

    def test_the_worst_ranking_scores_low(self):
        scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
        assert average_precision(scores, torch.tensor([0, 0, 1, 1])) < 0.6

    def test_no_positives_scores_zero(self):
        assert average_precision(torch.tensor([0.5, 0.4]), torch.tensor([0, 0])) == 0.0

    def test_it_ignores_the_threshold(self):
        """The point of AP over F1: a monotone rescaling must not change the score."""
        labels = torch.tensor([1, 0, 1, 0])
        scores = torch.tensor([0.9, 0.6, 0.5, 0.1])
        assert average_precision(scores, labels) == pytest.approx(
            average_precision(scores * 0.1, labels)
        )


class TestReport:
    def test_it_counts_the_confusion_matrix(self):
        scores = torch.tensor([0.9, 0.9, 0.1, 0.1])
        result = report(scores, torch.tensor([1, 0, 1, 0]))
        assert (result.true_positives, result.false_positives) == (1, 1)
        assert (result.true_negatives, result.false_negatives) == (1, 1)

    def test_a_collapsed_model_scores_high_accuracy_and_zero_recall(self):
        """Why early stopping tracks AP: this model is useless and 89% accurate."""
        labels = torch.tensor([0] * 894 + [1] * 110)
        result = report(torch.zeros(len(labels)), labels)
        assert result.accuracy == pytest.approx(894 / 1004, abs=1e-3)
        assert result.recall == 0.0
        assert result.f1 == pytest.approx(0.0)

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="do not match labels"):
            report(torch.zeros(3), torch.zeros(4, dtype=torch.long))


def test_resolve_device_honours_an_explicit_choice():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "mps", "cuda"}
