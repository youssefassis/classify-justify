import pytest
import torch

from classify_justify.evaluate import (
    faithfulness,
    model_randomization_test,
    pointing_game,
    relevance_mass,
    relevance_rank,
    spearman,
)
from classify_justify.explain import build, get


def _peaked_map(size: int, y: int, x: int) -> torch.Tensor:
    relevance = torch.zeros(1, 1, size, size)
    relevance[0, 0, y, x] = 1.0
    return relevance


def _mask(size: int, y: int, x: int, radius: int = 2) -> torch.Tensor:
    mask = torch.zeros(1, size, size, dtype=torch.bool)
    mask[0, y - radius : y + radius, x - radius : x + radius] = True
    return mask


class TestLocalization:
    def test_pointing_game_hits_and_misses(self):
        assert pointing_game(_peaked_map(32, 10, 10), _mask(32, 10, 10))
        assert not pointing_game(_peaked_map(32, 30, 30), _mask(32, 10, 10))

    def test_relevance_mass_is_a_share(self):
        inside = _peaked_map(32, 10, 10)
        assert relevance_mass(inside, _mask(32, 10, 10)) == pytest.approx(1.0)
        assert relevance_mass(inside, _mask(32, 28, 28)) == pytest.approx(0.0)

    def test_relevance_mass_ignores_negative_relevance(self):
        """Negative relevance argues against the class; it is not evidence for it."""
        relevance = torch.zeros(1, 1, 32, 32)
        relevance[0, 0, 10, 10] = 1.0
        relevance[0, 0, 28, 28] = -5.0
        assert relevance_mass(relevance, _mask(32, 10, 10)) == pytest.approx(1.0)

    def test_relevance_rank_normalises_for_defect_size(self):
        relevance = torch.rand(1, 1, 32, 32)
        mask = _mask(32, 16, 16, radius=4)
        relevance[0, 0][mask[0]] += 10.0  # make the defect unambiguously the top region
        assert relevance_rank(relevance, mask) == pytest.approx(1.0)

    def test_metrics_reject_bad_input(self):
        with pytest.raises(ValueError, match="mask is empty"):
            pointing_game(_peaked_map(32, 1, 1), torch.zeros(1, 32, 32, dtype=torch.bool))
        with pytest.raises(ValueError, match="does not match mask"):
            relevance_mass(_peaked_map(32, 1, 1), _mask(16, 8, 8))


class TestFaithfulness:
    def test_scores_are_bounded_probabilities(self, trained_model, defect_sample):
        image, _ = defect_sample
        relevance = build("integrated-gradients", trained_model).attribute(image, 1)
        score = faithfulness(trained_model, image, relevance, target=1, steps=16)

        assert 0.0 <= score.deletion_auc <= 1.0
        assert 0.0 <= score.insertion_auc <= 1.0
        assert len(score.deletion_curve) == len(score.insertion_curve)

    def test_the_curves_start_at_opposite_ends(self, trained_model, defect_sample):
        """Deletion begins on the intact image; insertion begins on an empty one."""
        image, _ = defect_sample
        relevance = build("integrated-gradients", trained_model).attribute(image, 1)
        score = faithfulness(trained_model, image, relevance, target=1, steps=16)

        with torch.no_grad():
            intact = float(trained_model(image).softmax(1)[0, 1])
            empty = float(trained_model(torch.zeros_like(image)).softmax(1)[0, 1])
        assert score.deletion_curve[0] == pytest.approx(intact, abs=1e-5)
        assert score.insertion_curve[0] == pytest.approx(empty, abs=1e-5)

    def test_a_good_map_beats_a_random_one(self, trained_model, defect_sample):
        """The metric has to rank a real explanation above noise, or it measures nothing."""
        image, _ = defect_sample
        torch.manual_seed(0)
        real = build("integrated-gradients", trained_model).attribute(image, 1)
        noise = torch.rand_like(real)

        good = faithfulness(trained_model, image, real, target=1, steps=32)
        bad = faithfulness(trained_model, image, noise, target=1, steps=32)
        assert good.insertion_auc > bad.insertion_auc

    def test_rejects_a_batch_and_a_size_mismatch(self, trained_model, defect_sample):
        image, _ = defect_sample
        relevance = torch.rand_like(image)
        with pytest.raises(ValueError, match="one image at a time"):
            faithfulness(trained_model, image.repeat(2, 1, 1, 1), relevance, 1)
        with pytest.raises(ValueError, match="does not match image"):
            faithfulness(trained_model, image, torch.rand(1, 1, 8, 8), 1)


class TestSanity:
    def test_spearman_endpoints(self):
        values = torch.randn(64)
        assert spearman(values, values) == pytest.approx(1.0, abs=1e-5)
        assert spearman(values, -values) == pytest.approx(-1.0, abs=1e-5)
        assert spearman(values, torch.zeros(64)) == 0.0  # no ranking to correlate

    def test_randomization_does_not_damage_the_caller_model(self, trained_model, defect_sample):
        image, _ = defect_sample
        before = trained_model.fc.weight.clone()
        model_randomization_test(trained_model, get("saliency"), image, 1, max_layers=2)
        assert torch.equal(trained_model.fc.weight, before)

    def test_a_weight_dependent_method_decorrelates(self, trained_model, defect_sample):
        """Saliency depends on the weights, so destroying them must change the map."""
        image, _ = defect_sample
        result = model_randomization_test(trained_model, get("saliency"), image, 1)
        assert result.passed()
        assert len(result.layers) == len(result.correlations)

    def test_guided_backprop_fails_the_check(self, trained_model, defect_sample):
        """Adebayo et al. (2018), reproduced: the control method must fail.

        Guided Backprop stays correlated with its original map after every layer has
        been randomised — it is reporting edges in the image, not the decision. If
        this ever starts passing, the sanity check has stopped working, not Guided
        Backprop started being trustworthy.
        """
        image, _ = defect_sample
        result = model_randomization_test(trained_model, get("guided-backprop"), image, 1)
        assert not result.passed()
        assert abs(result.correlations[0]) > 0.8  # unmoved by randomising the head
