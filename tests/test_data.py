import pytest
import torch

from classify_justify.data import SyntheticDefects, stratified_split
from classify_justify.data.kolektor import _is_image
from classify_justify.data.transforms import eval_transform, train_transform


class TestSyntheticDefects:
    def test_is_a_pure_function_of_seed_and_index(self):
        """A split is only reproducible if the data is identical across processes."""
        first = SyntheticDefects(size=8, seed=3)[5]
        second = SyntheticDefects(size=8, seed=3)[5]
        assert torch.equal(first[0], second[0])
        assert torch.equal(first[2], second[2])

    def test_a_different_seed_gives_different_images(self):
        assert not torch.equal(SyntheticDefects(size=8, seed=1)[5][0],
                               SyntheticDefects(size=8, seed=2)[5][0])

    def test_the_mask_marks_the_defect_and_only_the_defect(self):
        image, label, mask = SyntheticDefects(size=8, seed=0)[1]
        assert label == 1
        assert mask.any(), "a defective image must carry a mask"
        # The blob is the brightest thing present, so the peak belongs to the mask.
        assert mask.flatten()[int(image.flatten().argmax())]

    def test_clean_images_carry_an_empty_mask(self):
        _, label, mask = SyntheticDefects(size=8, seed=0)[0]
        assert label == 0
        assert not mask.any()

    def test_out_of_range_raises(self):
        with pytest.raises(IndexError):
            SyntheticDefects(size=4)[4]


class TestStratifiedSplit:
    def test_halves_are_disjoint_and_complete(self):
        labels = [0] * 100 + [1] * 12
        train, validation = stratified_split(labels, 0.2, seed=0)
        assert not set(train) & set(validation)
        assert sorted(train + validation) == list(range(len(labels)))

    def test_class_balance_is_preserved(self):
        labels = [0] * 2085 + [1] * 246
        _, validation = stratified_split(labels, 0.2, seed=0)
        held_out = sum(labels[i] for i in validation) / len(validation)
        assert held_out == pytest.approx(246 / 2331, abs=0.02)

    def test_a_rare_class_never_disappears(self):
        """With 2 positives and a 20% split, rounding would otherwise hold out zero."""
        labels = [0] * 50 + [1, 1]
        train, validation = stratified_split(labels, 0.2, seed=0)
        assert sum(labels[i] for i in validation) >= 1
        assert sum(labels[i] for i in train) >= 1

    def test_the_seed_controls_the_split(self):
        labels = [0] * 40 + [1] * 10
        assert stratified_split(labels, 0.2, seed=0) == stratified_split(labels, 0.2, seed=0)
        assert stratified_split(labels, 0.2, seed=0) != stratified_split(labels, 0.2, seed=1)

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
    def test_a_nonsense_fraction_is_rejected(self, fraction):
        with pytest.raises(ValueError, match="validation_fraction"):
            stratified_split([0, 1], fraction)


class TestKolektorFileFilter:
    def test_accepts_images_and_rejects_masks(self):
        assert _is_image("10301")
        assert not _is_image("10301_GT")

    def test_rejects_the_archives_stray_duplicates(self):
        """The published zip contains these two; both corrupt a run if let through."""
        assert not _is_image("10301 (copy)")
        assert not _is_image("10301_GT (copy)")


class TestTransforms:
    def test_eval_transform_is_deterministic(self):
        """Attribution explains one exact tensor; evaluation must not randomise it."""
        transform = eval_transform(0.3, 0.2)
        image = torch.rand(1, 32, 32)
        assert torch.equal(transform(image), transform(image))

    def test_train_transform_actually_augments(self):
        torch.manual_seed(0)
        transform = train_transform(0.3, 0.2, augment=True)
        image = torch.rand(1, 32, 32)
        outputs = [transform(image) for _ in range(12)]
        assert any(not torch.equal(outputs[0], other) for other in outputs[1:])

    def test_augmentation_can_be_turned_off(self):
        transform = train_transform(0.3, 0.2, augment=False)
        image = torch.rand(1, 32, 32)
        assert torch.equal(transform(image), transform(image))
