# classify-justify

Train a surface-defect classifier, explain it with seventeen attribution methods, and
**measure whether those explanations can be trusted**.

Most explainability code stops at a heatmap. A heatmap is easy to produce and easy to
believe: it is a picture, it lands somewhere plausible, and nothing about it says
whether it reflects the model's decision or merely the edges in the image. This repo
treats the heatmap as a claim and scores it — for faithfulness to the model, for
agreement with ground truth, and for whether it depends on the trained weights at all.

## Quickstart

No download, no GPU. Trains in about fifteen seconds on a laptop CPU:

```bash
uv venv && uv pip install -e ".[dev]"

python -m classify_justify train --dataset synthetic --output runs/synthetic --device cpu
python -m classify_justify evaluate --checkpoint runs/synthetic/model.pt --dataset synthetic --sanity
python -m classify_justify explain --checkpoint runs/synthetic/model.pt --image <some.png>
```

The synthetic dataset is generated, not downloaded: textured images, half carrying a
small bright blob, with the blob's exact position known. That known position is the
point — the localisation metrics need ground truth, and generating it means the whole
pipeline runs anywhere, including in CI.

## Attribution methods

Seventeen, behind one interface: `attribute(inputs, target)` returns relevance shaped
`(B, 1, H, W)` at input resolution for every one of them, which is what lets the
evaluation suite score them all in the same loop.

**CAM family** — weight a convolutional feature map by channel, sum, keep the positive
part. Captum ships none of these, so they are implemented in
[`explain/cam.py`](src/classify_justify/explain/cam.py); they differ only in where the
weights come from, which is the whole taxonomy:

| method | channel weights from | gradients? |
|---|---|---|
| `grad-cam` | spatially averaged gradient | yes |
| `grad-cam++` | gradient, reweighted so many small activations are not drowned out | yes |
| `xgrad-cam` | gradient, weighted by each position's share of its channel | yes |
| `layer-cam` | per-position positive gradient (weights vary *within* a channel) | yes |
| `score-cam` | forward passes with the channel as an input mask | no |
| `ablation-cam` | the logit drop when the channel is removed | no |

The last two cost one forward pass per channel but survive the gradient saturation
that makes the first four unreliable on a confident model.

**Gradient-based** — thin adapters over [Captum](https://captum.ai):

| method | idea |
|---|---|
| `saliency` | the raw gradient of the class logit |
| `input-x-gradient` | gradient multiplied by the input |
| `integrated-gradients` | gradient integrated from a black-image baseline |
| `smoothgrad` | gradients averaged over noisy copies of the input |
| `deeplift` | contribution against a reference activation |
| `gradient-shap` | expected gradients over a baseline distribution |
| `guided-backprop` | backward pass with negative gradients clipped |

`guided-backprop` is included **as a control**, not a recommendation: it draws clean
edges while depending only weakly on the weights.

**Perturbation-based** — change the input, watch the logit move. Immune to gradient
saturation; one to two orders of magnitude slower:

| method | idea |
|---|---|
| `occlusion` | slide a patch over the image and record the drop |
| `rise` | average many random masks, weighted by the score each produced |
| `lime` | fit a linear surrogate on superpixel on/off samples |
| `kernel-shap` | the same, with Shapley kernel weights |

## Measuring the explanations

**Faithfulness** — rank pixels by relevance, then remove them in order (deletion,
*lower is better*) or add them to an empty image in order (insertion, *higher is
better*). Both are reported: deletion alone rewards a map that finds fragile pixels,
insertion alone rewards one that highlights everything.

**Localisation** — does the map land on the defect? `pointing_game` asks whether the
most-relevant pixel falls inside the mask, `relevance_mass` measures the share of
positive relevance inside it, `relevance_rank` is the precision of the top-|mask|
pixels. Faithfulness and localisation disagree more often than you would expect.

**Sanity** — the model-randomisation test of Adebayo et al. (2018). Randomise layers
from the output backwards and watch how far the attribution moves; a map that survives
the destruction of the weights is describing the image, not the decision. Maps are
compared as magnitudes, following the paper.

## Results

Every number is from the **synthetic** dataset — a classifier at F1 1.000 on 512
held-out images, scored over 12 defective images. Industrial numbers need a GPU.

| method | deletion ↓ | insertion ↑ | pointing ↑ | mass ↑ | rank ↑ | sanity |
|---|---|---|---|---|---|---|
| `deeplift` | 0.103 | 0.984 | 1.00 | 0.734 | 0.786 | +0.53 |
| `input-x-gradient` | 0.105 | 0.984 | 1.00 | 0.653 | 0.750 | +0.48 |
| `gradient-shap` | 0.106 | 0.985 | 1.00 | 0.747 | 0.767 | +0.38 |
| `integrated-gradients` | 0.107 | 0.984 | 1.00 | 0.726 | 0.764 | +0.48 |
| `saliency` | 0.112 | 0.983 | 0.92 | 0.282 | 0.621 | +0.17 |
| `smoothgrad` | 0.123 | 0.986 | 1.00 | 0.640 | 0.803 | +0.22 |
| `guided-backprop` | 0.127 | 0.984 | 1.00 | 0.359 | 0.702 | +0.41 |
| `ablation-cam` | 0.155 | 0.985 | 0.67 | 0.059 | 0.515 | +0.71 |
| `grad-cam++` | 0.156 | 0.985 | 0.67 | 0.057 | 0.519 | +0.02 |
| `score-cam` | 0.158 | 0.984 | 0.67 | 0.060 | 0.536 | +0.37 |
| `grad-cam` | 0.161 | 0.984 | 0.67 | 0.060 | 0.510 | +0.54 |
| `xgrad-cam` | 0.162 | 0.984 | 0.67 | 0.060 | 0.509 | +0.56 |
| `rise` | 0.165 | 0.985 | 1.00 | 0.014 | 0.857 | +0.06 |
| `layer-cam` | 0.209 | 0.983 | 0.42 | 0.090 | 0.298 | +0.30 |
| `kernel-shap` | 0.238 | 0.978 | 0.08 | 0.113 | 0.116 | +0.10 |
| `occlusion` | 0.261 | 0.960 | 0.00 | 0.053 | 0.073 | +0.88 |
| `lime` | 0.269 | 0.982 | 0.08 | 0.121 | 0.114 | +0.49 |

These are the properties of a *toy* problem, and three columns are artefacts of it:

- **Insertion barely separates the methods** (0.96–0.99) — the synthetic classifier is
  effectively certain, so the probability recovers from almost any few pixels.
- **The CAM family localises poorly** (mass ≈ 0.06) — four pooling stages reduce a
  64×64 input to a 4×4 feature map, so every CAM is upsampled from sixteen values while
  the defect is four pixels across. A resolution limit, not a flaw in the method.
- **The Adebayo signature is directional, not absolute, at this scale.** Guided
  backprop decays 0.89 → 0.29, more clinging than `saliency` (0.26 → 0.09) but well
  short of the near-1.0-throughout result the paper obtains with Inception and VGG on
  ImageNet. A four-block CNN has little to destroy.

## Reproducing on KolektorSDD2

[KolektorSDD2](https://www.vicos.si/resources/kolektorsdd2/): 3,335 images of
production surfaces, 356 defective, each with a pixel-level mask — masks rather than
boxes, so localisation is scored against the defect's actual shape.

```bash
python -m classify_justify download --root data
python -m classify_justify train --dataset kolektor --data-root data/KolektorSDD2 --output runs/kolektor
python -m classify_justify evaluate --checkpoint runs/kolektor/model.pt --data-root data/KolektorSDD2 --sanity
```

Budget a GPU for training: at 256×640 it costs roughly **105 s/epoch on an Apple
M-series GPU** and about **9 min/epoch on CPU**, for up to 30 epochs. The dataset is
never vendored here — it is CC BY-NC-SA 4.0, non-commercial and share-alike, and cites:

> Božič, Tabernik, Skočaj (2021). *Mixed supervision for surface-defect detection:
> from weakly to fully supervised learning.* Computers in Industry.

The loader skips two stray files the published archive contains, `10301 (copy).png` and
`10301_GT (copy).png`. Both are traps: the second does not end in `_GT`, so the obvious
filter loads a **mask** as an input image, and the first is a duplicate that can land in
train while its original lands in validation.

## Design notes

**The classifier is built for explanation.** Its head is a global average pool plus one
linear layer, because CAM weighting is only defensible when the logit is a linear
function of pooled channels. Its ReLUs are deliberately not in-place, because Guided
Backprop and DeepLIFT rewrite ReLU gradients through backward hooks and an in-place
ReLU silently corrupts them. It is built from config alone, so a checkpoint loads on a
machine with no dataset.

**Splitting.** Augmentation is a per-sample transform applied at load time, so an
augmented copy can never outlive the split that separated it. Normalisation statistics
are fitted on the training split alone and stored in the checkpoint. Splits are
stratified, because at 10.6% defective a uniform split swings several points between
seeds.

**Early stopping ranks by `(average precision, F1)`.** Accuracy is useless at this
imbalance — predicting "clean" for everything scores 0.894 — and AP alone saturates at
1.000 the moment the ranking is perfect, locking in the first epoch to get there even
when its threshold is still badly calibrated.

## Development

```bash
python -m pytest        # 98 tests, ~30 s
python -m ruff check .
```

Tests train a small model on synthetic data and assert it reached accuracy > 0.9. That
guard matters: a collapsed model still produces heatmaps of the right shape, so every
contract test would pass while measuring nothing.

## Licence

MIT, see [LICENSE](LICENSE). KolektorSDD2 is licensed separately, as noted above.
