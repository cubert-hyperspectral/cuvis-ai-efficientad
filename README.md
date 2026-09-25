# cuvis-ai-efficientad

EfficientAD for [cuvis-ai](https://docs.cuvis.ai/latest/): the student-teacher + autoencoder anomaly
detector of Batzner et al. (WACV 2024), with the torch model of
[anomalib](https://github.com/open-edge-platform/anomalib) 2.1.0 used unchanged.

EfficientAD scores an image by two discrepancies:
- a student network against a frozen, pre-trained patch-description teacher;
- the student against an autoencoder.

The two maps are normalised by quantiles taken on normal validation images and averaged. The node
reproduces anomalib's inference path exactly:
1. the frame is resized with the stock pre-processor's antialiased bilinear resize;
2. the model runs in eval mode;
3. `anomaly_map` is resized back to the frame bilinearly.

Training stays in anomalib: `EfficientAd` + `Engine`, with its ImageNet penalty, hard-feature loss
and quantiles. The trained weights are imported with `load_anomalib_checkpoint` before the pipeline
is saved; the pipeline `.pt` then carries the whole detector.

## Nodes

### `cuvis_ai_efficientad.node.efficientad.EfficientAdDetector`

| Port | Direction | Shape / dtype | Notes |
|---|---|---|---|
| `rgb_image` | in | `[B, H, W, 3]` float32 in `[0, 1]` | e.g. a false-RGB projection after `JointPercentileStretch` (cuvis-ai-steervit); resized to `image_size` internally |
| `scores` | out | `[B, H, W, 1]` float32 | anomalib's `anomaly_map`, resized to the input |
| `anomaly_score` | out | `[B]` float32 | mean of the top `topk_frac` pixels of `scores` |

| hparam | default | meaning |
|---|---|---|
| `image_size` | 256 | model input size, int or `[h, w]`; must match the training size |
| `teacher_out_channels` | 384 | teacher feature channels |
| `model_size` | `small` | `small` (PDN-S) or `medium` (PDN-M) |
| `padding` | false | zero padding in the convolutions |
| `pad_maps` | true | pad the maps by 4 px before resizing when `padding` is false |
| `topk_frac` | 0.001 | pixel fraction averaged into `anomaly_score` |

Weights are frozen, and the model stays in eval mode under `pipeline.train()`, because anomalib
returns training losses in train mode. There is no Phase 1 and no `TRAINABLE_BUFFERS`.

## Build a pipeline from an anomalib checkpoint

```python
from cuvis_ai_efficientad import EfficientAdDetector

det = EfficientAdDetector(image_size=512, name="efficientad")
det.load_anomalib_checkpoint("model_final.ckpt")  # anomalib EfficientAd Lightning checkpoint
# ... wire det into a CuvisPipeline, then pipeline.save_to_file("efficientad.yaml")
```

The checkpoint must come from a model whose quantiles were set on normal images (anomalib does
this at the start of validation). An all-zero-quantile checkpoint is rejected.

A Lightning checkpoint also pickles the classes of the training module. If you trained a subclass
of `EfficientAd`, load the checkpoint where that class is importable, or keep only its tensors
first, `torch.save({"state_dict": ckpt["state_dict"]}, "weights.ckpt")`, which loads anywhere.
Deployment is unaffected: the saved pipeline `.pt` holds plain tensors.

## Install

Local development: a bare manifest pointing at the checkout. The path is relative to the
manifest.

```yaml
name: efficientad
path: "../cuvis-ai-efficientad"
package_name: cuvis-ai-efficientad
capabilities:
  - class_name: cuvis_ai_efficientad.node.efficientad.EfficientAdDetector
```

Frozen consumers use the git-tag form, which the loader clones at the tag and installs:

```yaml
name: efficientad
repo: "https://github.com/cubert-hyperspectral/cuvis-ai-efficientad.git"
tag: "v0.1.0"
package_name: cuvis-ai-efficientad
capabilities:
  - class_name: cuvis_ai_efficientad.node.efficientad.EfficientAdDetector
```

A pipeline that lists `efficientad` in its `plugins:` pulls anomalib 2.1.0 and its dependencies
into the composed child environment on first use.

## Development

```bash
uv sync --extra dev
uv run --extra dev pytest tests -m "not slow"
uv run --extra dev ruff format --check cuvis_ai_efficientad tests
uv run --extra dev ruff check cuvis_ai_efficientad tests
```

## References

- K. Batzner, L. Heckler, R. König, "EfficientAD: Accurate Visual Anomaly Detection at
  Millisecond-Level Latencies", WACV 2024.
- anomalib 2.1.0, `anomalib.models.image.efficient_ad` (Apache-2.0).

## License

Apache-2.0 (see `LICENSE`). anomalib is Apache-2.0.
