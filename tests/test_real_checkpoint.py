"""Real-checkpoint parity (slow): a trained anomalib EfficientAd checkpoint loads into the node, and
the node's maps equal anomalib's own inference path with the same weights.

Needs the checkpoint, so it is skipped unless ``EFFICIENTAD_CKPT`` names one: a Lightning checkpoint
where its training classes import, or a tensors-only ``{"state_dict": ...}`` file. Set
``EFFICIENTAD_IMAGE_SIZE`` (default 256) and ``EFFICIENTAD_MODEL_SIZE`` (default ``small``) to what
the checkpoint was trained with. Run with ``pytest -m slow``.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F
from anomalib.models import EfficientAd
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel, EfficientAdModelSize

from cuvis_ai_efficientad.node.efficientad import EfficientAdDetector

pytestmark = pytest.mark.slow

CKPT = os.environ.get("EFFICIENTAD_CKPT")
SIZE = int(os.environ.get("EFFICIENTAD_IMAGE_SIZE", "256"))
MODEL_SIZE = os.environ.get("EFFICIENTAD_MODEL_SIZE", "small")


@pytest.mark.skipif(not CKPT, reason="set EFFICIENTAD_CKPT to a trained EfficientAd checkpoint")
def test_trained_checkpoint_matches_anomalib_inference_path():
    node = EfficientAdDetector(image_size=SIZE, model_size=MODEL_SIZE, name="effad")
    node.load_anomalib_checkpoint(CKPT)

    ref = EfficientAdModel(
        model_size=EfficientAdModelSize.S if MODEL_SIZE == "small" else EfficientAdModelSize.M
    )
    state = torch.load(CKPT, map_location="cpu", weights_only=False)  # a trusted local file
    state = state.get("state_dict", state)
    ref.load_state_dict({k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")})

    rgb = torch.rand(2, 400, 432, 3, generator=torch.Generator().manual_seed(0))
    transform = EfficientAd.configure_pre_processor(image_size=(SIZE, SIZE)).transform
    with torch.no_grad():
        amap = ref.eval()(transform(rgb.permute(0, 3, 1, 2))).anomaly_map
    expected = F.interpolate(amap, size=rgb.shape[1:3], mode="bilinear", align_corners=False)

    out = node(rgb_image=rgb)
    assert torch.equal(out["scores"], expected.permute(0, 2, 3, 1))
    assert torch.isfinite(out["anomaly_score"]).all()
