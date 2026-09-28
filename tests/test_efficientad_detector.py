"""EfficientAdDetector: anomalib inference-path parity, port contract, frozen state, checkpoint
import, hparam validation."""

from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn.functional as F
from anomalib.models import EfficientAd
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel

from cuvis_ai_efficientad.node.efficientad import EfficientAdDetector
from tests.conftest import SIZE, set_trained_state

pytestmark = pytest.mark.unit


def _anomalib_map(model: EfficientAdModel, rgb: torch.Tensor) -> torch.Tensor:
    """anomalib's own inference path: stock pre-processor transform -> eval model -> anomaly_map,
    resized back to the frame bilinearly (anomalib predict + resize to the source image)."""
    transform = EfficientAd.configure_pre_processor(image_size=(SIZE, SIZE)).transform
    x = transform(rgb.permute(0, 3, 1, 2))
    with torch.no_grad():
        amap = model.eval()(x).anomaly_map
    return F.interpolate(amap, size=rgb.shape[1:3], mode="bilinear", align_corners=False).permute(
        0, 2, 3, 1
    )


def test_matches_anomalib_inference_path(detector, rgb):
    out = detector(rgb_image=rgb)
    assert torch.equal(out["scores"], _anomalib_map(detector.model, rgb))


def test_image_score_is_topk_mean(detector, rgb):
    out = detector(rgb_image=rgb)
    flat = out["scores"].reshape(1, -1)
    k = max(1, int(0.001 * flat.shape[1]))
    assert torch.allclose(out["anomaly_score"], torch.topk(flat, k).values.mean(dim=1))


def test_quantile_normalisation_is_applied(detector, rgb):
    """The map is 0.5 x normalised st + 0.5 x normalised st-ae: new quantiles change it."""
    a = detector(rgb_image=rgb)["scores"]
    with torch.no_grad():
        detector.model.quantiles["qb_st"].mul_(2.0)
    assert not torch.equal(a, detector(rgb_image=rgb)["scores"])


def test_port_contract(detector):
    x = torch.rand(2, 280, 260, 3)
    out = detector(rgb_image=x)
    assert set(out) == set(EfficientAdDetector.OUTPUT_SPECS)
    assert out["scores"].shape == (2, 280, 260, 1) and out["scores"].dtype == torch.float32
    assert out["anomaly_score"].shape == (2,) and out["anomaly_score"].dtype == torch.float32


def test_weights_frozen_and_eval_pinned(detector, rgb):
    before = detector(rgb_image=rgb)["scores"]
    detector.train()  # pipeline train mode must not flip anomalib into its loss branch
    assert detector.training and not detector.model.training
    assert not any(p.requires_grad for p in detector.model.parameters())
    assert torch.equal(detector(rgb_image=rgb)["scores"], before)


def test_load_anomalib_checkpoint(tmp_path, rgb):
    src = EfficientAdModel()
    set_trained_state(src, seed=7)
    ckpt = {
        "state_dict": {f"model.{k}": v for k, v in src.state_dict().items()},
        "hyper_parameters": {},
    }
    path = tmp_path / "effad.ckpt"
    torch.save(ckpt, path)
    node = EfficientAdDetector(image_size=SIZE)
    node.load_anomalib_checkpoint(path)
    for k, v in src.state_dict().items():
        assert torch.equal(node.model.state_dict()[k], v), k
    assert not any(p.requires_grad for p in node.model.parameters())
    assert torch.equal(node(rgb_image=rgb)["scores"], _anomalib_map(src, rgb))


def test_load_checkpoint_rejects_foreign_and_unvalidated(tmp_path):
    node = EfficientAdDetector(image_size=SIZE)
    foreign = tmp_path / "foreign.ckpt"
    torch.save({"state_dict": {"backbone.weight": torch.zeros(1)}}, foreign)
    with pytest.raises(ValueError, match="no 'model"):
        node.load_anomalib_checkpoint(foreign)
    raw = EfficientAdModel()  # all-zero quantiles: never validated on normal images
    unvalidated = tmp_path / "unvalidated.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in raw.state_dict().items()}}, unvalidated)
    with pytest.raises(ValueError, match="quantiles"):
        node.load_anomalib_checkpoint(unvalidated)


class _TrainingOnlyClass:
    """Stands in for a training subclass that is pickled into a Lightning checkpoint."""


def test_load_checkpoint_with_unimportable_training_class(tmp_path, monkeypatch):
    src = EfficientAdModel()
    set_trained_state(src, seed=3)
    module = types.ModuleType("_effad_training_code")
    module._TrainingOnlyClass = _TrainingOnlyClass
    monkeypatch.setattr(_TrainingOnlyClass, "__module__", module.__name__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    path = tmp_path / "subclassed.ckpt"
    torch.save(
        {
            "state_dict": {f"model.{k}": v for k, v in src.state_dict().items()},
            "hyper_parameters": {"module": _TrainingOnlyClass()},
        },
        path,
    )
    monkeypatch.delitem(sys.modules, module.__name__)  # the training code is not installed here
    node = EfficientAdDetector(image_size=SIZE)
    with pytest.raises(ValueError, match="cannot be imported here"):
        node.load_anomalib_checkpoint(path)
    tensors_only = tmp_path / "weights.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in src.state_dict().items()}}, tensors_only)
    node.load_anomalib_checkpoint(tensors_only)
    for k, v in src.state_dict().items():
        assert torch.equal(node.model.state_dict()[k], v), k


@pytest.mark.parametrize(
    "kw",
    [
        {"image_size": 32},
        {"image_size": [256]},
        {"model_size": "large"},
        {"topk_frac": 0.0},
        {"teacher_out_channels": 0},
    ],
)
def test_invalid_hparams_raise(kw):
    with pytest.raises(ValueError):
        EfficientAdDetector(**kw)


def test_hparams_recorded():
    node = EfficientAdDetector(image_size=[256, 320], model_size="medium", topk_frac=0.002)
    assert node.hparams["image_size"] == [256, 320]
    assert node.hparams["model_size"] == "medium"
    assert node.hparams["topk_frac"] == 0.002
