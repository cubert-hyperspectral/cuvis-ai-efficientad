"""Reload smoke: an EfficientAdDetector inside a CuvisPipeline survives save_to_file ->
load_pipeline (yaml + .pt) and reproduces its outputs: every tensor of the detector (networks,
mean / std, quantiles) lives in the .pt."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_efficientad.node.efficientad import EfficientAdDetector
from tests.conftest import SIZE, set_trained_state

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[1]
H, W = 270, 300


class _ConstantRGBSource(Node):
    """Module-scope test source: a deterministic random RGB frame in [0, 1]."""

    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"rgb": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 3))}

    def __init__(self, seed: int = 0, **kwargs) -> None:
        super().__init__(seed=seed, **kwargs)
        self.seed = int(seed)

    def forward(self, **_) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed)
        return {"rgb": torch.rand(1, H, W, 3, generator=g)}


def test_detector_pipeline_reloads(tmp_path):
    src = _ConstantRGBSource(seed=3, name="src")
    det = EfficientAdDetector(image_size=SIZE, name="effad")
    set_trained_state(det.model, seed=11)
    pipe = CuvisPipeline("efficientad_reload_smoke")
    pipe.connect(src.outputs.rgb, det.inputs.rgb_image)

    ctx = Context(stage=ExecutionStage.INFERENCE)
    before = pipe.forward(batch={}, context=ctx)
    assert before[("effad", "scores")].shape == (1, H, W, 1)

    yaml_path = tmp_path / "effad.yaml"
    pipe.save_to_file(str(yaml_path))
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    cfg["plugins"] = ["efficientad"]
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    registry = NodeRegistry()
    registry.register_plugin(str(REPO / "plugins.yaml"))
    restored = CuvisPipeline.load_pipeline(
        str(yaml_path),
        weights_path=str(yaml_path.with_suffix(".pt")),
        device="cpu",
        node_registry=registry,
    )
    node = next(n for n in restored.nodes if not isinstance(n, str) and n.name == "effad")
    assert isinstance(node, EfficientAdDetector) and node.hparams["image_size"] == [SIZE, SIZE]
    after = restored.forward(batch={}, context=ctx)
    for port in ("scores", "anomaly_score"):
        assert torch.equal(after[("effad", port)], before[("effad", port)]), port
