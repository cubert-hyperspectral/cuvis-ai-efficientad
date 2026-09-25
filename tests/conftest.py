"""Shared fixtures: a small EfficientAD detector with random weights but a 'trained-looking' state.

A trained model has non-zero teacher channel mean / std and map quantiles; with zeros anomalib
skips both normalisations, so the fixtures set seeded non-zero values to exercise the full inference
path.
"""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_efficientad.node.efficientad import EfficientAdDetector

SIZE = 256  # anomalib's default EfficientAD input size (the autoencoder decoder is built for it)


def set_trained_state(model, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():  # a random but well-scaled network
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
        c = model.teacher_out_channels
        model.mean_std["mean"].copy_(torch.randn((1, c, 1, 1), generator=g) * 0.1)
        model.mean_std["std"].copy_(torch.rand((1, c, 1, 1), generator=g) + 0.5)
        for key, val in (("qa_st", 0.01), ("qb_st", 0.2), ("qa_ae", 0.02), ("qb_ae", 0.3)):
            model.quantiles[key].copy_(torch.tensor(val))


@pytest.fixture
def detector() -> EfficientAdDetector:
    node = EfficientAdDetector(image_size=SIZE, name="effad")
    set_trained_state(node.model)
    return node


@pytest.fixture
def rgb() -> torch.Tensor:
    return torch.rand(1, 300, 320, 3, generator=torch.Generator().manual_seed(1))
