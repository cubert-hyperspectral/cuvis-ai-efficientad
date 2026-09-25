"""EfficientAD anomaly maps of an RGB frame, with anomalib 2.1.0's torch model used unchanged.

EfficientAD (Batzner et al., WACV 2024) scores an image by two discrepancies: a student network
against a frozen, pre-trained patch-description teacher, and the student against an autoencoder.
anomalib's ``EfficientAdModel`` holds all of it: teacher, student, autoencoder, the teacher's
channel mean / std and the map-normalisation quantiles. All of it is state, so a fitted pipeline
``.pt`` carries the complete detector.

The node reproduces anomalib's inference path exactly: the frame is resized to ``image_size`` with
the stock pre-processor's antialiased bilinear resize, the model runs in eval mode, and its
``anomaly_map`` (0.5 x the quantile-normalised student-teacher map + 0.5 x the student-autoencoder
map) is resized back to the frame bilinearly.

Training stays in anomalib (``EfficientAd`` + ``Engine``: ImageNet penalty, hard-feature loss,
quantiles on normal validation images); ``load_anomalib_checkpoint`` imports the trained weights
before the pipeline is saved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel, EfficientAdModelSize
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as TF

_SIZES = {"small": EfficientAdModelSize.S, "medium": EfficientAdModelSize.M}


class EfficientAdDetector(Node):
    """EfficientAD student-teacher + autoencoder anomaly map and image score of an RGB frame."""

    _category = NodeCategory.MODEL
    _tags = frozenset({NodeTag.ANOMALY, NodeTag.IMAGE, NodeTag.TORCH})

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB frame [B, H, W, 3] in [0, 1], e.g. a stretched false-RGB projection; "
            "resized to `image_size` internally (the model normalises with ImageNet statistics).",
        ),
    }
    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Anomaly map [B, H, W, 1]: anomalib's anomaly_map (mean of the quantile-"
            "normalised student-teacher and student-autoencoder maps), resized to the input.",
        ),
        "anomaly_score": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Image-level score [B]: mean of the top topk_frac pixels of `scores`.",
        ),
    }

    def __init__(
        self,
        image_size: int | list[int] | tuple[int, int] = 256,
        teacher_out_channels: int = 384,
        model_size: str = "small",
        padding: bool = False,
        pad_maps: bool = True,
        topk_frac: float = 0.001,
        **kwargs: Any,
    ) -> None:
        """Create an EfficientAD detector with untrained weights (restore or import them).

        Parameters
        ----------
        image_size : model input size, an int (square) or [height, width]; the frame is resized
            to it. anomalib's default is 256; the weights must have been trained at this size.
        teacher_out_channels : teacher feature channels (anomalib default 384).
        model_size : "small" (PDN-S) or "medium" (PDN-M).
        padding : zero padding in the convolutions (anomalib default False).
        pad_maps : pad the maps by 4 px before resizing when ``padding`` is False (anomalib
            default True).
        topk_frac : fraction of output pixels averaged into ``anomaly_score``.
        """
        name = type(self).__name__
        size = (
            [int(image_size)] * 2 if isinstance(image_size, int) else [int(s) for s in image_size]
        )
        if len(size) != 2 or min(size) < 64:
            raise ValueError(
                f"{name}: image_size must be an int or [h, w] >= 64, got {image_size!r}"
            )
        if model_size not in _SIZES:
            raise ValueError(
                f"{name}: model_size must be one of {sorted(_SIZES)}, got {model_size!r}"
            )
        if isinstance(teacher_out_channels, bool) or int(teacher_out_channels) < 1:
            raise ValueError(
                f"{name}: teacher_out_channels must be >= 1, got {teacher_out_channels!r}"
            )
        if not 0.0 < float(topk_frac) <= 1.0:
            raise ValueError(f"{name}: topk_frac must be in (0, 1], got {topk_frac}")
        self.image_size = size
        self.teacher_out_channels = int(teacher_out_channels)
        self.model_size = str(model_size)
        self.padding = bool(padding)
        self.pad_maps = bool(pad_maps)
        self.topk_frac = float(topk_frac)
        super().__init__(
            image_size=self.image_size,
            teacher_out_channels=self.teacher_out_channels,
            model_size=self.model_size,
            padding=self.padding,
            pad_maps=self.pad_maps,
            topk_frac=self.topk_frac,
            **kwargs,
        )
        self.model = EfficientAdModel(
            teacher_out_channels=self.teacher_out_channels,
            model_size=_SIZES[self.model_size],
            padding=self.padding,
            pad_maps=self.pad_maps,
        )
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> EfficientAdDetector:
        """Follow the pipeline's mode flag but keep the frozen EfficientAD model in eval mode.

        anomalib's model returns training losses instead of maps in train mode.
        """
        super().train(mode)
        self.model.eval()
        return self

    def load_anomalib_checkpoint(self, path: str | Path) -> None:
        """Import the weights of an anomalib ``EfficientAd`` Lightning checkpoint (``model.*``).

        All tensors load strictly: teacher, student, autoencoder, the teacher mean / std and the
        map quantiles. A checkpoint without quantiles (never validated on normal images) raises,
        because the model would then silently average un-normalised maps. A Lightning checkpoint
        also pickles the training module's classes, so it loads only where they are importable; a
        file holding just ``{"state_dict": ...}`` loads anywhere.
        """
        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(
                f"{path}: the checkpoint pickles a training class that cannot be imported here "
                f"({e}). Load it where the training code is importable, or keep only its "
                "tensors first: torch.save({'state_dict': ckpt['state_dict']}, 'weights.ckpt')"
            ) from e
        state = ckpt.get("state_dict", ckpt)
        model_state = {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
        if not model_state:
            raise ValueError(
                f"{path}: no 'model.*' tensors; not an anomalib EfficientAd checkpoint"
            )
        self.model.load_state_dict(model_state, strict=True)
        if not EfficientAdModel.is_set(self.model.quantiles):
            raise ValueError(
                f"{path}: the map-normalisation quantiles are all zero (not validated)"
            )
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, rgb_image: Tensor, **_: Any) -> dict[str, Tensor]:
        """Anomaly map [B, H, W, 1] + top-k image score [B] of an RGB frame [B, H, W, 3]."""
        b, h, w, _ = rgb_image.shape
        x = TF.resize(
            rgb_image.permute(0, 3, 1, 2),
            self.image_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        with torch.no_grad():
            amap = self.model(x).anomaly_map
        amap = F.interpolate(amap.float(), size=(h, w), mode="bilinear", align_corners=False)
        scores = amap.permute(0, 2, 3, 1).contiguous()
        k = max(1, int(self.topk_frac * h * w))
        anomaly_score = torch.topk(scores.reshape(b, -1), k, dim=1).values.mean(dim=1)
        return {"scores": scores, "anomaly_score": anomaly_score}
