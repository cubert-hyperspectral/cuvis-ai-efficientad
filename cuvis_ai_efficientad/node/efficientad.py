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

``backend="tensorrt"`` runs the model as a TensorRT engine built from these weights on the machine
(:mod:`cuvis_ai_efficientad.trt_engine`); the resize, the map upsampling and the score stay in
torch.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel, EfficientAdModelSize
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor, nn
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as TF

_SIZES = {"small": EfficientAdModelSize.S, "medium": EfficientAdModelSize.M}
_BACKENDS = ("torch", "tensorrt")
_AUTOCAST_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


@contextmanager
def _tf32_matmul(enabled: bool) -> Iterator[None]:
    """Allow TF32 tensor-core matmuls inside the block and restore the process setting after it."""
    if not enabled:
        yield
        return
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _drop_engine(module: nn.Module, _incompatible_keys: Any) -> None:
    """``load_state_dict`` post hook: new weights need their own TensorRT engine."""
    module._engine = None


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
        autocast_dtype: str | None = None,
        tf32: bool = False,
        backend: str = "torch",
        engine_dir: str | None = None,
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
        autocast_dtype : ``None`` (float32, default), ``"float16"`` or ``"bfloat16"``: run the
            EfficientAD model under CUDA autocast (tensor cores). Applied on CUDA inputs only; the
            outputs stay float32. The numerics change slightly, so re-validate and re-calibrate a
            pipeline first.
        tf32 : allow TF32 tensor-core matmuls in the float32 forward (float32 storage and
            accumulation; the convolutions use TF32 by PyTorch's default already), set around
            the forward and restored afterwards. CUDA inputs only; ignored when
            ``autocast_dtype`` is set.
        backend : ``"torch"`` (default) runs anomalib's model. ``"tensorrt"`` runs a TensorRT
            engine compiled from the same model and weights instead (CUDA only). Its precision
            follows the options above: ``autocast_dtype="float16"`` -> fp16 engine, ``tf32`` ->
            TF32 engine, neither -> IEEE float32 engine (TF32 off; PyTorch's float32 node runs its
            convolutions in TF32 by default). Engines are specific to one GPU, TensorRT version and
            set of weights; build them once per machine before the first run:
            ``python -m cuvis_ai_efficientad.trt_engine build-pipeline <pipeline.yaml>``. Needs the
            ``tensorrt`` package (the plugin's ``tensorrt`` extra); ``bfloat16`` is not offered.
        engine_dir : where the engines are kept (default:
            :func:`cuvis_ai_efficientad.trt_engine.default_engine_dir`). Engine file names carry the
            precision, a fingerprint of the weights, the input size, the GPU and the TensorRT
            version, so one directory can serve several machines and pipelines.
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
        if autocast_dtype is not None and autocast_dtype not in _AUTOCAST_DTYPES:
            raise ValueError(
                f"{name}: autocast_dtype must be None or one of {sorted(_AUTOCAST_DTYPES)}, "
                f"got {autocast_dtype!r}"
            )
        if backend not in _BACKENDS:
            raise ValueError(f"{name}: backend must be one of {_BACKENDS}, got {backend!r}")
        if backend == "tensorrt" and _AUTOCAST_DTYPES.get(autocast_dtype) is torch.bfloat16:
            raise ValueError(f"{name}: backend='tensorrt' has no bfloat16 engine; use float16")
        if engine_dir is not None and not str(engine_dir).strip():
            raise ValueError(f"{name}: engine_dir must be None or a non-empty path")
        self.image_size = size
        self.teacher_out_channels = int(teacher_out_channels)
        self.model_size = str(model_size)
        self.padding = bool(padding)
        self.pad_maps = bool(pad_maps)
        self.topk_frac = float(topk_frac)
        self.autocast_dtype = autocast_dtype
        self.tf32 = bool(tf32)
        self.backend = str(backend)
        self.engine_dir = str(engine_dir) if engine_dir is not None else None
        super().__init__(
            image_size=self.image_size,
            teacher_out_channels=self.teacher_out_channels,
            model_size=self.model_size,
            padding=self.padding,
            pad_maps=self.pad_maps,
            topk_frac=self.topk_frac,
            autocast_dtype=self.autocast_dtype,
            tf32=self.tf32,
            backend=self.backend,
            engine_dir=self.engine_dir,
            **kwargs,
        )
        self._amp_dtype = _AUTOCAST_DTYPES.get(autocast_dtype) if autocast_dtype else None
        self.model = EfficientAdModel(
            teacher_out_channels=self.teacher_out_channels,
            model_size=_SIZES[self.model_size],
            padding=self.padding,
            pad_maps=self.pad_maps,
        )
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        # The TensorRT engine is loaded at the first forward, once the pipeline has loaded the
        # weights, and dropped whenever new weights are loaded.
        self._engine: Any = None
        self.register_load_state_dict_post_hook(_drop_engine)

    # ------------------------------------------------------------------ TensorRT
    @property
    def engine_precision(self) -> str:
        """The TensorRT engine precision of this node's options: ``fp16``, ``tf32`` or ``fp32``."""
        if self._amp_dtype is torch.float16:
            return "fp16"
        return "tf32" if self.tf32 else "fp32"

    def engine_path(self, device: torch.device | str | int | None = None) -> str:
        """This machine's engine file for the node's current weights, precision and input size."""
        from cuvis_ai_efficientad import trt_engine

        extra = (
            f"EfficientAdModel|{self.model_size}|{self.teacher_out_channels}|{self.padding}|"
            f"{self.pad_maps}|{self.image_size}"
        )
        fingerprint = trt_engine.weights_fingerprint(self.model, extra)
        return os.path.join(
            self.engine_dir or trt_engine.default_engine_dir(),
            trt_engine.engine_file_name(
                self.engine_precision, fingerprint, self.image_size, device
            ),
        )

    def build_engine(self, force: bool = False) -> tuple[str, bool]:
        """Build this node's engine here (CUDA); returns its path and whether it was built now."""
        from cuvis_ai_efficientad import trt_engine

        device = next(self.model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(
                f"{self.name}: TensorRT engines are built on a CUDA device, the model is on "
                f"{device}"
            )
        path = self.engine_path(device)
        if os.path.exists(path) and not force:
            return path, False
        fingerprint = os.path.basename(path).split("_")[1]
        trt_engine.build_engine(
            self.model, self.image_size, self.engine_precision, path, fingerprint
        )
        self._engine = None
        return path, True

    def _load_engine(self, device: torch.device) -> Any:
        from cuvis_ai_efficientad import trt_engine

        if device.type != "cuda":
            raise RuntimeError(f"{self.name}: backend='tensorrt' needs a CUDA device, got {device}")
        path = self.engine_path(device)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{self.name}: no TensorRT engine for these weights on this machine at {path}. "
                "Build it once with: python -m cuvis_ai_efficientad.trt_engine build-pipeline "
                "<this pipeline's yaml>"
            )
        engine = trt_engine.TensorRTEngine(path, device)
        expected = (1, 3, *self.image_size)
        if tuple(engine.input_shape) != expected:
            raise RuntimeError(
                f"{self.name}: TensorRT engine {path} takes input {tuple(engine.input_shape)}, "
                f"the model needs {expected}; rebuild it."
            )
        return engine

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

        The file is unpickled (``weights_only=False``: a Lightning checkpoint needs it), and
        unpickling can execute code. Load only checkpoints you trust, e.g. your own training runs.
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
        if self.backend == "tensorrt":
            if self._engine is None:
                self._engine = self._load_engine(x.device)
            if b == 1:
                amap = self._engine(x)["anomaly_map"]
            else:  # the engine takes one frame; its output buffer is reused, so copy each result
                amap = torch.cat(
                    [self._engine(x[i : i + 1])["anomaly_map"].clone() for i in range(b)]
                )
        else:
            amp = self._amp_dtype is not None and x.is_cuda
            tf32 = self.tf32 and x.is_cuda and not amp
            with (
                torch.no_grad(),
                torch.autocast(device_type="cuda", dtype=self._amp_dtype, enabled=amp),
                _tf32_matmul(tf32),
            ):
                amap = self.model(x).anomaly_map
        amap = F.interpolate(amap.float(), size=(h, w), mode="bilinear", align_corners=False)
        scores = amap.permute(0, 2, 3, 1).contiguous()
        k = max(1, int(self.topk_frac * h * w))
        anomaly_score = torch.topk(scores.reshape(b, -1), k, dim=1).values.mean(dim=1)
        return {"scores": scores, "anomaly_score": anomaly_score}
