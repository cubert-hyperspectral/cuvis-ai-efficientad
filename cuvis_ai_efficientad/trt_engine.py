"""TensorRT engines for EfficientAdDetector, built per machine from the node's fitted model.

The node's anomalib ``EfficientAdModel`` (teacher, student, autoencoder, the teacher's channel
mean / std and the map quantiles) is exported to ONNX at the node's ``image_size``: a static
``1x3xHxW`` input ``input`` and the output ``anomaly_map``. The installed TensorRT compiles it into
an engine for THIS GPU and TensorRT version, which runs on torch CUDA tensors.

The weights are fitted state (a pipeline ``.pt`` carries them), so an engine file name carries a
fingerprint of them next to the precision, the input size, the GPU and the TensorRT version: an
engine never runs with weights it was not built from, and the engines of several machines and
pipelines can share one directory. A JSON file next to each engine records how it was built.

Precisions (the node derives its own from ``autocast_dtype`` / ``tf32``):

- ``fp32``: IEEE float32, TF32 disabled. Within ~5e-7 of the float32 model computed with IEEE
  convolutions; PyTorch's default runs the convolutions in TF32 (``torch.backends.cudnn`` TF32 is on
  by default), so against the plain float32 node the difference is the TF32 one (~3e-4 of the map).
  Slow on GPUs that rely on TF32 tensor cores (Jetson Thor).
- ``tf32``: TF32 tensor cores allowed (TensorRT's default float build), the counterpart of PyTorch's
  default.
- ``fp16``: TensorRT's FP16 builder flag (mixed precision: TensorRT keeps a layer in float32 where
  that is faster or needed). TensorRT 11 dropped the flag, so fp16 engines need TensorRT 10.

Optional dependencies, not installed with the plugin: the TensorRT Python package matching torch's
CUDA (``tensorrt-cu12`` / ``tensorrt-cu13``, version 10) and, to build engines, ``onnx`` (the
``tensorrt`` extra). Build the engines of a pipeline once per machine, before its first run::

    python -m cuvis_ai_efficientad.trt_engine build-pipeline pipeline.yaml [more.yaml ...]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn

PRECISIONS = ("fp32", "tf32", "fp16")
ENGINE_DIR_ENV = "CUVIS_AI_TRT_ENGINE_DIR"
_LOGGER: Any = None


def _tensorrt() -> Any:
    try:
        import tensorrt
    except ImportError as exc:
        raise ImportError(
            "EfficientAdDetector backend='tensorrt' needs the TensorRT Python package "
            "(version 10) matching torch's CUDA: pip install tensorrt-cu12 (CUDA 12 torch) or "
            "tensorrt-cu13 (CUDA 13 torch), or the plugin's 'tensorrt' extra; building engines "
            "also needs onnx."
        ) from exc
    if int(str(tensorrt.__version__).split(".")[0]) < 10:
        raise ImportError(f"TensorRT >= 10 is required, found {tensorrt.__version__}.")
    return tensorrt


def _logger(trt: Any) -> Any:
    """One TensorRT logger per process (TensorRT keeps the first and warns about others)."""
    global _LOGGER
    if _LOGGER is None or _LOGGER[0] is not trt:
        _LOGGER = (trt, trt.Logger(trt.Logger.WARNING))
    return _LOGGER[1]


def gpu_tag(device: torch.device | str | int | None = None) -> str:
    """``<device name>-sm<capability>``, filesystem-safe (e.g. ``NVIDIA-Thor-sm110``)."""
    name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)
    return f"{re.sub(r'[^A-Za-z0-9]+', '-', name).strip('-')}-sm{major}{minor}"


def weights_fingerprint(module: nn.Module, extra: str = "") -> str:
    """First 16 hex digits of the MD5 of a module's state (names, dtypes, shapes, values).

    ``extra`` carries what else shapes the exported graph (e.g. the architecture options and the
    input size). An identity check of fitted weights, not a security measure.
    """
    digest = hashlib.md5(extra.encode("utf-8"))  # noqa: S324
    for name, tensor in sorted(module.state_dict().items()):
        t = tensor.detach().to("cpu").contiguous()
        digest.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return digest.hexdigest()[:16]


def engine_file_name(
    precision: str,
    fingerprint: str,
    image_size: tuple[int, int] | list[int],
    device: torch.device | str | int | None = None,
) -> str:
    """``<precision>_<fingerprint>_<H>x<W>_<gpu tag>_trt<TensorRT version>.engine``."""
    h, w = (int(s) for s in image_size)
    return (
        f"{precision}_{fingerprint}_{h}x{w}_{gpu_tag(device)}_trt{_tensorrt().__version__}.engine"
    )


def default_engine_dir() -> str:
    """``$CUVIS_AI_TRT_ENGINE_DIR/efficientad`` or ``~/.cache/cuvis-ai/tensorrt/efficientad``."""
    root = os.environ.get(ENGINE_DIR_ENV) or os.path.join(
        os.path.expanduser("~"), ".cache", "cuvis-ai", "tensorrt"
    )
    return os.path.join(root, "efficientad")


@contextmanager
def sync_free(model: nn.Module) -> Iterator[None]:
    """Keep anomalib's EfficientAD forward free of host / device copies inside the block.

    anomalib 2.1.0 builds the ImageNet mean / std on the CPU and copies them to the device in every
    PDN / autoencoder forward, and checks its normalisation buffers with a Python bool of a device
    tensor. Both force a host synchronisation, which an ONNX trace turns into warnings and a
    CUDA-graph capture refuses. Inside the block the constants come from a per-device cache (same
    values) and ``is_set`` answers from one check made on entry, so the computation is unchanged.
    Restored on exit.
    """
    import anomalib.models.image.efficient_ad.torch_model as tm

    cache: dict[torch.device, tuple[Tensor, Tensor]] = {}

    def imagenet_norm_batch(x: Tensor) -> Tensor:
        if x.device not in cache:
            cache[x.device] = (
                torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(x.device),
                torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(x.device),
            )
        mean, std = cache[x.device]
        return (x - mean) / std

    answers = {
        id(getattr(model, k)): bool(tm.EfficientAdModel.is_set(getattr(model, k)))
        for k in ("mean_std", "quantiles")
    }
    original = tm.imagenet_norm_batch
    tm.imagenet_norm_batch = imagenet_norm_batch
    model.is_set = lambda p_dic: answers.get(  # type: ignore[method-assign]
        id(p_dic), bool(tm.EfficientAdModel.is_set(p_dic))
    )
    try:
        yield
    finally:
        tm.imagenet_norm_batch = original
        del model.is_set


class _AnomalyMap(nn.Module):
    """The exported graph: the model's anomaly map (1x1xHxW) for a 1x3xHxW input."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).anomaly_map


def export_onnx(model: nn.Module, image_size: tuple[int, int] | list[int], path: str) -> str:
    """Export the model's inference path to ONNX (opset 17; ``input`` -> ``anomaly_map``).

    The wrapper is traced in eval mode and left in eval mode (``torch.onnx.export`` restores the
    traced module's mode afterwards, and a model left in train mode would return training losses).
    """
    h, w = (int(s) for s in image_size)
    device = next(model.parameters()).device
    wrapper = _AnomalyMap(model).eval()
    example = torch.zeros(1, 3, h, w, device=device)
    with torch.no_grad(), sync_free(model):
        torch.onnx.export(
            wrapper,
            (example,),
            path,
            input_names=["input"],
            output_names=["anomaly_map"],
            opset_version=17,
            dynamo=False,
        )
    model.eval()
    return path


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_engine(
    model: nn.Module,
    image_size: tuple[int, int] | list[int],
    precision: str,
    engine_path: str,
    fingerprint: str,
) -> dict[str, Any]:
    """Export ``model``, build and save a TensorRT engine for this GPU; returns its build record."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}.")
    trt = _tensorrt()
    if precision == "fp16" and not hasattr(trt.BuilderFlag, "FP16"):
        raise RuntimeError(
            f"TensorRT {trt.__version__} has no FP16 builder flag (dropped in TensorRT 11): "
            "build fp16 engines with TensorRT 10, or use precision 'fp32' / 'tf32'."
        )
    t0 = time.perf_counter()
    logger = _logger(trt)
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = export_onnx(model, image_size, os.path.join(tmp, "efficientad.onnx"))
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx_path):
            errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT could not parse the exported EfficientAD model: {errors}")
        config = builder.create_builder_config()
        if precision == "fp32":
            config.clear_flag(trt.BuilderFlag.TF32)
        elif precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError(f"TensorRT engine build failed ({precision}).")
    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(blob)
    h, w = (int(s) for s in image_size)
    record = {
        "engine": os.path.basename(engine_path),
        "precision": precision,
        "fingerprint": fingerprint,
        "input": [1, 3, h, w],
        "tensorrt": trt.__version__,
        "torch": torch.__version__,
        "anomalib": _version("anomalib"),
        "cuvis_ai_efficientad": _version("cuvis-ai-efficientad"),
        "gpu": torch.cuda.get_device_name(),
        "gpu_tag": gpu_tag(),
        "build_seconds": round(time.perf_counter() - t0, 1),
        "built_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    with open(f"{engine_path}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    return record


class TensorRTEngine:
    """Run a serialized TensorRT engine (static shapes, one input) on torch CUDA tensors.

    The engine runs on its own CUDA stream (TensorRT adds a host synchronisation to every call on
    the default stream), ordered after the work already queued on torch's current stream, and
    torch's current stream waits for it, so torch ops that follow see the results in order. The
    output tensors are reused buffers, overwritten by the next call - consume or copy them before
    calling again.
    """

    def __init__(self, path: str, device: torch.device | str = "cuda") -> None:
        trt = _tensorrt()
        self.path = path
        self.device = torch.device(device)
        dtypes = {
            getattr(trt, name): dtype
            for name, dtype in (
                ("float32", torch.float32),
                ("float16", torch.float16),
                ("bfloat16", torch.bfloat16),
            )
            if hasattr(trt, name)
        }
        self._runtime = trt.Runtime(_logger(trt))
        with open(path, "rb") as f, torch.cuda.device(self.device):
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
            self._stream = torch.cuda.Stream(self.device)
        if self._engine is None:
            raise RuntimeError(
                f"TensorRT could not load {path}: an engine only runs on the GPU and TensorRT "
                "version it was built with - rebuild it on this machine "
                "(python -m cuvis_ai_efficientad.trt_engine build-pipeline ...)."
            )
        self._context = self._engine.create_execution_context()
        inputs: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {}
        self.outputs: dict[str, Tensor] = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            dtype = dtypes[self._engine.get_tensor_dtype(name)]
            shape = tuple(self._engine.get_tensor_shape(name))
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs[name] = (dtype, shape)
            else:
                buf = torch.empty(shape, dtype=dtype, device=self.device)
                self.outputs[name] = buf
                self._context.set_tensor_address(name, buf.data_ptr())
        if len(inputs) != 1:
            raise RuntimeError(f"{path}: expected one engine input, found {sorted(inputs)}.")
        self.input_name, (self.input_dtype, self.input_shape) = next(iter(inputs.items()))
        self._held: Tensor | None = None

    def __call__(self, x: Tensor) -> dict[str, Tensor]:
        x = x.to(device=self.device, dtype=self.input_dtype).contiguous()
        self._context.set_tensor_address(self.input_name, x.data_ptr())
        current = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current)  # x written, previous outputs consumed
        with torch.cuda.device(self.device):
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError(f"TensorRT execution failed for {self.path}.")
        current.wait_stream(self._stream)  # everything queued after this sees the outputs
        self._held = x  # the engine reads x asynchronously; keep it alive until the next call
        return self.outputs


def engine_nodes(pipeline: Any) -> list[Any]:
    """The ``backend='tensorrt'`` EfficientAdDetector nodes of a restored pipeline."""
    from cuvis_ai_efficientad.node.efficientad import EfficientAdDetector

    return [
        n for n in pipeline.nodes if isinstance(n, EfficientAdDetector) and n.backend == "tensorrt"
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m cuvis_ai_efficientad.trt_engine",
        description="Build the TensorRT engines of the backend='tensorrt' EfficientAdDetector "
        "nodes of pipelines.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    pipe = sub.add_parser(
        "build-pipeline", help="restore each pipeline with its weights and build its engines"
    )
    pipe.add_argument(
        "pipelines", nargs="+", help="pipeline yamls; the weights are the sibling .pt"
    )
    pipe.add_argument(
        "--plugins-dir", default=None, help="plugin catalog, only for yamls outside a cuvis-ai tree"
    )
    pipe.add_argument("--force", action="store_true", help="rebuild existing engines")
    args = ap.parse_args(argv)
    from cuvis_ai_core.utils.restore import restore_pipeline

    for yaml_path in args.pipelines:
        weights = os.path.splitext(yaml_path)[0] + ".pt"
        pipeline = restore_pipeline(
            yaml_path,
            weights_path=weights,
            device="cuda",
            plugins_dirs=[args.plugins_dir] if args.plugins_dir else None,
        )
        nodes = engine_nodes(pipeline)
        if not nodes:
            print(f"{yaml_path}: no backend='tensorrt' EfficientAdDetector", flush=True)
        for node in nodes:
            path, built = node.build_engine(force=args.force)
            print(f"{'built' if built else 'exists'}: {path} ({node.name})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
