"""TensorRT backend of EfficientAdDetector and the engine helpers.

Mocked tests (no TensorRT, no GPU) check the plumbing: the engine gets exactly the model input of
the torch path (so a network that computes the model gives outputs identical to the torch backend),
the precision follows the node's options, engines are keyed by a fingerprint of the weights, a
missing engine names the build command, new weights drop the loaded engine, the ONNX export leaves
the model and anomalib as they were, and the build writes the engine plus its record. The ``slow``
test builds and runs real engines where CUDA, TensorRT and onnx are available.
"""

from __future__ import annotations

import json
import os
import sys
import types

import anomalib.models.image.efficient_ad.torch_model as tm
import pytest
import torch

from cuvis_ai_efficientad import trt_engine
from cuvis_ai_efficientad.node.efficientad import EfficientAdDetector
from tests.conftest import SIZE, set_trained_state

CUDA = torch.cuda.is_available()


class FakeEngine:
    """Stands in for a TensorRT engine: computes the model in torch, records its inputs."""

    def __init__(self, model):
        self.model = model
        self.inputs: list[torch.Tensor] = []

    def __call__(self, x):
        self.inputs.append(x)
        with torch.no_grad():
            return {"anomaly_map": self.model(x.contiguous()).anomaly_map}


def make_node(**kw) -> EfficientAdDetector:
    node = EfficientAdDetector(image_size=SIZE, name="effad", **kw)
    set_trained_state(node.model)
    return node


# ------------------------------------------------------------------ options
def test_backend_defaults_hparams_and_validation():
    node = EfficientAdDetector(image_size=SIZE)
    assert node.backend == "torch" and node.engine_dir is None
    assert node.hparams["backend"] == "torch" and node.hparams["engine_dir"] is None
    trt_node = EfficientAdDetector(image_size=SIZE, backend="tensorrt", engine_dir="/engines")
    assert (
        trt_node.hparams["backend"] == "tensorrt" and trt_node.hparams["engine_dir"] == "/engines"
    )
    with pytest.raises(ValueError, match="backend"):
        EfficientAdDetector(backend="onnx")
    with pytest.raises(ValueError, match="bfloat16"):
        EfficientAdDetector(backend="tensorrt", autocast_dtype="bfloat16")
    with pytest.raises(ValueError, match="engine_dir"):
        EfficientAdDetector(engine_dir=" ")


@pytest.mark.parametrize(
    "kw, precision",
    [
        ({}, "fp32"),
        ({"tf32": True}, "tf32"),
        ({"autocast_dtype": "float16"}, "fp16"),
        ({"autocast_dtype": "fp16", "tf32": True}, "fp16"),
    ],
)
def test_engine_precision_follows_the_node_options(kw, precision):
    assert (
        EfficientAdDetector(image_size=SIZE, backend="tensorrt", **kw).engine_precision == precision
    )


# ------------------------------------------------------------------ fingerprints and names
def test_fingerprint_is_stable_and_tracks_weights_and_extra(detector):
    fp = trt_engine.weights_fingerprint(detector.model, "a")
    assert fp == trt_engine.weights_fingerprint(detector.model, "a") and len(fp) == 16
    assert fp != trt_engine.weights_fingerprint(detector.model, "b")
    with torch.no_grad():
        next(detector.model.parameters()).add_(1e-3)
    assert fp != trt_engine.weights_fingerprint(detector.model, "a")


@pytest.fixture
def fake_trt(monkeypatch):
    """A minimal ``tensorrt`` module: records builder flags, returns a fake serialized engine."""

    class BuilderFlag:
        TF32, FP16 = "TF32", "FP16"

    class Logger:
        WARNING = 1

        def __init__(self, level):
            self.level = level

    class Config:
        def __init__(self):
            self.flags = {BuilderFlag.TF32}

        def set_flag(self, flag):
            self.flags.add(flag)

        def clear_flag(self, flag):
            self.flags.discard(flag)

    class Builder:
        blob: bytes | None = b"engine"
        configs: list[Config] = []

        def __init__(self, logger):
            self.logger = logger

        def create_network(self, flags):
            return object()

        def create_builder_config(self):
            Builder.configs.append(Config())
            return Builder.configs[-1]

        def build_serialized_network(self, network, config):
            return Builder.blob

    class OnnxParser:
        ok = True

        def __init__(self, network, logger):
            self.num_errors = 0 if OnnxParser.ok else 1

        def parse_from_file(self, path):
            assert os.path.exists(path)
            return OnnxParser.ok

        def get_error(self, i):
            return "bad node"

    trt = types.SimpleNamespace(
        __version__="10.15.1.29",
        BuilderFlag=BuilderFlag,
        Logger=Logger,
        Builder=Builder,
        OnnxParser=OnnxParser,
    )
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setattr(trt_engine, "gpu_tag", lambda device=None: "Fake-GPU-sm00")
    monkeypatch.setattr(trt_engine, "_LOGGER", None)
    return trt


def test_engine_file_name_and_default_dir(fake_trt, monkeypatch, tmp_path):
    name = trt_engine.engine_file_name("tf32", "0123456789abcdef", [512, 256])
    assert name == "tf32_0123456789abcdef_512x256_Fake-GPU-sm00_trt10.15.1.29.engine"
    monkeypatch.setenv(trt_engine.ENGINE_DIR_ENV, str(tmp_path))
    assert trt_engine.default_engine_dir() == os.path.join(str(tmp_path), "efficientad")
    monkeypatch.delenv(trt_engine.ENGINE_DIR_ENV)
    assert trt_engine.default_engine_dir().endswith(
        os.path.join(".cache", "cuvis-ai", "tensorrt", "efficientad")
    )


def test_engine_path_carries_precision_fingerprint_and_size(fake_trt, tmp_path):
    fp32 = make_node(backend="tensorrt", engine_dir=str(tmp_path)).engine_path()
    fp16 = make_node(
        backend="tensorrt", engine_dir=str(tmp_path), autocast_dtype="float16"
    ).engine_path()
    assert os.path.dirname(fp32) == str(tmp_path)
    assert os.path.basename(fp32).startswith("fp32_") and f"_{SIZE}x{SIZE}_" in fp32
    assert os.path.basename(fp16).startswith("fp16_")
    assert fp32.split("_")[-4] == fp16.split("_")[-4]  # same weights, same fingerprint
    other = make_node(backend="tensorrt", engine_dir=str(tmp_path))
    set_trained_state(other.model, seed=3)
    assert other.engine_path() != fp32


def test_tensorrt_import_errors(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    with pytest.raises(ImportError, match="tensorrt-cu12"):
        trt_engine._tensorrt()
    monkeypatch.setitem(sys.modules, "tensorrt", types.SimpleNamespace(__version__="9.3.0"))
    with pytest.raises(ImportError, match=">= 10"):
        trt_engine._tensorrt()


# ------------------------------------------------------------------ forward plumbing
def test_tensorrt_forward_equals_torch_forward(detector, rgb, monkeypatch):
    node = make_node(backend="tensorrt")
    fake = FakeEngine(node.model)
    monkeypatch.setattr(node, "_load_engine", lambda device: fake)
    got, want = node(rgb_image=rgb), detector(rgb_image=rgb)
    assert torch.equal(got["scores"], want["scores"])
    assert torch.equal(got["anomaly_score"], want["anomaly_score"])
    assert tuple(fake.inputs[0].shape) == (1, 3, SIZE, SIZE)
    node(rgb_image=rgb)
    assert len(fake.inputs) == 2 and node._engine is fake  # loaded once, reused


def test_tensorrt_batch_runs_frame_by_frame(detector, monkeypatch):
    frames = torch.rand(3, 200, 210, 3, generator=torch.Generator().manual_seed(4))
    node = make_node(backend="tensorrt")
    fake = FakeEngine(node.model)
    monkeypatch.setattr(node, "_load_engine", lambda device: fake)
    got, want = node(rgb_image=frames), detector(rgb_image=frames)
    assert len(fake.inputs) == 3 and all(tuple(x.shape) == (1, 3, SIZE, SIZE) for x in fake.inputs)
    assert torch.allclose(got["scores"], want["scores"], atol=1e-6)
    assert torch.allclose(got["anomaly_score"], want["anomaly_score"], atol=1e-6)


def test_new_weights_drop_the_loaded_engine(detector):
    node = make_node(backend="tensorrt")
    node._engine = object()
    node.load_state_dict(detector.state_dict())
    assert node._engine is None


def test_tensorrt_needs_cuda_and_names_the_build_command(fake_trt, tmp_path, monkeypatch):
    node = make_node(backend="tensorrt", engine_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="CUDA"):
        node._load_engine(torch.device("cpu"))
    with pytest.raises(FileNotFoundError, match="trt_engine build-pipeline"):
        node._load_engine(torch.device("cuda"))
    with pytest.raises(RuntimeError, match="CUDA"):
        node.build_engine()


def test_engine_with_the_wrong_input_shape_is_refused(fake_trt, tmp_path, monkeypatch):
    node = make_node(backend="tensorrt", engine_dir=str(tmp_path))
    path = node.engine_path()
    open(path, "wb").close()
    monkeypatch.setattr(
        trt_engine, "TensorRTEngine", lambda p, d: types.SimpleNamespace(input_shape=(1, 3, 64, 64))
    )
    with pytest.raises(RuntimeError, match="rebuild"):
        node._load_engine(torch.device("cuda"))


# ------------------------------------------------------------------ export and build
def test_sync_free_is_bit_identical_and_restores_anomalib(detector):
    x = torch.rand(1, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(2))
    original = tm.imagenet_norm_batch
    with torch.no_grad():
        want = detector.model(x).anomaly_map
        with trt_engine.sync_free(detector.model):
            assert tm.imagenet_norm_batch is not original
            got = detector.model(x).anomaly_map
    assert torch.equal(got, want)
    assert tm.imagenet_norm_batch is original and "is_set" not in vars(detector.model)


def test_export_keeps_the_model_in_eval_mode(detector, tmp_path):
    pytest.importorskip("onnx")
    path = trt_engine.export_onnx(detector.model, [SIZE, SIZE], str(tmp_path / "e.onnx"))
    assert os.path.getsize(path) > 0
    assert not detector.model.training and not any(m.training for m in detector.model.modules())
    assert "is_set" not in vars(detector.model)


@pytest.mark.parametrize(
    "precision, flags", [("fp32", set()), ("tf32", {"TF32"}), ("fp16", {"TF32", "FP16"})]
)
def test_build_engine_writes_the_engine_and_its_record(
    fake_trt, monkeypatch, tmp_path, detector, precision, flags
):
    monkeypatch.setattr(trt_engine, "export_onnx", lambda m, s, p: open(p, "wb").close() or p)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Fake GPU")
    fake_trt.Builder.configs.clear()
    path = str(tmp_path / "e" / "x.engine")
    record = trt_engine.build_engine(detector.model, [SIZE, SIZE], precision, path, "fp")
    assert open(path, "rb").read() == b"engine"
    assert json.loads(open(path + ".json").read()) == record
    assert (
        record["precision"] == precision
        and record["fingerprint"] == "fp"
        and record["input"] == [1, 3, SIZE, SIZE]
    )
    assert fake_trt.Builder.configs[-1].flags == flags


def test_build_engine_failures(fake_trt, monkeypatch, tmp_path, detector):
    monkeypatch.setattr(trt_engine, "export_onnx", lambda m, s, p: open(p, "wb").close() or p)
    with pytest.raises(ValueError, match="precision"):
        trt_engine.build_engine(detector.model, [SIZE, SIZE], "int8", str(tmp_path / "a"), "fp")
    fake_trt.OnnxParser.ok = False
    try:
        with pytest.raises(RuntimeError, match="parse"):
            trt_engine.build_engine(detector.model, [SIZE, SIZE], "fp32", str(tmp_path / "a"), "fp")
    finally:
        fake_trt.OnnxParser.ok = True
    fake_trt.Builder.blob = None
    try:
        with pytest.raises(RuntimeError, match="build failed"):
            trt_engine.build_engine(detector.model, [SIZE, SIZE], "fp32", str(tmp_path / "a"), "fp")
    finally:
        fake_trt.Builder.blob = b"engine"
    monkeypatch.delattr(fake_trt.BuilderFlag, "FP16")
    with pytest.raises(RuntimeError, match="FP16"):
        trt_engine.build_engine(detector.model, [SIZE, SIZE], "fp16", str(tmp_path / "a"), "fp")


def test_node_build_engine_skips_an_existing_engine(monkeypatch, tmp_path):
    node = make_node(backend="tensorrt", engine_dir=str(tmp_path))
    engine = str(tmp_path / "fp32_0123456789abcdef_256x256_gpu_trt10.engine")
    monkeypatch.setattr(node, "engine_path", lambda device=None: engine)
    # the model stays on the CPU here: report a CUDA parameter so build_engine takes its CUDA path
    monkeypatch.setattr(
        node.model, "parameters", lambda: iter([types.SimpleNamespace(device=torch.device("cuda"))])
    )
    calls = []
    monkeypatch.setattr(
        trt_engine, "build_engine", lambda *a: calls.append(a) or open(a[3], "wb").close()
    )
    node._engine = object()
    path, built = node.build_engine()
    assert (path, built) == (engine, True) and node._engine is None
    assert calls[0][2] == "fp32" and calls[0][3] == engine and calls[0][4] == "0123456789abcdef"
    assert node.build_engine() == (engine, False) and len(calls) == 1
    assert node.build_engine(force=True) == (engine, True) and len(calls) == 2


def test_build_pipeline_cli_builds_each_tensorrt_node(monkeypatch, capsys):
    built = []
    trt_node = make_node(backend="tensorrt")
    torch_node = make_node()
    monkeypatch.setattr(
        trt_node, "build_engine", lambda force=False: built.append(force) or ("/e/x.engine", True)
    )
    pipeline = types.SimpleNamespace(nodes=[trt_node, torch_node, "not a node"])
    import cuvis_ai_core.utils.restore as restore

    seen = {}
    monkeypatch.setattr(
        restore, "restore_pipeline", lambda y, **kw: seen.update(y=y, **kw) or pipeline
    )
    assert trt_engine.main(["build-pipeline", "p.yaml", "--force"]) == 0
    assert built == [True] and seen["weights_path"] == "p.pt" and seen["device"] == "cuda"
    assert "built: /e/x.engine (effad)" in capsys.readouterr().out
    monkeypatch.setattr(
        restore, "restore_pipeline", lambda y, **kw: types.SimpleNamespace(nodes=[torch_node])
    )
    trt_engine.main(["build-pipeline", "q.yaml"])
    assert "no backend='tensorrt'" in capsys.readouterr().out


# ------------------------------------------------------------------ real TensorRT
@pytest.mark.slow
@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize(
    "kw, rel_tol", [({}, 1e-4), ({"tf32": True}, 5e-3), ({"autocast_dtype": "float16"}, 3e-2)]
)
def test_real_tensorrt_engine_matches_the_torch_backend(tmp_path, rgb, kw, rel_tol):
    pytest.importorskip("tensorrt")
    pytest.importorskip("onnx")
    ref = make_node(**kw).cuda()
    node = make_node(backend="tensorrt", engine_dir=str(tmp_path), **kw).cuda()
    path, built = node.build_engine()
    assert built and os.path.exists(path) and os.path.exists(path + ".json")
    x = rgb.cuda()
    got = node(rgb_image=x)
    # the fp32 engine is IEEE float32: compare it with IEEE convolutions (PyTorch's default is TF32)
    conv = torch.backends.cudnn.conv.fp32_precision
    torch.backends.cudnn.conv.fp32_precision = "ieee" if not kw else conv
    try:
        want = ref(rgb_image=x)
    finally:
        torch.backends.cudnn.conv.fp32_precision = conv
    scale = float(want["scores"].abs().max())
    assert float((got["scores"] - want["scores"]).abs().max()) <= rel_tol * scale
    assert float((got["anomaly_score"] - want["anomaly_score"]).abs().max()) <= rel_tol * scale
    assert not node.model.training
