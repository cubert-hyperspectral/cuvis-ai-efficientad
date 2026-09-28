"""Guard the torch index pins in ``pyproject.toml`` and the CI lock (the cuvis-ai pattern).

A local ``uv sync`` takes torch and torchvision from a PyTorch wheel index through the ``cuda``
dependency group: aarch64 Linux (Jetson Thor, CUDA 13) from cu130, every other platform from cu128.
Both entries stay scoped to the ``cuda`` group, so an environment that installs this plugin as a
path or git dependency (a cuvis.next child environment) inherits no index pin. The committed lock
is the CI lock: CI runs ``uv run --no-sources --locked``, so the lock must resolve torch from PyPI.
The base requirements are split along the same markers, because uv assigns an explicit index
only to a requirement inside one resolver fork.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.markers import Marker
from packaging.requirements import Requirement

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
FORKED = ("torch", "torchvision")
# The forked packages with a base requirement in [project].dependencies (anomalib declares
# neither, so the plugin declares both).
BASE_FORKED = FORKED
INDEX_URL = {
    "pytorch-cu128": "https://download.pytorch.org/whl/cu128",
    "pytorch-cu130": "https://download.pytorch.org/whl/cu130",
}
ENVIRONMENTS = {
    "jetson": {"sys_platform": "linux", "platform_machine": "aarch64"},
    "linux-x86_64": {"sys_platform": "linux", "platform_machine": "x86_64"},
    "windows": {"sys_platform": "win32", "platform_machine": "AMD64"},
}
EXPECTED_INDEX = {
    "jetson": "pytorch-cu130",
    "linux-x86_64": "pytorch-cu128",
    "windows": "pytorch-cu128",
}


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("package", FORKED)
def test_sources_are_scoped_to_the_cuda_group(pyproject, package):
    entries = pyproject["tool"]["uv"]["sources"][package]
    assert isinstance(entries, list) and len(entries) == 2
    assert all(entry["group"] == "cuda" for entry in entries), "an unscoped pin leaks to consumers"


@pytest.mark.parametrize("package", FORKED)
@pytest.mark.parametrize("platform", ENVIRONMENTS)
def test_each_platform_resolves_one_index(pyproject, package, platform):
    entries = pyproject["tool"]["uv"]["sources"][package]
    matching = [e["index"] for e in entries if Marker(e["marker"]).evaluate(ENVIRONMENTS[platform])]
    assert matching == [EXPECTED_INDEX[platform]]


@pytest.mark.parametrize("package", FORKED)
def test_base_requirements_are_split_along_the_fork_markers(pyproject, package):
    """One base requirement per index fork with one floor, or none (the cuvis-ai pattern).

    An unsplit requirement next to the two marker-gated pins fails every resolution that reads the
    sources (``uv sync``, ``uv run``, the release workflow) with "conflicting indexes for package
    torch in all marker environments". CI's ``--no-sources`` runs never read the sources.
    """
    requirements = [
        Requirement(dep)
        for dep in pyproject["project"]["dependencies"]
        if Requirement(dep).name == package
    ]
    expected = 2 if package in BASE_FORKED else 0
    assert len(requirements) == expected, f"{package}: {[str(r) for r in requirements]}"
    if requirements:
        assert len({str(req.specifier) for req in requirements}) == 1, f"{package}: floors differ"
        entries = pyproject["tool"]["uv"]["sources"][package]
        source_markers = {str(Marker(entry["marker"])) for entry in entries}
        assert {str(req.marker) for req in requirements} == source_markers


def test_indexes_are_explicit_pytorch_indexes(pyproject):
    indexes = {i["name"]: i for i in pyproject["tool"]["uv"]["index"]}
    for name, url in INDEX_URL.items():
        assert indexes[name]["url"] == url
        assert indexes[name]["explicit"] is True, "a non-explicit index would serve every package"


def test_cuda_group_carries_the_pins_and_is_installed_by_default(pyproject):
    assert set(pyproject["dependency-groups"]["cuda"]) == set(FORKED)
    assert "cuda" in pyproject["tool"]["uv"]["default-groups"]


def test_lock_resolves_torch_from_pypi_for_ci():
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    for package in FORKED:
        entries = [p for p in lock["package"] if p["name"] == package]
        assert entries, f"{package} missing from uv.lock"
        for entry in entries:
            assert entry["source"] == {"registry": "https://pypi.org/simple"}, (
                f"{package} {entry['version']} is locked from {entry['source']}; regenerate the "
                "lock with `uv lock --no-sources` (CI runs `uv run --no-sources --locked`)"
            )
