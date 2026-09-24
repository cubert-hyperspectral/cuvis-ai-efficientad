"""Manifest loading: the checked-in plugins.yaml resolves through NodeRegistry and every
capability imports as a Node subclass."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml
from cuvis_ai_core.node.node import Node
from cuvis_ai_core.utils.node_registry import NodeRegistry

from cuvis_ai_efficientad.node.efficientad import EfficientAdDetector

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "plugins.yaml"


def test_manifest_registers_plugin_and_resolves_nodes():
    registry = NodeRegistry()
    registry.register_plugin(str(MANIFEST))
    assert registry.list_plugins() == ["efficientad"]
    assert registry.get("EfficientAdDetector") is EfficientAdDetector


def test_manifest_capabilities_are_importable_nodes():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["name"] == "efficientad"
    assert manifest["package_name"] == "cuvis-ai-efficientad"
    assert manifest["capabilities"], "capabilities must not be empty"
    for entry in manifest["capabilities"]:
        module_name, _, cls_name = entry["class_name"].rpartition(".")
        cls = getattr(importlib.import_module(module_name), cls_name)
        assert issubclass(cls, Node), entry["class_name"]
