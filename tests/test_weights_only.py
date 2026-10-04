"""Regression: checkpoints are data. torch.load must refuse pickled code.

Does not download AlphaFace weights. It checks the call sites in source.
"""

from __future__ import annotations

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "src" / "deepfake"


def _torch_loads(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "load"
            and isinstance(func.value, ast.Name)
            and func.value.id == "torch"
        ):
            found.append(node)
    return found


def _weights_only_true(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "weights_only" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
            return True
    return False


def test_alphaface_checkpoint_uses_weights_only():
    path = PKG / "swap" / "alphaface.py"
    calls = _torch_loads(path)
    assert calls, "AlphaFace loader must call torch.load"
    assert all(_weights_only_true(c) for c in calls)


def test_package_torch_load_is_weights_only():
    offenders: list[str] = []
    for path in sorted(PKG.rglob("*.py")):
        for call in _torch_loads(path):
            if not _weights_only_true(call):
                offenders.append(f"{path.relative_to(PKG)}:{call.lineno}")
    assert offenders == []
