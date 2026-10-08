"""The oracle itself: nanoinfer is reachable and runs the model it will judge.

Every later phase is a comparison against nanoinfer, so the first thing worth
testing is that the comparison can be made at all, and that the compiler does
not secretly depend on the thing it is compared against.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from tests.oracle import REPO


def test_nanoinfer_runs_the_tiny_model(tiny_model_dir: Path) -> None:
    from nanoinfer.model import Qwen2

    model = Qwen2.from_model_dir(tiny_model_dir)
    logits = model.forward(np.array([3, 1, 4, 1, 5]))

    assert logits.shape == (5, model.config.vocab_size)
    assert logits.dtype == np.float32
    assert np.isfinite(logits).all()


@pytest.mark.reference
def test_nanoinfer_loads_the_real_model(real_model_dir: Path) -> None:
    from nanoinfer.config import ModelConfig

    config = ModelConfig.from_model_dir(real_model_dir)
    assert config.num_hidden_layers == 24
    assert config.hidden_size == 896


def test_compiler_never_imports_nanoinfer() -> None:
    """Checked from the source, so it holds even for code no test runs."""
    offenders = []
    for path in sorted((REPO / "nanocompile").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".")[0] in {"nanoinfer", "tests"} for name in names):
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")

    assert offenders == []
