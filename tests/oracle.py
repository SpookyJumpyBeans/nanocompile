"""Where nanoinfer and the real weights are, for the tests that need them.

nanoinfer is not a package on PyPI and is not installed here. It is a checkout,
by default the one next to this repository, and it is put on ``sys.path`` only
for tests and tools. ``nanocompile/`` must never import it: the compiler is
being checked against nanoinfer, and a compiler that borrowed nanoinfer's code
could pass that check by agreeing with itself.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parent.parent


def nanoinfer_root() -> Path | None:
    """The nanoinfer checkout, or ``None`` if there is not one."""
    configured = os.environ.get("NANOINFER_PATH")
    root = Path(configured) if configured else REPO.parent / "nanoinfer"
    return root if (root / "nanoinfer" / "model.py").is_file() else None


def model_dir() -> Path | None:
    """The real Qwen2.5-0.5B-Instruct weights, or ``None`` if not downloaded."""
    configured = os.environ.get("NANOCOMPILE_MODEL")
    if configured:
        path = Path(configured)
    else:
        root = nanoinfer_root()
        if root is None:
            return None
        path = root / "models" / "Qwen2.5-0.5B-Instruct"
    return path if (path / "config.json").is_file() else None


def import_nanoinfer() -> Path:
    """Put the nanoinfer checkout on ``sys.path`` and return its root.

    Raises rather than returning ``None``: a caller that got here needs the
    oracle, and the fixtures in conftest turn the error into a skip.
    """
    root = nanoinfer_root()
    if root is None:
        raise FileNotFoundError(
            "nanoinfer not found next to this repository; set NANOINFER_PATH"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def nanoinfer_tiny() -> ModuleType:
    """nanoinfer's ``tests/tiny.py``, which builds its 2-layer test model.

    Loaded from its file under another name, because ``tests`` here is this
    repository's package and importing ``tests.tiny`` would look in the wrong
    place.
    """
    root = import_nanoinfer()
    name = "nanoinfer_tests_tiny"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / "tests" / "tiny.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
