from __future__ import annotations

from pathlib import Path

import pytest

from tests import oracle


@pytest.fixture(scope="session")
def nanoinfer_root() -> Path:
    """The oracle's checkout, importable as ``nanoinfer``. Skips without it."""
    try:
        return oracle.import_nanoinfer()
    except FileNotFoundError as error:
        pytest.skip(str(error))


@pytest.fixture(scope="session")
def tiny_model_dir(nanoinfer_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """nanoinfer's 2-layer test model, written once per session.

    Built by nanoinfer's own test helper so both engines read the same bytes.
    """
    return oracle.nanoinfer_tiny().build_tiny_model(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="session")
def real_model_dir(nanoinfer_root: Path) -> Path:
    """The real 494M-parameter weights. Skips if they are not downloaded."""
    path = oracle.model_dir()
    if path is None:
        pytest.skip("Qwen2.5-0.5B-Instruct not downloaded; see nanoinfer's tools/download_model.py")
    return path
