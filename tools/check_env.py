"""Report what this machine offers the compiler, before anything is compiled.

    python tools/check_env.py

Every benchmark result records the machine it came from, and this is the same
information, checked up front: which C compiler, which vector instructions it
will target with ``-march=native``, and whether the oracle and the real weights
are where the tests will look for them.

Instruction sets are read from the compiler's own predefined macros rather
than from the OS. That answers the question that matters here, which is what
the generated code may use, and it works the same on Windows and Linux.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tests import oracle

# Macro gcc defines under -march=native -> what phase it matters for.
FEATURES = {
    "__AVX2__": "AVX2 (phase 4 vectorization)",
    "__FMA__": "FMA (phase 4 vectorization)",
    "__AVXVNNI__": "AVX-VNNI (phase 6 int8)",
    "__AVX512F__": "AVX-512 (not targeted)",
}


def c_compiler() -> tuple[str | None, str]:
    """The compiler on PATH and its version line."""
    cc = os.environ.get("CC") or shutil.which("gcc")
    if cc is None:
        return None, "none on PATH (needed from phase 2; set CC or install gcc)"
    version = subprocess.run([cc, "--version"], capture_output=True, text=True)
    return cc, version.stdout.splitlines()[0] if version.stdout else "unknown"


def native_features(cc: str) -> dict[str, bool]:
    """Which FEATURES macros ``-march=native`` defines."""
    result = subprocess.run(
        [cc, "-march=native", "-dM", "-E", "-"],
        input="", capture_output=True, text=True,
    )
    defined = {line.split()[1] for line in result.stdout.splitlines() if line.startswith("#define")}
    return {macro: macro in defined for macro in FEATURES}


def main() -> int:
    rows: list[tuple[str, str]] = [
        ("python", sys.version.split()[0]),
        ("numpy", np.__version__),
        ("platform", f"{platform.system()} {platform.machine()}"),
        ("cpu threads", str(os.cpu_count())),
    ]

    cc, version = c_compiler()
    rows.append(("c compiler", version if cc is None else f"{cc}: {version}"))
    if cc is not None:
        for macro, enabled in native_features(cc).items():
            rows.append((FEATURES[macro], "yes" if enabled else "no"))

    root = oracle.nanoinfer_root()
    rows.append(("nanoinfer", str(root) if root else "not found (set NANOINFER_PATH)"))
    weights = oracle.model_dir()
    rows.append(("real weights", str(weights) if weights else "not found (reference tests will skip)"))

    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value}")

    return 0 if cc is not None and root is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
