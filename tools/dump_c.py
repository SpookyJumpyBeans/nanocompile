"""Print the C that Qwen2's graph compiles to, at the real model's widths.

    python tools/dump_c.py --layers 1             # every distinct kernel
    python tools/dump_c.py --layers 1 --calls     # the call sequence instead

Kernels are deduplicated, so the source is the same length for 1 layer or 24;
``--calls`` shows the per-node sequence that reuses them.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanocompile import codegen_c
from nanocompile.lower import lower
from nanocompile.models.qwen2 import Qwen2Config, build_qwen2
from nanocompile.symbolic import format_shape
from tests import oracle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=oracle.model_dir())
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--last-only", action="store_true")
    parser.add_argument("--calls", action="store_true")
    args = parser.parse_args()

    if args.model is None:
        print("no model directory; pass --model or set NANOCOMPILE_MODEL", file=sys.stderr)
        return 1

    config = Qwen2Config.from_model_dir(args.model)
    if args.layers is not None:
        config = replace(config, num_hidden_layers=args.layers)
    program = lower(build_qwen2(config, last_only=args.last_only))

    if args.calls:
        for call in program.calls:
            out = call.output
            print(f"{codegen_c.kernel_name(call.kernel):<8} {call.primitive:<12} "
                  f"%{call.node:<5} -> {out.kind} {out.dtype}{format_shape(out.shape)}")
    else:
        print(codegen_c.render_library(program.kernels), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
