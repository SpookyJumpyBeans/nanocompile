"""Print Qwen2's graph IR, at the real model's widths.

    python tools/dump_graph.py                # all 24 layers, ~3,600 lines
    python tools/dump_graph.py --layers 1     # one block, the readable version
    python tools/dump_graph.py --last-only    # the decode-step variant

Needs only the model's ``config.json``, not its weights: building a graph
reads shapes, never values.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanocompile.models.qwen2 import Qwen2Config, build_qwen2
from tests import oracle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=oracle.model_dir())
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--last-only", action="store_true")
    args = parser.parse_args()

    if args.model is None:
        print("no model directory; pass --model or set NANOCOMPILE_MODEL", file=sys.stderr)
        return 1

    config = Qwen2Config.from_model_dir(args.model)
    if args.layers is not None:
        config = replace(config, num_hidden_layers=args.layers)
    print(build_qwen2(config, last_only=args.last_only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
