"""Run a PaddleOCR tool after preloading PyTorch DLLs on Windows."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch  # noqa: F401 - import order prevents a Windows DLL conflict.


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: paddleocr_runner.py TOOL_PATH [TOOL_ARGS ...]")

    tool_path = Path(sys.argv[1]).resolve()
    if not tool_path.is_file():
        raise FileNotFoundError(f"PaddleOCR tool not found: {tool_path}")

    sys.argv = [str(tool_path), *sys.argv[2:]]
    runpy.run_path(str(tool_path), run_name="__main__")


if __name__ == "__main__":
    main()
