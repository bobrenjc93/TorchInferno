#!/usr/bin/env python3
"""Build the pinned SM90 FP8 output adapter outside serving runtime paths."""

from torchinferno.kernels.sgl_fp8_out_builder import main

if __name__ == "__main__":
    raise SystemExit(main())
