#!/usr/bin/env python3
"""Launch mlx_lm LoRA with the single-process ring communication backend.

Some Conda installations expose MPICH. MLX's default ``any`` backend probes
that library first and aborts because MLX requires Open MPI. A one-process
local job does not need MPI, so this wrapper explicitly selects MLX's ring
backend with a group size of one without modifying the installed package.
"""

from __future__ import annotations

import os

os.environ.setdefault("MLX_RANK", "0")
os.environ.setdefault("MLX_SIZE", "1")

import mlx.core as mx  # noqa: E402


_original_distributed_init = mx.distributed.init


def _single_process_init(*args, **kwargs):
    return _original_distributed_init(strict=bool(kwargs.get("strict", False)), backend="ring")


mx.distributed.init = _single_process_init


def _single_process_all_sum(value, *args, **kwargs):
    # mlx_lm's trainer calls all_sum without passing the already-created
    # singleton group. Returning the input is the exact all-reduce result for
    # a group of size one and avoids a second default-backend MPI probe.
    return value


mx.distributed.all_sum = _single_process_all_sum

from mlx_lm.lora import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
