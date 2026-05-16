# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, Tuple
import math
import torch


_MAX_QUANT_CHUNK_VALUES = 1024 * 1024


def _num_blocks(numel: int, block_size: int) -> int:
    if numel == 0:
        return 0
    return math.ceil(numel / block_size)


def _blocks_per_chunk(block_size: int) -> int:
    return max(1, _MAX_QUANT_CHUNK_VALUES // block_size)


def _use_8bit_state(p: torch.Tensor, min_8bit_size: int) -> bool:
    return p.is_floating_point() and p.numel() >= min_8bit_size


def init_exp_avg_sq_state(state: Dict[str, Any], p: torch.Tensor,
                          block_size: int, min_8bit_size: int) -> None:
    if _use_8bit_state(p, min_8bit_size):
        num_blocks = _num_blocks(p.numel(), block_size)
        state['exp_avg_sq_q'] = torch.zeros(p.numel(), dtype=torch.uint8, device=p.device)
        state['exp_avg_sq_scale'] = torch.zeros(num_blocks, dtype=torch.float32, device=p.device)
        state['exp_avg_sq_block_size'] = block_size
    else:
        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)


def _ensure_quantized_state_device(state: Dict[str, Any],
                                   device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    q = state['exp_avg_sq_q']
    scale = state['exp_avg_sq_scale']

    if q.device != device:
        q = q.to(device)
        state['exp_avg_sq_q'] = q
    if scale.device != device or scale.dtype != torch.float32:
        scale = scale.to(device=device, dtype=torch.float32)
        state['exp_avg_sq_scale'] = scale

    return q, scale


def dequantize_exp_avg_sq(state: Dict[str, Any], p: torch.Tensor) -> torch.Tensor:
    q, scale = _ensure_quantized_state_device(state, p.device)
    numel = q.numel()
    exp_avg_sq = torch.empty(numel, dtype=p.dtype, device=p.device)

    if numel == 0:
        return exp_avg_sq.view_as(p)

    block_size = state['exp_avg_sq_block_size']
    blocks_per_chunk = _blocks_per_chunk(block_size)

    for block_start in range(0, scale.numel(), blocks_per_chunk):
        block_end = min(scale.numel(), block_start + blocks_per_chunk)
        start = block_start * block_size
        end = min(numel, block_end * block_size)

        chunk = q[start:end].to(dtype=torch.float32)
        chunk_scale = scale[block_start:block_end].repeat_interleave(block_size)[:end-start]
        exp_avg_sq[start:end].copy_(chunk.mul_(chunk_scale).to(dtype=p.dtype))

    return exp_avg_sq.view_as(p)


def quantize_exp_avg_sq(state: Dict[str, Any], exp_avg_sq: torch.Tensor) -> None:
    q, scale = _ensure_quantized_state_device(state, exp_avg_sq.device)
    numel = q.numel()

    if numel == 0:
        return

    block_size = state['exp_avg_sq_block_size']
    blocks_per_chunk = _blocks_per_chunk(block_size)
    flat = exp_avg_sq.detach().reshape(-1)

    for block_start in range(0, scale.numel(), blocks_per_chunk):
        block_end = min(scale.numel(), block_start + blocks_per_chunk)
        start = block_start * block_size
        end = min(numel, block_end * block_size)
        chunk_blocks = block_end - block_start
        padded_numel = chunk_blocks * block_size

        values = flat[start:end].to(dtype=torch.float32)
        if values.numel() != padded_numel:
            padded = values.new_zeros(padded_numel)
            padded[:values.numel()] = values
            values = padded

        blocks = values.view(chunk_blocks, block_size)
        max_values = blocks.amax(dim=1)
        safe_scale = torch.where(max_values > 0,
                                 max_values / 255.0,
                                 torch.ones_like(max_values))
        quantized = torch.round(blocks / safe_scale[:, None]).clamp_(0, 255)

        q[start:end].copy_(quantized.to(dtype=torch.uint8).reshape(-1)[:end-start])
        scale[block_start:block_end].copy_(
            torch.where(max_values > 0, safe_scale, torch.zeros_like(safe_scale)))
