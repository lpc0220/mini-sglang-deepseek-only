#!/usr/bin/env python3
"""
Benchmark: trtllm_batch_decode_with_kv_cache_mla (Kernel #8)
Source: flashinfer (TensorRT-LLM kernel)
Category: Attention (Mixed bound)
Ops: MLA decode attention (alternative implementation)

Based on: flashinfer/benchmarks/bench_trtllm_gen_mla.py

Usage:
    python bench_trtllm_mla.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    Nh,
    BenchmarkResult, PEAK_BANDWIDTH_GBS, PEAK_TFLOPS_FP16, RIDGE_FP16,
    benchmark_kernel, save_results, check_flashinfer
)

# DeepSeek MLA fixed dimensions (from flashinfer benchmark)
NUM_Q_HEADS = Nh  # 128
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
KV_LORA_RANK = 512
HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576


def bench_trtllm_mla(flashinfer, B: int, seq_len: int, page_size: int = 32,
                     device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark TensorRT-LLM MLA decode kernel.

    Based on flashinfer/benchmarks/bench_trtllm_gen_mla.py:
        - query: [B, q_len, num_heads, kv_lora_rank + qk_rope_head_dim]
        - kv_cache: [num_blocks, page_size, kv_lora_rank + qk_rope_head_dim]
        - Must unsqueeze(1) to add head dimension for API
        - workspace_buffer: 128MB as int8
    """
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    except ImportError:
        print("Warning: trtllm_batch_decode_with_kv_cache_mla not available (requires flashinfer)")
        return None

    torch.manual_seed(42)
    q_len_per_request = 1  # decode: single token per request

    # Query: [B, q_len, num_heads, 576]
    query = torch.randn(
        B, q_len_per_request, NUM_Q_HEADS, HEAD_DIM,
        dtype=torch.bfloat16, device=device
    )

    # Paged KV cache setup
    num_tokens = seq_len * B
    num_blocks = (num_tokens + page_size - 1) // page_size

    # Sequence lengths (use same seq_len for all, with last one = max)
    seq_lens_list = [seq_len] * B
    max_seq_len = max(seq_lens_list)
    seq_lens_tensor = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)

    blocks_per_seq = (seq_lens_tensor + page_size - 1) // page_size
    max_num_blocks_per_seq = blocks_per_seq.max().item()

    # Generate block table with unique block IDs
    total_blocks_needed = int(blocks_per_seq.sum().item())
    all_block_ids = torch.arange(total_blocks_needed, dtype=torch.int32, device=device)

    block_tables = torch.zeros(
        (B, max_num_blocks_per_seq), dtype=torch.int32, device=device
    )
    block_id = 0
    for i in range(B):
        num_blocks_needed = int(blocks_per_seq[i].item())
        block_tables[i, :num_blocks_needed] = all_block_ids[block_id:block_id + num_blocks_needed]
        block_id += num_blocks_needed

    # KV cache: [num_blocks, page_size, 576]
    kv_cache = torch.randn(
        num_blocks, page_size, HEAD_DIM,
        dtype=torch.bfloat16, device=device
    )

    # Workspace buffer - 128MB as int8 (from official benchmark)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # Scale factor: 1.0 / sqrt(128 + 64)
    sm_scale = 1.0 / ((QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) ** 0.5)

    def kernel_fn():
        trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache.unsqueeze(1),  # Add head dimension: [num_blocks, 1, page_size, 576]
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            block_tables=block_tables,
            seq_lens=seq_lens_tensor,
            max_seq_len=max_seq_len,
            bmm1_scale=sm_scale,
            bmm2_scale=1.0,
        )

    try:
        latency_ms = benchmark_kernel(kernel_fn)
    except Exception as e:
        print(f"Warning: Kernel failed for B={B}, seq_len={seq_len}: {e}")
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except:
            pass
        return None

    # FLOPS calculation (from official benchmark)
    # 2 * num_q_heads * (2 * kv_lora_rank + qk_rope_head_dim) * sum(seq_lens) * q_len_per_request
    flops = 2 * NUM_Q_HEADS * (2 * KV_LORA_RANK + QK_ROPE_HEAD_DIM) * sum(seq_lens_list) * q_len_per_request
    q_bytes = query.numel() * query.element_size()
    kv_bytes = kv_cache.numel() * kv_cache.element_size()
    bytes_transferred = q_bytes + kv_bytes
    gflops = flops / (latency_ms * 1e-3) / 1e9
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    bound = "memory" if arith_intensity < RIDGE_FP16 else "compute"
    if bound == "memory":
        peak_pct = (bandwidth_gbs / PEAK_BANDWIDTH_GBS) * 100
    else:
        peak_pct = (gflops / 1000 / PEAK_TFLOPS_FP16) * 100

    return BenchmarkResult(
        kernel="trtllm_batch_decode_with_kv_cache_mla",
        op="attn_mqa",
        phase="decode",
        B=B, S=seq_len,
        M=B * NUM_Q_HEADS, N=seq_len, K_dim=HEAD_DIM,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound=bound
    )


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run trtllm_mla benchmarks."""
    flashinfer = check_flashinfer()
    if not flashinfer:
        print("ERROR: flashinfer not available")
        return

    results = []

    print("\n=== Decode Phase (TRT-LLM MLA Attention) ===")
    for page_size in [32, 64]:
        print(f"\n  Page size: {page_size}")
        for B in batch_sizes:
            for seq_len in seq_lens:
                result = bench_trtllm_mla(flashinfer, B, seq_len, page_size)
                if result:
                    results.append(result)
                    print(f"    B={B}, seq_len={seq_len}: {result.latency_ms:.4f} ms, "
                          f"{result.bandwidth_gbs:.1f} GB/s, {result.bound}")

    save_results(results, output_dir, "trtllm_batch_decode_with_kv_cache_mla")


def main():
    parser = argparse.ArgumentParser(description="Benchmark trtllm_mla kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,4,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="1024,4096,8192",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: trtllm_batch_decode_with_kv_cache_mla (Kernel #8)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
