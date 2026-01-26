#!/usr/bin/env python3
"""
Benchmark: trtllm_batch_decode_with_kv_cache_mla (Kernel #8)
Source: flashinfer (TensorRT-LLM kernel)
Category: Attention (Mixed bound)
Ops: MLA decode attention (alternative implementation)

Usage:
    python bench_trtllm_mla.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    Nh, Lkv, Dr,
    BenchmarkResult, PEAK_BANDWIDTH_GBS, PEAK_TFLOPS_FP16, RIDGE_FP16,
    benchmark_kernel, save_results, check_flashinfer
)


def bench_trtllm_mla(flashinfer, B: int, seq_len: int,
                     device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark TensorRT-LLM MLA decode kernel.

    API signature:
        trtllm_batch_decode_with_kv_cache_mla(
            query,              # [B, q_len, num_heads, head_dim_qk] where head_dim_qk = qk_nope + qk_rope
            kv_cache,           # [num_pages, page_size, head_dim_ckv + head_dim_kpe]
            workspace_buffer,   # [num_semaphores, 4]
            qk_nope_head_dim,   # 128
            kv_lora_rank,       # 512
            qk_rope_head_dim,   # 64
            block_tables,       # [B, num_pages]
            seq_lens,           # [B]
            max_seq_len,        # int
            ...
        )
    """
    try:
        from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
    except ImportError:
        print("Warning: trtllm_batch_decode_with_kv_cache_mla not available (requires flashinfer)")
        return None

    # DeepSeek MLA dimensions
    qk_nope_head_dim = 128  # Dn
    kv_lora_rank = Lkv      # 512
    qk_rope_head_dim = Dr   # 64
    head_dim_qk = qk_nope_head_dim + qk_rope_head_dim  # 192
    head_dim_ckv_kpe = kv_lora_rank + qk_rope_head_dim  # 576 = 512 + 64

    block_size = 64
    q_len = 1  # decode: single token per request

    # Query: [B, q_len, num_heads, head_dim_qk]
    query = torch.randn(B, q_len, Nh, head_dim_qk, dtype=torch.bfloat16, device=device)

    # Paged KV cache: [num_pages, page_size, head_dim_ckv + head_dim_kpe]
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_pages = B * num_blocks_per_seq
    kv_cache = torch.randn(total_pages, block_size, head_dim_ckv_kpe, dtype=torch.bfloat16, device=device)

    # Block tables: [B, num_blocks_per_seq]
    block_tables = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(B, num_blocks_per_seq)

    # Sequence lengths: [B]
    seq_lens_tensor = torch.full((B,), seq_len, dtype=torch.int32, device=device)

    # Workspace buffer for multi-block mode
    num_semaphores = 256
    workspace_buffer = torch.zeros(num_semaphores, 4, dtype=torch.int32, device=device)

    # Scale factor
    sm_scale = 1.0 / (head_dim_qk ** 0.5)

    def kernel_fn():
        trtllm_batch_decode_with_kv_cache_mla(
            query,
            kv_cache,
            workspace_buffer,
            qk_nope_head_dim,
            kv_lora_rank,
            qk_rope_head_dim,
            block_tables,
            seq_lens_tensor,
            seq_len,  # max_seq_len
            sparse_mla_top_k=0,
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

    # Approximate FLOPS for attention
    flops = 4 * B * Nh * seq_len * head_dim_ckv_kpe
    q_bytes = query.numel() * 2
    kv_bytes = kv_cache.numel() * 2
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
        M=B * Nh, N=seq_len, K_dim=head_dim_ckv_kpe,
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
    for B in batch_sizes:
        for seq_len in seq_lens:
            result = bench_trtllm_mla(flashinfer, B, seq_len)
            if result:
                results.append(result)
                print(f"  B={B}, seq_len={seq_len}: {result.latency_ms:.4f} ms, "
                      f"{result.bandwidth_gbs:.1f} GB/s, {result.bound}")

    save_results(results, output_dir, "trtllm_batch_decode_with_kv_cache_mla")


def main():
    parser = argparse.ArgumentParser(description="Benchmark trtllm_mla kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048,4096",
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
