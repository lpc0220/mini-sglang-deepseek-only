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
    """Benchmark TensorRT-LLM MLA decode kernel."""
    try:
        from flashinfer.triton.mla import trtllm_batch_decode_with_kv_cache_mla
    except ImportError:
        print("Warning: trtllm_batch_decode_with_kv_cache_mla not available")
        return None

    d = Lkv + Dr  # 576
    block_size = 64

    # Setup inputs similar to cutlass_mla_decode
    q_nope = torch.randn(B, Nh, Lkv, dtype=torch.bfloat16, device=device)
    q_rope = torch.randn(B, Nh, Dr, dtype=torch.bfloat16, device=device)

    # Paged KV cache
    num_blocks = (seq_len + block_size - 1) // block_size
    kv_cache = torch.randn(B * num_blocks, block_size, d, dtype=torch.bfloat16, device=device)
    block_table = torch.arange(B * num_blocks, dtype=torch.int32, device=device).reshape(B, num_blocks)
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=device)

    def kernel_fn():
        trtllm_batch_decode_with_kv_cache_mla(
            q_nope, q_rope, kv_cache, block_table, seq_lens, 1.0 / (d ** 0.5)
        )

    latency_ms = benchmark_kernel(kernel_fn)

    # Approximate FLOPS for attention
    flops = 4 * B * Nh * seq_len * d
    q_bytes = (q_nope.numel() + q_rope.numel()) * 2
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
        M=B * Nh, N=seq_len, K_dim=d,
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
