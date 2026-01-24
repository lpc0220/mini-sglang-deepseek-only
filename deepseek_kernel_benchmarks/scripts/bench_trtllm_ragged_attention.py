#!/usr/bin/env python3
"""
Benchmark: trtllm_ragged_attention_deepseek (Kernel #9)
Source: flashinfer (TensorRT-LLM kernel)
Category: Attention (Mixed bound)
Ops: Prefill attention with ragged batching

Usage:
    python bench_trtllm_ragged_attention.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    Nh, Lkv, Dr, Dv,
    BenchmarkResult, PEAK_BANDWIDTH_GBS, PEAK_TFLOPS_FP16, RIDGE_FP16,
    benchmark_kernel, save_results, check_flashinfer
)


def bench_trtllm_ragged_attention(flashinfer, B: int, S: int,
                                   device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark TensorRT-LLM ragged attention for DeepSeek."""
    try:
        from flashinfer.triton.mla import trtllm_ragged_attention_deepseek
    except ImportError:
        print("Warning: trtllm_ragged_attention_deepseek not available")
        return None

    d = Lkv + Dr  # 576
    tokens = B * S

    # Setup inputs
    q_nope = torch.randn(tokens, Nh, Lkv, dtype=torch.bfloat16, device=device)
    q_rope = torch.randn(tokens, Nh, Dr, dtype=torch.bfloat16, device=device)
    kv_cache = torch.randn(tokens, d, dtype=torch.bfloat16, device=device)

    # Ragged batch info
    cu_seqlens = torch.arange(0, tokens + 1, S, dtype=torch.int32, device=device)

    def kernel_fn():
        trtllm_ragged_attention_deepseek(
            q_nope, q_rope, kv_cache, cu_seqlens, 1.0 / (d ** 0.5)
        )

    latency_ms = benchmark_kernel(kernel_fn)

    # Approximate FLOPS for attention: 4 * tokens * seq_len * d (simplified)
    flops = 4 * tokens * S * d
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
        kernel="trtllm_ragged_attention_deepseek",
        op="prefill_attn",
        phase="prefill",
        B=B, S=S,
        M=tokens * Nh, N=S, K_dim=d,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound=bound
    )


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run trtllm_ragged_attention benchmarks."""
    flashinfer = check_flashinfer()
    if not flashinfer:
        print("ERROR: flashinfer not available")
        return

    results = []

    print("\n=== Prefill Phase (TRT-LLM Ragged Attention) ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            result = bench_trtllm_ragged_attention(flashinfer, B, S)
            if result:
                results.append(result)
                print(f"  B={B}, S={S}: {result.latency_ms:.4f} ms, "
                      f"{result.gflops:.1f} GFLOPS, {result.bound}")

    save_results(results, output_dir, "trtllm_ragged_attention_deepseek")


def main():
    parser = argparse.ArgumentParser(description="Benchmark trtllm_ragged_attention kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: trtllm_ragged_attention_deepseek (Kernel #9)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
