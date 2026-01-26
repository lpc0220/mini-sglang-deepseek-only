#!/usr/bin/env python3
"""
Benchmark: apply_shuffle_mul_sum (Kernel #20)
Source: sgl-kernel
Category: MoE (Memory-bound)
Ops: Scatter and weighted sum of expert outputs

NOTE: This kernel may not have a standalone benchmark in sgl-kernel.
      This file provides a template for when the kernel is exposed.

Usage:
    python bench_apply_shuffle_mul_sum.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    H, E, K,
    BenchmarkResult, PEAK_BANDWIDTH_GBS,
    benchmark_kernel, save_results, check_sgl_kernel
)


def bench_apply_shuffle_mul_sum(sgl_kernel, B: int, S: int, hidden_size: int,
                                 num_experts: int, topk: int, phase: str,
                                 device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark apply_shuffle_mul_sum kernel."""
    try:
        from sgl_kernel import apply_shuffle_mul_sum
    except ImportError:
        print("Warning: apply_shuffle_mul_sum not available (kernel may be internal)")
        return None

    tokens = B * S if phase == "prefill" else B
    total_expert_tokens = tokens * topk

    # Expert outputs: [total_expert_tokens, hidden_size]
    expert_outputs = torch.randn(total_expert_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    # Expert weights: [tokens, topk]
    expert_weights = torch.randn(tokens, topk, dtype=torch.float32, device=device)
    # Token indices for scattering
    token_indices = torch.arange(tokens, dtype=torch.int32, device=device).repeat_interleave(topk)
    # Output: [tokens, hidden_size]
    output = torch.zeros(tokens, hidden_size, dtype=torch.bfloat16, device=device)

    def kernel_fn():
        apply_shuffle_mul_sum(expert_outputs, expert_weights, token_indices, output)

    try:
        latency_ms = benchmark_kernel(kernel_fn)
    except Exception as e:
        print(f"Warning: Kernel failed for B={B}, S={S}: {e}")
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except:
            pass
        return None

    # Memory: read expert_outputs, weights, indices; write output
    bytes_read = expert_outputs.numel() * 2 + expert_weights.numel() * 4 + token_indices.numel() * 4
    bytes_write = output.numel() * 2
    bytes_transferred = bytes_read + bytes_write
    flops = total_expert_tokens * hidden_size * 2  # multiply + accumulate
    gflops = flops / (latency_ms * 1e-3) / 1e9
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    peak_pct = (bandwidth_gbs / PEAK_BANDWIDTH_GBS) * 100

    return BenchmarkResult(
        kernel="apply_shuffle_mul_sum",
        op="experts_scatter",
        phase=phase,
        B=B, S=S,
        M=tokens, N=hidden_size, K_dim=topk,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound="memory"
    )


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run apply_shuffle_mul_sum benchmarks."""
    sgl_kernel = check_sgl_kernel()
    if not sgl_kernel:
        print("ERROR: sgl_kernel not available")
        return

    results = []

    # Decode phase (S=1)
    print("\n=== Decode Phase ===")
    for B in batch_sizes:
        result = bench_apply_shuffle_mul_sum(sgl_kernel, B, 1, H, E, K, "decode")
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    # Prefill phase
    print("\n=== Prefill Phase ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            result = bench_apply_shuffle_mul_sum(sgl_kernel, B, S, H, E, K, "prefill")
            if result:
                results.append(result)
                print(f"  B={B}, S={S}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    if results:
        save_results(results, output_dir, "apply_shuffle_mul_sum")
    else:
        print("\nNo results - kernel not available")


def main():
    parser = argparse.ArgumentParser(description="Benchmark apply_shuffle_mul_sum kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: apply_shuffle_mul_sum (Kernel #20)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
