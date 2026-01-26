#!/usr/bin/env python3
"""
Benchmark: prepare_moe_input (Kernel #17)
Source: sgl-kernel
Category: MoE (Memory-bound)
Ops: Prepare input for MoE experts (gather tokens by expert)

NOTE: This kernel may not have a standalone benchmark in sgl-kernel.
      This file provides a template for when the kernel is exposed.

Usage:
    python bench_prepare_moe_input.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    H, E, K,
    BenchmarkResult, PEAK_BANDWIDTH_GBS,
    benchmark_kernel, save_results, check_sgl_kernel
)


def bench_prepare_moe_input(sgl_kernel, B: int, S: int, hidden_size: int,
                            num_experts: int, topk: int, phase: str,
                            device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark prepare_moe_input kernel."""
    try:
        from sgl_kernel import prepare_moe_input
    except ImportError:
        print("Warning: prepare_moe_input not available (kernel may be internal)")
        return None

    tokens = B * S if phase == "prefill" else B
    # Hidden states: [tokens, hidden_size]
    hidden_states = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device=device)
    # Expert indices: [tokens, topk]
    expert_indices = torch.randint(0, num_experts, (tokens, topk), dtype=torch.int32, device=device)
    # Expert weights: [tokens, topk]
    expert_weights = torch.randn(tokens, topk, dtype=torch.float32, device=device)

    def kernel_fn():
        prepare_moe_input(hidden_states, expert_indices, expert_weights)

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

    # Memory: read hidden_states, indices, weights; write gathered inputs
    bytes_read = hidden_states.numel() * 2 + expert_indices.numel() * 4 + expert_weights.numel() * 4
    bytes_write = tokens * topk * hidden_size * 2  # gathered inputs
    bytes_transferred = bytes_read + bytes_write
    flops = 0  # Pure gather operation
    gflops = 0
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = 0

    peak_pct = (bandwidth_gbs / PEAK_BANDWIDTH_GBS) * 100

    return BenchmarkResult(
        kernel="prepare_moe_input",
        op="experts_gather",
        phase=phase,
        B=B, S=S,
        M=tokens, N=topk, K_dim=hidden_size,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound="memory"
    )


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run prepare_moe_input benchmarks."""
    sgl_kernel = check_sgl_kernel()
    if not sgl_kernel:
        print("ERROR: sgl_kernel not available")
        return

    results = []

    # Decode phase (S=1)
    print("\n=== Decode Phase ===")
    for B in batch_sizes:
        result = bench_prepare_moe_input(sgl_kernel, B, 1, H, E, K, "decode")
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    # Prefill phase
    print("\n=== Prefill Phase ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            result = bench_prepare_moe_input(sgl_kernel, B, S, H, E, K, "prefill")
            if result:
                results.append(result)
                print(f"  B={B}, S={S}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    if results:
        save_results(results, output_dir, "prepare_moe_input")
    else:
        print("\nNo results - kernel not available")


def main():
    parser = argparse.ArgumentParser(description="Benchmark prepare_moe_input kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: prepare_moe_input (Kernel #17)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
