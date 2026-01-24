#!/usr/bin/env python3
"""
Benchmark: moe_fused_gate (Kernel #16)
Source: sgl-kernel
Category: MoE Routing (Memory-bound)
Ops: Fused gate computation for MoE

Usage:
    python bench_moe_fused_gate.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    H, E, K,
    BenchmarkResult, PEAK_BANDWIDTH_GBS,
    benchmark_kernel, save_results, check_sgl_kernel
)


def bench_moe_fused_gate(sgl_kernel, B: int, S: int, hidden_size: int,
                          num_experts: int, topk: int, phase: str,
                          device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark moe_fused_gate kernel."""
    try:
        from sgl_kernel import moe_fused_gate
    except ImportError:
        print("Warning: moe_fused_gate not available")
        return None

    tokens = B * S if phase == "prefill" else B
    # Hidden states: [tokens, hidden_size]
    hidden_states = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device=device)
    # Gate weight: [num_experts, hidden_size]
    gate_weight = torch.randn(num_experts, hidden_size, dtype=torch.bfloat16, device=device)

    def kernel_fn():
        moe_fused_gate(hidden_states, gate_weight, topk)

    latency_ms = benchmark_kernel(kernel_fn)

    # Memory: read hidden_states, gate_weight; write routing outputs
    bytes_read = (hidden_states.numel() + gate_weight.numel()) * 2
    bytes_write = tokens * topk * (4 + 4)  # indices + weights
    bytes_transferred = bytes_read + bytes_write
    flops = 2 * tokens * num_experts * hidden_size  # GEMM for gate
    gflops = flops / (latency_ms * 1e-3) / 1e9
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    peak_pct = (bandwidth_gbs / PEAK_BANDWIDTH_GBS) * 100

    return BenchmarkResult(
        kernel="moe_fused_gate",
        op="fused_gate",
        phase=phase,
        B=B, S=S,
        M=tokens, N=num_experts, K_dim=hidden_size,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound="memory"
    )


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run moe_fused_gate benchmarks."""
    sgl_kernel = check_sgl_kernel()
    if not sgl_kernel:
        print("ERROR: sgl_kernel not available")
        return

    results = []

    # Decode phase (S=1)
    print("\n=== Decode Phase ===")
    for B in batch_sizes:
        result = bench_moe_fused_gate(sgl_kernel, B, 1, H, E, K, "decode")
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    # Prefill phase
    print("\n=== Prefill Phase ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            result = bench_moe_fused_gate(sgl_kernel, B, S, H, E, K, "prefill")
            if result:
                results.append(result)
                print(f"  B={B}, S={S}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    save_results(results, output_dir, "moe_fused_gate")


def main():
    parser = argparse.ArgumentParser(description="Benchmark moe_fused_gate kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: moe_fused_gate (Kernel #16)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
