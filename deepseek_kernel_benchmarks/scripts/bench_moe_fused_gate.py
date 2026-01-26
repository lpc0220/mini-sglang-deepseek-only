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
    E, K,
    BenchmarkResult, PEAK_BANDWIDTH_GBS,
    benchmark_kernel, save_results, check_sgl_kernel
)


def bench_moe_fused_gate(sgl_kernel, B: int, S: int, hidden_size: int,
                          num_experts: int, topk: int, phase: str,
                          device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark moe_fused_gate kernel.

    API from sgl_kernel/moe.py:
        moe_fused_gate(
            input_tensor,         # Gating output [num_tokens, num_experts]
            bias,                 # Correction bias [num_experts]
            num_expert_group,     # Number of expert groups (DeepSeek V3 uses 8)
            topk_group,           # Top-k groups to select (DeepSeek V3 uses 4)
            topk,                 # Total experts to select (DeepSeek V3 uses 8)
            num_fused_shared_experts=0,
            routed_scaling_factor=0,
            apply_routed_scaling_factor_on_output=False,
        )

    DeepSeek V3/R1: 256 experts, 8 groups, topk_group=4, topk=8
    """
    try:
        from sgl_kernel import moe_fused_gate
    except ImportError:
        print("Warning: moe_fused_gate not available")
        return None

    tokens = B * S if phase == "prefill" else B

    # DeepSeek V3 MoE parameters
    num_expert_group = 8   # Number of expert groups
    topk_group = 4         # Select top-4 groups

    # Input: gating output [tokens, num_experts]
    input_tensor = torch.randn(tokens, num_experts, dtype=torch.float32, device=device)
    # Bias: correction bias [num_experts]
    bias = torch.randn(num_experts, dtype=torch.float32, device=device)

    def kernel_fn():
        moe_fused_gate(input_tensor, bias, num_expert_group, topk_group, topk)

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

    # Memory: read input_tensor, bias; write routing outputs
    bytes_read = input_tensor.numel() * 4 + bias.numel() * 4  # float32
    bytes_write = tokens * topk * (4 + 4)  # indices + weights
    bytes_transferred = bytes_read + bytes_write
    flops = tokens * num_experts * 5  # sigmoid + topk ops (simplified)
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
        # hidden_size not used in fused gate (input is gating output)
        result = bench_moe_fused_gate(sgl_kernel, B, 1, 0, E, K, "decode")
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    # Prefill phase
    print("\n=== Prefill Phase ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            result = bench_moe_fused_gate(sgl_kernel, B, S, 0, E, K, "prefill")
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
