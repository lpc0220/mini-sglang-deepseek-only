#!/usr/bin/env python3
"""
Benchmark: silu_and_mul (Kernel #13)
Source: sgl-kernel
Category: Activation (Memory-bound)
Ops: act_fn (SiLU gating for MoE and FFN)

Usage:
    python bench_silu_and_mul.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    I,
    BenchmarkResult, PEAK_BANDWIDTH_GBS,
    compute_activation_flops, compute_activation_bytes,
    benchmark_kernel, save_results, check_sgl_kernel
)


def bench_silu_and_mul(sgl_kernel, B: int, S: int, dim: int, phase: str,
                       device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark SiLU activation kernel."""
    tokens = B * S if phase == "prefill" else B
    # Input is [tokens, 2*dim] for gate_proj and up_proj outputs
    x = torch.randn(tokens, 2 * dim, dtype=torch.bfloat16, device=device)

    def kernel_fn():
        sgl_kernel.silu_and_mul(x)

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

    flops = compute_activation_flops(tokens * dim)
    bytes_transferred = compute_activation_bytes(tokens * dim)
    gflops = flops / (latency_ms * 1e-3) / 1e9
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    # Activation is memory-bound
    peak_pct = (bandwidth_gbs / PEAK_BANDWIDTH_GBS) * 100

    return BenchmarkResult(
        kernel="silu_and_mul",
        op="act_fn",
        phase=phase,
        B=B, S=S,
        M=tokens, N=dim, K_dim=0,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound="memory"
    )


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run silu_and_mul benchmarks."""
    sgl_kernel = check_sgl_kernel()
    if not sgl_kernel:
        print("ERROR: sgl_kernel not available")
        return

    results = []

    # Decode phase (S=1)
    print("\n=== Decode Phase ===")
    for B in batch_sizes:
        result = bench_silu_and_mul(sgl_kernel, B, 1, I, "decode")
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    # Prefill phase
    print("\n=== Prefill Phase ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            result = bench_silu_and_mul(sgl_kernel, B, S, I, "prefill")
            if result:
                results.append(result)
                print(f"  B={B}, S={S}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    save_results(results, output_dir, "silu_and_mul")


def main():
    parser = argparse.ArgumentParser(description="Benchmark silu_and_mul kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: silu_and_mul (Kernel #13)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
