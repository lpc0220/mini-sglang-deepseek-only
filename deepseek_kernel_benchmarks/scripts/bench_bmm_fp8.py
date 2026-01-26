#!/usr/bin/env python3
"""
Benchmark: bmm_fp8 (Kernel #6)
Source: sgl-kernel
Category: BMM (Compute-bound)
Ops: q_nope * w_kc, attn * w_vc (MLA latent projections)

Usage:
    python bench_bmm_fp8.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    Nh, Lkv, Dn, Dv,
    BenchmarkResult, PEAK_TFLOPS_FP8,
    compute_bmm_flops,
    benchmark_kernel, save_results, check_sgl_kernel
)


def bench_bmm_fp8(sgl_kernel, B: int, phase: str, op_name: str,
                  batch: int, M: int, N: int, K: int,
                  device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark BMM FP8 kernel."""
    try:
        from sgl_kernel import bmm_fp8
    except ImportError:
        print("Warning: bmm_fp8 not available")
        return None

    # Create FP8 inputs
    a = torch.randn((batch, M, K), dtype=torch.bfloat16, device=device)
    b = torch.randn((batch, K, N), dtype=torch.bfloat16, device=device)

    # Convert to FP8
    a_fp8 = a.to(torch.float8_e4m3fn)
    b_fp8 = b.to(torch.float8_e4m3fn)
    a_scale = torch.ones(batch, dtype=torch.float32, device=device)
    b_scale = torch.ones(batch, dtype=torch.float32, device=device)

    def kernel_fn():
        bmm_fp8(a_fp8, b_fp8, a_scale, b_scale, torch.bfloat16)

    latency_ms = benchmark_kernel(kernel_fn)

    flops = compute_bmm_flops(batch, M, N, K)
    bytes_transferred = batch * (M * K + K * N + M * N)  # FP8 = 1 byte
    gflops = flops / (latency_ms * 1e-3) / 1e9
    tflops = gflops / 1000
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    # FP8 compute
    peak_pct = (tflops / PEAK_TFLOPS_FP8) * 100

    return BenchmarkResult(
        kernel="bmm_fp8",
        op=op_name,
        phase=phase,
        B=B, S=1,
        M=M, N=N, K_dim=K,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound="compute"
    )


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run bmm_fp8 benchmarks."""
    sgl_kernel = check_sgl_kernel()
    if not sgl_kernel:
        print("ERROR: sgl_kernel not available")
        return

    results = []

    # BMM operations in MLA:
    # 1. q_nope * w_kc: [Nh, B, Dn] @ [Nh, Dn, Lkv] -> [Nh, B, Lkv]
    # 2. attn * w_vc: [Nh, B, Lkv] @ [Nh, Lkv, Dv] -> [Nh, B, Dv]

    print("\n=== Decode Phase: q_nope * w_kc ===")
    for B in batch_sizes:
        result = bench_bmm_fp8(sgl_kernel, B, "decode", "q_nope*w_kc", Nh, B, Lkv, Dn)
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.gflops:.1f} GFLOPS, {result.peak_pct:.2f}% peak")

    print("\n=== Decode Phase: attn * w_vc ===")
    for B in batch_sizes:
        result = bench_bmm_fp8(sgl_kernel, B, "decode", "attn*w_vc", Nh, B, Dv, Lkv)
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.gflops:.1f} GFLOPS, {result.peak_pct:.2f}% peak")

    save_results(results, output_dir, "bmm_fp8")


def main():
    parser = argparse.ArgumentParser(description="Benchmark bmm_fp8 kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths (unused for this kernel)")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: bmm_fp8 (Kernel #6)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
