#!/usr/bin/env python3
"""
Benchmark: dsv3_fused_a_gemm (Kernel #4)
Source: sgl-kernel
Category: GEMM (Compute-bound)
Ops: fused_qkv_a_proj (low-latency path for B<=16)

Usage:
    python bench_dsv3_fused_a_gemm.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    H, Lq, Lkv, Dr,
    BenchmarkResult, PEAK_TFLOPS_FP16,
    compute_gemm_flops, compute_gemm_bytes,
    benchmark_kernel, save_results, check_sgl_kernel
)


def bench_dsv3_fused_a_gemm(sgl_kernel, B: int, phase: str,
                            device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark DSV3 fused A GEMM (low-latency path for B<=16)."""
    try:
        from sgl_kernel import dsv3_fused_a_gemm
    except ImportError:
        print("Warning: dsv3_fused_a_gemm not available")
        return None

    M, K, N = B, H, Lq + Lkv + Dr  # 7168 -> 2112

    mat_a = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    mat_b = torch.randn((N, K), dtype=torch.bfloat16, device=device).T

    def kernel_fn():
        dsv3_fused_a_gemm(mat_a, mat_b)

    latency_ms = benchmark_kernel(kernel_fn)

    flops = compute_gemm_flops(M, N, K)
    bytes_transferred = compute_gemm_bytes(M, N, K, dtype_size=2, weight_dtype_size=2)
    gflops = flops / (latency_ms * 1e-3) / 1e9
    tflops = gflops / 1000
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    # BF16 GEMM
    peak_pct = (tflops / PEAK_TFLOPS_FP16) * 100

    return BenchmarkResult(
        kernel="dsv3_fused_a_gemm",
        op="fused_qkv_a_proj",
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
    """Run dsv3_fused_a_gemm benchmarks."""
    sgl_kernel = check_sgl_kernel()
    if not sgl_kernel:
        print("ERROR: sgl_kernel not available")
        return

    results = []

    # This kernel is optimized for B<=16
    print("\n=== Decode Phase (B<=16, low-latency path) ===")
    for B in [b for b in batch_sizes if b <= 16]:
        result = bench_dsv3_fused_a_gemm(sgl_kernel, B, "decode")
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.gflops:.1f} GFLOPS, {result.peak_pct:.2f}% peak")

    save_results(results, output_dir, "dsv3_fused_a_gemm")


def main():
    parser = argparse.ArgumentParser(description="Benchmark dsv3_fused_a_gemm kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16",
                        help="Comma-separated batch sizes (max 16 for this kernel)")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print("=" * 60)
    print("Benchmark: dsv3_fused_a_gemm (Kernel #4)")
    print("=" * 60)
    run_benchmarks(batch_sizes, args.output)


if __name__ == "__main__":
    main()
