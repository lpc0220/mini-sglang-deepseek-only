#!/usr/bin/env python3
"""
Benchmark: dsv3_router_gemm (Kernel #5)
Source: sgl-kernel
Category: GEMM (Compute-bound)
Ops: gate (router) for MoE

Usage:
    python bench_dsv3_router_gemm.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    H, E,
    BenchmarkResult, PEAK_TFLOPS_FP16,
    compute_gemm_flops, compute_gemm_bytes,
    benchmark_kernel, save_results, check_sgl_kernel
)


def bench_dsv3_router_gemm(sgl_kernel, B: int, phase: str,
                           device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark DSV3 router GEMM (gate for MoE)."""
    try:
        from sgl_kernel import dsv3_router_gemm
    except ImportError:
        print("Warning: dsv3_router_gemm not available")
        return None

    M, K, N = B, H, E  # 7168 -> 256

    mat_a = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    mat_b = torch.randn((N, K), dtype=torch.bfloat16, device=device).T

    def kernel_fn():
        dsv3_router_gemm(mat_a, mat_b)

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
        kernel="dsv3_router_gemm",
        op="gate",
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
    """Run dsv3_router_gemm benchmarks."""
    sgl_kernel = check_sgl_kernel()
    if not sgl_kernel:
        print("ERROR: sgl_kernel not available")
        return

    results = []

    # Decode phase (S=1)
    print("\n=== Decode Phase ===")
    for B in batch_sizes:
        result = bench_dsv3_router_gemm(sgl_kernel, B, "decode")
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.gflops:.1f} GFLOPS, {result.peak_pct:.2f}% peak")

    # Prefill phase
    print("\n=== Prefill Phase ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            tokens = B * S
            result = bench_dsv3_router_gemm(sgl_kernel, tokens, "prefill")
            if result:
                result.B = B
                result.S = S
                results.append(result)
                print(f"  B={B}, S={S}: {result.latency_ms:.4f} ms, {result.gflops:.1f} GFLOPS, {result.peak_pct:.2f}% peak")

    save_results(results, output_dir, "dsv3_router_gemm")


def main():
    parser = argparse.ArgumentParser(description="Benchmark dsv3_router_gemm kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: dsv3_router_gemm (Kernel #5)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
