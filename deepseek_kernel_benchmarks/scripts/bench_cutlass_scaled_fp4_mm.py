#!/usr/bin/env python3
"""
Benchmark: cutlass_scaled_fp4_mm (Kernel #3)
Source: sgl-kernel
Category: GEMM (Compute-bound)
Ops: q_b_proj, kv_b_proj, o_proj, gate_proj, up_proj, down_proj (B>16)

Usage:
    python bench_cutlass_scaled_fp4_mm.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    H, Nh, Lq, Lkv, Dn, Dr, Dv, Dq, I,
    BenchmarkResult, PEAK_TFLOPS_FP4,
    compute_gemm_flops, compute_gemm_bytes,
    benchmark_kernel, save_results, check_sgl_kernel
)


def bench_cutlass_fp4_gemm(sgl_kernel, B: int, S: int, M: int, N: int, K: int,
                           op_name: str, phase: str, device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark Cutlass FP4 GEMM kernel."""
    try:
        from sgl_kernel import cutlass_scaled_fp4_mm, scaled_fp4_quant
    except ImportError:
        print("Warning: cutlass_scaled_fp4_mm not available")
        return None

    FLOAT4_E2M1_MAX = 6.0
    FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

    dtype = torch.bfloat16
    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((N, K), dtype=dtype, device=device)

    a_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a.flatten())
    b_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b.flatten())
    alpha = 1.0 / (a_global_scale.to(torch.float32) * b_global_scale.to(torch.float32))

    a_fp4, a_scale = scaled_fp4_quant(a, a_global_scale.to(torch.float32))
    b_fp4, b_scale = scaled_fp4_quant(b, b_global_scale.to(torch.float32))

    def kernel_fn():
        cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_scale, b_scale, alpha, dtype)

    latency_ms = benchmark_kernel(kernel_fn)

    flops = compute_gemm_flops(M, N, K)
    bytes_transferred = compute_gemm_bytes(M, N, K, dtype_size=2, weight_dtype_size=0.5)
    gflops = flops / (latency_ms * 1e-3) / 1e9
    tflops = gflops / 1000
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    # FP4 GEMM - compute bound
    peak_pct = (tflops / PEAK_TFLOPS_FP4) * 100

    return BenchmarkResult(
        kernel="cutlass_scaled_fp4_mm",
        op=op_name,
        phase=phase,
        B=B, S=S,
        M=M, N=N, K_dim=K,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound="compute"
    )


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run cutlass_scaled_fp4_mm benchmarks."""
    sgl_kernel = check_sgl_kernel()
    if not sgl_kernel:
        print("ERROR: sgl_kernel not available")
        return

    results = []

    # Define GEMM shapes for different ops
    # q_b_proj: [B, Lq] x [Lq, Nh*Dq] -> [B, Nh*Dq]
    # kv_b_proj: [B, Lkv] x [Lkv, Nh*(Dn+Dv)] -> [B, Nh*(Dn+Dv)]
    # o_proj: [B, Nh*Dv] x [Nh*Dv, H] -> [B, H]

    ops = [
        ("q_b_proj", Nh * Dq, Lq),
        ("kv_b_proj", Nh * (Dn + Dv), Lkv),
        ("o_proj", H, Nh * Dv),
    ]

    # Decode phase (S=1)
    print("\n=== Decode Phase ===")
    for B in batch_sizes:
        for op_name, N, K in ops:
            M = B
            result = bench_cutlass_fp4_gemm(sgl_kernel, B, 1, M, N, K, op_name, "decode")
            if result:
                results.append(result)
                print(f"  {op_name} B={B}: {result.latency_ms:.4f} ms, {result.gflops:.1f} GFLOPS, {result.peak_pct:.2f}% peak")

    # Prefill phase
    print("\n=== Prefill Phase ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            tokens = B * S
            for op_name, N, K in ops:
                M = tokens
                result = bench_cutlass_fp4_gemm(sgl_kernel, B, S, M, N, K, op_name, "prefill")
                if result:
                    results.append(result)
                    print(f"  {op_name} B={B}, S={S}: {result.latency_ms:.4f} ms, {result.gflops:.1f} GFLOPS, {result.peak_pct:.2f}% peak")

    save_results(results, output_dir, "cutlass_scaled_fp4_mm")


def main():
    parser = argparse.ArgumentParser(description="Benchmark cutlass_scaled_fp4_mm kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: cutlass_scaled_fp4_mm (Kernel #3)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
