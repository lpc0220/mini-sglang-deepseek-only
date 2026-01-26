#!/usr/bin/env python3
"""
Benchmark: cutlass_fp4_group_mm (Kernel #19)
Source: sgl-kernel
Category: MoE (Compute-bound)
Ops: experts (gate_up), experts (down) - grouped GEMM for MoE

Usage:
    python bench_cutlass_fp4_group_mm.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    H, E, K, I,
    BenchmarkResult, PEAK_TFLOPS_FP4,
    compute_gemm_flops,
    benchmark_kernel, save_results, check_sgl_kernel
)


def bench_cutlass_fp4_group_mm(sgl_kernel, B: int, S: int, num_experts: int,
                                topk: int, hidden_size: int, intermediate_size: int,
                                op_name: str, phase: str,
                                device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark cutlass_fp4_group_mm kernel for MoE experts."""
    try:
        from sgl_kernel import cutlass_fp4_group_mm, scaled_fp4_quant
    except ImportError:
        print("Warning: cutlass_fp4_group_mm not available")
        return None

    tokens = B * S if phase == "prefill" else B
    total_expert_tokens = tokens * topk

    FLOAT4_E2M1_MAX = 6.0
    FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

    # For gate_up: [total_expert_tokens, hidden_size] x [num_experts, hidden_size, 2*intermediate_size]
    # For down: [total_expert_tokens, intermediate_size] x [num_experts, intermediate_size, hidden_size]

    if op_name == "gate_up":
        M, K_dim, N = total_expert_tokens, hidden_size, 2 * intermediate_size
    else:  # down
        M, K_dim, N = total_expert_tokens, intermediate_size, hidden_size

    # Simplified: benchmark single group GEMM (actual kernel handles multiple experts)
    a = torch.randn((M, K_dim), dtype=torch.bfloat16, device=device)
    b = torch.randn((N, K_dim), dtype=torch.bfloat16, device=device)

    a_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a.flatten())
    b_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b.flatten())
    alpha = 1.0 / (a_global_scale.to(torch.float32) * b_global_scale.to(torch.float32))

    a_fp4, a_scale = scaled_fp4_quant(a, a_global_scale.to(torch.float32))
    b_fp4, b_scale = scaled_fp4_quant(b, b_global_scale.to(torch.float32))

    # Expert assignment (which expert handles which token)
    expert_ids = torch.randint(0, num_experts, (total_expert_tokens,), dtype=torch.int32, device=device)

    def kernel_fn():
        # Note: actual API may differ
        cutlass_fp4_group_mm(a_fp4, b_fp4, a_scale, b_scale, alpha, expert_ids, torch.bfloat16)

    try:
        latency_ms = benchmark_kernel(kernel_fn)
    except Exception as e:
        print(f"Warning: Kernel failed for B={B}, S={S}, op={op_name}: {e}")
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except:
            pass
        return None

    flops = compute_gemm_flops(M, N, K_dim)
    bytes_transferred = int(M * K_dim * 0.5 + K_dim * N * 0.5 + M * N * 2)  # FP4 inputs, bf16 output
    gflops = flops / (latency_ms * 1e-3) / 1e9
    tflops = gflops / 1000
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    peak_pct = (tflops / PEAK_TFLOPS_FP4) * 100

    return BenchmarkResult(
        kernel="cutlass_fp4_group_mm",
        op=f"experts_{op_name}",
        phase=phase,
        B=B, S=S,
        M=M, N=N, K_dim=K_dim,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound="compute"
    )


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run cutlass_fp4_group_mm benchmarks."""
    sgl_kernel = check_sgl_kernel()
    if not sgl_kernel:
        print("ERROR: sgl_kernel not available")
        return

    results = []

    # Decode phase (S=1)
    print("\n=== Decode Phase: gate_up ===")
    for B in batch_sizes:
        result = bench_cutlass_fp4_group_mm(sgl_kernel, B, 1, E, K, H, I, "gate_up", "decode")
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.gflops:.1f} GFLOPS, {result.peak_pct:.2f}% peak")

    print("\n=== Decode Phase: down ===")
    for B in batch_sizes:
        result = bench_cutlass_fp4_group_mm(sgl_kernel, B, 1, E, K, H, I, "down", "decode")
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.gflops:.1f} GFLOPS, {result.peak_pct:.2f}% peak")

    # Prefill phase
    print("\n=== Prefill Phase: gate_up ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            result = bench_cutlass_fp4_group_mm(sgl_kernel, B, S, E, K, H, I, "gate_up", "prefill")
            if result:
                results.append(result)
                print(f"  B={B}, S={S}: {result.latency_ms:.4f} ms, {result.gflops:.1f} GFLOPS, {result.peak_pct:.2f}% peak")

    print("\n=== Prefill Phase: down ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            result = bench_cutlass_fp4_group_mm(sgl_kernel, B, S, E, K, H, I, "down", "prefill")
            if result:
                results.append(result)
                print(f"  B={B}, S={S}: {result.latency_ms:.4f} ms, {result.gflops:.1f} GFLOPS, {result.peak_pct:.2f}% peak")

    if results:
        save_results(results, output_dir, "cutlass_fp4_group_mm")
    else:
        print("\nNo results - kernel not available")


def main():
    parser = argparse.ArgumentParser(description="Benchmark cutlass_fp4_group_mm kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: cutlass_fp4_group_mm (Kernel #19)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
