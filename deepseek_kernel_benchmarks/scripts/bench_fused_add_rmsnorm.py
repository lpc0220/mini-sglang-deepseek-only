#!/usr/bin/env python3
"""
Benchmark: fused_add_rmsnorm (Kernel #2)
Source: sgl-kernel
Category: Normalization (Memory-bound)
Ops: input_layernorm + residual, post_attention_layernorm + residual

Usage:
    python bench_fused_add_rmsnorm.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    H, BenchmarkResult, PEAK_BANDWIDTH_GBS,
    compute_norm_flops, compute_norm_bytes,
    benchmark_kernel, save_results, check_sgl_kernel
)


def bench_fused_add_rmsnorm(sgl_kernel, B: int, S: int, hidden_size: int, phase: str,
                             device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark fused add + RMSNorm kernel."""
    tokens = B * S if phase == "prefill" else B
    x = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device=device)
    residual = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device=device)
    weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)

    def kernel_fn():
        sgl_kernel.fused_add_rmsnorm(x, residual, weight, 1e-6)

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

    # Fused: read x, read residual, write x, write residual
    flops = compute_norm_flops(tokens * hidden_size) + tokens * hidden_size  # add + norm
    bytes_transferred = 4 * tokens * hidden_size * 2  # 4 tensors, bf16
    gflops = flops / (latency_ms * 1e-3) / 1e9
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    # Memory-bound
    peak_pct = (bandwidth_gbs / PEAK_BANDWIDTH_GBS) * 100

    return BenchmarkResult(
        kernel="fused_add_rmsnorm",
        op="post_attention_layernorm",
        phase=phase,
        B=B, S=S,
        M=tokens, N=hidden_size, K_dim=0,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound="memory"
    )


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run fused_add_rmsnorm benchmarks."""
    sgl_kernel = check_sgl_kernel()
    if not sgl_kernel:
        print("ERROR: sgl_kernel not available")
        return

    results = []

    # Decode phase (S=1)
    print("\n=== Decode Phase ===")
    for B in batch_sizes:
        result = bench_fused_add_rmsnorm(sgl_kernel, B, 1, H, "decode")
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    # Prefill phase
    print("\n=== Prefill Phase ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            result = bench_fused_add_rmsnorm(sgl_kernel, B, S, H, "prefill")
            if result:
                results.append(result)
                print(f"  B={B}, S={S}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    save_results(results, output_dir, "fused_add_rmsnorm")


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused_add_rmsnorm kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: fused_add_rmsnorm (Kernel #2)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
