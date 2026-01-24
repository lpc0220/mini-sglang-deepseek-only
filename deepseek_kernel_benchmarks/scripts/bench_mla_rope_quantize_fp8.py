#!/usr/bin/env python3
"""
Benchmark: mla_rope_quantize_fp8 (Kernel #10)
Source: flashinfer
Category: Attention/RoPE (Memory-bound)
Ops: Fused RoPE + FP8 quantization for MLA

Usage:
    python bench_mla_rope_quantize_fp8.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    Nh, Lkv, Dr,
    BenchmarkResult, PEAK_BANDWIDTH_GBS,
    benchmark_kernel, save_results, check_flashinfer
)


def bench_mla_rope_quantize_fp8(flashinfer, B: int, S: int,
                                 device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark MLA RoPE + FP8 quantization kernel."""
    try:
        from flashinfer.triton.mla import mla_rope_quantize_fp8
    except ImportError:
        print("Warning: mla_rope_quantize_fp8 not available")
        return None

    tokens = B * S
    d = Lkv + Dr  # 576

    # Input: q_rope before RoPE
    q_rope = torch.randn(tokens, Nh, Dr, dtype=torch.bfloat16, device=device)
    kv_latent = torch.randn(tokens, d, dtype=torch.bfloat16, device=device)

    # RoPE cos/sin cache
    cos_cache = torch.randn(S, Dr // 2, dtype=torch.bfloat16, device=device)
    sin_cache = torch.randn(S, Dr // 2, dtype=torch.bfloat16, device=device)
    positions = torch.arange(S, dtype=torch.int32, device=device).repeat(B)

    def kernel_fn():
        mla_rope_quantize_fp8(q_rope, kv_latent, cos_cache, sin_cache, positions)

    latency_ms = benchmark_kernel(kernel_fn)

    # Memory-bound: read q_rope, kv_latent, cos, sin; write quantized outputs
    bytes_read = (q_rope.numel() + kv_latent.numel() + cos_cache.numel() + sin_cache.numel()) * 2
    bytes_write = (tokens * Nh * Dr + tokens * d)  # FP8 outputs = 1 byte each
    bytes_transferred = bytes_read + bytes_write
    flops = tokens * (Nh * Dr * 4 + d)  # RoPE ops
    gflops = flops / (latency_ms * 1e-3) / 1e9
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    peak_pct = (bandwidth_gbs / PEAK_BANDWIDTH_GBS) * 100

    return BenchmarkResult(
        kernel="mla_rope_quantize_fp8",
        op="rope_quantize",
        phase="prefill" if S > 1 else "decode",
        B=B, S=S,
        M=tokens, N=Dr, K_dim=d,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound="memory"
    )


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run mla_rope_quantize_fp8 benchmarks."""
    flashinfer = check_flashinfer()
    if not flashinfer:
        print("ERROR: flashinfer not available")
        return

    results = []

    # Decode phase (S=1)
    print("\n=== Decode Phase ===")
    for B in batch_sizes:
        result = bench_mla_rope_quantize_fp8(flashinfer, B, 1)
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    # Prefill phase
    print("\n=== Prefill Phase ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            result = bench_mla_rope_quantize_fp8(flashinfer, B, S)
            if result:
                results.append(result)
                print(f"  B={B}, S={S}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    save_results(results, output_dir, "mla_rope_quantize_fp8")


def main():
    parser = argparse.ArgumentParser(description="Benchmark mla_rope_quantize_fp8 kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: mla_rope_quantize_fp8 (Kernel #10)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
