#!/usr/bin/env python3
"""
Benchmark: concat_mla_mha_k (Kernel #12)
Source: sgl-kernel
Category: Concat (Memory-bound)
Ops: Concatenate MLA latent with RoPE keys

Usage:
    python bench_concat_mla_mha_k.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    Nh, Lkv, Dr,
    BenchmarkResult, PEAK_BANDWIDTH_GBS,
    benchmark_kernel, save_results, check_sgl_kernel
)


def bench_concat_mla_mha_k(sgl_kernel, B: int, S: int,
                           device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark concat_mla_mha_k kernel."""
    try:
        from sgl_kernel import concat_mla_mha_k
    except ImportError:
        print("Warning: concat_mla_mha_k not available")
        return None

    tokens = B * S
    d = Lkv + Dr  # 576

    # MLA latent: [tokens, Lkv]
    mla_latent = torch.randn(tokens, Lkv, dtype=torch.bfloat16, device=device)
    # RoPE keys: [tokens, Dr]
    rope_k = torch.randn(tokens, Dr, dtype=torch.bfloat16, device=device)
    # Output: [tokens, d]
    output = torch.empty(tokens, d, dtype=torch.bfloat16, device=device)

    def kernel_fn():
        concat_mla_mha_k(mla_latent, rope_k, output)

    latency_ms = benchmark_kernel(kernel_fn)

    # Pure memory copy
    bytes_read = (mla_latent.numel() + rope_k.numel()) * 2
    bytes_write = output.numel() * 2
    bytes_transferred = bytes_read + bytes_write
    flops = 0  # Just memory copy
    gflops = 0
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = 0

    peak_pct = (bandwidth_gbs / PEAK_BANDWIDTH_GBS) * 100

    return BenchmarkResult(
        kernel="concat_mla_mha_k",
        op="concat_kv",
        phase="prefill" if S > 1 else "decode",
        B=B, S=S,
        M=tokens, N=d, K_dim=0,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound="memory"
    )


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run concat_mla_mha_k benchmarks."""
    sgl_kernel = check_sgl_kernel()
    if not sgl_kernel:
        print("ERROR: sgl_kernel not available")
        return

    results = []

    # Decode phase (S=1)
    print("\n=== Decode Phase ===")
    for B in batch_sizes:
        result = bench_concat_mla_mha_k(sgl_kernel, B, 1)
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    # Prefill phase
    print("\n=== Prefill Phase ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            result = bench_concat_mla_mha_k(sgl_kernel, B, S)
            if result:
                results.append(result)
                print(f"  B={B}, S={S}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    save_results(results, output_dir, "concat_mla_mha_k")


def main():
    parser = argparse.ArgumentParser(description="Benchmark concat_mla_mha_k kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: concat_mla_mha_k (Kernel #12)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
