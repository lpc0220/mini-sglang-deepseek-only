#!/usr/bin/env python3
"""
Benchmark: apply_rope_with_cos_sin_cache_inplace (Kernel #11)
Source: sgl-kernel
Category: RoPE (Memory-bound)
Ops: rotary_emb (apply rotary position embeddings)

Usage:
    python bench_apply_rope.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    Nh, Dr,
    BenchmarkResult, PEAK_BANDWIDTH_GBS,
    benchmark_kernel, save_results, check_sgl_kernel
)


def bench_apply_rope(sgl_kernel, B: int, S: int,
                     device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark apply_rope_with_cos_sin_cache_inplace kernel."""
    try:
        from sgl_kernel import apply_rope_with_cos_sin_cache_inplace
    except ImportError:
        print("Warning: apply_rope_with_cos_sin_cache_inplace not available")
        return None

    tokens = B * S
    head_dim = Dr  # 64

    # Query tensor: [tokens, Nh, head_dim]
    q = torch.randn(tokens, Nh, head_dim, dtype=torch.bfloat16, device=device)
    # Key tensor: [tokens, Nh, head_dim] (for standard attention, but MLA uses compressed)
    k = torch.randn(tokens, Nh, head_dim, dtype=torch.bfloat16, device=device)

    # RoPE cos/sin cache
    max_seq_len = max(S, 2048)
    cos_cache = torch.randn(max_seq_len, head_dim // 2, dtype=torch.bfloat16, device=device)
    sin_cache = torch.randn(max_seq_len, head_dim // 2, dtype=torch.bfloat16, device=device)
    positions = torch.arange(tokens, dtype=torch.int64, device=device) % S

    def kernel_fn():
        apply_rope_with_cos_sin_cache_inplace(q, k, cos_cache, sin_cache, positions)

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

    # Memory-bound: read q, k, cos, sin, positions; write q, k
    bytes_read = (q.numel() + k.numel()) * 2 + (cos_cache.numel() + sin_cache.numel()) * 2 + positions.numel() * 8
    bytes_write = (q.numel() + k.numel()) * 2
    bytes_transferred = bytes_read + bytes_write
    flops = tokens * Nh * head_dim * 4  # RoPE ops
    gflops = flops / (latency_ms * 1e-3) / 1e9
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    peak_pct = (bandwidth_gbs / PEAK_BANDWIDTH_GBS) * 100

    return BenchmarkResult(
        kernel="apply_rope_with_cos_sin_cache_inplace",
        op="rotary_emb",
        phase="prefill" if S > 1 else "decode",
        B=B, S=S,
        M=tokens, N=Nh, K_dim=head_dim,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound="memory"
    )


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run apply_rope benchmarks."""
    sgl_kernel = check_sgl_kernel()
    if not sgl_kernel:
        print("ERROR: sgl_kernel not available")
        return

    results = []

    # Decode phase (S=1)
    print("\n=== Decode Phase ===")
    for B in batch_sizes:
        result = bench_apply_rope(sgl_kernel, B, 1)
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    # Prefill phase
    print("\n=== Prefill Phase ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            result = bench_apply_rope(sgl_kernel, B, S)
            if result:
                results.append(result)
                print(f"  B={B}, S={S}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    save_results(results, output_dir, "apply_rope_with_cos_sin_cache_inplace")


def main():
    parser = argparse.ArgumentParser(description="Benchmark apply_rope kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: apply_rope_with_cos_sin_cache_inplace (Kernel #11)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
