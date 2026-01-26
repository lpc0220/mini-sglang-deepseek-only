#!/usr/bin/env python3
"""
Benchmark: moe_align_block_size (Kernel #21)
Source: sgl-kernel
Category: MoE (Memory-bound)
Ops: Align MoE expert assignments to block size for efficient processing

Usage:
    python bench_moe_align_block_size.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    E, K,
    BenchmarkResult, PEAK_BANDWIDTH_GBS,
    benchmark_kernel, save_results, check_sgl_kernel
)


def bench_moe_align_block_size(sgl_kernel, B: int, S: int, num_experts: int,
                                topk: int, block_size: int, phase: str,
                                device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark moe_align_block_size kernel."""
    try:
        from sgl_kernel import moe_align_block_size
    except ImportError:
        print("Warning: moe_align_block_size not available")
        return None

    tokens = B * S if phase == "prefill" else B
    # Expert indices: [tokens, topk]
    expert_indices = torch.randint(0, num_experts, (tokens, topk), dtype=torch.int32, device=device)

    def kernel_fn():
        moe_align_block_size(expert_indices, num_experts, block_size)

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

    # Memory: read expert_indices, write aligned indices and metadata
    bytes_read = expert_indices.numel() * 4
    bytes_write = expert_indices.numel() * 4 + num_experts * 4  # aligned indices + expert counts
    bytes_transferred = bytes_read + bytes_write
    flops = tokens * topk * 5  # sorting/alignment ops
    gflops = flops / (latency_ms * 1e-3) / 1e9
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    peak_pct = (bandwidth_gbs / PEAK_BANDWIDTH_GBS) * 100

    return BenchmarkResult(
        kernel="moe_align_block_size",
        op="align_experts",
        phase=phase,
        B=B, S=S,
        M=tokens, N=topk, K_dim=block_size,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound="memory"
    )


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run moe_align_block_size benchmarks."""
    sgl_kernel = check_sgl_kernel()
    if not sgl_kernel:
        print("ERROR: sgl_kernel not available")
        return

    results = []
    block_size = 64  # Common block size for MoE

    # Decode phase (S=1)
    print("\n=== Decode Phase ===")
    for B in batch_sizes:
        result = bench_moe_align_block_size(sgl_kernel, B, 1, E, K, block_size, "decode")
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    # Prefill phase
    print("\n=== Prefill Phase ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            result = bench_moe_align_block_size(sgl_kernel, B, S, E, K, block_size, "prefill")
            if result:
                results.append(result)
                print(f"  B={B}, S={S}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    if results:
        save_results(results, output_dir, "moe_align_block_size")
    else:
        print("\nNo results - kernel not available")


def main():
    parser = argparse.ArgumentParser(description="Benchmark moe_align_block_size kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: moe_align_block_size (Kernel #21)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
