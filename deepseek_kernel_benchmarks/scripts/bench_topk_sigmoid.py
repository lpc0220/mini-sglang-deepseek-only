#!/usr/bin/env python3
"""
Benchmark: topk_sigmoid (Kernel #15)
Source: sgl-kernel
Category: MoE Routing (Memory-bound)
Ops: topk (MoE expert selection with sigmoid, DeepSeek V3 variant)

Usage:
    python bench_topk_sigmoid.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    E, K,
    BenchmarkResult, PEAK_BANDWIDTH_GBS,
    benchmark_kernel, save_results, check_sgl_kernel
)


def bench_topk_sigmoid(sgl_kernel, B: int, S: int, num_experts: int, topk: int, phase: str,
                       device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark topk_sigmoid kernel for MoE routing (DeepSeek V3)."""
    try:
        from sgl_kernel import topk_sigmoid
    except ImportError:
        print("Warning: topk_sigmoid not available")
        return None

    tokens = B * S if phase == "prefill" else B
    # Router logits: [tokens, num_experts]
    router_logits = torch.randn(tokens, num_experts, dtype=torch.float32, device=device)

    def kernel_fn():
        topk_sigmoid(router_logits, topk)

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

    # Memory: read logits, write topk indices and weights
    bytes_read = router_logits.numel() * 4  # float32
    bytes_write = tokens * topk * (4 + 4)  # indices (int32) + weights (float32)
    bytes_transferred = bytes_read + bytes_write
    flops = tokens * num_experts * 5  # sigmoid + topk ops
    gflops = flops / (latency_ms * 1e-3) / 1e9
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    peak_pct = (bandwidth_gbs / PEAK_BANDWIDTH_GBS) * 100

    return BenchmarkResult(
        kernel="topk_sigmoid",
        op="topk",
        phase=phase,
        B=B, S=S,
        M=tokens, N=num_experts, K_dim=topk,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound="memory"
    )


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run topk_sigmoid benchmarks."""
    sgl_kernel = check_sgl_kernel()
    if not sgl_kernel:
        print("ERROR: sgl_kernel not available")
        return

    results = []

    # Decode phase (S=1)
    print("\n=== Decode Phase ===")
    for B in batch_sizes:
        result = bench_topk_sigmoid(sgl_kernel, B, 1, E, K, "decode")
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    # Prefill phase
    print("\n=== Prefill Phase ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            result = bench_topk_sigmoid(sgl_kernel, B, S, E, K, "prefill")
            if result:
                results.append(result)
                print(f"  B={B}, S={S}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    save_results(results, output_dir, "topk_sigmoid")


def main():
    parser = argparse.ArgumentParser(description="Benchmark topk_sigmoid kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: topk_sigmoid (Kernel #15)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
