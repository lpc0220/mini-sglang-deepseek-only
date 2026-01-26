#!/usr/bin/env python3
"""
Benchmark: cutlass_mla_decode (Kernel #7)
Source: sgl-kernel
Category: Attention (Mixed bound)
Ops: attn_mqa (MLA decode attention)

Usage:
    python bench_cutlass_mla_decode.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    Nh, Lkv, Dr,
    BenchmarkResult, PEAK_BANDWIDTH_GBS, PEAK_TFLOPS_FP16, RIDGE_FP16,
    benchmark_kernel, save_results, check_sgl_kernel
)


def bench_cutlass_mla_decode(sgl_kernel, B: int, seq_len: int,
                             device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark Cutlass MLA decode kernel.

    Note: Large B*seq_len combinations can cause CUDA crashes.
    On GB200, crashes observed at B=2, seq_len=1024 (B*seq_len=2048).
    Limit to B*seq_len <= 1024 to avoid illegal instruction errors.
    """
    try:
        from sgl_kernel import cutlass_mla_decode, cutlass_mla_get_workspace_size
    except ImportError:
        print("Warning: cutlass_mla_decode not available")
        return None

    # Skip large combinations that cause CUDA crashes
    # GB200 crashes at B=2, seq_len=1024 (2048), works at B=2, seq_len=512 (1024)
    if B * seq_len > 1024:
        print(f"  Skipping B={B}, seq_len={seq_len}: B*seq_len={B*seq_len} > 1024 (crash risk)")
        return None

    d = Lkv + Dr  # 576
    block_size = 64
    num_kv_splits = -1

    seq_lens_tensor = torch.full((B,), seq_len, dtype=torch.int32, device=device)
    block_num = (seq_len + block_size - 1) // block_size
    pack_factor = 128 // block_size
    block_num = ((block_num + pack_factor - 1) // pack_factor) * pack_factor

    qn = torch.randn(Nh, B, d - Dr, dtype=torch.bfloat16, device=device)
    qr = torch.randn(B, Nh, Dr, dtype=torch.bfloat16, device=device)
    block_table = torch.randint(0, B * block_num, (B, block_num),
                                dtype=torch.int32, device=device)
    kv_cache = torch.randn(block_table.numel(), block_size, d,
                           dtype=torch.bfloat16, device=device)

    try:
        workspace_size = cutlass_mla_get_workspace_size(block_num * block_size, B, num_kv_splits)
        workspace = torch.empty(workspace_size, device=device, dtype=torch.uint8)
    except Exception as e:
        print(f"Warning: Failed to get workspace size for B={B}, seq_len={seq_len}: {e}")
        return None

    def kernel_fn():
        cutlass_mla_decode(qn.transpose(0, 1), qr, kv_cache, seq_lens_tensor,
                          block_table, workspace, 1.44, num_kv_splits)

    try:
        latency_ms = benchmark_kernel(kernel_fn)
    except Exception as e:
        print(f"Warning: Kernel failed for B={B}, seq_len={seq_len}: {e}")
        # Try to clear CUDA error state and free memory
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except:
            pass
        return None

    # Approximate FLOPS for attention
    flops = 4 * B * Nh * seq_len * d
    q_bytes = qn.numel() * 2 + qr.numel() * 2
    kv_bytes = kv_cache.numel() * 2
    bytes_transferred = q_bytes + kv_bytes
    gflops = flops / (latency_ms * 1e-3) / 1e9
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    # Mixed bound depending on seq_len
    bound = "memory" if arith_intensity < RIDGE_FP16 else "compute"
    if bound == "memory":
        peak_pct = (bandwidth_gbs / PEAK_BANDWIDTH_GBS) * 100
    else:
        peak_pct = (gflops / 1000 / PEAK_TFLOPS_FP16) * 100

    return BenchmarkResult(
        kernel="cutlass_mla_decode",
        op="attn_mqa",
        phase="decode",
        B=B, S=seq_len,
        M=B * Nh, N=seq_len, K_dim=d,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound=bound
    )


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run cutlass_mla_decode benchmarks."""
    sgl_kernel = check_sgl_kernel()
    if not sgl_kernel:
        print("ERROR: sgl_kernel not available")
        return

    results = []

    print("\n=== Decode Phase (MLA Attention) ===")
    for B in batch_sizes:
        for seq_len in seq_lens:
            result = bench_cutlass_mla_decode(sgl_kernel, B, seq_len)
            if result:
                results.append(result)
                print(f"  B={B}, seq_len={seq_len}: {result.latency_ms:.4f} ms, "
                      f"{result.bandwidth_gbs:.1f} GB/s, {result.bound}")

    save_results(results, output_dir, "cutlass_mla_decode")


def main():
    parser = argparse.ArgumentParser(description="Benchmark cutlass_mla_decode kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048,4096",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: cutlass_mla_decode (Kernel #7)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
