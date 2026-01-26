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
    Nh, Dr,
    BenchmarkResult, PEAK_BANDWIDTH_GBS,
    benchmark_kernel, save_results, check_flashinfer
)


def bench_mla_rope_quantize_fp8(flashinfer, B: int, S: int,
                                 device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark MLA RoPE + FP8 quantization kernel.

    API signature:
        mla_rope_quantize_fp8(
            q_rope,         # [tokens, num_heads, qk_rope_head_dim]
            k_rope,         # [tokens, qk_rope_head_dim] (MLA: 2D)
            q_nope,         # [tokens, num_heads, qk_nope_head_dim]
            k_nope,         # [tokens, kv_lora_rank] (MLA: 2D)
            cos_sin_cache,  # [max_seq_len, qk_rope_head_dim] MUST be float32
            pos_ids,        # [tokens]
            ...
        )
    """
    try:
        from flashinfer.rope import mla_rope_quantize_fp8
    except ImportError:
        print("Warning: mla_rope_quantize_fp8 not available (requires flashinfer)")
        return None

    tokens = B * S

    # DeepSeek MLA dimensions
    # From flashinfer docs: q_nope shape is (nnz, num_heads, no_rope_dim)
    # For MLA: k_rope is 2D (nnz, rope_dim), k_nope is 2D (nnz, no_rope_dim)
    qk_nope_head_dim = 128  # no_rope_dim (NOT kv_lora_rank!)
    qk_rope_head_dim = Dr   # 64 (rope_dim)

    # Query inputs: (nnz, num_qo_heads, dim)
    q_rope = torch.randn(tokens, Nh, qk_rope_head_dim, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(tokens, Nh, qk_nope_head_dim, dtype=torch.bfloat16, device=device)

    # Key inputs: MLA uses 2D tensors (nnz, dim)
    k_rope = torch.randn(tokens, qk_rope_head_dim, dtype=torch.bfloat16, device=device)
    k_nope = torch.randn(tokens, qk_nope_head_dim, dtype=torch.bfloat16, device=device)  # 128, not 512!

    # RoPE cos/sin cache - MUST be float32
    max_seq_len = max(S, 2048)  # Use reasonable max
    cos_sin_cache = torch.randn(max_seq_len, qk_rope_head_dim, dtype=torch.float32, device=device)

    # Position IDs
    pos_ids = torch.arange(S, dtype=torch.int64, device=device).repeat(B)[:tokens]

    def kernel_fn():
        mla_rope_quantize_fp8(
            q_rope,
            k_rope,
            q_nope,
            k_nope,
            cos_sin_cache,
            pos_ids,
        )

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

    # Memory-bound: read q_rope, k_rope, q_nope, k_nope, cos_sin; write quantized outputs
    bytes_read = (q_rope.numel() + k_rope.numel() + q_nope.numel() + k_nope.numel()) * 2
    bytes_read += cos_sin_cache.numel() * 4  # float32
    bytes_write = (q_rope.numel() + k_rope.numel() + q_nope.numel() + k_nope.numel())  # FP8 outputs = 1 byte each
    bytes_transferred = bytes_read + bytes_write
    flops = tokens * (Nh * qk_rope_head_dim * 4 + qk_rope_head_dim * 4)  # RoPE ops
    gflops = flops / (latency_ms * 1e-3) / 1e9
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    peak_pct = (bandwidth_gbs / PEAK_BANDWIDTH_GBS) * 100

    return BenchmarkResult(
        kernel="mla_rope_quantize_fp8",
        op="rope_quantize",
        phase="prefill" if S > 1 else "decode",
        B=B, S=S,
        M=tokens, N=qk_rope_head_dim, K_dim=qk_nope_head_dim,
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
