#!/usr/bin/env python3
"""
Benchmark: rope_quantize_fp8 (MLA configuration) (Kernel #10)
Source: flashinfer
Category: Attention/RoPE (Memory-bound)
Ops: Fused RoPE + FP8 quantization for MLA

Based on: flashinfer/benchmarks/bench_rope_quantize_fp8.py

Usage:
    python bench_mla_rope_quantize_fp8.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    Nh,
    BenchmarkResult, PEAK_BANDWIDTH_GBS,
    benchmark_kernel, save_results, check_flashinfer
)

# DeepSeek MLA dimensions (from official flashinfer benchmark)
# MLA: 128 Q heads, 1 K head, 64 rope_dim + 512 no_rope_dim
NUM_QO_HEADS = Nh  # 128
NUM_KV_HEADS = 1   # MLA uses single KV head
ROPE_DIM = 64      # qk_rope_head_dim
NO_ROPE_DIM = 512  # kv_lora_rank (NOT 128!)
TOTAL_DIM = ROPE_DIM + NO_ROPE_DIM  # 576


def bench_mla_rope_quantize_fp8(flashinfer, num_tokens: int,
                                 device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark MLA RoPE + FP8 quantization kernel.

    Based on flashinfer/benchmarks/bench_rope_quantize_fp8.py:
        - MLA mode uses 2D K tensors: k_in shape [num_tokens, total_dim]
        - Q tensor shape: [num_tokens, num_qo_heads, total_dim]
        - Splits into rope and nope components
        - cos_sin_cache from RoPE computation (float32)
    """
    try:
        from flashinfer.rope import rope_quantize_fp8
    except ImportError:
        print("Warning: rope_quantize_fp8 not available (requires flashinfer)")
        return None

    torch.manual_seed(42)
    input_dtype = torch.bfloat16
    quant_dtype = torch.float8_e4m3fn

    # Create input tensors for MLA mode
    # Q: [num_tokens, num_qo_heads, total_dim]
    q_in = torch.randn(
        num_tokens, NUM_QO_HEADS, TOTAL_DIM,
        dtype=input_dtype, device=device
    )
    # K: [num_tokens, total_dim] - MLA uses 2D K tensor
    k_in = torch.randn(
        num_tokens, TOTAL_DIM,
        dtype=input_dtype, device=device
    )

    pos_ids = torch.arange(num_tokens, dtype=torch.int64, device=device)

    # Create cos_sin_cache for RoPE (must be float32)
    max_seq_len = max(num_tokens, 4096)
    # Simple RoPE cache: [max_seq_len, rope_dim] with cos and sin interleaved
    inv_freq = 1.0 / (10000 ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32) / ROPE_DIM))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_seq_len, rope_dim//2]
    cos = freqs.cos()
    sin = freqs.sin()
    cos_sin_cache = torch.cat([cos, sin], dim=-1).to(device)  # [max_seq_len, rope_dim]

    # Split tensors for RoPE vs non-RoPE components
    q_rope = q_in[..., :ROPE_DIM]   # [num_tokens, num_qo_heads, rope_dim]
    q_nope = q_in[..., ROPE_DIM:]   # [num_tokens, num_qo_heads, no_rope_dim]
    k_rope = k_in[..., :ROPE_DIM]   # [num_tokens, rope_dim]
    k_nope = k_in[..., ROPE_DIM:]   # [num_tokens, no_rope_dim]

    # Create output tensors (quantized)
    q_rope_out = torch.empty_like(q_rope, dtype=quant_dtype)
    q_nope_out = torch.empty_like(q_nope, dtype=quant_dtype)
    k_rope_out = torch.empty_like(k_rope, dtype=quant_dtype)
    k_nope_out = torch.empty_like(k_nope, dtype=quant_dtype)

    def kernel_fn():
        rope_quantize_fp8(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=pos_ids,
            is_neox=False,
            q_rope_out=q_rope_out,
            k_rope_out=k_rope_out,
            q_nope_out=q_nope_out,
            k_nope_out=k_nope_out,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
            enable_pdl=False,
        )

    try:
        latency_ms = benchmark_kernel(kernel_fn)
    except Exception as e:
        print(f"Warning: Kernel failed for num_tokens={num_tokens}: {e}")
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except:
            pass
        return None

    # Memory-bound: read inputs + cos_sin; write FP8 outputs
    bytes_read = (q_in.numel() + k_in.numel()) * q_in.element_size()
    bytes_read += cos_sin_cache.numel() * 4  # float32
    bytes_write = (q_rope_out.numel() + q_nope_out.numel() + k_rope_out.numel() + k_nope_out.numel())  # 1 byte each
    bytes_transferred = bytes_read + bytes_write
    flops = num_tokens * (NUM_QO_HEADS * ROPE_DIM * 4 + ROPE_DIM * 4)  # RoPE ops
    gflops = flops / (latency_ms * 1e-3) / 1e9
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    peak_pct = (bandwidth_gbs / PEAK_BANDWIDTH_GBS) * 100

    # Compute B and S from num_tokens (assume S=1 for decode-like scenarios)
    B = num_tokens
    S = 1

    return BenchmarkResult(
        kernel="mla_rope_quantize_fp8",
        op="rope_quantize",
        phase="decode" if num_tokens < 256 else "prefill",
        B=B, S=S,
        M=num_tokens, N=ROPE_DIM, K_dim=NO_ROPE_DIM,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound="memory"
    )


def run_benchmarks(token_counts: List[int], output_dir: str):
    """Run mla_rope_quantize_fp8 benchmarks."""
    flashinfer = check_flashinfer()
    if not flashinfer:
        print("ERROR: flashinfer not available")
        return

    results = []

    print("\n=== MLA RoPE + FP8 Quantization ===")
    for num_tokens in token_counts:
        result = bench_mla_rope_quantize_fp8(flashinfer, num_tokens)
        if result:
            results.append(result)
            print(f"  tokens={num_tokens}: {result.latency_ms:.4f} ms, "
                  f"{result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    save_results(results, output_dir, "mla_rope_quantize_fp8")


def main():
    parser = argparse.ArgumentParser(description="Benchmark mla_rope_quantize_fp8 kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--token-counts", type=str, default="1,2,4,8,16,32,64,128,256,384,512,768",
                        help="Comma-separated token counts")
    args = parser.parse_args()

    token_counts = [int(x) for x in args.token_counts.split(",")]

    print("=" * 60)
    print("Benchmark: rope_quantize_fp8 (MLA config) (Kernel #10)")
    print("MLA: 128 Q heads, 1 K head, 64 rope_dim + 512 no_rope_dim")
    print("=" * 60)
    run_benchmarks(token_counts, args.output)


if __name__ == "__main__":
    main()
