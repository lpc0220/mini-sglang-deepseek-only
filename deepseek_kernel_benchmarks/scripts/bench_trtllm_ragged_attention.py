#!/usr/bin/env python3
"""
Benchmark: trtllm_ragged_attention_deepseek (Kernel #9)
Source: flashinfer (TensorRT-LLM kernel)
Category: Attention (Mixed bound)
Ops: Prefill attention with ragged batching

Usage:
    python bench_trtllm_ragged_attention.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    Nh, Lkv, Dr, Dv,
    BenchmarkResult, PEAK_BANDWIDTH_GBS, PEAK_TFLOPS_FP16, RIDGE_FP16,
    benchmark_kernel, save_results, check_flashinfer
)


def bench_trtllm_ragged_attention(flashinfer, B: int, S: int,
                                   device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark TensorRT-LLM ragged attention for DeepSeek.

    API signature:
        trtllm_ragged_attention_deepseek(
            query,              # [num_tokens, num_heads, 192]  (q_nope + q_rope)
            key,                # [num_tokens, num_heads, 192]
            value,              # [num_tokens, num_heads, 128]
            workspace_buffer,
            seq_lens,           # [B]
            max_q_len,          # int
            max_kv_len,         # int
            bmm1_scale,         # float
            bmm2_scale,         # float
            o_sf_scale,         # float
            batch_size,         # int
            window_left,        # int
            cum_seq_lens_q,     # [B+1]
            cum_seq_lens_kv,    # [B+1]
            enable_pdl,         # bool
            is_causal,          # bool
            return_lse,         # bool
        )

    Note: query.shape[2] == 192, key.shape[2] == 192, value.shape[2] == 128
    """
    try:
        from flashinfer.prefill import trtllm_ragged_attention_deepseek
    except ImportError:
        print("Warning: trtllm_ragged_attention_deepseek not available (requires flashinfer)")
        return None

    # DeepSeek MLA dimensions
    head_dim_qk = 192  # 128 + 64
    head_dim_v = Dv    # 128

    tokens = B * S

    # Query: [num_tokens, num_heads, 192]
    query = torch.randn(tokens, Nh, head_dim_qk, dtype=torch.bfloat16, device=device)
    # Key: [num_tokens, num_heads, 192]
    key = torch.randn(tokens, Nh, head_dim_qk, dtype=torch.bfloat16, device=device)
    # Value: [num_tokens, num_heads, 128]
    value = torch.randn(tokens, Nh, head_dim_v, dtype=torch.bfloat16, device=device)

    # Sequence lengths: [B]
    seq_lens_tensor = torch.full((B,), S, dtype=torch.int32, device=device)

    # Cumulative sequence lengths: [B+1]
    cum_seq_lens_q = torch.arange(0, tokens + 1, S, dtype=torch.int32, device=device)
    cum_seq_lens_kv = cum_seq_lens_q.clone()

    # Workspace buffer - needs ~8MB for trtllm kernels
    workspace_size_bytes = 16 * 1024 * 1024  # 16MB to be safe
    workspace_buffer = torch.zeros(workspace_size_bytes // 4, dtype=torch.int32, device=device)

    # Scale factors
    sm_scale = 1.0 / (head_dim_qk ** 0.5)

    def kernel_fn():
        trtllm_ragged_attention_deepseek(
            query,
            key,
            value,
            workspace_buffer,
            seq_lens_tensor,
            S,  # max_q_len
            S,  # max_kv_len
            sm_scale,  # bmm1_scale
            1.0,  # bmm2_scale
            1.0,  # o_sf_scale
            B,  # batch_size
            -1,  # window_left (-1 for no windowing)
            cum_seq_lens_q,
            cum_seq_lens_kv,
            False,  # enable_pdl
            True,   # is_causal
            False,  # return_lse
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

    # Approximate FLOPS for attention: 4 * tokens * seq_len * head_dim (simplified)
    flops = 4 * tokens * S * head_dim_qk
    q_bytes = query.numel() * 2
    kv_bytes = (key.numel() + value.numel()) * 2
    bytes_transferred = q_bytes + kv_bytes
    gflops = flops / (latency_ms * 1e-3) / 1e9
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    bound = "memory" if arith_intensity < RIDGE_FP16 else "compute"
    if bound == "memory":
        peak_pct = (bandwidth_gbs / PEAK_BANDWIDTH_GBS) * 100
    else:
        peak_pct = (gflops / 1000 / PEAK_TFLOPS_FP16) * 100

    return BenchmarkResult(
        kernel="trtllm_ragged_attention_deepseek",
        op="prefill_attn",
        phase="prefill",
        B=B, S=S,
        M=tokens * Nh, N=S, K_dim=head_dim_qk,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound=bound
    )


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run trtllm_ragged_attention benchmarks."""
    flashinfer = check_flashinfer()
    if not flashinfer:
        print("ERROR: flashinfer not available")
        return

    results = []

    print("\n=== Prefill Phase (TRT-LLM Ragged Attention) ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            result = bench_trtllm_ragged_attention(flashinfer, B, S)
            if result:
                results.append(result)
                print(f"  B={B}, S={S}: {result.latency_ms:.4f} ms, "
                      f"{result.gflops:.1f} GFLOPS, {result.bound}")

    save_results(results, output_dir, "trtllm_ragged_attention_deepseek")


def main():
    parser = argparse.ArgumentParser(description="Benchmark trtllm_ragged_attention kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: trtllm_ragged_attention_deepseek (Kernel #9)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
