#!/usr/bin/env python3
"""
Benchmark: trtllm_ragged_attention_deepseek (Kernel #9)
Source: flashinfer (TensorRT-LLM kernel)
Category: Attention (Mixed bound)
Ops: Prefill attention with ragged batching

Based on: flashinfer/tests/attention/test_trtllm_ragged_kv_stride.py

Usage:
    python bench_trtllm_ragged_attention.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    Nh,
    BenchmarkResult, PEAK_BANDWIDTH_GBS, PEAK_TFLOPS_FP16, RIDGE_FP16,
    benchmark_kernel, save_results, check_flashinfer
)

# DeepSeek MLA dimensions (from flashinfer test)
NUM_KV_HEADS = Nh  # 128 (same as Q heads for this kernel)
HEAD_DIM_QK = 192  # 128 + 64
HEAD_DIM_VO = 128  # value/output dimension


def bench_trtllm_ragged_attention(flashinfer, batch_size: int, max_seq_len: int,
                                   device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark TensorRT-LLM ragged attention for DeepSeek.

    Based on flashinfer/tests/attention/test_trtllm_ragged_kv_stride.py:
        - query: [total_q, num_kv_heads, head_dim_qk]
        - key: [total_kv, num_kv_heads, head_dim_qk]
        - value: [total_kv, num_kv_heads, head_dim_vo]
        - workspace_buffer: 128MB as uint8
    """
    try:
        from flashinfer.prefill import trtllm_ragged_attention_deepseek
    except ImportError:
        print("Warning: trtllm_ragged_attention_deepseek not available (requires flashinfer)")
        return None

    torch.manual_seed(42)

    # Construct ragged Q with varying sequence lengths
    seq_lens_q = torch.randint(
        low=max(1, max_seq_len // 2), high=max_seq_len,
        size=(batch_size,), device=device, dtype=torch.int32
    )
    cum_seq_lens_q = torch.cat([
        torch.zeros(1, device=device, dtype=torch.int32),
        torch.cumsum(seq_lens_q, dim=0, dtype=torch.int32),
    ], dim=0)
    total_q = int(cum_seq_lens_q[-1].item())
    max_q_len = int(seq_lens_q.max().item())

    q = torch.randn(
        total_q, NUM_KV_HEADS, HEAD_DIM_QK,
        device=device, dtype=torch.bfloat16
    )

    # Construct ragged KV (use same length as Q for simplicity)
    seq_lens_kv = seq_lens_q.clone()
    cum_seq_lens_kv = cum_seq_lens_q.clone()
    total_kv = total_q
    max_kv_len = max_q_len

    k = torch.randn(
        total_kv, NUM_KV_HEADS, HEAD_DIM_QK,
        device=device, dtype=torch.bfloat16
    )
    v = torch.randn(
        total_kv, NUM_KV_HEADS, HEAD_DIM_VO,
        device=device, dtype=torch.bfloat16
    )

    # Workspace buffer - 128MB as uint8 (from official test)
    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    scale = float(1.0 / (HEAD_DIM_QK ** 0.5))

    def kernel_fn():
        trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=workspace_buffer,
            seq_lens=seq_lens_kv,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            bmm1_scale=scale,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=batch_size,
            window_left=-1,  # No windowing
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            enable_pdl=False,
            is_causal=True,
            return_lse=False,
        )

    try:
        latency_ms = benchmark_kernel(kernel_fn)
    except Exception as e:
        print(f"Warning: Kernel failed for B={batch_size}, S={max_seq_len}: {e}")
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except:
            pass
        return None

    # Approximate FLOPS for attention: 4 * total_q * avg_kv_len * head_dim
    avg_seq_len = total_q / batch_size
    flops = 4 * total_q * avg_seq_len * HEAD_DIM_QK
    q_bytes = q.numel() * q.element_size()
    kv_bytes = (k.numel() + v.numel()) * k.element_size()
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
        B=batch_size, S=max_seq_len,
        M=total_q * NUM_KV_HEADS, N=max_seq_len, K_dim=HEAD_DIM_QK,
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
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            result = bench_trtllm_ragged_attention(flashinfer, batch_size, seq_len)
            if result:
                results.append(result)
                print(f"  B={batch_size}, S={seq_len}: {result.latency_ms:.4f} ms, "
                      f"{result.gflops:.1f} GFLOPS, {result.bound}")

    save_results(results, output_dir, "trtllm_ragged_attention_deepseek")


def main():
    parser = argparse.ArgumentParser(description="Benchmark trtllm_ragged_attention kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="256,512,1024,2048,4096",
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
